// Correctness gate C (Phase 2 SPEC §5.2) for one back end. Writes a JSON report and exits non-zero unless every
// selected item has 0 mismatches. No tolerance anywhere: the shared checker (sf/check.hpp) compares bit patterns.
//
//   sf_gate --device emul|cuda|cpu [--path inc|hyb] [--microbatch B] [--threads N] [--parity ...]
//           [--partial] [--items 1,2,3,4,5,6,7,8] [--starts 64] [--ref-parity PC1,...] --out report.json
//   sf_gate --item10 REFERENCE_DIR RUN_DIR [--out report.json]
//
// References: the CPU shared (full-sort) and full methods on synthetic data and N <= 4,096 (items 1, 5), and the
// CPU-best incremental configuration (--ref-parity) on real data (items 3, 4). For the CPU back end itself the gate
// is a CPU self-check against those references. Each item reports a hash of its reference outputs ("golden"), so the
// golden outputs can be compared across machines (SPEC §5.3).
//
// Items:
//   1  synthetic suite: random and tie-heavy series, 18 policies x 42 events (lengths 2/30/256 at both ends and
//      inside, q 0/4/32, price-only and volume-only), stages and episodes vs shared, episodes and bits vs full
//   2  negative controls (GPU faults), each on a witness input where it must be detected; a control that is not
//      detected invalidates the run
//   3  real data, both series, P105: lengths 2..256 and 6, 120; all 17 task strengths and 0; both / price / volume;
//      64 starts per (series, L, q, kind) from stream purpose gate plus both series ends; alternating events
//   4  stage differential of item 3 (reported per first differing stage)
//   5  N <= 4,096 prefixes of both series: every start, back end vs CPU incremental vs full
//   6  determinism: one task with two batch configurations gives identical bits
//   7  numeric golden vectors: volume double rounding, half-integer fixed volumes, overflow rejection, tiny mn,
//      counts at the cap, rows with cap 0, a last partial batch
//   8  stream: Philox known answers, rejection sequence vs the independent Python implementation
//      (experiments/phase2-gpu/stream_vectors.json), 1,000,000 starts of the back end vs the host
//   9  (CPU only) Phase 1 hit files: experiments/phase2-gpu/check_phase1_hits.py
//   10 T-ref enumeration bits equal the CPU reference: --item10
//
//   sf_gate --oracle TASK REFERENCE_DIR [--threads N] [--ref-parity ...] [--out oracle.json]
// Reference oracle (protocol "Oracle for the reference"): per T-ref cell, 2,000 starts from stream purpose oracle are
// re-evaluated with the shared method (one scan, full-sort ranking) and 200 of them also with full recomputation;
// every stage and episode must equal the CPU-best incremental evaluation, and the hit bits the stored reference.
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "sf/check.hpp"
#include "sf/sha256.hpp"
#include "task.hpp"

using namespace sf;

namespace {

[[noreturn]] void fail(const std::string& msg) {
  std::fprintf(stderr, "sf_gate: %s\n", msg.c_str());
  std::exit(2);
}

const std::vector<int32_t> kTaskLengths = {2, 4, 8, 16, 32, 64, 128, 256};
const std::vector<double> kTaskStrengths = {1.0,  1.4142, 2.0,     2.8284, 4.0,  5.6569,  8.0,  11.3137, 16.0,
                                            22.6274, 32.0, 45.2548, 64.0,   90.5097, 128.0, 181.0193, 256.0};

struct Options {
  std::string device = "emul", out, ref_parity, item10_ref, item10_run, root = ".", oracle_task, oracle_ref;
  int64_t oracle_shared = 2000, oracle_full = 200;
  BackendConfig cfg;
  EvalOptions ref_eval;
  bool ref_table = false, partial = false;
  std::vector<int> items = {1, 2, 3, 4, 5, 6, 7, 8};
  int64_t starts = 64;
};

void parse_parity(const std::string& v, EvalOptions& e, bool& table, bool* group = nullptr) {
  std::stringstream ss(v);
  for (std::string it; std::getline(ss, it, ',');) {
    if (it == "PC1") table = true;
    else if (it == "PC2") e.s_table = true;
    else if (it == "PC3") e.cap_table = true;
    else if (it == "PC4") e.sparse_both = true;
    else if (it == "PC6" && group) *group = true;
    else if (it != "none" && it != "PC5" && it != "PC6") fail("unknown parity item " + it);
  }
}

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--partial") {
      o.partial = true;
      o.items = {1, 2, 3, 5, 7, 8};
      continue;
    }
    if (a == "--oracle") {
      if (i + 2 >= argc) fail("--oracle TASK REFERENCE_DIR");
      o.oracle_task = argv[++i];
      o.oracle_ref = argv[++i];
      continue;
    }
    if (a == "--item10") {
      if (i + 2 >= argc) fail("--item10 REF RUN");
      o.item10_ref = argv[++i];
      o.item10_run = argv[++i];
      continue;
    }
    if (i + 1 >= argc) fail("missing value for " + a);
    const std::string v = argv[++i];
    if (a == "--device") o.device = v;
    else if (a == "--out") o.out = v;
    else if (a == "--root") o.root = v;
    else if (a == "--path") o.cfg.path = v;
    else if (a == "--threads") o.cfg.threads = std::atoi(v.c_str());
    else if (a == "--microbatch") o.cfg.microbatch = v == "auto" ? 0 : std::atoll(v.c_str());
    else if (a == "--submission") o.cfg.submission = std::atoll(v.c_str());
    else if (a == "--starts") o.starts = std::atoll(v.c_str());
    else if (a == "--oracle-shared") o.oracle_shared = std::atoll(v.c_str());
    else if (a == "--oracle-full") o.oracle_full = std::atoll(v.c_str());
    else if (a == "--parity") parse_parity(v, o.cfg.eval, o.cfg.event_table, &o.cfg.group_by_cell);
    else if (a == "--ref-parity") parse_parity(v, o.ref_eval, o.ref_table), o.ref_parity = v;
    else if (a == "--items") {
      o.items.clear();
      std::stringstream ss(v);
      for (std::string it; std::getline(ss, it, ',');) o.items.push_back(std::atoi(it.c_str()));
    } else fail("unknown option " + a);
  }
  return o;
}

// ---- report ----

struct Item {
  explicit Item(int i = 0, std::string n = "") : id(i), name(std::move(n)) {}
  int id;
  std::string name;
  int64_t trials = 0, mismatches = 0;
  std::map<std::string, int64_t> by_stage;
  std::string first, golden, note;
  bool valid = true;  // false: the item could not be judged (for example a control without a witness)
  double seconds = 0;
  bool pass() const { return valid && mismatches == 0; }
};

std::string item_json(const Item& it) {
  std::ostringstream j;
  j << "{\"item\":" << it.id << ",\"name\":" << json_quote(it.name) << ",\"trials\":" << it.trials
    << ",\"mismatches\":" << it.mismatches << ",\"valid\":" << (it.valid ? "true" : "false")
    << ",\"pass\":" << (it.pass() ? "true" : "false") << ",\"by_stage\":{";
  bool first = true;
  for (const auto& s : it.by_stage) j << (first ? "" : ",") << json_quote(s.first) << ":" << s.second, first = false;
  j << "},\"first_mismatch\":" << json_quote(it.first) << ",\"golden_sha256\":" << json_quote(it.golden)
    << ",\"note\":" << json_quote(it.note) << ",\"seconds\":" << it.seconds << "}";
  return j.str();
}

// Serialised trace for the golden hash.
void hash_trace(Sha256& h, const TrialTrace& t, const uint64_t* hit, const uint64_t* ovl, int words) {
  for (const auto& r : t.rows) {
    h.update(&r.c0, 8);
    h.update(&r.c1, 8);
    h.update(r.range.data(), r.range.size() * 4);
    h.update(r.volume.data(), r.volume.size() * 4);
    h.update(r.both_start.data(), r.both_start.size() * 4);
    h.update(r.both_count.data(), r.both_count.size() * 4);
  }
  const auto eps = [&h](const std::vector<Episode>& v) {
    const uint64_t n = v.size();
    h.update(&n, 8);
    for (const auto& e : v) {
      h.update(&e.k, sizeof e.k);
      h.update(&e.start, 8);
      h.update(&e.window, 4);
      h.update(&e.count, 4);
      h.update(&e.S, 8);
    }
  };
  eps(t.ordered);
  for (const auto& p : t.episodes) eps(p);
  h.update(hit, words * 8);
  h.update(ovl, words * 8);
}

// ---- comparing a back end with the CPU reference on explicit trials ----

struct Trials {
  std::vector<CellPlan> plans;
  std::vector<int32_t> cells;
  std::vector<int64_t> starts;
};

enum class RefKind { Incremental, SharedFull };

// Evaluates all trials with the back end (with traces) and the reference, in chunks; records mismatches in `it`.
void compare_backend(const Options& o, const BaseState& b, const Trials& tr, RefKind ref, Item& it, bool stages,
                     const BackendConfig* cfg_override = nullptr, bool expect_detect = false) {
  const BackendConfig& cfg = cfg_override ? *cfg_override : o.cfg;
  auto be = make_backend(o.device, b, cfg);
  be->set_cells(tr.plans);
  const size_t P = b.policies.size();
  const int words = policy_words(P);
  const int64_t n = static_cast<int64_t>(tr.starts.size());
  const int64_t chunk = 256;
  const int T = std::max(1, omp_get_max_threads());
  EvalOptions eo = o.ref_eval;
  eo.trace = true;
  std::vector<std::unique_ptr<Evaluator>> inc(T), shared(T), full(T);
  Sha256 golden;
  for (int64_t c0 = 0; c0 < n; c0 += chunk) {
    const int64_t m = std::min(chunk, n - c0);
    std::vector<uint64_t> hit(m * words), ovl(m * words);
    std::vector<uint8_t> done(m);
    std::vector<TrialTrace> traces;
    BatchOutput out{words, hit.data(), ovl.data(), done.data(), &traces};
    be->evaluate(TrialBatch{nullptr, 0, m, tr.cells.data() + c0, tr.starts.data() + c0}, out, [] { return false; });
    std::vector<TrialTrace> ref_tr(m), ref_full(m);
    std::vector<uint64_t> rh(m * words, 0), ro(m * words, 0);
#pragma omp parallel for num_threads(T) schedule(dynamic, 1)
    for (int64_t i = 0; i < m; ++i) {
      const int t = omp_get_thread_num();
      const CellPlan& p = tr.plans[tr.cells[c0 + i]];
      const TestEvent e{tr.starts[c0 + i], p.length, p.strength, p.kind};
      std::vector<std::vector<Episode>> eps;
      if (ref == RefKind::Incremental) {
        if (!inc[t]) inc[t] = std::make_unique<Evaluator>(b, Method::Incremental, eo);
        if (o.ref_table) inc[t]->run(e, p.table, 1);
        else inc[t]->run(e, p.median_range, 1);
        ref_tr[i] = inc[t]->trace();
      } else {
        EvalOptions so;
        so.trace = true;
        if (!shared[t]) shared[t] = std::make_unique<Evaluator>(b, Method::Shared, so);
        if (!full[t]) full[t] = std::make_unique<Evaluator>(b, Method::Full, so);
        shared[t]->run(e, p.median_range, 1);
        full[t]->run(e, p.median_range, 1);
        ref_tr[i] = shared[t]->trace();
        ref_full[i] = full[t]->trace();
      }
      const auto& found = ref == RefKind::Incremental ? ref_tr[i].episodes : ref_full[i].episodes;
      for (size_t q = 0; q < P; ++q) {
        const Detection d = detection(found[q], b.episodes[q], e);
        if (d.hit) rh[i * words + q / 64] |= uint64_t{1} << (q % 64);
        if (d.overlap) ro[i * words + q / 64] |= uint64_t{1} << (q % 64);
      }
    }
    for (int64_t i = 0; i < m; ++i) {
      hash_trace(golden, ref_tr[i], rh.data() + i * words, ro.data() + i * words, words);
      CheckResult r = compare_traces(ref_tr[i], traces[i], stages);
      if (r.ok() && ref == RefKind::SharedFull) r = compare_traces(ref_full[i], traces[i], false);
      if (r.ok())
        r = compare_bits(rh.data() + i * words, ro.data() + i * words, hit.data() + i * words, ovl.data() + i * words, P);
      if (r.ok() && !done[i]) r = CheckResult{CheckStage::Bits, "trial not completed"};
      ++it.trials;
      if (!r.ok()) {
        ++it.mismatches;
        ++it.by_stage[check_stage_name(r.first)];
        if (it.first.empty())
          it.first = "trial " + std::to_string(c0 + i) + " (start " + std::to_string(tr.starts[c0 + i]) + ", L " +
                     std::to_string(tr.plans[tr.cells[c0 + i]].length) + "): " + check_stage_name(r.first) + ", " +
                     r.detail;
        if (expect_detect) return;  // a negative control needs one detection
      }
    }
  }
  it.golden = golden.hex();
}

Series synthetic(int64_t n, uint32_t seed, double tick = 0.0) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> step(0.0, 5e-4), spread(6e-4, 3e-4);
  std::lognormal_distribution<double> vol(2.5, 0.7);
  Series s;
  double close = 42000.0;
  const auto q = [tick](double p) { return tick > 0 ? std::round(p / tick) * tick : p; };
  for (int64_t i = 0; i < n; ++i) {
    close *= std::exp(step(rng));
    const double sp = std::fabs(spread(rng));
    s.ts_ns.push_back(i * 60'000'000'000LL);
    double lo = q(close * (1 - sp)), hi = q(close * (1 + sp));
    if (hi < lo) hi = lo;
    s.low.push_back(static_cast<float>(lo));
    s.high.push_back(static_cast<float>(hi));
    s.volume.push_back(static_cast<float>(std::round(vol(rng) * 4) / 4));
  }
  return s;
}

// The 42 events of the sf_test suite.
Trials suite_events(const Series& s, uint32_t seed) {
  Trials t;
  std::mt19937_64 rng(seed);
  std::map<std::pair<int32_t, std::pair<double, int>>, int32_t> plan_of;
  const auto add = [&](int64_t start, int32_t L, double q, EventKind k) {
    const auto key = std::make_pair(L, std::make_pair(q, static_cast<int>(k)));
    if (!plan_of.count(key)) {
      plan_of[key] = static_cast<int32_t>(t.plans.size());
      t.plans.push_back(make_cell_plan(s, L, q, k));
    }
    t.cells.push_back(plan_of[key]);
    t.starts.push_back(start);
  };
  for (int32_t L : {2, 30, 256}) {
    std::uniform_int_distribution<int64_t> pos(0, s.size() - L);
    for (int64_t start : {int64_t{0}, s.size() - L, pos(rng), pos(rng)})
      for (double q : {0.0, 4.0, 32.0}) add(start, L, q, EventKind::Both);
    add(pos(rng), L, 32.0, EventKind::Price);
    add(pos(rng), L, 32.0, EventKind::Volume);
  }
  return t;
}

std::vector<Policy> suite_policies() { return policy_grid({4, 8, 15}, {1.5, 2.0, 3.0}, {30, 60}); }

std::vector<Policy> grid(const std::vector<int>& ks, const std::vector<double>& ss, const std::vector<int64_t>& gs) {
  return policy_grid(ks, ss, gs);
}
std::vector<Policy> p105() { return grid({4, 8, 15}, {1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0}, {15, 30, 60, 120, 240}); }
std::vector<Policy> p24() { return grid({8, 15}, {1.5, 2.0, 2.5, 3.0}, {30, 60, 120}); }

struct RealSeries {
  std::string name, path;
  uint32_t code;
};
std::vector<RealSeries> real_series(const Options& o) {
  return {{"1m-quarter", o.root + "/data/BTCUSDT_1m_2024Q1.bin", 0}, {"1s-week", o.root + "/data/BTCUSDT_1s_20240108_14.bin", 1}};
}

// ---- items ----

Item item1(const Options& o) {
  Item it{1, "synthetic suite"};
  for (const auto& [s, tag] : {std::make_pair(synthetic(5000, 20261004), "random"),
                               std::make_pair(synthetic(5000, 20261007, 5.0), "tie-heavy")}) {
    const BaseState b = prepare_base(s, kDefaultWindows, suite_policies(), 256, true, 1);
    compare_backend(o, b, suite_events(s, 11), RefKind::SharedFull, it, true);
    it.note += std::string(tag) + " ";
  }
  return it;
}

// A witness input and the fault that must be detected on it.
Item item2(const Options& o) {
  Item it{2, "negative controls"};
  if (o.device == "cpu") {
    it.note = "GPU faults do not apply to the CPU back end; CPU controls are in sf_test";
    return it;
  }
  struct Control {
    std::string fault, witness;
  };
  const std::vector<Control> controls = {{"device-log10", "random"},      {"reversed-tie", "tie-heavy"},
                                         {"cap-minus-one", "tie-heavy"},  {"direct-float", "volume 2^54+2^30+1"},
                                         {"half-even", "volume 2^-25"},   {"fast-math", "random"}};
  std::string notes;
  for (const auto& c : controls) {
    Series s;
    std::vector<Policy> pol = suite_policies();
    Trials tr;
    if (c.witness == "random" || c.witness == "tie-heavy") {
      s = synthetic(5000, c.witness == "random" ? 20261004 : 20261007, c.witness == "random" ? 0.0 : 5.0);
      tr = suite_events(s, 11);
    } else {
      s = synthetic(600, 7);
      if (c.witness == "volume 2^54+2^30+1") {
        // fixed-point sum of the window [300, 303) is 2^54 + 2^30 + 1; policies use windows 2..4 so no window
        // sum can overflow
        s.volume[300] = std::ldexp(1.0f, 30);
        s.volume[301] = 64.0f;
        s.volume[302] = std::ldexp(1.0f, -24);
        pol = grid({3}, {1.5}, {30});
      } else {
        // fixed point exactly 0.5 on every bar near the event: half to even gives window sums of 0, half away
        // from zero gives w units, so the volume feature differs (a single such bar beside normal volumes would
        // vanish in the float rounding of the feature)
        for (int i = 250; i < 350; ++i) s.volume[i] = std::ldexp(1.0f, -25);
      }
      tr.plans.push_back(make_cell_plan(s, 2, 1.0, EventKind::Price));
      tr.cells = {0};
      tr.starts = {300};
    }
    const BaseState b = prepare_base(s, kDefaultWindows, pol, 256, true, 1);
    Item clean{0, ""}, faulty{0, ""};
    BackendConfig cfg = o.cfg;
    compare_backend(o, b, tr, RefKind::SharedFull, clean, true, &cfg);
    cfg.fault = c.fault;
    compare_backend(o, b, tr, RefKind::SharedFull, faulty, true, &cfg, true);
    const bool detected = faulty.mismatches > 0;
    it.trials += clean.trials;
    it.mismatches += clean.mismatches;  // the clean back end must match on every witness
    if (!detected) it.valid = false;
    notes += c.fault + ": " + (detected ? "detected at " + faulty.first : "NOT DETECTED (witness invalid)") + "; ";
  }
  it.note = notes;
  return it;
}

Item item3(const Options& o, Item& item4) {
  Item it{3, "real data"};
  item4 = Item{4, "stage differential of item 3"};
  std::vector<int32_t> lengths = kTaskLengths;
  lengths.push_back(6);
  lengths.push_back(120);
  std::vector<double> strengths = kTaskStrengths;
  strengths.insert(strengths.begin(), 0.0);
  std::vector<EventKind> kinds = {EventKind::Both, EventKind::Price, EventKind::Volume};
  auto series = real_series(o);
  auto pols = p105();
  if (o.partial) {  // Stage C: week, L = 32, q = 16, P24
    series = {series[1]};
    lengths = {32};
    strengths = {16.0};
    kinds = {EventKind::Both};
    pols = p24();
  }
  for (const auto& rs : series) {
    const Series s = load_series(rs.path);
    const BaseState b = prepare_base(s, kDefaultWindows, pols, 256, true, std::max(1, omp_get_max_threads()));
    Trials tr;
    for (size_t iL = 0; iL < lengths.size(); ++iL)
      for (size_t iq = 0; iq < strengths.size(); ++iq)
        for (size_t ik = 0; ik < kinds.size(); ++ik) {
          const int32_t L = lengths[iL];
          const int32_t plan = static_cast<int32_t>(tr.plans.size());
          tr.plans.push_back(make_cell_plan(s, L, strengths[iq], kinds[ik]));
          const uint32_t code = static_cast<uint32_t>((iL * strengths.size() + iq) * kinds.size() + ik);
          const StreamKey key{kTaskGate, rs.code, code, 0, kPurposeGate};
          for (int64_t j = 0; j < o.starts; ++j) {
            uint64_t st;
            if (!stream_draw(key, j, s.size() - L + 1, st)) fail("gate draw failed");
            tr.cells.push_back(plan);
            tr.starts.push_back(static_cast<int64_t>(st));
          }
          for (int64_t st : {int64_t{0}, s.size() - L}) {  // both series ends
            tr.cells.push_back(plan);
            tr.starts.push_back(st);
          }
        }
    // alternating events: the first event repeated between others must give the same result each time
    const int64_t a0 = tr.starts[0];
    const int32_t c0 = tr.cells[0];
    for (int r = 0; r < 4 && tr.plans.size() > 1; ++r) {
      tr.cells.push_back(static_cast<int32_t>(tr.plans.size()) - 1 - r % static_cast<int>(tr.plans.size()));
      tr.starts.push_back(s.size() / 2 + 17 * r);
      tr.cells.push_back(c0);
      tr.starts.push_back(a0);
    }
    Item part{3, ""};
    compare_backend(o, b, tr, RefKind::Incremental, part, true);
    it.trials += part.trials;
    it.mismatches += part.mismatches;
    for (const auto& kv : part.by_stage) it.by_stage[kv.first] += kv.second;
    if (it.first.empty() && !part.first.empty()) it.first = rs.name + ": " + part.first;
    it.golden += (it.golden.empty() ? "" : ",") + part.golden;
    it.note += rs.name + " " + std::to_string(part.trials) + " trials; ";
  }
  item4.trials = it.trials;
  item4.mismatches = it.mismatches;
  item4.by_stage = it.by_stage;
  item4.first = it.first;
  item4.note = "stage-level breakdown of item 3 (features -> tail counts -> ordered candidates -> episodes -> bits)";
  return it;
}

Item item5(const Options& o) {
  Item it{5, "N <= 4096, every start"};
  for (const auto& rs : real_series(o)) {
    const Series s = load_series(rs.path).head(4096);
    const auto pols = o.partial ? p24() : p105();
    const BaseState b = prepare_base(s, kDefaultWindows, pols, 256, true, 1);
    Trials tr;
    const std::vector<int32_t> lengths = o.partial ? std::vector<int32_t>{32} : std::vector<int32_t>{2, 8, 32, 128, 256};
    for (int32_t L : lengths) {
      tr.plans.push_back(make_cell_plan(s, L, 16.0, EventKind::Both));
      for (int64_t st = 0; st <= s.size() - L; ++st) {
        tr.cells.push_back(static_cast<int32_t>(tr.plans.size()) - 1);
        tr.starts.push_back(st);
      }
    }
    Item a{5, ""}, f{5, ""};
    compare_backend(o, b, tr, RefKind::Incremental, a, true);   // vs CPU incremental, all stages
    compare_backend(o, b, tr, RefKind::SharedFull, f, false);   // vs full (episodes, bits) and shared (stages)
    it.trials += a.trials;
    it.mismatches += a.mismatches + f.mismatches;
    for (const auto& kv : a.by_stage) it.by_stage[kv.first] += kv.second;
    for (const auto& kv : f.by_stage) it.by_stage[kv.first] += kv.second;
    if (it.first.empty()) it.first = !a.first.empty() ? a.first : f.first;
    it.golden += (it.golden.empty() ? "" : ",") + a.golden;
  }
  return it;
}

// Packed bits of a fixed-n task under one configuration.
std::string task_bits(const Options& o, const BaseState& b, const Series& s, const BackendConfig& cfg) {
  auto be = make_backend(o.device, b, cfg);
  std::vector<CellPlan> plans;
  for (int32_t L : {2, 32, 256}) plans.push_back(make_cell_plan(s, L, 16.0, EventKind::Both));
  be->set_cells(plans);
  TrialSpace sp{};
  sp.order = TrialOrder::CellMajor;
  sp.cells = 3;
  sp.per_cell = 333;  // not a multiple of any batch size used
  sp.total = 999;
  sp.key = StreamKey{kTaskGate, 1, 0, 0, kPurposeGate};
  for (int c = 0; c < 3; ++c) {
    sp.stream_cell[c] = 900 + c;
    sp.valid[c] = s.size() - plans[c].length + 1;
  }
  const int words = policy_words(b.policies.size());
  std::vector<uint64_t> hit(sp.total * words), ovl(sp.total * words);
  std::vector<uint8_t> done(sp.total);
  BatchOutput out{words, hit.data(), ovl.data(), done.data()};
  be->evaluate(TrialBatch{&sp, 0, sp.total}, out, [] { return false; });
  Sha256 h;
  h.update(hit.data(), hit.size() * 8);
  h.update(ovl.data(), ovl.size() * 8);
  h.update(done.data(), done.size());
  return h.hex();
}

Item item6(const Options& o) {
  Item it{6, "determinism across batch configurations"};
  const Series s = load_series(real_series(o)[1].path).head(151200);
  const BaseState b = prepare_base(s, kDefaultWindows, p24(), 256, true, 1);
  BackendConfig c1 = o.cfg, c2 = o.cfg;
  if (o.device == "cpu") {
    c1.threads = 1;
    c2.threads = std::max(2, o.cfg.threads);
  } else {
    c1.microbatch = 64;
    c1.submission = 1;
    c2.microbatch = 37;
    c2.submission = 3;
  }
  const std::string h1 = task_bits(o, b, s, c1), h2 = task_bits(o, b, s, c2);
  it.trials = 2 * 999;
  it.mismatches = h1 != h2;
  it.golden = h1;
  it.note = c1.describe() + " vs " + c2.describe();
  if (h1 != h2) it.first = h1 + " != " + h2;
  return it;
}

Item item7(const Options& o) {
  Item it{7, "numeric golden vectors"};
  std::string notes;
  const auto run = [&](const std::string& what, const Series& s, const std::vector<Policy>& pol, const Trials& tr,
                       int32_t max_length = 256) {
    const BaseState b = prepare_base(s, kDefaultWindows, pol, max_length, true, 1);
    Item part{7, ""};
    compare_backend(o, b, tr, RefKind::SharedFull, part, true);
    it.trials += part.trials;
    it.mismatches += part.mismatches;
    if (it.first.empty() && !part.first.empty()) it.first = what + ": " + part.first;
    notes += what + " " + std::to_string(part.trials) + "; ";
  };
  const auto one = [](const Series& s, int64_t start, int32_t L, double q, EventKind k) {
    Trials t;
    t.plans.push_back(make_cell_plan(s, L, q, k));
    t.cells = {0};
    t.starts = {start};
    return t;
  };
  {  // volume double rounding: the window sum 2^54 + 2^30 + 1
    Series s = synthetic(600, 7);
    s.volume[300] = std::ldexp(1.0f, 30);
    s.volume[301] = 64.0f;
    s.volume[302] = std::ldexp(1.0f, -24);
    run("volume 2^54+2^30+1", s, grid({3}, {1.5}, {30}), one(s, 300, 2, 1.0, EventKind::Price));
  }
  {  // half-integer fixed volumes
    Series s = synthetic(600, 8);
    for (int i = 0; i < 40; ++i) s.volume[280 + i] = std::ldexp(static_cast<float>(2 * i + 1), -25);
    Trials t = one(s, 290, 16, 3.0, EventKind::Both);
    run("half-integer fixed volumes", s, suite_policies(), t);
  }
  {  // tiny mn
    Series s = synthetic(600, 9);
    for (int i = 0; i < 8; ++i) s.low[300 + i] = 1e-30f;
    run("tiny mn", s, suite_policies(), one(s, 296, 8, 4.0, EventKind::Both));
  }
  {  // counts at the cap and cap + 1: tie-heavy data, several s_floor; checked that such counts occur
    const Series s = synthetic(3000, 20261007, 5.0);
    Trials t = suite_events(s, 21);
    const auto pol = grid({15}, {1.0, 1.5}, {30});
    run("counts at cap (tie-heavy)", s, pol, t);
    // the witness must contain both-set counts equal to the cap (a count of cap + 1 is outside the set)
    const BaseState b = prepare_base(s, kDefaultWindows, pol, 256, true, 1);
    EvalOptions eo;
    eo.trace = true;
    Evaluator ev(b, Method::Incremental, eo);
    int64_t at_cap = 0;
    for (size_t i = 0; i < t.starts.size(); ++i) {
      const CellPlan& p = t.plans[t.cells[i]];
      ev.run(TestEvent{t.starts[i], p.length, p.strength, p.kind}, p.median_range, 1);
      for (size_t k = 0; k < ev.trace().rows.size(); ++k)
        for (int32_t c : ev.trace().rows[k].both_count) at_cap += c == b.caps[k];
    }
    if (at_cap == 0) it.valid = false;
    notes += "(" + std::to_string(at_cap) + " both-set counts equal to the cap); ";
  }
  {  // rows with cap = 0: N_k * 10^-1.5 < 1 for the longest windows
    const Series s = synthetic(280, 10);
    Trials t;
    t.plans.push_back(make_cell_plan(s, 16, 8.0, EventKind::Both));
    for (int64_t st = 0; st <= s.size() - 16; st += 7) {
      t.cells.push_back(0);
      t.starts.push_back(st);
    }
    run("rows with cap 0", s, grid({15}, {1.5, 2.0}, {30}), t, 16);
  }
  {  // a last partial batch: trial count not a multiple of the microbatch
    const Series s = synthetic(3000, 11);
    Trials t;
    t.plans.push_back(make_cell_plan(s, 30, 16.0, EventKind::Both));
    const int64_t mb = o.cfg.microbatch > 0 ? o.cfg.microbatch : 64;
    for (int64_t i = 0; i < 2 * mb + 3; ++i) {
      t.cells.push_back(0);
      t.starts.push_back((i * 37) % (s.size() - 30));
    }
    run("last partial batch", s, suite_policies(), t);
  }
  {  // overflow rejection: both the CPU and the back end refuse the same trial
    Series s = synthetic(1000, 12);
    s.volume[500] = std::ldexp(1.0f, 29);  // passes the base check; doubled by q = 1 it reaches 2^30 = limit at w 256
    const BaseState b = prepare_base(s, kDefaultWindows, grid({15}, {2.0}, {60}), 256, true, 1);
    auto be = make_backend(o.device, b, o.cfg);
    be->set_cells({make_cell_plan(s, 4, 1.0, EventKind::Both)});
    bool cpu_threw = false, be_threw = false;
    try {
      Evaluator ev(b, Method::Incremental);
      ev.run(TestEvent{499, 4, 1.0, EventKind::Both}, median_range(view(s), 4), 1);
    } catch (const std::overflow_error&) {
      cpu_threw = true;
    }
    try {
      const int32_t c = 0;
      const int64_t st = 499;
      uint64_t h = 0, ov = 0;
      uint8_t d = 0;
      BatchOutput out{1, &h, &ov, &d};
      be->evaluate(TrialBatch{nullptr, 0, 1, &c, &st}, out, [] { return false; });
    } catch (const std::overflow_error&) {
      be_threw = true;
    }
    ++it.trials;
    if (!(cpu_threw && be_threw)) {
      ++it.mismatches;
      if (it.first.empty()) it.first = "overflow: cpu threw " + std::to_string(cpu_threw) + ", back end " + std::to_string(be_threw);
    }
    notes += "overflow rejection (cpu " + std::string(cpu_threw ? "rejects" : "accepts") + ", back end " +
             (be_threw ? "rejects" : "accepts") + "); ";
  }
  it.note = notes;
  return it;
}

Item item8(const Options& o) {
  Item it{8, "stream"};
  const BaseState b = prepare_base(synthetic(600, 1), kDefaultWindows, grid({4}, {2.0}, {60}), 8, true, 1);
  auto be = make_backend(o.device, b, o.cfg);
  // known answers, on the back end
  const uint32_t ck[18] = {0, 0, 0, 0, 0, 0, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u,
                           0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u, 0xa4093822u, 0x299f31d0u};
  const uint32_t want[12] = {0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u, 0x408f276du, 0x41c83b0eu,
                             0xa20bc7c6u, 0x6d5451fdu, 0xd16cfe09u, 0x94fdccebu, 0x5001e420u, 0x24126ea1u};
  uint32_t got[12];
  be->philox_blocks(ck, 3, got);
  it.trials += 3;
  if (!std::equal(got, got + 12, want)) ++it.mismatches, it.first = "Philox known answers";
  // rejection sequence against the Python implementation
  const std::string path = o.root + "/experiments/phase2-gpu/stream_vectors.json";
  try {
    const Json v = Json::parse(read_text(path));
    const Json& r = v["rejection"];
    const Json& k = r["key"];
    TrialSpace sp{};
    sp.order = TrialOrder::RoundRobin;
    sp.cells = 1;
    sp.total = INT64_MAX;
    sp.key = StreamKey{static_cast<uint32_t>(k["task"].int64()), static_cast<uint32_t>(k["series"].int64()), 0,
                       static_cast<uint32_t>(k["replicate"].int64()), static_cast<uint32_t>(k["purpose"].int64())};
    sp.stream_cell[0] = static_cast<uint32_t>(k["cell"].int64());
    sp.valid[0] = r["M"].int64();
    const auto& starts = r["starts"].items();
    const int64_t n = static_cast<int64_t>(starts.size());
    std::vector<int32_t> cell(n);
    std::vector<int64_t> st(n);
    be->resolve_starts(sp, 0, n, cell.data(), st.data());
    int64_t bad = 0, host_bad = 0;
    for (int64_t j = 0; j < n; ++j) {
      const int64_t w = std::strtoll(starts[j].str().c_str(), nullptr, 10);
      bad += st[j] != w;
      int32_t c;
      int64_t jj, hs;
      resolve_trial(sp, j, c, jj, hs);
      host_bad += hs != w;
    }
    it.trials += n;
    it.mismatches += bad + host_bad;
    if ((bad || host_bad) && it.first.empty()) it.first = "rejection sequence differs from stream.py";
    it.note += "rejection sequence: " + std::to_string(n) + " draws, M = " + std::to_string(sp.valid[0]) + ", " +
               std::to_string(r["rejected"].int64()) + " rejected candidates; ";
  } catch (const std::exception& e) {
    it.valid = false;
    it.note += std::string("stream vectors unavailable: ") + e.what() + "; ";
  }
  // 1,000,000 starts of a T2-sized space: back end vs host
  {
    TrialSpace sp{};
    sp.order = TrialOrder::CellMajor;
    sp.cells = 136;
    sp.per_cell = 7353;  // 136 x 7353 = 1,000,008
    sp.total = sp.cells * sp.per_cell;
    sp.key = StreamKey{kTaskT2, 1, 0, 0, kPurposeMeasured};
    for (int c = 0; c < sp.cells; ++c) {
      sp.stream_cell[c] = c;
      sp.valid[c] = 604800 - kTaskLengths[c / 17] + 1;
    }
    const int64_t n = 1000000;
    std::vector<int32_t> cell(n);
    std::vector<int64_t> st(n);
    be->resolve_starts(sp, 0, n, cell.data(), st.data());
    int64_t bad = 0;
#pragma omp parallel for reduction(+ : bad)
    for (int64_t g = 0; g < n; ++g) {
      int32_t c;
      int64_t j, s;
      resolve_trial(sp, g, c, j, s);
      bad += c != cell[g] || s != st[g];
    }
    it.trials += n;
    it.mismatches += bad;
    if (bad && it.first.empty()) it.first = std::to_string(bad) + " of 1,000,000 starts differ from the host";
  }
  return it;
}

int item10(const Options& o) {
  Item it{10, "T-ref enumeration bits equal the CPU reference"};
  const std::string a = read_text(o.item10_ref + "/ref_bits.bin"), b = read_text(o.item10_run + "/ref_bits.bin");
  it.trials = static_cast<int64_t>(a.size());
  it.golden = sha256_hex(a.data(), a.size());
  if (a != b) {
    it.mismatches = 1;
    size_t i = 0;
    while (i < std::min(a.size(), b.size()) && a[i] == b[i]) ++i;
    it.first = "first differing byte " + std::to_string(i) + " (sizes " + std::to_string(a.size()) + ", " +
               std::to_string(b.size()) + ")";
  }
  const std::string j = "{\"items\":[" + item_json(it) + "],\"pass\":" + (it.pass() ? "true" : "false") + "}\n";
  if (!o.out.empty()) std::ofstream(o.out) << j;
  std::fputs(j.c_str(), stdout);
  return it.pass() ? 0 : 1;
}

int oracle(const Options& o) {
  const auto t0 = std::chrono::steady_clock::now();
  const Task t = load_task(o.oracle_task);
  if (t.order != "enumerate") fail("the oracle checks an enumeration task (T-ref)");
  const std::string sha = sha256_file(t.series_path);
  if (!t.series_sha256.empty() && sha != t.series_sha256) fail("input SHA-256 mismatch");
  const Series s = load_series(t.series_path).head(t.prefix);
  const int T = std::max(1, o.cfg.threads);
  const BaseState b = prepare_base(s, t.ladder, t.policies, t.max_length, true, T);
  const TrialSpace sp = make_space(t, s.size(), kPurposeMeasured, 0);
  const std::string bits = read_text(o.oracle_ref + "/ref_bits.bin");
  const size_t P = b.policies.size();
  if (bits.size() != (static_cast<size_t>(sp.total) * P + 7) / 8) fail("reference bits have the wrong size");
  std::map<int32_t, double> med;
  std::vector<CellPlan> plans;
  for (const auto& c : t.cells) {
    if (!med.count(c.length)) med[c.length] = median_range(view(s), c.length);
    CellPlan p;
    p.length = c.length;
    p.strength = c.strength;
    p.kind = c.kind;
    p.median_range = med[c.length];
    p.table = make_event_table(c.length, c.strength, p.median_range);
    plans.push_back(p);
  }
  struct Job {
    int32_t cell;
    int64_t start;
    bool full;
  };
  std::vector<Job> jobs;
  for (int32_t c = 0; c < sp.cells; ++c) {
    const StreamKey key{t.task_code, t.series_code, sp.stream_cell[c], 0, kPurposeOracle};
    for (int64_t j = 0; j < o.oracle_shared; ++j) {
      uint64_t st;
      if (!stream_draw(key, j, sp.valid[c], st)) fail("oracle draw failed");
      jobs.push_back({c, static_cast<int64_t>(st), j < o.oracle_full});
    }
  }
  EvalOptions eo = o.ref_eval, so;
  eo.trace = so.trace = true;
  std::vector<int64_t> bad(sp.cells, 0), bad_full(sp.cells, 0);
  std::string first;
  std::vector<std::unique_ptr<Evaluator>> inc(T), shared(T), full(T);
#pragma omp parallel for num_threads(T) schedule(dynamic, 1)
  for (size_t i = 0; i < jobs.size(); ++i) {
    const int th = omp_get_thread_num();
    if (!inc[th]) {
      inc[th] = std::make_unique<Evaluator>(b, Method::Incremental, eo);
      shared[th] = std::make_unique<Evaluator>(b, Method::Shared, so);
      full[th] = std::make_unique<Evaluator>(b, Method::Full, so);
    }
    const Job& jb = jobs[i];
    const CellPlan& p = plans[jb.cell];
    const TestEvent e{jb.start, p.length, p.strength, p.kind};
    if (o.ref_table) inc[th]->run(e, p.table, 1);
    else inc[th]->run(e, p.median_range, 1);
    shared[th]->run(e, p.median_range, 1);
    CheckResult r = compare_traces(shared[th]->trace(), inc[th]->trace(), true);
    // hit bits of the shared evaluation against the stored reference
    const int64_t g = sp.offset[jb.cell] + jb.start;
    for (size_t q = 0; r.ok() && q < P; ++q) {
      const bool hit = detection(shared[th]->trace().episodes[q], b.episodes[q], e).hit;
      const size_t bit = static_cast<size_t>(g) * P + q;
      if (hit != static_cast<bool>(bits[bit / 8] >> (bit % 8) & 1))
        r = CheckResult{CheckStage::Bits, "reference bit of policy " + std::to_string(q)};
    }
    bool full_bad = false;
    if (r.ok() && jb.full) {
      full[th]->run(e, p.median_range, 1);
      const CheckResult f = compare_traces(full[th]->trace(), shared[th]->trace(), false);
      if (!f.ok()) r = f, full_bad = true;
    }
    if (!r.ok()) {
#pragma omp critical(sf_oracle)
      {
        ++bad[jb.cell];
        bad_full[jb.cell] += full_bad;
        if (first.empty())
          first = "cell " + std::to_string(jb.cell) + " start " + std::to_string(jb.start) + ": " +
                  check_stage_name(r.first) + ", " + r.detail;
      }
    }
  }
  int64_t total_bad = 0;
  std::ostringstream j;
  j << "{\"task\":" << json_quote(t.name) << ",\"reference\":" << json_quote(o.oracle_ref)
    << ",\"reference_sha256\":" << json_quote(sha256_hex(bits.data(), bits.size()))
    << ",\"shared_per_cell\":" << o.oracle_shared << ",\"full_per_cell\":" << o.oracle_full
    << ",\"reference_parity\":" << json_quote(o.ref_parity.empty() ? "none" : o.ref_parity) << ",\"cells\":[";
  for (int32_t c = 0; c < sp.cells; ++c) {
    total_bad += bad[c];
    j << (c ? "," : "") << "{\"cell\":" << c << ",\"L\":" << plans[c].length << ",\"q\":" << plans[c].strength
      << ",\"mismatches\":" << bad[c] << ",\"full_mismatches\":" << bad_full[c] << "}";
  }
  const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  j << "],\"mismatches\":" << total_bad << ",\"first_mismatch\":" << json_quote(first) << ",\"seconds\":" << sec
    << ",\"pass\":" << (total_bad == 0 ? "true" : "false") << "}\n";
  if (!o.out.empty()) std::ofstream(o.out) << j.str();
  std::fputs(j.str().c_str(), stdout);
  return total_bad == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  const Options o = parse(argc, argv);
  if (!o.item10_ref.empty()) return item10(o);
  if (!o.oracle_task.empty()) return oracle(o);
  if (!backend_available(o.device)) fail("back end not available in this build: " + o.device);
  std::vector<Item> items;
  bool all = true;
  Item item4;
  for (int id : o.items) {
    if (id == 4) continue;  // reported with item 3
    const auto t0 = std::chrono::steady_clock::now();
    Item it;
    try {
      switch (id) {
        case 1: it = item1(o); break;
        case 2: it = item2(o); break;
        case 3: it = item3(o, item4); break;
        case 5: it = item5(o); break;
        case 6: it = item6(o); break;
        case 7: it = item7(o); break;
        case 8: it = item8(o); break;
        default: fail("unknown item " + std::to_string(id));
      }
    } catch (const std::exception& e) {
      it = Item{id, "item " + std::to_string(id)};
      it.valid = false;
      it.note = std::string("error: ") + e.what();
    }
    it.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("item %2d %-44s %9lld trials %6lld mismatches  %s%s\n", it.id, it.name.c_str(),
                static_cast<long long>(it.trials), static_cast<long long>(it.mismatches), it.pass() ? "pass" : "FAIL",
                it.valid ? "" : " (invalid)");
    std::fflush(stdout);
    all &= it.pass();
    items.push_back(it);
    if (id == 3 && std::find(o.items.begin(), o.items.end(), 4) != o.items.end()) {
      item4.seconds = 0;
      items.push_back(item4);
      all &= item4.pass();
    }
  }
  std::ostringstream j;
  j << "{\"device\":" << json_quote(o.device) << ",\"config\":" << json_quote(o.cfg.describe())
    << ",\"reference_parity\":" << json_quote(o.ref_parity.empty() ? "none" : o.ref_parity)
    << ",\"partial\":" << (o.partial ? "true" : "false") << ",\"starts_per_combination\":" << o.starts << ",\"items\":[";
  for (size_t i = 0; i < items.size(); ++i) j << (i ? "," : "") << item_json(items[i]);
  j << "],\"pass\":" << (all ? "true" : "false") << "}\n";
  if (!o.out.empty()) std::ofstream(o.out) << j.str();
  std::printf("gate %s\n", all ? "passed" : "FAILED");
  return all ? 0 : 1;
}

// Correctness gate for the engine. Runs before any timing; exits non-zero on failure.
//
// Identity contract (sf/scan.hpp): every scan path gives the same feature bits, so counts and episodes must match
// exactly, not within a tolerance. Checks:
//   features   direct vs optimized vs threaded, bit for bit, including volume
//   counts     full sort and tail ranking vs the brute-force definition, at several thresholds, on tie-heavy data
//   methods    full / shared / tail / incremental give identical episodes for every policy of a grid, over events
//              that vary position (including both ends), length, strength (including q = 0), and kind
//   controls   broken variants that the gate must reject: off-by-one boundary, wrong window ladder, a stale
//              incremental base, and a tail cap set by the wrong s_min
#include <cmath>
#include <cstdio>
#include <random>
#include <string>

#include "sf/detector.hpp"
#include "sf/evaluate.hpp"
#include "sf/scenario.hpp"

using namespace sf;

namespace {
int failures = 0;

void check(bool ok, const std::string& name) {
  std::printf("%-76s %s\n", name.c_str(), ok ? "pass" : "FAIL");
  if (!ok) ++failures;
}

// tick > 0 rounds prices to a tick grid so equal ranges (rank ties) occur as well as equal volumes.
Series random_series(int64_t n, uint32_t seed, double tick = 0.0) {
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
    // quantize volume so equal values (rank ties) actually occur
    s.volume.push_back(static_cast<float>(std::round(vol(rng) * 4) / 4));
  }
  return s;
}

bool same_bits(float a, float b) { return a == b || (std::isnan(a) && std::isnan(b)); }

bool features_match(const Features& ref, const Features& got) {
  if (ref.range.size() != got.range.size() || got.min.size() != ref.min.size()) return false;
  for (size_t i = 0; i < ref.range.size(); ++i)
    if (!same_bits(ref.min[i], got.min[i]) || !same_bits(ref.max[i], got.max[i]) ||
        !same_bits(ref.range[i], got.range[i]) || !same_bits(ref.volume[i], got.volume[i]))
      return false;
  return true;
}

std::vector<int32_t> brute_counts(const Features& f) {
  std::vector<int32_t> c(f.range.size(), kInvalidCount);
  for (int k = 0; k < f.K(); ++k) {
    const int64_t nk = valid_starts(f.n, f.windows[k]);
    const size_t row = static_cast<size_t>(k) * f.n;
    for (int64_t s = 0; s < nk; ++s) {
      int32_t cr = 0, cv = 0;
      for (int64_t j = 0; j < nk; ++j) {
        cr += f.range[row + j] >= f.range[row + s];
        cv += f.volume[row + j] >= f.volume[row + s];
      }
      c[row + s] = std::max(cr, cv);
    }
  }
  return c;
}

// Tail semantics: exact where both feature counts are <= cap, kAboveCap where either exceeds it.
bool tail_matches(const Features& f, const std::vector<int32_t>& brute, const std::vector<int32_t>& tail,
                  double s_floor) {
  for (int k = 0; k < f.K(); ++k) {
    const int64_t nk = valid_starts(f.n, f.windows[k]), cap = count_cap(nk, s_floor);
    const size_t row = static_cast<size_t>(k) * f.n;
    for (int64_t s = 0; s < f.n; ++s) {
      const int32_t b = brute[row + s], t = tail[row + s];
      if (s >= nk ? t != kInvalidCount : (b <= cap ? t != b : t != kAboveCap)) return false;
    }
  }
  return true;
}

bool same_episodes(const std::vector<Episode>& a, const std::vector<Episode>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (a[i].k != b[i].k || a[i].start != b[i].start || a[i].window != b[i].window || a[i].count != b[i].count ||
        a[i].S != b[i].S)
      return false;
  return true;
}

bool same_all(const std::vector<std::vector<Episode>>& a, const std::vector<std::vector<Episode>>& b) {
  if (a.size() != b.size()) return false;
  for (size_t p = 0; p < a.size(); ++p)
    if (!same_episodes(a[p], b[p])) return false;
  return true;
}

std::vector<TestEvent> event_set(int64_t n, uint32_t seed) {
  std::mt19937_64 rng(seed);
  std::vector<TestEvent> out;
  for (int32_t L : {2, 30, 256}) {
    std::uniform_int_distribution<int64_t> pos(0, n - L);
    for (int64_t start : {int64_t{0}, n - L, pos(rng), pos(rng)})
      for (double q : {0.0, 4.0, 32.0}) out.push_back({start, L, q, EventKind::Both});
    out.push_back({pos(rng), L, 32.0, EventKind::Price});
    out.push_back({pos(rng), L, 32.0, EventKind::Volume});
  }
  return out;
}

using Results = std::vector<std::vector<std::vector<Episode>>>;  // [event][policy]

Results run_all(const BaseState& b, const std::vector<TestEvent>& events, Method m, int threads) {
  Evaluator ev(b, m);
  Results out;
  for (const auto& e : events) out.push_back(ev.run(e, median_range(view(*b.base), e.length), threads));
  return out;
}

// Number of events where method m differs from the reference results in any policy.
int mismatches(const BaseState& b, const std::vector<TestEvent>& events, const Results& ref, Method m, int threads) {
  const Results got = run_all(b, events, m, threads);
  int bad = 0;
  for (size_t i = 0; i < events.size(); ++i) bad += !same_all(ref[i], got[i]);
  return bad;
}
}  // namespace

int main() {
  const auto w = kDefaultWindows;
  const Series s = random_series(5000, 20261004);
  const Series ticks = random_series(5000, 20261007, 5.0);  // tie-heavy in both features
  const SeriesView x = view(s);
  ScanBuffers sb;

  // ---- features: bit-identical across scan paths ----
  Features direct, opt, opt_mt, direct_mt;
  scan_features(x, w, ScanKind::Direct, true, 1, direct, sb);
  scan_features(x, w, ScanKind::Optimized, true, 1, opt, sb);
  scan_features(x, w, ScanKind::Optimized, true, 4, opt_mt, sb);
  scan_features(x, w, ScanKind::Direct, true, 4, direct_mt, sb);
  check(features_match(direct, direct_mt), "features: direct, 4 threads == 1 thread (bits)");
  check(features_match(direct, opt), "features: optimized == direct (bits, volume included)");
  check(features_match(opt, opt_mt), "features: optimized, 4 threads == 1 thread (bits)");

  // Negative controls: the gate must reject both.
  Features bug = direct;
  for (int k = 0; k < bug.K(); ++k) {  // boundary off by one: last valid start dropped
    const size_t last = static_cast<size_t>(k) * bug.n + valid_starts(bug.n, w[k]) - 1;
    bug.range[last] = bug.volume[last] = bug.min[last] = bug.max[last] = NAN;
  }
  check(!features_match(direct, bug), "negative control: boundary off by one is caught");
  std::vector<int32_t> rotated(w.begin() + 1, w.end());
  rotated.push_back(w.front());
  Features wrong;
  scan_features(x, rotated, ScanKind::Direct, true, 1, wrong, sb);
  check(!features_match(direct, wrong), "negative control: wrong window length is caught");

  // ---- counts against the definition (brute force is O(n^2) per length, so a shorter series) ----
  for (const Series* src : {&s, &ticks}) {
    const std::string tag = src == &s ? "random" : "tie-heavy";
    const Series small = src->head(700);
    Features fs;
    scan_features(view(small), w, ScanKind::Optimized, false, 1, fs, sb);
    const auto brute = brute_counts(fs);
    RankBuffers rb;
    std::vector<int32_t> c1, c4;
    rarity_counts(fs, 1, c1, rb);
    rarity_counts(fs, 4, c4, rb);
    check(c1 == brute, "counts: full sort == brute force (" + tag + ")");
    check(c1 == c4, "counts: full sort, 4 threads == 1 thread (" + tag + ")");
    bool tail_ok = true;
    for (double sf : {0.5, 1.0, 1.5, 2.0, 3.0}) {
      std::vector<int32_t> t;
      rarity_counts(fs, sf == 1.5 ? 4 : 1, t, rb, RankKind::Tail, sf);
      tail_ok &= tail_matches(fs, brute, t, sf);
    }
    check(tail_ok, "counts: tail ranking == brute force up to the cap, s 0.5..3 (" + tag + ")");
  }

  // ---- single-policy detector ----
  DetectorConfig cfg;
  Workspace ws;
  cfg.scan = ScanKind::Direct;
  const auto top_direct = detect(x, cfg, 1, ws);
  cfg.scan = ScanKind::Optimized;
  const auto top_opt = detect(x, cfg, 1, ws);
  const auto top_opt_mt = detect(x, cfg, 4, ws);
  cfg.rank = RankKind::Tail;
  const auto top_tail = detect(x, cfg, 1, ws);
  check(!top_direct.empty(), "detector: returns episodes");
  check(same_episodes(top_direct, top_opt), "detector: optimized == direct top episodes");
  check(same_episodes(top_opt, top_opt_mt), "detector: 4 threads == 1 thread");
  check(same_episodes(top_opt, top_tail), "detector: tail ranking == full sort");

  // ---- methods: identical episodes for every policy and every event ----
  const auto policies = policy_grid({4, 8, 15}, {1.5, 2.0, 3.0}, {30, 60});
  for (const Series* src : {&s, &ticks}) {
    const std::string tag = src == &s ? "random" : "tie-heavy";
    const BaseState b = prepare_base(*src, w, policies, 256, true, 1);
    const auto events = event_set(src->size(), 11);
    const Results ref = run_all(b, events, Method::Full, 1);
    for (Method m : {Method::Shared, Method::Tail, Method::Incremental}) {
      const int bad1 = mismatches(b, events, ref, m, 1), bad4 = mismatches(b, events, ref, m, 4);
      check(bad1 == 0 && bad4 == 0, std::string("methods: ") + method_name(m) + " == full, " +
                                        std::to_string(events.size()) + " events x 18 policies, 1/4 threads (" +
                                        tag + ")");
    }
    // q = 0 leaves the series unchanged: every method must return the base episodes, so nothing is "detected"
    Evaluator inc(b, Method::Incremental);
    const TestEvent null{1234, 30, 0.0};
    const auto& got = inc.run(null, median_range(view(*src), 30), 1);
    bool null_ok = true;
    for (size_t p = 0; p < policies.size(); ++p)
      null_ok &= same_episodes(got[p], b.episodes[p]) && !detection(got[p], b.episodes[p], null).hit;
    check(null_ok, "control: q = 0 returns the base episodes and no detection (" + tag + ")");
  }

  // Negative controls for the methods: a stale incremental base and a tail cap from the wrong s_min.
  {
    const BaseState b = prepare_base(s, w, policies, 256, true, 1);
    Series moved = s;
    for (int64_t p = 4000; p < 4040; ++p) moved.volume[p] *= 50.0f;  // the base the evaluator should have used
    BaseState stale = prepare_base(moved, w, policies, 256, false, 1);
    stale.heads = b.heads;  // orderings left over from the old series
    const auto events = event_set(s.size(), 12);
    check(mismatches(stale, events, run_all(stale, events, Method::Full, 1), Method::Incremental, 1) > 0,
          "negative control: stale incremental base is caught");
    const BaseState capped = prepare_base(s, w, policies, 256, true, 1, 3.0);
    const Results ref = run_all(capped, events, Method::Full, 1);
    check(mismatches(capped, events, ref, Method::Tail, 1) > 0 &&
              mismatches(capped, events, ref, Method::Incremental, 1) > 0,
          "negative control: tail cap from too high an s_min is caught");
  }

  // ---- scenarios ----
  Series work = s;
  const TestEvent e{2500, 30, 32.0};
  apply_event(s, e, median_range(x, e.length), work);
  cfg.rank = RankKind::Sort;
  const auto top_event = detect(view(work), cfg, 1, ws);
  check(detection(top_event, top_opt, e).hit, "scenario: strong test event is detected as a new episode");
  restore(s, e, work);
  check(work.low == s.low && work.high == s.high && work.volume == s.volume, "scenario: restore is exact");

  std::printf("%s (%d failure%s)\n", failures ? "GATE FAILED" : "gate passed", failures, failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}

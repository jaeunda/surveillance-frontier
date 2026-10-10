// Unit checks of the Phase 2 additions; exits non-zero on failure. The full correctness gate is sf_gate.
//   stream     Philox4x32-10 known-answer vectors (Random123 kat_vectors), Lemire mapping, counter layout
//   numerics   SHA-256 test vectors, Clopper-Pearson edge cases, the device arithmetic of the GPU path against the
//              CPU definitions (fixed-point rounding, int64 -> double -> float volume)
//   engine     parity options PC2-PC4 and the event table give the Phase 1 episodes; traces of all methods agree
//   backends   cpu and emul back ends give the same bits and traces as the CPU reference on synthetic series
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>

#include "../cuda/gpu_kernels.hpp"
#include "sf/batch.hpp"
#include "sf/check.hpp"
#include "sf/interval.hpp"
#include "sf/sha256.hpp"

using namespace sf;

namespace {
int failures = 0;

void check(bool ok, const std::string& name) {
  std::printf("%-84s %s\n", name.c_str(), ok ? "pass" : "FAIL");
  if (!ok) ++failures;
}

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
    s.ts_ns.push_back(i * 1'000'000'000LL);
    double lo = q(close * (1 - sp)), hi = q(close * (1 + sp));
    if (hi < lo) hi = lo;
    s.low.push_back(static_cast<float>(lo));
    s.high.push_back(static_cast<float>(hi));
    s.volume.push_back(static_cast<float>(std::round(vol(rng) * 4) / 4));
  }
  return s;
}

// Every back end against the CPU incremental reference with traces, on events spread over the series.
int backend_mismatches(const std::string& name, const Series& s, const std::vector<Policy>& pol, BackendConfig cfg,
                       std::string* first = nullptr) {
  const BaseState b = prepare_base(s, kDefaultWindows, pol, 256, true, 1);
  std::vector<CellPlan> plans;
  std::vector<int32_t> cells;
  std::vector<int64_t> starts;
  std::mt19937_64 rng(5);
  int c = 0;
  for (int32_t L : {2, 30, 256})
    for (double q : {0.0, 4.0, 32.0})
      for (EventKind kind : {EventKind::Both, EventKind::Price, EventKind::Volume}) {
        plans.push_back(make_cell_plan(s, L, q, kind));
        std::uniform_int_distribution<int64_t> pos(0, s.size() - L);
        for (int64_t st : {int64_t{0}, s.size() - L, pos(rng), pos(rng), pos(rng)}) {
          cells.push_back(c);
          starts.push_back(st);
        }
        ++c;
      }
  auto be = make_backend(name, b, cfg);
  be->set_cells(plans);
  const int64_t n = static_cast<int64_t>(starts.size());
  const int words = policy_words(pol.size());
  std::vector<uint64_t> hit(n * words), ovl(n * words);
  std::vector<uint8_t> done(n);
  std::vector<TrialTrace> traces;
  BatchOutput out{words, hit.data(), ovl.data(), done.data(), &traces};
  be->evaluate(TrialBatch{nullptr, 0, n, cells.data(), starts.data()}, out, [] { return false; });
  EvalOptions eo;
  eo.trace = true;
  Evaluator shared(b, Method::Shared, eo), full(b, Method::Full, eo);
  int bad = 0;
  for (int64_t i = 0; i < n; ++i) {
    const CellPlan& p = plans[cells[i]];
    const TestEvent e{starts[i], p.length, p.strength, p.kind};
    shared.run(e, p.median_range, 1);
    full.run(e, p.median_range, 1);
    std::vector<uint64_t> h(words, 0), o(words, 0);
    for (size_t q = 0; q < pol.size(); ++q) {
      const Detection d = detection(full.trace().episodes[q], b.episodes[q], e);
      if (d.hit) h[q / 64] |= uint64_t{1} << (q % 64);
      if (d.overlap) o[q / 64] |= uint64_t{1} << (q % 64);
    }
    CheckResult r = compare_traces(shared.trace(), traces[i], true);
    if (r.ok()) r = compare_traces(full.trace(), traces[i], false);
    if (r.ok()) r = compare_bits(h.data(), o.data(), hit.data() + i * words, ovl.data() + i * words, pol.size());
    if (!r.ok() && first && first->empty())
      *first = std::string(check_stage_name(r.first)) + ": " + r.detail + " (event " + std::to_string(i) + ")";
    bad += !r.ok() || !done[i];
  }
  return bad;
}
}  // namespace

int main() {
  // ---- stream ----
  {
    const Philox4x32 a = philox4x32_10(Philox4x32{{0, 0, 0, 0}}, 0, 0);
    const Philox4x32 b = philox4x32_10(Philox4x32{{~0u, ~0u, ~0u, ~0u}}, ~0u, ~0u);
    const Philox4x32 c =
        philox4x32_10(Philox4x32{{0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u}}, 0xa4093822u, 0x299f31d0u);
    const auto eq = [](const Philox4x32& x, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3) {
      return x.v[0] == a0 && x.v[1] == a1 && x.v[2] == a2 && x.v[3] == a3;
    };
    check(eq(a, 0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u) &&
              eq(b, 0x408f276du, 0x41c83b0eu, 0xa20bc7c6u, 0x6d5451fdu) &&
              eq(c, 0xd16cfe09u, 0x94fdccebu, 0x5001e420u, 0x24126ea1u),
          "stream: Philox4x32-10 known-answer vectors (Random123)");
    const StreamKey k{2, 1, 135, 0, 0};
    const Philox4x32 ctr = stream_counter(k, (uint64_t{3} << 32) | 7, 2);
    check(ctr.v[0] == 7 && ctr.v[1] == 3 && ctr.v[2] == (2u << 24 | 1u << 20 | 135u) && ctr.v[3] == 2u,
          "stream: counter layout (j, task, series, cell, purpose, replicate, attempt)");
    bool in_range = true;
    int rejected = 0;
    for (uint64_t j = 0; j < 20000; ++j) {
      uint64_t s;
      in_range &= stream_draw(k, j, 604793, s) && s < 604793;
      stream_draw(StreamKey{5, 0, 7, 3, 5}, j, (uint64_t{1} << 62) + 1, s, &rejected);
    }
    check(in_range, "stream: draws fall in [0, M)");
    check(rejected > 3000 && rejected < 9000, "stream: M = 2^62 + 1 rejects about a quarter of the candidates (" +
                                                  std::to_string(rejected) + " of ~20000)");
  }

  // ---- numerics ----
  check(sha256_hex("abc", 3) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" &&
            sha256_hex("", 0) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "sha256: FIPS 180-4 test vectors");
  {
    std::string m(1000000, 'a');
    check(sha256_hex(m.data(), m.size()) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
          "sha256: one million 'a'");
  }
  {
    const Interval z = clopper_pearson(0, 100, 0.05), f = clopper_pearson(100, 100, 0.05);
    const Interval h = clopper_pearson(50, 100, 0.05);
    check(z.lower == 0 && std::fabs(z.upper - 0.036217) < 1e-6 && f.upper == 1 && std::fabs(f.lower - 0.963783) < 1e-6,
          "intervals: k = 0 and k = n (0.0362 / 0.9638 at n = 100)");
    check(std::fabs(h.lower - 0.39832) < 1e-5 && std::fabs(h.upper - 0.60168) < 1e-5,
          "intervals: k = 50, n = 100 gives [0.3983, 0.6017]");
  }
  {
    // device arithmetic of the GPU path vs the CPU definitions (sf/scan.hpp)
    bool ok = true;
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> u(0.0f, 1e6f);
    for (int i = 0; i < 200000; ++i) {
      const float v = i < 64 ? std::ldexp(static_cast<float>(2 * i + 1), -25) : u(rng);  // half-integer fixed volumes
      ok &= gpu::fixed_volume(v, gpu::kFaultNone) == volume_fixed(v);
    }
    check(ok, "numerics: device fixed-point rounding == llround (half away from zero), incl. half-integers");
    const int64_t witness = (int64_t{1} << 54) + (int64_t{1} << 30) + 1;
    check(gpu::volume_feature_d(witness, gpu::kFaultNone) == volume_feature(witness) &&
              gpu::volume_feature_d(witness, gpu::kFaultDirectFloat) != volume_feature(witness),
          "numerics: volume int64 -> double -> float; direct int64 -> float differs on 2^54 + 2^30 + 1");
    check(gpu::fixed_volume(std::ldexp(1.0f, -25), gpu::kFaultHalfEven) != volume_fixed(std::ldexp(1.0f, -25)),
          "numerics: half-to-even fixed point differs on 2^-25 (witness)");
    float nan_bits = gpu::nan_f(), cpu_nan = rel_range(0.0f, 1.0f);
    check(std::memcmp(&nan_bits, &cpu_nan, 4) == 0, "numerics: NaN of an invalid range has the CPU's bit pattern");
  }

  // ---- engine: parity options and the event table keep every episode ----
  {
    const Series s = random_series(6000, 20261010, 5.0);
    const auto pol = policy_grid({4, 8, 15}, {1.5, 2.0, 3.0}, {30, 60});
    const BaseState b = prepare_base(s, kDefaultWindows, pol, 256, true, 1);
    Evaluator ref(b, Method::Incremental);
    EvalOptions all;
    all.s_table = all.cap_table = all.sparse_both = true;
    Evaluator par(b, Method::Incremental, all);
    EvalOptions tr;
    tr.trace = true;
    Evaluator ti(b, Method::Incremental, tr), ts(b, Method::Tail, tr);
    EvalOptions trs = all;
    trs.trace = true;
    Evaluator tp(b, Method::Incremental, trs);
    std::mt19937_64 rng(9);
    int bad = 0, bad_trace = 0;
    for (int32_t L : {2, 30, 256})
      for (double q : {0.0, 4.0, 32.0}) {
        const double med = median_range(view(s), L);
        const EventTable table = make_event_table(L, q, med);
        std::uniform_int_distribution<int64_t> pos(0, s.size() - L);
        for (int64_t st : {int64_t{0}, s.size() - L, pos(rng), pos(rng)}) {
          const TestEvent e{st, L, q, EventKind::Both};
          const auto want = ref.run(e, med, 1);
          const auto& got = par.run(e, table, 1);
          for (size_t p = 0; p < pol.size(); ++p) bad += !same_episode_list(got[p], want[p]);
          ti.run(e, med, 1);
          ts.run(e, med, 1);
          tp.run(e, table, 1);
          bad_trace += !compare_traces(ts.trace(), ti.trace(), true).ok();
          bad_trace += !compare_traces(ts.trace(), tp.trace(), true).ok();
        }
      }
    check(bad == 0, "engine: PC1-PC4 together give the Phase 1 incremental episodes");
    check(bad_trace == 0, "engine: per-stage traces of tail, incremental and incremental + PC1-PC4 agree");
  }

  // ---- back ends ----
  {
    const Series s = random_series(4000, 20261004), ticks = random_series(4000, 20261007, 5.0);
    const auto pol = policy_grid({4, 8, 15}, {1.5, 2.0, 3.0}, {30, 60});
    for (const std::string name : {"cpu", "emul"})
      for (const Series* src : {&s, &ticks}) {
        BackendConfig cfg;
        cfg.threads = 4;
        cfg.microbatch = 7;  // forces several microbatches and a partial last one
        for (const std::string path : {"inc", "hyb"}) {
          if (name == "cpu" && path == "hyb") continue;
          cfg.path = path;
          std::string first;
          const int bad = backend_mismatches(name, *src, pol, cfg, &first);
          check(bad == 0, "backends: " + name + (name == "emul" ? " (" + path + ")" : "") +
                              " == CPU shared/full, 135 events x 18 policies, all stages (" +
                              (src == &s ? "random" : "tie-heavy") + ")" + (first.empty() ? "" : " [" + first + "]"));
        }
      }
  }

  std::printf("%s (%d failure%s)\n", failures ? "TESTS FAILED" : "tests passed", failures, failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}

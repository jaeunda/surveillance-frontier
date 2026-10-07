// One trial of a stress test: a set of policies evaluated on the base series with one test event applied.
//
// Four methods return the same episodes for every policy, bit for bit (gated by sf_test). They differ only in how
// much work they repeat, which is what Phase 1 measures:
//   Full         every policy re-runs the whole detector (scan, full-sort ranking, selection)
//   Shared       one scan and one full-sort ranking at the largest K serve all policies; peaks are found once at the
//                smallest s_min and each policy only filters and merges them
//   Tail         as Shared, with tail ranking capped at the smallest s_min (sf/rank.hpp)
//   Incremental  the base series is scanned and ranked once. A trial recomputes only the windows that overlap the
//                event (at most L + w - 1 starts per length) and merges them into the base series' precomputed
//                per-length orderings, which yields the exact tail of every row without a pass over the series
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "sf/detector.hpp"
#include "sf/scenario.hpp"

namespace sf {

enum class Method { Full, Shared, Tail, Incremental };

const char* method_name(Method m);
Method parse_method(const std::string& name);  // throws on unknown names

struct Policy {
  int K;  // window lengths used: the first K of the ladder
  SelectConfig select;
};

// Cartesian product K x s_min x episode_gap, in that nesting order.
std::vector<Policy> policy_grid(const std::vector<int>& ks, const std::vector<double>& s_mins,
                                const std::vector<int64_t>& gaps, int top_k = 20);

// Read-only state built once per series and shared by all workers.
struct BaseState {
  const Series* base = nullptr;
  std::vector<int32_t> windows;  // ladder prefix for the largest K among the policies
  std::vector<Policy> policies;
  double s_floor = 0;            // smallest s_min among the policies: the tail cap
  int32_t max_length = 0;        // longest test event the incremental heads are sized for
  // Incremental only: per length, the largest values of each feature on the base series, descending, enough of
  // them to recover the (cap + 1) largest after up to max_length + w - 1 starts are replaced.
  struct Head {
    std::vector<RankItem> range, volume;
  };
  std::vector<Head> heads;
  std::vector<std::vector<Episode>> episodes;  // per policy, on the base series
};

// s_floor_override > 0 replaces the smallest s_min (only the gate's negative control uses it).
BaseState prepare_base(const Series& base, const std::vector<int32_t>& ladder, const std::vector<Policy>& policies,
                       int32_t max_length, bool incremental, int threads, double s_floor_override = 0);

class Evaluator {
 public:
  Evaluator(const BaseState& base, Method method);

  // Episodes per policy (same order as base.policies) for the base series with e applied. threads > 1
  // parallelizes inside the trial over window lengths. The returned reference is valid until the next call.
  const std::vector<std::vector<Episode>>& run(const TestEvent& e, double median_range_L, int threads,
                                               StageTimes* times = nullptr);

 private:
  struct TailEntry {
    int32_t start, count;
  };
  struct Row {  // incremental scratch for one window length
    int64_t c0 = 0, c1 = -1;  // starts whose windows overlap the event
    std::vector<float> new_range, new_volume;
    std::vector<int64_t> vfix;
    std::vector<int32_t> qmin, qmax, both;
    std::vector<RankItem> changed, merged;
    std::vector<TailEntry> tail_range, tail_volume;
    std::vector<int64_t> peaks;
  };

  void merge_tail(const std::vector<RankItem>& head, const std::vector<float>& values, int64_t nk, int64_t cap,
                  Row& row, std::vector<TailEntry>& tail);

  void run_full(int threads, StageTimes& st);
  void run_shared(RankKind kind, int threads, StageTimes& st);
  void run_incremental(const TestEvent& e, int threads, StageTimes& st);
  void merge_policies();

  const BaseState& b_;
  Method method_;
  Series work_;
  Workspace ws_;
  std::vector<DetectorConfig> configs_;  // Full: one detector per policy
  std::vector<int32_t> cc_;              // Incremental: combined counts, kAboveCap except at the current tails
  std::vector<Row> rows_;
  std::vector<std::vector<Episode>> out_;
};

// A policy detects the event if it reports an episode centred inside [start, start + length) that is not among its
// episodes on the base series (same k and start). `overlap` drops the second condition (the Phase 0 definition,
// which also counts episodes that were already there).
struct Detection {
  bool hit, overlap;
};
Detection detection(const std::vector<Episode>& found, const std::vector<Episode>& on_base, const TestEvent& e);

}  // namespace sf

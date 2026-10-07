// The full detector: features -> rarity counts -> candidates -> top episodes. One call is one evaluation of a
// policy setting on one series; stress tests repeat it over many scenarios.
#pragma once

#include <cstdint>
#include <vector>

#include "sf/rank.hpp"
#include "sf/scan.hpp"
#include "sf/select.hpp"
#include "sf/series.hpp"

namespace sf {

// Window ladder used in Phase 0 (roughly sqrt(2) steps from 2 to 256 bars).
inline const std::vector<int32_t> kDefaultWindows = {2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 91, 128, 181, 256};

struct DetectorConfig {
  std::vector<int32_t> windows = kDefaultWindows;
  SelectConfig select;
  ScanKind scan = ScanKind::Optimized;
  RankKind rank = RankKind::Sort;  // Tail uses select.s_min as its floor
};

struct StageTimes {
  double scan_ms = 0, rank_ms = 0, select_ms = 0;
  double total_ms() const { return scan_ms + rank_ms + select_ms; }
};

// All per-evaluation memory; reuse one per thread to keep allocation out of timed loops.
struct Workspace {
  Features features;
  ScanBuffers scan;
  RankBuffers rank;
  std::vector<int32_t> counts;
  std::vector<int64_t> candidates;
  SelectBuffers select;
};

// threads > 1 parallelizes inside the evaluation; for many evaluations, run one per thread with threads == 1.
std::vector<Episode> detect(const SeriesView& x, const DetectorConfig& cfg, int threads, Workspace& ws,
                            StageTimes* times = nullptr);

}  // namespace sf

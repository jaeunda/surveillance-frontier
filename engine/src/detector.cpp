#include "sf/detector.hpp"

#include <chrono>

namespace sf {

namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
}  // namespace

std::vector<Episode> detect(const SeriesView& x, const DetectorConfig& cfg, int threads, Workspace& ws,
                            StageTimes* times) {
  // 1) features of every (window length, start)
  auto t0 = Clock::now();
  scan_features(x, cfg.windows, cfg.scan, false, threads, ws.features, ws.scan);
  const double scan_ms = ms_since(t0);

  // 2) per-length rarity counts (with a full sort, the expensive part: 2K sorts)
  t0 = Clock::now();
  rarity_counts(ws.features, threads, ws.counts, ws.rank, cfg.rank, cfg.select.s_min);
  const double rank_ms = ms_since(t0);

  // 3) candidates and top episodes
  t0 = Clock::now();
  local_peaks(ws.counts, x.n, cfg.windows, cfg.select.s_min, threads, ws.candidates, ws.select);
  auto top = select_top(ws.candidates, ws.counts, x.n, cfg.windows, cfg.select, ws.select);
  const double select_ms = ms_since(t0);

  if (times) *times = {scan_ms, rank_ms, select_ms};
  return top;
}

}  // namespace sf

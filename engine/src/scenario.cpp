#include "sf/scenario.hpp"

#include <algorithm>
#include <cmath>

#include "sf/scan.hpp"

namespace sf {

double median_range(const SeriesView& x, int32_t length) {
  Features f;
  ScanBuffers buf;
  scan_features(x, {length}, ScanKind::Optimized, false, 1, f, buf);
  std::vector<float> r;
  r.reserve(f.range.size());
  for (float v : f.range)
    if (!std::isnan(v)) r.push_back(v);
  if (r.empty()) return 0.0;
  // nanmedian: mean of the two middle values for even counts
  const size_t m = r.size() / 2;
  std::nth_element(r.begin(), r.begin() + m, r.end());
  const double hi = r[m];
  if (r.size() % 2) return hi;
  const double lo = *std::max_element(r.begin(), r.begin() + m);
  return 0.5 * (lo + hi);
}

void apply_event(const Series& base, const TestEvent& e, double median_range_L, Series& dst) {
  const bool price = e.kind != EventKind::Volume, vol = e.kind != EventKind::Price;
  const float vol_factor = static_cast<float>(1.0 + e.strength);
  for (int32_t j = 0; j < e.length; ++j) {
    const double t = (j + 0.5) / e.length;
    const double shape = t < 2.0 / 3.0 ? t / (2.0 / 3.0) : (1.0 - t) / (1.0 / 3.0);
    const float factor = static_cast<float>(1.0 + e.strength * median_range_L * shape);
    const int64_t p = e.start + j;
    if (price) {
      dst.low[p] = base.low[p] * factor;
      dst.high[p] = base.high[p] * factor;
    }
    if (vol) dst.volume[p] = base.volume[p] * vol_factor;
  }
}

void restore(const Series& base, const TestEvent& e, Series& dst) {
  const auto b = e.start, end = e.start + e.length;
  std::copy(base.low.begin() + b, base.low.begin() + end, dst.low.begin() + b);
  std::copy(base.high.begin() + b, base.high.begin() + end, dst.high.begin() + b);
  std::copy(base.volume.begin() + b, base.volume.begin() + end, dst.volume.begin() + b);
}

}  // namespace sf

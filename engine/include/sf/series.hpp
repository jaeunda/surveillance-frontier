// Time series input: one value per bar (low, high, volume) plus timestamps.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace sf {

struct Series {
  std::vector<int64_t> ts_ns;  // bar open time, nanoseconds since epoch (UTC)
  std::vector<float> low, high, volume;

  int64_t size() const { return static_cast<int64_t>(low.size()); }
  Series head(int64_t n) const;  // first n bars (all if n <= 0 or n >= size)
};

// Non-owning view used by the engine; all three arrays have length n.
struct SeriesView {
  const float* low;
  const float* high;
  const float* volume;
  int64_t n;
};

inline SeriesView view(const Series& s) { return {s.low.data(), s.high.data(), s.volume.data(), s.size()}; }

// Binary format (little endian): "SFSER001", uint64 n, int64 ts[n], float32 low[n], high[n], volume[n].
Series load_series(const std::string& path);
void save_series(const std::string& path, const Series& s);

}  // namespace sf

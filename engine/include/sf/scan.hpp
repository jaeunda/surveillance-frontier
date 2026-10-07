// Stage 1: features of every (window length, start) pair.
//
// For window length w = windows[k] and start s (0 <= s <= n - w), over bars s .. s + w - 1:
//   min = min(low), max = max(high), range = (max - min) / min, volume = sum(volume).
// Arrays are stored row-major as [k * n + s]; starts with s > n - w are NaN.
//
// Identity contract: every scan path (direct, optimized, incremental) returns the same bits. min and max are exact;
// range is one float expression of them; volume is summed in fixed point (each bar rounded to a multiple of
// 2^-kVolumeFracBits, exact int64 sums), so the summation order cannot change the result. Equal features give equal
// integer counts in stage 2, hence identical episodes for every threshold and policy.
#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "sf/series.hpp"

namespace sf {

enum class ScanKind {
  Direct,     // every start reads its own window: O(n * sum(w)), mirrors the GPU kernel
  Optimized,  // monotonic-deque sliding min/max + running fixed-point sum: O(n * K), the CPU baseline
};

constexpr int kVolumeFracBits = 24;

inline int64_t volume_fixed(float v) { return std::llround(std::ldexp(static_cast<double>(v), kVolumeFracBits)); }
inline float volume_feature(int64_t sum) {
  return static_cast<float>(std::ldexp(static_cast<double>(sum), -kVolumeFracBits));
}
inline float rel_range(float mn, float mx) {
  return mn > 0.0f ? (mx - mn) / mn : std::numeric_limits<float>::quiet_NaN();
}

struct Features {
  int64_t n = 0;
  std::vector<int32_t> windows;
  std::vector<float> range, volume;  // [K * n]
  std::vector<float> min, max;       // [K * n], filled only when requested (correctness gate)
  int K() const { return static_cast<int>(windows.size()); }
};

// Reusable buffers so repeated evaluations do not allocate.
struct ScanBuffers {
  std::vector<int64_t> vfix;                      // [n] fixed-point volume per bar
  std::vector<std::vector<int32_t>> qmin, qmax;   // per thread slot, [n]
};

// Fixed-point volumes of bars [0, n); throws if a window sum of up to max_window bars could overflow int64.
void fixed_volumes(const SeriesView& x, int64_t max_window, std::vector<int64_t>& vfix);

// Features of starts 0 .. count - 1 of window length w on x (sliding deques; x and vfix may be offset views).
// qmin/qmax need count + w - 1 entries. mn/mx may be null.
void scan_starts(const SeriesView& x, const int64_t* vfix, int64_t w, int64_t count, float* range, float* volume,
                 float* mn, float* mx, int32_t* qmin, int32_t* qmax);

// threads == 1 runs serially; threads > 1 parallelizes over window lengths (Optimized) or over
// (k, start) pairs (Direct).
void scan_features(const SeriesView& x, const std::vector<int32_t>& windows, ScanKind kind, bool keep_minmax,
                   int threads, Features& out, ScanBuffers& buf);

}  // namespace sf

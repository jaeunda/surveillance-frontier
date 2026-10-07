#include "sf/scan.hpp"

#include <omp.h>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace sf {

namespace {
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

void prepare(const SeriesView& x, const std::vector<int32_t>& windows, bool keep_minmax, Features& out) {
  const size_t cells = windows.size() * static_cast<size_t>(x.n);
  out.n = x.n;
  out.windows = windows;
  out.range.resize(cells);
  out.volume.resize(cells);
  out.min.resize(keep_minmax ? cells : 0);
  out.max.resize(keep_minmax ? cells : 0);
}

void fill_invalid(Features& out, int k, int64_t from) {
  const int64_t n = out.n;
  const bool mm = !out.min.empty();
  for (int64_t s = std::max<int64_t>(from, 0); s < n; ++s) {
    const size_t i = static_cast<size_t>(k) * n + s;
    out.range[i] = out.volume[i] = kNaN;
    if (mm) out.min[i] = out.max[i] = kNaN;
  }
}

// Reference scan: each (k, s) cell reads its whole window. Simple and obviously correct; used by the gate.
void scan_direct(const SeriesView& x, int threads, Features& out) {
  const int64_t n = x.n, K = out.K();
  const bool mm = !out.min.empty();
#pragma omp parallel for num_threads(threads) if (threads > 1) schedule(static)
  for (int64_t idx = 0; idx < K * n; ++idx) {
    const int64_t k = idx / n, s = idx - k * n, w = out.windows[k];
    if (s + w > n) {
      out.range[idx] = out.volume[idx] = kNaN;
      if (mm) out.min[idx] = out.max[idx] = kNaN;
      continue;
    }
    float mn = std::numeric_limits<float>::infinity(), mx = -mn;
    int64_t vs = 0;
    for (int64_t p = s; p < s + w; ++p) {
      mn = std::min(mn, x.low[p]);
      mx = std::max(mx, x.high[p]);
      vs += volume_fixed(x.volume[p]);
    }
    out.range[idx] = rel_range(mn, mx);
    out.volume[idx] = volume_feature(vs);
    if (mm) {
      out.min[idx] = mn;
      out.max[idx] = mx;
    }
  }
}

// Baseline scan: one pass over the series per window length (scan_starts over all valid starts).
void scan_optimized(const SeriesView& x, int threads, Features& out, ScanBuffers& buf) {
  const int64_t n = x.n;
  const int K = out.K();
  const bool mm = !out.min.empty();
  fixed_volumes(x, *std::max_element(out.windows.begin(), out.windows.end()), buf.vfix);
  const int slots = std::max(threads, 1);
  if (static_cast<int>(buf.qmin.size()) < slots) {
    buf.qmin.resize(slots);
    buf.qmax.resize(slots);
  }
  for (int t = 0; t < slots; ++t) {
    buf.qmin[t].resize(n);
    buf.qmax[t].resize(n);
  }
#pragma omp parallel for num_threads(threads) if (threads > 1) schedule(dynamic, 1)
  for (int k = 0; k < K; ++k) {
    const int slot = threads > 1 ? omp_get_thread_num() : 0;
    const int64_t w = out.windows[k], row = static_cast<int64_t>(k) * n;
    fill_invalid(out, k, n - w + 1);
    if (n < w) continue;
    scan_starts(x, buf.vfix.data(), w, n - w + 1, out.range.data() + row, out.volume.data() + row,
                mm ? out.min.data() + row : nullptr, mm ? out.max.data() + row : nullptr, buf.qmin[slot].data(),
                buf.qmax[slot].data());
  }
}
}  // namespace

void fixed_volumes(const SeriesView& x, int64_t max_window, std::vector<int64_t>& vfix) {
  // every window sum must fit into int64: max bar * max_window < 2^63
  const double limit = std::ldexp(1.0, 62 - kVolumeFracBits) / static_cast<double>(std::max<int64_t>(max_window, 1));
  vfix.resize(x.n);
  for (int64_t i = 0; i < x.n; ++i) {
    if (!(x.volume[i] >= 0.0f && x.volume[i] < limit)) throw std::overflow_error("volume out of fixed-point range");
    vfix[i] = volume_fixed(x.volume[i]);
  }
}

void scan_starts(const SeriesView& x, const int64_t* vfix, int64_t w, int64_t count, float* range, float* volume,
                 float* mn_out, float* mx_out, int32_t* qmin, int32_t* qmax) {
  // qmin/qmax are monotonic deques of bar indices whose front is the min(low) / max(high) of the current window;
  // vs is the exact fixed-point sum of the current window
  int64_t h1 = 0, t1 = 0, h2 = 0, t2 = 0, vs = 0;
  for (int64_t i = 0; i < count + w - 1; ++i) {
    // push bar i, dropping entries it dominates; then drop the front once it leaves the window [i - w + 1, i]
    while (t1 > h1 && x.low[qmin[t1 - 1]] >= x.low[i]) --t1;
    qmin[t1++] = static_cast<int32_t>(i);
    while (t2 > h2 && x.high[qmax[t2 - 1]] <= x.high[i]) --t2;
    qmax[t2++] = static_cast<int32_t>(i);
    if (qmin[h1] <= i - w) ++h1;
    if (qmax[h2] <= i - w) ++h2;
    vs += vfix[i];
    const int64_t s = i - w + 1;  // window that ends at bar i starts at s
    if (s >= 0) {
      const float mn = x.low[qmin[h1]], mx = x.high[qmax[h2]];
      range[s] = rel_range(mn, mx);
      volume[s] = volume_feature(vs);
      if (mn_out) mn_out[s] = mn;
      if (mx_out) mx_out[s] = mx;
      vs -= vfix[s];
    }
  }
}

void scan_features(const SeriesView& x, const std::vector<int32_t>& windows, ScanKind kind, bool keep_minmax,
                   int threads, Features& out, ScanBuffers& buf) {
  prepare(x, windows, keep_minmax, out);
  if (kind == ScanKind::Direct)
    scan_direct(x, threads, out);
  else
    scan_optimized(x, threads, out, buf);
}

}  // namespace sf

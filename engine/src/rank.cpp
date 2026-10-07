#include "sf/rank.hpp"

#include <omp.h>

#include <algorithm>

namespace sf {

namespace {
// out[s] = max(out[s], #{x >= values[s]}) over s < nk. Sort (value, index) ascending; equal values share the count
// nk - (position of the first equal value), which equals nk - lower_bound(sorted, v).
void count_at_least(const float* values, int64_t nk, int32_t* out, std::vector<RankItem>& items) {
  items.resize(nk);
  for (int64_t s = 0; s < nk; ++s) items[s] = {values[s], static_cast<int32_t>(s)};
  std::sort(items.begin(), items.end(), [](const auto& a, const auto& b) { return a.v < b.v; });
  int64_t first = 0;
  // walk the sorted order; `first` marks where the current run of equal values began
  for (int64_t p = 0; p < nk; ++p) {
    if (items[p].v != items[first].v) first = p;
    const int32_t c = static_cast<int32_t>(nk - first);
    int32_t& o = out[items[p].i];
    if (c > o) o = c;
  }
}

// Same result as count_at_least for every start whose count is <= cap; kAboveCap for all others.
// y = the (cap + 1)-th largest value; a start has #{x >= v} <= cap exactly when v > y.
void count_tail(const float* values, int64_t nk, int64_t cap, int32_t* out, std::vector<RankItem>& items) {
  if (cap >= nk) return count_at_least(values, nk, out, items);  // every start is in the tail
  items.resize(nk);
  for (int64_t s = 0; s < nk; ++s) items[s] = {values[s], static_cast<int32_t>(s)};
  const auto desc = [](const auto& a, const auto& b) { return a.v > b.v; };
  std::nth_element(items.begin(), items.begin() + cap, items.end(), desc);
  const float y = items[cap].v;
  const auto mid = std::partition(items.begin(), items.begin() + cap, [y](const auto& a) { return a.v > y; });
  for (auto it = mid; it != items.end(); ++it) out[it->i] = kAboveCap;
  std::sort(items.begin(), mid, desc);
  // descending order: every start of a run of equal values has count = position after the run
  const int64_t m = mid - items.begin();
  for (int64_t p = 0, q = 0; p < m; p = q) {
    while (q < m && items[q].v == items[p].v) ++q;
    for (int64_t r = p; r < q; ++r) {
      int32_t& o = out[items[r].i];
      if (q > o) o = static_cast<int32_t>(q);
    }
  }
}
}  // namespace

void rarity_counts(const Features& f, int threads, std::vector<int32_t>& counts, RankBuffers& buf, RankKind kind,
                   double s_floor) {
  const int64_t n = f.n;
  const int K = f.K();
  counts.assign(static_cast<size_t>(K) * n, kInvalidCount);
  const int slots = std::max(threads, 1);
  if (static_cast<int>(buf.items.size()) < slots) buf.items.resize(slots);
  // 2K independent rankings; the two features of one length write the same row, so a task is one length.
#pragma omp parallel for num_threads(threads) if (threads > 1) schedule(dynamic, 1)
  for (int k = 0; k < K; ++k) {
    auto& items = buf.items[threads > 1 ? omp_get_thread_num() : 0];
    const int64_t nk = valid_starts(n, f.windows[k]);
    const size_t row = static_cast<size_t>(k) * n;
    // start from 0, then take the max of the range count and the volume count
    std::fill(counts.begin() + row, counts.begin() + row + nk, 0);
    if (kind == RankKind::Sort) {
      count_at_least(f.range.data() + row, nk, counts.data() + row, items);
      count_at_least(f.volume.data() + row, nk, counts.data() + row, items);
    } else {
      const int64_t cap = count_cap(nk, s_floor);
      count_tail(f.range.data() + row, nk, cap, counts.data() + row, items);
      count_tail(f.volume.data() + row, nk, cap, counts.data() + row, items);
    }
  }
}

}  // namespace sf

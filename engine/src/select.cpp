#include "sf/select.hpp"

#include <algorithm>
#include <cmath>

#include "sf/rank.hpp"

namespace sf {

double rarity_score(int32_t count, int64_t nk) {
  return -std::log10(static_cast<double>(count) / static_cast<double>(nk));
}

bool is_peak(const int32_t* row, int64_t nk, int64_t s, int64_t r) {
  const int32_t c = row[s];
  for (int64_t j = std::max<int64_t>(0, s - r), end = std::min(nk - 1, s + r); j <= end; ++j)
    if (j != s && peak_blocked_by(row[j], j, c, s)) return false;
  return true;
}

void local_peaks(const std::vector<int32_t>& counts, int64_t n, const std::vector<int32_t>& windows, double s_min,
                 int threads, std::vector<int64_t>& flat, SelectBuffers& buf) {
  const int K = static_cast<int>(windows.size());
  buf.per_k.resize(std::max<size_t>(buf.per_k.size(), K));
#pragma omp parallel for num_threads(threads) if (threads > 1) schedule(dynamic, 1)
  for (int k = 0; k < K; ++k) {
    auto& out = buf.per_k[k];
    out.clear();
    const int64_t w = windows[k], nk = valid_starts(n, windows[k]);
    const int64_t thr = count_cap(nk, s_min), r = std::max<int64_t>(1, w / 2);
    const int32_t* row = counts.data() + static_cast<size_t>(k) * n;
    // keep s if it passes the S >= s_min threshold and no neighbour within +-r is rarer
    for (int64_t s = 0; s < nk; ++s)
      if (row[s] <= thr && is_peak(row, nk, s, r)) out.push_back(static_cast<int64_t>(k) * n + s);
  }
  flat.clear();
  for (int k = 0; k < K; ++k) flat.insert(flat.end(), buf.per_k[k].begin(), buf.per_k[k].end());
}

void order_candidates(const std::vector<int64_t>& flat, const std::vector<int32_t>& counts, int64_t n,
                      const std::vector<int32_t>& windows, std::vector<Episode>& ordered) {
  // flat index -> (k, start, count, S)
  ordered.clear();
  for (int64_t idx : flat) {
    const int k = static_cast<int>(idx / n);
    const int64_t s = idx - static_cast<int64_t>(k) * n;
    const double S = rarity_score(counts[idx], valid_starts(n, windows[k]));
    ordered.push_back({k, s, windows[k], counts[idx], S});
  }
  std::sort(ordered.begin(), ordered.end(), [](const Episode& a, const Episode& b) {
    return episode_before(a.S, a.k, a.start, b.S, b.k, b.start);
  });
}

void merge_episodes(const std::vector<Episode>& ordered, int64_t n, int K, const SelectConfig& cfg,
                    std::vector<Episode>& kept) {
  // greedy episode merge: walk from the rarest down, skip anything near an episode already kept
  kept.clear();
  for (const Episode& e : ordered) {
    if (e.k >= K || e.count > count_cap(valid_starts(n, e.window), cfg.s_min)) continue;
    const int64_t a0 = e.start, a1 = e.start + e.window;
    const bool separate = std::all_of(kept.begin(), kept.end(), [&](const Episode& o) {
      return episodes_separate(a0, a1, o.start, o.start + o.window, cfg.episode_gap);
    });
    if (separate) {
      kept.push_back(e);
      if (static_cast<int>(kept.size()) == cfg.top_k) break;
    }
  }
}

std::vector<Episode> select_top(const std::vector<int64_t>& flat, const std::vector<int32_t>& counts, int64_t n,
                                const std::vector<int32_t>& windows, const SelectConfig& cfg, SelectBuffers& buf) {
  order_candidates(flat, counts, n, windows, buf.ordered);
  std::vector<Episode> kept;
  merge_episodes(buf.ordered, n, static_cast<int>(windows.size()), cfg, kept);
  return kept;
}

}  // namespace sf

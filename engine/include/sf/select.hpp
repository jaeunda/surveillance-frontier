// Stage 3: candidate selection.
//
// A start is a candidate if its count is at most count_cap(N_k, s_min) (sf/rank.hpp) and it is the rarest point among
// its +-max(1, w/2) neighbours at the same length (ties go to the earlier start). Candidates are ordered by S
// descending (ties by k, then start); a candidate is dropped if it lies within `episode_gap` bars of, or overlaps, an
// already chosen one. The first top_k survivors are the reported episodes.
//
// Whether a start is a local peak does not depend on s_min, K or the gap, so several policies can share one peak
// list: peaks are found once at the smallest s_min, ordered once, and each policy filters and merges that list.
#pragma once

#include <cstdint>
#include <vector>

namespace sf {

struct Episode {
  int k;
  int64_t start;
  int32_t window;
  int32_t count;
  double S;
};

struct SelectConfig {
  double s_min = 2.0;
  int top_k = 20;
  int64_t episode_gap = 60;
};

// Reusable buffers so repeated selections do not allocate.
struct SelectBuffers {
  std::vector<std::vector<int64_t>> per_k;
  std::vector<Episode> ordered;
};

// Start s of a row (nk valid starts, count row `row`) is a local peak within +-r.
bool is_peak(const int32_t* row, int64_t nk, int64_t s, int64_t r);

// Flat indices k * n + s of candidates, in increasing order.
void local_peaks(const std::vector<int32_t>& counts, int64_t n, const std::vector<int32_t>& windows, double s_min,
                 int threads, std::vector<int64_t>& flat, SelectBuffers& buf);

// Candidates (flat indices, any order) as episodes ordered by S descending, then k, then start.
void order_candidates(const std::vector<int64_t>& flat, const std::vector<int32_t>& counts, int64_t n,
                      const std::vector<int32_t>& windows, std::vector<Episode>& ordered);

// One policy from an ordered candidate list: keep candidates with k < K and count <= count_cap(N_k, s_min), then the
// greedy episode merge.
void merge_episodes(const std::vector<Episode>& ordered, int64_t n, int K, const SelectConfig& cfg,
                    std::vector<Episode>& kept);

std::vector<Episode> select_top(const std::vector<int64_t>& flat, const std::vector<int32_t>& counts, int64_t n,
                                const std::vector<int32_t>& windows, const SelectConfig& cfg, SelectBuffers& buf);

}  // namespace sf

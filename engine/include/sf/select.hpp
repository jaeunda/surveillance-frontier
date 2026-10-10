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

#include "sf/hd.hpp"

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

// The rarity S = -log10(count / nk) of a count; the only definition (candidate ordering and the S tables use it).
double rarity_score(int32_t count, int64_t nk);

// Candidate order: S descending, then k, then start. Shared by every back end.
SF_HD bool episode_before(double s_a, int k_a, int64_t start_a, double s_b, int k_b, int64_t start_b) {
  if (s_a != s_b) return s_a > s_b;
  if (k_a != k_b) return k_a < k_b;
  return start_a < start_b;
}

// Local-peak tie rule: neighbour j (count cj) keeps start s (count c) from being a peak if it is rarer, or equally
// rare and earlier.
SF_HD bool peak_blocked_by(int32_t cj, int64_t j, int32_t c, int64_t s) { return cj < c || (cj == c && j < s); }

// Greedy merge: a candidate [a0, a1) is separate from a kept episode [o0, o1) if the gap between them is >= gap.
SF_HD bool episodes_separate(int64_t a0, int64_t a1, int64_t o0, int64_t o1, int64_t gap) {
  return (a0 > o0 ? a0 : o0) - (a1 < o1 ? a1 : o1) >= gap;
}

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

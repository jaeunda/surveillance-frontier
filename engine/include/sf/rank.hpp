// Stage 2: per-length rarity counts.
//
// Among the N_k = n - w + 1 valid starts of window length k, c(v) = #{x >= v} (at least 1), computed for range and
// volume separately. The combined count is c = max(c_range, c_volume) and the rarity is S = -log10(c / N_k):
// "S = 3" means both features are in the top 0.1% of all intervals of that length. Counts are integers, so every
// implementation must agree exactly.
//
// Selection only ever looks at starts with c <= floor(N_k * 10^-s_min) (sf/select.hpp), and a start outside that set
// cannot outrank one inside it. Tail ranking uses this: it finds each feature's threshold value by selection
// (O(N_k)) and sorts only the values above it. Counts are exact up to the cap and kAboveCap beyond it, which gives the
// same candidates and episodes as a full sort for every s_min >= the s_floor that set the cap.
#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "sf/scan.hpp"

namespace sf {

constexpr int32_t kInvalidCount = std::numeric_limits<int32_t>::max();
constexpr int32_t kAboveCap = kInvalidCount - 1;

enum class RankKind {
  Sort,  // full sort of every row: exact counts everywhere (reference)
  Tail,  // selection + sort of the tail: exact counts up to the cap, kAboveCap beyond
};

inline int64_t valid_starts(int64_t n, int32_t w) { return n >= w ? n - w + 1 : 0; }

// The largest count that still passes S >= s: floor(N_k * 10^-s). Selection and tail ranking both use this.
inline int64_t count_cap(int64_t nk, double s) {
  return static_cast<int64_t>(std::floor(static_cast<double>(nk) * std::pow(10.0, -s)));
}

struct RankItem {
  float v;
  int32_t i;
};

struct RankBuffers {
  std::vector<std::vector<RankItem>> items;  // per thread slot, [n]
};

// counts[k * n + s]; kInvalidCount for invalid starts. Features must be finite on valid starts.
// Tail: the cap of row k is count_cap(N_k, s_floor).
void rarity_counts(const Features& f, int threads, std::vector<int32_t>& counts, RankBuffers& buf,
                   RankKind kind = RankKind::Sort, double s_floor = 0.0);

}  // namespace sf

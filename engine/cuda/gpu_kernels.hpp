// Device code of the GPU back end (Phase 2 SPEC §4, §6). Every function here is SF_HD: nvcc compiles it for the
// device, and the host emulation (gpu_backend_emul.cpp) compiles the same source for the CPU, so the exactness tests
// run the real kernel bodies without a GPU. Kernels are flat: one body call per index, no shared memory or barriers.
//
// One microbatch holds b trials. Per trial and window row k, the pipeline is
//   rows      c0, count of the starts whose windows overlap the event
//   features  range and volume of those starts, each window read directly (exact min/max, fixed-point volume)
//   tails     changed values sorted, merged with the base head of the row: the exact tail (start, count) of each
//             feature, written as runs of equal values (no dense count array, no truncation)
//   both      both tails sorted by start and intersected: the both-set with combined count, sorted by start
//   peaks     local-peak test on the both-set by scanning neighbours within +-r (starts outside it have count
//             kAboveCap and can never block)
//   compact   peaks as candidates with S from the host table
//   order     (G-inc) candidates sorted by S desc, k, start; merge: greedy episode merge per (trial, policy), then the
//             detection bits. G-hyb does order and merge on the host with the CPU code.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#include "sf/batch.hpp"
#include "sf/hd.hpp"
#include "sf/select.hpp"

namespace sf {
namespace gpu {

// Negative controls of gate item 2 (SPEC §5.2): each must be detected by the checker on its witness.
enum Fault : int32_t {
  kFaultNone = 0,
  kFaultDeviceLog10 = 1,   // S computed on the device instead of the host table
  kFaultReversedTie = 2,   // local-peak ties go to the later start
  kFaultCapMinusOne = 3,   // tail cap one too small
  kFaultDirectFloat = 4,   // int64 -> float volume directly (single rounding)
  kFaultHalfEven = 5,      // fixed-point volume rounded half to even
  kFaultFastMath = 6,      // approximate division in the range feature (what --use_fast_math substitutes)
};

enum ErrorBit : int32_t { kErrDraw = 1, kErrHeadShort = 2 };

struct RankItemD {  // same layout as sf::RankItem
  float v;
  int32_t i;
};
struct TailD {
  int32_t start, count;
};
struct CandD {
  double S;
  int32_t start, count, k, pad;
};
struct EpisodeD {  // base episodes: detection compares k and start only
  int64_t start;
  int32_t k, pad;
};

struct RowD {
  int32_t w, head_len;
  int64_t nk, cap, r;
  int64_t head_off;  // into head_range / head_volume
  int64_t feat_off;  // into the per-trial feature slots
  int32_t feat_cap;  // max_length + w - 1
  int32_t pad;
  int64_t tail_off;  // into the per-trial tail / both / candidate slots
  int64_t s_off;     // into s_table
};
struct CellD {
  int32_t length, kind;  // kind: 0 both, 1 price, 2 volume (EventKind order)
  int64_t factor_off;
  float vol_factor;
  int32_t pad;
};
struct PolicyD {
  int32_t K, top_k;
  int64_t gap;
  int64_t base_off;
  int32_t base_len, pad;
};

// Resident state and the microbatch scratch, all device pointers.
struct Args {
  int32_t K, P, top_k_max, fault;
  int64_t n;
  int64_t F;     // feature slots per trial = sum_k feat_cap
  int64_t T;     // tail / both / candidate slots per trial = sum_k cap
  const float *low, *high, *volume;
  const RowD* rows;
  const RankItemD *head_range, *head_volume;
  const double* s_table;
  const int32_t* policy_caps;  // [P * K]
  const PolicyD* policies;
  const EpisodeD* base_eps;
  const CellD* cells;
  const float* factors;
  const TrialSpace* space;
  int64_t g0;
  // scratch, sized for the microbatch
  int32_t* cell;
  int64_t* start;
  int64_t* c0;       // [b * K]
  int32_t* cnt;      // [b * K]
  float *f_range, *f_volume;  // [b * F]
  RankItemD* chg;    // [b * 2 * F]
  TailD* tail;       // [b * 2 * T]
  int32_t* tail_len; // [b * 2 * K]
  TailD* both;       // [b * T]
  int32_t* both_len; // [b * K]
  uint8_t* peak;     // [b * T]
  CandD* cand;       // [b * T]
  int32_t* cand_len; // [b]
  int32_t* kept;     // [b * P * top_k_max], indices into the trial's candidates
  int32_t* kept_len; // [b * P]
  uint8_t *hit, *overlap;  // [b * P]
  int32_t* error;    // [1], ErrorBit flags
};

// ---- exact arithmetic (SPEC §4) ----

SF_HD float fmul_rn(float a, float b) {
#if defined(__CUDA_ARCH__)
  return __fmul_rn(a, b);
#else
  return a * b;
#endif
}
SF_HD float fsub_rn(float a, float b) {
#if defined(__CUDA_ARCH__)
  return __fsub_rn(a, b);
#else
  return a - b;
#endif
}
SF_HD float fdiv_rn(float a, float b) {
#if defined(__CUDA_ARCH__)
  return __fdiv_rn(a, b);
#else
  return a / b;
#endif
}
SF_HD float fdiv_fast(float a, float b) {
#if defined(__CUDA_ARCH__)
  return __fdividef(a, b);
#else
  return a * (1.0f / b);
#endif
}
SF_HD double ll2double_rn(int64_t x) {
#if defined(__CUDA_ARCH__)
  return __ll2double_rn(x);
#else
  return static_cast<double>(x);
#endif
}
SF_HD float double2float_rn(double x) {
#if defined(__CUDA_ARCH__)
  return __double2float_rn(x);
#else
  return static_cast<float>(x);
#endif
}
SF_HD float ll2float_rn(int64_t x) {
#if defined(__CUDA_ARCH__)
  return __ll2float_rn(x);
#else
  return static_cast<float>(x);
#endif
}
SF_HD float nan_f() {
  const uint32_t bits = 0x7fc00000u;  // std::numeric_limits<float>::quiet_NaN()
  float f;
  memcpy(&f, &bits, sizeof f);
  return f;
}

// llround(ldexp(double(v), 24)): the exact double, rounded half away from zero (SPEC §4).
SF_HD int64_t fixed_volume(float v, int32_t fault) {
  const double x = static_cast<double>(v) * 16777216.0;  // exact: a float times a power of two
  if (fault == kFaultHalfEven) {
#if defined(__CUDA_ARCH__)
    return __double2ll_rn(x);
#else
    return static_cast<int64_t>(std::nearbyint(x));  // default rounding mode: half to even
#endif
  }
  const double t = trunc(x);
  int64_t r = static_cast<int64_t>(t);
  const double f = x - t;  // exact
  if (f >= 0.5) ++r;
  else if (f <= -0.5) --r;
  return r;
}

SF_HD float range_feature(float mn, float mx, int32_t fault) {
  if (!(mn > 0.0f)) return nan_f();
  return fault == kFaultFastMath ? fdiv_fast(fsub_rn(mx, mn), mn) : fdiv_rn(fsub_rn(mx, mn), mn);
}

SF_HD float volume_feature_d(int64_t sum, int32_t fault) {
  if (fault == kFaultDirectFloat) return ll2float_rn(sum) * 5.9604644775390625e-08f;  // 2^-24
  return double2float_rn(ll2double_rn(sum) * 5.9604644775390625e-08);              // exact scaling
}

// ---- small sorts (any order of equal keys is fine where they are used) ----

// Sorts a[0, n) so that no element comes "before" an earlier one.
template <typename T, typename Before>
SF_HD void heap_sort(T* a, int64_t n, Before before) {
  // max-heap under "before": the root is the element that belongs last
  for (int64_t i = n / 2 - 1; i >= 0; --i) {
    for (int64_t r = i;;) {
      int64_t c = 2 * r + 1;
      if (c >= n) break;
      if (c + 1 < n && before(a[c], a[c + 1])) ++c;
      if (!before(a[r], a[c])) break;
      T t = a[r]; a[r] = a[c]; a[c] = t;
      r = c;
    }
  }
  for (int64_t end = n - 1; end > 0; --end) {
    T t = a[0]; a[0] = a[end]; a[end] = t;
    for (int64_t r = 0;;) {
      int64_t c = 2 * r + 1;
      if (c >= end) break;
      if (c + 1 < end && before(a[c], a[c + 1])) ++c;
      if (!before(a[r], a[c])) break;
      T u = a[r]; a[r] = a[c]; a[c] = u;
      r = c;
    }
  }
}

struct ValueDesc {
  SF_HD bool operator()(const RankItemD& a, const RankItemD& b) const { return a.v > b.v; }
};
struct StartAsc {
  SF_HD bool operator()(const TailD& a, const TailD& b) const { return a.start < b.start; }
};
struct CandBefore {
  SF_HD bool operator()(const CandD& a, const CandD& b) const {
    return episode_before(a.S, a.k, a.start, b.S, b.k, b.start);
  }
};

SF_HD void set_error(int32_t* e, int32_t bit) {
#if defined(__CUDA_ARCH__)
  atomicOr(e, bit);
#else
  __atomic_fetch_or(e, bit, __ATOMIC_RELAXED);
#endif
}

SF_HD int row_of(const RowD* rows, int K, int64_t off, bool tail) {
  int k = 0;
  while (k + 1 < K && (tail ? rows[k + 1].tail_off : rows[k + 1].feat_off) <= off) ++k;
  return k;
}

// ---- kernel bodies; index i covers the stated domain ----

// i < b: trial cell and start from the trial space
SF_HD void k_resolve(int64_t i, const Args& a) {
  int64_t j;
  if (!resolve_trial(*a.space, a.g0 + i, a.cell[i], j, a.start[i])) set_error(a.error, kErrDraw);
}

// Gate item 8 on the device: i < n draws of an arbitrary trial space (into cell/start), or Philox blocks.
struct StreamArgs {
  const TrialSpace* space;
  int64_t g0;
  int32_t* cell;
  int64_t* start;
  const uint32_t* ck;
  uint32_t* out;
  int32_t* error;
};
SF_HD void k_stream(int64_t i, const StreamArgs& a) {
  int64_t j;
  if (!resolve_trial(*a.space, a.g0 + i, a.cell[i], j, a.start[i])) set_error(a.error, kErrDraw);
}
SF_HD void k_philox(int64_t i, const StreamArgs& a) {
  const Philox4x32 r = philox4x32_10(Philox4x32{{a.ck[6 * i], a.ck[6 * i + 1], a.ck[6 * i + 2], a.ck[6 * i + 3]}},
                                     a.ck[6 * i + 4], a.ck[6 * i + 5]);
  for (int w = 0; w < 4; ++w) a.out[4 * i + w] = r.v[w];
}

// i < b * K
SF_HD void k_rows(int64_t i, const Args& a) {
  const int64_t t = i / a.K;
  const int k = static_cast<int>(i - t * a.K);
  const RowD& row = a.rows[k];
  const int64_t s0 = a.start[t], L = a.cells[a.cell[t]].length;
  const int64_t c0 = s0 - row.w + 1 > 0 ? s0 - row.w + 1 : 0;
  const int64_t c1 = s0 + L - 1 < row.nk - 1 ? s0 + L - 1 : row.nk - 1;
  a.c0[i] = c0;
  a.cnt[i] = static_cast<int32_t>(c1 - c0 + 1);
}

// i < b * F: one changed start of one row, its window read directly
SF_HD void k_features(int64_t i, const Args& a) {
  const int64_t t = i / a.F, off = i - t * a.F;
  const int k = row_of(a.rows, a.K, off, false);
  const RowD& row = a.rows[k];
  const int64_t j = off - row.feat_off;
  if (j >= a.cnt[t * a.K + k]) return;
  const int64_t s = a.c0[t * a.K + k] + j;
  const CellD& cell = a.cells[a.cell[t]];
  const int64_t e0 = a.start[t], e1 = e0 + cell.length;
  const bool price = cell.kind != 2, vol = cell.kind != 1;
  float mn = INFINITY, mx = -INFINITY;
  int64_t vs = 0;
  for (int64_t p = s; p < s + row.w; ++p) {
    float lo = a.low[p], hi = a.high[p], v = a.volume[p];
    if (p >= e0 && p < e1) {
      if (price) {
        const float f = a.factors[cell.factor_off + (p - e0)];
        lo = fmul_rn(lo, f);
        hi = fmul_rn(hi, f);
      }
      if (vol) v = fmul_rn(v, cell.vol_factor);
    }
    mn = lo < mn ? lo : mn;
    mx = hi > mx ? hi : mx;
    vs += fixed_volume(v, a.fault);
  }
  const float r = range_feature(mn, mx, a.fault), v = volume_feature_d(vs, a.fault);
  a.f_range[i] = r;
  a.f_volume[i] = v;
  a.chg[(t * 2) * a.F + off] = RankItemD{r, static_cast<int32_t>(s)};
  a.chg[(t * 2 + 1) * a.F + off] = RankItemD{v, static_cast<int32_t>(s)};
}

// i < b * 2 * K: exact tail of one feature of one row (Evaluator::merge_tail, emitted as runs)
SF_HD void k_tails(int64_t i, const Args& a) {
  const int64_t t = i / (2 * a.K);
  const int64_t rem = i - t * 2 * a.K;
  const int f = static_cast<int>(rem / a.K), k = static_cast<int>(rem % a.K);
  const RowD& row = a.rows[k];
  const int64_t c0 = a.c0[t * a.K + k], cnt = a.cnt[t * a.K + k], c1 = c0 + cnt - 1;
  RankItemD* chg = a.chg + (t * 2 + f) * a.F + row.feat_off;
  heap_sort(chg, cnt, ValueDesc{});  // descending values
  const RankItemD* head = (f == 0 ? a.head_range : a.head_volume) + row.head_off;
  int64_t cap = row.cap;
  if (a.fault == kFaultCapMinusOne && cap > 0) cap -= 1;
  const int64_t want = cap + 1 < row.nk ? cap + 1 : row.nk;
  TailD* out = a.tail + (t * 2 + f) * a.T + row.tail_off;
  int64_t h = 0, c = 0, run = 0;
  float run_v = 0;
  for (int64_t pos = 0; pos < want; ++pos) {
    while (h < row.head_len && head[h].i >= c0 && head[h].i <= c1) ++h;  // replaced by the changed values
    RankItemD it;
    if (h < row.head_len && (c == cnt || head[h].v >= chg[c].v)) it = head[h++];
    else if (c < cnt) it = chg[c++];
    else {
      set_error(a.error, kErrHeadShort);
      a.tail_len[(t * 2 + f) * a.K + k] = 0;
      return;
    }
    if (pos > 0 && it.v != run_v) {  // the previous run ends here: its count is pos
      for (int64_t r = run; r < pos; ++r) out[r].count = static_cast<int32_t>(pos);
      run = pos;
    }
    run_v = it.v;
    if (pos < row.cap) out[pos] = TailD{it.i, 0};
  }
  int64_t m;
  if (cap < row.nk) {
    m = run;  // the last run holds the (cap + 1)-th value: its starts have count > cap
  } else {
    for (int64_t r = run; r < want; ++r) out[r].count = static_cast<int32_t>(want);
    m = want;
  }
  a.tail_len[(t * 2 + f) * a.K + k] = static_cast<int32_t>(m);
}

// i < b * K: both-set of one row, sorted by start
SF_HD void k_both(int64_t i, const Args& a) {
  const int64_t t = i / a.K;
  const int k = static_cast<int>(i - t * a.K);
  const RowD& row = a.rows[k];
  TailD* tr = a.tail + (t * 2) * a.T + row.tail_off;
  TailD* tv = a.tail + (t * 2 + 1) * a.T + row.tail_off;
  const int64_t nr = a.tail_len[(t * 2) * a.K + k], nv = a.tail_len[(t * 2 + 1) * a.K + k];
  heap_sort(tr, nr, StartAsc{});
  heap_sort(tv, nv, StartAsc{});
  TailD* out = a.both + t * a.T + row.tail_off;
  int64_t m = 0;
  for (int64_t x = 0, y = 0; x < nr && y < nv;) {
    if (tr[x].start < tv[y].start) ++x;
    else if (tv[y].start < tr[x].start) ++y;
    else {
      out[m++] = TailD{tr[x].start, tr[x].count > tv[y].count ? tr[x].count : tv[y].count};
      ++x, ++y;
    }
  }
  a.both_len[i] = static_cast<int32_t>(m);
}

// i < b * T: local-peak test of one both-set member
SF_HD void k_peaks(int64_t i, const Args& a) {
  const int64_t t = i / a.T, off = i - t * a.T;
  const int k = row_of(a.rows, a.K, off, true);
  const RowD& row = a.rows[k];
  const int64_t idx = off - row.tail_off, len = a.both_len[t * a.K + k];
  if (idx >= len) return;
  const TailD* both = a.both + t * a.T + row.tail_off;
  const int64_t s = both[idx].start;
  const int32_t c = both[idx].count;
  const bool reversed = a.fault == kFaultReversedTie;
  bool peak = true;
  for (int64_t j = idx - 1; peak && j >= 0 && both[j].start >= s - row.r; --j) {
    const int32_t cj = both[j].count;
    peak = reversed ? !(cj < c || (cj == c && both[j].start > s)) : !peak_blocked_by(cj, both[j].start, c, s);
  }
  for (int64_t j = idx + 1; peak && j < len && both[j].start <= s + row.r; ++j) {
    const int32_t cj = both[j].count;
    peak = reversed ? !(cj < c || (cj == c && both[j].start > s)) : !peak_blocked_by(cj, both[j].start, c, s);
  }
  a.peak[i] = peak;
}

// i < b: candidates of one trial (rows in order, starts ascending)
SF_HD void k_compact(int64_t i, const Args& a) {
  CandD* out = a.cand + i * a.T;
  int32_t m = 0;
  for (int k = 0; k < a.K; ++k) {
    const RowD& row = a.rows[k];
    const TailD* both = a.both + i * a.T + row.tail_off;
    const uint8_t* pk = a.peak + i * a.T + row.tail_off;
    for (int64_t x = 0, len = a.both_len[i * a.K + k]; x < len; ++x) {
      if (!pk[x]) continue;
      const int32_t c = both[x].count;
      double S;
      if (a.fault == kFaultDeviceLog10) {
#if defined(__CUDA_ARCH__)
        S = -log10(static_cast<double>(c) / static_cast<double>(row.nk));
#else
        S = -(std::log10(static_cast<double>(c)) - std::log10(static_cast<double>(row.nk)));  // host stand-in
#endif
      } else {
        S = a.s_table[row.s_off + c];
      }
      out[m++] = CandD{S, both[x].start, c, k, 0};
    }
  }
  a.cand_len[i] = m;
}

// i < b (G-inc): candidates in episode order
SF_HD void k_order(int64_t i, const Args& a) { heap_sort(a.cand + i * a.T, a.cand_len[i], CandBefore{}); }

// i < b * P (G-inc): greedy merge of one policy, then its detection bits (merge_episodes, detection)
SF_HD void k_merge(int64_t i, const Args& a) {
  const int64_t t = i / a.P;
  const int p = static_cast<int>(i - t * a.P);
  const PolicyD& pol = a.policies[p];
  const CandD* cand = a.cand + t * a.T;
  int32_t* kept = a.kept + i * a.top_k_max;
  int32_t m = 0;
  for (int32_t x = 0, len = a.cand_len[t]; x < len && m < pol.top_k; ++x) {
    const CandD& e = cand[x];
    if (e.k >= pol.K || e.count > a.policy_caps[p * a.K + e.k]) continue;
    const int64_t a0 = e.start, a1 = e.start + a.rows[e.k].w;
    bool separate = true;
    for (int32_t y = 0; separate && y < m; ++y) {
      const CandD& o = cand[kept[y]];
      separate = episodes_separate(a0, a1, o.start, o.start + a.rows[o.k].w, pol.gap);
    }
    if (separate) kept[m++] = x;
  }
  a.kept_len[i] = m;
  const int64_t e0 = a.start[t], len = a.cells[a.cell[t]].length;
  bool hit = false, overlap = false;
  for (int32_t y = 0; y < m; ++y) {
    const CandD& e = cand[kept[y]];
    const double centre = static_cast<double>(static_cast<int64_t>(e.start)) + a.rows[e.k].w / 2.0;
    if (centre < static_cast<double>(e0) || centre >= static_cast<double>(e0 + len)) continue;
    overlap = true;
    bool known = false;
    for (int32_t z = 0; !known && z < pol.base_len; ++z) {
      const EpisodeD& o = a.base_eps[pol.base_off + z];
      known = o.k == e.k && o.start == e.start;
    }
    hit |= !known;
  }
  a.hit[i] = hit;
  a.overlap[i] = overlap;
}

}  // namespace gpu
}  // namespace sf

#include "sf/evaluate.hpp"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <exception>
#include <stdexcept>

namespace sf {

namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
const auto kDesc = [](const RankItem& a, const RankItem& b) { return a.v > b.v; };

// The largest m values of a feature row, descending.
std::vector<RankItem> head_of(const float* values, int64_t nk, int64_t m) {
  std::vector<RankItem> items(nk);
  for (int64_t s = 0; s < nk; ++s) items[s] = {values[s], static_cast<int32_t>(s)};
  std::partial_sort(items.begin(), items.begin() + m, items.end(), kDesc);
  items.resize(m);
  return items;
}
}  // namespace

const char* method_name(Method m) {
  switch (m) {
    case Method::Full: return "full";
    case Method::Shared: return "shared";
    case Method::Tail: return "tail";
    case Method::Incremental: return "incremental";
  }
  return "?";
}

Method parse_method(const std::string& name) {
  for (Method m : {Method::Full, Method::Shared, Method::Tail, Method::Incremental})
    if (name == method_name(m)) return m;
  throw std::invalid_argument("unknown method " + name);
}

std::vector<Policy> policy_grid(const std::vector<int>& ks, const std::vector<double>& s_mins,
                                const std::vector<int64_t>& gaps, int top_k) {
  std::vector<Policy> out;
  for (int k : ks)
    for (double s : s_mins)
      for (int64_t g : gaps) out.push_back({k, {s, top_k, g}});
  return out;
}

BaseState prepare_base(const Series& base, const std::vector<int32_t>& ladder, const std::vector<Policy>& policies,
                       int32_t max_length, bool incremental, int threads, double s_floor_override) {
  if (policies.empty()) throw std::invalid_argument("no policies");
  BaseState b;
  b.base = &base;
  b.policies = policies;
  b.max_length = max_length;
  int k_max = 0;
  b.s_floor = policies[0].select.s_min;
  for (const auto& p : policies) {
    if (p.K < 1 || p.K > static_cast<int>(ladder.size())) throw std::invalid_argument("policy K out of range");
    k_max = std::max(k_max, p.K);
    b.s_floor = std::min(b.s_floor, p.select.s_min);
  }
  if (s_floor_override > 0) b.s_floor = s_floor_override;
  b.windows.assign(ladder.begin(), ladder.begin() + k_max);
  for (int32_t w : b.windows) {
    const int64_t nk = valid_starts(base.size(), w), cap = std::min(count_cap(nk, b.s_floor), nk);
    b.caps.push_back(cap);
    b.s_table.emplace_back(cap + 1, 0.0);
    for (int64_t c = 1; c <= cap; ++c) b.s_table.back()[c] = rarity_score(static_cast<int32_t>(c), nk);
  }
  for (const auto& p : policies) {
    b.policy_caps.emplace_back();
    for (int32_t w : b.windows)
      b.policy_caps.back().push_back(static_cast<int32_t>(count_cap(valid_starts(base.size(), w), p.select.s_min)));
  }

  // each policy's episodes on the unchanged series (a q = 0 event changes nothing)
  b.episodes = Evaluator(b, Method::Tail).run({0, 1, 0.0}, 0.0, threads);

  if (incremental) {
    const int64_t n = base.size();
    if (max_length < 1 || max_length > n) throw std::invalid_argument("max_length out of range");
    Features f;
    ScanBuffers sb;
    scan_features(view(base), b.windows, ScanKind::Optimized, false, threads, f, sb);
    b.heads.resize(b.windows.size());
#pragma omp parallel for num_threads(threads) if (threads > 1) schedule(dynamic, 1)
    for (int k = 0; k < static_cast<int>(b.windows.size()); ++k) {
      const int32_t w = b.windows[k];
      const int64_t nk = valid_starts(n, w), row = static_cast<int64_t>(k) * n;
      // (cap + 1) values survive in the head even if every replaced start was in it
      const int64_t m = std::min<int64_t>(nk, count_cap(nk, b.s_floor) + 1 + max_length + w - 1);
      b.heads[k].range = head_of(f.range.data() + row, nk, m);
      b.heads[k].volume = head_of(f.volume.data() + row, nk, m);
    }
  }
  return b;
}

Evaluator::Evaluator(const BaseState& base, Method method, const EvalOptions& options)
    : b_(base), method_(method), opt_(options), work_(*base.base), out_(base.policies.size()) {
  if (method_ == Method::Full) {
    for (const auto& p : b_.policies) {
      DetectorConfig cfg;
      cfg.windows.assign(b_.windows.begin(), b_.windows.begin() + p.K);
      cfg.select = p.select;
      configs_.push_back(cfg);
    }
  } else if (method_ == Method::Incremental) {
    if (b_.heads.size() != b_.windows.size()) throw std::invalid_argument("base state has no incremental heads");
    if (!opt_.sparse_both) cc_.assign(b_.windows.size() * static_cast<size_t>(work_.size()), kAboveCap);
    rows_.resize(b_.windows.size());
  }
}

const std::vector<std::vector<Episode>>& Evaluator::run(const TestEvent& e, double median_range_L, int threads,
                                                        StageTimes* times) {
  apply_event(*b_.base, e, median_range_L, work_);
  evaluate(e, threads, times);
  return out_;
}

const std::vector<std::vector<Episode>>& Evaluator::run(const TestEvent& e, const EventTable& table, int threads,
                                                        StageTimes* times) {
  apply_event(*b_.base, e, table, work_);
  evaluate(e, threads, times);
  return out_;
}

void Evaluator::evaluate(const TestEvent& e, int threads, StageTimes* times) {
  StageTimes st;
  counts_ = TrialCounts{};
  switch (method_) {
    case Method::Full: run_full(threads, st); break;
    case Method::Shared: run_shared(e, RankKind::Sort, threads, st); break;
    case Method::Tail: run_shared(e, RankKind::Tail, threads, st); break;
    case Method::Incremental: run_incremental(e, threads, st); break;
  }
  if (opt_.trace) {
    trace_.episodes = out_;
    trace_.has_stages = method_ != Method::Full;
    if (method_ != Method::Full) trace_.ordered = ws_.select.ordered;
  }
  restore(*b_.base, e, work_);
  if (times) *times = st;
}

void Evaluator::run_full(int threads, StageTimes& st) {
  for (size_t p = 0; p < configs_.size(); ++p) {
    StageTimes one;
    out_[p] = detect(view(work_), configs_[p], threads, ws_, &one);
    st.scan_ms += one.scan_ms;
    st.rank_ms += one.rank_ms;
    st.select_ms += one.select_ms;
  }
}

void Evaluator::merge_policies() {
  if (!opt_.cap_table) {
    for (size_t p = 0; p < b_.policies.size(); ++p)
      merge_episodes(ws_.select.ordered, work_.size(), b_.policies[p].K, b_.policies[p].select, out_[p]);
    return;
  }
  // PC3: merge_episodes with the count caps from the table
  for (size_t p = 0; p < b_.policies.size(); ++p) {
    const Policy& pol = b_.policies[p];
    const int32_t* caps = b_.policy_caps[p].data();
    auto& kept = out_[p];
    kept.clear();
    for (const Episode& e : ws_.select.ordered) {
      if (e.k >= pol.K || e.count > caps[e.k]) continue;
      const int64_t a0 = e.start, a1 = e.start + e.window;
      const bool separate = std::all_of(kept.begin(), kept.end(), [&](const Episode& o) {
        return episodes_separate(a0, a1, o.start, o.start + o.window, pol.select.episode_gap);
      });
      if (separate) {
        kept.push_back(e);
        if (static_cast<int>(kept.size()) == pol.select.top_k) break;
      }
    }
  }
}

// PC4: peaks of one row from its both-set sorted by start; starts outside the set have count kAboveCap and can never
// block a start inside it, so only set members within +-r are compared.
void Evaluator::sparse_peaks(Row& row, int k, int64_t nk, int64_t r) {
  (void)nk;  // every member is a valid start, so the +-r scan never passes nk - 1
  row.peak_eps.clear();
  row.neighbours = 0;
  const auto& both = row.both_sparse;
  const int32_t w = b_.windows[k];
  for (size_t i = 0; i < both.size(); ++i) {
    const int64_t s = both[i].start;
    const int32_t c = both[i].count;
    bool peak = true;
    for (size_t j = i; peak && j-- > 0 && both[j].start >= s - r;) {
      row.neighbours += opt_.diag;
      peak = !peak_blocked_by(both[j].count, both[j].start, c, s);
    }
    for (size_t j = i + 1; peak && j < both.size() && both[j].start <= s + r; ++j) {
      row.neighbours += opt_.diag;
      peak = !peak_blocked_by(both[j].count, both[j].start, c, s);
    }
    if (peak) {
      const double S = opt_.s_table ? b_.s_table[k][c] : rarity_score(c, valid_starts(work_.size(), w));
      row.peak_eps.push_back({k, s, w, c, S});
    }
  }
}

void Evaluator::order_sparse() {
  auto& ordered = ws_.select.ordered;
  ordered.clear();
  for (const Row& row : rows_) ordered.insert(ordered.end(), row.peak_eps.begin(), row.peak_eps.end());
}

// Features of the starts whose windows overlap the event and the both-set, from full feature and count rows.
void Evaluator::trace_rows(const TestEvent& e) {
  const int64_t n = work_.size();
  const int K = static_cast<int>(b_.windows.size());
  trace_.rows.assign(K, {});
  for (int k = 0; k < K; ++k) {
    auto& r = trace_.rows[k];
    const int64_t w = b_.windows[k], nk = valid_starts(n, b_.windows[k]);
    r.c0 = std::max<int64_t>(0, e.start - w + 1);
    r.c1 = std::min<int64_t>(nk - 1, e.start + e.length - 1);
    const size_t row = static_cast<size_t>(k) * n;
    r.range.assign(ws_.features.range.begin() + row + r.c0, ws_.features.range.begin() + row + r.c1 + 1);
    r.volume.assign(ws_.features.volume.begin() + row + r.c0, ws_.features.volume.begin() + row + r.c1 + 1);
    for (int64_t s = 0; s < nk; ++s) {
      const int32_t c = ws_.counts[row + s];
      if (c <= b_.caps[k]) {
        r.both_start.push_back(static_cast<int32_t>(s));
        r.both_count.push_back(c);
      }
    }
  }
}

void Evaluator::run_shared(const TestEvent& e, RankKind kind, int threads, StageTimes& st) {
  const SeriesView x = view(work_);
  auto t0 = Clock::now();
  scan_features(x, b_.windows, ScanKind::Optimized, false, threads, ws_.features, ws_.scan);
  st.scan_ms = ms_since(t0);
  t0 = Clock::now();
  rarity_counts(ws_.features, threads, ws_.counts, ws_.rank, kind, b_.s_floor);
  st.rank_ms = ms_since(t0);
  t0 = Clock::now();
  local_peaks(ws_.counts, x.n, b_.windows, b_.s_floor, threads, ws_.candidates, ws_.select);
  order_candidates(ws_.candidates, ws_.counts, x.n, b_.windows, ws_.select.ordered);
  merge_policies();
  st.select_ms = ms_since(t0);
  if (opt_.trace) trace_rows(e);
}

// The first (cap + 1) values of the perturbed row, descending: the base head without the replaced starts
// [c0, c1], merged with their new values. Starts with a value above the (cap + 1)-th one have count <= cap; their
// count is the position after their run of equal values. Appends (start, count) of those starts to `tail`.
void Evaluator::merge_tail(const std::vector<RankItem>& head, const std::vector<float>& values, int64_t nk,
                           int64_t cap, Row& row, std::vector<TailEntry>& tail) {
  row.changed.clear();
  for (int64_t j = 0; j <= row.c1 - row.c0; ++j) row.changed.push_back({values[j], static_cast<int32_t>(row.c0 + j)});
  std::sort(row.changed.begin(), row.changed.end(), kDesc);
  const size_t want = static_cast<size_t>(std::min(cap + 1, nk));
  row.merged.clear();
  size_t h = 0, c = 0;
  while (row.merged.size() < want) {
    while (h < head.size() && head[h].i >= row.c0 && head[h].i <= row.c1) ++h;  // replaced by `changed`
    const bool from_head = h < head.size() && (c == row.changed.size() || head[h].v >= row.changed[c].v);
    if (from_head)
      row.merged.push_back(head[h++]);
    else if (c < row.changed.size())
      row.merged.push_back(row.changed[c++]);
    else
      throw std::logic_error("incremental head too short");
  }
  size_t m = want;  // cap >= nk: every start is in the tail
  if (cap < nk) {
    const float y = row.merged[cap].v;
    m = cap;
    while (m > 0 && row.merged[m - 1].v == y) --m;
  }
  tail.clear();
  for (size_t p = 0, q = 0; p < m; p = q) {
    while (q < m && row.merged[q].v == row.merged[p].v) ++q;
    for (size_t r = p; r < q; ++r) tail.push_back({row.merged[r].i, static_cast<int32_t>(q)});
  }
}

void Evaluator::run_incremental(const TestEvent& e, int threads, StageTimes& st) {
  if (e.length > b_.max_length) throw std::invalid_argument("event longer than the incremental heads allow");
  const int64_t n = work_.size();
  const int K = static_cast<int>(b_.windows.size());
  const SeriesView x = view(work_);

  // 1) new features of the starts whose windows overlap the event; an error (fixed-point overflow) is rethrown after
  // the parallel loop, since an exception must not leave an OpenMP region
  auto t0 = Clock::now();
  std::exception_ptr error;
#pragma omp parallel for num_threads(threads) if (threads > 1) schedule(dynamic, 1)
  for (int k = 0; k < K; ++k) {
    try {
    Row& row = rows_[k];
    const int64_t w = b_.windows[k], nk = valid_starts(n, b_.windows[k]);
    row.c0 = std::max<int64_t>(0, e.start - w + 1);
    row.c1 = std::min<int64_t>(nk - 1, e.start + e.length - 1);
    const int64_t cnt = row.c1 - row.c0 + 1, bars = cnt + w - 1;
    row.new_range.resize(cnt);
    row.new_volume.resize(cnt);
    row.qmin.resize(bars);
    row.qmax.resize(bars);
    const SeriesView sub{x.low + row.c0, x.high + row.c0, x.volume + row.c0, bars};
    fixed_volumes(sub, w, row.vfix);
    scan_starts(sub, row.vfix.data(), w, cnt, row.new_range.data(), row.new_volume.data(), nullptr, nullptr,
                row.qmin.data(), row.qmax.data());
    } catch (...) {
#pragma omp critical(sf_incremental_error)
      if (!error) error = std::current_exception();
    }
  }
  if (error) std::rethrow_exception(error);
  st.scan_ms = ms_since(t0);

  // 2) exact tails of both features; combined counts where both are in their tail, kAboveCap elsewhere
  t0 = Clock::now();
#pragma omp parallel for num_threads(threads) if (threads > 1) schedule(dynamic, 1)
  for (int k = 0; k < K; ++k) {
    Row& row = rows_[k];
    const int64_t nk = valid_starts(n, b_.windows[k]), cap = count_cap(nk, b_.s_floor);
    merge_tail(b_.heads[k].range, row.new_range, nk, cap, row, row.tail_range);
    merge_tail(b_.heads[k].volume, row.new_volume, nk, cap, row, row.tail_volume);
    if (opt_.sparse_both) {
      // PC4: both tails sorted by start, intersected; the combined count is the larger of the two
      const auto by_start = [](const TailEntry& a, const TailEntry& b) { return a.start < b.start; };
      std::sort(row.tail_range.begin(), row.tail_range.end(), by_start);
      std::sort(row.tail_volume.begin(), row.tail_volume.end(), by_start);
      row.both_sparse.clear();
      for (size_t i = 0, j = 0; i < row.tail_range.size() && j < row.tail_volume.size();) {
        const auto &a = row.tail_range[i], &b = row.tail_volume[j];
        if (a.start < b.start) ++i;
        else if (b.start < a.start) ++j;
        else {
          row.both_sparse.push_back({a.start, std::max(a.count, b.count)});
          ++i, ++j;
        }
      }
      continue;
    }
    int32_t* cc = cc_.data() + static_cast<size_t>(k) * n;
    // range counts first; a start in both tails is marked negative (-c - 1) when its volume count arrives;
    // unmarked range-only starts go back to kAboveCap
    for (const auto& t : row.tail_range) cc[t.start] = t.count;
    row.both.clear();
    for (const auto& t : row.tail_volume) {
      if (cc[t.start] == kAboveCap) continue;
      cc[t.start] = -std::max(cc[t.start], t.count) - 1;
      row.both.push_back(t.start);
    }
    for (const auto& t : row.tail_range) cc[t.start] = cc[t.start] < 0 ? -cc[t.start] - 1 : kAboveCap;
  }
  st.rank_ms = ms_since(t0);

  // 3) peaks among the starts in both tails, then per-policy merge
  t0 = Clock::now();
  auto t1 = t0;
  const auto lap = [&t1](double& acc) {
    const auto now = Clock::now();
    acc += std::chrono::duration<double, std::milli>(now - t1).count();
    t1 = now;
  };
  if (opt_.sparse_both) {
#pragma omp parallel for num_threads(threads) if (threads > 1) schedule(dynamic, 1)
    for (int k = 0; k < K; ++k) {
      const int64_t w = b_.windows[k];
      sparse_peaks(rows_[k], k, valid_starts(n, b_.windows[k]), std::max<int64_t>(1, w / 2));
    }
    lap(st.peak_ms);
    order_sparse();
    lap(st.collect_ms);
    std::sort(ws_.select.ordered.begin(), ws_.select.ordered.end(), [](const Episode& a, const Episode& b) {
      return episode_before(a.S, a.k, a.start, b.S, b.k, b.start);
    });
    lap(st.order_ms);
    merge_policies();
    lap(st.merge_ms);
  } else {
#pragma omp parallel for num_threads(threads) if (threads > 1) schedule(dynamic, 1)
    for (int k = 0; k < K; ++k) {
      Row& row = rows_[k];
      const int64_t w = b_.windows[k], nk = valid_starts(n, b_.windows[k]), r = std::max<int64_t>(1, w / 2);
      const int32_t* cc = cc_.data() + static_cast<size_t>(k) * n;
      row.peaks.clear();
      row.neighbours = 0;
      if (opt_.diag) {
        for (int32_t s : row.both) {
          bool peak = true;
          for (int64_t j = std::max<int64_t>(0, s - r), end = std::min(nk - 1, s + r); peak && j <= end; ++j)
            if (j != s) ++row.neighbours, peak = !peak_blocked_by(cc[j], j, cc[s], s);
          if (peak) row.peaks.push_back(static_cast<int64_t>(k) * n + s);
        }
        continue;
      }
      for (int32_t s : row.both)
        if (is_peak(cc, nk, s, r)) row.peaks.push_back(static_cast<int64_t>(k) * n + s);
    }
    lap(st.peak_ms);
    ws_.candidates.clear();
    for (const Row& row : rows_) ws_.candidates.insert(ws_.candidates.end(), row.peaks.begin(), row.peaks.end());
    lap(st.collect_ms);
    if (opt_.s_table) {
      // PC2: order_candidates with S from the table
      auto& ordered = ws_.select.ordered;
      ordered.clear();
      for (int64_t idx : ws_.candidates) {
        const int k = static_cast<int>(idx / n);
        const int64_t s = idx - static_cast<int64_t>(k) * n;
        ordered.push_back({k, s, b_.windows[k], cc_[idx], b_.s_table[k][cc_[idx]]});
      }
      std::sort(ordered.begin(), ordered.end(), [](const Episode& a, const Episode& b) {
        return episode_before(a.S, a.k, a.start, b.S, b.k, b.start);
      });
    } else {
      order_candidates(ws_.candidates, cc_, n, b_.windows, ws_.select.ordered);
    }
    lap(st.order_ms);
    merge_policies();
    lap(st.merge_ms);
  }
  if (opt_.diag || opt_.trace) {
    for (const Row& row : rows_) {
      counts_.both += opt_.sparse_both ? row.both_sparse.size() : row.both.size();
      counts_.neighbours += row.neighbours;
      counts_.peaks += opt_.sparse_both ? row.peak_eps.size() : row.peaks.size();
    }
    counts_.candidates = static_cast<int64_t>(ws_.select.ordered.size());
  }
  if (opt_.trace) {
    trace_.rows.assign(K, {});
    for (int k = 0; k < K; ++k) {
      const Row& row = rows_[k];
      auto& r = trace_.rows[k];
      r.c0 = row.c0;
      r.c1 = row.c1;
      r.range = row.new_range;
      r.volume = row.new_volume;
      std::vector<TailEntry> both = row.both_sparse;
      if (!opt_.sparse_both) {
        both.clear();
        for (int32_t s : row.both) both.push_back({s, cc_[static_cast<size_t>(k) * n + s]});
        std::sort(both.begin(), both.end(), [](const TailEntry& a, const TailEntry& b) { return a.start < b.start; });
      }
      for (const auto& t : both) {
        r.both_start.push_back(t.start);
        r.both_count.push_back(t.count);
      }
    }
  }
  // leave cc at kAboveCap for the next trial
  if (!opt_.sparse_both)
    for (int k = 0; k < K; ++k)
      for (int32_t s : rows_[k].both) cc_[static_cast<size_t>(k) * n + s] = kAboveCap;
  lap(st.reset_ms);
  st.select_ms = ms_since(t0);
}

Detection detection(const std::vector<Episode>& found, const std::vector<Episode>& on_base, const TestEvent& e) {
  Detection d{false, false};
  for (const auto& ep : found) {
    const double centre = ep.start + ep.window / 2.0;
    if (centre < e.start || centre >= e.start + e.length) continue;
    d.overlap = true;
    const bool known = std::any_of(on_base.begin(), on_base.end(),
                                   [&](const Episode& o) { return o.k == ep.k && o.start == ep.start; });
    d.hit |= !known;
  }
  return d;
}

}  // namespace sf

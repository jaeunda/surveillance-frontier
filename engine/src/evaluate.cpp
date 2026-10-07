#include "sf/evaluate.hpp"

#include <omp.h>

#include <algorithm>
#include <chrono>
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

Evaluator::Evaluator(const BaseState& base, Method method)
    : b_(base), method_(method), work_(*base.base), out_(base.policies.size()) {
  if (method_ == Method::Full) {
    for (const auto& p : b_.policies) {
      DetectorConfig cfg;
      cfg.windows.assign(b_.windows.begin(), b_.windows.begin() + p.K);
      cfg.select = p.select;
      configs_.push_back(cfg);
    }
  } else if (method_ == Method::Incremental) {
    if (b_.heads.size() != b_.windows.size()) throw std::invalid_argument("base state has no incremental heads");
    cc_.assign(b_.windows.size() * static_cast<size_t>(work_.size()), kAboveCap);
    rows_.resize(b_.windows.size());
  }
}

const std::vector<std::vector<Episode>>& Evaluator::run(const TestEvent& e, double median_range_L, int threads,
                                                        StageTimes* times) {
  StageTimes st;
  apply_event(*b_.base, e, median_range_L, work_);
  switch (method_) {
    case Method::Full: run_full(threads, st); break;
    case Method::Shared: run_shared(RankKind::Sort, threads, st); break;
    case Method::Tail: run_shared(RankKind::Tail, threads, st); break;
    case Method::Incremental: run_incremental(e, threads, st); break;
  }
  restore(*b_.base, e, work_);
  if (times) *times = st;
  return out_;
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
  for (size_t p = 0; p < b_.policies.size(); ++p)
    merge_episodes(ws_.select.ordered, work_.size(), b_.policies[p].K, b_.policies[p].select, out_[p]);
}

void Evaluator::run_shared(RankKind kind, int threads, StageTimes& st) {
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

  // 1) new features of the starts whose windows overlap the event
  auto t0 = Clock::now();
#pragma omp parallel for num_threads(threads) if (threads > 1) schedule(dynamic, 1)
  for (int k = 0; k < K; ++k) {
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
  }
  st.scan_ms = ms_since(t0);

  // 2) exact tails of both features; combined counts where both are in their tail, kAboveCap elsewhere
  t0 = Clock::now();
#pragma omp parallel for num_threads(threads) if (threads > 1) schedule(dynamic, 1)
  for (int k = 0; k < K; ++k) {
    Row& row = rows_[k];
    const int64_t nk = valid_starts(n, b_.windows[k]), cap = count_cap(nk, b_.s_floor);
    merge_tail(b_.heads[k].range, row.new_range, nk, cap, row, row.tail_range);
    merge_tail(b_.heads[k].volume, row.new_volume, nk, cap, row, row.tail_volume);
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
#pragma omp parallel for num_threads(threads) if (threads > 1) schedule(dynamic, 1)
  for (int k = 0; k < K; ++k) {
    Row& row = rows_[k];
    const int64_t w = b_.windows[k], nk = valid_starts(n, b_.windows[k]), r = std::max<int64_t>(1, w / 2);
    const int32_t* cc = cc_.data() + static_cast<size_t>(k) * n;
    row.peaks.clear();
    for (int32_t s : row.both)
      if (is_peak(cc, nk, s, r)) row.peaks.push_back(static_cast<int64_t>(k) * n + s);
  }
  ws_.candidates.clear();
  for (const Row& row : rows_) ws_.candidates.insert(ws_.candidates.end(), row.peaks.begin(), row.peaks.end());
  order_candidates(ws_.candidates, cc_, n, b_.windows, ws_.select.ordered);
  merge_policies();
  // leave cc at kAboveCap for the next trial
  for (int k = 0; k < K; ++k)
    for (int32_t s : rows_[k].both) cc_[static_cast<size_t>(k) * n + s] = kAboveCap;
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

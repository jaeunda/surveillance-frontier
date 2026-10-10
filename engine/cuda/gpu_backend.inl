// Host side of the GPU back end (G-inc and G-hyb, Phase 2 protocol "Implementations compared"). Included by
// gpu_backend_cuda.cu (CUDA) and gpu_backend_emul.cpp (host emulation) with SF_GPU_NS set; see gpu_kernels.hpp for
// the device pipeline.
//
// Resident state (uploaded at construction): the series, per-row constants, the base heads of both features, the S
// table, per-policy caps, the policies and their base episodes. Per task (set_cells): event factor tables. Per
// microbatch: the scratch of SPEC §6, sized for the base state's maximum event length. A submission is `submission`
// microbatches; the host waits for each microbatch's results (bits, or candidates on G-hyb) before the next.
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_kernels.hpp"
#include "runtime.hpp"
#include "sf/rank.hpp"
#include "sf/scan.hpp"

namespace sf {
namespace SF_GPU_NS {

using gpu::Args;
using gpu::CandD;
using gpu::CellD;
using gpu::EpisodeD;
using gpu::PolicyD;
using gpu::RankItemD;
using gpu::RowD;
using gpu::TailD;

namespace {

template <typename T>
struct DevBuf {
  T* p = nullptr;
  size_t n = 0;
  DevBuf() = default;
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  ~DevBuf() { reset(); }
  void reset() {
    if (p) dev_free(p);
    p = nullptr;
    n = 0;
  }
  size_t bytes() const { return n * sizeof(T); }
  void alloc(size_t count, int64_t& total) {
    reset();
    p = static_cast<T*>(dev_malloc(count * sizeof(T)));
    n = count;
    total += static_cast<int64_t>(bytes());
  }
  void upload(const std::vector<T>& h, int64_t& total, int64_t& moved) {
    alloc(h.size(), total);
    h2d(p, h.data(), bytes());
    moved += static_cast<int64_t>(bytes());
  }
};

int32_t fault_code(const std::string& f) {
  if (f.empty() || f == "none") return gpu::kFaultNone;
  if (f == "device-log10") return gpu::kFaultDeviceLog10;
  if (f == "reversed-tie") return gpu::kFaultReversedTie;
  if (f == "cap-minus-one") return gpu::kFaultCapMinusOne;
  if (f == "direct-float") return gpu::kFaultDirectFloat;
  if (f == "half-even") return gpu::kFaultHalfEven;
  if (f == "fast-math") return gpu::kFaultFastMath;
  throw std::invalid_argument("unknown fault " + f);
}

class GpuBackend : public BatchEvaluator {
 public:
  GpuBackend(const BaseState& base, const BackendConfig& cfg) : b_(base), cfg_(cfg) {
    if (cfg_.path != "inc" && cfg_.path != "hyb") throw std::invalid_argument("GPU path must be inc or hyb");
    if (b_.heads.size() != b_.windows.size()) throw std::invalid_argument("GPU back end needs incremental heads");
    launch_threads() = std::max(1, cfg_.threads);
    init(cfg_.device);
    a_ = Args{};
    a_.fault = fault_code(cfg_.fault);
    a_.K = static_cast<int32_t>(b_.windows.size());
    a_.P = static_cast<int32_t>(b_.policies.size());
    a_.n = b_.base->size();
    // free device memory is read once, after context creation and before any allocation of ours (SPEC §6), so the
    // shared state is subtracted exactly once when the microbatch is sized
    mem_info(free_at_context_, total_bytes_);
    upload_resident();
    size_microbatch();
    alloc_scratch();
    size_t total = 0;
    mem_info(free_after_alloc_, total);
  }

  std::string name() const override { return cfg_.path == "inc" ? "gpu-inc" : "gpu-hyb"; }
  std::string device_info() const override {
    return device_name() + " path=" + cfg_.path + " microbatch=" + std::to_string(b_micro_) +
           " working_set_bytes_per_trial=" + std::to_string(ws_bytes_) +
           " shared_bytes=" + std::to_string(shared_bytes_) + " scratch_bytes=" + std::to_string(scratch_bytes_) +
           " b_max=" + std::to_string(b_max_) + " memory_fraction=" + std::to_string(cfg_.memory_fraction) +
           " free_at_context_bytes=" + std::to_string(free_at_context_) +
           " free_after_alloc_bytes=" + std::to_string(free_after_alloc_) + " total_bytes=" + std::to_string(total_bytes_);
  }

  void set_cells(const std::vector<CellPlan>& cells) override {
    std::vector<CellD> hc;
    std::vector<float> factors;
    check_cells_.assign(cells.size(), 0);
    const double limit_w = std::ldexp(1.0, 62 - kVolumeFracBits);
    for (size_t c = 0; c < cells.size(); ++c) {
      const CellPlan& p = cells[c];
      if (p.length > b_.max_length) throw std::invalid_argument("cell length exceeds the base state's max length");
      if (p.table.length != p.length || static_cast<int32_t>(p.table.price.size()) != p.length)
        throw std::invalid_argument("cell without event table");
      hc.push_back(CellD{p.length, static_cast<int32_t>(p.kind), static_cast<int64_t>(factors.size()),
                         p.table.volume, 0});
      factors.insert(factors.end(), p.table.price.begin(), p.table.price.end());
      // fixed-point overflow (SPEC §4): if no event-modified bar can reach the smallest limit, no trial of this
      // cell can fail the CPU's check; otherwise every trial is checked on the host exactly as the CPU does
      // (float multiplication is monotone, so the largest scaled bar is the largest bar scaled)
      const double limit = limit_w / b_.windows.back();
      const float m = p.kind != EventKind::Price ? max_volume_ * p.table.volume : max_volume_;
      check_cells_[c] = !(min_volume_ >= 0.0f && max_volume_ < limit && m >= 0.0f && m < limit);
    }
    cells_ = cells;
    cells_d_.upload(hc, shared_bytes_alloc_, bytes_h2d_);
    factors_d_.upload(factors, shared_bytes_alloc_, bytes_h2d_);
    a_.cells = cells_d_.p;
    a_.factors = factors_d_.p;
  }

  void evaluate(const TrialBatch& batch, BatchOutput& out, const std::function<bool()>& stop) override {
    const int64_t n = batch.count();
    if (n <= 0) return;
    if (out.words != policy_words(b_.policies.size())) throw std::invalid_argument("GPU back end: output words");
    if (cells_.empty()) throw std::invalid_argument("GPU back end: no cells");
    if (batch.space) {
      if (!space_d_.p) space_d_.alloc(1, shared_bytes_alloc_);
      h2d(space_d_.p, batch.space, sizeof(TrialSpace));
      bytes_h2d_ += sizeof(TrialSpace);
      a_.space = space_d_.p;
    }
    if (out.traces) out.traces->assign(n, {});
    const auto t0 = std::chrono::steady_clock::now();
    const int64_t sub = b_micro_ * std::max<int64_t>(1, cfg_.submission);
    WorkerRecord rec;
    for (int64_t pos = 0; pos < n;) {
      if (stop()) break;
      const int64_t end = std::min(n, pos + sub);
      for (; pos < end;) {
        const int64_t m = std::min(b_micro_, end - pos);
        run_micro(batch, pos, m, out);
        pos += m;
        rec.trials += m;
      }
      rec.last_done_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
    rec.busy_s = rec.last_done_s;
    stats_.trials += rec.trials;
    stats_.workers = {rec};
  }

  void sync() override { dev_sync(); }

  void resolve_starts(const TrialSpace& sp, int64_t g0, int64_t n, int32_t* cell, int64_t* start) override {
    DevBuf<TrialSpace> spd;
    DevBuf<int32_t> c;
    DevBuf<int64_t> s;
    int64_t unused = 0;
    spd.alloc(1, unused);
    h2d(spd.p, &sp, sizeof sp);
    const int64_t chunk = 1 << 20;
    c.alloc(chunk, unused);
    s.alloc(chunk, unused);
    dev_memset(err_d_.p, 0, sizeof(int32_t));
    for (int64_t pos = 0; pos < n; pos += chunk) {
      const int64_t m = std::min(chunk, n - pos);
      launch<gpu::StreamArgs, gpu::k_stream>(m, gpu::StreamArgs{spd.p, g0 + pos, c.p, s.p, nullptr, nullptr, err_d_.p});
      d2h(cell + pos, c.p, m * sizeof(int32_t));
      d2h(start + pos, s.p, m * sizeof(int64_t));
      check_error();
    }
  }

  void philox_blocks(const uint32_t* ck, int64_t n, uint32_t* out) override {
    DevBuf<uint32_t> in, o;
    int64_t unused = 0;
    in.alloc(6 * n, unused);
    o.alloc(4 * n, unused);
    h2d(in.p, ck, 6 * n * sizeof(uint32_t));
    launch<gpu::StreamArgs, gpu::k_philox>(n, gpu::StreamArgs{nullptr, 0, nullptr, nullptr, in.p, o.p, nullptr});
    d2h(out, o.p, 4 * n * sizeof(uint32_t));
  }

  BackendStats take_stats() override {
    stats_.bytes_h2d = bytes_h2d_;
    stats_.bytes_d2h = bytes_d2h_;
    stats_.peak_device_bytes = shared_bytes_alloc_ + scratch_bytes_;
    BackendStats s = std::move(stats_);
    stats_ = BackendStats{};
    bytes_h2d_ = bytes_d2h_ = 0;
    return s;
  }

 private:
  void upload_resident() {
    const BaseState& b = b_;
    const int K = a_.K;
    std::vector<RowD> rows(K);
    std::vector<RankItemD> hr, hv;
    std::vector<double> st;
    int64_t F = 0, T = 0;
    for (int k = 0; k < K; ++k) {
      RowD& r = rows[k];
      r.w = b.windows[k];
      r.nk = valid_starts(a_.n, r.w);
      r.cap = b.caps[k];
      r.r = std::max<int64_t>(1, r.w / 2);
      if (b.heads[k].range.size() != b.heads[k].volume.size()) throw std::logic_error("head sizes differ");
      r.head_off = static_cast<int64_t>(hr.size());
      r.head_len = static_cast<int32_t>(b.heads[k].range.size());
      for (const auto& x : b.heads[k].range) hr.push_back(RankItemD{x.v, x.i});
      for (const auto& x : b.heads[k].volume) hv.push_back(RankItemD{x.v, x.i});
      r.feat_off = F;
      r.feat_cap = b.max_length + r.w - 1;
      F += r.feat_cap;
      r.tail_off = T;
      T += std::max<int64_t>(r.cap, 1);
      r.s_off = static_cast<int64_t>(st.size());
      st.insert(st.end(), b.s_table[k].begin(), b.s_table[k].end());
    }
    a_.F = F;
    a_.T = T;
    std::vector<int32_t> pcaps;
    std::vector<PolicyD> pols;
    std::vector<EpisodeD> eps;
    a_.top_k_max = 1;
    for (size_t p = 0; p < b.policies.size(); ++p) {
      const Policy& pol = b.policies[p];
      pcaps.insert(pcaps.end(), b.policy_caps[p].begin(), b.policy_caps[p].end());
      pols.push_back(PolicyD{pol.K, pol.select.top_k, pol.select.episode_gap, static_cast<int64_t>(eps.size()),
                             static_cast<int32_t>(b.episodes[p].size()), 0});
      for (const auto& e : b.episodes[p]) eps.push_back(EpisodeD{e.start, e.k, 0});
      a_.top_k_max = std::max(a_.top_k_max, pol.select.top_k);
    }
    const Series& x = *b.base;
    min_volume_ = *std::min_element(x.volume.begin(), x.volume.end());
    max_volume_ = *std::max_element(x.volume.begin(), x.volume.end());
    low_d_.upload(x.low, shared_bytes_alloc_, bytes_h2d_);
    high_d_.upload(x.high, shared_bytes_alloc_, bytes_h2d_);
    vol_d_.upload(x.volume, shared_bytes_alloc_, bytes_h2d_);
    rows_d_.upload(rows, shared_bytes_alloc_, bytes_h2d_);
    hr_d_.upload(hr, shared_bytes_alloc_, bytes_h2d_);
    hv_d_.upload(hv, shared_bytes_alloc_, bytes_h2d_);
    st_d_.upload(st, shared_bytes_alloc_, bytes_h2d_);
    pcaps_d_.upload(pcaps, shared_bytes_alloc_, bytes_h2d_);
    pols_d_.upload(pols, shared_bytes_alloc_, bytes_h2d_);
    eps_d_.upload(eps.empty() ? std::vector<EpisodeD>(1) : eps, shared_bytes_alloc_, bytes_h2d_);
    a_.low = low_d_.p;
    a_.high = high_d_.p;
    a_.volume = vol_d_.p;
    a_.rows = rows_d_.p;
    a_.head_range = hr_d_.p;
    a_.head_volume = hv_d_.p;
    a_.s_table = st_d_.p;
    a_.policy_caps = pcaps_d_.p;
    a_.policies = pols_d_.p;
    a_.base_eps = eps_d_.p;
    rows_h_ = rows;
    shared_bytes_ = shared_bytes_alloc_ + static_cast<int64_t>(sizeof(TrialSpace));
  }

  // bytes of scratch per trial (SPEC §6 working set, as laid out here)
  int64_t per_trial_bytes() const {
    const int64_t K = a_.K, P = a_.P, F = a_.F, T = a_.T;
    return 4 + 8 + K * (8 + 4) + F * (4 + 4) + 2 * F * 8 + 2 * T * 8 + 2 * K * 4 + T * 8 + K * 4 + T * 1 +
           T * static_cast<int64_t>(sizeof(CandD)) + 4 + P * a_.top_k_max * 4 + P * 4 + 2 * P;
  }

  void size_microbatch() {
    ws_bytes_ = per_trial_bytes();
    const double room =
        cfg_.memory_fraction * static_cast<double>(free_at_context_) - static_cast<double>(shared_bytes_);
    b_max_ = std::max<int64_t>(0, static_cast<int64_t>(room / static_cast<double>(ws_bytes_)));
    b_micro_ = cfg_.microbatch > 0 ? cfg_.microbatch : b_max_;
    if (b_micro_ < 1) throw std::runtime_error("GPU back end: no microbatch fits (b_max = 0)");
  }

  void alloc_scratch() {
    const int64_t b = b_micro_, K = a_.K, P = a_.P, F = a_.F, T = a_.T;
    int64_t& s = scratch_bytes_;
    s = 0;
    try {
      cell_d_.alloc(b, s);
      start_d_.alloc(b, s);
      c0_d_.alloc(b * K, s);
      cnt_d_.alloc(b * K, s);
      fr_d_.alloc(b * F, s);
      fv_d_.alloc(b * F, s);
      chg_d_.alloc(b * 2 * F, s);
      tail_d_.alloc(b * 2 * T, s);
      tail_len_d_.alloc(b * 2 * K, s);
      both_d_.alloc(b * T, s);
      both_len_d_.alloc(b * K, s);
      peak_d_.alloc(b * T, s);
      cand_d_.alloc(b * T, s);
      cand_len_d_.alloc(b, s);
      kept_d_.alloc(b * P * a_.top_k_max, s);
      kept_len_d_.alloc(b * P, s);
      hit_d_.alloc(b * P, s);
      ovl_d_.alloc(b * P, s);
      err_d_.alloc(1, s);
    } catch (const std::exception& e) {
      throw std::runtime_error(std::string("GPU allocation failed at microbatch ") + std::to_string(b) + ": " +
                               e.what());
    }
    a_.cell = cell_d_.p;
    a_.start = start_d_.p;
    a_.c0 = c0_d_.p;
    a_.cnt = cnt_d_.p;
    a_.f_range = fr_d_.p;
    a_.f_volume = fv_d_.p;
    a_.chg = chg_d_.p;
    a_.tail = tail_d_.p;
    a_.tail_len = tail_len_d_.p;
    a_.both = both_d_.p;
    a_.both_len = both_len_d_.p;
    a_.peak = peak_d_.p;
    a_.cand = cand_d_.p;
    a_.cand_len = cand_len_d_.p;
    a_.kept = kept_d_.p;
    a_.kept_len = kept_len_d_.p;
    a_.hit = hit_d_.p;
    a_.overlap = ovl_d_.p;
    a_.error = err_d_.p;
    dev_memset(err_d_.p, 0, sizeof(int32_t));
  }

  // the CPU's fixed_volumes check on every row window of one trial (only for cells that could overflow)
  void check_overflow(int32_t cell, int64_t start) const {
    const CellPlan& p = cells_[cell];
    const Series& x = *b_.base;
    for (int k = 0; k < a_.K; ++k) {
      const int64_t w = b_.windows[k], nk = valid_starts(a_.n, b_.windows[k]);
      const int64_t c0 = std::max<int64_t>(0, start - w + 1), c1 = std::min<int64_t>(nk - 1, start + p.length - 1);
      const double limit = std::ldexp(1.0, 62 - kVolumeFracBits) / static_cast<double>(w);
      for (int64_t q = c0; q <= c1 + w - 1; ++q) {
        float v = x.volume[q];
        if (q >= start && q < start + p.length && p.kind != EventKind::Price) v = v * p.table.volume;
        if (!(v >= 0.0f && v < limit)) throw std::overflow_error("volume out of fixed-point range");
      }
    }
  }

  void run_micro(const TrialBatch& batch, int64_t pos, int64_t m, BatchOutput& out) {
    const int64_t K = a_.K, P = a_.P;
    Args a = a_;
    a.g0 = batch.g0 + pos;
    const bool timed = cfg_.eval.diag;
    PhaseTimer t_begin, t_feat, t_tail, t_both, t_peak, t_compact, t_order, t_merge;
    if (timed) t_begin.mark();
    std::vector<int32_t> cell(m);
    std::vector<int64_t> start(m);
    if (batch.cells) {
      std::copy(batch.cells + pos, batch.cells + pos + m, cell.begin());
      std::copy(batch.starts + pos, batch.starts + pos + m, start.begin());
      h2d(cell_d_.p, cell.data(), m * sizeof(int32_t));
      h2d(start_d_.p, start.data(), m * sizeof(int64_t));
      bytes_h2d_ += m * 12;
    } else {
      launch<Args, gpu::k_resolve>(m, a);
      d2h(cell.data(), cell_d_.p, m * sizeof(int32_t));
      d2h(start.data(), start_d_.p, m * sizeof(int64_t));
      bytes_d2h_ += m * 12;
      check_error();
    }
    for (int64_t i = 0; i < m; ++i) {
      if (cell[i] < 0 || cell[i] >= static_cast<int32_t>(cells_.size()))
        throw std::invalid_argument("GPU back end: trial cell has no plan");
      if (check_cells_[cell[i]]) check_overflow(cell[i], start[i]);
    }
    launch<Args, gpu::k_rows>(m * K, a);
    launch<Args, gpu::k_features>(m * a.F, a);
    if (timed) t_feat.mark();
    launch<Args, gpu::k_tails>(m * 2 * K, a);
    if (timed) t_tail.mark();
    launch<Args, gpu::k_both>(m * K, a);
    if (timed) t_both.mark();
    launch<Args, gpu::k_peaks>(m * a.T, a);
    if (timed) t_peak.mark();
    launch<Args, gpu::k_compact>(m, a);
    if (timed) t_compact.mark();

    std::vector<uint8_t> hit(m * P), ovl(m * P);
    std::vector<std::vector<Episode>> host_ordered;  // G-hyb, or traces
    std::vector<std::vector<std::vector<Episode>>> host_kept;
    if (cfg_.path == "inc") {
      launch<Args, gpu::k_order>(m, a);
      if (timed) t_order.mark();
      launch<Args, gpu::k_merge>(m * P, a);
      if (timed) t_merge.mark();
      d2h(hit.data(), hit_d_.p, hit.size());
      d2h(ovl.data(), ovl_d_.p, ovl.size());
      bytes_d2h_ += 2 * m * P;
      check_error();
    } else {
      check_error();
      // G-hyb: candidates to the host; ordering, merge and detection with the CPU code
      std::vector<int32_t> clen(m);
      d2h(clen.data(), cand_len_d_.p, m * sizeof(int32_t));
      bytes_d2h_ += m * 4;
      std::vector<std::vector<CandD>> cand(m);
      for (int64_t i = 0; i < m; ++i) {
        cand[i].resize(clen[i]);
        d2h(cand[i].data(), cand_d_.p + i * a.T, clen[i] * sizeof(CandD));
        bytes_d2h_ += clen[i] * static_cast<int64_t>(sizeof(CandD));
      }
      if (timed) t_order.mark();
      host_ordered.resize(m);
      host_kept.resize(m);
#pragma omp parallel for num_threads(std::max(1, cfg_.threads)) schedule(dynamic, 1)
      for (int64_t i = 0; i < m; ++i) {
        auto& ord = host_ordered[i];
        for (const CandD& c : cand[i]) ord.push_back({c.k, c.start, b_.windows[c.k], c.count, c.S});
        std::sort(ord.begin(), ord.end(), [](const Episode& x, const Episode& y) {
          return episode_before(x.S, x.k, x.start, y.S, y.k, y.start);
        });
        const CellPlan& cp = cells_[cell[i]];
        const TestEvent e{start[i], cp.length, cp.strength, cp.kind};
        host_kept[i].resize(P);
        for (int64_t p = 0; p < P; ++p) {
          merge_episodes(ord, a_.n, b_.policies[p].K, b_.policies[p].select, host_kept[i][p]);
          const Detection d = detection(host_kept[i][p], b_.episodes[p], e);
          hit[i * P + p] = d.hit;
          ovl[i * P + p] = d.overlap;
        }
      }
      if (timed) t_merge.mark();
    }
    for (int64_t i = 0; i < m; ++i) {
      const int64_t o = pos + i;
      for (int w = 0; w < out.words; ++w) out.hit[o * out.words + w] = out.overlap[o * out.words + w] = 0;
      for (int64_t p = 0; p < P; ++p) set_bits(out, o, static_cast<size_t>(p), hit[i * P + p], ovl[i * P + p]);
      out.done[o] = 1;
    }
    if (out.traces) fill_traces(pos, m, cell, start, host_ordered, host_kept, out);
    if (timed) {
      t_merge.wait();
      StageTimes& s = stats_.sum;
      s.scan_ms += t_feat.ms_since(t_begin);
      s.rank_ms += t_both.ms_since(t_feat);
      s.peak_ms += t_peak.ms_since(t_both);
      s.collect_ms += t_compact.ms_since(t_peak);
      s.order_ms += t_order.ms_since(t_compact);
      s.merge_ms += t_merge.ms_since(t_order);
      s.select_ms += t_merge.ms_since(t_both);
      (void)t_tail;
    }
  }

  void check_error() {
    int32_t e = 0;
    d2h(&e, err_d_.p, sizeof e);
    if (e & gpu::kErrDraw) throw std::runtime_error("trial stream: no accepted draw within 256 attempts");
    if (e & gpu::kErrHeadShort) throw std::logic_error("incremental head too short");
  }

  void fill_traces(int64_t pos, int64_t m, const std::vector<int32_t>& cell, const std::vector<int64_t>& start,
                   const std::vector<std::vector<Episode>>& host_ordered,
                   const std::vector<std::vector<std::vector<Episode>>>& host_kept, BatchOutput& out) {
    const int64_t K = a_.K, P = a_.P, F = a_.F, T = a_.T;
    std::vector<int64_t> c0(m * K);
    std::vector<int32_t> cnt(m * K), blen(m * K), clen(m), klen(m * P), kept(m * P * a_.top_k_max);
    std::vector<float> fr(m * F), fv(m * F);
    std::vector<TailD> both(m * T);
    std::vector<CandD> cand(m * T);
    d2h(c0.data(), c0_d_.p, c0.size() * 8);
    d2h(cnt.data(), cnt_d_.p, cnt.size() * 4);
    d2h(fr.data(), fr_d_.p, fr.size() * 4);
    d2h(fv.data(), fv_d_.p, fv.size() * 4);
    d2h(blen.data(), both_len_d_.p, blen.size() * 4);
    d2h(both.data(), both_d_.p, both.size() * sizeof(TailD));
    d2h(clen.data(), cand_len_d_.p, clen.size() * 4);
    d2h(cand.data(), cand_d_.p, cand.size() * sizeof(CandD));
    d2h(klen.data(), kept_len_d_.p, klen.size() * 4);
    d2h(kept.data(), kept_d_.p, kept.size() * 4);
    for (int64_t i = 0; i < m; ++i) {
      TrialTrace& tr = (*out.traces)[pos + i];
      tr.has_stages = true;
      tr.rows.assign(K, {});
      for (int64_t k = 0; k < K; ++k) {
        const RowD& r = rows_h_[k];
        auto& row = tr.rows[k];
        row.c0 = c0[i * K + k];
        row.c1 = row.c0 + cnt[i * K + k] - 1;
        const float* a = fr.data() + i * F + r.feat_off;
        const float* v = fv.data() + i * F + r.feat_off;
        row.range.assign(a, a + cnt[i * K + k]);
        row.volume.assign(v, v + cnt[i * K + k]);
        for (int32_t x = 0; x < blen[i * K + k]; ++x) {
          row.both_start.push_back(both[i * T + r.tail_off + x].start);
          row.both_count.push_back(both[i * T + r.tail_off + x].count);
        }
      }
      if (cfg_.path == "inc") {
        for (int32_t x = 0; x < clen[i]; ++x) {
          const CandD& c = cand[i * T + x];
          tr.ordered.push_back({c.k, c.start, b_.windows[c.k], c.count, c.S});
        }
        tr.episodes.assign(P, {});
        for (int64_t p = 0; p < P; ++p)
          for (int32_t y = 0; y < klen[i * P + p]; ++y)
            tr.episodes[p].push_back(tr.ordered[kept[(i * P + p) * a_.top_k_max + y]]);
      } else {
        tr.ordered = host_ordered[i];
        tr.episodes = host_kept[i];
      }
    }
    (void)cell;
    (void)start;
  }

  const BaseState& b_;
  BackendConfig cfg_;
  Args a_{};
  std::vector<RowD> rows_h_;
  std::vector<CellPlan> cells_;
  std::vector<uint8_t> check_cells_;
  float min_volume_ = 0, max_volume_ = 0;
  int64_t ws_bytes_ = 0, shared_bytes_ = 0, shared_bytes_alloc_ = 0, scratch_bytes_ = 0;
  int64_t b_max_ = 0, b_micro_ = 0, bytes_h2d_ = 0, bytes_d2h_ = 0;
  size_t free_at_context_ = 0, free_after_alloc_ = 0, total_bytes_ = 0;
  BackendStats stats_;
  DevBuf<float> low_d_, high_d_, vol_d_, fr_d_, fv_d_, factors_d_;
  DevBuf<RowD> rows_d_;
  DevBuf<RankItemD> hr_d_, hv_d_, chg_d_;
  DevBuf<double> st_d_;
  DevBuf<int32_t> pcaps_d_, cell_d_, cnt_d_, tail_len_d_, both_len_d_, cand_len_d_, kept_d_, kept_len_d_, err_d_;
  DevBuf<PolicyD> pols_d_;
  DevBuf<EpisodeD> eps_d_;
  DevBuf<CellD> cells_d_;
  DevBuf<TrialSpace> space_d_;
  DevBuf<int64_t> start_d_, c0_d_;
  DevBuf<TailD> tail_d_, both_d_;
  DevBuf<uint8_t> peak_d_, hit_d_, ovl_d_;
  DevBuf<CandD> cand_d_;
};

}  // namespace

std::unique_ptr<BatchEvaluator> make_backend(const BaseState& base, const BackendConfig& cfg) {
  return std::make_unique<GpuBackend>(base, cfg);
}

void device_init(int device) { init(device); }

}  // namespace SF_GPU_NS
}  // namespace sf

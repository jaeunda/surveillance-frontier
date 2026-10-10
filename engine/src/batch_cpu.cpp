// CPU back end: one incremental Evaluator per worker thread, trials distributed dynamically over the workers.
#include <omp.h>
#include <sched.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <mutex>
#include <numeric>
#include <stdexcept>

#include "sf/batch.hpp"
#include "sf/rank.hpp"

namespace sf {

std::string BackendConfig::describe() const {
  std::string s = "threads=" + std::to_string(threads) + ";parity=";
  std::string pc;
  if (event_table) pc += "PC1,";
  if (eval.s_table) pc += "PC2,";
  if (eval.cap_table) pc += "PC3,";
  if (eval.sparse_both) pc += "PC4,";
  if (group_by_cell) pc += "PC6,";
  if (!pc.empty()) pc.pop_back();
  s += pc.empty() ? "none" : pc;
  s += ";path=" + path + ";microbatch=" + std::to_string(microbatch) + ";submission=" + std::to_string(submission);
  if (eval.diag) s += ";diag";
  if (!fault.empty()) s += ";fault=" + fault;
  return s;
}

void BatchEvaluator::resolve_starts(const TrialSpace& sp, int64_t g0, int64_t n, int32_t* cell, int64_t* start) {
  for (int64_t i = 0; i < n; ++i) {
    int64_t j;
    if (!resolve_trial(sp, g0 + i, cell[i], j, start[i])) throw std::runtime_error("trial stream: draw failed");
  }
}

void BatchEvaluator::philox_blocks(const uint32_t* ck, int64_t n, uint32_t* out) {
  for (int64_t i = 0; i < n; ++i) {
    const Philox4x32 r = philox4x32_10(Philox4x32{{ck[6 * i], ck[6 * i + 1], ck[6 * i + 2], ck[6 * i + 3]}},
                                       ck[6 * i + 4], ck[6 * i + 5]);
    for (int w = 0; w < 4; ++w) out[4 * i + w] = r.v[w];
  }
}

CellPlan make_cell_plan(const Series& base, int32_t length, double strength, EventKind kind) {
  CellPlan c;
  c.length = length;
  c.strength = strength;
  c.kind = kind;
  c.median_range = median_range(view(base), length);
  c.table = make_event_table(length, strength, c.median_range);
  return c;
}

std::string affinity_text() {
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof set, &set) != 0) return "?";
  std::string out;
  for (int c = 0; c < CPU_SETSIZE;) {
    if (!CPU_ISSET(c, &set)) {
      ++c;
      continue;
    }
    int e = c;
    while (e + 1 < CPU_SETSIZE && CPU_ISSET(e + 1, &set)) ++e;
    out += (out.empty() ? "" : ",") + std::to_string(c) + (e > c ? "-" + std::to_string(e) : "");
    c = e + 1;
  }
  return out;
}

namespace {
using Clock = std::chrono::steady_clock;

void add_times(StageTimes& a, const StageTimes& b) {
  a.scan_ms += b.scan_ms;
  a.rank_ms += b.rank_ms;
  a.select_ms += b.select_ms;
  a.peak_ms += b.peak_ms;
  a.collect_ms += b.collect_ms;
  a.order_ms += b.order_ms;
  a.merge_ms += b.merge_ms;
  a.reset_ms += b.reset_ms;
}

class CpuBackend : public BatchEvaluator {
 public:
  CpuBackend(const BaseState& base, const BackendConfig& cfg) : b_(base), cfg_(cfg) {
    if (cfg_.threads < 1) throw std::invalid_argument("cpu back end: threads < 1");
    workers_.resize(cfg_.threads);
    // each worker allocates and first-touches its scratch on the thread that will use it
#pragma omp parallel num_threads(cfg_.threads)
    workers_[omp_get_thread_num()] = std::make_unique<Evaluator>(b_, Method::Incremental, cfg_.eval);
  }

  std::string name() const override { return "cpu"; }
  std::string device_info() const override { return "cpu threads=" + std::to_string(cfg_.threads); }

  void set_cells(const std::vector<CellPlan>& cells) override { cells_ = cells; }

  void evaluate(const TrialBatch& batch, BatchOutput& out, const std::function<bool()>& stop) override {
    const int64_t n = batch.count();
    if (n <= 0) return;
    const size_t P = b_.policies.size();
    if (out.words != policy_words(P)) throw std::invalid_argument("cpu back end: output word count");
    // resolve every trial first (cheap), so the order of work can be grouped by cell (PC6)
    cell_.resize(n);
    start_.resize(n);
    std::atomic<bool> bad_draw{false};
#pragma omp parallel for num_threads(cfg_.threads) schedule(static)
    for (int64_t i = 0; i < n; ++i) {
      if (batch.cells) {
        cell_[i] = batch.cells[i];
        start_[i] = batch.starts[i];
      } else {
        int64_t j;
        if (!resolve_trial(*batch.space, batch.g0 + i, cell_[i], j, start_[i])) bad_draw = true;
      }
    }
    if (bad_draw) throw std::runtime_error("trial stream: no accepted draw within 256 attempts");
    for (int64_t i = 0; i < n; ++i)
      if (cell_[i] < 0 || cell_[i] >= static_cast<int32_t>(cells_.size()))
        throw std::invalid_argument("cpu back end: trial cell has no plan");
    order_.resize(n);
    std::iota(order_.begin(), order_.end(), int64_t{0});
    if (cfg_.group_by_cell)
      std::stable_sort(order_.begin(), order_.end(), [&](int64_t a, int64_t b) { return cell_[a] < cell_[b]; });

    if (out.traces) {  // gate: serial, with a tracing evaluator
      out.traces->assign(n, {});
      EvalOptions o = cfg_.eval;
      o.trace = true;
      Evaluator ev(b_, Method::Incremental, o);
      for (int64_t i = 0; i < n; ++i) {
        run_one(ev, i, out, nullptr);
        (*out.traces)[i] = ev.trace();
      }
      stats_.trials += n;
      return;
    }

    const auto t0 = Clock::now();
    std::atomic<bool> stopped{false};
    std::exception_ptr error;  // the first error, rethrown unchanged after the parallel region
    std::mutex error_mu;
    std::vector<WorkerRecord> rec(cfg_.threads);
    std::vector<StageTimes> times(cfg_.threads);
    std::vector<TrialCounts> counts(cfg_.threads);
#pragma omp parallel num_threads(cfg_.threads)
    {
      const int t = omp_get_thread_num();
      WorkerRecord& r = rec[t];
      r.cpu_start = sched_getcpu();
      r.affinity_start = affinity_text();
#pragma omp for schedule(dynamic, 1)
      for (int64_t o = 0; o < n; ++o) {
        if (stopped.load(std::memory_order_relaxed)) continue;
        if (stop()) {
          stopped = true;
          continue;
        }
        const auto a = Clock::now();
        try {
          StageTimes st;
          run_one(*workers_[t], order_[o], out, &st);
          add_times(times[t], st);
          if (cfg_.eval.diag) {
            const auto& c = workers_[t]->counts();
            counts[t].both += c.both;
            counts[t].neighbours += c.neighbours;
            counts[t].peaks += c.peaks;
            counts[t].candidates += c.candidates;
          }
        } catch (...) {
          std::lock_guard<std::mutex> lock(error_mu);
          if (!error) error = std::current_exception();
          stopped = true;
          continue;
        }
        const auto z = Clock::now();
        ++r.trials;
        r.busy_s += std::chrono::duration<double>(z - a).count();
        r.last_done_s = std::chrono::duration<double>(z - t0).count();
      }
      r.cpu_end = sched_getcpu();
      r.affinity_end = affinity_text();
    }
    if (error) std::rethrow_exception(error);
    for (int t = 0; t < cfg_.threads; ++t) {
      stats_.trials += rec[t].trials;
      add_times(stats_.sum, times[t]);
      stats_.counts.both += counts[t].both;
      stats_.counts.neighbours += counts[t].neighbours;
      stats_.counts.peaks += counts[t].peaks;
      stats_.counts.candidates += counts[t].candidates;
    }
    stats_.workers = rec;
  }

  BackendStats take_stats() override {
    BackendStats s = std::move(stats_);
    stats_ = BackendStats{};
    return s;
  }

 private:
  void run_one(Evaluator& ev, int64_t i, BatchOutput& out, StageTimes* st) {
    const CellPlan& c = cells_[cell_[i]];
    const TestEvent e{start_[i], c.length, c.strength, c.kind};
    const auto& got = cfg_.event_table ? ev.run(e, c.table, 1, st) : ev.run(e, c.median_range, 1, st);
    for (size_t w = 0; w < static_cast<size_t>(out.words); ++w)
      out.hit[i * out.words + w] = out.overlap[i * out.words + w] = 0;
    for (size_t p = 0; p < got.size(); ++p) {
      const Detection d = detection(got[p], b_.episodes[p], e);
      set_bits(out, i, p, d.hit, d.overlap);
    }
    out.done[i] = 1;
  }

  const BaseState& b_;
  BackendConfig cfg_;
  std::vector<std::unique_ptr<Evaluator>> workers_;
  std::vector<CellPlan> cells_;
  std::vector<int32_t> cell_;
  std::vector<int64_t> start_, order_;
  BackendStats stats_;
};
}  // namespace

std::unique_ptr<BatchEvaluator> make_cpu_backend(const BaseState& base, const BackendConfig& cfg) {
  return std::make_unique<CpuBackend>(base, cfg);
}

}  // namespace sf

// Batch evaluation of stress-test trials for Phase 2 (SPEC §1): one interface, several back ends.
//
// A task is a set of cells (event length, strength, kind) and a trial space that maps a global trial index g to
// (cell, j, start). Starts come from the counter-based stream (sf/stream.hpp) or, for enumeration, are j itself, so
// a trial's result never depends on the back end, thread count, batch, or schedule. Back ends evaluate any subset of
// a range of g in any order and write per-trial hit and overlap bits; the runner aggregates.
//
//   cpu    incremental Evaluator per worker thread (CPU-best = this plus the adopted parity options)
//   emul   the GPU back end's device code run on the host (exactness tests without a GPU; never timed)
//   cuda   the GPU back end on a CUDA device (built with SF_CUDA=ON)
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "sf/evaluate.hpp"
#include "sf/hd.hpp"
#include "sf/stream.hpp"

namespace sf {

constexpr int kMaxCells = 1024;

enum class TrialOrder : int32_t {
  CellMajor = 0,   // g = c * per_cell + j (fixed-n tasks)
  RoundRobin = 1,  // g = j * cells + c (deadline tasks; unbounded)
  Enumerate = 2,   // cell-major, j = start = every valid start of the cell
};

// Plain data so it can be copied to a device as is.
struct TrialSpace {
  TrialOrder order = TrialOrder::CellMajor;
  int32_t cells = 0;
  int64_t per_cell = 0;               // CellMajor
  StreamKey key{};                    // key.cell is replaced by stream_cell[c]
  uint32_t stream_cell[kMaxCells];    // stream cell code of each cell
  int64_t valid[kMaxCells];           // M_c = N - L_c + 1
  int64_t offset[kMaxCells + 1];      // Enumerate: first g of each cell
  int64_t total = 0;                  // number of trials (RoundRobin: INT64_MAX)
};

// g -> (cell, j, start). Returns false only if the stream exhausted its attempts (the runner aborts).
SF_HD bool resolve_trial(const TrialSpace& sp, int64_t g, int32_t& cell, int64_t& j, int64_t& start) {
  if (sp.order == TrialOrder::CellMajor) {
    cell = static_cast<int32_t>(g / sp.per_cell);
    j = g - static_cast<int64_t>(cell) * sp.per_cell;
  } else if (sp.order == TrialOrder::RoundRobin) {
    cell = static_cast<int32_t>(g % sp.cells);
    j = g / sp.cells;
  } else {
    int32_t lo = 0, hi = sp.cells - 1;  // last cell whose offset <= g
    while (lo < hi) {
      const int32_t mid = (lo + hi + 1) / 2;
      if (sp.offset[mid] <= g) lo = mid;
      else hi = mid - 1;
    }
    cell = lo;
    j = g - sp.offset[lo];
    start = j;
    return true;
  }
  StreamKey key = sp.key;
  key.cell = sp.stream_cell[cell];
  uint64_t s = 0;
  const bool ok = stream_draw(key, static_cast<uint64_t>(j), static_cast<uint64_t>(sp.valid[cell]), s);
  start = static_cast<int64_t>(s);
  return ok;
}

// Per-length preparation of one cell (stage S4).
struct CellPlan {
  int32_t length = 0;
  double strength = 0;
  EventKind kind = EventKind::Both;
  double median_range = 0;
  EventTable table;
};
CellPlan make_cell_plan(const Series& base, int32_t length, double strength, EventKind kind);

// Trials [g0, g1) of a space, or an explicit list (cells/starts non-null, count = g1 - g0).
struct TrialBatch {
  const TrialSpace* space = nullptr;
  int64_t g0 = 0, g1 = 0;
  const int32_t* cells = nullptr;
  const int64_t* starts = nullptr;
  int64_t count() const { return g1 - g0; }
};

// Results of trial i = g - g0: policy p's bits at word i * words + p / 64, bit p % 64.
struct BatchOutput {
  int words = 0;
  uint64_t* hit = nullptr;
  uint64_t* overlap = nullptr;
  uint8_t* done = nullptr;                   // 1 once trial i is complete and its bits are in host memory
  std::vector<TrialTrace>* traces = nullptr;  // optional: per-stage data of every trial (gate)
};

inline int policy_words(size_t policies) { return static_cast<int>((policies + 63) / 64); }

// Per-worker (CPU) or per-device (GPU) records of one evaluate() call.
struct WorkerRecord {
  int cpu_start = -1, cpu_end = -1;
  std::string affinity_start, affinity_end;
  int64_t trials = 0;
  double busy_s = 0, last_done_s = 0;  // since the call began
};

struct BackendStats {
  int64_t trials = 0;
  StageTimes sum;            // summed over trials (CPU); GPU: kernel-phase wall times
  TrialCounts counts;        // summed (diag)
  int64_t bytes_h2d = 0, bytes_d2h = 0;
  int64_t peak_device_bytes = 0;
  std::vector<WorkerRecord> workers;
};

struct BackendConfig {
  int threads = 1;           // cpu: workers; GPU paths: host threads for host-side work
  EvalOptions eval;          // cpu: parity options PC2-PC4 (+ diag)
  bool event_table = false;  // PC1
  bool group_by_cell = false;  // PC6
  // GPU
  std::string path = "inc";  // inc (G-inc) or hyb (G-hyb)
  int64_t microbatch = 64;   // trials resident at once; 0 = largest that fits (b_max)
  int64_t submission = 1;    // microbatches per host round trip
  double memory_fraction = 0.8;
  int device = 0;
  std::string fault;         // gate negative controls only (SPEC §5.2 item 2)
  std::string describe() const;  // stable text used in the base-state key and the run record
};

class BatchEvaluator {
 public:
  virtual ~BatchEvaluator() = default;
  virtual std::string name() const = 0;
  // S4: cells of the next batches; replaces earlier plans
  virtual void set_cells(const std::vector<CellPlan>& cells) = 0;
  // S5: evaluates trials of the batch; before starting a trial (CPU) or a submission (GPU) it calls stop(), and
  // once that returns true it issues no new work. In-flight work is completed.
  virtual void evaluate(const TrialBatch& batch, BatchOutput& out, const std::function<bool()>& stop) = 0;
  // Gate item 8: trial resolution and Philox blocks computed by the back end itself (on the device for GPU paths).
  // ctr_key holds 6 words per block (counter c0..c3, key k0, k1); out receives 4 words per block.
  virtual void resolve_starts(const TrialSpace& sp, int64_t g0, int64_t n, int32_t* cell, int64_t* start);
  virtual void philox_blocks(const uint32_t* ctr_key, int64_t n, uint32_t* out);
  // Synchronises outstanding device work (stage boundaries only).
  virtual void sync() {}
  virtual BackendStats take_stats() = 0;  // and reset
  virtual std::string device_info() const = 0;
};

// Stage S2 of a back end: thread pool start (cpu) or device context creation (cuda); nothing for emul.
void device_init(const std::string& name, const BackendConfig& cfg);

// name: cpu | emul | cuda. The base state must outlive the back end.
std::unique_ptr<BatchEvaluator> make_backend(const std::string& name, const BaseState& base, const BackendConfig& cfg);
bool backend_available(const std::string& name);

// Detection bits of one trial into the output words.
inline void set_bits(BatchOutput& out, int64_t i, size_t p, bool hit, bool overlap) {
  const uint64_t m = uint64_t{1} << (p % 64);
  const size_t w = static_cast<size_t>(i) * out.words + p / 64;
  if (hit) out.hit[w] |= m;
  if (overlap) out.overlap[w] |= m;
}

}  // namespace sf

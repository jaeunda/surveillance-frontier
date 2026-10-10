# Phase 2 implementation specification

Companion to the [protocol](README.md); parameters in [`protocol.json`](protocol.json). This file says what has to
be built and which behaviour is part of the contract. Section numbers are referenced from the protocol. Items marked
**contract** may not change after the freeze without a dated amendment; everything else is implementation detail.

## 1. Components

| Component | Where | Purpose |
|---|---|---|
| `sf::BatchEvaluator` interface | `engine/include/sf/` | Evaluate a batch of (cell, trial id) for all policies of a task; CPU and CUDA back ends behind one interface |
| CPU back end | `engine/src/` | Wraps the existing incremental `Evaluator`, one per worker thread; CPU-best = this plus adopted parity items (§7) |
| CUDA back end | `engine/cuda/`, CMake option `SF_CUDA=ON` (off by default) | G-inc or G-hyb (§6). The CPU-only build must not change when the option is off |
| `sf_run` | `engine/apps/sf_run.cpp` | Task runner (§2), used for every Phase 2 timing |
| `sf_gate` | `engine/apps/` + `engine/tests/` | Correctness gate C (§5) and the shared exactness checker |
| Harness | `experiments/phase2-gpu/run_task.py` | External clock, cold / warm / deadline runs, run records (§8) |
| Launch driver | `experiments/phase2-gpu/run_launch.sh` | One launch: gates, seeded condition order, upload of results, `--shutdown` |
| Analysis | `experiments/phase2-gpu/analyze.py`, `make_figures.py` | Verdicts and figures from stored records only (§10) |

The CUDA build ships SASS only, so no JIT runs at start-up: `sm_89` (L40S, the measured build), `sm_120` (desktop
RTX 5060 Ti: development, partial gate, D3), and `sm_75` (optional Colab T4 exactness check). Each architecture's
binary is gated on its own; only the `sm_89` binary produces verdict timings. The CUDA toolkit version must support
all three and is pinned at the freeze.

## 2. Task runner `sf_run`

**Input.** A task file (JSON) with the series path and SHA-256, the policy list, cells (L, q, event kind), trials per
cell or enumeration, coverage and α, the stream key and replicate, the deadline mode, and the output directory. The
T1/T2/T-ref/T-ref-est task files are generated from `protocol.json` and committed.

**Modes.**

| Mode | Invocation | Notes |
|---|---|---|
| Cold | `sf_run TASK --device cpu\|cuda --out DIR` | One task, then exit |
| Warm | `sf_run --serve SOCKET --device ...` | Persistent. Receives task files, answers after the output fsync. Base state cached by key (§2.2) |
| Deadline | cold + `--deadline-ns T --drain-ms M` | H4. T is an absolute `CLOCK_MONOTONIC` time from the harness |
| Enumerate | task with `"enumerate": true` | T-ref; every valid start of each cell, start order = index order |

**2.1 Stages (contract).** The runner writes a `CLOCK_MONOTONIC` nanosecond timestamp at each boundary S0–S8
([protocol](README.md#measurement)). On the GPU, a stage boundary is taken after the work of that stage is complete
on the device (stream synchronisation at boundaries only; never inside S5 in timed runs). Warm-up trials use stream
purpose `warmup` (§3) and are inside the measured time on both devices; their number per device is set at the freeze.

**2.2 Base-state key (contract).** The cache key of a prepared base state is (input SHA-256, window ladder, policy
list, s<sub>floor</sub>, max event length, back end and its configuration). A warm run reuses a base state only on
an exact key match; T1 and T2 have different keys.

**2.3 Outputs.**

| Task type | Files |
|---|---|
| Fixed n (T1, T2) | `cells.csv`: per cell and policy k, n, lower, upper, half-width; `hits.bin`: bit-packed per (trial, policy), trial-major; `stages.json` |
| Enumeration (T-ref) | `ref_bits.bin`: bit-packed per (cell, start, policy); `rates.csv`: exact rate per cell and policy; `stages.json` |
| Deadline (T-ref-est) | `prefix.json`: global prefix length G, N<sub>c</sub>, k<sub>c</sub> per cell and policy; intervals at the largest reached checkpoint; completed-but-unused trial count; `stages.json` |

All outputs are fsynced (file and directory) before the runner exits or acknowledges. Each output carries a SHA-256
used for the cross-device identity check.

**2.4 Intervals (contract).** Clopper–Pearson from the regularised incomplete beta inverse, in C++ on the host,
shared by both back ends. Before Stage A it is validated against SciPy `beta.ppf` for every k at n = 2,449
(α = 0.05) and n = 11,726 (α = 0.05/3,264), and at every checkpoint n with α = 0.05/(360·11): absolute difference
≤ 10<sup>−12</sup>, and the same pass/fail for the width check. T2's worst half-width is 3·10<sup>−9</sup> below h,
so the check uses the C++ value and logs the SciPy cross-check.

**2.5 Spot verification.** After a fixed-n run, outside the timed region, 256 trial ids drawn from stream purpose
`spot` are re-evaluated with the CPU tail method and compared in all fields (§5.1).

**2.6 Deadline behaviour (contract).** No new work is issued after `T − drain`. In-flight batches are drained. A
trial counts as complete if its result is in host memory when the prefix is computed. The prefix is the largest G
such that all global indices g < G are complete, and N<sub>c</sub> = number of g < G with g mod 15 = c. Aggregation,
intervals, writing, and fsync follow. The run is **late** if the harness sees the exit after the deadline; late runs
and runs without valid output count as N<sub>c</sub> = 0 for every cell. The drain margin per device is 1.5 × the
largest (exit − stop) time in 20 calibration runs (10 at B = 10 s, 10 at B = 300 s, stream purpose `calibration`),
rounded up to 10 ms, and is fixed at the freeze. The scheduler may process trials in any order, but trials outside
the prefix never count.

## 3. Trial stream `sf-stream-v1` (contract)

A trial's start depends only on (key, task, series, cell, j, replicate, purpose), never on device, thread count,
batch, or schedule.

- Generator: Philox4x32-10 (Random123 specification), implemented once in a header usable from C++ and CUDA.
- Key (k0, k1) = (seed mod 2<sup>32</sup>, seed div 2<sup>32</sup>), seed = 20261010.
- Counter: c0 = j mod 2<sup>32</sup>, c1 = j div 2<sup>32</sup>,
  c2 = task (8 bits) ≪ 24 | series (4 bits) ≪ 20 | cell (20 bits),
  c3 = purpose (4 bits) ≪ 28 | replicate (20 bits) ≪ 8 | attempt (8 bits).
  Codes for task, series, and purpose are in `protocol.json`. Distinct trials never share a counter.
- Output words x0..x3 give u<sub>a</sub> = x0 | x1 ≪ 32 and u<sub>b</sub> = x2 | x3 ≪ 32. Candidates in order:
  attempt 0 u<sub>a</sub>, attempt 0 u<sub>b</sub>, attempt 1 u<sub>a</sub>, …
- Mapping to [0, M), M = N − L + 1 (Lemire): m = u·M as a 128-bit product, l = low 64 bits. If l < M, compute
  t = (2<sup>64</sup> − M) mod M and reject when l < t. Otherwise start = high 64 bits of m. After 256 attempts
  (probability below 10<sup>−1000</sup> for M < 2<sup>32</sup>) the runner aborts with an error.
- Tests: the Random123 `kat_vectors` entries for `philox4x32_10`, identical on CPU and GPU; a test-only M near
  2<sup>63</sup> that forces rejections, with the expected sequence computed independently in Python; equality of
  1,000,000 starts between CPU and GPU.
- Phase 1's `seed + worker_id` streams remain only for reproducing Phase 1.

Cell index: T1/T2 c = i<sub>L</sub>·17 + i<sub>q</sub> (lengths and strengths ascending); T-ref and T-ref-est
c = i<sub>L</sub>·5 + i<sub>q</sub> over L (8, 32, 128) and q (16, 32, 64, 128, 256). Replicates: 0 for fixed-n tasks;
H4 seeds of launch pair i are replicates 10(i − 1) + 1 … 10i.

## 4. Numerical contract (contract)

The CPU code in `engine/` defines the semantics. The GPU reproduces them operation by operation; nothing is
recomputed in a different way.

| Quantity | CPU definition | GPU rule |
|---|---|---|
| Price event factor | `float(1.0 + q·r_L·shape(j))` in double, `apply_event` | Host table per (L, q) from **the same function** (refactored out of `apply_event` and called by both); uploaded as float |
| Event bar values | `base·factor` (float), volume `base·float(1 + q)` | `__fmul_rn`, factors from the host |
| Fixed-point volume | `llround(ldexp(double(v), 24))`: round half **away from zero** | Explicit half-away-from-zero rounding of the exact double; `__double2ll_rn` (half to even) is forbidden |
| Overflow | `fixed_volumes` rejects windows that could overflow int64 | Host performs the same check before upload; same error |
| Volume feature | int64 → double (round to nearest) → × 2<sup>−24</sup> → float | `__ll2double_rn`, exact scaling, `__double2float_rn`; direct int64 → float is forbidden (double rounding differs, e.g. sum = 2<sup>54</sup> + 2<sup>30</sup> + 1) |
| Range | `(mx − mn) / mn` in float, NaN if mn ≤ 0 | `__fsub_rn`, `__fdiv_rn`, same NaN rule |
| Tail cap | `floor(N_k · pow(10, −s))` | Host table per (k, s) |
| S | `−log10(c / N_k)` in double | Host table per (k, count ≤ cap); comparing integer ratios instead is forbidden unless proven equivalent |
| Order | S descending, then k, then start; local peak ties go to the earlier start | Same comparison function, shared source |
| Greedy merge | Sequential in S order within a trial | Never parallelised within a trial; parallelism over trials and policies only |

Compiler: no `--use_fast_math`; `--fmad=false` for device code on the feature path; host code keeps the Phase 1 flags.
Host results must not depend on the CPU: the desktop (AVX2) and the reference CPU (AVX-512) both build with
`-march=native`, and GCC contracts floating-point expressions into FMA by default. Item 9 runs on both machines and
the reference oracle subset is recomputed on the reference CPU (§5.3). If either differs, `-ffp-contract=off` is
applied to the event-factor and feature code, item 9 is re-checked, and the change is recorded.
The Phase 2 CPU build must reproduce the 18 Phase 1 raw hit files byte for byte (§5.2 item 9) so that shared
refactoring cannot drift.

## 5. Correctness gate C

### 5.1 Exactness checker (contract)

One shared checker compares two results per (trial, policy): the number of episodes and, for each, k, start, window,
count, and S **by bit pattern**; then the hit and overlap bits. For stage differentials, valid feature slots are
compared by bit pattern; invalid slots must be NaN on both sides. It reports the first stage that differs (features
→ tail counts → ordered candidates → episodes → hit/overlap) and the trial id. This replaces the Phase 1 `verify`
comparison (k, start, S) and the NaN-tolerant equality in `sf_test`.

### 5.2 Gate items (all must give 0 mismatches)

| # | Item |
|---|---|
| 1 | `sf_test` synthetic suite (tie-heavy data, 18 policies × 42 events, both series ends, the 4 existing negative controls) on every back end |
| 2 | GPU negative controls, each with a **witness** input on which the faulty variant differs: device `log10` for S; reversed peak tie rule; tail cap − 1; direct int64 → float volume; half-to-even fixed-point rounding; a fast-math build of the feature path. Each must be detected. A control that does not differ on its witness invalidates the gate run until the witness is fixed |
| 3 | Real data, both series: P105; lengths of the task grid (2…256) and 6, 120; **all 17 task strengths** (including non-integer values) and 0; price-only and volume-only; starts at both series ends; a sequence of alternating (L, q) events applied and restored with state equality after it. 64 starts per (series, L, q, kind) from purpose `gate` |
| 4 | Stage differential (§5.1) on item 3 |
| 5 | N ≤ 4,096: every start, GPU vs CPU incremental vs full |
| 6 | Determinism: the same task with two batch sizes and launch configurations gives the same output hashes |
| 7 | Numeric golden vectors: the volume double-rounding witness, half-integer fixed volumes, overflow rejection, tiny mn, counts at cap and cap + 1, rows with cap = 0, a last partial batch |
| 8 | Stream tests (§3) |
| 9 | CPU only: the 18 Phase 1 hit files (64k prefix enumeration) reproduced byte for byte |
| 10 | T-ref enumeration bits equal the CPU reference |

The partial gate of Stage C runs items 1, 2, 5, 7, 8 and item 3 restricted to week / L = 32 / q = 16 / P24, first on
the desktop GPU during development and then on the L40S before path selection counts. The full gate (D0) runs
everything on the frozen `sm_89` build on the L40S; the same full gate on the desktop is required before D3 timings,
and on Colab it is optional. The GPU gate compares against CPU-best golden outputs, plus the full method on item 5.

### 5.3 Where CPU references come from

Golden outputs and the T-ref reference are computed on the desktop with the frozen commit. Their validity does not
depend on the machine, but this is checked rather than assumed: the gate step of every Stage D CPU launch recomputes
the golden outputs on the reference CPU, and every T-ref timing run there regenerates all reference bits; hashes must
match. A mismatch stops Stage D until the cause is found (§4).

## 6. GPU design constraints and memory plan

- **No dense per-trial count arrays.** The CPU's `cc_` is K × N int32 per worker (34.6 MiB per trial on the week).
  The GPU keeps per trial and row only the merged tails and the set of starts in both tails, sorted by start.
  A start outside that set has count `kAboveCap`, which is exactly the CPU semantics. The peak test finds
  neighbours within ±r by binary search on that set. The tie rules are unchanged.
- **No truncation** of tails or candidates (for example to top_k) before the greedy merge.
- **Worst-case working set per trial** (bytes, row k, cap<sub>k</sub> from s<sub>floor</sub>, W<sub>k</sub> = L + w<sub>k</sub> − 1):
  Σ<sub>k</sub> [16·W<sub>k</sub> + 72·(cap<sub>k</sub> + 1)]: changed values and fixed volumes, two merged tails and
  tail lists, the both-set, peaks, and candidates. On the week at s<sub>floor</sub> = 1.5 this is about 21 MB.
  Shared resident state: series, heads, S and cap tables, base episodes.
- **Batches.** The *microbatch* (trials resident at once) is the largest b with
  b · working set + shared state ≤ 0.8 × free device memory after context creation. The *submission batch* (trials
  per host round trip) is a multiple of it. Batch size per (series, policy grid) is chosen from {64, 256, 1,024,
  b<sub>max</sub>} with the same design as the CPU thread count (protocol, *Implementations compared*): highest
  geometric mean of median steady-state throughput over L ∈ {2, 32, 256} at q = 16 (3 × 5 s each), then a 60 s
  sustained check of the two best at L = 32 that decides if it disagrees. It is fixed at the freeze. Allocation failures are results: they are recorded with the size that failed.
- **Two-pass sizing** (count, then allocate exactly) is allowed if it stays exact.
- **G-hyb**: candidates go to the host per trial; host ordering and merge use the CPU code.

## 7. CPU parity

| ID | Candidate | Also on GPU |
|---|---|---|
| PC1 | Event factor table per (L, q) instead of per-trial computation | yes |
| PC2 | S table lookup instead of `log10` per candidate | yes |
| PC3 | Cap / threshold table per (k, s<sub>min</sub>) | yes |
| PC4 | Sparse both-set with binary-search peak test instead of the dense `cc_` | yes |
| PC5 | Scratch reuse across trials and cells | yes |
| PC6 | Trial batching by cell (shared per-length preparation) | yes |

Rule (protocol): a candidate is adopted on the CPU if the CPU gate passes and the geometric-mean steady-state
throughput over three calibration points (week P24 L = 32; week P1 L = 2; quarter P24 L = 256; q = 16) improves by
≥ 2% (3 × 5 s each). Otherwise the measured result is recorded as the reason. Improvements found later are added to
this table with the same rule. The log `parity.md` in the results lists each candidate, its measurements, and the
decision.

## 8. Instrumentation and harness

- **Select sub-stages** (CPU incremental and both GPU paths): peak test, candidate collection, S and sort,
  per-policy merge, reset; timers in every build (overhead must be < 1% at the canonical point, checked once against
  a timer-free build). In diagnostic mode only, counts are recorded: both-set size per row, neighbour comparisons,
  peaks, candidates. Phase 1's `select_ms` is not read as greedy-merge cost.
- **Harness.** Cold: `sync; echo 3 > /proc/sys/vm/drop_caches`, then `t0 = time.monotonic_ns()` immediately
  before spawn, `t1` at `waitpid`. Warm: the server is started and receives a first task outside the timing (its
  key matches the timed task). Then each submit is timed until the acknowledgement. Deadline: the harness passes
  `t0 + B` and records late runs.
- **Resource sampler** (contract for what is recorded, not how): each run executes in its own cgroup v2; the harness
  reads `memory.peak` at exit and samples `MemAvailable`, `/proc/vmstat` (pswpin, pswpout), `/proc/stat` steal, and
  per-core MHz from `/proc/cpuinfo` at 1 Hz, and on GPU instances `nvidia-smi` clocks and memory at 1 Hz. The
  sampler runs on the harness side; its overhead is checked once at the canonical point (< 1% throughput change).
  The runner records `getrusage` major faults and peak RSS (MiB, from `ru_maxrss`), and each worker's
  `sched_getcpu()` and affinity mask at loop start and end.
- **Profiling runs** (exploratory, never timed): `perf stat` with cycles, instructions, LLC loads and misses, and
  stalled cycles where available, at the canonical point and the three calibration lengths; if the VM does not
  expose a counter, that is recorded.
- **Timeouts**: 3 × the CPU-best projected time of the condition (fixed at the freeze); the process is killed and
  the run recorded as censored at the timeout.
- **Order**: every launch runs all its conditions and repetitions in one order shuffled with seed
  `20261010 + launch index`, written to the launch record before the first run.

## 9. Run records and manifest

Per launch, `env.json`: full commit SHA and dirty flag, binary SHA-256, compiler, nvcc, driver, and CUDA runtime
versions, compile flags, `lscpu`, `numactl --hardware`, governor, THP, `OMP_*`, `nvidia-smi -q` (clocks, ECC,
persistence, power limit), instance type, region, availability zone, instance id, tenancy, purchase option, launch
and termination times (from the EC2 API, not run timers), on-demand price from the official AWS price list on the
run date (SKU and SHA-256 of the price file), spot price of the zone at launch and interruption notices (spot only),
hardware identity check result (protocol, *Execution environment*), input SHA-256s, stream version, Python package
versions. Thread and batch calibration records, including the sustained check and any flip, go to
`calibration.jsonl`.

Per run, one line in `runs.jsonl`: condition, mode, repetition, position in the order, external times, stage
timestamps, status (ok / late / timeout / error), output SHA-256, identity check result, peak RSS, cgroup peak memory,
minimum `MemAvailable`, swap and major-fault counts, steal time, clock summary, start/end affinity, peak device
memory, bytes transferred, and the per-worker or per-batch completion counts and times.

Layout: `results/<env>_<date>/launch-<i>-<cpu|gpu>/{env.json, calibration.jsonl, runs.jsonl, outputs/, gate/}`, `x2-c7i-8xlarge/` (same layout), plus `reference/`
(T-ref bits, oracle log), `parity.md`, `protocol-freeze.json`, and `SHA256SUMS`. Large binaries (reference bits,
per-trial hits) go into compressed archives with checksums. If the public repository cannot hold them, only their
hashes and an external archive location are committed.

## 10. Analysis

- `analyze.py` computes every verdict from the stored records alone: R1, R2 (re-projection with the Phase 1
  `scenario_space.project` method on Phase 2 rates), C, H3 and H4 (launch-pair log-ratios, t-intervals, categories,
  censoring), cost per task on the stated basis (instance-hours × on-demand price; spot and interruptions
  separately), D1 (non-negative least squares on rows scaled by 1/observed cost, held-out errors), and the
  exploratory tables.
- D1 model (contract): per-trial cost t = β0 + β1·C + β2·C<sub>w</sub> + β3·W + β4·P (+ β5/b on the GPU), with
  C = Σ<sub>k</sub>(cap<sub>k</sub> + 1), C<sub>w</sub> = Σ<sub>k</sub>(cap<sub>k</sub> + 1)·max(1, ⌊w<sub>k</sub>/2⌋),
  W = Σ<sub>k</sub>(L + w<sub>k</sub> − 1), P = policy count, b = batch size; sums over the task's windows k < K<sub>max</sub>.
  Post-execution counts (both-set size, candidates) are used only in a separate explanatory fit.
- Figures and tables are regenerated from stored results, and byte identity on regeneration is checked, as in Phase 1.

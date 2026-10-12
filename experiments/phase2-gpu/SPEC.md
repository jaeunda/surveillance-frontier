# Phase 2 implementation specification

Companion to the [protocol](README.md) (design r2); parameters in [`protocol.json`](protocol.json). This file says what
has to be built and which behaviour is part of the contract. Items marked **contract** apply to every machine, pilot
and formal alike, and may not change after the freeze without a dated entry in the protocol; everything else is
implementation detail.

## 1. Components

| Component | Where | Purpose |
|---|---|---|
| `sf::BatchEvaluator` interface | `engine/include/sf/` | Evaluate a batch of (cell, trial id) for all policies of a task; CPU and CUDA back ends behind one interface |
| CPU back end | `engine/src/` | The incremental `Evaluator`, one per worker thread; CPU-best = this plus adopted parity items (§7) |
| CUDA back end | `engine/cuda/`, CMake option `SF_CUDA=ON` (off by default) | G-inc or G-hyb (§6). The CPU-only build does not change when the option is off |
| `sf_run` | `engine/apps/sf_run.cpp` | Task runner (§2), used for every Phase 2 timing, pilot and formal |
| `sf_gate` | `engine/apps/` + `engine/tests/` | Correctness gate C (§5), the shared exactness checker, the reference oracle |
| Experiment library | `experiments/phase2-gpu/p2/` | Tasks, machines and identity, the harness (§8), calibration designs, projections (§11), statistics |
| Pilot and scaling | `pilot.py`, `scaling.py` | Part I and Part II of the protocol (§11) |
| Formal | `calibrate.py`, `reference.py`, `freeze.py`, `launch.py`, `run_launch.sh` | Part III |
| Analysis | `analyze.py`, `make_figures.py` | Verdicts and figures from stored records only (§10) |

The CUDA build ships SASS only, so no JIT runs at start-up. Each machine builds for its own architecture (`sm_120`
lab RTX 5060 Ti; Colab `sm_75`/`sm_80`/`sm_89` as assigned; `sm_89` L40S), and each architecture's binary passes the
gate on that device before any of its timings count. The formal GPU's toolkit and architecture are pinned at the freeze.

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
on the device. Inside S5 a timed run adds no synchronisation for timing: GPU stage timers exist only in diagnostic runs
(`--diag`). The host does wait for each microbatch's results (bits, or candidates on G-hyb) before using them; in the
current prototype these copies are synchronous, a performance gap recorded in `IMPLEMENTATION.md`, not a measurement
boundary. Warm-up trials use stream purpose `warmup` (§3) and are inside the measured time on both devices; their
number per device is set at the freeze.

**2.2 Base-state key (contract).** The cache key of a prepared base state is (input SHA-256, window ladder, policy
list, s<sub>floor</sub>, max event length, back end and its configuration). A warm run reuses a base state only on
an exact key match; T1 and T2 have different keys.

**2.3 Outputs.**

| Task type | Files |
|---|---|
| Fixed n (T1, T2) | `cells.csv`: per cell and policy k, n, lower, upper, half-width; `hits.bin`: bit-packed per (trial, policy), trial-major; `stages.json` |
| Enumeration (T-ref) | `ref_bits.bin`: bit-packed per (cell, start, policy); `rates.csv`: exact rate per cell and policy; `stages.json` |
| Deadline (T-ref-est) | `prefix.json`: global prefix length G, N<sub>c</sub>, k<sub>c</sub> per cell and policy; intervals at the largest reached checkpoint; completed-but-unused trial count; `stages.json` |

All outputs are fsynced (file and directory) before the runner exits or acknowledges. Each output carries a SHA-256.
The result hash used for the cross-device identity check covers the whole result: for fixed n, the SHA-256 of the
`cells.csv` and `hits.bin` hashes together (so two trials whose bits are swapped within a cell, which keeps every k, still
change it); for T-ref, `ref_bits.bin`; for deadline runs, `prefix.json`.

**2.4 Intervals (contract).** Clopper–Pearson from the regularised incomplete beta inverse, in C++ on the host,
shared by both back ends. Before Stage A it is validated against SciPy `beta.ppf` for every k at n = 2,449
(α = 0.05) and n = 11,726 (α = 0.05/3,264), and at every checkpoint n with α = 0.05/(360·11): absolute difference
≤ 10<sup>−12</sup>, and the same pass/fail for the width check. T2's worst half-width is 3·10<sup>−9</sup> below h,
so the check uses the C++ value and logs the SciPy cross-check.

**2.5 Spot verification.** After a fixed-n run, outside the timed region, 256 trial ids drawn from stream purpose
`spot` are re-evaluated with the CPU tail method and compared in all fields (§5.1).

**2.6 Deadline behaviour (contract).** The *cutoff* is `T − drain`: from then no new trial (CPU) or microbatch (GPU,
checked before each microbatch, also inside a submission) is issued. Work in flight at the cutoff finishes. A trial
counts as complete if its result is in host memory when the prefix is computed. The prefix is the largest G such
that all global indices g < G are complete, and N<sub>c</sub> = number of g < G with g mod 15 = c. Aggregation,
intervals, writing, and fsync follow. `prefix.json` records `cutoff_ns`, `stop_seen_ns` (first check past the
cutoff), `loop_end_ns` (in-flight work done) and `completed_unused`. The run is **late** if the harness sees the exit
after the deadline; late runs, runs stopped by their timeout (2B + 120 s), and runs without valid output count as
N<sub>c</sub> = 0 for every cell. The drain margin per device is 1.5 × the largest (exit − cutoff) in 20 calibration
runs with drain 0 (10 at B = 10 s, 10 at B = 300 s, stream purpose `calibration`, cache-cold, the frozen
configuration), rounded up to 10 ms, and is fixed at the freeze. So the margin covers the in-flight remainder as well
as aggregation and output (r2.1: r2 measured from the end of the loop and missed the in-flight part). A drain check
with the margin (2 runs at every H4 budget) is recorded in the freeze. The scheduler may process trials in any order,
but trials outside the prefix never count.

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
Host results must not depend on the CPU: an AVX2 machine (lab PC) and an AVX-512 machine (`c7i`) both build with
`-march=native`, and GCC contracts floating-point expressions into FMA by default. Item 9 runs on every machine (pilot
P0 and every formal CPU gate step), and every formal T-ref run regenerates the reference bits (§5.3). If either differs, `-ffp-contract=off` is
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

The partial gate runs items 1, 2, 5, 7, 8 and item 3 restricted to week / L = 32 / q = 16 / P24. It runs on the
device itself before any GPU timing of that machine counts: in pilot P0 for each path, and on the formal GPU before
path selection. The full gate (D0) runs everything on the frozen build on the formal GPU. Host emulation (`emul`) runs
the same kernel bodies for development checks but never counts as a device gate. The GPU gate compares against CPU-best
golden outputs, plus the full method on item 5.

### 5.3 Where CPU references come from

Golden outputs and the T-ref reference (formal Stage B) may be computed on any machine with a gated build of the
frozen commit, including the pilot machine. Their validity does not depend on the machine, but this is checked rather
than assumed: the gate step of every formal CPU launch recomputes the golden outputs, and every formal T-ref timing run
regenerates all reference bits; hashes must match. A pilot reference built before the freeze is used only after these
hashes agree. A mismatch stops Stage D until the cause is found (§4).

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
- **Batches (contract for the sizing rule).** The *microbatch* (trials resident at once) is at most
  b<sub>max</sub> = ⌊(0.8 × F − M<sub>shared</sub>) / M<sub>trial</sub>⌋, where F is the free device memory read
  **once, after context creation and before any allocation of the back end**, so shared state is subtracted exactly
  once. An explicit microbatch above b<sub>max</sub> is refused (`no microbatch fits`). The back end reports F, total
  memory, M<sub>shared</sub>, M<sub>trial</sub>, scratch bytes, b<sub>max</sub>, the free memory after its resident
  allocations and after the per-task tables (the last allocation), the live and peak live bytes (a re-allocation
  replaces the earlier buffer in the count); the harness adds the `nvidia-smi` maximum during the run. Planned bytes
  (M<sub>shared</sub> + b · M<sub>trial</sub>) are always reported next to the reported allocation (F − free after
  allocation). The *submission batch* (trials per host round trip) is a multiple of the microbatch. Pilot: a sweep up to
  b<sub>max</sub>. Formal: chosen per (series, grid) from {64, 256, 1,024, b<sub>max</sub>} with the thread-count design
  (short + sustained). Allocation failures are results, recorded with the size that failed.
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
  per-policy merge, reset. CPU timers in every build (overhead < 1% at the canonical point, checked once against a
  timer-free build); GPU stage events only in diagnostic runs (`--diag`), created once and reused. On G-hyb the
  "order" span includes the candidate copy and the "merge" span is host work, so neither is device time. Counts
  (both-set size, neighbour comparisons, peaks, candidates) in diagnostic mode only, CPU only.
- **Configurations.** A run's back end is a configuration: CPU (threads, optional CPU list for `taskset`, parity set,
  warm-up trials) or GPU (path, microbatch, submission, host threads, warm-up trials), identified by a key that
  contains every one of them (`cpu:t6@0-5:none`, `cuda:inc:b256:h8:s1:w0`); two runs with one key ran one
  configuration. Host threads are one count per (series, grid), never a table. Formal batches are integers (b<sub>max</sub>
  is resolved at calibration). OpenMP runs with `OMP_PLACES=cores OMP_PROC_BIND=spread` inside the CPU list.
- **Clock (contract).** `t0 = time.monotonic_ns()` immediately before spawn (cold, deadline) or before the request
  (warm); `t1` at `waitpid` or at the acknowledgement sent after the output fsync.
- **Cold kinds (contract).** The harness runs `sync; echo 3 > /proc/sys/vm/drop_caches` (directly or by `sudo -n`).
  If that write succeeds the run is *cache-cold*, otherwise *process-cold*; the kind is in the record and the two are
  never aggregated together. The capability in the manifest is such an actual drop, not a guess from root or sudo.
  Formal launches refuse to start where the cache cannot be dropped. A pilot plan may fix `--cold-kind process-cold`
  (a P4 verification of a process-cold baseline).
- **Warm blocks.** A server (`sf_run --serve`) is primed with a task of the same base-state key outside the timing;
  then the timed requests of that condition follow. A reply with `base_cached = false` makes the run `not_warm`. A warm
  request does not repeat S1: the input is hashed once per file identity (device, inode, size, mtime). The server is
  closed before any cold or deadline run, so a cached state never holds memory during other runs. Its cgroup peak covers
  the server's lifetime (priming and earlier requests) and is labelled so.
- **Records that survive interruptions (contract).** Before every run the harness appends a `started` record (run id,
  kind, condition, configuration key) to `attempts.jsonl`; the terminal record carries the same run id and is appended
  as soon as the run ends (warm blocks too). A started run without a terminal record is a lost run: it invalidates a
  formal launch and counts against completion in the pilot. A harness failure is a record (`harness_error`).
- **Bounds on every run.** Formal runs: their timeout; deadline runs 2B + 120 s; warm requests and priming the
  condition's timeout (the socket times out and the server is killed); benches their seconds + 900 s. Pilot runs: the
  time box, i.e. the plan's remaining hours (`time box` status, the step stops).
- **Order.** Cold repetitions are single items and each warm block is one item; all items of a step (pilot) or launch
  (formal) run in one order shuffled with a recorded seed, written before the first run.
- **Resource sampler** (contract for what is recorded): each run in its own cgroup v2 group where permitted
  (`memory.peak` at exit); 1 Hz `MemAvailable`, `/proc/vmstat` swap and major faults, `/proc/stat` steal, per-core MHz;
  on GPU machines `nvidia-smi` SM/memory clocks, memory used, power, utilisation. The runner records `getrusage` and each
  worker's CPU and affinity at loop start and end.
- **Identity during runs.** Every fixed-n output is compared with the reference hash (pilot: the first CPU result of
  that label with 0 spot mismatches; formal: Stage B). After a fixed-n run, outside the timing, 256 trial ids from
  stream purpose `spot` (draws `--spot-from` k … k + 255; the pilot advances k by 256 per earlier check of the same
  condition) are re-evaluated by the run's own back end and configuration and by the CPU tail method and compared with
  each other and with the stored bits (§2.5). Pilot: an identity or spot mismatch of a GPU path excludes the path
  from the rest of the step (`not started (path excluded)`), later steps and all metrics; of a CPU output, it stops
  the step and the session; a check that cannot finish is `spot_error`. When a label first gets a CPU reference hash,
  every earlier output of the task is compared with it (`reverify.jsonl`); a mismatch excludes as above. An output
  checked only by the spot sample is reported as exploratory.
- **Profiling runs** (exploratory, never timed): `perf stat` cycles, instructions, LLC loads and misses, stalled cycles
  where available; unavailable counters are recorded.
- **Timeouts** (formal): 3 × the conservative CPU-best projection of the condition; the run is killed and recorded as
  censored at exactly the timeout. A condition with a censored run gets no category (r2.1, README "Verdict rules").

## 9. Run records and manifest

**Manifest** (`env.json`, per pilot session and per formal launch): profile and identity check, runtime (WSL2, Colab,
EC2, Linux), git commit and dirty flag, binary SHA-256s, CMake cache (compiler, flags, CUDA architectures), compiler and
`nvcc` versions, GPU state (model, driver, total/used/free memory, other processes, persistence mode), `nvidia-smi -q`,
`lscpu`, topology (logical CPUs, physical cores, SMT, NUMA, P/E mapping and its source), `/proc/meminfo` totals,
uptime, governor, THP, `OMP_*`, capabilities (cache drop, cgroup, persistence, perf, taskset), EC2 metadata and the
official on-demand price on AWS (with the raw Price List response), input SHA-256s, protocol SHA-256, Python packages
and `pip freeze`, the cgroup CPU quota and the CPUs this process may use (topology is restricted to them), the code
fingerprint (SHA-256 over every tracked code, protocol and task file; results and documentation excluded), and the
hardware fingerprint (runtime, CPU model, allowed CPUs, quota, RAM, GPU, driver, instance type).

**Runs** (`runs.jsonl`, one line per run): mode, cold kind, condition, configuration and key, position, external
times, stage timestamps, status (ok / late / timeout / time box / error / width_fail / not_warm / identity_mismatch /
spot_mismatch / spot_error / not started (reason)), output SHA-256 and identity result, spot mismatches and the first
mismatch, the output files with their SHA-256 (`out_manifest`, pilot), runner record (device memory fields, transfers,
select sub-stage sums, per-worker trials and affinity), cgroup peak, sampler record. **Benches** (`bench.jsonl`):
point, metadata, configuration, rate, setup, device memory fields, sampler record, or the failure with its stderr.

**Layout.**

```
results/pilot/<label>/      pilot-plan.json, amendments.jsonl, hashes.json, reference/attempt-<k>/,
                            session-<k>/{env.json, checks/attempt-<j>/, attempts.jsonl, steps.jsonl, runs.jsonl,
                            bench.jsonl, parity.md, outputs/<step>/attempt-<j>/}, pilot-summary.json, pilot-summary.md, figs/
results/scaling/            scaling-decision.json, verification_<label>_action<i>.json
results/formal/<label>/     reference/, calibration/<machine>/, protocol-freeze.json, d0/,
                            stage-a/launch-<i>/attempt-<k>/, stage-d/pair-<i>/<cpu|gpu>/attempt-<k>/,
                            stage-d/size-control/attempt-<k>/, verdicts_*.json, tables_*.md, figs/
  each attempt:             env.json, order.json, attempts.jsonl, runs.jsonl, calibration.jsonl, rates.jsonl,
                            tdiag.jsonl, threads.json, gate/, cpu_host/, outputs/, outputs_bits.tar.gz, SHA256SUMS,
                            ec2.json (after termination)
```

Attempts are created fresh and never reused or overwritten; an interrupted or invalid attempt stays and is replaced
by the next attempt of the same slot (same order and seeds). An archive is written once per attempt and read back
before the archived files are removed.

Large binaries go into compressed archives with checksums; if the public repository cannot hold them, only hashes and
an external archive location are committed.

## 10. Analysis

- `analyze.py` computes every formal verdict from stored records alone: R1, R2 (projection with the launch's own rates
  at every task length, §11), C, H3 and H4 (launch-pair log-ratios, t-intervals, categories, censoring; only
  cache-cold runs enter cold conditions), cost, D1, and the exploratory D3 (`cross-env`) and spend summary.
- **Validation before verdicts (contract, `p2/validate.py`).** An attempt counts only if: its env names the slot's
  stage, role and index; status complete; formal identity matched; it ran under the committed scaling decision, the
  freeze and the frozen code fingerprint (Stage D); every order item has exactly its records (cold 1, warm block its
  repetitions, deadline 1); every T1/T2/T-ref run is ok or censored, cache-cold when cold, reproduces the reference hash
  (Stage D) and has 0 spot mismatches; and no started run lacks a terminal record. The first valid attempt of a slot is
  used; later attempts of a valid slot never are. A pair counts only if both launches are on distinct recorded
  instances (also distinct from every other used launch), in one region, on one UTC day, in the registered order, and
  its CPU gate has the reference golden hashes. Gates count only for what they show: device, full or partial,
  64 starts (full), every item of that coverage passed, and the code and GPU they ran with.
- **H4 identity (contract).** The reference bits and rates must carry the frozen hashes; exact rates are recounted
  from the bits as float. For every live run the replicate, the round-robin structure (N<sub>c</sub> from G), the
  cells, 0 ≤ k ≤ N, and every k<sub>c</sub> against the reference bits at the stream's starts are checked; each budget
  needs exactly the registered seeds per device and pair. A failure blocks that budget's category ("invalid").
- D1 model (contract): per-trial cost t = β0 + β1·C + β2·C<sub>w</sub> + β3·W + β4·P (+ β5/b on the GPU), with
  C = Σ<sub>k</sub>(cap<sub>k</sub> + 1), C<sub>w</sub> = Σ<sub>k</sub>(cap<sub>k</sub> + 1)·max(1, ⌊w<sub>k</sub>/2⌋),
  W = Σ<sub>k</sub>(L + w<sub>k</sub> − 1), P = policy count, b = batch size; sums over windows k < K<sub>max</sub>.
  Non-negative least squares on rows scaled by 1/observed cost, fitted on the calibration split, judged on the 6
  held-out points whose workload (series and prefix, policies, cells, maximum length; `p2.tasks.workload_fingerprint`)
  is never run before D1 (`t_diag.heldout_judged`, r2.1). The other 7 held-out points have the workload of a
  calibration or T-ref cell point and are reported apart as repeat predictions. Post-execution counts are used only in
  a separate explanatory fit.
- Figures and tables are regenerated from stored results; `make_figures.py … --check` renders twice and requires byte
  identity. The H4 figure states each budget's category, late or invalid runs, and identity failures.

## 11. Pilot, projections, and scaling records

**Pilot plan (contract).** `pilot.py plan` writes `pilot-plan.json` once (never rewritten): purpose (pilot or
verification), machine profile and identity, topology, GPU state, capabilities, git state, binary, input, task and
protocol hashes, the steps, sessions, repetitions, thread candidates (from the topology; the P-core list is given
explicitly where the kernel does not expose it), microbatch sweep, bench settings, available hours, cold kind, code
fingerprint, stop rules, and any cut with its reason. Changes are appended to `amendments.jsonl` with a reason before
the run they affect; each step records how many amendments applied.

**Sessions (contract).** Every session is its own machine. `pilot.py check --session K` runs that session's P0 (CPU
regression, stream, intervals, CPU gate, item 9, the partial gate on the session's own device per GPU path) and
records its hardware fingerprint, code fingerprint and binaries. A step refuses to run without passed checks of its
session, or if the hardware, code or binaries differ from them (a new Colab assignment is a new session), a GPU step
without a path that passed that session's device gate, and any step once the hours are used up ("not started
(time)"). Hours used count checks, finished steps, and steps cut off without a finish record (their immutable
start/heartbeat records, plus one heartbeat interval under the pilot's lost-attempt policy). A step share, including
`check`, is separate from the whole session's limit; reaching only that share leaves later steps eligible.
Threads, parity and microbatch are chosen per session and used only in that session; values are pooled
across sessions only within one hardware fingerprint. The reference build starts only if its conservative projection
× 1.1 fits the remaining hours; otherwise the per-cell rates and the projection are the result.

**Colab preservation (2026-10-12).** [COLAB_RUNBOOK.md](COLAB_RUNBOOK.md) defines the operator sequence.
Python dependencies are pinned transitively in `requirements-colab.lock`; setup logs, input bytes, built binaries,
VM boot ID and occupancy observations accompany the research records. A restored session must pass artifact checks
before completed work or P0 checks are reused. Checks have a file manifest; each check item has start/end records.
Priming, fixed-task, reference and deadline outputs have manifests and are mirrored after completion outside timing.
Missing outputs invalidate metrics without altering raw records. Backup files are replaced only after SHA-256
readback, and final archives have independent checksum sidecars. Setup/backup occupancy is separate from performance
E2E and the plan's check/step hours. A new VM uses a new session even with identical hardware specifications.

**Projections (contract).**

- Fixed-n tasks: T = T<sub>setup</sub> + Σ<sub>cells</sub> n<sub>c</sub> / r(series, grid, L) (+ T<sub>output</sub>),
  with r measured at q = 16 at every task length (`calib/` points, 3 × 5 s). Scenarios use the slowest (conservative),
  median (base), and fastest (optimistic) repetition. An unmeasured length takes the nearest measured length in log L
  and is flagged interpolated or extrapolated. Warm projections have no setup term.
- T-ref: Σ<sub>c</sub> (N − L<sub>c</sub> + 1) / r<sub>c</sub> from each cell's own rate (`pilot/tref-cell_c*`),
  9,071,175 trials in total.
- Before a pilot projection is used for T-ref, the T1/T2 warm projections of the same configuration are compared with
  the observed warm E2E (observed / base is reported).
- Memory: planned M<sub>shared</sub> + b · M<sub>trial</sub> against the reported allocation and the `nvidia-smi`
  maximum (§6).

**Scaling record.** `scaling.py draft` copies the pilot observations, failures, environments, and a catalogue of every
measured metric with its value; a person completes the six items of the protocol and the actions. `scaling.py check`
(required by every later step) rejects: any `TODO` or empty judgement (schedule, alternatives, constraint, hypothesis,
falsification, stop, formal reason); no pilot observations; an action outside {none, environment, cpu-host, gpu};
`none` mixed with other actions; a target that is not a machine profile; an action without metrics; a metric without
a mode (e2e, completion, spread), a baseline equal to the value recomputed from the baseline label's raw records, three
finite positive predictions whose gains are ordered conservative ≤ base ≤ optimistic, or a gain threshold ≥ 1;
completion or spread metrics for anything but an `environment` action; a missing cost or instance-hours; formal
machines without an identity rule; a formal plan without itemised costs (`budget.formal_items`); spent + planned above
the cap; a pilot summary changed since the draft. A verification plan and every formal step require the record to pass
this check and be committed; formal steps also require each action's committed P4 verification.

**Metrics.** `e2e`: task, series, mode (cache-cold, process-cold, warm) and configuration, value = median over all ok
runs (lower is better). `rate`: task point and configuration, value = median trials/s over ok benches (probes
excluded; higher is better). `completion`: ok runs / planned runs of the condition, lost and not-started runs included
(higher is better; 0 if none finished); ok / started is reported beside it. `spread`: max / min of the ok times (lower
is better). `e2e` and `spread` exist only for an uncensored condition (every planned run ok); a condition with any
timeout, time box, lost, failed or unstarted run gives no value and no `best` choice. Excluded GPU paths and sessions
(output check failures) give no metric. A metric may name a session; a label whose sessions ran
on different hardware must. The configuration may be a key or `best`, `best-cpu`, `best-gpu` (each machine's own best).
Gain and verdicts as in the protocol (P4).

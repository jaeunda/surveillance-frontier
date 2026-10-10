# Phase 2 — GPU engine: when does a bit-identical GPU implementation beat the fastest exact CPU method end to end?

Status: **design frozen, not started** (2026-10-10; amended twice the same day before any run: execution environment,
then CPU calibration, cost basis, and launch records after a review of the Phase 1 environment). No Phase 2 measurement has been run. This file is the protocol;
[`SPEC.md`](SPEC.md) is the implementation specification and [`protocol.json`](protocol.json) holds every frozen
parameter in machine-readable form. Values that this protocol says are fixed at a later stage (thread counts, batch
size, deadline margin, timeouts) are fixed **by a rule stated here** and committed in `protocol-freeze.json`
before the measurements they govern.

## Starting point

Phase 1 ([write-up](../../docs/phase1.md)) changed the baseline: the incremental method runs 349x more trials per second
than full recomputation (week, K = 15, one policy, one thread), and the stress-testing tasks T1 and T2 are projected
at 43 s and 12.1 min on a 32-core CPU. H2 is not supported for them. Phase 2 does not revisit that verdict.

| Kept from Phase 1, unchanged | How Phase 2 treats it |
|---|---|
| H2 verdict for T1/T2, the 1-hour line, the stored result files (including the F4 label) | Preserved. Nothing in Phase 2 re-judges H2 |
| Target quantity: conditional detection rate p<sub>π</sub>(L, q \| x) on a fixed series | Unchanged. Not a damage bound |
| T1, T2 definitions (n = 2,449 and 11,726 trials per cell, h = 0.02, Clopper–Pearson) | Unchanged; they remain reference tasks |
| Identity contract (feature bits, integer counts, tie rules, greedy merge) | Unchanged; extended to the GPU without tolerance |
| CPU baseline = fastest verified exact method (incremental) | Kept; re-measured at 32 and 64 threads for P1, P24, P105 |

Phase 2 does **not assume a GPU advantage**. A CPU win, a tie, and a GPU win are all reportable outcomes with the same
evidence requirements. Tasks and accuracy targets are not enlarged after results are seen.

## Questions

- **RQ1 (prerequisite).** Do T1 and T2 actually complete end to end — input read to interval file on disk — within
  the requested interval widths, and how long do they take cold and warm? (Phase 1 projected these times but did not
  run a complete task.)
- **RQ2 (main).** At the same estimand, trial stream, estimator, and bit-identical output, which of the fastest exact
  CPU configuration and a GPU implementation reaches the result sooner end to end, and under which workload
  conditions? Can a cost model with inputs known before execution predict per-trial cost on held-out conditions?
- **RQ3 (bounded extension).** On the full 1-second week, how long does building an exact reference for selected
  cells take on each device, and at a fixed wall-clock budget, how far is each device's estimate from that reference?

Out of scope: variance-aware or adaptive allocation, new detectors, multiple assets or periods, account- vs
group-level policy comparison (H1), damage bounds. These belong to later phases.

## Target quantity, accuracy, and exactness

- **Estimand**: as in Phase 1. A trial draws a start for one cell (L, q) and evaluates every policy of the task.
- **Implementation exactness**: for the same (series, policy, event), the GPU's episode list equals the CPU
  reference in every field (k, start, window, count, and the bit pattern of S), and so do the detection and overlap
  bits. There is no tolerance. This is a finite differential check, not a proof ([`SPEC.md` §5](SPEC.md#5-correctness-gate-c)).
- **Statistical accuracy**: fixed-n Clopper–Pearson as in Phase 1 (T1 pointwise α = 0.05; T2 Bonferroni
  α = 0.05/3,264). Half-width means (upper − lower)/2. For fixed-n tasks the CPU and GPU evaluate the same trial ids,
  so their outputs are identical and H3 is a pure time comparison.
- **Exact reference** (RQ3): the per-start detection bits from evaluating every valid start of a cell. "Exact" means
  the exact average over the finite population of starts as the implementation defines it; its correctness rests
  on the gate and the oracle check below, not on a proof.

## Tasks

| Task | Definition | Role |
|---|---|---|
| T1-curve | Phase 1 T1, on the 1-minute quarter and the 1-second week | Reference task (R1, R2, H3) |
| T2-policy-map | Phase 1 T2, on both series | Reference task (R1, R2, H3) |
| T-ref | 1-second week, P24, every valid start of 15 cells: L ∈ {8, 32, 128} × q ∈ {16, 32, 64, 128, 256}; 9,071,175 trials | Cost of building an exact reference (H3); produces the reference for H4 |
| T-ref-est | The same 360 rates (15 cells × 24 policies) estimated by sampling within a wall-clock budget B ∈ {10, 30, 100, 300} s | H4 |
| T-diag | Per-trial cost over controlled factors (below) | D1 |

No new feasibility line (such as one hour) is set for T-ref; it is judged only by the H3 rule. T-ref is a
reference-building task done once per study, not a recurring analysis, and its result is reported as such. T1 and T2
are always reported next to it. Two larger tasks named during design (all 136 cells of the week enumerated; several
independent series per trial) are **not part of this protocol**; each would need its own registered criteria.

**T-diag factors** (steady-state per-trial cost; each line varies one factor around the canonical point
L = 32, q = 16, P24; exact policy lists in `protocol.json`):

| Factor | Levels |
|---|---|
| Tail cap s<sub>floor</sub> | 1.5, 2.0, 3.0 (3 policies: K = 15, gaps 30/60/120, s<sub>min</sub> = s<sub>floor</sub>) |
| Policy count | 1, 3, 24 at s<sub>floor</sub> = 1.5, K<sub>max</sub> = 15 |
| K<sub>max</sub> | 8, 15 (one policy, s<sub>min</sub> = 2, gap 60) |
| Event length L | 2, 32, 256 (calibration); 8, 128 (held out) |
| Strength q | 0, 16, 256 |
| Series length N | nested prefixes of the 1-second week: 75,600, 151,200, 302,400 (calibration); 604,800 (held out) |
| GPU batch size | 64, 256, 1,024, and the largest that fits (b<sub>max</sub>); crossed with s<sub>floor</sub> {1.5, 3.0} and L {2, 256} at N = 302,400 |

Nested prefixes of one week vary N without changing the data source. The 1-minute quarter is a different series
(bar size, period, distribution) and is reported as a cross-series check, not as an N level.

## Implementations compared

**CPU-best** is the fastest exact CPU configuration *among those examined*: currently incremental, across-trial
parallel, on AWS `c7i.16xlarge`. It is not a claim about the best possible CPU algorithm.

- **Threads**: per (series, policy grid), 32 or 64 threads, chosen in a dedicated calibration and never from the
  final measurement runs. Phase 1 measured the 64-thread (SMT) gain only for one policy at L = 30, so the choice is
  not carried over between grids or lengths. Short calibration: q = 16, L ∈ {2, 32, 256}, 3 × 5 s each; the
  candidate with the higher geometric mean of the three median throughputs wins. Sustained check: both thread
  counts run once for 60 s at L = 32; if the sustained order disagrees with the short one, the sustained winner is
  used and the flip is recorded. P1 and P24 set task threads; P105 is calibrated the same way and reported only.
  Whether the choice flips across L or grids is reported. Repeated before the freeze if CPU-best changes. The GPU
  batch size is chosen with the same design ([`SPEC.md` §6](SPEC.md#6-gpu-design-constraints-and-memory-plan)).
- **Parity**: every device-independent improvement considered for the GPU (lookup tables for event factors, S, and
  caps; sparse count representation; scratch reuse; batch scheduling; anything found later) is also tried on the
  CPU. It is adopted if it passes the CPU gate and raises the geometric-mean throughput of three calibration points
  by ≥ 2%; otherwise the measured reason for not adopting it is recorded. The adopted set defines CPU-best at the
  freeze. Both sides' tuning attempts are logged.
- **CPU-host** (descriptive): CPU-best on the GPU instance's own host, with its thread count chosen the same way
  among {physical cores, all vCPUs} of that host; never with the `c7i` thread count.

**GPU**: two candidate designs are prototyped; one is carried forward.

| Path | Design |
|---|---|
| G-inc | Batched incremental on the device: base orderings resident, changed windows recomputed, tails merged, peaks, ordering, and per-policy merge on the GPU |
| G-hyb | GPU recomputes changed windows, merges tails, and extracts peak candidates; the host orders candidates and runs the per-policy merge |

Selection rule: on the L40S, at week / L = 32 / q = 16 / P24, the path with the higher steady-state trials per second
(transfers and synchronisation included, each at its best batch size) is chosen; within 10% of each other, G-inc is
chosen. G-inc runs on `g6e.2xlarge`, G-hyb on `g6e.4xlarge` (the larger host does the selection work). Comparisons are
always against the whole GPU instance (GPU + host).

## Hypotheses

### Confirmatory (verdict rules fixed below)

| ID | Statement | Pass / verdict |
|---|---|---|
| R1 | The CPU runner completes T1 and T2 on both series end to end, cold and warm, with every interval at half-width ≤ 0.02 | Every interval's half-width ≤ h; counts recomputed from stored per-trial bits match; 0 mismatches on spot verification; on both Stage A launches |
| R2 | The Phase 1 projection method, re-applied with the Phase 2 binary and thread count, predicts warm end-to-end time | Per task × series: \|median warm E2E / re-projection − 1\| ≤ 0.15 |
| C | The GPU implementation is bit-identical to the CPU reference over the full gate | 0 mismatches in every gate item |
| **H3** | The GPU reaches the same result in less end-to-end time | Category per condition (§ Verdict rules) |
| **H4** | At equal wall-clock budget, the GPU's estimate is closer to the exact reference | Category per budget (§ Verdict rules) |
| D1 | A per-device cost model with pre-execution inputs predicts per-trial cost on held-out conditions | Per device: median \|relative error\| ≤ 0.20 and maximum ≤ 0.50 over the 13 held-out points |

H3 and H4 are phrased in the GPU's favour because that is the project hypothesis, but their verdicts are
symmetric: GPU advantage, CPU advantage, equivalence, and inconclusive are all outcomes, none of them a failure.

### Exploratory (reported, no verdict)

| ID | Content |
|---|---|
| D2 | End-to-end winner predicted from the D1 models plus measured fixed costs, against the H3/H4 categories; with the trivial predictors "always CPU" and "always GPU", the number of inconclusive conditions, and the full denominator |
| D3 | Transfer of the L40S cost model to a second, different GPU (the desktop RTX 5060 Ti, consumer Blackwell) with one scale factor fitted on 3 pre-set points, predicting T2 week and the other T-diag points there |
| X1 | Diagnosis of the non-monotone rate dips (Phase 1 prefix, L = 120, q 64 → 128): at which step (global rank, local peak, gap exclusion, top-20 cut, base-episode membership) each traced flip happens; then whether the full-week reference shows dips |
| X2 | CPU instance-size control: CPU-best on one `c7i.8xlarge` launch (16 cores / 32 vCPU / 64 GiB), threads chosen among {16, 32} by the same calibration, T1 and T2 on both series cold (3 runs each); end-to-end time, peak memory, and cost per task next to `c7i.16xlarge` and the GPU instance. Keeps the cost comparison from resting on one CPU size only; it does not change CPU-best or any verdict |
| — | Select-stage breakdown (peak test, candidate collection, S and sort, per-policy merge, reset) and counts, with hardware counters (`perf stat`) from separate profiling runs where the VM exposes them (unavailability recorded); a bottleneck is named only with that evidence; same-box CPU-host; cost per task in USD (basis below); warm-mode H4; amortised time over several tasks per prepared base; GPU power from `nvidia-smi`; fixed-n replay projection of H4 from measured throughput |

Anything added after results are seen (for example extra transition cells, at most 6) is labelled post hoc.

## Verdict rules

### Unit and repetitions

The **independent unit is an instance launch**. Runs inside a launch share host, VM, and placement, so they are
technical repetitions. Stage D uses **m = 3 launch pairs** (one CPU and one GPU launch each, same region, same
calendar day, alternating which goes first). Inside each launch every condition runs in one seeded shuffled order.

- T1, T2: 3 cold and 3 warm runs per launch; the launch value is the median of the 3.
- T-ref: 1 cold run per launch (about an hour on the CPU; within-run noise averages out).
- T-ref-est: 10 seeds per launch pair and budget, the same seeds on both devices.

**Choice of m.** With m = 3, a 90% interval of a mean log-ratio has half-width 2.92·√2·σ/√3 ≈ 2.38σ (σ = SD of log
launch medians per device, assumed similar on both devices). To keep it within log(1.25)/2 ≈ 0.11, σ must be
≤ 0.047. From the two Stage A CPU launches, σ is estimated from T2 week cold medians. If it exceeds 0.047, m = 5
(half-width ≈ 1.35σ). This decision uses CPU data only, before any GPU timing, and the resource plan covers m = 5.

### H3: time to the same result

For condition c and launch pair i, ℓ<sub>i</sub> = log(T<sub>CPU,i</sub> / T<sub>GPU,i</sub>) with T the launch value
above. ρ = exp(mean ℓ), and its 90% interval is exp(mean ℓ ± t<sub>0.95, m−1</sub> · SD(ℓ)/√m). The paired t-interval
assumes approximately normal launch log-ratios; with m = 3 this is a weak check, so per-launch ratios are always shown.

Equivalence margin **Δ = 1.25**, fixed now and not derived from measured variance:

| Category | 90% interval [lo, hi] of ρ |
|---|---|
| GPU advantage | lo ≥ 1.25 |
| CPU advantage | hi ≤ 0.80 |
| Equivalent | 0.80 < lo and hi < 1.25 |
| Inconclusive | otherwise. If the interval excludes 1 it is annotated "direction X, size unresolved" |

A difference smaller than 25% in time to a result that takes seconds to an hour does not change which device a user
should pick. Δ is close to the 15–20% scale of the Phase 1 projection criteria, and it is chosen before any Phase 2
timing.

- **Primary conditions** (cold): T1 quarter, T1 week, T2 quarter, T2 week, T-ref week. **Secondary** (warm): T1 and
  T2 on both series. Each condition gets its own verdict. No pooled claim ("GPU wins most tasks") is made; any
  cross-condition summary is descriptive and lists the conditions.
- A run must reproduce the reference output hash (T1/T2: all (k, n) and intervals; T-ref: all hit bits);
  otherwise it is not a valid timing.
- **Timeouts**: a GPU or CPU run is stopped at 3 × the CPU-best projected time of its condition (fixed at the
  freeze). A stopped run counts as a time of exactly the timeout. This lower bound can only make the other device
  look slower than it is, so a CPU advantage may still be concluded from such runs; a GPU advantage may not.

### H4: error at equal wall-clock

Each live run is a **cold process with a deadline** B (setup, aggregation, interval computation, and fsync of the
output all inside B, measured by the harness from spawn to exit). Trials are ordered globally round-robin over the
15 cells (g = j·15 + c); only the **contiguous prefix** of completed trials counts, which gives N<sub>c</sub>(B) per
cell. The estimate is k<sub>c</sub>/N<sub>c</sub>, compared with the exact rate.

- Per run: MSE = mean over 360 rates of (p̂ − p)². Per device and launch pair: RMSE = √(mean of MSE over its 10 seeds).
- ε<sub>i</sub> = RMSE<sub>CPU,i</sub> / RMSE<sub>GPU,i</sub> (> 1 means the GPU is more accurate). The interval and
  categories are exactly as for H3, with ε in place of ρ and the same Δ = 1.25. In samples, Δ = 1.25 in RMSE
  corresponds to about 1.56x as many trials, which is deliberately stricter than the time margin.
- If both RMSEs are 0 the verdict is "equal (both exact)"; if one is 0, the result is reported without an interval.
- Not reaching anything is a result too: a cell with N<sub>c</sub> = 0 is estimated as 1/2; a **late run** (exit
  after B, or no valid output) counts as N<sub>c</sub> = 0 for all cells. No run or cell is dropped.
- Secondary metrics: maximum absolute error, RMSE over transition rates (exact p ∈ [0.05, 0.95]; "undefined" if the
  set is empty, never imputed), N<sub>c</sub>(B) distribution, late-run count, checkpoint reached, mean interval
  half-width at the reached checkpoint (unreached cells counted as half-width 0.5).
- **Intervals under a time budget**: intervals are reported only at fixed checkpoints n<sub>j</sub> = 2<sup>8</sup>,
  2<sup>9</sup>, …, 2<sup>18</sup> (J = 11), with α/(360·11) per interval (union bound over rates and checkpoints),
  at the largest checkpoint reached. The guarantee comes from that construction; coverage observed over the live runs
  is a sanity check, not a demonstration.
- **Replay** of stored exact bits at the measured N(B) is reported separately as a fixed-n projection. It is not the
  H4 measurement: completed sample counts may depend on which starts were drawn, and replay does not preserve that.
- Every live run's (N<sub>c</sub>, k<sub>c</sub>) must equal what the exact reference bits give for the same trial ids.

Interpretation fixed in advance: because both devices use the same estimator, H4 mostly follows throughput. What
it adds beyond H3 is the effect of setup, deadlines, and the stopping rule on error, most visible at small B.

### Status of every registered item

Each confirmatory item ends in exactly one status, all of them reported:

| Status | Meaning |
|---|---|
| Judged | Verdict or category as above |
| Not measured | A prerequisite failed (for example C or the reference oracle); the reason and the failing stage are recorded |
| Incomplete | Fewer than m complete launch pairs, a time box, or the budget stopped the work. Raw data of incomplete launches is kept but not used in verdicts |

A whole launch is the smallest unit dropped when work stops. Every launch runs all conditions, so stopping cannot
favour quick conditions. "Not measured" and "incomplete" are never reported as "CPU advantage" or "equivalent".

A comparison is **invalid** (not a result) if the trial stream, estimator, or measurement boundary differ between the
devices, if parity was skipped without a recorded reason, if a non-identical output is timed, or if tasks, accuracy,
Δ, budgets, or cells were changed after results were seen.

## Measurement

| Stage | Content | Cold | Warm |
|---|---|---|---|
| S0 | Process start, argument parsing | ✓ | — |
| S1 | Read inputs, verify SHA-256 | ✓ | — |
| S2 | Device init: thread pool / CUDA context, allocation, first touch | ✓ | — |
| S3 | Base state: scan, ranks, heads, base episodes (per base-state key) | ✓ | — (resident) |
| S4 | Per-length preparation: median range, factor / S / cap tables, uploads | ✓ | ✓ |
| S5 | Trial loop (batch transfers included) | ✓ | ✓ |
| S6–S8 | Aggregation per cell and policy; intervals and width check; write and fsync | ✓ | ✓ |

- **End-to-end time** is measured by the external harness: from just before spawn (cold) or task submission (warm)
  to process exit or the result acknowledgement after fsync. Stage timestamps inside the runner give the breakdown.
- **Cold**: new process, page cache dropped, GPU persistence mode on, precompiled device code (no JIT). **Warm**:
  the same persistent process already ran a task with the same base-state key (input hash, window ladder, policy
  set, s<sub>floor</sub>, maximum event length). **Boot-cold** (first run after instance start) is recorded once,
  descriptively.
- Profiling runs that serialise the pipeline, or that read hardware counters, are kept apart from timed runs and
  never used for verdicts.
- Resource records per run, taken by the harness outside the runner's critical path: process-tree peak memory
  (cgroup v2 `memory.peak`), minimum `MemAvailable`, swap-in/out and major faults, CPU steal time, 1 Hz CPU clock
  samples, GPU clocks and memory, and thread affinity at loop start and end. They describe the runs; they are not
  a minimum-memory test.
- Data download and build are excluded and recorded separately.

## Stages and dependencies

| Stage | Work | Exit condition | If not met |
|---|---|---|---|
| A. CPU runner | Runner, trial stream, intervals, harness and Phase 1 regression built and checked on the desktop; on the reference CPU: thread calibration, R1, R2 on 2 launches, σ for m | R1 passes | No speed verdicts until fixed; GPU development may continue |
| B. CPU diagnostics and reference (desktop) | Select-stage breakdown, exploratory T-diag on CPU, parity candidates implemented, T-ref reference built with CPU-best and checked by the oracle, X1 trace | Oracle: 0 mismatches | T-ref H3 and H4 "not measured" until fixed |
| C. GPU feasibility | Prototypes of G-inc and G-hyb and the partial gate (week, L = 32, q = 16, P24 + all-start enumeration at N ≤ 4,096) on the desktop GPU; memory plan; path selection on the L40S | Chosen path passes the partial gate on the L40S and its working set fits at batch ≥ 64 there | GPU path "not validated": H3, H4, the GPU side of D1, D2, D3 "not measured"; CPU results (R1, R2, CPU D1, reference, X1) are reported |
| Pre-freeze calibration | On the measurement machines only: parity adoption and thread counts (reference CPU); batch sizes and deadline margins (both); timeouts from the CPU projections | All values recorded | — |
| Freeze | CPU-best and GPU commits and the values above → `protocol-freeze.json` | Committed | — |
| D0. Full gate | Gate C on the frozen GPU build over every reported task range | 0 mismatches | Correctness fixes allowed, then re-freeze and the full gate again; no timing counts before it passes |
| D. Comparison | m launch pairs with all conditions; D1 on pair 1; D3 on the desktop with the frozen code; X2 on one `c7i.8xlarge` launch with the frozen CPU build | — | Status rules above |

**Oracle for the reference (fixed now).** Per T-ref cell, 2,000 starts drawn from a separate stream are re-evaluated
with the *shared* method (one scan and full-sort ranking, independent of the incremental tail merge) and 200 of them
also with *full* recomputation. All episode fields and hit bits must match. Afterwards the GPU's own T-ref enumeration
must reproduce the reference bits exactly (part of D0 and of every T-ref timing run).

**Stage C engineering rule.** If the chosen prototype is projected to finish at least one primary condition ≥ 1.5x
faster than CPU-best (same projection method on both sides), the GPU path is optimised further before the freeze.
Otherwise the prototype is frozen as it is. **Either way Stage D runs the same complete protocol** on the frozen
build, including the full gate, all primary conditions, H4, and D1. The 1.5x rule only decides how much optimisation
effort goes in; results are labelled "optimised" or "prototype" accordingly.

## Execution environment

Each step runs on the cheapest machine that does not weaken what it is used for. Only the machines that produce
verdict timings must be controlled, launchable more than once, and identical to what others can rent; every other
step depends only on bits (correctness, the reference), on development, or on exploratory timings.

| Machine | Used for | Not used for |
|---|---|---|
| **Reference CPU**: AWS `c7i.16xlarge` (32 cores / 64 vCPU, as Phase 1), us-east-1, Ubuntu 24.04, built on the instance, `-O3 -march=native`, `OMP_PLACES=cores OMP_PROC_BIND=spread`; spot capacity, on-demand after two interruptions | All CPU timings with a verdict (R1, R2, H3, H4, D1), thread calibration, parity adoption, CPU deadline margin | — |
| **Measurement GPU**: NVIDIA L40S, AWS `g6e.2xlarge` (G-inc) or `g6e.4xlarge` (G-hyb), on-demand, us-east-1 | Path selection, full gate D0, all GPU timings with a verdict, GPU batch size and deadline margin, CPU-host | Development |
| **Desktop**: Intel Core Ultra 5 250K Plus (6 P + 12 E cores), 32 GB, NVIDIA RTX 5060 Ti 8 GB (Blackwell), Windows 11 + WSL2 Ubuntu 24.04 | Development of all code; correctness tests; Phase 1 regression; T-ref reference build and oracle; X1; exploratory CPU diagnostics; GPU prototypes and partial gate; D3 | Any verdict timing |
| **CPU size control**: AWS `c7i.8xlarge` (16 cores / 32 vCPU / 64 GiB), same build and settings as the reference CPU, spot, one launch | X2 only | Any verdict; CPU-best |
| **Colab** (free T4, optional) | An extra exactness check of the frozen GPU code on a third architecture (sm_75); skipped without consequence if no GPU is available | Any timing |

- **Hardware identity.** A verdict launch must report the same CPU model as Phase 1 (`c7i`: Xeon Platinum 8488C,
  32 cores / 64 threads, one NUMA node) or an L40S (`g6e`). A launch that does not is kept, labelled
  "hardware mismatch", excluded from verdicts, and replaced by a new launch.
- **Cost basis.** Cost per task is reported for every machine at the official AWS on-demand price for us-east-1 Linux
  on the run date (price-list SKU and file hash recorded). Spot launches additionally report the spot price of their
  availability zone at launch and the instance-hours lost to interruptions. Instance-hours come from launch and
  termination times, not from summed run timers. Bills, where available, are reported separately; storage and
  transfer are listed apart from compute.
- Spot and on-demand capacity run the same instance type; the purchase option is recorded per launch. An
  interrupted launch is incomplete and is replaced by a new launch, never resumed.
- Desktop and Colab timings are never results, except D3, which is exploratory and labelled with its machine. The
  desktop is a shared personal machine without GPU persistence mode, with a display attached, and with mixed core
  types, which is why it does not produce verdict timings.
- Bits built on the desktop are re-checked on the reference CPU at no extra cost: each Stage D CPU launch's gate step
  recomputes the gate golden outputs, and every T-ref timing run there regenerates all reference bits; hashes must
  match.
- Toolchain versions (GCC, CUDA, driver) are pinned in `protocol-freeze.json` and recorded per run. Comparing GPU
  generations is not a goal; results are not stated as hardware laws. Energy is not measured, because CPU power is
  not observable on these VMs.

## What is fixed when

| Fixed now (this commit) | Fixed by a rule stated here, committed before the stage that uses it | Free during implementation |
|---|---|---|
| Questions, tasks, cells, budgets, checkpoints; hypotheses and verdict rules (Δ, interval method, m rule, status labels, hardware identity); thread and batch calibration design; cost basis; resource records; stage boundaries, cold/warm/deadline definitions; oracle; trial stream v1; numerical contract; gate items; D1 model form, calibration and held-out split; Stage C selection and 1.5x rules; seeds | Thread counts (Stage A, pre-freeze); m (Stage A, CPU only); CPU-best composition (parity rule); GPU path (Stage C); batch sizes, deadline drain margins, timeouts (pre-freeze calibration) | Code structure, kernels, launch configuration, file formats, scripts (until the freeze); correctness fixes at any time, followed by the full gate and re-runs of any affected timing |

Changing anything in the first column after this commit requires a dated amendment in this file, made before the
affected measurement, stating the reason. Amendments made after seeing results make the affected comparison
exploratory.

## What this phase can and cannot claim

- It can show, for this detector, these tasks, and these two machines, whether the GPU implementation reaches the
  same exact result sooner, by how much, at what cost, and with which per-trial cost model.
- It does not claim a cost-optimal or minimum-memory configuration: X2 covers one smaller CPU size, and the memory
  records are observations at full instance memory, not runs under a limit. Stage-time slowdowns are not attributed
  to cache, bandwidth, or clock without the counter evidence above.
- It cannot show that GPUs in general are necessary or unnecessary, a lower bound on CPU cost, behaviour on other
  assets, periods, or detectors, or anything about damage bounds or real surveillance effectiveness. More trials
  reduce sampling error for this fixed series and synthetic event only.
- A loss or a missing GPU path says something about the implementation that was built within the stated effort, not
  about what a GPU can do.
- The incremental method's speed-up over full recomputation, the Phase 0 GPU scan speed-up, Clopper–Pearson
  intervals, and the counter-based random stream are earlier results or standard methods, not results of this phase.

## Files

| File | Content |
|---|---|
| [`README.md`](README.md) | This protocol |
| [`SPEC.md`](SPEC.md) | Implementation specification: runner, trial stream, numerical contract, gate, memory, harness, manifest, analysis |
| [`protocol.json`](protocol.json) | Frozen parameters (tasks, cells, policy lists, budgets, checkpoints, margins, seeds, model form) |

Scripts and result folders are added during implementation. Results go to `results/<env>_<date>/launch-<i>/`, as in
Phase 1.

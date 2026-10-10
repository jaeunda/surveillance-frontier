# Phase 2 — GPU engine: what limits the exact computation on the machines at hand, and does the matching resource help?

Status: **design r2, not started** (2026-10-10). No Phase 2 measurement has been run. This file is the protocol;
[`SPEC.md`](SPEC.md) is the implementation specification, [`protocol.json`](protocol.json) holds every parameter, and
[`IMPLEMENTATION.md`](IMPLEMENTATION.md) records how the code carries it out.

**Revision r2 (2026-10-10, before any measurement).** The first design (r1, commit `6f4aad9`) fixed the comparison
machines in advance (AWS `c7i.16xlarge` and an L40S) and gave the locally available machines only development and
correctness work. r2 reverses the order of evidence: the work is first run on a machine at hand, its limits are
measured, and a resource is added only for an observed reason and checked against that reason. The estimand, the
tasks and their accuracy, the exactness contract, the trial stream, the estimator, and the verdict rules of H3/H4 are
unchanged from r1. Because nothing had been measured, r2 replaces r1 as a whole rather than amending it.

## Starting point

Phase 1 ([write-up](../../docs/phase1.md)) changed the baseline: the incremental method runs 349x more trials per second
than full recomputation (week, K = 15, one policy, one thread), and the stress-testing tasks T1 and T2 are projected
at 43 s and 12.1 min on a 32-core CPU. H2 is not supported for them. Phase 2 does not revisit that verdict.

| Kept from Phase 1, unchanged | How Phase 2 treats it |
|---|---|
| H2 verdict for T1/T2, the 1-hour line, the stored result files (including the F4 label) | Preserved. Nothing in Phase 2 re-judges H2, and the 1-hour line is not reused as a reason to rent hardware |
| Target quantity: conditional detection rate p<sub>π</sub>(L, q \| x) on a fixed series | Unchanged. Not a damage bound |
| T1, T2 definitions (n = 2,449 and 11,726 trials per cell, h = 0.02, Clopper–Pearson) | Unchanged; they remain reference tasks |
| Identity contract (feature bits, integer counts, tie rules, greedy merge) | Unchanged; extended to the GPU without tolerance |
| CPU baseline = fastest verified exact method (incremental) | Kept; its configuration is chosen anew on every machine |

Phase 2 does **not assume a GPU advantage, nor that more hardware is needed.** A CPU win, a tie, a GPU win, and "no
scaling was justified" are all reportable outcomes. Tasks and accuracy targets are not enlarged after results are seen,
and **scaling the hardware is kept apart from scaling the problem**: every scaling step first re-runs the same work.

## Questions

- **RQ0 (constraints).** At the registered accuracy, how long do T1 and T2 take end to end on an available machine,
  how fast is each GPU path there, and what limits the work: time, CPU, host, device memory, or the controllability of
  the environment?
- **RQ1 (scaling).** If a resource is added because of such an observation, does the same work improve as predicted
  (time to the same result, rate, completion), and where does it not?
- **RQ2 (comparison).** At the same estimand, trial stream, estimator, and bit-identical output, which of the fastest
  exact CPU configuration and a GPU implementation reaches the result sooner end to end, on the machines chosen by
  RQ1, and under which workload conditions? Can a per-device cost model with pre-execution inputs predict per-trial cost
  on held-out conditions?
- **RQ3 (bounded extension).** On the full 1-second week, how long does building an exact reference for selected
  cells take on each device, and at a fixed wall-clock budget, how far is each device's estimate from that reference?

Out of scope: variance-aware or adaptive allocation, new detectors, multiple assets or periods, account- vs
group-level policy comparison (H1), damage bounds.

## Target quantity, accuracy, and exactness

- **Estimand**: as in Phase 1. A trial draws a start for one cell (L, q) and evaluates every policy of the task.
- **Implementation exactness**: for the same (series, policy, event), the GPU's episode list equals the CPU
  reference in every field (k, start, window, count, and the bit pattern of S), and so do the detection and overlap
  bits. No tolerance. This is a finite differential check, not a proof ([`SPEC.md` §5](SPEC.md#5-correctness-gate-c)).
- **Statistical accuracy**: fixed-n Clopper–Pearson as in Phase 1 (T1 pointwise α = 0.05; T2 Bonferroni
  α = 0.05/3,264). For fixed-n tasks every device evaluates the same trial ids, so outputs are identical and every
  comparison of them is a pure time comparison. **A faster machine does not make a fixed-n result more accurate**;
  accuracy at a fixed time is H4's question only.
- **Exact reference** (RQ3): the per-start detection bits from evaluating every valid start of a cell; its correctness
  rests on the gate and the oracle below.

## Tasks

| Task | Definition | Role |
|---|---|---|
| T1-curve | Phase 1 T1, on the 1-minute quarter and the 1-second week | Pilot baseline; formal R1, R2, H3 |
| T2-policy-map | Phase 1 T2, on both series | Pilot baseline; formal R1, R2, H3 |
| T-ref | 1-second week, P24, every valid start of 15 cells: L ∈ {8, 32, 128} × q ∈ {16, 32, 64, 128, 256}; n<sub>c</sub> = N − L + 1, 9,071,175 trials | Reference-building cost (H3); the reference for H4 |
| T-ref-est | The same 360 rates estimated by sampling within a wall-clock budget B ∈ {10, 30, 100, 300} s | H4 |
| T-diag | Per-trial cost over controlled factors (below) | D1; pilot diagnosis on the calibration split only |

T-ref is built once per study; its cost is reported as such, never as a recurring workload. No feasibility line is
set for it. **H4 needs the full-week reference; a reference on a prefix of the week never stands in for it.** Larger
tasks (all 136 week cells, several series) are not part of this protocol.

**T-diag factors** (steady-state per-trial cost; each line varies one factor around L = 32, q = 16, P24; exact
policy lists in `protocol.json`): tail cap s<sub>floor</sub> 1.5, 2.0, 3.0; policy count 1, 3, 24; K<sub>max</sub> 8, 15;
L 2, 32, 256 (calibration) and 8, 128 (held out); q 0, 16, 256; series length N as nested prefixes of the week
75,600, 151,200, 302,400 (calibration) and 604,800 (held out); GPU batch 64, 256, 1,024, b<sub>max</sub>, crossed with
s<sub>floor</sub> {1.5, 3.0} × L {2, 256} at N = 302,400. The 13 held-out points are never run before the formal D1.

## Implementations

**CPU-best** is the fastest exact CPU configuration *examined on a given machine*: incremental, across-trial parallel,
with that machine's thread choice and the adopted parity items. A choice made on one machine is never carried to
another: thread counts, pinning, and parity adoption are re-established on each machine by the same rule.

**GPU**: two designs, G-inc (batched incremental on the device, merge on the device) and G-hyb (device recomputes
changed windows and extracts peak candidates; the host orders and merges). Both are exercised in the pilot; the formal
path is chosen on the formal GPU by the path rule (Part III), whatever the pilot preferred. Comparisons are always
against the whole GPU machine (GPU + host), and a GPU result is always reported next to the same box's CPU.

**Parity**: every device-independent improvement considered for the GPU (lookup tables for event factors, S, and caps;
sparse count representation; scratch reuse; batch scheduling) is also tried on the CPU, and adopted there if it passes
the CPU gate and raises the geometric-mean throughput of three calibration points by ≥ 2% ([`SPEC.md` §7](SPEC.md#7-cpu-parity)).

---

## Part I. Pilot on an available machine (P0–P2)

| Machine | Profile | What is recorded every session |
|---|---|---|
| Lab PC | `lab-desktop`: Core Ultra 5 250K Plus (hybrid P/E cores, 18 logical CPUs), 32 GiB host RAM of which WSL2 sees about 15 GiB, RTX 5060 Ti 8 GB (sm_120), Windows 11 + WSL2 | Free VRAM and other GPU processes, MemAvailable, swap, driver and toolkit, P/E mapping |
| Google Colab | `colab`: whatever the session is assigned | GPU model, host CPU, RAM, session start; **each session is its own hardware**, never "Colab" in general |

One available machine suffices; using both is not required. The lab PC is used for execution and measurement only.

| Step | Work | Passes on |
|---|---|---|
| **P0 plan and feasibility** | `pilot-plan.json` written and committed before the first timing: machine, code, binary, input, and task hashes, steps, repetitions, stop rules, available hours. CPU regression (`sf_test`, `sf_test2`, Phase 1 hit files), stream and interval checks, CPU gate; **the partial gate on the device itself** for each GPU path | Which GPU paths passed on real hardware; emulation and a successful build never count as device execution |
| **P1 baseline** | `cpu-baseline`: T1 and T2 on both series, cold and warm, CPU on every logical CPU. `gpu-rates`: G-inc and G-hyb at a small and the representative point (week, L = 32, q = 16, P24), microbatch 64 and b<sub>max</sub>. `gpu-tasks`: T1 and T2 cold and warm on every path that passed P0 | E2E with stage times, output identity with the CPU, failures and their conditions |
| **P2 diagnosis** | `threads`: candidates from the machine's topology (P-core set, physical cores, all logical CPUs). `microbatch`: sweep up to b<sub>max</sub> with memory records. `rates`: steady-state rates at every task length. `diag`: T-diag calibration split. `tref-rates`: every T-ref cell. `tref-build`: the reference and its oracle if the hours allow. `deadline`: H4 deadline mechanics on a 75,600-bar prefix | Throughput curves over threads and microbatch, RAM/VRAM peaks, host vs device time, projections of everything still to run |

**Repetitions.** Two sessions at different times of day; per session, T1/T2 cold 3 and warm 3. Warm repetitions of a
condition run as one block after one untimed priming task. A smaller plan is allowed only if recorded before the first
measurement (`--cut`); later changes are dated amendments made before the run they affect. Every run is kept,
including slow, failed, late, out-of-memory, and retried ones. Each session reports its own median and range;
repetitions on one PC are not independent launches, and no interval about hardware is computed from them.

**Measurement in the pilot.** The harness times from just before spawn to process exit after the output fsync, with
the runner's S0–S8 stamps for the breakdown (initialisation, transfer, loop, output). If the page cache cannot be dropped
the run is **process-cold** and is aggregated apart from **cache-cold** runs. Missing controls (cgroup memory peak,
GPU persistence mode, which WSL2 does not support) are recorded with the substitute measure. Results are written to the
machine's local disk; copies to Drive or elsewhere happen after the timed region.

**What the pilot is not.** Pilot times are results about that machine and are reported as such. They are never pooled
with formal launches, and they never set m, Δ, tasks, accuracy targets, or cells. A GPU path that loses in the pilot is
not dropped from the formal comparison on that basis, and the pilot never runs the 13 held-out T-diag points.

## Part II. Scaling decision (P3) and verification (P4)

**P3: one record before any paid run.** `results/scaling/scaling-decision.json` is drafted from the pilot summaries
(observations, the catalogue of measured metrics, projections) and completed by hand with six items:

1. **Required outputs and schedule** — which T1/T2, reference, and repeated measurements must finish by when. The
   research schedule is not the H4 budget B.
2. **Observations** — E2E, stage times, rates, peak memory, failure rates, session-to-session variation, raw logs.
3. **Alternatives without scaling** — what was tried within the machine (threads and pinning, parity, scratch reuse,
   microbatch) and over which range.
4. **Needed resource and expected effect** — which constraint is relieved by how much; conservative, base and
   optimistic scenarios; projections outside the observed range are marked.
5. **Falsifying measurement** — the same-work comparison, with a gain threshold below which the hypothesis is withdrawn.
   "Runs on the larger machine" is not an improvement in speed or accuracy.
6. **Cost and stop conditions** — instance-hours including gates, calibration, repetitions, failure reserve, storage,
   and shutdown; a spending and development-time limit kept regardless of the outcome.

The action is one of: **none** (report the pilot; formal H3/H4 "not measured (no scaling)"), **environment**
(a controllable machine of a similar class: repeatable launches, controlled cold state), **cpu-host** (a larger CPU or
GPU host), **gpu** (a different or larger GPU). Projections use

- time: T ≈ T<sub>setup</sub> + Σ<sub>c</sub> n<sub>c</sub> / r<sub>c</sub> + T<sub>output</sub>, with r<sub>c</sub>
  measured on that machine, code, grid, length, and strength (T-ref: n<sub>c</sub> = N − L + 1, 9,071,175 trials);
- memory: M<sub>shared</sub> + b · M<sub>trial</sub> against the allocation the device reported, with free memory read
  after context creation and before the code's own allocations (so shared state is subtracted once), and RAM, VRAM,
  and pinned memory kept apart. Neither a data-sheet capacity nor an old snapshot sets b<sub>max</sub>.

| Pilot observation | Check first | Follow-up it can justify |
|---|---|---|
| Large microbatches do not fit, and throughput still rises over the batches that do | Allocation accounting, free VRAM, duplicated buffers, resident vs submitted trials | Same work at the local batches and the new larger ones on a larger-memory GPU. No gain withdraws the capacity hypothesis |
| Device time dominates, host wait and transfers are small, and the needed schedule is hard | Sequential sorts in kernels, low parallelism, tuning; never GPU utilisation alone | GPU change, measured end to end; never projected from data-sheet FLOPS |
| G-hyb leaves host ordering and merging as the largest share | Host threads and affinity, transfer volume, G-inc | Same GPU class with a larger host; not a larger-VRAM GPU |
| CPU throughput keeps rising with threads and CPU completion or the reference limits the schedule | Oversubscription, mixed cores, swap, duplicated memory | A larger CPU on the same work (`c7i.16xlarge` also continues the Phase 1 baseline) |
| RAM peak, swap, or pinned buffers limit | Concurrent jobs, per-worker copies, streaming, the WSL2 memory limit (host RAM ≠ WSL2 RAM) | More RAM or another environment |
| Session ends, assignment changes, or a shared PC prevents repeatable conditions | Bookable times, logging and recovery | A controllable cloud machine, possibly of the same class; this is about control, not speed |
| T1/T2 are fast enough and throughput saturates with batch | Whether another question still needs answering | No scaling; any later cloud comparison states its own reason (stronger CPU control, reproducibility) |

One out-of-memory run does not show that a larger GPU is needed: if a smaller microbatch completes the task, the
question is whether a larger batch saves time. A change of machine also changes architecture, host, and OS, so a gain is
never attributed to one resource alone; where possible the target re-runs the pilot's batches, the newly possible ones,
and its own best setting.

**P4: verification.** The target machine runs the same workload through the same pilot harness
(`pilot.py plan --purpose verification`). Each metric of the record gets gain = baseline / observed (times) or
observed / baseline (rates): **supported** if gain ≥ the record's threshold, **not supported** below it,
**indeterminate** if the run is missing or failed. Whether the gain fell inside the predicted range is reported too. This
is descriptive (one machine per side, medians, no interval); an unexpected absence of gain is a result.

**Budget.** The cap stays at **USD 250** and is not raised automatically. The record re-estimates every paid step from
pilot rates and must fit what remains; the formal comparison stops at a launch boundary when the next complete launch
pair cannot be paid for.

## Part III. Formal comparison

Runs only if the committed scaling record says so, on the machines it names (default candidates: `c7i.16xlarge` for
the CPU, an L40S `g6e` for the GPU). It is a comparison **conditional on those machines**; it is not a claim about the
best CPU or GPU.

### Hypotheses

| ID | Statement | Pass / verdict |
|---|---|---|
| R1 | The CPU runner completes T1 and T2 on both series end to end, cache-cold and warm, every interval at half-width ≤ 0.02 | Every half-width ≤ h; counts recomputed from stored per-trial bits match; 0 spot mismatches; on both Stage A launches |
| R2 | The projection Σ n<sub>c</sub>/r<sub>c</sub> with the launch's own rates at every task length predicts warm E2E | Per task × series: \|median warm E2E / projection − 1\| ≤ 0.15 |
| C | The GPU implementation is bit-identical to the CPU reference over the full gate | 0 mismatches in every gate item |
| **H3** | The GPU reaches the same result in less end-to-end time | Category per condition (verdict rules) |
| **H4** | At equal wall-clock budget, the GPU's estimate is closer to the exact reference | Category per budget (verdict rules) |
| D1 | A per-device cost model with pre-execution inputs predicts per-trial cost on held-out conditions | Per device: median \|relative error\| ≤ 0.20 and maximum ≤ 0.50 over the 13 held-out points |

Exploratory, no verdict: **D2** (end-to-end winner predicted from D1 plus fixed costs, against trivial predictors);
**D3** (consistency of per-point GPU rates between the pilot GPU and the formal GPU; the pilot was seen before the
formal choices, so this is not a held-out prediction test); **X1** (trace of the non-monotone rate dips of Phase 1);
**X2** (CPU size control on one smaller CPU launch, if the record names one); select-stage breakdown with `perf stat`
counters from separate profiling runs; same-box CPU-host; cost per task; warm H4; GPU power.
Anything added after results are seen is labelled post hoc.

### Verdict rules

**Unit.** The independent unit is an **instance launch**; runs inside a launch are technical repetitions. Stage D uses
m launch pairs (one CPU and one GPU launch each, same region and calendar day, alternating which goes first). T1/T2:
3 cache-cold runs and one warm block of 3 per launch, launch value = median; T-ref: 1 cold run per launch; T-ref-est:
10 seeds per launch pair and budget, the same seeds on both devices.

**m.** m = 3 if σ ≤ 0.047, else m = 5, where σ is the SD of log T2-week cold medians over the two formal Stage A CPU
launches. **Only formal Stage A CPU data set m**; pilot data, GPU or CPU, never do.

**H3.** ℓ<sub>i</sub> = log(T<sub>CPU,i</sub> / T<sub>GPU,i</sub>); ρ = exp(mean ℓ), 90% interval
exp(mean ℓ ± t<sub>0.95, m−1</sub> · SD(ℓ)/√m). Equivalence margin **Δ = 1.25**:

| Category | 90% interval [lo, hi] of ρ |
|---|---|
| GPU advantage | lo ≥ 1.25 |
| CPU advantage | hi ≤ 0.80 |
| Equivalent | 0.80 < lo and hi < 1.25 |
| Inconclusive | otherwise; "direction X, size unresolved" if the interval excludes 1 |

Primary conditions (cold): T1 and T2 on both series, T-ref. Secondary (warm): T1 and T2 on both series. Each gets its
own verdict; no pooled claim. A run must reproduce the reference output hash, and formal cold runs must be cache-cold.
Timeouts: 3 × the conservative CPU-best projection of the condition; a stopped run counts as exactly the timeout, which
can support the other device's advantage but never the stopped device's advantage or equivalence.

**H4.** Each run is a cold process with a deadline B (setup, aggregation, intervals, and fsync inside B). Trials are
ordered round-robin over the 15 cells and only the contiguous prefix counts, giving N<sub>c</sub>(B). Per run, MSE over
the 360 rates; per device and pair, RMSE over its 10 seeds; ε<sub>i</sub> = RMSE<sub>CPU,i</sub> / RMSE<sub>GPU,i</sub>,
with the H3 interval and categories. N<sub>c</sub> = 0 is estimated as 1/2; a late or invalid run counts as
N<sub>c</sub> = 0 everywhere; nothing is dropped. Intervals are reported only at checkpoints 2<sup>8</sup> … 2<sup>18</sup>
with α/(360·11) each. Every live run's (N<sub>c</sub>, k<sub>c</sub>) must equal what the reference bits give for the
same trial ids. Replay of stored bits at the measured N(B) is reported separately and is not H4.

**Status of every item.** Judged; **not measured** (a prerequisite failed, or the scaling record chose not to proceed;
the reason is recorded); **incomplete** (fewer than m complete pairs, a time box, or the budget). A whole launch is the
smallest unit dropped. "Not measured" and "incomplete" are never reported as an advantage or equivalence. A comparison
is invalid if the trial stream, estimator, or measurement boundary differ between devices, if parity was skipped
without a recorded reason, if a non-identical output is timed, or if tasks, accuracy, Δ, budgets, or cells changed after
results were seen.

### Stages

| Stage | Work | Exit | If not met |
|---|---|---|---|
| A. CPU runner | On the formal CPU, 2 launches: thread calibration (that machine's candidates), rates at every task length, R1, R2, σ → m | R1 passes | No speed verdicts until fixed |
| B. Reference | T1/T2 reference hashes, T-ref enumeration with CPU-best and the oracle (2,000 shared + 200 full re-evaluations per cell), golden gate; any machine with a gated build | Oracle 0 mismatches | T-ref H3 and H4 "not measured" |
| C. GPU path | On the formal GPU: partial gate for both paths, batch calibration per path, path selection (higher steady-state rate at the representative point; within 10%, G-inc) | Chosen path passes the partial gate | GPU side "not measured"; CPU results reported |
| Pre-freeze | Parity adoption, threads, rates and timeouts (formal CPU); batch, drain margins, warm-up counts (both); optional optimisation: if the chosen path is projected ≥ 1.5x faster than CPU-best on a primary condition, it is optimised further, otherwise frozen as a prototype | All values recorded | — |
| Freeze | `protocol-freeze.json` with the scaling record's hash, machines, m, configurations, timeouts, reference hashes | Committed | — |
| D0. Full gate | Gate C on the frozen GPU build over every reported task range | 0 mismatches | Fixes, re-freeze, full gate again |
| D. Comparison | m launch pairs with all conditions; D1 on pair 1; CPU-host on every GPU launch; X2 if named | — | Status rules above |

The 1.5x rule only sets optimisation effort; it is never a reason to scale hardware. The formal GPU's best path and
batch are re-established there even if the pilot chose otherwise.

**Formal machines.** A launch must match the identity rule of the profile named in the scaling record (for
`c7i.16xlarge`: Xeon Platinum 8488C, 32 cores / 64 threads, one NUMA node; for `g6e`: an L40S). Otherwise it is kept,
labelled "hardware mismatch", excluded, and replaced. A launch on which the page cache cannot be dropped is not run.
Spot and on-demand capacity run the same instance type; an interrupted launch is incomplete and replaced, never resumed.
Cost per task is reported at the official on-demand price on the run date (SKU and price-file hash), instance-hours
from launch to termination; spot prices and lost hours separately.

## Measurement

| Stage | Content | Cold | Warm |
|---|---|---|---|
| S0 | Process start, argument parsing | ✓ | — |
| S1 | Read inputs, verify SHA-256 | ✓ | — |
| S2 | Device init: thread pool / CUDA context | ✓ | — |
| S3 | Base state: scan, ranks, heads, base episodes; device allocation and uploads | ✓ | — (resident) |
| S4 | Per-length preparation: median range, factor / S / cap tables, uploads | ✓ | ✓ |
| S5 | Trial loop (batch transfers included) | ✓ | ✓ |
| S6–S8 | Aggregation per cell and policy; intervals and width check; write and fsync | ✓ | ✓ |

**Cache-cold**: new process after the page cache was dropped, precompiled device code (no JIT). **Process-cold**: new
process without the cache drop (pilot only, reported apart). **Warm**: the same process already ran a task with the same
base-state key. Resource records per run: cgroup v2 memory peak where permitted, minimum MemAvailable, swap and major
faults, CPU steal, 1 Hz CPU clock, GPU clocks, memory and power, thread affinity at loop start and end. Profiling runs
are kept apart from timed runs. Data download and build are excluded.

## What is fixed when

| Fixed in this revision | Fixed later by a rule stated here | Free during implementation |
|---|---|---|
| Questions, tasks, cells, budgets, checkpoints; hypotheses and verdict rules (Δ, interval, m rule, statuses); pilot steps, repetitions, cold kinds, never-rules; scaling record items, actions, verification rule, budget cap; calibration designs; oracle; trial stream v1; numerical contract; gate items; D1 model and split | Pilot plan (P0, before the pilot's first run); scaling action, formal machines, thresholds (P3, before any paid run); formal threads, m, CPU-best, GPU path, batch, drain, timeouts (Part III, before the freeze) | Code structure, kernels, launch configuration, file formats, scripts (until the freeze); correctness fixes at any time, followed by the gate and re-runs of affected timings |

A change to the first column needs a dated entry in this file before the affected measurement. Choices made after
seeing results make the affected comparison exploratory.

## What this phase can and cannot claim

- It can show what limits this exact computation on the machine at hand, whether the resource added for an observed
  reason helped the same work as predicted, and, if the formal comparison runs, whether a GPU implementation reaches the
  same exact result sooner than CPU-best on the chosen machines, at what cost, and with which per-trial cost model.
- It does not claim a cost-optimal or minimum-memory configuration, or that the chosen machines are the best ones.
  Stage-time slowdowns are not attributed to cache, bandwidth, or clock without counter evidence.
- It cannot show that GPUs in general are necessary or unnecessary, a lower bound on CPU cost, behaviour on other
  assets, periods, or detectors, or anything about damage bounds or real surveillance effectiveness.
- A loss or a missing GPU path says something about the implementation built within the stated effort, not about what
  a GPU can do.

## Files

| File | Content |
|---|---|
| [`README.md`](README.md) | This protocol |
| [`SPEC.md`](SPEC.md) | Implementation specification |
| [`protocol.json`](protocol.json) | Parameters: tasks, machines, pilot, scaling, formal |
| [`IMPLEMENTATION.md`](IMPLEMENTATION.md) | Code map, workflow by step, what has and has not been verified |

Results: `results/pilot/<label>/` (Part I and P4), `results/scaling/` (Part II), `results/formal/<label>/` (Part III).

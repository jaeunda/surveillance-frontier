# Phase 1 — feasibility: what does a stated stress-testing task cost on a strong CPU?

Status: **done**. Reference run on 2026-10-09 (AWS `c7i.16xlarge`, commit `bc9cd9c`), results in
[`results/aws-c7i-16xlarge_2026-10-09/`](results/aws-c7i-16xlarge_2026-10-09/), write-up and verdicts in
[`docs/phase1.md`](../../docs/phase1.md). The plan below (Question through Measurement protocol) is kept as it was
frozen before the run; the file tables only list scripts and outputs added afterwards.

Post-run audit note: the stored F4 `supported=True` checks trial-loop time only, whereas the frozen hypothesis
below asks for end-to-end validation. That broader claim remains unverified; see the corrected
[results discussion](../../docs/phase1.md#how-far-the-projection-was-checked-f4). The 660 sweep records comprise
600 timed repetitions, 33 gate records, and 27 fixed-trial validation records; validation was not repeated three times.

## Question

Phase 0 showed that the number of detector re-evaluations per unit of time decides how precisely a detector's
behaviour can be measured. Phase 1 asks whether that limit binds for concrete tasks, and makes sure that the CPU side
of that answer is not an artefact of a wasteful implementation:

1. **What exactly is estimated, and how many trials does a stated accuracy need?** (target quantity, tasks)
2. **What is the cheapest exact way to run a trial on a CPU?** Re-running the full detector for every policy and
   every trial is one option, not the floor.
3. **How long does each task take on the reference machine** with each method, measured on real policy batches and
   checked against an end-to-end run?

This is hypothesis H2 of the project. A GPU engine is out of scope; it is compared against this baseline next.

## Target quantity

For a policy π, a fixed series x, and a test event of length L and strength q (the Phase 0 shape, applied at a
start s), the **conditional detection rate** is

> p<sub>π</sub>(L, q | x) = share of all valid starts s at which π reports a **new** episode centred inside the
> event, i.e. one that is not among π's episodes on the unchanged series.

- It is conditional on x: its intervals do not cover other weeks, regimes, or assets.
- It is **not a damage bound**. A worst-case undetected effect needs a damage function, an admissible scenario set,
  and a bound direction; a maximum found by search would be a lower bound on the worst case, and a high-budget run
  is not a certified reference. Those definitions belong to Phase 3 and are not claimed here.
- It is exact on small problems: evaluating every start (enumeration) gives p without sampling error, which is how
  the estimator is checked (E1).
- Controls reported with it: q = 0 (no change, must give 0), price-only and volume-only events (the detector needs
  both features), and the *overlap* rate (an episode centred inside the event, new or not; the Phase 0 definition).

A Monte Carlo trial draws one start uniformly and evaluates **all policies of the task** on it. Intervals are
Clopper–Pearson, whose coverage is at least nominal for every p. (Wilson intervals were the first choice; the smoke
run's estimator check showed them undercovering near p = 0, worst cell 0.84, and the Bonferroni joint coverage of
432 intervals at 0.86–0.91, so they were replaced before the reference run.) Trials per cell = the smallest n whose
interval has half-width ≤ h at the worst case p = 0.5, at α = 0.05 for **pointwise** intervals (each cell on its
own) or α = 0.05 / (cells × policies) for **simultaneous** intervals (the whole surface at once).

Two things the rate does not give for free, both measured by the estimator check: a ±h error in p is not a ±h
error in the strength at which detection sets in (that depends on the local slope), and p need not increase with q
(global re-ranking and the greedy episode merge can make it dip; the smoke run found such dips).

## Tasks ([`scenario_space.json`](scenario_space.json))

| Task | Question | Policies | Cells (L × q) | Coverage | Trials per cell |
|---|---|---|---|---|---|
| T1-curve | How does the default policy's rate change with event length and strength? | 1 (K = 15, s_min = 2, gap = 60) | 8 × 17 (L = 2…256 x2, q = 1…256 x√2) | pointwise, h = 0.02 | 2,449 |
| T2-policy-map | The same for 24 policy settings, every interval holding at once | 24 (K 8/15 × s_min 1.5–3 × gap 30/60/120) | 8 × 17 | simultaneous, h = 0.02 | 11,726 |

Around T1, one factor at a time: half-width (0.05, 0.02, 0.01), strength step (x2, x√2, x2^¼), number of lengths
(1, 4, 8, 15), policy grid (1, 24, 105), coverage (pointwise, simultaneous). With a p-map from the estimator check,
`scenario_space.py --pmap` also shows **variance-aware allocation** (cells far from p = 0.5 need fewer trials).

Units (independent series per trial, e.g. one per key) multiply every time linearly and are reported apart; all
times are for one unit. The two series (1-minute quarter, 1-second week) are two workloads of different length, not
a resolution comparison: their windows, gaps, and periods differ in real time.

## Trial methods (bit-identical outputs)

| Method | Per trial | Why it is exact |
|---|---|---|
| full | every policy re-runs scan + full-sort ranking + selection | the reference |
| shared | one scan and one full-sort ranking at the largest K; peaks once at the smallest s_min; per-policy filter and merge | rows are independent per length; peaks do not depend on s_min, K, or gap |
| tail | as shared, ranking only the values above each feature's (cap + 1)-th largest (selection + small sort) | a start outside the cap cannot be a candidate or outrank one |
| incremental | recompute the ≤ L + w − 1 windows per length that overlap the event; merge them into the base series' precomputed orderings | the exact tail of the changed row follows from the old ordering minus the replaced starts plus their new values |

Identity contract: all scan paths return the same feature bits (exact min/max, fixed-point volume sums), so counts and
episodes must match exactly, never within a tolerance. `sf_test` checks it on synthetic tie-heavy data for 18
policies × 42 events (both series ends, L = 2/30/256, q = 0/4/32, three event kinds) and with four negative controls;
the `verify` part checks it again on the real series before any timing.

## Hypotheses (revised 2026-10-07, frozen before the reference run)

| ID | Hypothesis | Pass criterion |
|---|---|---|
| G | Gate: on the real series, shared / tail / incremental return the same episodes as full for all 24 policies | 0 mismatching trials |
| F1 | Full recomputation is not the CPU's cost floor | Best exact method ≥ 10x the trials/s of full at N = 604,800, K = 15, one policy, 1 thread |
| F2 | Each method's cost follows its model: full a + b·NK + c·NK·log N; tail and incremental a + b·NK | Model fitted on N ≤ 64,000 predicts N = 604,800 within 20% for every K |
| F3 | Independent trials scale near-linearly over physical cores, for every method | Across-mode throughput at 32 threads ≥ 0.7 × 32 × single-thread, at N = 64,000 and 604,800 |
| F4 | Measured batch throughput predicts real time-to-solution | Projection vs an end-to-end run of the validation task within 15%, per method |
| E1 | The interval method keeps its stated coverage against exact rates | No cell and policy has replayed coverage below 0.95 by more than 4 replay standard errors (0.931 at 2,000 replays); Bonferroni intervals all hold in ≥ 95% of runs |

Prior evidence, stated up front: a laptop smoke run gave about 1,650 / 294 / 4.3 ms per trial for full / tail /
incremental on the 1-second week (K = 15), so F1 is expected to hold; its projection put T1 on the 1-second week at
about 2 days (full), 9 hours (tail), and 4 minutes (incremental) on 9 cores. Within mode and the SMT gain (64 vs 32
threads) are reported descriptively. If F3 fails, the per-stage slowdown at 32 threads is reported next to it; it
shows which stage stops scaling, not why (bandwidth, SMT, or frequency would need counters).

**Decision rule for H2.** For each task, the time-to-solution of the fastest exact method on the reference machine
decides: under 1 hour, the CPU is not the bottleneck for that task and the GPU comparison of Phase 2 moves to larger
tasks (more units, a more expensive detector, or bounds once Phase 3 defines them); over 1 hour, Phase 2 compares
CPU and GPU on that task with the same estimator, the same trial stream, and error against an exact reference.
"Full recomputation is slow" alone does not support H2.

## Reference machine

AWS `c7i.16xlarge` (64 vCPU = 32 physical cores, 128 GB, single socket, on-demand, us-east-1), Ubuntu 24.04,
engine built on the instance, `OMP_PLACES=cores OMP_PROC_BIND=spread`, thread ladder 1, 2, 4, 8, 16, 32, 64. The
laptop used during setup is a secondary environment only.

## Measurement protocol

- Every timed configuration is repeated 3 times (5 s each); all repetitions of all parts run in one seeded shuffled
  order. Reported values are medians over repetitions; nothing is picked as the best of several runs.
- Throughput is the steady-state rate: per worker, trials completed / time of its last completion, summed over
  workers. Trials that finish after the budget are counted in that rate but not in `trials_by_deadline`.
- Each worker allocates, first touches, and warms up its own memory on its own thread; the CPU each worker ran on,
  `OMP_*`, `lscpu`, `numactl --hardware`, the governor and THP setting are recorded in `env.json`.
- The projection uses the batch rates (every policy of the task evaluated per trial, all physical cores), at the
  nearest measured event length (2, 30, 256). The `scenario` part shows how much cost varies with length, strength,
  and event kind; the validation task (lengths 6/30/120, strengths 8/32/128) checks the projection end to end.

## Files

| File | Content |
|---|---|
| [`scenario_space.json`](scenario_space.json) | Target quantity, policy grids, tasks, one-factor ladders, validation task |
| [`check_phase0.py`](check_phase0.py) | Gate: the C++ engine reproduces the Phase 0 top 20 on 2024 Q1 |
| [`run_sweep.py`](run_sweep.py) | Real-data gate and CPU measurements (parts verify, cost, scaling, batch, scenario, validate) |
| [`validate_estimator.py`](validate_estimator.py) | Exact rates by enumeration on a small problem; coverage, monotonicity, transition error (E1) |
| [`scenario_space.py`](scenario_space.py) | Trials per task and projected time-to-solution from measured batch rates |
| [`make_figures.py`](make_figures.py) | Figures 1-4, `tables.md`, and the verdicts for one run |
| [`make_doc_figures.py`](make_doc_figures.py) | Figures of `docs/phase1.md` and the README from one run (`docs/assets/phase1/`) |
| [`replay_estimator.py`](replay_estimator.py) | Post-run check: exact rates and the E1 coverage replay from a run's `raw_hits.tar.gz`, compared with its CSVs |
| [`run_reference.sh`](run_reference.sh) | The whole protocol on the reference machine, in order |
| [`requirements.txt`](requirements.txt) | Python dependencies of the scripts |

## Run

On the reference machine, from the repository root, with a clean commit checked out:

```bash
experiments/phase1-feasibility/run_reference.sh aws-c7i-16xlarge
```

It builds the engine, runs `sf_test` and `check_phase0.py`, downloads the data, runs the sweep (verify gate first),
the estimator check, and the figures. Step by step:

```bash
cmake -S . -B build && cmake --build build -j && ./build/sf_test        # build + engine gate
python3 -m venv .venv && .venv/bin/pip install -r experiments/phase1-feasibility/requirements.txt
.venv/bin/python cases/finance/fetch_klines.py 1m-2024q1                 # data/ (not committed)
.venv/bin/python cases/finance/fetch_klines.py 1s-2024-01-08
.venv/bin/python experiments/phase1-feasibility/check_phase0.py         # reproduction gate
.venv/bin/python experiments/phase1-feasibility/run_sweep.py --env <machine-label>
.venv/bin/python experiments/phase1-feasibility/validate_estimator.py --env <machine-label>
.venv/bin/python experiments/phase1-feasibility/make_figures.py \
    experiments/phase1-feasibility/results/<env>_<date>
```

`run_sweep.py --quick` runs a few configurations as a smoke test.

## Result files (per run)

| File | Content |
|---|---|
| `env.json` | CPU, cores, OS, compiler, git commit, OpenMP binding, `lscpu`, `numactl`, governor, THP |
| `cpu_sweep.jsonl` | One line per configuration: part, repetition, order, method, policy grid, trials/s, stage medians, per-policy hit and overlap rates, worker CPUs, mismatches |
| `estimator_cells.csv`, `estimator.csv` | Exact rates per cell and policy; coverage, monotonicity, transition error |
| `tables.md`, `hypotheses.csv` | Gates and verdicts, cost tables, stage slowdown, time-to-solution, projection check |
| `figs/fig1_cost_vs_n.png` | Cost per trial vs N per method, with cost models (F1, F2) |
| `figs/fig2_stages.png` | Where one trial spends its time, per method |
| `figs/fig3_thread_scaling.png` | Speedup vs threads per method (F3) |
| `figs/fig4_time_to_solution.png` | Time-to-solution per task and method, projection error on the validation task (F4) |
| `figs/fig5_estimator.png` | Exact detection rates and interval coverage (E1) |
| `tasks_variance_aware.csv` | Exploratory projection with variance-aware allocation (not a verified result; see `docs/phase1.md`) |
| `raw_hits.tar.gz` | Per-start detection bits from enumeration (`raw/hits_L*_q*.u8`, uint8, starts × policies), input of the coverage replay; `raw/` itself is not committed |
| `SHA256SUMS` | Checksums of the run's files (recorded after copying them back), including the unpacked `raw/` files |

## How the pieces connect

1. `cases/finance/fetch_klines.py` writes `data/*.bin` (the engine's input format, `engine/include/sf/series.hpp`).
2. `build/sf_test` and `check_phase0.py` must pass; the `verify` part repeats the method identity on real data.
3. `run_sweep.py` calls `build/sf_bench` once per configuration; each call builds the base state once, runs trials
   (`engine/apps/sf_bench.cpp`), and prints one JSON line.
4. `validate_estimator.py` calls `sf_bench --enumerate` for exact rates and replays Monte Carlo runs from them.
5. `scenario_space.py` turns the tasks into trial counts and divides by the measured batch rates;
   `make_figures.py` adds the cost models, scaling, verdicts, and figures.

Engine internals and a reading order: [`engine/README.md`](../../engine/README.md).

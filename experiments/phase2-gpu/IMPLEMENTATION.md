# Phase 2 implementation

The code that carries out the [protocol](README.md) (design r2) and the [specification](SPEC.md). Where SPEC leaves a
choice open (code structure, file formats, scripts), this file records the choice. **No Phase 2 measurement has been
run.** The engine (C++/CUDA) is the instrument and is shared by every part; the experiment code in this directory was
rebuilt for r2 around the pilot → scaling decision → formal comparison sequence.

## Engine (SPEC §1–§7)

| SPEC component | File | Notes |
|---|---|---|
| `sf::BatchEvaluator` | [`engine/include/sf/batch.hpp`](../../engine/include/sf/batch.hpp) | Trial space (g → cell, j, start) as plain data shared with the device; per-cell plans (S4); packed hit/overlap words |
| Trial stream `sf-stream-v1` | [`engine/include/sf/stream.hpp`](../../engine/include/sf/stream.hpp) | Philox4x32-10 + Lemire, one header for host and device |
| CPU back end | [`engine/src/batch_cpu.cpp`](../../engine/src/batch_cpu.cpp) | One incremental `Evaluator` per OpenMP worker; dynamic schedule; per-worker CPU and affinity at loop start and end |
| CPU parity PC1–PC4 | [`engine/src/evaluate.cpp`](../../engine/src/evaluate.cpp) (`EvalOptions`) | All off = the Phase 1 method. PC5 is what the per-worker `Evaluator` already does. PC6 groups trials by cell |
| GPU back end (G-inc, G-hyb) | [`engine/cuda/`](../../engine/cuda/) | `SF_HD` device functions run as CUDA kernels (`SF_CUDA=ON`) or as host loops (back end `emul`, never timed). Free device memory is read once after context creation and before the back end's allocations; `device_info` reports it with total memory, shared and scratch bytes, the per-trial working set, b_max, and the free memory after allocation (SPEC §6) |
| Shared functions of the numerical contract | `scenario.cpp`, `select.hpp`/`.cpp` | Refactored out of the Phase 1 code; the GPU uses host tables built by these functions |
| Exactness checker | [`engine/include/sf/check.hpp`](../../engine/include/sf/check.hpp) | Stage by stage: features → tail counts → ordered candidates → episodes → hit/overlap |
| Clopper–Pearson | [`engine/src/interval.cpp`](../../engine/src/interval.cpp) | Checked against SciPy by `check_intervals.py` |
| `sf_run`, `sf_gate`, `sf_test2` | [`engine/apps/`](../../engine/apps/), [`engine/tests/test_phase2.cpp`](../../engine/tests/test_phase2.cpp) | Runner modes cold, deadline, warm, enumerate, bench, spot, cp; gate items 1–8 and 10, `--partial`, `--oracle` |

## Experiment code (this directory)

| Part | File | Content |
|---|---|---|
| Library | [`p2/common.py`](p2/common.py) | Paths, protocol, hashing, fsynced JSONL records, git state |
| | [`p2/tasks.py`](p2/tasks.py) | Task files from `protocol.json` into `tasks/{main,calib,parity,tdiag,pilot}/` |
| | [`p2/machine.py`](p2/machine.py) | Manifest, topology (P/E mapping from sysfs or per-CPU max MHz, else explicit), GPU state, capabilities, identity against a machine profile, EC2 metadata and official prices |
| | [`p2/harness.py`](p2/harness.py) | Configurations and keys, `taskset` pinning, cold (cache-/process-cold) / warm / deadline runs, bench, cgroup and 1 Hz sampler |
| | [`p2/design.py`](p2/design.py) | Short + sustained choice, parity rule, path selection, drain calibration, `perf stat` profiling |
| | [`p2/projection.py`](p2/projection.py) | Time projections with scenarios and rate flags; T-ref per-cell projection; planned vs reported device memory |
| | [`p2/records.py`](p2/records.py) | Metric values of a label (e2e, rate; `best` per machine), gains, the metric catalogue |
| | [`p2/stats.py`](p2/stats.py), [`p2/formal.py`](p2/formal.py) | Interval and categories; formal machines from the committed scaling record, launch directories, condition lists |
| Part I | [`pilot.py`](pilot.py) | `plan`, `amend`, `check` (P0), `run --step` (P1/P2), `summary` |
| Part II | [`scaling.py`](scaling.py) | `draft`, `check`, `verify` (P4) |
| Part III | [`calibrate.py`](calibrate.py), [`reference.py`](reference.py), [`freeze.py`](freeze.py), [`launch.py`](launch.py), [`run_launch.sh`](run_launch.sh) | Formal calibration, reference and oracle, freeze, Stage A/D launches |
| Analysis | [`analyze.py`](analyze.py), [`make_figures.py`](make_figures.py) | `stage-a`, `stage-d`, `cross-env` (D3), `spend`; formal and pilot figures |
| Checks | [`stream.py`](stream.py), [`check_intervals.py`](check_intervals.py), [`check_phase1_hits.py`](check_phase1_hits.py), [`test_p2.py`](test_p2.py), [`make_tasks.py`](make_tasks.py) `--check`, [`collect_env.py`](collect_env.py) | Independent stream, SciPy interval check, gate item 9, synthetic tests of the analysis and decision logic |

### Choices made where SPEC leaves them open

- **GPU pipeline.** Per microbatch: rows → features → tails (changed values sorted, merged with the base head) →
  both-set → peaks → candidates. G-inc orders and merges per (trial, policy) on the device; G-hyb copies candidates to
  the host and uses the CPU code. No dense count array and no truncation. Sorts inside kernels are per-thread heap sorts:
  correct for any tie order, but slow; the pilot measures how slow.
- **Working set.** Per trial as laid out in `gpu_backend.inl::per_trial_bytes` (about 14 MB on the week at
  s_floor = 1.5; the SPEC upper estimate is 21 MB). Which of the two the allocation matches is a pilot measurement.
- **Pilot reference hash.** Within a pilot label, the first CPU result of a task with 0 spot mismatches is the identity
  reference for all later runs of that label (`hashes.json`); formal runs use the Stage B reference.
- **Pilot CPU configuration.** `cpu-baseline` uses every logical CPU without pinning. After `threads`, the choice at
  1s-week/P24 is the CPU configuration of `rates`, `diag`, `tref-rates`, `tref-build`, and `deadline`. GPU steps use the
  `microbatch` step's best batch, else 64.
- **Warm blocks.** One priming task per (condition, configuration), then the timed requests. Priming T2 on the week is a
  full task run; blocks avoid repeating it per repetition.
- **Cold exit.** After the last output and `stages.json` are fsynced, `sf_run` calls `_Exit`; teardown is not part of
  the end-to-end time, for every back end.
- **Outputs.** `hits.bin` and `ref_bits.bin` hold bit t·P + p, least significant bit first. `result_sha256` hashes
  `cells.csv` (fixed n), `ref_bits.bin` (T-ref) or `prefix.json` (deadline).
- **Censoring.** A timed-out run is a lower bound on its own device's time; `analyze.py` does not conclude that
  device's advantage or equivalence from it.

## Workflow

| Step | Machine | Commands (from the repository root; `PY=.venv/bin/python`, `H=experiments/phase2-gpu`) |
|---|---|---|
| Build | any | `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release [-DSF_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120-real]`, `cmake --build build -j`; `$PY $H/make_tasks.py --check`; `$PY $H/test_p2.py` |
| P0 | lab PC or Colab | `$PY $H/pilot.py plan --label lab_2026-10-12 --machine lab-desktop --hours H [--p-cores LIST]`; commit the plan; `$PY $H/pilot.py check --label …` |
| P1 | same | `pilot.py run --label … --session 1 --step cpu-baseline`, then `gpu-rates`, `gpu-tasks` |
| P2 | same | `--step threads`, `microbatch`, `rates`, `diag`, `tref-rates`, `tref-build`, `deadline`; session 2 at another time of day; `pilot.py summary --label …`; `make_figures.py pilot …` |
| P3 | any | `scaling.py draft --pilot …`; complete the record; `scaling.py check`; commit |
| P4 | target | `pilot.py plan --label … --machine <target> --purpose verification --scaling-record $H/results/scaling/scaling-decision.json`; the same steps; `scaling.py verify --action I --target-label …`. After termination: `collect_env.py --ec2-times ID --out …/session-K/ec2.json` |
| Part III | formal machines | `run_launch.sh --stage A --role cpu --launch 1|2 --label L`; `analyze.py stage-a L`; `reference.py --label L`; `calibrate.py parity|threads|rates|drain` (CPU); `calibrate.py batch --path inc|hyb`, `path`, `drain --config gpu`, `threads --gpu-host` (GPU); `freeze.py`; commit; D0: `build/sf_gate --device cuda --path P --out $H/results/formal/L/d0/gate_cuda.json`; `run_launch.sh --stage D --role cpu|gpu --launch i`; `analyze.py stage-d L`; `make_figures.py formal L` |

## Verified (laptop, no GPU)

| Check | Result |
|---|---|
| `sf_test` (Phase 1 gate) after the refactoring | passes |
| Gate item 9: the 18 Phase 1 hit files, GCC 13.3 build | 18/18 byte-identical |
| `sf_test2`: Philox known-answer vectors, counter layout, SHA-256 vectors, interval edge cases, device arithmetic vs CPU definitions, PC1–PC4 episodes and per-stage traces, `cpu` and `emul` (G-inc, G-hyb) against CPU shared/full on random and tie-heavy series (135 events × 18 policies, every stage) | 20/20 pass |
| `sf_gate --device emul`, items 1, 2, 6, 7, 8 | pass. All six negative controls are detected on their witnesses: device log10 and reversed tie at the ordered candidates; cap − 1 at the tail counts; direct float, half-to-even and fast math at the features. The rejection sequence matches `stream.py` (4,096 draws, 1,424 rejections). 1,000,000 starts match the host |
| `sf_gate --partial`, items 3 and 5, both GPU paths (emulated) | pass (66 real-data trials; 8,130 trials at N ≤ 4,096, every start) |
| `sf_gate --device emul --path inc`, items 1, 3, 4, 5 with 2 gate starts per combination (instead of 64) | pass: real data on both series with P105, all 10 lengths, all 18 strengths and all 3 event kinds, 4,336 trials, 0 mismatches at every stage; every start at N ≤ 4,096, 40,118 trials, 0 mismatches |
| `sf_gate --device cpu --parity PC1,PC2,PC3,PC4,PC6`, items 1, 3–8, same reduced starts | pass (same trial counts, 0 mismatches). Item 7 first failed because the CPU back end rethrew worker errors as a generic error, which hid the overflow rejection. It now rethrows the original exception, and item 7 passes with and without parity |
| Clopper–Pearson vs SciPy, every k at T1, T2 and all 11 checkpoints | pass; max \|difference\| ≤ 1e-13 up to n = 32,768, rising to 7.8e-13 at n = 262,144 (tolerance 1e-12: passes with little margin at the largest checkpoints); same width verdicts |
| `sf_run` cold, reduced T1 on the quarter: `cpu`, `cpu` with PC1+PC2+PC3+PC4+PC6, `emul` | identical `cells.csv` and `hits.bin` hashes; spot verification 0/256 mismatches |
| T-ref enumeration on a 20,000-bar prefix: `cpu` vs `emul` G-hyb; `sf_gate --item10`; reference oracle (40 shared + 4 full per cell) | identical bits; oracle 0 mismatches |
| Deadline runs (`cpu`, `emul`) and the H4 identity replay: `stream.py` regenerates every start, and (N_c, k_c) equal the reference bits | match on both back ends |
| Warm mode through the harness: the base state stays resident, and the result hash equals the cold run's | yes (r1 harness; re-checked with the r2 harness below) |
| `test_p2.py` (r2): categories and censoring, H3 with interrupted / mismatched / process-cold launches, H4 ε, D1 recovery of a linear cost on the 13 held-out points, projections and rate flags, T-ref trial count, memory check, configuration keys and pinning, thread candidates, task generation (15 T-ref cells, 13 held-out points), scaling-record validation, gains | 30/30 pass |
| r2 pilot chain on the laptop, with a smoke plan amended to T1 quarter only and 1 s benches: `plan` → `amend` → `check` (sf_test, sf_test2, intervals, stream, `test_p2.py`, CPU gate items 1, 6, 7, 8) → `cpu-baseline` (2 cold + warm block of 2) → `gpu-tasks` (refused: no device gate) → `deadline` → `tref-rates` → `summary` → `scaling.py draft/check` → a verification label → `scaling.py verify` | all steps ran. Cold runs were labelled process-cold (no root); warm runs reused the base state; every deadline run with drain 0 exited 30–80 ms after its deadline ("late"), which is what the formal drain margin is for; T-ref projection on this laptop 7.2–8.0 h; runs were flagged for memory pressure because the laptop was already swapping. Smoke outputs were deleted |
| CPU gate item 5 (every start at N ≤ 4,096 against full recomputation) on the CPU back end | did not finish within 5 min on the laptop; it stays in the device partial gate and the formal CPU gate, not in pilot P0 |
| CUDA back end compiled for sm_75, sm_89, sm_120 (SASS only) and linked into `sf_run` and `sf_gate` with `SF_CUDA=ON` | builds without warnings with **nvcc 13.4** (NVIDIA's pip wheel). This is not the planned 12.8.1 toolkit. Without a driver the binary stops with a CUDA error and does not crash |

## Not verified (needs hardware or access this machine does not have)

- **Any CUDA execution.** Kernel launches, device memory sizing (b_max), transfers, the device `log10`
  negative-control witness on real hardware, event-based phase timing, and the partial and full gate on sm_120,
  sm_89 or sm_75. The device code has only run under the host emulation. Exactness on the device still depends on
  `__fmul_rn`/`__fdiv_rn`/`__ll2double_rn`/`__double2float_rn` behaving as specified, and the gate exists to test
  exactly that.
- **GPU performance.** Nothing has been measured. The prototype's per-thread sorts and sequential compaction are
  obvious Stage C work.
- **Toolchain.** No build with CUDA 12.8.1 has been tried. The toolchain is pinned at the freeze.
- **Root-only parts of the harness.** Dropping the page cache and cgroup v2 `memory.peak` both need root, which was
  unavailable here. The runs record "not dropped" and `null`.
- **AWS parts.** IMDS, the EC2 and Price List APIs, spot price history, interruption polling, S3 upload, and
  shutdown. These code paths were not executed, and the price filter values should be checked once against a real
  `get-products` response.
- **Full-size runs.** T1/T2 on the week, T-ref (9,071,175 trials), the full oracle, and the full gate with 64 starts were not run here, for lack of time on this machine.
- **Overhead checks of SPEC §8** (timers < 1% against a timer-free build; sampler < 1%). No timer-free build
  variant exists yet.
- **Not implemented:** the X1 trace tool (exploratory).
- **Not exercised:** the r2 steps `threads`, `rates`, `gpu-rates`, `microbatch`, `diag`, `tref-build` with full settings; every Part III script (`calibrate.py`, `reference.py`, `freeze.py`, `launch.py`, `analyze.py stage-a/stage-d/cross-env/spend` on real records); the new device-memory fields of `device_info` on a real GPU.

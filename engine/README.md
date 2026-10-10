# engine — detector evaluation core (C++17, OpenMP)

Domain-agnostic: a series of (low, high, volume) bars goes in, the top rare episodes come out. One call of
`sf::detect` is one evaluation of one policy; a stress-test trial evaluates a set of policies on the series with one
test event applied (`sf::Evaluator`), by one of four methods that must return the same bits.

## Reading order

| Step | File | What to look for |
|---|---|---|
| 1 | [`include/sf/series.hpp`](include/sf/series.hpp) | Input: `Series` (owning) and `SeriesView` (pointers + n); binary file format |
| 2 | [`include/sf/detector.hpp`](include/sf/detector.hpp), [`src/detector.cpp`](src/detector.cpp) | The whole pipeline in three calls; `Workspace` holds all reusable memory |
| 3 | [`include/sf/scan.hpp`](include/sf/scan.hpp), [`src/scan.cpp`](src/scan.cpp) | Stage 1 and the identity contract (exact min/max, fixed-point volume). `scan_direct` (reference) vs `scan_starts` (sliding deques) |
| 4 | [`include/sf/rank.hpp`](include/sf/rank.hpp), [`src/rank.cpp`](src/rank.cpp) | Stage 2. Full sort vs tail ranking (selection, then sort above the cap) |
| 5 | [`include/sf/select.hpp`](include/sf/select.hpp), [`src/select.cpp`](src/select.cpp) | Stage 3. Local peaks, ordering by rarity, per-policy filter and episode merge |
| 6 | [`include/sf/scenario.hpp`](include/sf/scenario.hpp), [`src/scenario.cpp`](src/scenario.cpp) | One scenario = base series + one synthetic test event; controls (q = 0, price-only, volume-only); `restore` |
| 7 | [`include/sf/evaluate.hpp`](include/sf/evaluate.hpp), [`src/evaluate.cpp`](src/evaluate.cpp) | One trial over a policy set: full / shared / tail / incremental; the new-detection rule |
| 8 | [`tests/test_engine.cpp`](tests/test_engine.cpp) | Correctness gate: bit identity, brute-force counts, method differential, negative controls |
| 9 | [`apps/sf_bench.cpp`](apps/sf_bench.cpp) | Timing loop: `within` vs `across`, steady-state throughput, enumeration, `--verify` |

`apps/sf_detect.cpp` runs one evaluation and prints the episodes (used by the Phase 0 reproduction gate).

### Phase 2 additions

| Step | File | What to look for |
|---|---|---|
| 10 | [`include/sf/stream.hpp`](include/sf/stream.hpp) | Trial stream sf-stream-v1: Philox4x32-10 counter, Lemire mapping; host and device |
| 11 | [`include/sf/batch.hpp`](include/sf/batch.hpp), [`src/batch_cpu.cpp`](src/batch_cpu.cpp) | `BatchEvaluator`: trial space g -> (cell, start), cell plans, packed bits; the CPU back end |
| 12 | [`cuda/gpu_kernels.hpp`](cuda/gpu_kernels.hpp), [`cuda/gpu_backend.inl`](cuda/gpu_backend.inl) | The GPU pipeline (G-inc, G-hyb) as host/device functions; `runtime.hpp` runs it on CUDA or as a host emulation |
| 13 | [`include/sf/check.hpp`](include/sf/check.hpp), [`include/sf/interval.hpp`](include/sf/interval.hpp) | Exactness checker (first differing stage); Clopper-Pearson |
| 14 | [`apps/sf_run.cpp`](apps/sf_run.cpp), [`apps/sf_gate.cpp`](apps/sf_gate.cpp), [`tests/test_phase2.cpp`](tests/test_phase2.cpp) | Task runner, correctness gate, unit checks |

`EvalOptions` in `evaluate.hpp` holds the CPU parity options (all off = the Phase 1 method), the select sub-stage
timers, and the per-stage trace used by the gate. See [`experiments/phase2-gpu/IMPLEMENTATION.md`](../experiments/phase2-gpu/IMPLEMENTATION.md).

## Build

```bash
cmake -S . -B build && cmake --build build -j && ./build/sf_test && ./build/sf_test2   # from the repository root
cmake -S . -B build -DSF_CUDA=ON && cmake --build build -j                              # adds the CUDA back end
```

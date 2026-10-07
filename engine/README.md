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

## Build

```bash
cmake -S . -B build && cmake --build build -j && ./build/sf_test     # from the repository root
```

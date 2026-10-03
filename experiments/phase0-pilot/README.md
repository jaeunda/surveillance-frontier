# Phase 0 — pilot: exhaustive multi-scale window scan on the GPU

Scan every (start time, window length) pair of a Bitcoin price series for intervals where price range and volume are
both unusually large, on CPU and GPU, and measure when the GPU helps, where the time goes, and what a fixed time
budget buys. Results write-up: [`docs/phase0.md`](../../docs/phase0.md).

| File | Content |
|---|---|
| [`pilot.py`](pilot.py) | Notebook source in percent format. Edit this file, then regenerate the `.ipynb` |
| [`pilot.ipynb`](pilot.ipynb) | Colab notebook (environment → data → CPU/GPU implementations → correctness gate → events → injection → time budget → cost model → sweep → breakdown → layout → verdicts) |
| [`requirements.txt`](requirements.txt) | Python dependencies for local runs |
| [`results/colab-t4_2026-10-04/`](results/colab-t4_2026-10-04/) | Reference run on Colab Tesla T4: CSV/JSON outputs and figures |

## Run on Colab

1. Upload `pilot.ipynb` to Google Colab.
2. *Runtime → Change runtime type → T4 GPU*, then *Run all*.
3. Results are written section by section to `/content/pilot_results/` (`progress.json` records completed sections).
   The last cell zips them without the cached market data and starts a download. Store the contents as
   `results/<environment>_<date>/`.

## Regenerate the notebook

From the repository root:

```bash
python3 tools/py2nb.py experiments/phase0-pilot/pilot.py experiments/phase0-pilot/pilot.ipynb
```

## Local dry run (no GPU)

```bash
python3 -m venv .venv && .venv/bin/pip install -r experiments/phase0-pilot/requirements.txt
.venv/bin/python tools/run_notebook.py experiments/phase0-pilot/pilot.ipynb /tmp/pilot-dry
```

`tools/fakecupy` compiles each RawKernel's CUDA C with g++ and runs it once per (block, thread). Kernel logic, gates,
and figures are exercised, but there are no warps or memory hierarchy, so **dry-run timings and verdicts are
meaningless.**

## Result files

| File | Section | Content |
|---|---|---|
| `env.json` | 1, 16 | GPU/CPU, driver and library versions, GPU clocks before/after the sweep |
| `correctness_gate.csv` | 8 | Gate checks and negative controls |
| `top20_real_1m.csv` | 10 | Top-20 episodes on the 2024 Q1 1-minute series with the nearest listed event |
| `injection_trials.csv`, `injection_summary.csv` | 13 | Controlled injections by length × strength (H5, H6) |
| `budget_same_workload.csv`, `budget_same_time.csv` | 14 | Same-workload and same-budget re-evaluation (H7) |
| `fixed_cost_model.json` | 15 | Calibrated cost model (H1) |
| `sweep.csv` | 16 | CPU direct / CPU optimized / GPU kernel / GPU end-to-end over N × window sets |
| `e2e_breakdown.csv` | 17 | Per-segment timings of the CPU-only path and three GPU output paths (H2, H3) |
| `layout_blocksize.csv` | 18 | Thread layout × block size (H4) |
| `hypotheses.csv`, `summary.json` | 19 | Verdicts and run summary |
| `figs/` | 11–18 | Figures 1–9 |

## Data and scope

Binance BTCUSDT spot klines: 1-minute bars for 2024 Q1 (131,040 rows) and 1-second bars for 2024-01-08..14
(604,800 rows), downloaded at run time from the public archive. The scan does not judge whether trading was
manipulative; scores are ranks, not probabilities.

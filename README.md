# surveillance-frontier

**How well can a surveillance policy be stress-tested within a fixed time budget, and how much does GPU parallelism
change that?** Evaluating a policy means re-running its detector over many scenarios; this project measures how far
and how finely that search can go on CPUs and GPUs, using market surveillance as the first domain.

![Python](https://img.shields.io/badge/python-3-blue)
![CUDA](https://img.shields.io/badge/CUDA-C%2B%2B%20kernels-76B900)
![Status](https://img.shields.io/badge/phase%200-complete-success)

## Question

Surveillance increasingly judges behaviour by groups of accounts that act together rather than by single accounts.
Whether such a policy holds up is a worst-case question: the largest effect it can miss, over a large space of
scenarios and policy settings. Every point in that space means re-running detection, so the answer is limited by
how many evaluations fit into the time available.

- **H1.** Under account-level policies the undetected effect grows with the number of accounts used; under
  group-level policies it is bounded.
- **H2.** The search is large enough that an optimized multi-core CPU is the bottleneck.
- **H3.** A GPU implementation with identical results covers a wider and finer search in the same time.
- **H4.** At equal wall-clock time, the GPU's estimate of the bound is closer to a high-budget reference.

## Latest result — Phase 0

<p align="center"><img src="docs/assets/phase0/sec_event_1s.png" width="100%" alt="Price, volume, and the multi-scale rarity map around the fake SEC post on 2024-01-09, 1-second bars"></p>

The fake SEC post on 2024-01-09 at 1-second resolution. Each column is a moment, each row a window length from 2 s
to 4 min; darker means the price range and the volume of that window were both rarer. The jump (21:12), the
pullback (21:17), and the drop (21:25) appear as three separate cones, out of 9 million windows scored for the week.

The pilot scanned every (start, length) window of a Bitcoin price series in this way, on a Colab Tesla T4 against
an optimized CPU baseline:

- After a correctness gate (including two deliberately broken kernels that it caught), CPU and GPU return the same
  top-20 intervals; the top three sit within 10 minutes of real news events.
- The GPU kernel is 59x faster than the optimized CPU, but **end to end the time is in ranking and data movement**,
  not the kernel. Keeping everything on the GPU and returning only candidates cuts one scan from 3.5 s to 38.5 ms.
- **Within a 60-second budget the GPU re-runs the full detector 53x more often** (6,312 vs 119 runs), and its
  estimate of the detector's sensitivity is 6.7x more precise. The precision the GPU reaches in 10 s would take the
  CPU about eight minutes.

That last point turned a speed comparison into this project's question. Full write-up: [Phase 0](docs/phase0.md).

## Phases

| Phase | Question | Status | Document |
|---|---|---|---|
| **0. Pilot** | When does a GPU help an exhaustive window scan, where does the time go, and what does a time budget buy? | Done | [Phase 0 results](docs/phase0.md) |
| 1. Feasibility | How large is the scenario space, and how long does an optimized CPU take to search it? | Next | — |

## Repository layout

```text
docs/                       phase write-ups (phase0.md), references (references.md), figures (assets/)
experiments/phase0-pilot/   pilot notebook (.py source + generated .ipynb) and results
tools/                      notebook converter, GPU-free dry-run runner, CPU stand-in for CuPy (fakecupy)
```

## Quick start

Open [`experiments/phase0-pilot/pilot.ipynb`](experiments/phase0-pilot/pilot.ipynb) in Google Colab, select a
**T4 GPU** runtime, and run all cells. Market data downloads automatically from the Binance public archive. To edit
the notebook, change the `.py` source and regenerate:

```bash
python3 tools/py2nb.py experiments/phase0-pilot/pilot.py experiments/phase0-pilot/pilot.ipynb
```

## Scope and conduct

- Results are reported as damage bounds per policy setting. The repository does not provide procedures for evading
  any real surveillance system.
- Detectors here are research reimplementations, not copies of any exchange's or regulator's system.
- No real account is labeled as a manipulator; publicly reported incidents are used only to sanity-check detectors.
- Background, data sources, and related work: [docs/references.md](docs/references.md).

## License

TBD.

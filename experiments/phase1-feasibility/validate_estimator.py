"""Estimator check on a small problem with a known answer (E1).

1. Exact rates. On a prefix of the 1-second week, sf_bench --enumerate evaluates every event position of each cell
   (length L x strength q) for all 24 policies of P24 (incremental method). The share of detected positions is the
   exact conditional detection rate p(L, q | x) of each policy; no sampling is involved.
2. Coverage. A Monte Carlo run of n trials draws n positions uniformly with replacement, as sf_bench does. Replaying
   that R times from the enumerated hits gives the real coverage of the 95% Clopper-Pearson intervals: per cell
   and policy (pointwise), and for all cells and policies at once, with and without a Bonferroni correction
   (simultaneous). Trial counts are those scenario_space.py plans for h = 0.05 and 0.02 (pointwise).
   One draw of positions serves all policies, as in a real trial, so policies are correlated within a cell.
3. Shape. Whether p increases with q for every (L, policy), and how far a +-h error in p moves the strength at
   which p crosses 50% (local slope of the exact curve).

Writes into results/<env>_<date>/: estimator_cells.csv (exact p per cell and policy), estimator.csv (coverage and
verdict), figs/fig5_estimator.png.

    python experiments/phase1-feasibility/validate_estimator.py --env local-wsl [--n 64000] [--reps 2000]
"""

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_sweep import SERIES, grid_args, physical_cores  # noqa: E402
from scenario_space import clopper_pearson, trials_needed  # noqa: E402

LENGTHS = [6, 30, 120]
STRENGTHS = [4, 8, 16, 32, 64, 128]
HALF_WIDTHS = [0.05, 0.02]
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK_2, SURFACE = "#52514e", "#fcfcfb"


def enumerate_cells(args, space, raw):
    grid = space["policy_grids"]["P24"]
    policies = [(k, s, g) for k in grid["K"] for s in grid["s_min"] for g in grid["episode_gap"]]
    hits, rows = {}, []
    for L in LENGTHS:
        for q in STRENGTHS:
            path = raw / f"hits_L{L}_q{q}.u8"
            cmd = [str(args.build / "sf_bench"), str(SERIES["1s-week"]), "--method", "incremental",
                   *grid_args(space, "P24"), "--n", str(args.n), "--mode", "across", "--threads", str(args.threads),
                   "--length", str(L), "--strength", str(q), "--enumerate", "1", "--hits-out", str(path),
                   "--warmup", "0"]
            rec = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)
            h = np.fromfile(path, np.uint8).reshape(rec["trials"], len(policies)).astype(bool)
            hits[(L, q)] = h
            for j, (k, s, g) in enumerate(policies):
                rows.append({"length": L, "strength": q, "policy": j, "K": k, "s_min": s, "episode_gap": g,
                             "positions": rec["trials"], "p": h[:, j].mean()})
            print(f"L={L:>3} q={q:>3}: {rec['trials']:,} positions in {rec['wall_s']:.1f} s, "
                  f"p (K=15, s=2, gap=60) = {h[:, policies.index((15, 2.0, 60))].mean():.3f}", flush=True)
    return pd.DataFrame(rows), hits, policies


def coverage(cells, hits, half_width, reps, rng):
    """Pointwise and simultaneous coverage of Clopper-Pearson intervals at the planned trial count."""
    m = len(cells)
    n = trials_needed(half_width, 0.05)
    exact = cells.set_index(["length", "strength", "policy"]).p
    inside, inside_b = [], []
    for key, h in hits.items():
        p = exact.loc[key].to_numpy()
        idx = rng.integers(0, h.shape[0], size=(reps, n))
        k = h[idx].sum(axis=1)                          # reps x policies
        lo, hi = clopper_pearson(k, n, 0.05)
        lo_b, hi_b = clopper_pearson(k, n, 0.05 / m)
        tol = 1e-12
        inside.append((lo <= p + tol) & (p - tol <= hi))
        inside_b.append((lo_b <= p + tol) & (p - tol <= hi_b))
    inside, inside_b = np.concatenate(inside, axis=1), np.concatenate(inside_b, axis=1)  # reps x (cells*policies)
    return {"half_width": half_width, "trials": n, "intervals": m, "pointwise_mean": inside.mean(),
            "pointwise_min": inside.mean(axis=0).min(), "simultaneous": inside.all(axis=1).mean(),
            "simultaneous_bonferroni": inside_b.all(axis=1).mean()}


def shape(cells, half_width):
    """Monotonicity in q, and the strength error that +-h in p implies near the 50% crossing."""
    viol, spans = 0, []
    for (L, j), g in cells.groupby(["length", "policy"]):
        p = g.sort_values("strength").p.to_numpy()
        q = np.sort(g.strength.to_numpy())
        viol += int((np.diff(p) < -1e-12).sum())
        i = np.flatnonzero((p[:-1] < 0.5) & (p[1:] >= 0.5))
        if len(i):
            i = i[0]
            slope = (p[i + 1] - p[i]) / np.log2(q[i + 1] / q[i])    # change in p per doubling of q
            spans.append(2 ** (half_width / slope) - 1 if slope > 0 else np.inf)
    return viol, (np.median(spans) if spans else np.nan)


def figure(cells, cov, out):
    fig, axes = plt.subplots(1, len(LENGTHS), figsize=(12, 3.8), sharey=True)
    for ax, L in zip(axes, LENGTHS):
        d = cells[cells.length == L]
        for color, (k, s) in zip(SERIES_COLORS, [(15, 1.5), (15, 2.0), (15, 3.0), (8, 2.0)]):
            g = d[(d.K == k) & (d.s_min == s) & (d.episode_gap == 60)].sort_values("strength")
            ax.plot(g.strength, g.p, "o-", color=color, label=f"K = {k}, s_min = {s}")
        ax.axhline(0.5, color=INK_2, lw=1, ls=":")
        ax.set(xscale="log", xlabel="strength q", title=f"L = {L}", xticks=STRENGTHS, xticklabels=STRENGTHS)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].set_ylabel("exact detection rate (all positions)")
    axes[0].legend(fontsize=8, loc="upper left")
    c = cov.set_index("half_width")
    fig.suptitle("Fig. 5 - Exact detection rates by enumeration; interval coverage pointwise "
                 + ", ".join(f"{c.pointwise_mean[h]:.3f}" for h in c.index) + "; all intervals at once "
                 + ", ".join(f"{c.simultaneous[h]:.3f}" for h in c.index) + " (Bonferroni "
                 + ", ".join(f"{c.simultaneous_bonferroni[h]:.3f}" for h in c.index) + ")", fontsize=10)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--build", type=Path, default=ROOT / "build")
    ap.add_argument("--space", type=Path, default=HERE / "scenario_space.json")
    ap.add_argument("--n", type=int, default=64_000, help="prefix of the 1-second week")
    ap.add_argument("--reps", type=int, default=2000, help="replayed Monte Carlo runs")
    ap.add_argument("--threads", type=int, default=physical_cores() or 1)
    ap.add_argument("--seed", type=int, default=20261007)
    args = ap.parse_args()
    space = json.loads(args.space.read_text())
    out_dir = HERE / "results" / f"{args.env}_{dt.date.today().isoformat()}"
    raw = out_dir / "raw"
    (out_dir / "figs").mkdir(parents=True, exist_ok=True)
    raw.mkdir(exist_ok=True)

    cells, hits, _ = enumerate_cells(args, space, raw)
    rng = np.random.default_rng(args.seed)
    rows = []
    for h in HALF_WIDTHS:
        c = coverage(cells, hits, h, args.reps, rng)
        c["monotone_violations"], c["strength_error_at_50pct"] = shape(cells, h)
        rows.append(c)
    cov = pd.DataFrame(rows)
    # E1: no cell's replayed coverage is below 0.95 by more than 4 replay standard errors (the minimum over hundreds
    # of noisy estimates sits well below its true value), and the Bonferroni intervals hold jointly
    cov["pointwise_floor"] = 0.95 - 4 * np.sqrt(0.95 * 0.05 / args.reps)
    cov["supported"] = (cov.pointwise_min >= cov.pointwise_floor) & (cov.simultaneous_bonferroni >= 0.95)
    cells.to_csv(out_dir / "estimator_cells.csv", index=False)
    cov.to_csv(out_dir / "estimator.csv", index=False)
    figure(cells, cov, out_dir / "figs/fig5_estimator.png")
    print(cov.to_string(index=False))


if __name__ == "__main__":
    main()

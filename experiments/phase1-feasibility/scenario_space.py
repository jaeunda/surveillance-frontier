"""Size of each Phase 1 task and its projected time-to-solution, from scenario_space.json and a measured sweep.

Intervals are Clopper-Pearson (exact binomial), whose coverage is at least the nominal level for every p; the
smoke run showed Wilson intervals undercovering near p = 0. Trials per cell = the smallest n whose interval has
half-width <= h at the worst case p = 0.5 (unless a p-map is given), at level
  pointwise     alpha = 0.05: each cell's interval holds with 95% on its own
  simultaneous  alpha = 0.05 / m over m = cells x policies (Bonferroni): all intervals hold at once, >= 95%
With a p-map (exact rates from validate_estimator.py), each cell needs only the n for its own p, at least 50, taking
the largest need over the task's policies since one trial serves them all (variance-aware allocation).

Projected time-to-solution of method M on series S = setup + sum over cells of trials / rate. The rate is the
measured batch throughput of M on S (trials per second, every policy of the task evaluated in each trial) in the
fixed configuration of the `batch` part (across mode, all physical cores, median over repetitions), taken at the
nearest measured event length. Combinations that were not measured stay empty; nothing is extrapolated.

    python experiments/phase1-feasibility/scenario_space.py [--sweep results/<env>_<date>/cpu_sweep.jsonl]
                                                             [--pmap results/<env>_<date>/estimator_cells.csv]
"""

import argparse
import json
import math
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import beta

HERE = Path(__file__).resolve().parent
METHODS = ["full", "shared", "tail", "incremental"]
MIN_TRIALS = 50


def human(seconds):
    if seconds != seconds:
        return "not measured"
    for unit, size in [("years", 365 * 86400), ("days", 86400), ("h", 3600), ("min", 60)]:
        if seconds >= size:
            return f"{seconds / size:,.1f} {unit}"
    return f"{seconds:,.1f} s"


def strengths(spec):
    if isinstance(spec, list):
        return spec
    out, q = [], float(spec["from"])
    while q <= spec["to"] * (1 + 1e-9):
        out.append(round(q, 4))
        q *= spec["ratio"]
    return out


def policy_count(space, grid):
    return math.prod(len(v) for v in space["policy_grids"][grid].values())


def alpha_for(coverage, m):
    return 0.05 if coverage == "pointwise" else 0.05 / m


def clopper_pearson(k, n, alpha):
    k, n = np.asarray(k), np.asarray(n)
    lo = np.where(k > 0, beta.ppf(alpha / 2, k, n - k + 1), 0.0)
    hi = np.where(k < n, beta.ppf(1 - alpha / 2, k + 1, n - k), 1.0)
    return lo, hi


@lru_cache(maxsize=None)
def trials_needed(h, alpha, p=0.5):
    """Smallest n whose Clopper-Pearson interval at k = round(n p) has half-width <= h."""
    def half(n):
        lo, hi = clopper_pearson(round(n * p), n, alpha)
        return (hi - lo) / 2
    lo_n, hi_n = 1, 2
    while half(hi_n) > h:
        lo_n, hi_n = hi_n, hi_n * 2
    while hi_n - lo_n > 1:
        mid = (lo_n + hi_n) // 2
        lo_n, hi_n = (lo_n, mid) if half(mid) <= h else (mid, hi_n)
    return hi_n


def tasks(space):
    """(name, factor, value, task dict): the named tasks, then one-factor ladders around base_task."""
    base = space["base_task"]
    out = [(name, "task", name, {**base, **{k: v for k, v in t.items() if not k.startswith("_")}})
           for name, t in space["tasks"].items()]
    for factor, values in space["ladders"].items():
        key = {"strength_ratio": "strengths"}.get(factor, factor)
        for v in values:
            value = {**base["strengths"], "ratio": v} if factor == "strength_ratio" else v
            label = f"{len(v)} lengths" if factor == "lengths" else (f"x{v:.3g}" if factor == "strength_ratio" else v)
            out.append((f"base, {factor} = {label}", factor, label, {**base, key: value}))
    return out


def trials_per_cell(space, task, pmap=None):
    """Trials for every (length, strength) cell of the task, in row-major order."""
    cells = [(L, q) for L in task["lengths"] for q in strengths(task["strengths"])]
    m = len(cells) * policy_count(space, task["policies"])
    alpha, h = alpha_for(task["coverage"], m), task["half_width"]
    if pmap is None:
        return cells, [trials_needed(h, alpha)] * len(cells)
    need = []
    for L, q in cells:
        # nearest measured cell in log space; the policy whose p is closest to 0.5 there needs the most trials
        d = (np.log(pmap.length / L)) ** 2 + (np.log(pmap.strength.clip(lower=0.5) / q)) ** 2
        near = pmap[d == d.min()]
        p = float(near.p.iloc[np.argmin(np.abs(near.p - 0.5))])
        need.append(max(MIN_TRIALS, trials_needed(h, alpha, round(min(p, 1 - p), 3))))
    return cells, need


def batch_rates(sweep, space):
    """Median batch throughput and setup per (method, series, grid, length) from the `batch` part."""
    b = sweep[sweep.part == "batch"].copy()
    names = {n: s for s, n in space["series_bars"].items()}
    b["series_name"] = b.n.map(names)
    return b.groupby(["method", "series_name", "grid", "length"], as_index=False)[["trials_per_s", "setup_s"]].median()


def project(cells, trials, rates, method, series, grid):
    """Projected seconds (setup, trials) for one method; NaN if that combination was not measured."""
    r = rates[(rates.method == method) & (rates.series_name == series) & (rates.grid == grid)]
    if r.empty:
        return math.nan, math.nan
    total = 0.0
    for (L, _), n in zip(cells, trials):
        row = r.iloc[np.argmin(np.abs(np.log(r.length.to_numpy() / L)))]
        total += n / row.trials_per_s
    return r.setup_s.max(), total


def table(space, sweep=None, pmap=None):
    rows = []
    for name, factor, value, task in tasks(space):
        cells, trials = trials_per_cell(space, task, pmap)
        for series in space["series_bars"]:
            row = {"task": name, "factor": factor, "value": value, "series": series,
                   "policies": policy_count(space, task["policies"]), "cells": len(cells),
                   "trials_per_cell": round(sum(trials) / len(trials)), "trials": sum(trials)}
            if sweep is not None:
                rates = batch_rates(sweep, space)
                for m in METHODS:
                    setup, run = project(cells, trials, rates, m, series, task["policies"])
                    row[f"{m}_s"] = setup + run
            rows.append(row)
    return pd.DataFrame(rows)


def validation(space, sweep):
    """Projected vs actual trial time of the validation task, per method (setup excluded on both sides)."""
    v = space["validation_task"]
    rates = batch_rates(sweep, space)
    runs = sweep[sweep.part == "validate"]
    rows = []
    for m in v["methods"]:
        got = runs[runs.method == m]
        if got.empty:
            continue
        cells = list(zip(got.length, got.strength))
        _, projected = project(cells, list(got.trials), rates, m, v["series"], v["policies"])
        actual = got.wall_s.sum()
        rows.append({"method": m, "cells": len(got), "trials": int(got.trials.sum()), "actual_s": actual,
                     "projected_s": projected, "error": (projected - actual) / actual})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", type=Path, default=HERE / "scenario_space.json")
    ap.add_argument("--sweep", type=Path, help="cpu_sweep.jsonl from run_sweep.py")
    ap.add_argument("--pmap", type=Path, help="estimator_cells.csv from validate_estimator.py")
    ap.add_argument("--out", type=Path, help="write the table as CSV")
    args = ap.parse_args()
    space = json.loads(args.space.read_text())
    sweep = pd.read_json(args.sweep, lines=True) if args.sweep else None
    pmap = pd.read_csv(args.pmap) if args.pmap else None
    t = table(space, sweep, pmap)
    shown = t.copy()
    for col in [c for c in shown if c.endswith("_s")]:
        shown[col] = [human(s) for s in shown[col]]
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.max_rows", 200):
        print(shown.to_string(index=False))
        if sweep is not None:
            print(validation(space, sweep).to_string(index=False))
    if args.out:
        t.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()

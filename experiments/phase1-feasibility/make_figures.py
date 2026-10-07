"""Figures, tables, and hypothesis verdicts for one Phase 1 run.

Reads results/<env>_<date>/cpu_sweep.jsonl (+ env.json, and estimator.csv if validate_estimator.py ran) and
scenario_space.json; writes into the same folder:
  figs/fig1_cost_vs_n.png          single-thread cost per trial vs N for each method, with its cost model (F1, F2)
  figs/fig2_stages.png             where one trial spends its time, per method (N = 604,800, K = 15)
  figs/fig3_thread_scaling.png     across-mode speedup vs threads per method, with the F3 line (F3)
  figs/fig4_time_to_solution.png   projected time-to-solution of T1 and T2 per method and series (F4 checked)
  tables.md                        the same numbers as Markdown tables
  hypotheses.csv                   gates, F1-F4, E1: measured value, criterion, verdict

    python experiments/phase1-feasibility/make_figures.py experiments/phase1-feasibility/results/<env>_<date>
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from scenario_space import human, table, validation  # noqa: E402

# Categorical slots in fixed order (validated light palette), ink and grid tokens.
COLORS = {"full": "#eb6834", "shared": "#eda100", "tail": "#2a78d6", "incremental": "#1baf7a"}
STAGES = [("scan_ms_median", "scan / update", "#2a78d6"), ("rank_ms_median", "rank", "#eb6834"),
          ("select_ms_median", "select", "#1baf7a")]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
FIT_MAX_N, BIG_N = 64_000, 604_800
HOUR, DAY = 3600, 86400
# cost models per method (columns of the design matrix); see the Phase 1 README
MODELS = {"full": lambda n, k: [np.ones_like(n), n * k, n * k * np.log2(n)],
          "tail": lambda n, k: [np.ones_like(n), n * k],
          "incremental": lambda n, k: [np.ones_like(n), n * k]}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "lines.linewidth": 2, "lines.markersize": 6, "legend.frameon": False,
})


def load(run_dir):
    sweep = pd.read_json(run_dir / "cpu_sweep.jsonl", lines=True)
    sweep["k"] = sweep.k_set.map(max)
    env = json.loads((run_dir / "env.json").read_text())
    est = pd.read_csv(run_dir / "estimator.csv") if (run_dir / "estimator.csv").exists() else None
    return sweep, env, est


def cost_table(sweep):
    """Single-thread cost per (method, n, k): median over repetitions of the per-run medians."""
    c = sweep[sweep.part == "cost"]
    return c.groupby(["method", "n", "k"], as_index=False)[["trial_ms_median", "scan_ms_median", "rank_ms_median",
                                                            "select_ms_median", "trials_per_s"]].median()


def design(method, n, k):
    n, k = np.asarray(n, float), np.asarray(k, float)
    return np.column_stack(MODELS[method](n, k))


def fit(cost):
    coefs = {}
    for m, g in cost.groupby("method"):
        train = g[g.n <= FIT_MAX_N]
        coefs[m], *_ = np.linalg.lstsq(design(m, train.n, train.k), train.trial_ms_median, rcond=None)
    return coefs


def scaling_table(sweep):
    s = sweep[(sweep.part == "scaling")]
    return s.groupby(["method", "mode", "n", "threads"], as_index=False)[
        ["trials_per_s", "trial_ms_median", "scan_ms_median", "rank_ms_median", "select_ms_median"]].median()


# ---------- figures ----------

def fig_cost(cost, coefs, out):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    g15 = cost[cost.k == cost.k.max()]
    grid_n = np.geomspace(g15.n.min(), g15.n.max(), 100)
    for m, g in g15.groupby("method"):
        g = g.sort_values("n")
        ax.plot(g.n, g.trial_ms_median, "o-", color=COLORS[m], label=m)
        ax.plot(grid_n, design(m, grid_n, np.full_like(grid_n, g.k.iloc[0])) @ coefs[m], "--", color=COLORS[m],
                lw=1, alpha=0.7)
        ax.annotate(f"{g.trial_ms_median.iloc[-1]:,.1f} ms", (g.n.iloc[-1], g.trial_ms_median.iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK_2)
    ax.axvspan(g15.n.min() * 0.9, FIT_MAX_N, color=GRID, alpha=0.5, lw=0)
    ax.set(xscale="log", yscale="log", xlabel="series length N (bars)", ylabel="ms per trial (1 thread, 1 policy)",
           xticks=sorted(g15.n.unique()), xticklabels=[f"{n / 1000:,.0f}k" for n in sorted(g15.n.unique())],
           title=f"Fig. 1 - Cost of one trial by method, K = {g15.k.max()} (dashed: cost model fitted in grey)")
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_stages(cost, out):
    g = cost[(cost.n == cost.n.max()) & (cost.k == cost.k.max())].set_index("method")
    g = g.loc[[m for m in COLORS if m in g.index]]
    fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(g) + 1.6))
    total = g[[c for c, *_ in STAGES]].sum(axis=1)
    left = np.zeros(len(g))
    for col, name, color in STAGES:
        share = (g[col] / total).to_numpy()
        ax.barh(g.index, share, left=left, color=color, edgecolor=SURFACE, linewidth=2, height=0.6, label=name)
        left += share
    for y, t in enumerate(total):
        ax.text(1.01, y, f"{t:,.1f} ms", va="center", fontsize=9, color=INK)
    ax.set(xlim=(0, 1.15), xticks=[0, 0.25, 0.5, 0.75, 1], xlabel="share of one trial (1 thread)",
           title=f"Fig. 2 - Where one trial spends its time (N = {cost.n.max():,}, K = {cost.k.max()})")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_scaling(sc, env, out):
    ns = sorted(sc.n.unique())
    phys = env.get("physical_cores_used") or env.get("physical_cores")
    fig, axes = plt.subplots(1, len(ns), figsize=(5.5 * len(ns), 4), sharey=True, squeeze=False)
    for ax, n in zip(axes[0], ns):
        d = sc[sc.n == n]
        t = np.sort(d.threads.unique())
        ax.plot(t, t, ":", color=INK_2, lw=1, label="ideal")
        ax.plot(t, 0.7 * t, ":", color=INK_2, lw=1, alpha=0.5)
        for (m, mode), g in d.groupby(["method", "mode"]):
            g = g.sort_values("threads")
            base = g[g.threads == 1].trials_per_s.iloc[0]
            ax.plot(g.threads, g.trials_per_s / base, "o-" if mode == "across" else "s--", color=COLORS[m],
                    label=f"{m}, {mode}", alpha=1 if mode == "across" else 0.6)
        if phys:
            ax.axvline(phys, color=INK_2, lw=1, ls="--")
            ax.text(phys, ax.get_ylim()[1] * 0.98, " physical cores", fontsize=8, color=INK_2, va="top")
        ax.set(xlabel="threads", title=f"N = {n:,}, K = 15")
    axes[0][0].set_ylabel("throughput vs 1 thread")
    axes[0][-1].legend(loc="upper left", fontsize=8)
    fig.suptitle("Fig. 3 - Parallel scaling per method (dotted: ideal and 0.7 x ideal, the F3 line)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_tts(tts, val, env, out):
    t = tts[tts.factor == "task"]
    labels = [f"{r.task} · {r.series}" for r in t.itertuples()][::-1]
    methods = [m for m in COLORS if f"{m}_s" in t]
    y = np.arange(len(t))
    h = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(8.5, 0.9 * len(t) + 2))
    for i, m in enumerate(methods):
        vals = t[f"{m}_s"].to_numpy()[::-1]
        pos = y + (len(methods) / 2 - i - 0.5) * h
        ax.barh(pos, np.nan_to_num(vals, nan=0), height=h, color=COLORS[m], edgecolor=SURFACE, lw=1, label=m)
        for yy, v in zip(pos, vals):
            ax.text(v if v == v else ax.get_xlim()[0], yy, f"  {human(v)}", va="center", fontsize=8, color=INK)
    for x, name in [(HOUR, "1 hour"), (DAY, "1 day")]:
        ax.axvline(x, color=INK_2, lw=1, ls="--")
        ax.text(x, len(t) - 0.4, f" {name}", fontsize=8, color=INK_2)
    ax.set_yticks(y, labels)
    note = "" if val is None or val.dropna().empty else "\nprojection error on the validation task: " + ", ".join(
        f"{r.method} {100 * r.error:+.0f}%" for r in val.dropna().itertuples())
    ax.set(xscale="log", xlabel=f"projected time-to-solution, all physical cores ({env.get('cpu', 'CPU')})",
           title="Fig. 4 - Time-to-solution per task and method" + note)
    ax.title.set_fontsize(10)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=len(methods), fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------- hypotheses and tables ----------

def verdicts(sweep, cost, coefs, sc, env, val, est):
    rows = []
    v = sweep[sweep.part == "verify"]
    rows.append(("G", f"{int(v.mismatches.sum())} mismatching trials in {int(v.trials.sum())} verified trials",
                 "0 (real data, all methods vs full, 24 policies)", bool(len(v)) and v.mismatches.sum() == 0))
    # F1: full recomputation is not the CPU floor
    big = cost[(cost.n == cost.n.max()) & (cost.k == cost.k.max())].set_index("method").trials_per_s
    best = big.drop("full").idxmax()
    gain = big[best] / big["full"]
    rows.append(("F1", f"{best} / full at N = {cost.n.max():,}, K = 15: {gain:,.0f}x", ">= 10x", gain >= 10))
    # F2: each method's cost model, fitted on N <= 64,000, predicts the largest N
    errs = {}
    for m, g in cost.groupby("method"):
        b = g[g.n == g.n.max()]
        errs[m] = (np.abs(design(m, b.n, b.k) @ coefs[m] - b.trial_ms_median) / b.trial_ms_median).max()
    rows.append(("F2", ", ".join(f"{m} {100 * e:.0f}%" for m, e in errs.items()),
                 f"< 20% at N = {cost.n.max():,} for every K and method", max(errs.values()) < 0.20))
    # F3: across-mode efficiency at the physical core count
    phys = env.get("physical_cores_used") or env.get("physical_cores")
    a = sc[sc["mode"] == "across"]
    eff = {}
    for (m, n), g in a.groupby(["method", "n"]):
        one, top = g[g.threads == 1].trials_per_s, g[g.threads == phys].trials_per_s
        if len(one) and len(top):
            eff[(m, n)] = top.iloc[0] / (phys * one.iloc[0])
    worst = min(eff, key=eff.get) if eff else None
    rows.append(("F3", f"lowest efficiency at {phys} threads: {eff[worst]:.2f} ({worst[0]}, N = {worst[1]:,})"
                 if worst else "not measured", ">= 0.7 for every method and N", bool(eff) and min(eff.values()) >= 0.7))
    # F4: projection vs a real end-to-end run
    if val is not None and not val.empty:
        rows.append(("F4", ", ".join(f"{r.method} " + (f"{100 * r.error:+.1f}%" if r.error == r.error else "no batch rate")
                                     for r in val.itertuples()),
                     "|error| <= 15% for every method", bool((val.error.abs() <= 0.15).all())))
    else:
        rows.append(("F4", "validation task not run", "|error| <= 15% for every method", False))
    if est is not None:
        rows.append(("E1", "; ".join(f"h = {r.half_width}: pointwise min {r.pointwise_min:.3f}, all at once "
                                     f"{r.simultaneous:.3f}, Bonferroni {r.simultaneous_bonferroni:.3f}"
                                     for r in est.itertuples()),
                     f"pointwise >= {est.pointwise_floor.iloc[0]:.3f} (0.95 - 4 replay SE) in every cell, Bonferroni joint >= 0.95", bool(est.supported.all())))
    return pd.DataFrame(rows, columns=["id", "measured", "criterion", "supported"])


def md_table(df, floatfmt=".2f"):
    return df.to_markdown(index=False, floatfmt=floatfmt)


def write_tables(run_dir, env, cost, coefs, sc, tts, val, hyp):
    per_n = cost[cost.k == cost.k.max()].pivot_table(index="n", columns="method", values="trial_ms_median")
    per_n.columns = [f"{m} (ms)" for m in per_n.columns]
    stages = cost[(cost.n == cost.n.max()) & (cost.k == cost.k.max())][
        ["method", "scan_ms_median", "rank_ms_median", "select_ms_median", "trial_ms_median"]]
    phys = env.get("physical_cores_used") or env.get("physical_cores")
    inflation = []  # how much each stage slows down per trial when all physical cores run trials
    for (m, n), g in sc[sc["mode"] == "across"].groupby(["method", "n"]):
        one, top = g[g.threads == 1], g[g.threads == phys]
        if len(one) and len(top):
            inflation.append({"method": m, "n": n, **{name: top[c].iloc[0] / max(one[c].iloc[0], 1e-9)
                                                      for c, name, _ in STAGES}})
    shown = tts.copy()
    for col in [c for c in shown if c.endswith("_s")]:
        shown[col] = [human(s) for s in shown[col]]
    models = "; ".join(f"{m}: " + " + ".join(f"{c:.3g}·{t}" for c, t in zip(
        coefs[m], ["1", "N·K", "N·K·log2 N"])) for m in coefs)
    text = [
        f"# Phase 1 results - {run_dir.name}", "",
        f"CPU: {env.get('cpu')} ({env.get('physical_cores')} physical / {env.get('logical_cores')} logical cores), "
        f"compiler: {env.get('compiler')}, commit {env.get('git_commit')}{' (dirty)' if env.get('git_dirty') else ''}, "
        f"OMP_PLACES={env.get('omp_places')}, OMP_PROC_BIND={env.get('omp_proc_bind')}, {env.get('reps')} repetitions.",
        "", "## Gates and hypotheses", "", md_table(hyp),
        "", "## Single-thread ms per trial, K = 15, one policy", "", md_table(per_n.reset_index().astype({"n": str}), ".2f"),
        "", f"Cost models (ms), fitted on N <= {FIT_MAX_N:,}: {models}",
        "", f"## Stage times at N = {cost.n.max():,}, K = 15 (1 thread, ms)", "", md_table(stages, ".2f"),
        "", f"## Stage slowdown per trial at {phys} threads vs 1 (across mode)", "",
        md_table(pd.DataFrame(inflation), ".2f") if inflation else "not measured",
        "", "## Time-to-solution per task (projected from measured batch throughput)", "", md_table(shown, ".3g"),
        "", "## Projection check on the validation task", "",
        md_table(val, ".3f") if val is not None and not val.empty else "not run", "",
    ]
    (run_dir / "tables.md").write_text("\n".join(text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--space", type=Path, default=HERE / "scenario_space.json")
    args = ap.parse_args()
    sweep, env, est = load(args.run_dir)
    space = json.loads(args.space.read_text())
    figs = args.run_dir / "figs"
    figs.mkdir(exist_ok=True)

    cost = cost_table(sweep)
    coefs = fit(cost)
    sc = scaling_table(sweep)
    tts = table(space, sweep)
    val = validation(space, sweep)
    hyp = verdicts(sweep, cost, coefs, sc, env, val, est)

    fig_cost(cost, coefs, figs / "fig1_cost_vs_n.png")
    fig_stages(cost, figs / "fig2_stages.png")
    fig_scaling(sc, env, figs / "fig3_thread_scaling.png")
    fig_tts(tts, val, env, figs / "fig4_time_to_solution.png")
    hyp.to_csv(args.run_dir / "hypotheses.csv", index=False)
    write_tables(args.run_dir, env, cost, coefs, sc, tts, val, hyp)
    print(hyp.to_string(index=False))
    print(f"wrote {figs}/, tables.md, hypotheses.csv")


if __name__ == "__main__":
    main()

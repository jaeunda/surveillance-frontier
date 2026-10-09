"""Figures for docs/phase1.md and the README, from one Phase 1 run (read only; the run folder is not changed).

Draws the time-to-solution summary against the H2 decision line, labels the stage and estimator plots with their
audited scope, and copies the cost/scaling plots. Historical result files are never changed.

Usage (from the repository root):
    python experiments/phase1-feasibility/make_doc_figures.py \
        experiments/phase1-feasibility/results/aws-c7i-16xlarge_2026-10-09

Writes five figures under docs/assets/phase1/.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "docs" / "assets" / "phase1"
sys.path.insert(0, str(HERE))
from make_figures import COLORS, GRID, INK, INK_2, SURFACE, HOUR, STAGES, cost_table  # noqa: E402
from scenario_space import human, table  # noqa: E402

# full and shared are close in hue, so every method also has its own marker and a direct label
MARKERS = {"full": "s", "shared": "D", "tail": "^", "incremental": "o"}
COPIES = {"fig1_cost_vs_n.png": "cost_vs_n.png", "fig3_thread_scaling.png": "thread_scaling.png"}
SERIES = {"1m-quarter": "1-min bars, 2024 Q1", "1s-week": "1-s bars, one week"}
TASKS = {"T1-curve": "T1  one policy, pointwise", "T2-policy-map": "T2  24 policies, simultaneous"}


def fig_time_to_solution(tts, env, out):
    t = tts[tts.factor == "task"].reset_index(drop=True)
    methods = [m for m in COLORS if f"{m}_s" in t]
    y = np.arange(len(t))[::-1]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.axvspan(HOUR, 1e7, color=GRID, alpha=0.45, lw=0)
    ax.axvline(HOUR, color=INK_2, lw=1, ls="--")
    ax.text(HOUR * 1.3, len(t) - 0.45, "H2 rule applies to the fastest method", fontsize=8.5, color=INK_2, va="center")
    ax.text(HOUR / 1.3, len(t) - 0.45, "under 1 hour", fontsize=8.5, color=INK_2, va="center", ha="right")
    seen = set()
    for yy, r in zip(y, t.itertuples()):
        vals = {m: getattr(r, f"{m}_s") for m in methods if getattr(r, f"{m}_s") == getattr(r, f"{m}_s")}
        ax.plot([min(vals.values()), max(vals.values())], [yy, yy], color=INK_2, lw=1, zorder=1)
        for m, v in vals.items():
            ax.scatter(v, yy, s=70, marker=MARKERS[m], color=COLORS[m], edgecolor=SURFACE, lw=1.5, zorder=3,
                       label=None if m in seen else m)
            seen.add(m)
        best = vals["incremental"]
        ax.text(best / 1.25, yy, human(best), ha="right", va="center", fontsize=9, color=INK, fontweight="bold")
        ax.text(vals["full"] * 1.25, yy, f"{human(vals['full'])} (full)", ha="left", va="center", fontsize=8.5,
                color=INK_2)
    ax.set_yticks(y, [f"{TASKS[r.task]}\n{SERIES[r.series]}" for r in t.itertuples()], fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(1, 1e7)
    ax.set_ylim(-0.6, len(t) - 0.1)
    ax.grid(axis="y", visible=False)
    ticks = [1, 60, HOUR, 86400, 7 * 86400]
    ax.set_xticks(ticks, ["1 s", "1 min", "1 h", "1 day", "1 week"])
    ax.minorticks_off()
    ax.set_xlabel(f"projected time-to-solution on {env.get('physical_cores_used')} physical cores "
                  f"({env.get('cpu', 'CPU')}), log scale", fontsize=9)
    ax.set_title("Best measured exact CPU method: all task projections below 1 hour", loc="left",
                 fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(m) for m in methods if m in labels]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center",
              bbox_to_anchor=(0.5, -0.2), ncol=len(order), fontsize=9, handletextpad=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_stages(sweep, out):
    c = sweep.assign(k=sweep.k_set.map(max))
    c = cost_table(c)
    g = c[(c.n == 604800) & (c.k == 15)].set_index("method").loc[["full", "tail", "incremental"]]
    total = g[[col for col, *_ in STAGES]].sum(axis=1)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    left = np.zeros(len(g))
    for col, label, color in STAGES:
        share = (g[col] / total).to_numpy()
        ax.barh(g.index, share, left=left, color=color, edgecolor=SURFACE, height=0.6, label=label)
        left += share
    for y, value in enumerate(total):
        ax.text(1.01, y, f"{value:,.2f} ms", va="center", fontsize=9)
    ax.set(xlim=(0, 1.19), xticks=[0, .25, .5, .75, 1],
           xlabel="share of summed stage medians (1 thread)",
           title="Stage medians: N = 604,800, K = 15, L = 30, q = 16")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.26), ncol=3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_estimator(cells, out):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    strengths = sorted(cells.strength.unique())
    for ax, length in zip(axes, [6, 30, 120]):
        for color, (k, s) in zip(COLORS.values(), [(15, 1.5), (15, 2), (15, 3), (8, 2)]):
            g = cells[(cells.length == length) & (cells.K == k) & (cells.s_min == s)
                      & (cells.episode_gap == 60)].sort_values("strength")
            ax.plot(g.strength, g.p, "o-", color=color, label=f"K = {k}, s_min = {s:g}")
        ax.axhline(.5, color=INK_2, lw=1, ls=":")
        ax.set(xscale="log", xlabel="strength q", title=f"L = {length}", xticks=strengths,
               xticklabels=[str(q) for q in strengths])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].set_ylabel("exact conditional new-episode rate")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Enumeration on the 64,000-bar prefix; four policies shown, all with gap = 60", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--space", type=Path, default=HERE / "scenario_space.json")
    args = ap.parse_args()
    sweep = pd.read_json(args.run_dir / "cpu_sweep.jsonl", lines=True)
    env = json.loads((args.run_dir / "env.json").read_text())
    space = json.loads(args.space.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    fig_time_to_solution(table(space, sweep), env, OUT / "time_to_solution.png")
    fig_stages(sweep, OUT / "stages.png")
    fig_estimator(pd.read_csv(args.run_dir / "estimator_cells.csv"), OUT / "estimator.png")
    for src, dst in COPIES.items():
        shutil.copyfile(args.run_dir / "figs" / src, OUT / dst)
    print(f"wrote {OUT}/")


if __name__ == "__main__":
    main()

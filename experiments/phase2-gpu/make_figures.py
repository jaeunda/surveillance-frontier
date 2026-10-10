"""Figures from stored summaries only (nothing is recomputed here).

    make_figures.py formal LABEL   from results/formal/LABEL/verdicts_stage-d.json (analyze.py stage-d)
    make_figures.py pilot LABEL    from results/pilot/LABEL/pilot-summary.json (pilot.py summary)

Formal:
  figs/fig1_h3.png   time ratio T_CPU / T_GPU per condition: per-pair values, the 90% interval, and the
                     equivalence band 0.80-1.25 (> 1 means the GPU reached the same result sooner)
  figs/fig2_h4.png   RMSE against the exact reference vs wall-clock budget, per device (per-pair lines)
  figs/fig3_d1.png   D1: predicted vs observed per-trial cost per device; held-out points filled
Pilot:
  figs/pilot_microbatch.png   steady-state rate vs GPU microbatch at the representative point, one line per path
                              (allocation failures are marked on the axis; memory is in pilot-summary.md)

"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

# Categorical slots in fixed order (validated light palette, as Phase 1), ink and grid tokens.
DEVICE = {"gpu": "#2a78d6", "cpu": "#eb6834"}
INK, INK_2, GRID, SURFACE, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#ecebe7"
DELTA = 1.25


def plain(ax, axis):
    f = matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}")
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_formatter(f)
    a.set_minor_formatter(matplotlib.ticker.NullFormatter())


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_h3(v, out):
    rows = [r for r in v["H3"] if r.get("per_pair_ratio")] if isinstance(v["H3"], list) else []
    if not rows:
        return False
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(rows) + 1.4), facecolor=SURFACE)
    style(ax)
    ax.axvspan(1 / DELTA, DELTA, color=BAND, zorder=0)
    ax.axvline(1, color=INK_2, linewidth=1)
    for y, r in enumerate(rows):
        ax.scatter(r["per_pair_ratio"], [y] * len(r["per_pair_ratio"]), s=14, color=DEVICE["gpu"], alpha=0.45,
                   linewidths=0, zorder=2)
        if r.get("rho"):
            ax.plot([r["lo"], r["hi"]], [y, y], color=DEVICE["gpu"], linewidth=2, solid_capstyle="round", zorder=3)
            ax.scatter([r["rho"]], [y], s=64, color=DEVICE["gpu"], edgecolors=SURFACE, linewidths=2, zorder=4)
        ax.text(1.02, y, r["category"] or r["status"], transform=ax.get_yaxis_transform(), va="center", fontsize=8,
                color=INK_2)
    ax.set_xscale("log")
    ax.set_xticks([t for t in (0.25, 0.5, 0.8, 1, 1.25, 2, 4, 8) if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]])
    plain(ax, "x")
    ax.set_yticks(range(len(rows)), [r["condition"] for r in rows], fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("T_CPU / T_GPU (log scale; > 1: GPU sooner; band: within 1.25x)", fontsize=8, color=INK_2)
    ax.set_title(f"H3: time to the same result, m = {v['m']} launch pairs, 90% intervals", fontsize=10, color=INK,
                 loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return True


def fig_h4(v, out):
    rows = v["H4"] if isinstance(v["H4"], list) else []
    if not rows:
        return False
    fig, ax = plt.subplots(figsize=(6, 4), facecolor=SURFACE)
    style(ax)
    budgets = [r["budget_s"] for r in rows]
    for dev in ("cpu", "gpu"):
        pairs = sorted({k.split("-")[0] for r in rows for k in r["per_device"] if k.endswith(dev)})
        for p in pairs:
            ys = [r["per_device"].get(f"{p}-{dev}", {}).get("rmse") for r in rows]
            ax.plot(budgets, ys, color=DEVICE[dev], linewidth=2, marker="o", markersize=5,
                    markeredgecolor=SURFACE, alpha=0.8)
        last = [r["per_device"].get(f"{pairs[0]}-{dev}", {}).get("rmse") for r in rows][-1] if pairs else None
        if last:
            ax.annotate(dev.upper(), (budgets[-1], last), xytext=(6, 0), textcoords="offset points", fontsize=8,
                        color=INK, va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(budgets, [f"{b:g} s" for b in budgets])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    plain(ax, "y")
    ax.set_xlabel("wall-clock budget B (cold process, setup included)", fontsize=8, color=INK_2)
    ax.set_ylabel("RMSE over 360 rates vs exact reference", fontsize=8, color=INK_2)
    ax.set_title("H4: error at equal wall-clock (one line per launch)", fontsize=10, color=INK, loc="left")
    ax.legend(handles=[plt.Line2D([], [], color=DEVICE[d], linewidth=2, marker="o", label=d.upper()) for d in DEVICE],
              frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return True


def fig_d1(v, out):
    devs = [d for d in ("cpu", "gpu") if v["D1"][d].get("points")]
    if not devs:
        return False
    fig, axes = plt.subplots(1, len(devs), figsize=(4.2 * len(devs), 4), facecolor=SURFACE, squeeze=False)
    for ax, dev in zip(axes[0], devs):
        style(ax)
        pts = v["D1"][dev]["points"]
        obs = np.array([p["t_obs"] for p in pts]) * 1e3
        pred = np.array([p["t_pred"] for p in pts]) * 1e3
        held = np.array([p["heldout"] for p in pts])
        lim = [min(obs.min(), pred.min()) * 0.8, max(obs.max(), pred.max()) * 1.25]
        ax.plot(lim, lim, color=INK_2, linewidth=1)
        ax.scatter(obs[~held], pred[~held], s=36, facecolors="none", edgecolors=DEVICE[dev], linewidths=1.5,
                   label="calibration")
        ax.scatter(obs[held], pred[held], s=36, color=DEVICE[dev], edgecolors=SURFACE, linewidths=1.5,
                   label="held out")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        plain(ax, "x")
        plain(ax, "y")
        d = v["D1"][dev]
        ax.set_title(f"{dev.upper()}: median |err| {d['median_abs_rel_err']:.2f}, max {d['max_abs_rel_err']:.2f}",
                     fontsize=9, color=INK, loc="left")
        ax.set_xlabel("observed ms per trial", fontsize=8, color=INK_2)
        ax.set_ylabel("predicted ms per trial", fontsize=8, color=INK_2)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return True


PATH = {"inc": DEVICE["gpu"], "hyb": DEVICE["cpu"]}


def fig_microbatch(summary, out):
    rows = [m for m in summary.get("microbatch", [])]
    if not rows:
        return False
    fig, ax = plt.subplots(figsize=(6, 4), facecolor=SURFACE)
    style(ax)
    for path in ("inc", "hyb"):
        pts = sorted((m["microbatch"], (m["rate"] or {}).get("median")) for m in rows if m["path"] == path)
        ok = [(b, r) for b, r in pts if r]
        if ok:
            ax.plot([b for b, _ in ok], [r for _, r in ok], color=PATH[path], linewidth=2, marker="o", markersize=6,
                    markeredgecolor=SURFACE, label=f"G-{path}")
            ax.annotate(f"G-{path}", ok[-1], xytext=(6, 0), textcoords="offset points", fontsize=8, color=INK,
                        va="center")
        failed = [b for b, r in pts if not r]
        if failed:
            ax.scatter(failed, [ax.get_ylim()[0]] * len(failed), marker="x", color=PATH[path], s=40, clip_on=False)
    ax.set_xscale("log", base=2)
    plain(ax, "x")
    ax.set_xlabel("microbatch (trials resident on the device)", fontsize=8, color=INK_2)
    ax.set_ylabel("trials per second (median of repetitions)", fontsize=8, color=INK_2)
    ax.set_title(f"Pilot {summary['label']} ({summary['machine']}): rate vs microbatch", fontsize=10, color=INK,
                 loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["formal", "pilot"])
    ap.add_argument("label")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent / "results" / args.kind / args.label
    figs = root / "figs"
    figs.mkdir(exist_ok=True)
    if args.kind == "pilot":
        s = json.loads((root / "pilot-summary.json").read_text())
        print(f"pilot_microbatch.png: {'written' if fig_microbatch(s, figs / 'pilot_microbatch.png') else 'no data'}")
        return
    v = json.loads((root / "verdicts_stage-d.json").read_text())
    for name, fn in [("fig1_h3.png", fig_h3), ("fig2_h4.png", fig_h4), ("fig3_d1.png", fig_d1)]:
        print(f"{name}: {'written' if fn(v, figs / name) else 'no data'}")


if __name__ == "__main__":
    main()

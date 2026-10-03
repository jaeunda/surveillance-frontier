"""Redraw the pilot's rarity-map figures for the documentation, with aligned time axes.

Recomputes the same rank-based rarity as the notebook (Sections 5 and 7) on the CPU. Ranks are integers and the
correctness gate showed CPU and GPU counts are identical, so the map equals the one computed on the GPU in the run.

Usage (from the repository root):
    python3 experiments/phase0-pilot/make_figures.py [cache_dir]

Writes docs/assets/phase0/sec_event_1s.png and docs/assets/phase0/sec_event_1m.png.
Dependencies: numpy, numba, pandas, matplotlib, requests (see requirements.txt).
"""
import io
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.colors import LinearSegmentedColormap
from numba import njit

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "phase0"
CACHE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "binance"
ARCHIVE = "https://data.binance.vision/data/spot/{freq}/klines/BTCUSDT/{iv}/BTCUSDT-{iv}-{period}.zip"
WINDOWS = np.array([2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 91, 128, 181, 256], dtype=np.int32)
EVENT = pd.Timestamp("2024-01-09 21:11", tz="UTC")

# Sequential ramp (one hue, light -> dark): light = common interval, dark = rare interval.
RAMP = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("rarity", RAMP)
INK, INK_MUTED, GRID = "#1f1f1e", "#6b6a66", "#e6e5e1"
S_TICKS = [0, 1, 2, 3, 4, 5]
S_LABELS = ["all", "top 10%", "top 1%", "top 0.1%", "top 0.01%", "top 0.001%"]


def load(iv, periods, freq):
    frames = []
    for period in periods:
        path = CACHE / f"BTCUSDT-{iv}-{period}.zip"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            r = requests.get(ARCHIVE.format(freq=freq, iv=iv, period=period), timeout=120)
            r.raise_for_status()
            path.write_bytes(r.content)
        with zipfile.ZipFile(path) as zf:
            frames.append(pd.read_csv(zf.open(zf.namelist()[0]), header=None, usecols=[0, 2, 3, 4, 5],
                                      names=["t", "high", "low", "close", "volume"]))
    df = pd.concat(frames, ignore_index=True)
    df = df[pd.to_numeric(df.t, errors="coerce").notna()].astype(float)
    t = df.t.astype(np.int64)
    df["ts"] = pd.to_datetime(t, unit="us" if t.iloc[0] > 10**14 else "ms", utc=True)
    return df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


@njit(cache=False)
def features(low, high, volume, windows):
    """Range (max-min)/min and volume sum for every (length, start): monotonic deques + float64 prefix sum."""
    n, K = low.size, windows.size
    rng = np.full((K, n), np.nan, np.float32)
    vol = np.full((K, n), np.nan, np.float32)
    pre = np.zeros(n + 1)
    for i in range(n):
        pre[i + 1] = pre[i] + volume[i]
    qmin = np.empty(n, np.int64)
    qmax = np.empty(n, np.int64)
    for k in range(K):
        w = windows[k]
        h1 = t1 = h2 = t2 = 0
        for i in range(n):
            while t1 > h1 and low[qmin[t1 - 1]] >= low[i]:
                t1 -= 1
            qmin[t1] = i; t1 += 1
            while t2 > h2 and high[qmax[t2 - 1]] <= high[i]:
                t2 -= 1
            qmax[t2] = i; t2 += 1
            if qmin[h1] <= i - w:
                h1 += 1
            if qmax[h2] <= i - w:
                h2 += 1
            s = i - w + 1
            if s >= 0:
                mn, mx = low[qmin[h1]], high[qmax[h2]]
                rng[k, s] = np.float32((mx - mn) / mn)
                vol[k, s] = np.float32(pre[s + w] - pre[s])
    return rng, vol


def rarity(df):
    low, high, vol = (np.ascontiguousarray(df[c].to_numpy(np.float32)) for c in ["low", "high", "volume"])
    r, v = features(low, high, vol, WINDOWS)
    n = low.size
    S = np.full((len(WINDOWS), n), np.nan)
    for k, w in enumerate(WINDOWS):
        nk = n - int(w) + 1
        cr = nk - np.searchsorted(np.sort(r[k, :nk]), r[k, :nk], side="left")
        cv = nk - np.searchsorted(np.sort(v[k, :nk]), v[k, :nk], side="left")
        s_k = -np.log10(np.maximum(cr, cv) / nk)
        h = int(w) // 2                                   # plot at the window centre
        S[k, h:h + nk] = s_k[: n - h]
    return S


def draw(df, S, lo, hi, unit, title, path, bar_width):
    t = df.ts.dt.tz_convert(None).to_numpy()[lo:hi]
    fig = plt.figure(figsize=(12, 7.2), dpi=150)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.25, 0.55, 1.6], width_ratios=[60, 1], hspace=0.08, wspace=0.02)
    ax_p, ax_v, ax_m = (fig.add_subplot(gs[i, 0]) for i in range(3))
    cax = fig.add_subplot(gs[2, 1])
    for ax in (ax_p, ax_v):
        ax.sharex(ax_m)
        ax.tick_params(labelbottom=False)
    ax_p.fill_between(t, df.low[lo:hi], df.high[lo:hi], color="#9ec5f4", lw=0)
    ax_p.plot(t, df.close[lo:hi], color=INK, lw=1.0)
    ax_v.bar(t, df.volume[lo:hi], width=bar_width, color=INK_MUTED, lw=0)
    mesh = ax_m.pcolormesh(t, np.arange(len(WINDOWS)), S[:, lo:hi], shading="nearest", cmap=CMAP, vmin=0, vmax=5,
                           rasterized=True)
    ax_m.set_yticks(np.arange(len(WINDOWS))[::2], [str(w) for w in WINDOWS[::2]])
    cb = fig.colorbar(mesh, cax=cax)
    cb.set_ticks(S_TICKS, labels=S_LABELS)
    cb.outline.set_visible(False)
    ev = EVENT.tz_convert(None)
    for ax in (ax_p, ax_v, ax_m):
        ax.axvline(ev, color=INK, lw=1.0, ls=(0, (4, 3)))
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=9)
    for ax in (ax_p, ax_v):
        ax.grid(axis="y", color=GRID, lw=0.8)
    ax_p.annotate("fake SEC post, 21:11 UTC", (ev, 1.0), xycoords=("data", "axes fraction"), xytext=(-6, -4),
                  textcoords="offset points", va="top", ha="right", fontsize=9, color=INK)
    ax_p.set_ylabel("price (USDT)", color=INK_MUTED, fontsize=9)
    ax_v.set_ylabel("volume (BTC)", color=INK_MUTED, fontsize=9)
    ax_m.set_ylabel(f"window length ({unit})", color=INK_MUTED, fontsize=9)
    ax_m.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.suptitle(title, x=0.125, y=0.915, ha="left", fontsize=12, color=INK)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s1 = load("1s", [f"2024-01-{d:02d}" for d in range(8, 15)], "daily")
    i = int((s1.ts - EVENT).abs().idxmin())
    draw(s1, rarity(s1), i - 1200, i + 1200, "s",
         "Every (start, length) window of one week of 1-second bars scored for rarity (604,800 bars × 15 lengths)",
         OUT / "sec_event_1s.png", 1 / 86400)
    m1 = load("1m", ["2024-01", "2024-02", "2024-03"], "monthly")
    i = int((m1.ts - EVENT).abs().idxmin())
    draw(m1, rarity(m1), i - 180, i + 180, "min",
         "The same event on 1-minute bars of 2024 Q1 (131,040 bars × 15 lengths)", OUT / "sec_event_1m.png", 1 / 1440)


if __name__ == "__main__":
    main()

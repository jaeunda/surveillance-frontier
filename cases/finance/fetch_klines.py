"""Download Binance spot klines from the public archive and write an engine series file.

Gaps are filled as in Phase 0: missing bars take the previous close for open/high/low/close and zero volume.

    python cases/finance/fetch_klines.py 1m-2024q1        # -> data/BTCUSDT_1m_2024Q1.bin   (131,040 bars)
    python cases/finance/fetch_klines.py 1s-2024-01-08    # -> data/BTCUSDT_1s_20240108_14.bin (604,800 bars)
"""

import argparse
import io
import struct
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ARCHIVE = "https://data.binance.vision/data/spot/{freq}/klines/{symbol}/{iv}/{symbol}-{iv}-{period}.zip"
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades",
        "taker_base", "taker_quote", "ignore"]
PRESETS = {
    "1m-2024q1": ("1m", "monthly", ["2024-01", "2024-02", "2024-03"], "BTCUSDT_1m_2024Q1"),
    "1s-2024-01-08": ("1s", "daily", [f"2024-01-{d:02d}" for d in range(8, 15)], "BTCUSDT_1s_20240108_14"),
}
MAGIC = b"SFSER001"


def fetch(symbol, iv, freq, periods):
    frames = []
    for period in periods:
        r = requests.get(ARCHIVE.format(freq=freq, symbol=symbol, iv=iv, period=period), timeout=120)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            raw = pd.read_csv(zf.open(zf.namelist()[0]), header=None, names=COLS)
        frames.append(raw[pd.to_numeric(raw.open_time, errors="coerce").notna()])   # drop a header row if present
    df = pd.concat(frames, ignore_index=True)
    t = pd.to_numeric(df.open_time).astype(np.int64)
    df["timestamp"] = pd.to_datetime(t, unit="us" if t.iloc[0] > 10**14 else "ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"]).drop_duplicates("timestamp")
    df = df.sort_values("timestamp")
    step = "1min" if iv == "1m" else "1s"
    grid = pd.date_range(df.timestamp.iloc[0].floor(step), df.timestamp.iloc[-1].floor(step), freq=step, tz="UTC")
    df = df.set_index("timestamp").reindex(grid)
    filled = int(df.close.isna().sum())
    df["close"] = df.close.ffill()
    for c in ["open", "high", "low"]:
        df[c] = df[c].fillna(df.close)
    df["volume"] = df.volume.fillna(0.0)
    return df.rename_axis("timestamp").reset_index(), filled


def write_series(path, df):
    ts = df.timestamp.dt.tz_convert(None).to_numpy().astype("datetime64[ns]").astype("<i8")
    arrays = [df[c].to_numpy(np.float32).astype("<f4") for c in ["low", "high", "volume"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<Q", len(ts)))
        for a in [ts, *arrays]:
            f.write(a.tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preset", choices=sorted(PRESETS))
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    args = ap.parse_args()
    iv, freq, periods, name = PRESETS[args.preset]
    out = args.out_dir / f"{name.replace('BTCUSDT', args.symbol)}.bin"
    if out.exists():
        print(f"{out} exists, skipping")
        return
    df, filled = fetch(args.symbol, iv, freq, periods)
    write_series(out, df)
    print(f"{out}: {len(df):,} bars {df.timestamp.iloc[0]} .. {df.timestamp.iloc[-1]}, {filled} filled")


if __name__ == "__main__":
    main()

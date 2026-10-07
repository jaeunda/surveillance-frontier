"""Gate: the C++ engine must reproduce the Phase 0 top-20 episodes on the 2024 Q1 1-minute series.

Both scan kinds are checked against experiments/phase0-pilot/results/colab-t4_2026-10-04/top20_real_1m.csv:
identical (k, start, window) in the same order, and S within 1e-9.

    python experiments/phase1-feasibility/check_phase0.py [--build build] [--data data/BTCUSDT_1m_2024Q1.bin]
"""

import argparse
import io
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "experiments/phase0-pilot/results/colab-t4_2026-10-04/top20_real_1m.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, default=ROOT / "build")
    ap.add_argument("--data", type=Path, default=ROOT / "data/BTCUSDT_1m_2024Q1.bin")
    args = ap.parse_args()
    ref = pd.read_csv(REFERENCE)
    ok = True
    for scan in ["direct", "optimized"]:
        out = subprocess.run([str(args.build / "sf_detect"), str(args.data), "--scan", scan],
                             check=True, capture_output=True, text=True).stdout
        got = pd.read_csv(io.StringIO(out))
        same = (len(got) == len(ref)
                and (got[["k", "start", "window"]].values == ref[["k", "start", "window"]].values).all()
                and (got.S - ref.S).abs().max() < 1e-9)
        print(f"{scan:<10} top-{len(ref)} vs Phase 0: {'match' if same else 'MISMATCH'}")
        ok &= bool(same)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

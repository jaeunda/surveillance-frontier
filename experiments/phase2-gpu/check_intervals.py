"""Validate the C++ Clopper-Pearson bounds against SciPy beta.ppf (SPEC §2.4), before Stage A.

For every k at n = 2,449 (alpha 0.05), n = 11,726 (alpha 0.05 / 3,264), and every checkpoint n = 2^8 .. 2^18 at the
per-interval alpha of T-ref-est: |C++ - SciPy| <= 1e-12 for both bounds, and the same pass/fail of the half-width
check (h = 0.02) for the fixed-n tasks. Writes a JSON summary.

    python experiments/phase2-gpu/check_intervals.py [--build build] [--out results/.../intervals.json]
"""

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTOCOL = json.loads((HERE / "protocol.json").read_text())
TOL = 1e-12


def scipy_bounds(n, alpha):
    k = np.arange(n + 1)
    lo = np.where(k > 0, beta.ppf(alpha / 2, k, n - k + 1), 0.0)
    hi = np.where(k < n, beta.ppf(1 - alpha / 2, k + 1, n - k), 1.0)
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, default=ROOT / "build")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--max-checkpoint", type=int, default=1 << 18)
    args = ap.parse_args()
    t = PROTOCOL["tasks"]
    cases = [("T1", t["T1-curve"]["trials_per_cell"], t["T1-curve"]["alpha"], t["T1-curve"]["half_width"]),
             ("T2", t["T2-policy-map"]["trials_per_cell"], t["T2-policy-map"]["alpha"], t["T2-policy-map"]["half_width"])]
    cases += [(f"checkpoint {n}", n, t["T-ref-est"]["alpha_per_interval"], None)
              for n in t["T-ref-est"]["checkpoints"] if n <= args.max_checkpoint]
    rows, ok = [], True
    for name, n, alpha, h in cases:
        out = subprocess.run([str(args.build / "sf_run"), "--cp", f"{n},{alpha!r}"], check=True, capture_output=True,
                             text=True).stdout
        cpp = pd.read_csv(io.StringIO(out))
        lo, hi = scipy_bounds(n, alpha)
        d = max(np.max(np.abs(cpp.lower.to_numpy() - lo)), np.max(np.abs(cpp.upper.to_numpy() - hi)))
        row = {"case": name, "n": n, "alpha": alpha, "max_abs_diff": float(d), "pass": bool(d <= TOL)}
        if h is not None:
            w_cpp = (cpp.upper - cpp.lower).to_numpy() / 2 <= h
            w_sci = (hi - lo) / 2 <= h
            row["width_check_same"] = bool(np.array_equal(w_cpp, w_sci))
            row["worst_half_width_cpp"] = float(np.max((cpp.upper - cpp.lower).to_numpy() / 2))
            row["pass"] &= row["width_check_same"]
        ok &= row["pass"]
        rows.append(row)
        print(f"{name:<20} n={n:>7} max |diff| = {d:.3e}  {'pass' if row['pass'] else 'FAIL'}", flush=True)
    report = {"tolerance": TOL, "pass": ok, "cases": rows}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

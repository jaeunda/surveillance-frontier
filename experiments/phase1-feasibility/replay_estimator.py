"""Re-check E1 from a stored run, without the engine or the market data.

Reads the enumerated hits from <run>/raw_hits.tar.gz (in memory; nothing is unpacked or written), recomputes the
exact rates, and replays the coverage check with the functions and seed of validate_estimator.py. Prints whether the
rates match <run>/estimator_cells.csv and the coverage matches <run>/estimator.csv.

    python experiments/phase1-feasibility/replay_estimator.py \
        experiments/phase1-feasibility/results/aws-c7i-16xlarge_2026-10-09
"""

import argparse
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from validate_estimator import HALF_WIDTHS, LENGTHS, STRENGTHS, coverage, shape  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261007)
    args = ap.parse_args()
    cells = pd.read_csv(args.run_dir / "estimator_cells.csv")
    stored = pd.read_csv(args.run_dir / "estimator.csv")
    policies = cells.policy.nunique()

    hits = {}
    with tarfile.open(args.run_dir / "raw_hits.tar.gz") as tar:
        members = {Path(m.name).name: m for m in tar.getmembers() if m.isfile()}
        for L in LENGTHS:                               # same order as validate_estimator.py, so the same draws
            for q in STRENGTHS:
                data = tar.extractfile(members[f"hits_L{L}_q{q}.u8"]).read()
                hits[(L, q)] = np.frombuffer(data, np.uint8).reshape(-1, policies).astype(bool)

    p = pd.Series({(L, q, j): h[:, j].mean() for (L, q), h in hits.items() for j in range(policies)})
    exact = cells.set_index(["length", "strength", "policy"]).p
    rates_ok = np.allclose(p.loc[exact.index].to_numpy(), exact.to_numpy(), rtol=0, atol=1e-15)  # CSV round trip
    print(f"exact rates: {len(exact)} cells x policies, match estimator_cells.csv: {rates_ok}")

    rng = np.random.default_rng(args.seed)
    rows = []
    for h in HALF_WIDTHS:
        c = coverage(cells, hits, h, args.reps, rng)
        c["monotone_violations"], c["strength_error_at_50pct"] = shape(cells, h)
        rows.append(c)
    cov = pd.DataFrame(rows)
    cols = [c for c in cov.columns if c in stored.columns]
    print(cov.to_string(index=False))
    cov_ok = np.allclose(cov[cols].to_numpy(float), stored[cols].to_numpy(float), rtol=0, atol=1e-12)
    print(f"coverage replay matches estimator.csv: {cov_ok}")
    sys.exit(0 if rates_ok and cov_ok else 1)


if __name__ == "__main__":
    main()

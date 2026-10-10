"""Gate item 9 (SPEC §5.2): the Phase 2 CPU build reproduces the 18 Phase 1 raw hit files byte for byte.

Re-runs the Phase 1 enumeration (validate_estimator.py: 1-second week, first 64,000 bars, P24, incremental, every
start of L in {6, 30, 120} x q in {4, ..., 128}) with build/sf_bench and compares each file with the copy in the
Phase 1 archive (read in memory; the archive is not modified). Writes a JSON report and exits non-zero on any
difference.

    python experiments/phase2-gpu/check_phase1_hits.py [--build build] [--threads N] [--out gate/item9.json]
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PHASE1 = ROOT / "experiments/phase1-feasibility"
ARCHIVE = PHASE1 / "results/aws-c7i-16xlarge_2026-10-09/raw_hits.tar.gz"
SERIES = ROOT / "data/BTCUSDT_1s_20240108_14.bin"
LENGTHS = [6, 30, 120]
STRENGTHS = [4, 8, 16, 32, 64, 128]
P24 = ["--k-set", "8,15", "--s-set", "1.5,2.0,2.5,3.0", "--gap-set", "30,60,120"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, default=ROOT / "build")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    with tarfile.open(ARCHIVE) as tar:
        stored = {Path(m.name).name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
    rows, bad = [], 0
    with tempfile.TemporaryDirectory() as tmp:
        for L in LENGTHS:
            for q in STRENGTHS:
                name = f"hits_L{L}_q{q}.u8"
                path = Path(tmp) / name
                cmd = [str(args.build / "sf_bench"), str(SERIES), "--method", "incremental", *P24, "--n", "64000",
                       "--mode", "across", "--threads", str(args.threads), "--length", str(L), "--strength", str(q),
                       "--enumerate", "1", "--hits-out", str(path), "--warmup", "0"]
                subprocess.run(cmd, check=True, capture_output=True)
                got = path.read_bytes()
                same = got == stored[name]
                bad += not same
                rows.append({"file": name, "bytes": len(got), "sha256": hashlib.sha256(got).hexdigest(),
                             "expected_sha256": hashlib.sha256(stored[name]).hexdigest(), "identical": same})
                print(f"{name:<22} {'identical' if same else 'DIFFERENT'}", flush=True)
    report = {"item": 9, "files": len(rows), "mismatching_files": bad, "pass": bad == 0, "rows": rows}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"item 9: {len(rows) - bad}/{len(rows)} files identical")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()

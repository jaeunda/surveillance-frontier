"""Reference outputs for the formal comparison (Part III, Stage B), on any machine with a gated build.

With the CPU-best configuration: T1 and T2 on both series (their result hashes are what every timed run must
reproduce) with spot verification, the T-ref enumeration (the exact reference for H4) and its oracle (2,000 shared +
200 full re-evaluations per cell), and the golden CPU gate. Writes results/formal/<label>/reference/. Bits do not
depend on the machine; this is checked, not assumed: every formal CPU launch recomputes the gate and every formal
T-ref run regenerates the bits, and the hashes must match.

    reference.py --label L --threads N [--parity SET] [--skip fixed,tref,oracle,gate]
"""

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from p2 import formal, harness, machine  # noqa: E402
from p2.common import FORMAL, ROOT, git_state, read_json, run_logged, write_json  # noqa: E402
from p2.tasks import main_task  # noqa: E402


def last_json(stdout):
    return json.loads(stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--build", type=Path, default=ROOT / "build")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--parity", default="none")
    ap.add_argument("--skip", default="")
    a = ap.parse_args()
    out = formal.formal_dir(a.label) / "reference"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "reference.log"
    skip = set(filter(None, a.skip.split(",")))
    cfg = harness.cpu_config(a.threads, parity=a.parity)
    sf = [a.build / "sf_run"]
    be = harness.sf_args(cfg)
    hp = out / "hashes.json"
    hashes = read_json(hp) if hp.exists() else {}
    write_json(out / "env.json", {**machine.environment("reference", a.build), "config": cfg, "git": git_state()})

    def run(cmd):
        r = run_logged(cmd, log)
        if r.returncode != 0:
            sys.exit(f"failed: {' '.join(map(str, cmd))}\n{r.stderr[-2000:]}")
        return r.stdout

    if "fixed" not in skip:
        for task in ("T1-curve", "T2-policy-map"):
            for series in ("1m-quarter", "1s-week"):
                tf, d = main_task(task, series), out / f"{task}_{series}"
                r = last_json(run([*sf, tf, "--out", d, *be]))
                spot = last_json(run([*sf, tf, "--spot", d, *be]))
                if r["status"] != "ok" or spot["mismatches"] != 0:
                    sys.exit(f"{task} {series}: status {r['status']}, spot mismatches {spot['mismatches']}")
                hashes[f"{task}/{series}"] = r["result_sha256"]
                write_json(hp, hashes)
    if "tref" not in skip:
        r = last_json(run([*sf, main_task("T-ref", "1s-week"), "--out", out / "T-ref_1s-week", *be]))
        hashes["T-ref/1s-week"] = r["result_sha256"]
        write_json(hp, hashes)
    if "oracle" not in skip:
        o = FORMAL["oracle"]
        run([a.build / "sf_gate", "--oracle", main_task("T-ref", "1s-week"), out / "T-ref_1s-week", "--threads", a.threads,
             "--ref-parity", a.parity, "--oracle-shared", o["starts_per_cell_shared"], "--oracle-full",
             o["starts_per_cell_full"], "--out", out / "oracle.json"])
    if "gate" not in skip:
        run([a.build / "sf_gate", "--device", "cpu", "--threads", a.threads, "--parity", a.parity, "--ref-parity",
             a.parity, "--items", "1,3,4,5,6,7,8", "--out", out / "gate_cpu.json"])
        run([sys.executable, HERE / "check_phase1_hits.py", "--build", a.build, "--out", out / "item9.json"])
    print(json.dumps(hashes, indent=1))


if __name__ == "__main__":
    main()

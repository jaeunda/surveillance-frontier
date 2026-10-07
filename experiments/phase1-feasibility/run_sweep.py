"""Phase 1 CPU measurements. Each configuration is one call of build/sf_bench (one JSON line).

Parts (all by default; --parts selects):
  verify    gate on real data: shared / tail / incremental against the full method, 24 policies, several event
            lengths and strengths including q = 0; any mismatching trial fails the run
  cost      one policy, 1 thread: cost per trial vs N (prefixes of the 1-second week) and K, per method (F1, F2)
  scaling   one policy, K = 15: across mode over the thread ladder per method, within mode for full and tail (F3)
  batch     policy grids P1 / P24 / P105 at all physical cores, across mode: trials per second with every policy
            evaluated in each trial, per method, series, and event length (the rates the projection uses)
  scenario  P24, all physical cores: cost per trial vs event length, strength and kind (is one length enough?)
  validate  the validation task of scenario_space.json, run cell by cell with fixed trial counts (F4)

Every part except verify and validate is repeated --reps times; all configurations of all repetitions run in one
shuffled order (seeded), so drift and ordering effects spread over configurations instead of biasing some.

    python experiments/phase1-feasibility/run_sweep.py --env aws-c7i-16xlarge [--parts cost,batch] [--quick]
"""

import argparse
import datetime as dt
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from scenario_space import trials_per_cell  # noqa: E402

SERIES = {
    "1m-quarter": ROOT / "data/BTCUSDT_1m_2024Q1.bin",
    "1s-week": ROOT / "data/BTCUSDT_1s_20240108_14.bin",
}
N_1S = [16_000, 32_000, 64_000, 128_000, 256_000, 604_800]   # prefixes of the 1-second week
K_SET = [4, 8, 15]
PARTS = ["verify", "cost", "scaling", "batch", "scenario", "validate"]
MAX_LENGTH = 256                                             # incremental heads sized for every event length


def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()


def first(cmd):
    try:
        return run(cmd).splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return None


def physical_cores():
    """Distinct (physical id, core id) pairs in /proc/cpuinfo."""
    ids, phys = set(), None
    for line in open("/proc/cpuinfo"):
        key, _, val = line.partition(":")
        if key.strip() == "physical id":
            phys = val.strip()
        elif key.strip() == "core id":
            ids.add((phys, val.strip()))
    return len(ids) or None


def environment():
    def full(cmd):
        try:
            return run(cmd)
        except (OSError, subprocess.CalledProcessError):
            return None
    cpu = next((line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name")),
               platform.processor())
    read = lambda p: Path(p).read_text().strip() if Path(p).exists() else None  # noqa: E731
    return {
        "date": dt.date.today().isoformat(),
        "cpu": cpu,
        "logical_cores": os.cpu_count(),
        "physical_cores": physical_cores(),
        "os": platform.platform(),
        "compiler": first(["c++", "--version"]),
        "git_commit": first(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]),
        "git_dirty": bool(first(["git", "-C", str(ROOT), "status", "--porcelain"])),
        "omp_places": os.environ.get("OMP_PLACES"),
        "omp_proc_bind": os.environ.get("OMP_PROC_BIND"),
        "governor": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "thp": read("/sys/kernel/mm/transparent_hugepage/enabled"),
        "lscpu": full(["lscpu"]),
        "numactl": full(["numactl", "--hardware"]),
    }


def thread_ladder(phys, logical):
    t, out = 1, []
    while t < phys:
        out.append(t)
        t *= 2
    return out + [phys] + ([logical] if logical > phys else [])


def grid_args(space, grid):
    g = space["policy_grids"][grid]
    return ["--k-set", ",".join(map(str, g["K"])), "--s-set", ",".join(map(str, g["s_min"])),
            "--gap-set", ",".join(map(str, g["episode_gap"]))]


def configs(args, space, phys, logical):
    """(part, rep, label, sf_bench arguments) for every configuration of the selected parts."""
    reps, sec = args.reps, str(args.seconds)
    one = lambda k: ["--k-set", str(k), "--s-set", "2", "--gap-set", "60"]  # noqa: E731
    out = []
    quick = args.quick
    n_1s = [16_000, 64_000] if quick else N_1S
    ladder = [1, phys] if quick else thread_ladder(phys, logical)
    lengths = [2, 30, 256]
    for part in args.parts:
        if part == "verify":
            for series, events in [("1m-quarter", [(L, q) for L in lengths for q in (0, 16, 128)]),
                                   ("1s-week", [(30, 0), (30, 128)])][:1 if quick else 2]:
                for method in ["shared", "tail", "incremental"]:
                    for L, q in (events[:2] if quick else events):
                        out.append((part, 0, series, [str(SERIES[series]), "--method", method, *grid_args(space, "P24"),
                                                      "--mode", "across", "--threads", str(phys),
                                                      "--trials", str(2 * phys if not quick else phys),
                                                      "--length", str(L), "--strength", str(q),
                                                      "--max-length", str(MAX_LENGTH), "--verify", "--warmup", "0"]))
        for rep in range(reps if part not in ("verify", "validate") else 0):
            if part == "cost":
                for method in ["full", "tail", "incremental"]:
                    for n in n_1s:
                        for k in K_SET:
                            out.append((part, rep, "1s-week", [str(SERIES["1s-week"]), "--method", method, *one(k),
                                                               "--n", str(n), "--threads", "1", "--seconds", sec,
                                                               "--strength", "16", "--warmup", "1"]))
            elif part == "scaling":
                for method in ["full", "tail", "incremental"]:
                    for n in [64_000, 604_800]:
                        for mode in ["across", "within"] if method != "incremental" else ["across"]:
                            if mode == "within" and n != 604_800:
                                continue
                            for t in ladder:
                                out.append((part, rep, "1s-week",
                                            [str(SERIES["1s-week"]), "--method", method, *one(15), "--n", str(n),
                                             "--mode", mode, "--threads", str(t), "--seconds", sec,
                                             "--strength", "16", "--warmup", "1"]))
            elif part == "batch":
                combos = [("P1", m) for m in ["full", "tail", "incremental"]] + \
                         [("P24", m) for m in ["full", "shared", "tail", "incremental"]] + \
                         [("P105", m) for m in ["shared", "tail", "incremental"]]
                if quick:  # enough for the projection and the validation task, without full on P24
                    combos = [("P1", "full")] + [("P24", m) for m in ["shared", "tail", "incremental"]]
                for grid, method in combos:
                    for series in SERIES:
                        for L in ([2, 256] if quick else lengths):
                            out.append((part, rep, series, [str(SERIES[series]), "--method", method,
                                                            *grid_args(space, grid), "--mode", "across",
                                                            "--threads", str(phys), "--seconds", sec,
                                                            "--length", str(L), "--max-length", str(MAX_LENGTH),
                                                            "--strength", "16", "--warmup", "1"]))
            elif part == "scenario":
                for method in ["tail", "incremental"]:
                    for L in lengths:
                        for q in (0, 8, 128):
                            for kind in ["both"] if q != 128 else ["both", "price", "volume"]:
                                out.append((part, rep, "1s-week",
                                            [str(SERIES["1s-week"]), "--method", method, *grid_args(space, "P24"),
                                             "--mode", "across", "--threads", str(phys), "--seconds", sec,
                                             "--length", str(L), "--max-length", str(MAX_LENGTH), "--strength", str(q),
                                             "--event", kind, "--warmup", "1"]))
        if part == "validate":
            v = space["validation_task"]
            task = {"policies": v["policies"], "lengths": v["lengths"], "strengths": v["strengths"],
                    "half_width": v["half_width"], "coverage": v["coverage"]}
            cells, trials = trials_per_cell(space, task)
            for method in v["methods"]:
                for (L, q), n in zip(cells, trials):
                    out.append((part, 0, v["series"], [str(SERIES[v["series"]]), "--method", method,
                                                      *grid_args(space, v["policies"]), "--mode", "across",
                                                      "--threads", str(phys), "--trials", str(n if not quick else phys),
                                                      "--length", str(L), "--max-length", str(MAX_LENGTH),
                                                      "--strength", str(q), "--warmup", "1"]))
    return out


def grid_of(cmd, space):
    """Name of the policy grid a command line uses (or None for a single ad-hoc policy)."""
    get = lambda flag: cmd[cmd.index(flag) + 1]  # noqa: E731
    for name, g in space["policy_grids"].items():
        if (get("--k-set") == ",".join(map(str, g["K"])) and get("--s-set") == ",".join(map(str, g["s_min"]))
                and get("--gap-set") == ",".join(map(str, g["episode_gap"]))):
            return name
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, help="environment label, e.g. aws-c7i-16xlarge or local-wsl")
    ap.add_argument("--build", type=Path, default=ROOT / "build")
    ap.add_argument("--space", type=Path, default=HERE / "scenario_space.json")
    ap.add_argument("--parts", default=",".join(PARTS))
    ap.add_argument("--reps", type=int, default=3, help="independent repetitions of every timed configuration")
    ap.add_argument("--seconds", type=float, default=5.0, help="time budget per configuration")
    ap.add_argument("--physical-cores", type=int, default=physical_cores() or os.cpu_count())
    ap.add_argument("--logical-cores", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=20261007, help="order of the shuffled configurations")
    ap.add_argument("--quick", action="store_true", help="few configurations, short budget (smoke test)")
    args = ap.parse_args()
    args.parts = args.parts.split(",")
    if args.quick:
        args.reps, args.seconds = 1, 0.5
    space = json.loads(args.space.read_text())

    out_dir = HERE / "results" / f"{args.env}_{dt.date.today().isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = environment()
    env.update({"physical_cores_used": args.physical_cores, "reps": args.reps, "seconds": args.seconds,
                "seed": args.seed, "parts": args.parts, "quick": args.quick})
    (out_dir / "env.json").write_text(json.dumps(env, indent=2) + "\n")

    todo = configs(args, space, args.physical_cores, args.logical_cores)
    gates = [c for c in todo if c[0] == "verify"]          # gates run first, in order
    timed = [c for c in todo if c[0] != "verify"]
    random.Random(args.seed).shuffle(timed)
    bench = str(args.build / "sf_bench")
    failed = 0
    with open(out_dir / "cpu_sweep.jsonl", "w") as f:
        for i, (part, rep, series, cmd) in enumerate(gates + timed):
            proc = subprocess.run([bench, *cmd], capture_output=True, text=True)
            if proc.returncode and part != "verify":
                raise RuntimeError(f"sf_bench failed: {' '.join(cmd)}\n{proc.stderr}")
            rec = {"part": part, "rep": rep, "order": i, "series": series, "grid": grid_of(cmd, space),
                   **json.loads(proc.stdout)}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            failed += part == "verify" and rec["mismatches"] > 0
            print(f"[{i + 1}/{len(todo)}] {part:<8} {series:<10} {rec['method']:<11} {rec['grid'] or '-':<4} "
                  f"n={rec['n']:>7} {rec['mode']:<6} t={rec['threads']:>2} L={rec['length']:>3} q={rec['strength']:>5g}"
                  f": {rec['trials_per_s']:10.2f} trials/s"
                  + (f", mismatches {rec['mismatches']}" if rec["verified"] else ""), flush=True)
            if part == "verify" and failed and i == len(gates) - 1:
                sys.exit(f"verify gate failed ({failed} configurations with mismatches); timing not run")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()

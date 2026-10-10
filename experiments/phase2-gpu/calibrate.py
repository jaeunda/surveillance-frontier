"""Pre-freeze calibration on the formal machines (Part III). Every choice follows the registered rule; inputs and
decisions go to results/formal/<label>/calibration/<machine>/. The formal machines are those of the committed
scaling decision; this script refuses to run on any other hardware. Pilot choices are never carried over.

    calibrate.py threads --label L [--parity SET]               CPU threads per (series, grid): short + sustained design
    calibrate.py parity  --label L --threads N                   CPU parity rule (PC1..PC6) -> parity.md
    calibrate.py rates   --label L --config cpu|gpu [...]        rates at every task length and every T-ref cell (timeouts)
    calibrate.py batch   --label L --path inc|hyb                GPU microbatch among {64, 256, 1024, b_max}
    calibrate.py path    --label L                               G-inc vs G-hyb at the representative point
    calibrate.py drain   --label L --config cpu|gpu              deadline drain margin
    calibrate.py profile --label L --config cpu|gpu              perf stat counters (exploratory, never timed)

--config cpu uses the CPU-best of earlier steps (threads.json, parity.json); --config gpu the chosen path and batch.
"""

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from p2 import design, formal, harness, machine  # noqa: E402
from p2.common import FORMAL, PROTOCOL, ROOT, read_json, write_json  # noqa: E402
from p2.tasks import calib_task, point  # noqa: E402

TC = FORMAL["thread_calibration"]


def setup(a, role):
    rec = formal.decision()
    name, prof = formal.profile_of(rec, role)
    ident = machine.identity(name)
    if ident["match"] is not True:
        sys.exit(f"this machine is not the formal {role} machine {name}: {ident}")
    d = formal.formal_dir(a.label) / "calibration" / name
    d.mkdir(parents=True, exist_ok=True)
    write_json(d / f"env_{a.what}.json", machine.environment(name, a.build))
    return d, name, prof


def cpu_best(d):
    t = read_json(d / "threads.json")
    par = read_json(d / "parity.json")["parity"] if (d / "parity.json").exists() else "none"
    return harness.cpu_config(t["threads"]["1s-week/P24"], parity=par)


def gpu_best(d):
    p = read_json(d / "path.json")["choice"]
    b = read_json(d / f"batch_{p}.json")["microbatch"]["1s-week/P24"]
    return harness.gpu_config(p, b, host_threads=os.cpu_count())


def cmd_threads(a):
    role = "gpu" if a.gpu_host else "cpu"
    d, name, prof = setup(a, role)
    topo = machine.topology()
    cands = formal.thread_candidates({} if a.gpu_host else prof, topo)
    cfgs = [harness.cpu_config(n, parity=a.parity) for n in cands]
    out, recs = {}, []
    for series in PROTOCOL["series"]:
        for grid in TC["grids"]["set_task_threads"] + TC["grids"]["report_only"]:
            r = design.choose(series, grid, cfgs, d / "calibration.jsonl", build=a.build, label="threads")
            recs.append(r)
            if grid in TC["grids"]["set_task_threads"] and r.get("config"):
                out[f"{series}/{grid}"] = r["config"]["threads"]
            print(f"threads {series} {grid}: {r.get('choice')} (flip {r.get('flip')})", flush=True)
    write_json(d / ("threads_host.json" if a.gpu_host else "threads.json"),
               {"machine": name, "candidates": cands, "threads": out, "records": recs,
                "choice_differs_across_grids_or_series": len({r.get("choice") for r in recs}) > 1})


def cmd_parity(a):
    d, name, _ = setup(a, "cpu")
    adopted = design.parity(harness.cpu_config(a.threads), d / "calibration.jsonl", d / "parity.md", build=a.build,
                            min_gain=FORMAL["verdicts"]["parity_min_gain"])
    write_json(d / "parity.json", {"machine": name, "threads": a.threads, "parity": adopted})
    print(f"CPU-best parity set: {adopted}; repeat calibrate.py threads with --parity {adopted} if it is not 'none'")


def cmd_rates(a):
    d, name, _ = setup(a, a.config)
    cfg = cpu_best(d) if a.config == "cpu" else gpu_best(d)
    log = d / f"rates_{a.config}.jsonl"
    for series in PROTOCOL["series"]:
        for grid in ("P1", "P24"):
            for L in PROTOCOL["tasks"]["T1-curve"]["lengths"]:
                for rep in range(3):
                    design.bench_point(calib_task(series, grid, L), cfg, 5, log, build=a.build, label="rate", rep=rep)
    for c in range(15):
        for rep in range(3):
            design.bench_point(point(f"pilot/tref-cell_c{c}"), cfg, 5, log, build=a.build, label="tref-rate", rep=rep)
    print(f"rates written: {log}")


def cmd_batch(a):
    d, name, _ = setup(a, "gpu")
    host = os.cpu_count()
    cfgs = [harness.gpu_config(a.path, b, host) for b in (64, 256, 1024, "auto")]
    out, recs = {}, []
    for series in PROTOCOL["series"]:
        for grid in TC["grids"]["set_task_threads"] + TC["grids"]["report_only"]:
            r = design.choose(series, grid, cfgs, d / "calibration.jsonl", build=a.build, label=f"batch-{a.path}")
            recs.append(r)
            if grid in TC["grids"]["set_task_threads"] and r.get("config"):
                out[f"{series}/{grid}"] = r["config"]["microbatch"]
            print(f"batch {a.path} {series} {grid}: {r.get('choice')}", flush=True)
    write_json(d / f"batch_{a.path}.json", {"machine": name, "path": a.path, "microbatch": out, "records": recs})


def cmd_path(a):
    d, name, _ = setup(a, "gpu")
    cfgs = {}
    for p in ("inc", "hyb"):
        f = d / f"batch_{p}.json"
        if not f.exists():
            sys.exit(f"run calibrate.py batch --path {p} first")
        cfgs[p] = harness.gpu_config(p, read_json(f)["microbatch"]["1s-week/P24"], os.cpu_count())
    rec = design.select_path(cfgs, d / "calibration.jsonl", build=a.build)
    write_json(d / "path.json", {"machine": name, **rec})
    print(f"path: {rec['choice']} (rates {rec['rates']})")


def cmd_drain(a):
    d, name, _ = setup(a, a.config)
    cfg = cpu_best(d) if a.config == "cpu" else gpu_best(d)
    rec = design.drain(point("calib/T-ref-est_drain"), cfg, d / "calibration.jsonl", HERE / "results" / "_drain_runs",
                       build=a.build)
    write_json(d / f"drain_{a.config}.json", {"machine": name, **rec})
    print(f"drain margin {a.config}: {rec['drain_ms']} ms")


def cmd_profile(a):
    d, name, _ = setup(a, a.config)
    cfg = cpu_best(d) if a.config == "cpu" else gpu_best(d)
    write_json(d / f"profile_{a.config}.json", {"machine": name, "profile": design.profile(cfg, d / "profile.jsonl",
                                                                                            build=a.build)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["threads", "parity", "rates", "batch", "path", "drain", "profile"])
    ap.add_argument("--label", required=True)
    ap.add_argument("--build", type=Path, default=ROOT / "build")
    ap.add_argument("--parity", default="none")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--path", choices=["inc", "hyb"])
    ap.add_argument("--config", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--gpu-host", action="store_true", help="threads: calibrate the GPU machine's host (CPU-host)")
    a = ap.parse_args()
    a.build = a.build.resolve()
    if a.what == "parity" and not a.threads:
        sys.exit("--threads is required")
    if a.what == "batch" and not a.path:
        sys.exit("--path is required")
    {"threads": cmd_threads, "parity": cmd_parity, "rates": cmd_rates, "batch": cmd_batch, "path": cmd_path,
     "drain": cmd_drain, "profile": cmd_profile}[a.what](a)


if __name__ == "__main__":
    main()

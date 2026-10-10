"""One formal launch (Part III): Stage A (CPU) or Stage D (CPU, GPU, or the CPU size control). Called by run_launch.sh.

    launch.py --stage A --role cpu --launch I --label L [--parity SET]
    launch.py --stage D --role cpu|gpu|size-control --launch I --label L

Preconditions (each stops the launch before any timing): the scaling decision is complete, committed, and proceeds;
this machine matches the profile it names for the role (otherwise the launch is kept as "hardware mismatch",
excluded, and replaced); the page cache can be dropped (cold = cache-cold in the formal comparison); Stage D needs
protocol-freeze.json committed.

Steps:
  1  env.json (manifest, identity, capabilities)
  2  Stage A: thread calibration on this launch and rates at every task length (R2 re-projection)
     Stage D: boot-cold (descriptive), then the gate step; a failed gate stops the launch
  3  all conditions in one order shuffled with seed 20261010 + launch index, written before the first run.
     Cold repetitions are single items; warm repetitions of a condition are one block (priming outside the timing)
  4  Stage D, pair 1: the D1 points (T-diag, 3 x 5 s; calibration and held-out split)
  5  Stage D, GPU launches: CPU-host (descriptive), threads chosen on this host by the same design
  6  spot verification (outside the timing), archive of large outputs, SHA256SUMS
A spot interruption notice (polled every 5 s) marks the launch incomplete; it is never resumed.
"""

import argparse
import datetime as dt
import hashlib
import json
import random
import sys
import tarfile
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from p2 import design, formal, harness, machine  # noqa: E402
from p2.common import (FORMAL, GRID_OF_TASK, PROTOCOL, ROOT, append_jsonl, committed_unchanged, read_json,  # noqa: E402
                       run_logged, sha256_file, write_json)
from p2.tasks import calib_task, main_task  # noqa: E402


class InterruptWatch:
    def __init__(self, path):
        self.path, self.notice = path, None
        self.stop = threading.Event()
        self.t = threading.Thread(target=self.loop, daemon=True)

    def loop(self):
        while not self.stop.wait(5):
            n = machine.spot_interruption_notice()
            if n:
                self.notice = n
                append_jsonl(self.path, {"time": dt.datetime.now(dt.timezone.utc).isoformat(), "notice": n})


def config_for(role, task, series, freeze, host=None):
    """The frozen configuration of a role for one task (Stage A: the launch's own thread choice)."""
    key = f"{series}/{GRID_OF_TASK.get(task, 'P24')}"
    if role in ("cpu", "size-control") or host:
        c = host or freeze["cpu"]
        return harness.cpu_config(c["threads"].get(key, c["threads"].get("1s-week/P24")), parity=freeze["cpu"]["parity"],
                                  warmup=c.get("warmup_trials", 0))
    g = freeze["gpu"]
    return harness.gpu_config(g["path"], g["microbatch"].get(key, g["microbatch"]["1s-week/P24"]), g["host_threads"]
                              or 1, g["submission"], g["warmup_trials"])


def gate_step(role, d, build, freeze):
    gd = d / "gate"
    log = gd / "gate.log"
    res = {}
    if role in ("cpu", "size-control"):
        par = freeze["cpu"]["parity"]
        threads = machine.topology()["logical"]
        res["sf_test"] = run_logged([build / "sf_test"], log).returncode == 0
        res["sf_test2"] = run_logged([build / "sf_test2"], log).returncode == 0
        res["sf_gate_cpu"] = run_logged([build / "sf_gate", "--device", "cpu", "--threads", threads, "--parity", par,
                                         "--ref-parity", par, "--items", "1,3,4,5,6,7,8", "--out",
                                         gd / "gate_cpu.json"], log).returncode == 0
        res["item9"] = run_logged([sys.executable, HERE / "check_phase1_hits.py", "--build", build, "--out",
                                   gd / "item9.json"], log).returncode == 0
    else:
        g = freeze["gpu"]
        res["sf_gate_cuda"] = run_logged([build / "sf_gate", "--device", "cuda", "--path", g["path"], "--microbatch",
                                          g["microbatch"]["1s-week/P24"], "--threads", g["host_threads"] or 1,
                                          "--items", "1,2,7,8", "--out", gd / "gate_cuda.json"], log).returncode == 0
    write_json(gd / "gate_summary.json", res)
    return all(res.values()), res


def spot(rec, build, cfg):
    import subprocess
    r = subprocess.run([str(build / "sf_run"), rec["task_file"], "--spot", rec["out"], *harness.sf_args(cfg)],
                       capture_output=True, text=True, cwd=ROOT)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1]).get("mismatches")
    except (ValueError, IndexError):
        return None


def archive(d):
    big = sorted(d.rglob("*.bin"))
    if big:
        with tarfile.open(d / "outputs_bits.tar.gz", "w:gz") as tar:
            for p in big:
                tar.add(p, arcname=str(p.relative_to(d)))
        for p in big:
            p.unlink()
    lines = [f"{sha256_file(p)}  ./{p.relative_to(d)}" for p in sorted(x for x in d.rglob("*")
                                                                      if x.is_file() and x.name != "SHA256SUMS")]
    (d / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def run_items(items, d, role, build, freeze, timeouts, ref, watch, env, host=None, kind_label=None):
    runs = d / "runs.jsonl"
    spot_cfg = harness.cpu_config(machine.topology()["logical"])  # verification outside the timing, CPU method
    for pos, c in enumerate(items):
        if watch.notice:
            env["status"] = "incomplete: spot interruption"
            return
        cfg = config_for(role, c["task"], c["series"], freeze, host)
        tf = formal.task_file(c)
        key = formal.condition_key(c)
        cond = {"task": c["task"], "series": c["series"], "task_file": str(tf), "position": pos,
                "condition": key, **({"kind": kind_label} if kind_label else {})}
        name = f"{pos:04d}_{key.replace('/', '_')}_r{c.get('rep', 0)}"
        out = d / "outputs" / name
        expected = ref.get(f"{c['task']}/{c['series']}")
        to = timeouts.get(key)
        if c["kind"] == "cold":
            recs = [harness.run_cold(tf, out, cfg, {**cond, "rep": c["rep"]}, build=build, timeout_s=to,
                                     expected_sha=expected)]
        elif c["kind"] == "deadline":
            recs = [harness.run_cold(tf, out, cfg, {**cond, "rep": c["rep"], "budget_s": c["budget_s"]}, build=build,
                                     budget_s=c["budget_s"], drain_ms=freeze["gpu" if role == "gpu" else "cpu"]["drain_ms"],
                                     replicate=c["replicate"])]
        else:
            recs = []
            try:
                srv = harness.WarmServer(cfg, build=build)
                try:
                    srv.prime(tf, Path(f"{out}_prime"))
                    for r in range(c["reps"]):
                        recs.append(srv.timed(tf, Path(f"{out}_w{r}"), {**cond, "rep": r}, expected_sha=expected,
                                              timeout_s=to))
                finally:
                    srv.close()
            except (RuntimeError, OSError) as e:
                recs.append({"mode": "warm", **cond, "config": cfg, "config_key": harness.config_key(cfg),
                             "status": "error", "error": repr(e)})
        for rec in recs:
            if rec.get("status") == "ok" and c["task"] in ("T1-curve", "T2-policy-map"):
                rec["spot_mismatches"] = spot(rec, build, spot_cfg)
                if rec["spot_mismatches"] != 0:
                    rec["status"] = "spot_mismatch"
            append_jsonl(runs, rec)
            print(f"[{pos + 1}/{len(items)}] {key} rep {rec.get('rep')}: {rec.get('e2e_s', float('nan')):.3f} s "
                  f"{rec.get('status')} {rec.get('cold_kind', '')}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["A", "D"], required=True)
    ap.add_argument("--role", choices=formal.ROLES, required=True)
    ap.add_argument("--launch", type=int, required=True, help="launch index i (launch pair i in Stage D)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--build", type=Path, default=ROOT / "build")
    ap.add_argument("--parity", default="none", help="Stage A: CPU parity set")
    ap.add_argument("--dry-run", action="store_true", help="env.json and order.json only")
    a = ap.parse_args()
    build = a.build.resolve()
    if a.stage == "A" and a.role != "cpu":
        sys.exit("Stage A launches are CPU launches")
    rec = formal.decision()
    prof_name, prof = formal.profile_of(rec, a.role)
    fd = formal.formal_dir(a.label)
    if a.stage == "D":
        fz = fd / "protocol-freeze.json"
        if not fz.exists() or not committed_unchanged(fz):
            sys.exit("Stage D needs a committed protocol-freeze.json (freeze.py)")
        freeze = read_json(fz)
        if freeze["scaling_decision_sha256"] != sha256_file(formal.DECISION):
            sys.exit("the scaling decision changed after the freeze")
    else:
        freeze = {"cpu": {"parity": a.parity, "threads": {}, "warmup_trials": 0, "drain_ms": 0}, "timeouts_s": {},
                  "reference_sha256": {}}
        refp = fd / "reference" / "hashes.json"
        if refp.exists():
            freeze["reference_sha256"] = read_json(refp)
    d = formal.launch_dir(a.label, a.role, a.launch)
    d.mkdir(parents=True, exist_ok=True)
    env = machine.environment(prof_name, build)
    env.update({"stage": a.stage, "role": a.role, "launch_index": a.launch,
                "scaling_decision_sha256": sha256_file(formal.DECISION),
                "freeze_sha256": sha256_file(fd / "protocol-freeze.json") if a.stage == "D" else None})
    ident = env["identity"]
    env["status"] = "running"
    if ident["match"] is not True:
        env["status"] = "hardware mismatch"
    elif not env["capabilities"]["drop_page_cache"]:
        env["status"] = "not run: page cache cannot be dropped (formal cold runs are cache-cold)"
    write_json(d / "env.json", env)
    if env["status"] != "running" and not a.dry_run:
        sys.exit(env["status"])

    seed = PROTOCOL["stream"]["seed"] + a.launch
    items = formal.conditions(a.role, a.stage, a.launch)
    random.Random(seed).shuffle(items)
    write_json(d / "order.json", {"seed": seed, "order": items})
    if a.dry_run:
        print(f"dry run: {len(items)} items in {d / 'order.json'}")
        return

    watch = InterruptWatch(d / "interruptions.jsonl")
    if env["ec2"].get("purchase_option") == "spot":
        watch.t.start()
    timeouts = freeze["timeouts_s"]
    ref = freeze["reference_sha256"]
    cal = d / "calibration.jsonl"

    if a.stage == "A":
        cands = [harness.cpu_config(n, parity=a.parity) for n in formal.thread_candidates(prof, env["topology"])]
        threads = {}
        tc = FORMAL["thread_calibration"]
        for series in PROTOCOL["series"]:
            for grid in tc["grids"]["set_task_threads"] + tc["grids"]["report_only"]:
                r = design.choose(series, grid, cands, cal, build=build, label="threads")
                if grid in tc["grids"]["set_task_threads"] and r.get("config"):
                    threads[f"{series}/{grid}"] = r["config"]["threads"]
        freeze["cpu"]["threads"] = threads
        write_json(d / "threads.json", {"threads": threads})
        for series in PROTOCOL["series"]:
            for grid in ("P1", "P24"):
                cfg = harness.cpu_config(threads[f"{series}/{grid}"], parity=a.parity)
                for L in PROTOCOL["tasks"]["T1-curve"]["lengths"]:
                    for rep in range(3):
                        design.bench_point(calib_task(series, grid, L), cfg, 5, d / "rates.jsonl", build=build,
                                           label="rate", rep=rep)
    else:
        tf = main_task("T1-curve", "1m-quarter")
        append_jsonl(d / "runs.jsonl", harness.run_cold(
            tf, d / "outputs" / "boot-cold", config_for(a.role, "T1-curve", "1m-quarter", freeze),
            {"kind": "boot-cold", "task": "T1-curve", "series": "1m-quarter", "task_file": str(tf), "rep": 0},
            build=build, expected_sha=ref.get("T1-curve/1m-quarter")))
        ok, res = gate_step(a.role, d, build, freeze)
        if not ok:
            env["status"] = "gate failed: no timings"
            write_json(d / "env.json", env)
            archive(d)
            sys.exit(f"gate failed: {res}")

    run_items(items, d, a.role, build, freeze, timeouts, ref, watch, env)

    if a.stage == "D" and a.launch == 1 and a.role in ("cpu", "gpu") and env["status"] == "running":
        cfg = config_for(a.role, "T-diag", "1s-week", freeze)
        td = PROTOCOL["t_diag"]
        for f in sorted((HERE / "tasks" / "tdiag").glob("*.json")):
            if f.name.startswith("cross_") and a.role != "gpu":
                continue
            batches = [None]
            if a.role == "gpu" and f.name.startswith("cross_"):
                batches = td["gpu_cross"]["batch"]
            elif a.role == "gpu" and f.name == f"count_24_n{td['gpu_cross']['n']}.json":
                batches = td["gpu_batch"]
            for b in batches:
                c = cfg if b is None else {**cfg, "microbatch": "auto" if b == "b_max" else b}
                for rep in range(td["repetitions"]):
                    design.bench_point(f, c, td["seconds"], d / "tdiag.jsonl", build=build, label="tdiag", rep=rep)

    if a.role == "gpu" and a.stage == "D" and env["status"] == "running":
        hd = d / "cpu_host"
        topo = env["topology"]
        cands = [harness.cpu_config(n, parity=freeze["cpu"]["parity"]) for n in sorted({topo["physical_cores"], topo["logical"]})]
        r = design.choose("1s-week", "P24", cands, hd / "calibration.jsonl", build=build, label="cpu-host")
        host = {"threads": {"1s-week/P24": r["config"]["threads"]} if r.get("config") else {"1s-week/P24": topo["logical"]},
                "warmup_trials": freeze["cpu"]["warmup_trials"]}
        jobs = [{"kind": "cold", "task": t, "series": s, "rep": k} for t in ("T1-curve", "T2-policy-map")
                for s in PROTOCOL["series"] for k in range(3)] + [{"kind": "cold", "task": "T-ref", "series": "1s-week", "rep": 0}]
        random.Random(seed + 1000).shuffle(jobs)
        run_items(jobs, hd, "cpu", build, freeze, timeouts, ref, watch, env, host=host, kind_label="cpu-host")

    watch.stop.set()
    if env["status"] == "running":
        env["status"] = "complete"
    env["finished"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(d / "env.json", env)
    archive(d)
    print(f"launch done: {d} ({env['status']})")


if __name__ == "__main__":
    main()

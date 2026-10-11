"""One formal launch (Part III): Stage A (CPU) or Stage D (CPU, GPU, or the CPU size control). Called by run_launch.sh.

    launch.py --stage A --launch I --label L [--parity SET]
    launch.py --stage D --role cpu|gpu|size-control --launch I --label L --spent-usd X

Every call writes a new attempt directory of its slot (formal.slot_dir: stage-a/launch-<i>/attempt-<k>,
stage-d/pair-<i>/<role>/attempt-<k>, stage-d/size-control/attempt-<k>); nothing is resumed or overwritten. An
interrupted or invalid attempt is replaced by the next attempt of the same slot (same order, same seeds).

Preconditions (each stops the launch before any timing, recorded in env.json): the scaling decision is complete,
committed, proceeds, and every scaling action has its P4 verification; this machine matches the profile named for the
role (instance type, region, CPU/GPU model); the page cache can actually be dropped. Stage D also needs: the committed
freeze, the same code fingerprint as the freeze, the passed full device gate D0 of the frozen code (GPU), and room in
the budget for the next complete pair (spent so far, the larger of --spent-usd and the recorded EC2 costs, plus the
frozen pair estimate, within the cap).

Steps:
  1  env.json (manifest, identity, capabilities, code fingerprint), attempts.jsonl ('started' record of every run)
  2  gate on this instance: CPU launches sf_test, sf_test2, CPU gate (items 1, 3-8), item 9, intervals, stream;
     GPU launches the device gate items 1, 2, 7, 8 with the frozen configuration. A failed gate stops the launch
  3  Stage A: thread calibration per (series, grid) and rates at every task length with that grid's configuration
     Stage D: boot-cold (descriptive); size control: its own thread calibration
  4  all conditions in one order shuffled with seed 20261010 + launch index, written before the first run.
     Cold repetitions are single items; warm repetitions of a condition are one block (priming outside the timing)
  5  Stage D, pair 1: the D1 points (T-diag, 3 x 5 s)
  6  Stage D, GPU launches: CPU-host (descriptive), threads chosen on this host per (series, grid) by the same design
  7  spot verification (outside the timing), archive of large outputs, SHA256SUMS
A spot interruption notice (polled every 5 s) marks the attempt incomplete.
"""

import argparse
import datetime as dt
import json
import random
import subprocess
import sys
import tarfile
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from p2 import cost, design, formal, harness, machine, validate  # noqa: E402
from p2.common import (FORMAL, GRID_OF_TASK, PROTOCOL, RESULTS, ROOT, append_jsonl, code_fingerprint,  # noqa: E402
                       committed_unchanged, new_attempt_dir, read_json, run_logged, sha256_file, write_json)
from p2.tasks import calib_task, main_task  # noqa: E402

GRIDS = FORMAL["thread_calibration"]["grids"]


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


def task_key(task, series):
    return f"{series}/{GRID_OF_TASK.get(task, 'P24')}"


def config_for(role, task, series, freeze, host=None):
    """The configuration of a role for one task: per (series, grid) threads for the CPU, per (series, grid) batch
    and host threads for the GPU (Stage A and the size control pass their own thread choice as host)."""
    key = task_key(task, series)
    if role in ("cpu", "size-control") or host:
        c = host or freeze["cpu"]
        return harness.cpu_config(c["threads"][key], parity=c.get("parity", freeze["cpu"]["parity"]),
                                  warmup=c.get("warmup_trials", 0))
    g = freeze["gpu"]
    return harness.gpu_config(g["path"], g["microbatch"][key], g["host_threads"][key], g["submission"],
                              g["warmup_trials"])


def gate_step(role, d, build, freeze):
    gd = d / "gate"
    log = gd / "gate.log"
    res = {}
    if role in ("cpu", "size-control"):
        par = freeze["cpu"]["parity"]
        threads = machine.topology()["logical"]
        py = sys.executable
        res["sf_test"] = run_logged([build / "sf_test"], log).returncode == 0
        res["sf_test2"] = run_logged([build / "sf_test2"], log).returncode == 0
        res["sf_gate_cpu"] = run_logged([build / "sf_gate", "--device", "cpu", "--threads", threads, "--parity", par,
                                         "--ref-parity", par, "--items", "1,3,4,5,6,7,8", "--out",
                                         gd / "gate_cpu.json"], log).returncode == 0
        res["item9"] = run_logged([py, HERE / "check_phase1_hits.py", "--build", build, "--out",
                                   gd / "item9.json"], log).returncode == 0
        res["intervals"] = run_logged([py, HERE / "check_intervals.py", "--build", build, "--out",
                                       gd / "intervals.json"], log).returncode == 0
        res["stream"] = run_logged([py, HERE / "stream.py", "--check"], log).returncode == 0
        if res["sf_gate_cpu"]:
            res["gate_cpu_complete"] = not validate.gate_problems(gd / "gate_cpu.json", device="cpu", full=True)
    else:
        g = freeze["gpu"]
        k = "1s-week/P24"
        res["sf_gate_cuda"] = run_logged([build / "sf_gate", "--device", "cuda", "--path", g["path"], "--microbatch",
                                          g["microbatch"][k], "--threads", g["host_threads"][k], "--items", "1,2,7,8",
                                          "--out", gd / "gate_cuda.json"], log).returncode == 0
    write_json(gd / "gate_summary.json", res)
    return all(res.values()), res


def spot(rec, build, cfg):
    try:
        r = subprocess.run([str(build / "sf_run"), rec["task_file"], "--spot", rec["out"], *harness.sf_args(cfg)],
                           capture_output=True, text=True, cwd=ROOT, timeout=3600)
        return json.loads(r.stdout.strip().splitlines()[-1]).get("mismatches")
    except (ValueError, IndexError, OSError, subprocess.TimeoutExpired):
        return None


def archive(d):
    """Large outputs into outputs_bits.tar.gz (never overwritten), then SHA256SUMS of every file."""
    big = sorted(d.rglob("*.bin"))
    tar_path = d / "outputs_bits.tar.gz"
    if big:
        if tar_path.exists():
            raise FileExistsError(f"{tar_path} exists; an archive is written once per attempt")
        with tarfile.open(tar_path, "w:gz") as tar:
            for p in big:
                tar.add(p, arcname=str(p.relative_to(d)))
        with tarfile.open(tar_path) as tar:  # read back before deleting anything
            names = set(tar.getnames())
        if names != {str(p.relative_to(d)) for p in big}:
            raise RuntimeError("archive does not contain every output; nothing deleted")
        for p in big:
            p.unlink()
    lines = [f"{sha256_file(p)}  ./{p.relative_to(d)}" for p in sorted(x for x in d.rglob("*")
                                                                      if x.is_file() and x.name != "SHA256SUMS")]
    (d / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def deadline_timeout(B):
    return 2 * B + 120


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

        def keep(rec):
            if rec.get("status") == "ok" and c["task"] in ("T1-curve", "T2-policy-map"):
                rec["spot_mismatches"] = spot(rec, build, spot_cfg)
                if rec["spot_mismatches"] != 0:
                    rec["status"] = "spot_mismatch"
            append_jsonl(runs, rec)
            print(f"[{pos + 1}/{len(items)}] {key} rep {rec.get('rep')}: {rec.get('e2e_s', float('nan')):.3f} s "
                  f"{rec.get('status')} {rec.get('cold_kind', '')}", flush=True)

        if c["kind"] == "cold":
            keep(harness.run_cold(tf, out, cfg, {**cond, "rep": c["rep"]}, build=build, timeout_s=to,
                                  expected_sha=expected))
        elif c["kind"] == "deadline":
            dev = "gpu" if role == "gpu" and not host else "cpu"
            keep(harness.run_cold(tf, out, cfg, {**cond, "rep": c["rep"], "budget_s": c["budget_s"]}, build=build,
                                  budget_s=c["budget_s"], drain_ms=freeze[dev]["drain_ms"], replicate=c["replicate"],
                                  timeout_s=deadline_timeout(c["budget_s"])))
        else:
            harness.warm_block(cfg, tf, out, c["reps"], cond, build=build, expected_sha=expected, timeout_s=to,
                               prime_timeout_s=to, on_record=keep)


def thread_choice(series_grids, cands, log, build, label, seed):
    """Per (series, grid) thread choice on this machine by the short + sustained design."""
    out = {}
    for i, (series, grid) in enumerate(series_grids):
        r = design.choose(series, grid, cands, log, build=build, label=label, seed=seed + i)
        if grid in GRIDS["set_task_threads"]:
            if not r.get("config"):
                raise SystemExit(f"thread calibration failed for {series}/{grid}: {r.get('failed')}")
            out[f"{series}/{grid}"] = r["config"]["threads"]
    return out


def budget_check(label, freeze, spent_arg):
    """Room for one more complete launch pair: max(operator figure, recorded EC2 costs) + pair estimate <= cap."""
    dec = read_json(formal.DECISION)
    roots = [formal.formal_dir(label)] + [RESULTS / "pilot" / x for x in freeze.get("verification_labels", [])]
    recorded, missing = cost.recorded_usd(roots)
    spent = max(spent_arg, recorded + (dec["budget"]["spent_usd"] or 0))
    need = spent + freeze["pair_cost_usd"]
    rec = {"spent_usd_operator": spent_arg, "spent_usd_recorded": recorded, "records_without_cost": missing,
           "spent_usd_used": spent, "pair_cost_usd": freeze["pair_cost_usd"], "cap_usd": dec["budget"]["cap_usd"],
           "fits": need <= dec["budget"]["cap_usd"]}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["A", "D"], required=True)
    ap.add_argument("--role", choices=formal.ROLES, default="cpu")
    ap.add_argument("--launch", type=int, required=True, help="launch index i (launch pair i in Stage D)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--build", type=Path, default=ROOT / "build")
    ap.add_argument("--parity", default="none", help="Stage A: CPU parity set")
    ap.add_argument("--spent-usd", type=float, help="Stage D: total spent so far from the billing console")
    ap.add_argument("--dry-run", action="store_true", help="env.json and order.json only")
    a = ap.parse_args()
    build = a.build.resolve()
    if a.stage == "A" and (a.role != "cpu" or a.launch not in (1, 2)):
        sys.exit("Stage A has two CPU launches, 1 and 2")
    rec = formal.decision()
    prof_name, prof = formal.profile_of(rec, a.role)
    fd = formal.formal_dir(a.label)
    code = code_fingerprint()
    pre = []  # precondition failures
    if a.stage == "D":
        fz = fd / "protocol-freeze.json"
        if not fz.exists() or not committed_unchanged(fz):
            sys.exit("Stage D needs a committed protocol-freeze.json (freeze.py)")
        freeze = read_json(fz)
        freeze["_sha256"] = sha256_file(fz)
        if freeze["scaling_decision_sha256"] != sha256_file(formal.DECISION):
            sys.exit("the scaling decision changed after the freeze")
        if freeze["code_sha256"] != code:
            sys.exit("the code differs from the frozen code (code fingerprint); re-freeze and re-run D0")
        if a.role != "size-control" and not 1 <= a.launch <= freeze["m"]:
            sys.exit(f"pair {a.launch} is outside the registered pairs 1..{freeze['m']}")
        if a.role == "gpu":
            pr = validate.gate_problems(fd / "d0" / "gate_cuda.json", device="cuda", full=True,
                                        path_name=freeze["gpu"]["path"], code=freeze["code_sha256"])
            if pr or not committed_unchanged(fd / "d0" / "gate_cuda.json"):
                sys.exit(f"D0 full gate missing, uncommitted or not passed: {pr}")
        if a.spent_usd is None:
            sys.exit("--spent-usd (total spent so far, from the billing console) is required in Stage D")
    else:
        freeze = {"cpu": {"parity": a.parity, "threads": {}, "warmup_trials": 0, "drain_ms": 0}, "timeouts_s": {},
                  "reference_sha256": {}, "code_sha256": code}
        refp = fd / "reference" / "hashes.json"
        if refp.exists():
            freeze["reference_sha256"] = read_json(refp)
    slot = formal.slot_dir(a.label, a.stage, a.role, a.launch)
    if formal.attempts(slot):
        done, _ = validate.chosen(slot, a.stage, a.role, a.launch, freeze if a.stage == "D" else None,
                                  sha256_file(formal.DECISION))
        if done:
            sys.exit(f"{slot} already has a valid attempt ({done.name}); a valid slot is never re-run")
    d = new_attempt_dir(slot)
    harness.ATTEMPT_LOG = d / "attempts.jsonl"
    env = machine.environment(prof_name, build)
    env.update({"stage": a.stage, "role": a.role, "launch_index": a.launch, "attempt": d.name,
                "started": dt.datetime.now(dt.timezone.utc).isoformat(),
                "scaling_decision_sha256": sha256_file(formal.DECISION),
                "freeze_sha256": freeze.get("_sha256"), "reference_available": bool(freeze["reference_sha256"])})
    env["status"] = "running"
    if env["identity"]["match"] is not True:
        pre.append("hardware mismatch")
    if not env["capabilities"]["drop_page_cache"]:
        pre.append("not run: page cache cannot be dropped (formal cold runs are cache-cold)")
    if a.stage == "D" and a.role != "size-control" and formal.first_role(a.launch) == a.role:
        b = budget_check(a.label, freeze, a.spent_usd)
        env["budget_check"] = b
        if not b["fits"]:
            pre.append("not run: budget (the next complete pair does not fit the cap); the comparison is incomplete")
    if pre:
        env["status"] = "; ".join(pre)
    write_json(d / "env.json", env)
    if pre and not a.dry_run:
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
    topo = env["topology"]
    pairs = [(s, g) for s in PROTOCOL["series"] for g in GRIDS["set_task_threads"] + GRIDS["report_only"]]

    ok, res = gate_step(a.role, d, build, freeze)
    if not ok:
        env["status"] = "gate failed: no timings"
        write_json(d / "env.json", env)
        archive(d)
        sys.exit(f"gate failed: {res}")

    if a.stage == "A":
        cands = [harness.cpu_config(n, parity=a.parity) for n in formal.thread_candidates(prof, topo)]
        threads = thread_choice(pairs, cands, cal, build, "threads", seed)
        freeze["cpu"]["threads"] = threads
        write_json(d / "threads.json", {"threads": threads, "candidates": [c["threads"] for c in cands]})
        for series in PROTOCOL["series"]:
            for grid in GRIDS["set_task_threads"]:
                cfg = harness.cpu_config(threads[f"{series}/{grid}"], parity=a.parity)
                for L in PROTOCOL["tasks"]["T1-curve"]["lengths"]:
                    for rep in range(3):
                        design.bench_point(calib_task(series, grid, L), cfg, 5, d / "rates.jsonl", build=build,
                                           label="rate", rep=rep)
    else:
        if a.role == "size-control":  # its own thread choice on this machine; parity carried (recorded)
            cands = [harness.cpu_config(n, parity=freeze["cpu"]["parity"]) for n in formal.thread_candidates(prof, topo)]
            host = {"threads": thread_choice(pairs, cands, cal, build, "threads", seed),
                    "parity": freeze["cpu"]["parity"], "warmup_trials": freeze["cpu"]["warmup_trials"],
                    "parity_note": "parity set of the formal CPU, not re-calibrated on the size control"}
            freeze = {**freeze, "cpu": {**freeze["cpu"], **host}}
            write_json(d / "threads.json", host)
        tf = main_task("T1-curve", "1m-quarter")
        append_jsonl(d / "runs.jsonl", harness.run_cold(
            tf, d / "outputs" / "boot-cold", config_for(a.role, "T1-curve", "1m-quarter", freeze),
            {"kind": "boot-cold", "task": "T1-curve", "series": "1m-quarter", "task_file": str(tf), "rep": 0},
            build=build, expected_sha=ref.get("T1-curve/1m-quarter")))

    run_items(items, d, a.role, build, freeze, timeouts, ref, watch, env)

    if a.stage == "D" and a.launch == 1 and a.role in ("cpu", "gpu") and env["status"] == "running":
        td = PROTOCOL["t_diag"]
        for f in sorted((HERE / "tasks" / "tdiag").glob("*.json")):
            if f.name.startswith("cross_") and a.role != "gpu":
                continue
            cfg = config_for(a.role, "T-diag", "1s-week", freeze)
            batches = [None]
            if a.role == "gpu" and f.name.startswith("cross_"):
                batches = td["gpu_cross"]["batch"]
            elif a.role == "gpu" and f.name == f"count_24_n{td['gpu_cross']['n']}.json":
                batches = td["gpu_batch"]
            for b in batches:
                c = cfg if b is None else {**cfg, "microbatch": freeze["gpu"]["b_max"] if b == "b_max" else b}
                for rep in range(td["repetitions"]):
                    design.bench_point(f, c, td["seconds"], d / "tdiag.jsonl", build=build, label="tdiag", rep=rep)

    if a.role == "gpu" and a.stage == "D" and env["status"] == "running":
        hd = d / "cpu_host"
        hd.mkdir()
        cands = [harness.cpu_config(n, parity=freeze["cpu"]["parity"])
                 for n in sorted({topo["physical_cores"], topo["logical"]})]
        host = {"threads": thread_choice(pairs, cands, hd / "calibration.jsonl", build, "cpu-host", seed + 1000),
                "parity": freeze["cpu"]["parity"], "warmup_trials": freeze["cpu"]["warmup_trials"],
                "parity_note": "parity set of the formal CPU, not re-calibrated on the GPU host (descriptive CPU-host)"}
        write_json(hd / "threads.json", host)
        jobs = [{"kind": "cold", "task": t, "series": s, "rep": k} for t in ("T1-curve", "T2-policy-map")
                for s in PROTOCOL["series"] for k in range(3)] + [{"kind": "cold", "task": "T-ref", "series": "1s-week", "rep": 0}]
        random.Random(seed + 1000).shuffle(jobs)
        write_json(hd / "order.json", {"seed": seed + 1000, "order": jobs})
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

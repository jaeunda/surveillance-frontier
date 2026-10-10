"""Part I of the protocol: the pilot on an available machine (P0-P2), and the same workload on a target machine for
the scaling verification (P4). Results are descriptive and labelled with their machine; they are never pooled with
formal launches.

    pilot.py plan     --label L --machine lab-desktop --hours 12 [--p-cores 0-5] [--sessions 2] [--cut "reason"]
                      [--skip-step tref-build ...] [--purpose verification --scaling-record FILE]
    pilot.py amend    --label L --reason "..." --set key=json ...      before the measurement it affects
    pilot.py check    --label L [--paths inc,hyb]                        P0: CPU regression, stream, intervals, CUDA gate
    pilot.py run      --label L --session K --step STEP                 P1/P2 steps (see protocol.json pilot.steps)
    pilot.py summary  --label L                                          pilot-summary.json / .md (all sessions)

Results: results/pilot/<label>/{pilot-plan.json, amendments.jsonl, checks/, hashes.json, session-<K>/, pilot-summary.*}.
Every run appends one line to session-<K>/runs.jsonl or bench.jsonl, including failed, late, and slow runs.
"""

import argparse
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from p2 import design, harness, machine, projection  # noqa: E402
from p2.common import (FORMAL, MACHINES, PILOT, PROTOCOL, RESULTS, ROOT, append_jsonl, git_state,  # noqa: E402
                       input_hashes, now, read_json, read_jsonl, run_logged, sha256_file, task_hashes, write_json)
from p2.stats import describe  # noqa: E402
from p2.tasks import calib_task, main_task, point  # noqa: E402

STEPS = list(PILOT["steps"])
CPU_STEPS = {"cpu-baseline", "threads", "rates", "diag", "tref-rates", "tref-build", "deadline"}
GPU_STEPS = {"gpu-rates", "microbatch", "gpu-tasks"}
SEED = PROTOCOL["stream"]["seed"]


def label_dir(label):
    return RESULTS / "pilot" / label


# ---- plan ----

def effective_plan(d):
    """The plan with its amendments applied in order; plan['amendments'] counts them."""
    plan = read_json(d / "pilot-plan.json")
    amendments = read_jsonl(d / "amendments.jsonl")
    for a in amendments:
        for key, value in a["set"].items():
            node = plan
            *path, last = key.split(".")
            for k in path:
                node = node.setdefault(k, {})
            node[last] = value
    plan["amendments"] = len(amendments)
    return plan


def cmd_plan(a):
    d = label_dir(a.label)
    if (d / "pilot-plan.json").exists():
        sys.exit(f"{d / 'pilot-plan.json'} exists; a plan is never rewritten (use amend)")
    if a.machine not in MACHINES:
        sys.exit(f"unknown machine profile {a.machine}; profiles: {', '.join(MACHINES)}")
    if a.purpose == "pilot" and a.machine not in PILOT["machines"]:
        sys.exit(f"pilot machines are {PILOT['machines']}; a cloud profile is used with --purpose verification")
    if a.purpose == "verification" and not a.scaling_record:
        sys.exit("--purpose verification needs --scaling-record (the committed scaling decision)")
    topo = machine.topology()
    ident = machine.identity(a.machine)
    if ident["match"] is False:
        sys.exit(f"this machine does not match profile {a.machine}: {ident}")
    if MACHINES[a.machine]["kind"] == "available" and a.machine == "lab-desktop" and not (a.p_cores or topo["p_cores"]):
        print("note: P/E core mapping unknown here; the P-core candidate is left out until --p-cores is given "
              "(check the mapping on the Windows host)", file=sys.stderr)
    steps = [s for s in STEPS if s not in a.skip_step]
    reps = dict(PILOT["reps_per_session"])
    plan = {
        "created": now(),
        "purpose": a.purpose,
        "label": a.label,
        "machine": a.machine,
        "machine_profile": MACHINES[a.machine],
        "identity": ident,
        "topology": topo,
        "gpu": machine.gpu_state(),
        "capabilities": machine.capabilities(),
        "git": git_state(),
        "binaries": machine.build_info(a.build)["binary_sha256"],
        "inputs_sha256": input_hashes(),
        "tasks_sha256": task_hashes(),
        "protocol_sha256": sha256_file(HERE / "protocol.json"),
        "scaling_record": None if not a.scaling_record else {"path": a.scaling_record,
                                                                "sha256": sha256_file(a.scaling_record)},
        "hours_available": a.hours,
        "sessions": a.sessions,
        "steps": steps,
        "skipped_steps": a.skip_step,
        "fixed_tasks": PILOT["fixed_tasks"],
        "reps_per_session": reps,
        "cpu_baseline": harness.cpu_config(topo["logical"], name="all"),
        "thread_candidates": machine.thread_candidates(topo, a.p_cores),
        "microbatch_sweep": PILOT["microbatch_sweep"],
        "bench": PILOT["bench"],
        "stop_rules": [
            "a step does not start when the session's elapsed hours plus the hours already used reach hours_available; "
            "it is recorded as 'not started (time)'",
            "tref-build starts only if its conservative projection fits the remaining hours",
            "a failed, late, timed-out or out-of-memory run is recorded and the step continues with its next run",
            "a run with min MemAvailable < 2 GiB or any swap-in is flagged 'memory pressure' in the summary",
            "an identity mismatch stops the step and all GPU steps of that path",
        ],
        "cut": a.cut,
    }
    if a.purpose == "pilot" and a.sessions < PILOT["sessions"] and not a.cut:
        sys.exit(f"fewer than {PILOT['sessions']} sessions needs --cut with the reason, recorded before the first run")
    write_json(d / "pilot-plan.json", plan)
    print(f"plan written: {d / 'pilot-plan.json'} ({len(steps)} steps, {a.sessions} sessions, {a.hours} h)")
    print("commit it before the first measurement so that the plan precedes the results")


def cmd_amend(a):
    d = label_dir(a.label)
    if not (d / "pilot-plan.json").exists():
        sys.exit("no plan")
    sets = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        sets[k] = json.loads(v)
    started = [p.name for p in sorted(d.glob("session-*")) if (p / "steps.jsonl").exists()]
    append_jsonl(d / "amendments.jsonl", {"time": now(), "reason": a.reason, "set": sets,
                                          "sessions_with_runs_before": started})
    print(f"amendment {len(read_jsonl(d / 'amendments.jsonl'))} recorded")


# ---- P0 checks ----

def cmd_check(a):
    d = label_dir(a.label)
    plan = effective_plan(d)
    c = d / "checks"
    log = c / "checks.log"
    build = a.build
    res = {"time": now(), "machine": plan["machine"], "env": machine.environment(plan["machine"], build)}
    write_json(c / "env.json", res.pop("env"))
    py = sys.executable
    items = {
        "sf_test": [build / "sf_test"],
        "sf_test2": [build / "sf_test2"],
        "intervals": [py, HERE / "check_intervals.py", "--build", build, "--out", c / "intervals.json"],
        "stream": [py, HERE / "stream.py", "--check"],
        "harness_tests": [py, HERE / "test_p2.py"],
        "gate_cpu": [build / "sf_gate", "--device", "cpu", "--threads", plan["topology"]["logical"], "--items",
                     "1,6,7,8", "--out", c / "gate_cpu.json"],
    }
    if not a.skip_phase1_hits:
        items["phase1_hits"] = [py, HERE / "check_phase1_hits.py", "--build", build, "--out", c / "item9.json"]
    out = {}
    for name, cmd in items.items():
        r = run_logged(cmd, log)
        out[name] = r.returncode == 0
        print(f"{name:<16} {'pass' if out[name] else 'FAIL'}", flush=True)
    res["cpu"] = out
    res["cpu_pass"] = all(out.values())
    # CUDA: the partial gate on the device itself. A build without SF_CUDA, or no GPU, is 'unavailable', and an
    # emulation pass never counts as device execution.
    gpu = {}
    has_cuda = "ON" in machine.build_info(build)["cmake_cache"].get("SF_CUDA", "")
    for path in a.paths.split(","):
        if not has_cuda or not plan["gpu"].get("present"):
            gpu[path] = {"execution": "unavailable", "pass": False,
                         "reason": "build without SF_CUDA=ON" if not has_cuda else "no GPU visible"}
            continue
        r = run_logged([build / "sf_gate", "--device", "cuda", "--path", path, "--microbatch", "64", "--partial",
                        "--out", c / f"gate_cuda_{path}.json"], log)
        gpu[path] = {"execution": "device", "pass": r.returncode == 0, "returncode": r.returncode,
                     "stderr_tail": r.stderr[-1000:]}
        print(f"cuda {path:<11} {'pass' if gpu[path]['pass'] else 'FAIL'} (device execution)", flush=True)
    res["gpu"] = gpu
    res["validated_paths"] = [p for p, v in gpu.items() if v["pass"]]
    write_json(c / "checks.json", res)
    sys.exit(0 if res["cpu_pass"] else 1)


# ---- P1/P2 steps ----

class Session:
    def __init__(self, label, k, build):
        self.d = label_dir(label)
        self.plan = effective_plan(self.d)
        if k < 1 or k > self.plan["sessions"]:
            sys.exit(f"session {k} not in the plan (1..{self.plan['sessions']})")
        self.k, self.build = k, build
        self.sd = self.d / f"session-{k}"
        self.checks = read_json(self.d / "checks" / "checks.json") if (self.d / "checks" / "checks.json").exists() else None
        self.runs, self.benches = self.sd / "runs.jsonl", self.sd / "bench.jsonl"
        if not (self.sd / "env.json").exists():
            write_json(self.sd / "env.json", machine.environment(self.plan["machine"], build))
        self.hashes_path = self.d / "hashes.json"

    # identity of fixed-n outputs: the first complete CPU result of a task is the reference within this label
    def expected(self, task, series):
        h = read_json(self.hashes_path) if self.hashes_path.exists() else {}
        return h.get(f"{task}/{series}")

    def remember(self, task, series, rec):
        h = read_json(self.hashes_path) if self.hashes_path.exists() else {}
        key = f"{task}/{series}"
        if key not in h and rec["status"] == "ok" and rec["config"]["device"] == "cpu" and rec.get("spot_mismatches") == 0:
            h[key] = rec["result_sha256"]
            write_json(self.hashes_path, h)

    def hours_used(self):
        used = 0.0
        for s in sorted(self.d.glob("session-*/steps.jsonl")):
            for r in read_jsonl(s):
                used += r.get("elapsed_s", 0) / 3600
        return used

    def decision(self, kind):
        """Latest decision of a kind in any session of this label (threads, microbatch)."""
        found = None
        for s in sorted(self.d.glob("session-*/bench.jsonl")):
            for r in read_jsonl(s):
                if r.get("kind") == kind:
                    found = r
        return found

    def cpu_config(self):
        """CPU configuration for rates, diag and the reference: the threads decision for 1s-week/P24, else all."""
        dec = self.decision("threads-decision")
        if dec and dec.get("config"):
            return dec["config"]
        return self.plan["cpu_baseline"]

    def validated_paths(self):
        if not self.checks:
            return []
        return [p for p in self.checks["validated_paths"] if not self.path_stopped(p)]

    def path_stopped(self, path):
        for s in sorted(self.d.glob("session-*/runs.jsonl")):
            for r in read_jsonl(s):
                if r.get("status") == "identity_mismatch" and r["config"].get("path") == path:
                    return True
        return False

    def gpu_config(self, path):
        dec = self.decision(f"microbatch-decision-{path}")
        mb = dec["choice"] if dec and dec.get("choice") else 64
        return harness.gpu_config(path, mb, host_threads=self.plan["topology"]["logical"])

    def spot(self, rec):
        r = subprocess.run([str(self.build / "sf_run"), rec["task_file"], "--spot", rec["out"],
                            *harness.sf_args(self.plan["cpu_baseline"])], capture_output=True, text=True, cwd=ROOT)
        try:
            return json.loads(r.stdout.strip().splitlines()[-1]).get("mismatches")
        except (ValueError, IndexError):
            return None


def safe(name):
    return name.replace(":", "_").replace("@", "_").replace(",", "-")


def fixed_task_runs(s, step, configs, seed):
    """Cold repetitions individually and warm repetitions as one block per condition (one priming task, then the
    timed requests), all in one seeded order written before the first run."""
    reps = s.plan["reps_per_session"]
    items = []
    for cfg in configs:
        for task, series in s.plan["fixed_tasks"]:
            items += [{"kind": "cold", "task": task, "series": series, "rep": r, "config": cfg}
                      for r in range(reps["cold"])]
            if reps["warm"]:
                items.append({"kind": "warm-block", "task": task, "series": series, "reps": reps["warm"], "config": cfg})
    random.Random(seed).shuffle(items)
    write_json(s.sd / f"order_{step}.json", {"seed": seed, "order": items})
    for pos, it in enumerate(items):
        tf = main_task(it["task"], it["series"])
        cond = {"step": step, "session": s.k, "task": it["task"], "series": it["series"], "task_file": str(tf),
                "position": pos}
        out = s.sd / "outputs" / step / safe(f"{pos:03d}_{it['task']}_{it['series']}_{harness.config_key(it['config'])}")
        if it["kind"] == "cold":
            recs = [harness.run_cold(tf, out, it["config"], {**cond, "rep": it["rep"]}, build=s.build,
                                     expected_sha=s.expected(it["task"], it["series"]))]
        else:
            recs = []
            try:
                srv = harness.WarmServer(it["config"], build=s.build)
                try:
                    srv.prime(tf, Path(f"{out}_prime"))
                    for r in range(it["reps"]):
                        recs.append(srv.timed(tf, Path(f"{out}_w{r}"), {**cond, "rep": r},
                                              expected_sha=s.expected(it["task"], it["series"])))
                finally:
                    srv.close()
            except (RuntimeError, OSError) as e:
                recs.append({"mode": "warm", **cond, "config": it["config"], "config_key": harness.config_key(it["config"]),
                             "status": "error", "error": repr(e)})
        for rec in recs:
            if rec.get("status") == "ok":
                rec["spot_mismatches"] = s.spot(rec)
                if rec["spot_mismatches"] != 0:
                    rec["status"] = "spot_mismatch"
            s.remember(it["task"], it["series"], rec)
            append_jsonl(s.runs, rec)
            print(f"[{pos + 1}/{len(items)}] {rec['mode']:<5} {it['task']} {it['series']} {rec['config_key']}: "
                  f"{rec.get('e2e_s', float('nan')):.2f} s {rec['status']} {rec.get('cold_kind', '')}", flush=True)
            if rec.get("status") == "identity_mismatch":
                raise SystemExit("identity mismatch: the output differs from the CPU result; step stopped")


def step_cpu_baseline(s, seed):
    fixed_task_runs(s, "cpu-baseline", [s.plan["cpu_baseline"]], seed)


def step_threads(s, seed):
    cands = [harness.cpu_config(c["threads"], c["cpus"], name=c["name"]) for c in s.plan["thread_candidates"]]
    for series in PROTOCOL["series"]:
        for grid in ("P1", "P24"):
            rec = design.choose(series, grid, cands, s.benches, build=s.build, label="threads-grid")
            print(f"threads {series} {grid}: {rec.get('choice')} (flip {rec.get('flip')})", flush=True)
    # the configuration used for rates, diag and the reference: the week P24 choice
    week = [r for r in read_jsonl(s.benches) if r.get("kind") == "threads-grid-decision"
            and r["series"] == "1s-week" and r["grid"] == "P24"][-1]
    append_jsonl(s.benches, {**week, "kind": "threads-decision", "rule": "choice at 1s-week P24"})


def configs_for_rates(s):
    cfgs = [s.cpu_config()]
    cfgs += [s.gpu_config(p) for p in s.validated_paths()]
    return cfgs


def step_rates(s, seed):
    b = s.plan["bench"]
    lengths = PROTOCOL["tasks"]["T1-curve"]["lengths"]
    for cfg in configs_for_rates(s):
        for series in PROTOCOL["series"]:
            for grid in ("P1", "P24"):
                for L in lengths:
                    for rep in range(b["repetitions"]):
                        r = design.bench_point(calib_task(series, grid, L), cfg, b["seconds"], s.benches, build=s.build,
                                               label="rate", rep=rep, sample=cfg["device"] == "cuda")
                        print(f"rate {harness.config_key(cfg)} {series} {grid} L={L}: {r.get('trials_per_s')}",
                              flush=True)


def step_gpu_rates(s, seed):
    b = s.plan["bench"]
    for path in s.validated_paths():
        for mb in (64, "auto"):
            cfg = harness.gpu_config(path, mb, s.plan["topology"]["logical"])
            for name, pt in PILOT["gpu_points"].items():
                for rep in range(b["repetitions"]):
                    r = design.bench_point(point(pt), cfg, b["seconds"], s.benches, build=s.build, label="gpu-rate",
                                           rep=rep, point_name=name, sample=True)
                    print(f"gpu {path} b={mb} {name}: {r.get('trials_per_s')} {r['status']}", flush=True)


def step_microbatch(s, seed):
    b = s.plan["bench"]
    rep_point = point(PILOT["gpu_points"]["representative"])
    for path in s.validated_paths():
        probe = design.bench_point(rep_point, harness.gpu_config(path, "auto", s.plan["topology"]["logical"]), 1,
                                   s.benches, build=s.build, label="microbatch-probe", sample=True)
        b_max = (probe.get("device") or {}).get("b_max")
        sweep = [m for m in s.plan["microbatch_sweep"] if b_max is None or m <= b_max]
        if b_max is not None and b_max not in sweep:
            sweep.append(b_max)
        rates = {}
        for mb in sweep:
            cfg = harness.gpu_config(path, mb, s.plan["topology"]["logical"])
            recs = [design.bench_point(rep_point, cfg, b["seconds"], s.benches, build=s.build, label="microbatch",
                                       rep=rep, sample=True, b_max_at_probe=b_max) for rep in range(b["repetitions"])]
            rates[mb] = design.median_rate(recs)
            mem = projection.memory_check(recs[-1].get("device"))
            print(f"microbatch {path} b={mb}: {rates[mb]} ({recs[-1]['status']}; memory {mem})", flush=True)
        ok = {m: r for m, r in rates.items() if r}
        choice = max(ok, key=ok.get) if ok else None
        append_jsonl(s.benches, {"kind": f"microbatch-decision-{path}", "b_max": b_max, "median_rates": rates,
                                 "choice": choice, "rule": "highest median rate at the representative point"})


def step_gpu_tasks(s, seed):
    fixed_task_runs(s, "gpu-tasks", [s.gpu_config(p) for p in s.validated_paths()], seed)


def step_diag(s, seed):
    b = s.plan["bench"]
    pts = []
    for f in sorted((HERE / "tasks" / "tdiag").glob("*.json")):
        t = read_json(f)
        if t["tdiag"]["n"] in PROTOCOL["t_diag"]["heldout_n"]:
            continue  # held-out points of D1 are never run in the pilot
        pts.append(f)
    for cfg in configs_for_rates(s):
        for f in pts:
            if f.name.startswith("cross_") and cfg["device"] == "cpu":
                continue
            for rep in range(b["repetitions"]):
                design.bench_point(f, cfg, b["seconds"], s.benches, build=s.build, label="diag", rep=rep, diag=True,
                                   sample=cfg["device"] == "cuda")
        print(f"diag {harness.config_key(cfg)}: {len(pts)} points", flush=True)


def step_tref_rates(s, seed):
    b = s.plan["bench"]
    for cfg in configs_for_rates(s):
        for c in range(15):
            for rep in range(b["repetitions"]):
                r = design.bench_point(point(f"pilot/tref-cell_c{c}"), cfg, b["seconds"], s.benches, build=s.build,
                                       label="tref-rate", rep=rep, sample=cfg["device"] == "cuda")
            print(f"tref cell {c} {harness.config_key(cfg)}: {r.get('trials_per_s')}", flush=True)


def tref_projection(benches, cfg_key):
    cells = {}
    for r in benches:
        if r.get("kind") == "tref-rate" and r.get("config_key") == cfg_key and r.get("status") == "ok":
            cells.setdefault(r["meta"]["tref_cell"]["cell"], []).append(r["trials_per_s"])
    return projection.project_tref(cells)


def step_tref_build(s, seed):
    cfg = s.cpu_config()
    benches = [r for p in sorted(s.d.glob("session-*/bench.jsonl")) for r in read_jsonl(p)]
    proj = tref_projection(benches, harness.config_key(cfg))
    remaining = s.plan["hours_available"] - s.hours_used()
    if proj["status"] != "ok":
        return "not started (run tref-rates first)"
    need_h = proj["seconds"]["conservative"] / 3600 * 1.1
    if need_h > remaining:
        print(f"T-ref not started: conservative projection {need_h:.1f} h > remaining {remaining:.1f} h; the per-cell "
              "rates and the projection are the result", flush=True)
        return f"not started (time: projected {need_h:.2f} h > remaining {remaining:.2f} h)"
    ref = s.d / "reference"
    tf = main_task("T-ref", "1s-week")
    rec = harness.run_cold(tf, ref / "T-ref_1s-week", cfg, {"step": "tref-build", "session": s.k, "task": "T-ref",
                                                             "series": "1s-week", "task_file": str(tf), "rep": 0},
                           build=s.build)
    rec["projection_s"] = proj["seconds"]
    append_jsonl(s.runs, rec)
    print(f"T-ref: {rec['e2e_s'] / 3600:.2f} h {rec['status']}", flush=True)
    if rec["status"] == "ok":
        o = FORMAL["oracle"]
        r = run_logged([s.build / "sf_gate", "--oracle", tf, ref / "T-ref_1s-week", "--threads", cfg["threads"],
                        "--oracle-shared", o["starts_per_cell_shared"], "--oracle-full", o["starts_per_cell_full"],
                        "--out", ref / "oracle.json"], ref / "oracle.log")
        write_json(ref / "reference.json", {"result_sha256": rec["result_sha256"], "oracle_pass": r.returncode == 0,
                                            "config": cfg, "session": s.k, "git": git_state(),
                                            "binaries": machine.build_info(s.build)["binary_sha256"]})
        print(f"oracle: {'pass' if r.returncode == 0 else 'FAIL'}", flush=True)


def step_deadline(s, seed):
    dl = PILOT["deadline"]
    tf = point(dl["task"])
    cfgs = [s.cpu_config()] + [s.gpu_config(p) for p in s.validated_paths()]
    items = [(cfg, B, k) for cfg in cfgs for B in dl["budgets_s"] for k in range(dl["reps"])]
    random.Random(seed).shuffle(items)
    for pos, (cfg, B, k) in enumerate(items):
        out = s.sd / "outputs" / "deadline" / safe(f"{pos:03d}_B{B}_{harness.config_key(cfg)}")
        rec = harness.run_cold(tf, out, cfg, {"step": "deadline", "session": s.k, "task": "T-ref-est-prefix",
                                              "series": "week-prefix-75600", "task_file": str(tf), "rep": k},
                               build=s.build, budget_s=B, drain_ms=0, replicate=k + 1)
        try:
            pre = json.loads((out / "prefix.json").read_text())
            rec["exit_minus_stop_ms"] = (rec["t1_ns"] - pre["stop_ns"]) / 1e6
            rec["prefix_trials"] = pre.get("G")
        except (OSError, ValueError, KeyError):
            pass
        append_jsonl(s.runs, rec)
        print(f"deadline B={B} {rec['config_key']}: {rec['status']} exit-stop {rec.get('exit_minus_stop_ms')}", flush=True)


STEP_FN = {"cpu-baseline": step_cpu_baseline, "threads": step_threads, "rates": step_rates,
           "gpu-rates": step_gpu_rates, "microbatch": step_microbatch, "gpu-tasks": step_gpu_tasks,
           "diag": step_diag, "tref-rates": step_tref_rates, "tref-build": step_tref_build, "deadline": step_deadline}


def cmd_run(a):
    s = Session(a.label, a.session, a.build)
    if a.step not in s.plan["steps"]:
        sys.exit(f"step {a.step} is not in the plan {s.plan['steps']}")
    if s.checks is None or not s.checks["cpu_pass"]:
        sys.exit("P0 checks missing or failed (pilot.py check): no timed runs")
    if a.step in GPU_STEPS and not s.validated_paths():
        append_jsonl(s.sd / "steps.jsonl", {"step": a.step, "status": "not run (no GPU path passed the device gate)"})
        sys.exit("no GPU path passed the device gate")
    used = s.hours_used()
    if used >= s.plan["hours_available"]:
        append_jsonl(s.sd / "steps.jsonl", {"step": a.step, "status": "not started (time)", "hours_used": used})
        sys.exit(f"hours used {used:.2f} >= available {s.plan['hours_available']}")
    seed = SEED + 1000 * s.k + STEPS.index(a.step)
    t0 = time.monotonic()
    status = "complete"
    try:
        status = STEP_FN[a.step](s, seed) or "complete"
    except SystemExit as e:
        status = f"stopped: {e}"
        raise
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        append_jsonl(s.sd / "steps.jsonl", {"step": a.step, "status": status, "started_mono": t0,
                                            "elapsed_s": time.monotonic() - t0, "finished": now(),
                                            "plan_amendments": s.plan["amendments"]})


# ---- summary ----

def cmd_summary(a):
    d = label_dir(a.label)
    plan = effective_plan(d)
    runs, benches, sessions = [], [], {}
    for sd in sorted(d.glob("session-*")):
        k = int(sd.name.split("-")[1])
        env = read_json(sd / "env.json") if (sd / "env.json").exists() else {}
        sessions[k] = {"collected": env.get("collected"), "runtime": env.get("runtime"), "gpu": env.get("gpu"),
                       "cpu_model": (env.get("identity") or {}).get("cpu_model"), "meminfo": env.get("meminfo"),
                       "capabilities": env.get("capabilities"),
                       "steps": read_jsonl(sd / "steps.jsonl")}
        runs += read_jsonl(sd / "runs.jsonl")
        benches += read_jsonl(sd / "bench.jsonl")
    summary = {"label": a.label, "machine": plan["machine"], "purpose": plan["purpose"], "plan_amendments": plan["amendments"],
               "sessions": sessions, "hours_used": sum(r.get("elapsed_s", 0) for s in sessions.values()
                                                       for r in s["steps"]) / 3600,
               "checks": read_json(d / "checks" / "checks.json") if (d / "checks" / "checks.json").exists() else None}

    # end-to-end times of the fixed tasks: per condition and session, median and range; statuses kept
    e2e = {}
    for r in runs:
        if r.get("task") not in ("T1-curve", "T2-policy-map"):
            continue
        kind = r.get("cold_kind", "warm") if r["mode"] == "cold" else r["mode"]
        key = f"{r['task']}|{r['series']}|{kind}|{r['config_key']}"
        e = e2e.setdefault(key, {"task": r["task"], "series": r["series"], "mode": kind, "config": r["config_key"],
                                 "sessions": {}, "statuses": [], "memory_pressure": 0})
        e["statuses"].append(r["status"])
        if r.get("status") == "ok":
            e["sessions"].setdefault(str(r["session"]), []).append(r["e2e_s"])
        if (r.get("min_mem_available_mib") or 1e9) < 2048 or (r.get("pswpin") or 0) > 0:
            e["memory_pressure"] += 1
    for e in e2e.values():
        e["per_session"] = {k: describe(v) for k, v in e["sessions"].items()}
        e["all"] = describe([x for v in e["sessions"].values() for x in v])
        stages = [r["runner"]["stage_ms_sum"] for r in runs if r.get("runner") and r.get("status") == "ok"
                  and f"{r['task']}|{r['series']}" == f"{e['task']}|{e['series']}" and r["config_key"] == e["config"]]
        e["select_stage_ms_median"] = {k: statistics.median(s[k] for s in stages) for k in stages[0]} if stages else None
    summary["e2e"] = list(e2e.values())

    # decisions and sweeps
    summary["decisions"] = [r for r in benches if str(r.get("kind", "")).endswith("decision")
                            or str(r.get("kind", "")).startswith("microbatch-decision")]
    mb = {}
    for r in benches:
        if r.get("kind") == "microbatch":
            k = (r["config"]["path"], r["config"]["microbatch"])
            m = mb.setdefault(k, {"path": k[0], "microbatch": k[1], "rates": [], "statuses": [], "memory": None,
                                  "gpu_memory_used_mib_max": None})
            m["statuses"].append(r["status"])
            if r.get("trials_per_s"):
                m["rates"].append(r["trials_per_s"])
            m["memory"] = projection.memory_check(r.get("device")) or m["memory"]
            used = ((r.get("sampler") or {}).get("gpu") or {}).get("memory_used_mib_max")
            m["gpu_memory_used_mib_max"] = max(filter(None, [used, m["gpu_memory_used_mib_max"]]), default=None)
    summary["microbatch"] = [{**m, "rate": describe(m.pop("rates"))} for m in mb.values()]

    # projections: T1/T2 against observed warm E2E on the same configuration, then T-ref
    proj = []
    for cfg_key in sorted({r["config_key"] for r in benches if r.get("kind") in ("rate", "tref-rate")}):
        rates, setups = projection.rate_table([r for r in benches if r.get("kind") == "rate" and r["config_key"] == cfg_key])
        for task, series in plan["fixed_tasks"] if rates else []:
            p = projection.project_fixed(task, series, rates, warm=True)
            obs = [e["all"]["median"] for e in summary["e2e"] if e["task"] == task and e["series"] == series
                   and e["mode"] == "warm" and e["config"] == cfg_key and e["all"]]
            proj.append({"config": cfg_key, "task": task, "series": series, "projection_warm": p,
                         "observed_warm_median_s": obs[0] if obs else None,
                         "observed_over_projected_base": obs[0] / p["seconds"]["base"] if obs and p["status"] == "ok" else None})
        tp = tref_projection(benches, cfg_key)
        proj.append({"config": cfg_key, "task": "T-ref", "series": "1s-week", "projection_cold_loop": tp})
    summary["projections"] = proj
    summary["tref_build"] = read_json(d / "reference" / "reference.json") if (d / "reference" / "reference.json").exists() else None
    summary["tref_runs"] = [{k: r.get(k) for k in ("status", "e2e_s", "projection_s", "config_key")}
                            for r in runs if r.get("task") == "T-ref"]
    summary["deadline"] = [{k: r.get(k) for k in ("config_key", "budget_s", "status", "e2e_s", "exit_minus_stop_ms",
                                                   "prefix_trials")} for r in runs if r.get("step") == "deadline"]
    summary["failures"] = [{k: r.get(k) for k in ("step", "session", "task", "series", "mode", "config_key", "status",
                                                   "stderr_tail")} for r in runs + benches
                           if r.get("status") not in (None, "ok")]
    summary["source_sha256"] = {str(p.relative_to(d)): sha256_file(p) for p in sorted(d.glob("session-*/*.jsonl"))}
    write_json(d / "pilot-summary.json", summary)
    (d / "pilot-summary.md").write_text(summary_md(summary))
    print(summary_md(summary))


def summary_md(s):
    f = lambda x: "-" if x is None else f"{x:.3g}"  # noqa: E731
    lines = [f"# Pilot summary: {s['label']} ({s['machine']}, {s['purpose']})", "",
             f"Sessions: {len(s['sessions'])}; hours used {s['hours_used']:.2f}; plan amendments {s['plan_amendments']}. "
             "Descriptive results of one machine; not pooled with formal launches.", "",
             "## End-to-end time of the fixed tasks", "",
             "| task | series | mode | config | per session median [min, max] s | statuses | memory pressure |",
             "|---|---|---|---|---|---|---|"]
    for e in s["e2e"]:
        per = "; ".join(f"S{k}: {f(v['median'])} [{f(v['min'])}, {f(v['max'])}]" for k, v in e["per_session"].items() if v)
        st = ", ".join(f"{x}×{e['statuses'].count(x)}" for x in sorted(set(e["statuses"])))
        lines.append(f"| {e['task']} | {e['series']} | {e['mode']} | {e['config']} | {per or '-'} | {st} | "
                     f"{e['memory_pressure']} |")
    lines += ["", "## GPU microbatch sweep (representative point)", "",
              "| path | microbatch | median rate /s | statuses | planned MiB | reported MiB | nvidia-smi max MiB |",
              "|---|---|---|---|---|---|---|"]
    for m in s["microbatch"]:
        mem = m["memory"] or {}
        mib = lambda b: "-" if b is None else f"{b / 2**20:.0f}"  # noqa: E731
        lines.append(f"| {m['path']} | {m['microbatch']} | {f((m['rate'] or {}).get('median'))} | "
                     f"{','.join(sorted(set(m['statuses'])))} | {mib(mem.get('planned_bytes'))} | "
                     f"{mib(mem.get('reported_bytes'))} | {f(m['gpu_memory_used_mib_max'])} |")
    lines += ["", "## Projections", "", "| config | task | series | conservative / base / optimistic s | observed warm s | "
              "observed / base |", "|---|---|---|---|---|---|"]
    for p in s["projections"]:
        pr = p.get("projection_warm") or p.get("projection_cold_loop") or {}
        sec = pr.get("seconds")
        band = " / ".join(f(sec[k]) for k in projection.SCENARIOS) if sec else pr.get("status", "-")
        lines.append(f"| {p['config']} | {p['task']} | {p['series']} | {band} | {f(p.get('observed_warm_median_s'))} | "
                     f"{f(p.get('observed_over_projected_base'))} |")
    lines += ["", f"Failures and non-ok runs: {len(s['failures'])} (pilot-summary.json lists each)", ""]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--label", required=True)
    p.add_argument("--machine", required=True)
    p.add_argument("--hours", type=float, required=True, help="hours available on this machine for all sessions")
    p.add_argument("--sessions", type=int, default=PILOT["sessions"])
    p.add_argument("--p-cores", help="CPU list of the P-cores if the kernel does not expose it, e.g. 0-5")
    p.add_argument("--skip-step", nargs="*", default=[], choices=STEPS)
    p.add_argument("--cut", help="reason for any reduction of sessions or steps, recorded before the first run")
    p.add_argument("--purpose", choices=["pilot", "verification"], default="pilot")
    p.add_argument("--scaling-record", help="verification: the committed scaling-decision.json")
    p = sub.add_parser("amend")
    p.add_argument("--label", required=True)
    p.add_argument("--reason", required=True)
    p.add_argument("--set", nargs="+", required=True, help="key=json, e.g. sessions=1 or hours_available=6")
    p = sub.add_parser("check")
    p.add_argument("--label", required=True)
    p.add_argument("--paths", default="inc,hyb")
    p.add_argument("--skip-phase1-hits", action="store_true", help="skip gate item 9 (recorded as not run)")
    p = sub.add_parser("run")
    p.add_argument("--label", required=True)
    p.add_argument("--session", type=int, required=True)
    p.add_argument("--step", required=True, choices=STEPS)
    p = sub.add_parser("summary")
    p.add_argument("--label", required=True)
    for sp in sub.choices.values():
        sp.add_argument("--build", type=Path, default=ROOT / "build")
    a = ap.parse_args()
    a.build = a.build.resolve()
    {"plan": cmd_plan, "amend": cmd_amend, "check": cmd_check, "run": cmd_run, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()

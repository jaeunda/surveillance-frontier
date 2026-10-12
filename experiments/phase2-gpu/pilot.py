"""Part I of the protocol: the pilot on an available machine (P0-P2), and the same workload on a target machine for
the scaling verification (P4). Results are descriptive and labelled with their machine; they are never pooled with
formal launches.

    pilot.py plan     --label L --machine lab-desktop --hours 12 [--p-cores 0-5] [--sessions 2] [--cut "reason"]
                      [--skip-step tref-build ...] [--cold-kind auto|process-cold]
                      [--session-hours H] [--run-cap-min M]              (a run cap is a reduction: needs --cut)
                      [--step-hours STEP=H ...]                         per-step share of a session's hours
                      [--purpose verification --scaling-record FILE]
    pilot.py amend    --label L --reason "..." --set key=json ...      before the measurement it affects
    pilot.py check    --label L --session K [--paths inc,hyb]           P0 of a session: CPU regression, stream,
                                                                         intervals, CPU gate, device gate per GPU path
    pilot.py run      --label L --session K --step STEP                 P1/P2 steps (see protocol.json pilot.steps)
    pilot.py summary  --label L                                          pilot-summary.json / .md (all sessions)

Results: results/pilot/<label>/{pilot-plan.json, amendments.jsonl, hashes.json, session-<K>/, pilot-summary.*}, with
session-<K>/{env.json, checks/, attempts.jsonl, runs.jsonl, bench.jsonl, steps.jsonl, outputs/<step>/attempt-<k>/}.

Every session is its own machine: its CPU baseline and thread candidates come from its own topology (by the plan's
rule, written to session-<K>/cpu-setup.json by its P0 checks), it runs its own P0 checks and device gates, its own
thread, parity and microbatch choices, and its runs are refused if the hardware, code or binaries differ from what its
checks validated. Every run writes a 'started' record first and a terminal record as soon as it ends, including failed,
late, timed-out and slow runs; runs a step did not start are recorded as 'not started (...)'. Checks, steps, every run
and the reference oracle are bounded by the plan's remaining hours (the time box); time is counted from immutable
start, heartbeat and finish records of each attempt, never from later file times. A step is 'complete' only if it
returned normally with every planned run accounted for; any exception records it as failed.

Output checks: a finished fixed-task run is re-evaluated at 256 trials by its own back end and configuration and
compared field by field with the CPU tail method and its stored bits (sf_run --spot; repeated checks of one condition
draw further trial ids). This is a sample check, not bit identity of the whole output: a GPU output is 'exploratory
(spot check only)' until the label has a full CPU result hash for the task, and then every earlier output is compared
with it (reverify.jsonl). A spot or identity mismatch of a GPU path excludes that path from every later step and from
the summary's metrics; a mismatch of a CPU output stops the session.
"""

import argparse
import json
import math
import random
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from p2 import design, harness, machine, projection, records  # noqa: E402
from p2.common import (FORMAL, MACHINES, PILOT, PROTOCOL, RESULTS, ROOT, append_jsonl, code_fingerprint,  # noqa: E402
                       committed_unchanged, git_state, input_hashes, manifest, mirror_tree, new_attempt_dir, now,
                       read_json, read_jsonl, run_logged, sha256_file, task_hashes, write_json)
from p2.stats import describe  # noqa: E402
from p2.tasks import calib_task, main_task, point  # noqa: E402

STEPS = list(PILOT["steps"])
CPU_STEPS = {"cpu-baseline", "threads", "parity", "rates", "diag", "tref-rates", "tref-build", "deadline"}
GPU_STEPS = {"gpu-rates", "microbatch", "gpu-tasks"}
FIXED_STEPS = {"cpu-baseline", "gpu-tasks"}
SEED = PROTOCOL["stream"]["seed"]
HEARTBEAT_S = 60  # an attempt cut off without a finish record is counted to its last heartbeat plus this interval
CPU_SETUP_RULE = ("per session, from that session's allowed CPUs: baseline = every allowed logical CPU, unpinned; "
                  "thread candidates = machine.thread_candidates (the plan's P-core list if all of it is allowed, all "
                  "physical cores if SMT, all logical CPUs); GPU host threads and the spot check follow the session's "
                  "CPU configuration")


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
    if a.purpose == "verification":
        if not a.scaling_record:
            sys.exit("--purpose verification needs --scaling-record (the committed scaling decision)")
        import scaling  # noqa: E402
        pr = scaling.problems(read_json(a.scaling_record))
        if pr:
            sys.exit("the scaling record does not pass scaling.py check: " + "; ".join(pr[:5]))
        if not committed_unchanged(a.scaling_record):
            sys.exit("commit the scaling record before the verification it justifies")
    topo = machine.topology()
    ident = machine.identity(a.machine)
    if ident["match"] is False:
        sys.exit(f"this machine does not match profile {a.machine}: {ident}")
    if MACHINES[a.machine]["kind"] == "available" and a.machine == "lab-desktop" and not (a.p_cores or topo["p_cores"]):
        print("note: P/E core mapping unknown here; the P-core candidate is left out until --p-cores is given "
              "(check the mapping on the Windows host)", file=sys.stderr)
    steps = [s for s in STEPS if s not in a.skip_step]
    reps = dict(PILOT["reps_per_session"])
    step_hours = {}
    for kv in a.step_hours:
        k, v = kv.split("=", 1)
        if k not in STEPS + ["check"]:
            sys.exit(f"--step-hours: unknown step {k}")
        step_hours[k] = float(v)
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
        "code_sha256": code_fingerprint(),
        "binaries": machine.build_info(a.build)["binary_sha256"],
        "inputs_sha256": input_hashes(),
        "tasks_sha256": task_hashes(),
        "protocol_sha256": sha256_file(HERE / "protocol.json"),
        "scaling_record": None if not a.scaling_record else {"path": a.scaling_record,
                                                                "sha256": sha256_file(a.scaling_record)},
        "hours_available": a.hours,
        "hours_per_session": a.session_hours,
        "run_cap_s": None if a.run_cap_min is None else a.run_cap_min * 60,
        "sessions": a.sessions,
        "steps": steps,
        "skipped_steps": a.skip_step,
        "fixed_tasks": PILOT["fixed_tasks"],
        "reps_per_session": reps,
        "cold_kind": a.cold_kind,
        "p_cores": a.p_cores,
        "cpu_setup_rule": CPU_SETUP_RULE,
        "cpu_at_plan": session_cpu_setup({"p_cores": a.p_cores}, topo),  # the plan machine's, never used for a run
        "step_hours": step_hours,
        "microbatch_sweep": PILOT["microbatch_sweep"],
        "bench": PILOT["bench"],
        "stop_rules": [
            "P0 checks and steps do not start when the hours already used (checks and steps, finished or cut off) reach "
            "the remaining hours; they are recorded as 'not started (time)'",
            "the remaining hours are the smaller of hours_available and, if set, hours_per_session minus the hours "
            "used in that session and, if set, step_hours[step] minus the hours that step used in that session",
            "hours used: finished attempts by their elapsed time; an attempt cut off without a finish record from its "
            "start to its last heartbeat plus one heartbeat interval (an upper bound; the summary also gives the lower)",
            "every check item, run, spot check and the reference oracle is bounded by the remaining hours (time box); "
            "a run cut by it is recorded as 'time box', the step stops and its remaining runs are recorded as 'not "
            "started (time)'",
            "if run_cap_s is set, every fixed-task run (cold run, warm priming, warm request) is killed at run_cap_s "
            "and recorded as 'timeout' (censored at the cap); the remaining runs of that condition (task, series, "
            "configuration) in the step are recorded as 'not started (run cap)' and not run",
            "a failed, late, timed-out or out-of-memory run is recorded and the step continues with its next run",
            "a spot or identity mismatch of a GPU output excludes that path: its remaining runs are recorded as 'not "
            "started (path excluded)', later steps skip it and its records give no metric; a mismatch of a CPU output "
            "stops the step and the session",
            "a step is complete only if it returned normally and every planned run of it has its records",
            "tref-build starts only if 15 T-ref cell rates exist for one CPU configuration, the projection was checked "
            "against an uncensored observed warm E2E on the same configuration in this session, and the conservative "
            "projection, scaled by the worst observed / projected ratio if above 1, fits the remaining hours",
            "a run with min MemAvailable < 2 GiB or any swap-in is flagged 'memory pressure' in the summary",
        ],
        "cut": a.cut,
    }
    if a.purpose == "pilot" and a.sessions < PILOT["sessions"] and not a.cut:
        sys.exit(f"fewer than {PILOT['sessions']} sessions needs --cut with the reason, recorded before the first run")
    if (a.run_cap_min is not None or a.skip_step) and not a.cut:
        sys.exit("a run cap or a skipped step reduces the plan: give --cut with the reason")
    if (a.run_cap_min is not None and a.run_cap_min <= 0) or (a.session_hours is not None and a.session_hours <= 0):
        sys.exit("--run-cap-min and --session-hours must be positive")
    if any(v <= 0 for v in step_hours.values()):
        sys.exit("--step-hours values must be positive")
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


# ---- per-session CPU setup, time accounting, step attempts ----

def quota_cpus(topo):
    """CPUs' worth of the cgroup v2 CPU quota ('200000 100000' -> 2), or None if there is no quota."""
    f = str(topo.get("cgroup_cpu_max") or "").split()
    if len(f) != 2 or f[0] == "max":
        return None
    return max(1, math.ceil(int(f[0]) / int(f[1])))


def session_cpu_setup(plan, topo):
    """CPU baseline and thread candidates of one session from its own topology (CPU_SETUP_RULE): its allowed CPUs, its
    P-core mapping (the plan's list only if every CPU of it is allowed here), and its CPU quota, which caps the thread
    count of every candidate."""
    allowed = set(machine._cpulist(topo["allowed_cpus"])) if topo.get("allowed_cpus") else None
    pc, notes = plan.get("p_cores"), []
    if pc and allowed is not None and not set(machine._cpulist(pc)) <= allowed:
        notes.append(f"plan P-core list {pc} not within this session's allowed CPUs {topo['allowed_cpus']}: not used")
        pc = None
    q = quota_cpus(topo)
    cands = machine.thread_candidates(topo, pc)
    if q is not None and q < topo["logical"]:
        notes.append(f"CPU quota {topo['cgroup_cpu_max']} = {q} CPUs: thread counts capped at {q}")
        seen, capped = set(), []
        for c in cands:
            c = {**c, "threads": min(c["threads"], q)}
            if (c["threads"], c["cpus"]) not in seen:
                seen.add((c["threads"], c["cpus"]))
                capped.append(c)
        cands = capped
    threads = topo["logical"] if q is None else min(q, topo["logical"])
    return {"rule": CPU_SETUP_RULE, "logical": topo["logical"], "allowed_cpus": topo.get("allowed_cpus"),
            "cgroup_cpu_max": topo.get("cgroup_cpu_max"), "baseline": harness.cpu_config(threads, name="all"),
            "thread_candidates": cands, "note": "; ".join(notes) or None}


class Heartbeat:
    """Appends {step_id, wall} to session-<K>/heartbeats.jsonl every HEARTBEAT_S while an attempt runs (mirrored with
    every record), so a lost VM leaves a bound on how long the attempt ran."""

    def __init__(self, sd, step_id, interval=None):
        self.path, self.step_id = Path(sd) / "heartbeats.jsonl", step_id
        self.interval = HEARTBEAT_S if interval is None else interval
        self.stop_flag = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self.stop_flag.wait(self.interval):
            try:
                append_jsonl(self.path, {"step_id": self.step_id, "wall": time.time()})
            except OSError as e:
                print(f"warning: heartbeat not written: {e}", file=sys.stderr)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop_flag.set()
        self.thread.join()


def session_time(sd):
    """Hours of the checks and steps of one session, from their own records only: finished attempts by elapsed time;
    an attempt without a finish record from its start to its last heartbeat (lower) plus one heartbeat interval
    (upper). Returns {"finished_h", "unfinished_lower_h", "unfinished_upper_h", "used_h" (= finished + upper),
    "by_step": {step: used hours}}."""
    sd = Path(sd)
    recs = read_jsonl(sd / "steps.jsonl")
    beats = {}
    for h in read_jsonl(sd / "heartbeats.jsonl"):
        beats[h["step_id"]] = max(beats.get(h["step_id"], 0.0), h["wall"])
    finished = {r.get("step_id") for r in recs if r.get("event") != "started" and r.get("step_id")}
    out = {"finished_h": 0.0, "unfinished_lower_h": 0.0, "unfinished_upper_h": 0.0, "by_step": {}}
    for r in recs:
        if r.get("event") == "started":
            if r["step_id"] in finished:
                continue
            lower = max(0.0, beats.get(r["step_id"], r["wall"]) - r["wall"]) / 3600
            upper = lower + HEARTBEAT_S / 3600
            out["unfinished_lower_h"] += lower
            out["unfinished_upper_h"] += upper
            h = upper
        else:
            h = r.get("elapsed_s", 0) / 3600
            out["finished_h"] += h
        out["by_step"][r.get("step")] = out["by_step"].get(r.get("step"), 0.0) + h
    out["used_h"] = out["finished_h"] + out["unfinished_upper_h"]
    return out


def session_hours_used(sd):
    return session_time(sd)["used_h"]


def hours_used(d):
    """Hours of checks and steps over every session of a label (conservative for attempts without a finish record)."""
    return sum(session_hours_used(sd) for sd in sorted(Path(d).glob("session-*")))


def hours_left(plan, d, sd, step=None):
    """The plan's remaining hours for a check or step of session sd: hours_available minus the label's hours used,
    bounded by hours_per_session and by step_hours[step] (that step's hours in this session) if the plan sets them."""
    left = plan["hours_available"] - hours_used(d)
    st = session_time(sd)
    if plan.get("hours_per_session"):
        left = min(left, plan["hours_per_session"] - st["used_h"])
    cap = (plan.get("step_hours") or {}).get(step)
    if cap:
        left = min(left, cap - st["by_step"].get(step, 0.0))
    return left


def run_attempt(sd, step, left_h, fn, **extra):
    """One attempt of a check or step: a 'started' record, heartbeats, the time box, then a finish record whose status
    is 'complete' (or the status fn returns) only if fn returned normally. A time box gives 'stopped (time)'; any other
    exception is recorded as failed with its type and message and propagates."""
    step_id = f"{step}-{time.time_ns()}"
    append_jsonl(Path(sd) / "steps.jsonl", {"event": "started", "step": step, "step_id": step_id, "wall": time.time(),
                                            "hours_left": left_h, **extra})
    harness.TIME_BOX_NS = time.monotonic_ns() + int(left_h * 3600e9)
    t0 = time.monotonic()
    status, error = "failed", None
    try:
        with Heartbeat(sd, step_id):
            status = fn() or "complete"
    except harness.TimeBox as e:
        status = f"stopped (time): {e}"
    except SystemExit as e:
        status = f"stopped: {e}"
        raise
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except BaseException as e:
        status, error = "failed", f"{type(e).__name__}: {e}"
        raise
    finally:
        append_jsonl(Path(sd) / "steps.jsonl", {"step": step, "step_id": step_id, "status": status, "error": error,
                                                "elapsed_s": time.monotonic() - t0, "finished": now(), **extra})
    return status


# ---- P0 checks ----

def cmd_check(a):
    d = label_dir(a.label)
    plan = effective_plan(d)
    if a.session < 1 or a.session > plan["sessions"]:
        sys.exit(f"session {a.session} not in the plan (1..{plan['sessions']}); a further session needs "
                 "pilot.py amend --set sessions=N first (the hours stay the plan's total)")
    sd = d / f"session-{a.session}"
    left = hours_left(plan, d, sd, "check")
    if left <= 0:
        append_jsonl(sd / "steps.jsonl", {"step": "check", "status": "not started (time)", "hours_used": hours_used(d),
                                          "finished": now()})
        sys.exit(f"no hours left for the P0 checks (used {hours_used(d):.2f} of {plan['hours_available']}; this "
                 f"session {session_hours_used(sd):.2f} of {plan.get('hours_per_session')})")
    build = a.build
    env = machine.environment(plan["machine"], build)
    if plan["machine_profile"].get("identity") and env["identity"]["match"] is False:
        sys.exit(f"this machine does not match profile {plan['machine']}: {env['identity']}")
    if env["code_sha256"] != plan["code_sha256"]:
        sys.exit("the code differs from the plan's code fingerprint (record the change with pilot.py amend "
                 "--set code_sha256=... and its reason first)")
    if (sd / "env.json").exists() and read_json(sd / "env.json")["hardware_fingerprint"] != env["hardware_fingerprint"]:
        sys.exit(f"session {a.session} ran on other hardware ({read_json(sd / 'env.json')['hardware_fingerprint']}, "
                 f"now {env['hardware_fingerprint']}): a new assignment is a new session")
    setup = session_cpu_setup(plan, env["topology"])
    if (sd / "cpu-setup.json").exists():
        old = read_json(sd / "cpu-setup.json")
        if (old["baseline"], old["thread_candidates"]) != (setup["baseline"], setup["thread_candidates"]):
            sys.exit(f"session {a.session} was set up for other CPUs ({old['allowed_cpus']}, now "
                     f"{setup['allowed_cpus']}): a new assignment is a new session")
    c = new_attempt_dir(sd / "checks")
    write_json(c / "env.json", env)
    if not (sd / "env.json").exists():
        write_json(sd / "env.json", env)
    if not (sd / "cpu-setup.json").exists():
        write_json(sd / "cpu-setup.json", setup)  # before any measurement of this session
    print(f"session CPU setup: baseline {harness.config_key(setup['baseline'])}; candidates "
          f"{[x['name'] for x in setup['thread_candidates']]}{'; ' + setup['note'] if setup['note'] else ''}",
          flush=True)
    res = {"time": now(), "machine": plan["machine"], "session": a.session, "hardware_fingerprint":
           env["hardware_fingerprint"], "code_sha256": env["code_sha256"], "binaries": env["build"]["binary_sha256"],
           "cpu_setup": setup}
    t0 = time.monotonic()

    def checks():
        py = sys.executable
        log = c / "checks.log"
        items = {
            "sf_test": [build / "sf_test"],
            "sf_test2": [build / "sf_test2"],
            "intervals": [py, HERE / "check_intervals.py", "--build", build, "--out", c / "intervals.json"],
            "stream": [py, HERE / "stream.py", "--check"],
            "harness_tests": [py, HERE / "test_p2.py"],
            "gate_cpu": [build / "sf_gate", "--device", "cpu", "--threads", setup["baseline"]["threads"], "--items",
                         "1,6,7,8", "--out", c / "gate_cpu.json"],
        }
        if not a.skip_phase1_hits:
            items["phase1_hits"] = [py, HERE / "check_phase1_hits.py", "--build", build, "--out", c / "item9.json",
                                    "--threads", setup["baseline"]["threads"]]
        # CUDA: the partial gate on this session's device. A build without SF_CUDA, or no GPU, is 'unavailable', and
        # an emulation pass never counts as device execution.
        has_cuda = "ON" in machine.build_info(build)["cmake_cache"].get("SF_CUDA", "")
        gpu = {}
        for path in a.paths.split(","):
            if not has_cuda or not env["gpu"].get("present"):
                gpu[path] = {"execution": "unavailable", "pass": False,
                             "reason": "build without SF_CUDA=ON" if not has_cuda else "no GPU visible"}
                continue
            items[f"cuda_{path}"] = [build / "sf_gate", "--device", "cuda", "--path", path, "--microbatch", "64",
                                     "--partial", "--threads", setup["baseline"]["threads"], "--out",
                                     c / f"gate_cuda_{path}.json"]
        out, status = {}, "complete"
        for name, cmd in items.items():
            left_s = harness.time_left_s()
            if left_s <= 0:
                out[name], status = "not run (time box)", "stopped (time): time box before " + name
                continue
            try:
                append_jsonl(c / "items.jsonl", {"item": name, "event": "started", "time": now(),
                                                "timeout_s": left_s})
                r = run_logged(cmd, log, timeout=left_s)
                result = r.returncode == 0
            except subprocess.TimeoutExpired:
                r, result = None, "time box"
                status = f"stopped (time): time box during {name}"
            finally:
                mirror_tree(c)
            append_jsonl(c / "items.jsonl", {"item": name, "event": "finished", "time": now(), "result": result})
            if name.startswith("cuda_"):
                path = name[5:]
                gpu[path] = {"execution": "device", "pass": result is True, "returncode": r and r.returncode,
                             "gpu": env["gpu"].get("name"), "stderr_tail": (r.stderr[-1000:] if r else "time box")}
            else:
                out[name] = result
            print(f"{name:<16} {'pass' if result is True else 'FAIL' if result is False else result}", flush=True)
        res["cpu"] = out
        res["cpu_pass"] = all(v is True for v in out.values())
        res["gpu"] = gpu
        res["validated_paths"] = [p for p, v in gpu.items() if v["pass"]]
        res["elapsed_s"] = time.monotonic() - t0
        res["artifacts"] = manifest(c)
        write_json(c / "checks.json", res)
        mirror_tree(c)
        if status != "complete":
            return status
        return "complete" if res["cpu_pass"] else "failed (CPU checks)"

    status = run_attempt(sd, "check", left, checks, checks=c.name)
    sys.exit(0 if status == "complete" else 1)


# ---- P1/P2 steps ----

class Session:
    def __init__(self, label, k, build):
        self.d = label_dir(label)
        self.label = label
        self.plan = effective_plan(self.d)
        if k < 1 or k > self.plan["sessions"]:
            sys.exit(f"session {k} not in the plan (1..{self.plan['sessions']}); a further session needs "
                     "pilot.py amend --set sessions=N first")
        self.k, self.build = k, build
        self.sd = self.d / f"session-{k}"
        checks = sorted(self.sd.glob("checks/attempt-*/checks.json"), key=lambda p: int(p.parent.name.split("-")[1]))
        self.checks = read_json(checks[-1]) if checks else None
        self.setup = read_json(self.sd / "cpu-setup.json") if (self.sd / "cpu-setup.json").exists() else None
        self.runs, self.benches = self.sd / "runs.jsonl", self.sd / "bench.jsonl"
        self.hashes_path = self.d / "hashes.json"
        self.env = machine.environment(self.plan["machine"], build)
        self.topo = self.env["topology"]
        harness.ATTEMPT_LOG = self.sd / "attempts.jsonl"

    def refuse_if_changed(self):
        """Runs only on the machine, code and binaries this session's checks validated, and only while no CPU output
        of this session failed its check."""
        if self.checks is None or not self.checks["cpu_pass"]:
            return "P0 checks of this session missing or failed (pilot.py check --session K): no timed runs"
        if self.setup is None:
            return "this session has no cpu-setup.json (run pilot.py check --session K again)"
        if self.env["hardware_fingerprint"] != self.checks["hardware_fingerprint"]:
            return "the hardware differs from the one this session's checks ran on (a new assignment is a new session)"
        if self.env["topology"].get("allowed_cpus") != self.setup.get("allowed_cpus"):
            return "the allowed CPUs differ from this session's CPU setup (a new assignment is a new session)"
        if self.env["code_sha256"] != self.checks["code_sha256"] or self.env["code_sha256"] != self.plan["code_sha256"]:
            return "the code differs from the checked code or the plan"
        if self.env["build"]["binary_sha256"] != self.checks["binaries"]:
            return "the binaries differ from the checked build: run pilot.py check again"
        if self.env["inputs_sha256"] != self.plan["inputs_sha256"] or task_hashes() != self.plan["tasks_sha256"]:
            return "the inputs or tasks differ from the plan"
        from colab_ops import audit_results
        problems = audit_results(self.d)
        if problems:
            return "restore validation failed: " + "; ".join(problems[:3])
        if self.k in self.invalid()["sessions"]:
            return "a CPU output of this session failed its check (identity or spot): the session's validation stops"
        return None

    def invalid(self):
        runs, _ = records.label_records(self.label)
        return records.invalid(self.label, runs)

    # identity of fixed-n outputs: the first complete CPU result of a task is the reference within this label
    def expected(self, task, series):
        h = read_json(self.hashes_path) if self.hashes_path.exists() else {}
        return h.get(f"{task}/{series}")

    def remember(self, task, series, rec):
        """A first complete, spot-checked CPU result becomes the label's reference hash of the task; every earlier
        output of the task that was checked only by the spot sample is then compared with it (reverify.jsonl)."""
        h = read_json(self.hashes_path) if self.hashes_path.exists() else {}
        key = f"{task}/{series}"
        if key in h or rec["status"] != "ok" or rec["config"]["device"] != "cpu" or rec.get("spot_mismatches") != 0:
            return
        h[key] = rec["result_sha256"]
        write_json(self.hashes_path, h)
        for sd in sorted(self.d.glob("session-*")):
            for r in read_jsonl(sd / "runs.jsonl"):
                if r.get("task") == task and r.get("series") == series and r.get("result_sha256") \
                        and r.get("run_id") != rec.get("run_id") and r.get("identity_ok") is not True:
                    append_jsonl(self.d / "reverify.jsonl", {
                        "time": now(), "task": task, "series": series, "run_id": r.get("run_id"),
                        "session": int(sd.name.split("-")[1]), "config_key": r.get("config_key"),
                        "status_at_run": r.get("status"), "result_sha256": r["result_sha256"],
                        "reference_sha256": rec["result_sha256"], "reference_run_id": rec.get("run_id"),
                        "match": r["result_sha256"] == rec["result_sha256"]})

    def hours_used(self):
        return hours_used(self.d)

    def hours_left(self, step=None):
        return hours_left(self.plan, self.d, self.sd, step)

    def decision(self, kind):
        """Latest decision of a kind in this session (threads, parity, microbatch): never from another session."""
        found = None
        for r in read_jsonl(self.benches):
            if r.get("kind") == kind:
                found = r
        return found

    def cpu_baseline(self):
        return dict(self.setup["baseline"])

    def cpu_config(self):
        """CPU configuration for rates, diag and the reference: this session's threads choice for 1s-week/P24 with
        its adopted parity set, else this session's baseline."""
        dec = self.decision("threads-decision")
        cfg = dict(dec["config"]) if dec and dec.get("config") else self.cpu_baseline()
        par = self.decision("parity-adopted")
        cfg["parity"] = par["parity"] if par else "none"
        return cfg

    def validated_paths(self):
        if not self.checks:
            return []
        return [p for p in self.checks["validated_paths"] if not self.path_stopped(p)]

    def path_stopped(self, path):
        """A GPU path is excluded after any spot or identity mismatch, now or found later (reverify)."""
        return path in self.invalid()["paths"]

    def host_threads(self):
        return self.cpu_config()["threads"]

    def gpu_config(self, path):
        dec = self.decision(f"microbatch-decision-{path}")
        mb = dec["choice"] if dec and dec.get("choice") else 64
        return harness.gpu_config(path, mb, host_threads=self.host_threads())

    def drop(self):
        return self.plan.get("cold_kind", "auto") != "process-cold"

    def out_dir(self, step):
        return new_attempt_dir(self.sd / "outputs" / step)

    def spot(self, rec):
        """256-trial spot check of a finished run by its own back end and configuration (CPU tail and stored bits as
        the comparison). Each further check of the same (task, series, configuration) in the label draws the next 256
        trial ids. Bounded by the time box. Returns {"mismatches": n or None, "spot_from", "first", "error"}."""
        prior = sum(1 for sd in self.d.glob("session-*") for r in read_jsonl(sd / "runs.jsonl")
                    if "spot" in r and r.get("task") == rec.get("task") and r.get("series") == rec.get("series")
                    and r.get("config_key") == rec["config_key"])
        spot_from = 256 * prior
        left = harness.time_left_s()
        out = {"mismatches": None, "spot_from": spot_from, "first": None, "error": None}
        if left is not None and left <= 0:
            out["error"] = "time box before the spot check"
            return out
        try:
            r = run_logged([self.build / "sf_run", rec["task_file"], "--spot", rec["out"], "--spot-from", spot_from,
                            *harness.sf_args(rec["config"])], Path(rec["out"]) / "spot.log",
                           timeout=3600 if left is None else max(1.0, min(3600, left)))
            j = json.loads(r.stdout.strip().splitlines()[-1])
            out.update({"mismatches": j.get("mismatches"), "first": j.get("first")})
        except subprocess.TimeoutExpired:
            out["error"] = "spot check cut by the time box"
        except (ValueError, IndexError, OSError, KeyError) as e:
            out["error"] = repr(e)
        return out


def safe(name):
    return name.replace(":", "_").replace("@", "_").replace(",", "-")


CONFIG_ROLE = {"cpu-baseline": "starting CPU configuration (every allowed CPU, no parity; before the threads and "
                                "parity choices): not CPU-best",
               "gpu-tasks": "GPU configuration at the time of the step (microbatch choice if made, else 64): not "
                            "GPU-best unless the microbatch step ran before"}


def fixed_task_runs(s, step, configs, seed):
    """Cold repetitions individually and warm repetitions as one block per condition (one priming task, then the
    timed requests), all in one seeded order written before the first run. Each record is written when its run ends.
    Every planned run gets a record: runs not started because of the run cap, an excluded path, the time box or a
    stopped step are recorded as 'not started (...)'. Returns 'complete' only if every planned run is accounted for."""
    reps = s.plan["reps_per_session"]
    items = []
    for cfg in configs:
        for task, series in s.plan["fixed_tasks"]:
            items += [{"kind": "cold", "task": task, "series": series, "rep": r, "config": cfg}
                      for r in range(reps["cold"])]
            if reps["warm"]:
                items.append({"kind": "warm-block", "task": task, "series": series, "reps": reps["warm"], "config": cfg})
    random.Random(seed).shuffle(items)
    od = s.out_dir(step)
    cap = s.plan.get("run_cap_s")
    write_json(od / "order.json", {"seed": seed, "order": items, "run_cap_s": cap})
    capped = set()  # conditions with a run killed at the run cap
    got = {}  # position -> {"cold"} / {"priming", 0, 1, ..} records written
    cold_kind = {}  # condition -> cold kind of its last cold run (a run not started inherits it)

    def cond_of(pos, it):
        return {"step": step, "session": s.k, "task": it["task"], "series": it["series"],
                "task_file": str(main_task(it["task"], it["series"])), "position": pos, "outputs": od.name,
                "config_role": CONFIG_ROLE.get(step)}

    def write(pos, rec):
        append_jsonl(s.runs, rec)
        got.setdefault(pos, set()).add("cold" if rec["mode"] == "cold" else
                                       "priming" if rec["mode"] == "warm-priming" else rec.get("rep"))

    def not_started(pos, it, why):
        """Records for every run of item pos that has none yet."""
        have = got.get(pos, set())
        base = {**cond_of(pos, it), "config": it["config"], "config_key": harness.config_key(it["config"]),
                "status": f"not started ({why})", "runs_not_started": 1, "run_cap_s": cap, "finished_at": now()}
        if it["kind"] == "cold":
            if "cold" not in have:
                ck = (it["task"], it["series"], base["config_key"])
                write(pos, {"mode": "cold", **base, "rep": it["rep"],
                            "cold_kind": "process-cold" if not s.drop() else cold_kind.get(ck, "unknown")})
            return
        if "priming" not in have:
            write(pos, {"mode": "warm-priming", **base})
        for r in range(it["reps"]):
            if r not in have:
                write(pos, {"mode": "warm", **base, "rep": r})

    pos = -1
    try:
        for pos, it in enumerate(items):
            tf = main_task(it["task"], it["series"])
            cond = cond_of(pos, it)
            out = od / safe(f"{pos:03d}_{it['task']}_{it['series']}_{harness.config_key(it['config'])}")
            expected = s.expected(it["task"], it["series"])
            ckey = (it["task"], it["series"], harness.config_key(it["config"]))
            path = it["config"].get("path") if it["config"]["device"] != "cpu" else None
            why = ("path excluded: an output of this path failed its check" if path and s.path_stopped(path)
                   else "run cap" if ckey in capped else None)
            if why:
                not_started(pos, it, why)
                print(f"[{pos + 1}/{len(items)}] {it['kind']} {it['task']} {it['series']} {ckey[2]}: not started "
                      f"({why})", flush=True)
                continue

            def keep(rec, pos=pos, it=it, expected=expected, ckey=ckey):
                rec["run_cap_s"] = cap
                if rec.get("mode") == "warm-priming":
                    if rec.get("out") and Path(rec["out"]).is_dir():
                        rec["out_manifest"] = manifest(rec["out"])
                        rec["out_mirrored"] = mirror_tree(rec["out"])
                    if rec.get("status") in ("timeout", "time box"):
                        capped.add(ckey)
                    write(pos, rec)
                    if rec.get("status") == "time box":
                        raise harness.TimeBox("time box reached during warm priming")
                    return
                if rec.get("status") == "ok":
                    sp = s.spot(rec)
                    rec.update({"spot_mismatches": sp["mismatches"], "spot_from": sp["spot_from"],
                                "spot_first": sp["first"], "spot_error": sp["error"]})
                    if sp["mismatches"] is None:
                        rec["status"] = "spot_error"  # verification not completed: not an ok run
                    elif sp["mismatches"] != 0:
                        rec["status"] = "spot_mismatch"
                    rec["spot"] = True
                # what the output was compared with: the label's full CPU result hash, or only the 256-trial sample
                rec["identity_basis"] = ("cpu reference hash + 256-trial spot check" if expected else
                                         "256-trial spot check only (exploratory: no full CPU result hash yet)")
                if rec.get("out") and Path(rec["out"]).is_dir() and not str(rec.get("status")).startswith("not started"):
                    rec["out_manifest"] = manifest(rec["out"])
                    rec["out_mirrored"] = mirror_tree(rec["out"])
                if rec.get("status") == "timeout":
                    capped.add(ckey)
                if rec.get("mode") == "cold" and rec.get("cold_kind") in ("cache-cold", "process-cold"):
                    cold_kind[ckey] = rec["cold_kind"]
                s.remember(it["task"], it["series"], rec)
                write(pos, rec)
                print(f"[{pos + 1}/{len(items)}] {rec['mode']:<5} {it['task']} {it['series']} {rec['config_key']}: "
                      f"{rec.get('e2e_s', float('nan')):.2f} s {rec['status']} {rec.get('cold_kind', '')}", flush=True)
                if rec.get("status") in records.OUTPUT_FAILURES:
                    if rec["config"]["device"] == "cpu":
                        raise SystemExit(f"{rec['status']}: a CPU output failed its check; step and session stopped")
                    print(f"GPU path {rec['config'].get('path')} excluded: {rec['status']}", flush=True)
                if rec.get("status") == "time box":
                    raise harness.TimeBox("time box reached during the step")

            if it["kind"] == "cold":
                keep(harness.run_cold(tf, out, it["config"], {**cond, "rep": it["rep"]}, build=s.build, timeout_s=cap,
                                      expected_sha=expected, drop=s.drop()))
            else:
                harness.warm_block(it["config"], tf, out, it["reps"], cond, build=s.build, expected_sha=expected,
                                   timeout_s=cap, prime_timeout_s=cap, on_record=keep)
    except harness.TimeBox:
        for q in range(max(pos, 0), len(items)):
            not_started(q, items[q], "time")
        raise
    except SystemExit:
        for q in range(max(pos, 0), len(items)):
            not_started(q, items[q], "step stopped")
        raise
    missing = [q for q, it in enumerate(items)
               if got.get(q, set()) != ({"cold"} if it["kind"] == "cold" else {"priming", *range(it["reps"])})]
    return f"incomplete: positions {missing} lack records" if missing else "complete"


def step_cpu_baseline(s, seed):
    return fixed_task_runs(s, "cpu-baseline", [s.cpu_baseline()], seed)


def step_threads(s, seed):
    cands = [harness.cpu_config(c["threads"], c["cpus"], name=c["name"]) for c in s.setup["thread_candidates"]]
    for i, (series, grid) in enumerate((x, g) for x in PROTOCOL["series"] for g in ("P1", "P24")):
        rec = design.choose(series, grid, cands, s.benches, build=s.build, label="threads-grid", seed=seed + i)
        print(f"threads {series} {grid}: {rec.get('choice')} (flip {rec.get('flip')})", flush=True)
    # the configuration used for rates, diag and the reference: this session's week P24 choice
    week = [r for r in read_jsonl(s.benches) if r.get("kind") == "threads-grid-decision"
            and r["series"] == "1s-week" and r["grid"] == "P24"][-1]
    append_jsonl(s.benches, {**week, "kind": "threads-decision", "rule": "choice at 1s-week P24"})


def step_parity(s, seed):
    if not s.decision("threads-decision"):
        return "not started (run threads first in this session)"
    cfg = {**s.cpu_config(), "parity": "none"}
    log_md = s.sd / "parity.md"
    adopted = design.parity(cfg, s.benches, log_md, build=s.build, min_gain=FORMAL["verdicts"]["parity_min_gain"],
                            seed=seed, partial_gate=True)
    append_jsonl(s.benches, {"kind": "parity-adopted", "parity": adopted, "threads_config": harness.config_key(cfg)})
    print(f"parity adopted on this session: {adopted}", flush=True)


def configs_for_rates(s):
    cfgs = [s.cpu_config()]
    cfgs += [s.gpu_config(p) for p in s.validated_paths()]
    return cfgs


def link_configs(s):
    """Configurations of this session's fixed-task runs that differ from the current ones (the CPU baseline before the
    threads/parity choices, GPU runs before the microbatch choice): the rates step measures them too, so every observed
    E2E has rates on its own configuration and the projection can be checked against it. Excluded paths are left out."""
    have = {harness.config_key(c) for c in configs_for_rates(s)}
    bad = s.invalid()["paths"]
    out = {}
    for r in read_jsonl(s.runs):
        if r.get("step") in FIXED_STEPS and r.get("config") and r.get("config_key") not in have \
                and r["config"].get("path") not in bad:
            out[r["config_key"]] = r["config"]
    return list(out.values())


def step_rates(s, seed):
    b = s.plan["bench"]
    lengths = PROTOCOL["tasks"]["T1-curve"]["lengths"]
    for cfg in configs_for_rates(s) + link_configs(s):
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
            cfg = harness.gpu_config(path, mb, s.host_threads())
            for name, pt in PILOT["gpu_points"].items():
                for rep in range(b["repetitions"]):
                    r = design.bench_point(point(pt), cfg, b["seconds"], s.benches, build=s.build, label="gpu-rate",
                                           rep=rep, point_name=name, sample=True)
                    print(f"gpu {path} b={mb} {name}: {r.get('trials_per_s')} {r['status']}", flush=True)


def step_microbatch(s, seed):
    b = s.plan["bench"]
    rep_point = point(PILOT["gpu_points"]["representative"])
    for path in s.validated_paths():
        probe = design.bench_point(rep_point, harness.gpu_config(path, "auto", s.host_threads()), 1,
                                   s.benches, build=s.build, label="microbatch-probe", sample=True)
        b_max = (probe.get("device") or {}).get("b_max")
        sweep = [m for m in s.plan["microbatch_sweep"] if b_max is None or m <= b_max]
        if b_max is not None and b_max not in sweep:
            sweep.append(b_max)
        rates = {}
        for mb in sweep:
            cfg = harness.gpu_config(path, mb, s.host_threads())
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
    return fixed_task_runs(s, "gpu-tasks", [s.gpu_config(p) for p in s.validated_paths()], seed)


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


def projection_checks(runs, benches, fixed_tasks, bad=None):
    """Projection of each fixed task's warm E2E from the rates of a configuration, against the observed warm E2E of
    the same configuration on the same machine. Observed = median of an uncensored warm condition (all planned runs
    ok); otherwise the projection is 'unverified'. Records must be of one hardware fingerprint (the caller groups)."""
    out = []
    rate_recs = [r for r in benches if r.get("kind") == "rate" and not (bad and records.excluded(r, bad))]
    for cfg_key in sorted({r["config_key"] for r in rate_recs}):
        rates, _ = projection.rate_table([r for r in rate_recs if r["config_key"] == cfg_key])
        if not rates:
            continue
        for task, series in fixed_tasks:
            p = projection.project_fixed(task, series, rates, warm=True)
            m = {"task": task, "series": series, "mode": "warm", "config": cfg_key}
            recs = records.conditions(runs, m, bad).get(cfg_key, [])
            c = records.counts(recs)
            obs = statistics.median(r["e2e_s"] for r in recs if r.get("status") == "ok") if c["uncensored"] else None
            ratio = obs / p["seconds"]["base"] if obs and p["status"] == "ok" else None
            out.append({"config": cfg_key, "task": task, "series": series, "projection_warm": p, "counts": c,
                        "observed_warm_median_s": obs, "observed_over_projected_base": ratio,
                        "check": "verified" if ratio is not None else
                        "unverified: no uncensored observed warm E2E on this configuration" if p["status"] == "ok"
                        else f"no projection ({p['status']})"})
    return out


def tref_basis(s):
    """(configuration, projection, note) for the T-ref build: the current CPU configuration if all 15 cell rates
    exist for it in this session; else the latest configuration whose 15 cell rates exist (e.g. before a parity set
    adopted later), stated as such; (None, None, why) if none."""
    benches = read_jsonl(s.benches)
    cur = s.cpu_config()
    proj = tref_projection(benches, harness.config_key(cur))
    if proj["status"] == "ok":
        return cur, proj, "current CPU configuration"
    for r in reversed(benches):
        if r.get("kind") == "tref-rate" and r.get("config", {}).get("device") == "cpu":
            p = tref_projection(benches, r["config_key"])
            if p["status"] == "ok":
                return dict(r["config"]), p, (f"configuration of the T-ref cell rates ({r['config_key']}); the current "
                                              f"CPU configuration {harness.config_key(cur)} has no cell rates and is "
                                              "not used")
    return None, None, "no CPU configuration with all 15 T-ref cell rates in this session"


def step_tref_build(s, seed):
    cfg, proj, basis = tref_basis(s)
    if cfg is None:
        return f"not started ({basis})"
    runs, benches = records.label_records(s.label)
    checks = projection_checks([r for r in runs if r["_session"] == s.k], [r for r in benches if r["_session"] == s.k],
                               s.plan["fixed_tasks"], s.invalid())
    ratios = [c["observed_over_projected_base"] for c in checks if c["check"] == "verified" and c["config"].startswith("cpu:")]
    if not ratios:
        print("T-ref not started: no CPU projection of this session was checked against an observed warm E2E on the "
              "same configuration; the cell rates and the unverified projection are the result", flush=True)
        return "not started (projection unverified)"
    remaining = s.hours_left("tref-build")
    scale = max(1.0, max(ratios))
    need_h = proj["seconds"]["conservative"] / 3600 * scale * 1.1
    if need_h > remaining:
        print(f"T-ref not started: conservative projection x {scale:.2f} (worst observed / projected) = {need_h:.1f} h "
              f"> remaining {remaining:.1f} h; the per-cell rates and the projection are the result", flush=True)
        return f"not started (time: projected {need_h:.2f} h > remaining {remaining:.2f} h)"
    ref = new_attempt_dir(s.d / "reference")
    tf = main_task("T-ref", "1s-week")
    rec = harness.run_cold(tf, ref / "T-ref_1s-week", cfg, {"step": "tref-build", "session": s.k, "task": "T-ref",
                                                             "series": "1s-week", "task_file": str(tf), "rep": 0},
                           build=s.build, drop=s.drop())
    rec.update({"projection_s": proj["seconds"], "projection_scale": scale, "config_basis": basis,
                "reference_dir": ref.name})
    if Path(rec.get("out", "")).is_dir():
        rec["out_manifest"] = manifest(rec["out"])
        rec["out_mirrored"] = mirror_tree(rec["out"])
    append_jsonl(s.runs, rec)
    print(f"T-ref: {rec.get('e2e_s', 0) / 3600:.2f} h {rec['status']} ({basis})", flush=True)
    if rec["status"] == "time box":
        raise harness.TimeBox("time box reached during the T-ref build")
    if rec["status"] == "ok":
        o = FORMAL["oracle"]
        left = harness.time_left_s()
        oracle = "not run (time box)"
        try:
            if left is None or left > 0:
                r = run_logged([s.build / "sf_gate", "--oracle", tf, ref / "T-ref_1s-week", "--threads", cfg["threads"],
                                "--ref-parity", cfg.get("parity") or "none",
                                "--oracle-shared", o["starts_per_cell_shared"], "--oracle-full",
                                o["starts_per_cell_full"], "--out", ref / "oracle.json"], ref / "oracle.log", timeout=left)
                oracle = "pass" if r.returncode == 0 else "FAIL"
        except subprocess.TimeoutExpired:
            oracle = "time box"
        write_json(ref / "reference.json", {"result_sha256": rec["result_sha256"], "oracle": oracle,
                                            "oracle_pass": oracle == "pass", "config": cfg, "config_basis": basis,
                                            "session": s.k, "git": git_state(),
                                            "artifacts": manifest(ref),
                                            "binaries": machine.build_info(s.build)["binary_sha256"]})
        mirror_tree(ref)
        print(f"oracle: {oracle}", flush=True)
        if oracle in ("time box", "not run (time box)"):
            raise harness.TimeBox("time box reached before the oracle finished")
        if oracle != "pass":
            return "failed (reference oracle)"
    elif rec["status"] != "ok":
        return "incomplete: reference build " + rec["status"]


def step_deadline(s, seed):
    dl = PILOT["deadline"]
    tf = point(dl["task"])
    cfgs = [s.cpu_config()] + [s.gpu_config(p) for p in s.validated_paths()]
    items = [(cfg, B, k) for cfg in cfgs for B in dl["budgets_s"] for k in range(dl["reps"])]
    random.Random(seed).shuffle(items)
    od = s.out_dir("deadline")
    write_json(od / "order.json", {"seed": seed, "order": [[harness.config_key(c), B, k] for c, B, k in items]})
    for pos, (cfg, B, k) in enumerate(items):
        out = od / safe(f"{pos:03d}_B{B}_{harness.config_key(cfg)}")
        rec = harness.run_cold(tf, out, cfg, {"step": "deadline", "session": s.k, "task": "T-ref-est-prefix",
                                              "series": "week-prefix-75600", "task_file": str(tf), "rep": k,
                                              "position": pos}, build=s.build, budget_s=B, drain_ms=0,
                               replicate=k + 1, drop=s.drop(), timeout_s=2 * B + 120)
        try:
            pre = json.loads((out / "prefix.json").read_text())
            rec["exit_minus_cutoff_ms"] = (rec["t1_ns"] - pre["cutoff_ns"]) / 1e6
            rec["loop_end_minus_cutoff_ms"] = (pre["loop_end_ns"] - pre["cutoff_ns"]) / 1e6
            rec["prefix_trials"] = pre.get("G")
            rec["completed_unused"] = pre.get("completed_unused")
        except (OSError, ValueError, KeyError):
            pass
        rec["out_manifest"] = manifest(out)
        rec["out_mirrored"] = mirror_tree(out)
        append_jsonl(s.runs, rec)
        print(f"deadline B={B} {rec['config_key']}: {rec['status']} exit-cutoff {rec.get('exit_minus_cutoff_ms')}",
              flush=True)
        if rec.get("status") == "time box":
            raise harness.TimeBox("time box reached during the step")


STEP_FN = {"cpu-baseline": step_cpu_baseline, "threads": step_threads, "parity": step_parity, "rates": step_rates,
           "gpu-rates": step_gpu_rates, "microbatch": step_microbatch, "gpu-tasks": step_gpu_tasks,
           "diag": step_diag, "tref-rates": step_tref_rates, "tref-build": step_tref_build, "deadline": step_deadline}


def cmd_run(a):
    s = Session(a.label, a.session, a.build)
    if a.step not in s.plan["steps"]:
        sys.exit(f"step {a.step} is not in the plan {s.plan['steps']}")
    why = s.refuse_if_changed()
    if why:
        append_jsonl(s.sd / "steps.jsonl", {"step": a.step, "status": f"not run: {why}", "finished": now()})
        sys.exit(why)
    if a.step in GPU_STEPS and not s.validated_paths():
        append_jsonl(s.sd / "steps.jsonl", {"step": a.step, "status": "not run (no GPU path passed the device gate)",
                                            "finished": now()})
        sys.exit("no GPU path passed this session's device gate (or every passing path was excluded)")
    if not (s.sd / "env.json").exists():
        write_json(s.sd / "env.json", s.env)
    used = s.hours_used()
    left = s.hours_left(a.step)
    if left <= 0:
        append_jsonl(s.sd / "steps.jsonl", {"step": a.step, "status": "not started (time)", "hours_used": used,
                                            "session_hours_used": session_hours_used(s.sd), "finished": now()})
        sys.exit(f"no hours left (used {used:.2f} of {s.plan['hours_available']}; this session "
                 f"{session_hours_used(s.sd):.2f} of {s.plan.get('hours_per_session')}; step share "
                 f"{(s.plan.get('step_hours') or {}).get(a.step)})")
    seed = SEED + 1000 * s.k + STEPS.index(a.step)
    status = run_attempt(s.sd, a.step, left, lambda: STEP_FN[a.step](s, seed), plan_amendments=s.plan["amendments"])
    print(f"step {a.step}: {status}")
    if status.startswith(("incomplete", "failed")):
        sys.exit(1)


# ---- summary ----

def artifact_problems(runs):
    """Runs whose terminal record lists output files (out_manifest) that are missing or changed now, e.g. after a
    restore from Drive: 'artifact_missing'. The records themselves are never changed."""
    out = []
    for r in runs:
        man = r.get("out_manifest")
        if not man:
            continue
        now_ = manifest(r["out"]) if r.get("out") else {}
        bad = sorted(f for f, h in man.items() if now_.get(f) != h)
        if bad:
            out.append({"run_id": r.get("run_id"), "_session": r.get("_session"), "task": r.get("task"),
                        "series": r.get("series"), "config_key": r.get("config_key"), "status": r.get("status"),
                        "artifact_missing": bad[:10], "files_missing_or_changed": len(bad)})
    return out


def cmd_summary(a):
    d = label_dir(a.label)
    plan = effective_plan(d)
    runs, benches = records.label_records(a.label)  # tagged with session and hardware; lost runs included
    bad = records.invalid(a.label, runs)
    sessions = {}
    for sd in sorted(d.glob("session-*")):
        k = int(sd.name.split("-")[1])
        env = read_json(sd / "env.json") if (sd / "env.json").exists() else {}
        checks = sorted(sd.glob("checks/attempt-*/checks.json"), key=lambda p: int(p.parent.name.split("-")[1]))
        sessions[k] = {"collected": env.get("collected"), "runtime": env.get("runtime"), "gpu": env.get("gpu"),
                       "cpu_model": (env.get("identity") or {}).get("cpu_model"), "meminfo": env.get("meminfo"),
                       "topology": env.get("topology"), "capabilities": env.get("capabilities"),
                       "hardware_fingerprint": env.get("hardware_fingerprint") or machine.hardware_fingerprint(env),
                       "cpu_setup": read_json(sd / "cpu-setup.json") if (sd / "cpu-setup.json").exists() else None,
                       "time": session_time(sd), "excluded": k in bad["sessions"],
                       "checks": read_json(checks[-1]) if checks else None, "steps": read_jsonl(sd / "steps.jsonl")}
    summary = {"label": a.label, "machine": plan["machine"], "purpose": plan["purpose"], "plan_amendments": plan["amendments"],
               "sessions": sessions, "hours_used": hours_used(d),
               "hours_unconfirmed": {"lower": sum(v["time"]["unfinished_lower_h"] for v in sessions.values()),
                                     "upper": sum(v["time"]["unfinished_upper_h"] for v in sessions.values())},
               "machines": len({v["hardware_fingerprint"] for v in sessions.values()}),
               "excluded": {"gpu_paths": sorted(bad["paths"]), "sessions": sorted(bad["sessions"]),
                            "reason": "an output of the path (session: a CPU output) failed its identity or spot check, "
                                      "now or on later comparison with the CPU reference"},
               "pooling": "values are pooled across sessions only within one hardware fingerprint",
               "comparison_limits": [
                   "cpu-baseline is the starting CPU configuration (before the threads and parity choices) and "
                   "gpu-tasks uses the microbatch of that time: neither is CPU-best or GPU-best",
                   "the CPU steps ran before or after the GPU steps, not interleaved: time of day and host load are "
                   "confounded with the device; a direct ratio uses only the same session, cold kind and validity, "
                   "uncensored conditions",
                   "the threads choice at 1s-week/P24 is applied to the other workloads; it is not CPU-best there",
                   "a censored condition (timeout, time box, lost, failed or not started) has no E2E; its ok-only "
                   "median is conditional and the cap is a lower bound on its completion time",
                   "GPU outputs without a full CPU result hash are checked by a 256-trial sample only (exploratory)",
                   "Colab sessions are not independent hardware samples; kernel time alone is not an E2E advantage"]}

    # end-to-end times of the fixed tasks: per condition (task, series, mode, configuration, machine) the counts of
    # planned, started, ok, censored, lost, failed and not-started runs; the ok-only median is conditional unless the
    # condition is uncensored
    e2e = {}
    for r in runs:
        if r.get("task") not in records.FIXED or r.get("mode") not in ("cold", "warm"):
            continue
        kind = records._mode(r)
        key = f"{r['task']}|{r['series']}|{kind}|{r['config_key']}|{r['_hw']}"
        e = e2e.setdefault(key, {"task": r["task"], "series": r["series"], "mode": kind, "config": r["config_key"],
                                 "hardware": r["_hw"], "steps": set(), "config_role": set(), "recs": [],
                                 "memory_pressure": 0, "run_cap_s": r.get("run_cap_s")})
        e["recs"].append(r)
        e["steps"].add(r.get("step"))
        if r.get("config_role"):
            e["config_role"].add(r["config_role"])
        if (r.get("min_mem_available_mib") or 1e9) < 2048 or (r.get("pswpin") or 0) > 0:
            e["memory_pressure"] += 1
    for e in e2e.values():
        recs = e.pop("recs")
        e["steps"], e["config_role"] = sorted(filter(None, e["steps"])), sorted(e["config_role"])
        e["excluded"] = records.excluded(recs[0], bad)
        e["counts"] = records.counts(recs)
        e["statuses"] = [r.get("status") for r in recs]
        by_s = {}
        for r in recs:
            if r.get("status") == "ok":
                by_s.setdefault(str(r["_session"]), []).append(r["e2e_s"])
        e["ok_only_per_session"] = {k: describe(v) for k, v in by_s.items()}
        e["ok_only_all"] = describe([x for v in by_s.values() for x in v])
        e["uncensored"] = e["counts"]["uncensored"] and not e["excluded"]
        e["e2e"] = e["ok_only_all"] if e["uncensored"] else None
        e["censored_lower_bound_s"] = e["run_cap_s"] if e["counts"]["censored"] else None
        stages = [r["runner"]["stage_ms_sum"] for r in recs if r.get("runner") and r.get("status") == "ok"]
        e["select_stage_ms_median"] = {k: statistics.median(s[k] for s in stages) for k in stages[0]} if stages else None
    summary["e2e"] = list(e2e.values())
    summary["priming"] = [{k: r.get(k) for k in ("_session", "task", "series", "config_key", "status", "e2e_s", "error")}
                          for r in runs if r.get("mode") == "warm-priming"]

    # decisions and sweeps
    summary["decisions"] = [r for r in benches if str(r.get("kind", "")).endswith("decision")
                            or str(r.get("kind", "")).startswith("microbatch-decision")]
    mb = {}
    for r in benches:
        if r.get("kind") == "microbatch":
            k = (r["_session"], r["config"]["path"], r["config"]["microbatch"])
            m = mb.setdefault(k, {"session": k[0], "path": k[1], "microbatch": k[2], "rates": [], "statuses": [],
                                  "memory": None, "gpu_memory_used_mib_max": None})
            m["statuses"].append(r["status"])
            if r.get("trials_per_s"):
                m["rates"].append(r["trials_per_s"])
            m["memory"] = projection.memory_check(r.get("device")) or m["memory"]
            used = ((r.get("sampler") or {}).get("gpu") or {}).get("memory_used_mib_max")
            m["gpu_memory_used_mib_max"] = max(filter(None, [used, m["gpu_memory_used_mib_max"]]), default=None)
    summary["microbatch"] = [{**m, "rate": describe(m.pop("rates"))} for m in mb.values()]

    # projections per machine (hardware fingerprint): T1/T2 against an uncensored observed warm E2E on the same
    # configuration ('verified'), else 'unverified'; then T-ref from its cell rates (unverified unless a CPU projection
    # of that machine was verified)
    proj = []
    for hw in sorted({r["_hw"] for r in benches}):
        hb = [r for r in benches if r["_hw"] == hw]
        hr = [r for r in runs if r["_hw"] == hw]
        checks = projection_checks(hr, hb, plan["fixed_tasks"], bad)
        proj += [{"hardware": hw, **c} for c in checks]
        cpu_verified = [c["observed_over_projected_base"] for c in checks if c["check"] == "verified"
                        and c["config"].startswith("cpu:")]
        for cfg_key in sorted({r["config_key"] for r in hb if r.get("kind") == "tref-rate"}):
            proj.append({"hardware": hw, "config": cfg_key, "task": "T-ref", "series": "1s-week",
                         "projection_cold_loop": tref_projection(hb, cfg_key),
                         "check": (f"projection method verified on this machine (worst observed / projected "
                                   f"{max(cpu_verified):.3g})" if cpu_verified else
                                   "unverified estimate: no CPU projection of this machine was checked against an "
                                   "observed E2E; no claim on feasibility or cost of the T-ref build")})
    summary["projections"] = proj
    summary["tref_build"] = [read_json(p) for p in sorted(d.glob("reference/attempt-*/reference.json"))]
    summary["tref_runs"] = [{k: r.get(k) for k in ("_session", "status", "e2e_s", "projection_s", "projection_scale",
                                                   "config_key", "config_basis", "reference_dir")}
                            for r in runs if r.get("task") == "T-ref"]
    summary["deadline"] = [{k: r.get(k) for k in ("_session", "config_key", "budget_s", "status", "e2e_s",
                                                   "exit_minus_cutoff_ms", "loop_end_minus_cutoff_ms", "prefix_trials",
                                                   "completed_unused")} for r in runs if r.get("step") == "deadline"]
    summary["failures"] = [{k: r.get(k) for k in ("step", "_session", "task", "series", "mode", "kind", "config_key",
                                                   "status", "stderr_tail", "error", "spot_first", "spot_error")}
                           for r in runs + benches if r.get("status") not in (None, "ok")]
    summary["artifacts"] = artifact_problems(runs)
    # identity of fixed-task outputs: every distinct result hash per task and series, with the devices that produced it
    ident = {}
    ref = read_json(d / "hashes.json") if (d / "hashes.json").exists() else {}
    reverify = read_jsonl(d / "reverify.jsonl")
    for r in runs:
        if r.get("task") not in records.FIXED or not r.get("result_sha256"):
            continue
        k = f"{r['task']}/{r['series']}"
        e = ident.setdefault(k, {"task": r["task"], "series": r["series"], "cpu_reference": ref.get(k), "hashes": {}})
        h = e["hashes"].setdefault(r["result_sha256"], {"configs": set(), "statuses": []})
        h["configs"].add(r["config_key"])
        h["statuses"].append(r["status"])
    for k, e in ident.items():
        e["hashes"] = {h: {"configs": sorted(v["configs"]), "statuses": v["statuses"],
                           "matches_cpu_reference": None if not e["cpu_reference"] else h == e["cpu_reference"]}
                       for h, v in e["hashes"].items()}
        e["distinct"] = len(e["hashes"])
        e["reverified"] = [v for v in reverify if f"{v['task']}/{v['series']}" == k]
        e["basis"] = ("full CPU result hash (outputs compared with it, earlier ones re-verified)" if e["cpu_reference"]
                      else "no full CPU result hash: outputs checked by the 256-trial spot sample only (exploratory)")
    summary["identity"] = list(ident.values())
    summary["source_sha256"] = {str(p.relative_to(d)): sha256_file(p) for p in sorted(d.glob("session-*/*.jsonl"))}
    write_json(d / "pilot-summary.json", summary)
    (d / "pilot-summary.md").write_text(summary_md(summary))
    print(summary_md(summary))


def summary_md(s):
    f = lambda x: "-" if x is None else f"{x:.3g}"  # noqa: E731
    hu = s["hours_unconfirmed"]
    lines = [f"# Pilot summary: {s['label']} ({s['machine']}, {s['purpose']})", "",
             f"Sessions: {len(s['sessions'])} on {s['machines']} machine(s); hours used {s['hours_used']:.2f} "
             f"(of which cut-off attempts {hu['lower']:.2f}–{hu['upper']:.2f}); plan amendments "
             f"{s['plan_amendments']}. Descriptive results of one machine; not pooled with formal launches.", ""]
    ex = s["excluded"]
    if ex["gpu_paths"] or ex["sessions"]:
        lines += [f"**Excluded** (output check failed): GPU paths {ex['gpu_paths'] or '-'}, sessions "
                  f"{ex['sessions'] or '-'}. Their records give no metric.", ""]
    lines += ["Limits of comparison: " + "; ".join(s["comparison_limits"]) + ".", "",
              "## End-to-end time of the fixed tasks", "",
              "E2E is given only for an uncensored condition (every planned run ok). Otherwise the ok-only median is a "
              "conditional statistic, shown in brackets, and the run cap is a lower bound on the completion time.", "",
              "| task | series | mode | config | step | machine | planned / started / ok / censored / lost / failed / "
              "not started | completion (planned; started) | E2E median [min, max] s | memory pressure |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for e in s["e2e"]:
        c = e["counts"]
        cnt = f"{c['planned']} / {c['started']} / {c['ok']} / {c['censored']} / {c['lost']} / {c['failed']} / {c['not_started']}"
        comp = f"{f(c['completion'])}; {f(c['completion_started'])}"
        o = e["ok_only_all"]
        if e["excluded"]:
            val = "excluded"
        elif e["uncensored"]:
            val = f"{f(o['median'])} [{f(o['min'])}, {f(o['max'])}]"
        else:
            val = (f"censored{' (> ' + f(e['censored_lower_bound_s']) + ' s)' if e['censored_lower_bound_s'] else ''}"
                   f"{'; ok-only (' + f(o['median']) + ')' if o else ''}")
        lines.append(f"| {e['task']} | {e['series']} | {e['mode']} | {e['config']} | {','.join(e['steps'])} | "
                     f"{e['hardware']} | {cnt} | {comp} | {val} | {e['memory_pressure']} |")
    lines += ["", "## GPU microbatch sweep (representative point)", "",
              "| session | path | microbatch | median rate /s | statuses | planned MiB | reported MiB | nvidia-smi max MiB |",
              "|---|---|---|---|---|---|---|---|"]
    for m in s["microbatch"]:
        mem = m["memory"] or {}
        mib = lambda b: "-" if b is None else f"{b / 2**20:.0f}"  # noqa: E731
        lines.append(f"| {m['session']} | {m['path']} | {m['microbatch']} | {f((m['rate'] or {}).get('median'))} | "
                     f"{','.join(sorted(set(m['statuses'])))} | {mib(mem.get('planned_bytes'))} | "
                     f"{mib(mem.get('reported_bytes'))} | {f(m['gpu_memory_used_mib_max'])} |")
    lines += ["", "## Projections", "", "| config | task | series | conservative / base / optimistic s | observed warm s | "
              "observed / base | check |", "|---|---|---|---|---|---|---|"]
    for p in s["projections"]:
        pr = p.get("projection_warm") or p.get("projection_cold_loop") or {}
        sec = pr.get("seconds")
        band = " / ".join(f(sec[k]) for k in projection.SCENARIOS) if sec else pr.get("status", "-")
        lines.append(f"| {p['config']} | {p['task']} | {p['series']} | {band} | {f(p.get('observed_warm_median_s'))} | "
                     f"{f(p.get('observed_over_projected_base'))} | {p.get('check', '-')} |")
    lines += ["", "## Output identity of the fixed tasks", "",
              "| task | series | distinct result hashes | CPU reference | re-verified (mismatches) | basis |",
              "|---|---|---|---|---|---|"]
    for e in s.get("identity", []):
        rv = e["reverified"]
        lines.append(f"| {e['task']} | {e['series']} | {e['distinct']} | {'yes' if e['cpu_reference'] else 'no'} | "
                     f"{len(rv)} ({sum(1 for v in rv if v['match'] is False)}) | {e['basis']} |")
    lines += ["", f"Failures and non-ok runs: {len(s['failures'])}; warm primings {len(s['priming'])}; runs whose "
              f"output files are missing or changed (artifact_missing): {len(s['artifacts'])} "
              "(pilot-summary.json lists each)", ""]
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
    p.add_argument("--cold-kind", choices=["auto", "process-cold"], default="auto",
                   help="process-cold: never drop the page cache (a verification of a process-cold baseline)")
    p.add_argument("--scaling-record", help="verification: the committed scaling-decision.json")
    p.add_argument("--session-hours", type=float, help="hours available per session (default: only the total)")
    p.add_argument("--run-cap-min", type=float,
                   help="minutes after which a fixed-task run is killed and its condition stops (needs --cut)")
    p.add_argument("--step-hours", nargs="*", default=[],
                   help="STEP=H: at most H hours of a session for that step (e.g. cpu-baseline=1.5)")
    p = sub.add_parser("amend")
    p.add_argument("--label", required=True)
    p.add_argument("--reason", required=True)
    p.add_argument("--set", nargs="+", required=True, help="key=json, e.g. sessions=1 or hours_available=6")
    p = sub.add_parser("check")
    p.add_argument("--label", required=True)
    p.add_argument("--session", type=int, required=True)
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

"""Validation of formal records before any verdict (README "Verdict rules", "Status of every item").

A launch attempt counts only if its record is complete and consistent: the right stage, role and index; the formal
machine's identity; the committed scaling decision, freeze and code it ran with; every planned run present exactly
once (order.json against runs.jsonl and the started records); every fixed-n and T-ref run valid, cache-cold when
cold, and reproducing the reference output. Anything else excludes the whole launch, which is then replaced by a new
attempt in the same slot. Pairs must be on distinct instances, in one region, on one UTC calendar day, in the
registered order. Gates are checked for what they claim (device, full or partial, starts, items, code).
"""

from pathlib import Path

from . import formal
from .common import FORMAL, read_json, read_jsonl

FIXED_TASKS = ("T1-curve", "T2-policy-map", "T-ref")
COUNTED_OK = ("ok", "timeout")  # timeout = censored at exactly the timeout
FULL_GATE_ITEMS = {1, 2, 3, 4, 5, 6, 7, 8}
PARTIAL_GATE_ITEMS = {1, 2, 3, 5, 7, 8}


def expected_records(item):
    return item["reps"] if item["kind"] == "warm-block" else 1


def unfinished(d):
    """Run ids started but never finished (lost to an interruption, OOM kill or crash of the harness)."""
    started = {r["run_id"] for r in read_jsonl(Path(d) / "attempts.jsonl") if r.get("event") == "started"}
    done = set()
    for f in ("runs.jsonl", "rates.jsonl", "calibration.jsonl", "tdiag.jsonl"):
        done |= {r.get("run_id") for r in read_jsonl(Path(d) / f)}
    for sub in Path(d).glob("*/"):
        for f in ("runs.jsonl", "calibration.jsonl"):
            done |= {r.get("run_id") for r in read_jsonl(sub / f)}
    return sorted(started - done)


def launch_problems(d, stage, role, i, freeze=None, decision_sha=None):
    """Problems of one launch attempt (empty list = valid)."""
    d = Path(d)
    if not (d / "env.json").exists():
        return ["no env.json"]
    env = read_json(d / "env.json")
    out = []
    if env.get("stage") != stage or env.get("role") != role or env.get("launch_index") != i:
        out.append(f"env says stage {env.get('stage')} role {env.get('role')} index {env.get('launch_index')}")
    if env.get("status") != "complete":
        out.append(f"status {env.get('status')}")
    if (env.get("identity") or {}).get("match") is not True:
        out.append("hardware mismatch")
    if decision_sha and env.get("scaling_decision_sha256") != decision_sha:
        out.append("ran under another scaling decision")
    if freeze is not None:
        if env.get("freeze_sha256") != freeze["_sha256"]:
            out.append("ran under another freeze")
        if env.get("code_sha256") != freeze["code_sha256"]:
            out.append("code differs from the frozen code")
    if not (d / "order.json").exists():
        return out + ["no order.json"]
    order = read_json(d / "order.json")["order"]
    runs = [r for r in read_jsonl(d / "runs.jsonl") if r.get("kind") != "boot-cold"]
    by_pos = {}
    for r in runs:
        by_pos.setdefault(r.get("position"), []).append(r)
    for pos, item in enumerate(order):
        got = by_pos.pop(pos, [])
        if len(got) != expected_records(item):
            out.append(f"order item {pos} ({item['kind']} {item['task']} {item['series']}): {len(got)} records, "
                       f"{expected_records(item)} expected")
            continue
        if item["kind"] == "deadline":
            continue  # late/invalid deadline runs are penalised in H4, not excluded
        for r in got:
            bad = []
            if r.get("status") not in COUNTED_OK:
                bad.append(f"status {r.get('status')}")
            if r.get("status") == "ok" and (r.get("identity_ok") is False or
                                            (stage == "D" and r.get("identity_ok") is not True)):
                bad.append("output differs from the reference or was not compared with it")
            if r.get("mode") == "cold" and r.get("cold_kind") != "cache-cold":
                bad.append(f"cold kind {r.get('cold_kind')}")
            if r.get("status") == "ok" and r.get("task") in ("T1-curve", "T2-policy-map") and r.get("spot_mismatches") != 0:
                bad.append(f"spot mismatches {r.get('spot_mismatches')}")
            if bad:
                out.append(f"order item {pos} rep {r.get('rep')}: {', '.join(bad)}")
    if by_pos:
        out.append(f"records outside the order: positions {sorted(by_pos, key=str)}")
    lost = unfinished(d)
    if lost:
        out.append(f"{len(lost)} started runs without a terminal record")
    return out


def chosen(slot, stage, role, i, freeze=None, decision_sha=None):
    """(attempt dir or None, report) for one slot: the first attempt that validates. Attempts after a valid one are
    reported and never used (no choice among valid attempts)."""
    report, pick = [], None
    for a in formal.attempts(slot):
        pr = launch_problems(a, stage, role, i, freeze, decision_sha)
        report.append({"attempt": a.name, "problems": pr, "used": not pr and pick is None})
        if not pr and pick is None:
            pick = a
        elif not pr:
            report[-1]["problems"] = ["a later valid attempt of an already valid slot: not used"]
    return pick, report


def pair_problems(cpu_d, gpu_d, i):
    ce, ge = read_json(Path(cpu_d) / "env.json"), read_json(Path(gpu_d) / "env.json")
    out = []
    ci, gi = (ce.get("ec2") or {}), (ge.get("ec2") or {})
    if not ci.get("instance_id") or ci.get("instance_id") == gi.get("instance_id"):
        out.append("the two launches are not on two distinct recorded instances")
    if ci.get("region") != gi.get("region"):
        out.append(f"regions differ ({ci.get('region')}, {gi.get('region')})")
    if str(ce.get("started", ""))[:10] != str(ge.get("started", ""))[:10]:
        out.append(f"not on one UTC calendar day ({ce.get('started')}, {ge.get('started')})")
    first = "cpu" if str(ce.get("started")) <= str(ge.get("started")) else "gpu"
    if first != formal.first_role(i):
        out.append(f"{first} launch went first; pair {i} is registered to start with {formal.first_role(i)}")
    return out


def distinct_instances(dirs):
    """Instance ids that occur in more than one launch."""
    seen, dup = {}, set()
    for d in dirs:
        iid = (read_json(Path(d) / "env.json").get("ec2") or {}).get("instance_id")
        if iid in seen:
            dup.add(iid)
        seen[iid] = d
    return sorted(x for x in dup if x)


def gate_problems(path, *, device, full, path_name=None, code=None, gpu_name=None):
    """What a gate report must show to count: the device it ran on, full or partial coverage, the registered number
    of starts, every item of that coverage passed, and (if given) the code and GPU it ran with."""
    if not Path(path).exists():
        return [f"{path} missing"]
    g = read_json(path)
    out = []
    if g.get("device") != device:
        out.append(f"device {g.get('device')}, {device} required")
    if bool(g.get("partial")) == full:
        out.append("partial gate where the full gate is required" if full else "a partial gate is required")
    if full and g.get("starts_per_combination") != FORMAL["gate"]["starts_per_combination"]:
        out.append(f"starts per combination {g.get('starts_per_combination')}")
    items = {it.get("item"): it for it in g.get("items", [])}
    need = FULL_GATE_ITEMS if full else PARTIAL_GATE_ITEMS
    if device == "cpu":
        need = need - {2}  # item 2 is the device stream check
    for k in sorted(need):
        it = items.get(k)
        if not it or it.get("pass") is not True:
            out.append(f"item {k} missing or failed")
    if g.get("pass") is not True:
        out.append("gate did not pass")
    if path_name and path_name not in str(g.get("config", "")):
        out.append(f"gate config {g.get('config')} is not path {path_name}")
    meta = Path(path).with_suffix(".meta.json")
    if code or gpu_name:
        m = read_json(meta) if meta.exists() else {}
        if code and m.get("code_sha256") != code:
            out.append("gate ran with other code")
        if gpu_name and gpu_name not in str(m.get("gpu_name")):
            out.append(f"gate ran on {m.get('gpu_name')}")
    return out


def goldens(gate_path):
    g = read_json(gate_path)
    return {it["item"]: it.get("golden_sha256") for it in g.get("items", []) if it.get("golden_sha256")}

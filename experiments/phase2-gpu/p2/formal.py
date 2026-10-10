"""Formal comparison plumbing: the machines named by the committed scaling decision, launch directories, and the
condition list of a launch."""

import sys

from .common import FORMAL, MACHINES, PROTOCOL, RESULTS, committed_unchanged, read_json
from .tasks import main_task

DECISION = RESULTS / "scaling" / "scaling-decision.json"
ROLES = ("cpu", "gpu", "size-control")


def decision(require_commit=True):
    """The scaling decision, which must be complete, committed, and proceed to the formal comparison."""
    sys_path = str(DECISION.parents[2])
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    import scaling  # noqa: E402  (experiments/phase2-gpu/scaling.py)
    if not DECISION.exists():
        raise SystemExit(f"{DECISION} missing: the formal comparison follows a recorded scaling decision")
    rec = read_json(DECISION)
    pr = scaling.problems(rec)
    if pr:
        raise SystemExit("scaling decision incomplete: " + "; ".join(pr[:5]))
    if require_commit and not committed_unchanged(DECISION):
        raise SystemExit("commit the scaling decision before any formal launch")
    if rec["formal"]["proceed"] is not True:
        raise SystemExit("the scaling decision does not proceed to the formal comparison "
                         "(formal H3/H4 are 'not measured (no scaling)')")
    return rec


def profile_of(rec, role):
    key = {"cpu": "cpu", "gpu": "gpu", "size-control": "cpu_size_control"}[role]
    name = rec["formal"].get(key)
    if not name:
        raise SystemExit(f"the scaling decision names no machine for role {role}")
    return name, MACHINES[name]


def thread_candidates(profile, topo):
    c = profile.get("thread_candidates")
    if c:
        return list(c)
    return sorted({topo["physical_cores"], topo["logical"]})


def formal_dir(label):
    return RESULTS / "formal" / label


def launch_dir(label, role, i):
    return formal_dir(label) / (f"launch-{i}-{role}" if role != "size-control" else "size-control")


def conditions(role, stage, pair):
    """Every run of one launch before shuffling. Cold repetitions are separate items; the warm repetitions of a
    condition are one block (one priming task outside the timing, then the timed requests)."""
    v = FORMAL["verdicts"]["reps_per_launch"]
    fixed = [(t, s) for t in ("T1-curve", "T2-policy-map") for s in PROTOCOL["tasks"][t]["series"]]
    out = []
    if role == "size-control":
        return [{"kind": "cold", "task": t, "series": s, "rep": r} for t, s in fixed for r in range(3)]
    for t, s in fixed:
        out += [{"kind": "cold", "task": t, "series": s, "rep": r} for r in range(v["T1/T2 cold"])]
        out.append({"kind": "warm-block", "task": t, "series": s, "reps": v["T1/T2 warm"]})
    if stage == "A":
        return out
    out += [{"kind": "cold", "task": "T-ref", "series": "1s-week", "rep": r} for r in range(v["T-ref cold"])]
    est = PROTOCOL["tasks"]["T-ref-est"]
    k = est["seeds_per_launch_pair"]
    for B in est["budgets_s"]:
        out += [{"kind": "deadline", "task": "T-ref-est", "series": "1s-week", "budget_s": B,
                 "replicate": k * (pair - 1) + j + 1, "rep": j} for j in range(k)]
    return out


def condition_key(c):
    if c["kind"] == "deadline":
        return f"T-ref-est/B{c['budget_s']}"
    mode = "warm" if c["kind"] in ("warm", "warm-block") else "cold"
    return f"{c['task']}/{c['series']}/{mode}"


def task_file(c):
    return main_task(c["task"], c["series"])

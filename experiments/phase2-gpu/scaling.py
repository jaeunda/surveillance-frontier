"""Part II of the protocol: the scaling decision (P3) and its verification on the target machine (P4).

    scaling.py draft  --pilot LABEL [--pilot LABEL ...]       writes results/scaling/scaling-decision.json (draft)
    scaling.py check  [FILE]                                   completeness and budget; required before paid runs
    scaling.py verify [FILE] --action I --target-label LABEL   P4: same workload on the target, per metric

The draft fills in what the pilot measured (observations, the metric catalogue with values, projections); every
judgement is left as "TODO" for a person to write: required outputs and schedule, alternatives tried without
scaling, the actions with their hypotheses, expected effects, falsifying measurements, cost and stop conditions, and
whether and on which machines the formal comparison proceeds. `check` refuses a record with any TODO, an action
without a falsifiable metric, formal machines without an identity rule, or planned cost above the remaining cap.
The record must be committed before any run it justifies (pilot.py plan --purpose verification, launch.py check this).
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from p2 import records  # noqa: E402
from p2.common import MACHINES, RESULTS, SCALING, now, read_json, sha256_file, write_json  # noqa: E402

DEFAULT = RESULTS / "scaling" / "scaling-decision.json"
TODO = "TODO"


def cmd_draft(a):
    if a.out.exists() and not a.force:
        sys.exit(f"{a.out} exists (use --force to rebuild the draft; a committed record is amended, not rebuilt)")
    pilots = []
    for label in a.pilot:
        d = RESULTS / "pilot" / label
        if not (d / "pilot-summary.json").exists():
            sys.exit(f"{label}: run pilot.py summary first")
        s = read_json(d / "pilot-summary.json")
        pilots.append({"label": label, "machine": s["machine"], "summary_sha256": sha256_file(d / "pilot-summary.json"),
                       "sessions": len(s["sessions"]), "hours_used": s["hours_used"],
                       "observations": {k: s[k] for k in ("e2e", "microbatch", "projections", "tref_runs", "deadline",
                                                          "decisions")},
                       "failures": s["failures"],
                       "environment": {k: {kk: v.get(kk) for kk in ("runtime", "gpu", "cpu_model", "meminfo",
                                                                     "capabilities")}
                                       for k, v in s["sessions"].items()},
                       "metric_catalog": records.catalog(label)})
    rec = {
        "status": "draft",
        "created": now(),
        "pilot": pilots,
        "outputs_and_schedule": {"required_outputs": TODO, "schedule": TODO,
                                 "note": "the research schedule is not the H4 budget B; the Phase 1 one-hour line is "
                                         "not a criterion here"},
        "alternatives_without_scaling": [{"what": TODO, "range_checked": TODO, "result": TODO}],
        "actions": [{
            "action": TODO, "target": TODO, "constraint_observed": TODO, "hypothesis": TODO,
            "metrics": [{"name": TODO, "kind": TODO, "baseline_label": TODO, "baseline": None,
                         "predicted": {"conservative": None, "base": None, "optimistic": None},
                         "outside_observed_range": TODO, "threshold_gain": None}],
            "falsification": TODO,
            "cost": {"instance_hours": None, "usd_on_demand": None, "includes": TODO},
            "stop": TODO,
        }],
        "formal": {"proceed": TODO, "cpu": TODO, "gpu": TODO, "cpu_size_control": None, "reason": TODO},
        "budget": {"cap_usd": SCALING["budget_cap_usd"], "spent_usd": None, "planned_usd": None,
                   "rule": SCALING["budget_rule"]},
        "conditionality": "a comparison on machines chosen after seeing pilot results is conditional on those machines; "
                          "it is not a claim about the best CPU or GPU",
    }
    write_json(a.out, rec)
    print(f"draft written: {a.out}; replace every TODO, set baselines from the metric catalogue, then scaling.py check")


def _todos(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _todos(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _todos(v, f"{path}[{i}]")
    elif node == TODO:
        yield path


def problems(rec):
    out = [f"TODO at {p}" for p in _todos(rec)]
    actions = rec.get("actions") or []
    kinds = [x.get("action") for x in actions]
    if not actions:
        out.append("no action (use action 'none' to record a decision not to scale)")
    for i, act in enumerate(actions):
        k = act.get("action")
        if k not in SCALING["actions"]:
            out.append(f"actions[{i}]: action must be one of {list(SCALING['actions'])}")
            continue
        if k == "none":
            continue
        if act.get("target") not in MACHINES:
            out.append(f"actions[{i}]: target must be a machine profile ({', '.join(MACHINES)})")
        if not act.get("metrics"):
            out.append(f"actions[{i}]: no metric to verify")
        for j, m in enumerate(act.get("metrics") or []):
            where = f"actions[{i}].metrics[{j}]"
            if m.get("kind") not in ("e2e", "rate"):
                out.append(f"{where}: kind must be e2e or rate")
            if not isinstance(m.get("baseline"), (int, float)) or m["baseline"] <= 0:
                out.append(f"{where}: baseline value missing (take it from the metric catalogue)")
            pred = m.get("predicted") or {}
            if not all(isinstance(pred.get(s), (int, float)) for s in ("conservative", "base", "optimistic")):
                out.append(f"{where}: predicted conservative/base/optimistic missing")
            if not isinstance(m.get("threshold_gain"), (int, float)) or m["threshold_gain"] < 1:
                out.append(f"{where}: threshold_gain (>= 1) missing: the gain below which the hypothesis is withdrawn")
        c = act.get("cost") or {}
        if not isinstance(c.get("usd_on_demand"), (int, float)):
            out.append(f"actions[{i}]: cost.usd_on_demand missing")
    if "none" in kinds and len(kinds) > 1:
        out.append("action 'none' excludes other actions")
    f = rec.get("formal") or {}
    if f.get("proceed") not in (True, False):
        out.append("formal.proceed must be true or false")
    if f.get("proceed") is True:
        for role in ("cpu", "gpu"):
            prof = MACHINES.get(f.get(role))
            if not prof or not prof.get("identity"):
                out.append(f"formal.{role}: a machine profile with an identity rule is required for verdict launches")
        if f.get("cpu_size_control") and f["cpu_size_control"] not in MACHINES:
            out.append("formal.cpu_size_control: unknown profile")
    if kinds == ["none"] and f.get("proceed") is True:
        out.append("no scaling but formal.proceed: name the action that moves to the formal machines")
    b = rec.get("budget") or {}
    planned = sum((act.get("cost") or {}).get("usd_on_demand") or 0 for act in actions)
    if not isinstance(b.get("spent_usd"), (int, float)):
        out.append("budget.spent_usd missing")
    if not isinstance(b.get("planned_usd"), (int, float)):
        out.append("budget.planned_usd missing (all paid work: verification, gates, calibration, launches, reserve)")
    elif b["planned_usd"] < planned:
        out.append(f"budget.planned_usd {b['planned_usd']} is below the actions' sum {planned:.2f}")
    if isinstance(b.get("planned_usd"), (int, float)) and isinstance(b.get("spent_usd"), (int, float)):
        if b["spent_usd"] + b["planned_usd"] > b.get("cap_usd", SCALING["budget_cap_usd"]):
            out.append(f"spent + planned = {b['spent_usd'] + b['planned_usd']:.2f} USD exceeds the cap "
                       f"{b.get('cap_usd')} (the cap is not raised automatically)")
    if b.get("cap_usd") != SCALING["budget_cap_usd"]:
        out.append("budget.cap_usd differs from protocol.json")
    for p in rec.get("pilot") or []:
        s = RESULTS / "pilot" / p["label"] / "pilot-summary.json"
        if not s.exists() or sha256_file(s) != p["summary_sha256"]:
            out.append(f"pilot {p['label']}: summary changed since the draft (re-draft or explain in an amendment)")
    return out


def cmd_check(a):
    if not a.file.exists():
        sys.exit(f"{a.file} missing (scaling.py draft)")
    rec = read_json(a.file)
    pr = problems(rec)
    for p in pr:
        print(f"- {p}")
    print("complete" if not pr else f"{len(pr)} problems")
    sys.exit(1 if pr else 0)


def cmd_verify(a):
    rec = read_json(a.file)
    if problems(rec):
        sys.exit("the scaling record does not pass scaling.py check")
    act = rec["actions"][a.action]
    target = RESULTS / "pilot" / a.target_label
    plan = read_json(target / "pilot-plan.json")
    if plan.get("purpose") != "verification" or (plan.get("scaling_record") or {}).get("sha256") != sha256_file(a.file):
        sys.exit("the target label was not planned as a verification of this record (pilot.py plan --purpose "
                 "verification --scaling-record FILE)")
    if plan["machine"] != act["target"]:
        sys.exit(f"target label ran on {plan['machine']}, the action names {act['target']}")
    rows = []
    for m in act["metrics"]:
        value, cfg, n = records.observed(a.target_label, m)
        g = records.gain(m, m["baseline"], value)
        pg = {s: records.gain(m, m["baseline"], m["predicted"][s]) for s in ("conservative", "base", "optimistic")}
        if g is None:
            verdict = "indeterminate"
        elif g >= m["threshold_gain"]:
            verdict = "supported"
        else:
            verdict = "not supported"
        lo, hi = min(pg.values()), max(pg.values())
        rows.append({"name": m["name"], "metric": {k: m.get(k) for k in ("kind", "task", "series", "mode", "point", "config")},
                     "baseline": m["baseline"], "observed": value, "observed_config": cfg, "n": n, "gain": g,
                     "predicted_gain": pg, "within_predicted": None if g is None else lo <= g <= hi,
                     "threshold_gain": m["threshold_gain"], "verdict": verdict})
    out = {"record": str(a.file), "record_sha256": sha256_file(a.file), "action": a.action, "target": act["target"],
           "target_label": a.target_label, "time": now(), "metrics": rows,
           "note": "descriptive: one baseline machine and one target machine, medians over sessions; no interval. "
                   "A machine change also changes architecture, host and OS; the gain is not attributed to one "
                   "resource alone"}
    path = a.file.parent / f"verification_{a.target_label}_action{a.action}.json"
    write_json(path, out)
    f = lambda x: "-" if x is None else f"{x:.3g}"  # noqa: E731
    print("| metric | baseline | observed | gain | predicted gain | threshold | verdict |\n|---|---|---|---|---|---|---|")
    for r in rows:
        pg = r["predicted_gain"]
        print(f"| {r['name']} | {f(r['baseline'])} | {f(r['observed'])} | {f(r['gain'])} | "
              f"{f(pg['conservative'])}–{f(pg['optimistic'])} | {r['threshold_gain']} | {r['verdict']} |")
    print(f"written: {path}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("draft")
    p.add_argument("--pilot", action="append", required=True)
    p.add_argument("--out", type=Path, default=DEFAULT)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("check")
    p.add_argument("file", type=Path, nargs="?", default=DEFAULT)
    p = sub.add_parser("verify")
    p.add_argument("file", type=Path, nargs="?", default=DEFAULT)
    p.add_argument("--action", type=int, required=True)
    p.add_argument("--target-label", required=True)
    a = ap.parse_args()
    {"draft": cmd_draft, "check": cmd_check, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    main()

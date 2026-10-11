"""Observed values of a pilot or verification label, read from its raw records (runs.jsonl, bench.jsonl, attempts.jsonl).

A metric names what to compare between a baseline machine and a target machine:

    {"name": "...", "kind": "e2e", "task": "T2-policy-map", "series": "1s-week", "mode": "cache-cold"|"process-cold"|"warm",
     "config": "<config_key>"|"best"|"best-cpu"|"best-gpu", "session": K (optional)}
    {"name": "...", "kind": "rate", "point": "calib/1s-week_P24_L32", "config": "<config_key>"|"best"|...}
    {"name": "...", "kind": "completion" | "spread", "task": ..., "series": ..., "mode": ..., "config": ...}

E2E values are the median over all ok runs (lower is better); rates are the median trials/s over all ok bench runs at
the point (higher is better); completion is ok runs / all started runs of the condition (higher is better); spread is
max / min of the ok times (lower is better). "best" is each machine's own best configuration for that metric.
Records of sessions with different hardware fingerprints are never pooled: such a label needs "session" in the metric.
"""

import statistics

from .common import RESULTS, ROOT, read_json, read_jsonl

LOWER_IS_BETTER = ("e2e", "spread")


def label_records(label):
    """(runs, benches) of a label, each record tagged with its session and the session's hardware fingerprint; lost
    runs (started without a terminal record) are added to runs with status 'lost'."""
    from .machine import hardware_fingerprint
    d = RESULTS / "pilot" / label
    runs, benches = [], []
    for sd in sorted(d.glob("session-*")):
        k = int(sd.name.split("-")[1])
        env = read_json(sd / "env.json") if (sd / "env.json").exists() else {}
        hw = env.get("hardware_fingerprint") or hardware_fingerprint(env)
        tag = {"_session": k, "_hw": hw}
        r = [{**x, **tag} for x in read_jsonl(sd / "runs.jsonl")]
        b = [{**x, **tag} for x in read_jsonl(sd / "bench.jsonl")]
        done = {x.get("run_id") for x in r + b}
        for st in read_jsonl(sd / "attempts.jsonl"):
            if st.get("event") == "started" and st["run_id"] not in done and st.get("kind") in ("cold", "warm", "deadline"):
                r.append({**(st.get("condition") or {}), "mode": st["kind"], "status": "lost", "run_id": st["run_id"],
                          "config_key": st.get("config_key"), **tag})
        runs += r
        benches += b
    return runs, benches


def _point_of(bench_rec):
    t = bench_rec["task"]
    t = t[len("experiments/phase2-gpu/tasks/"):] if t.startswith("experiments/phase2-gpu/tasks/") else t
    return t[:-5] if t.endswith(".json") else t


def _config_match(key, want):
    if want == "best":
        return True
    if want == "best-cpu":
        return key.startswith("cpu:")
    if want == "best-gpu":
        return not key.startswith("cpu:")
    return key == want


def _mode(r):
    return r.get("cold_kind", "process-cold" if r.get("status") == "lost" else None) if r["mode"] == "cold" else r["mode"]


def values_by_config(runs, benches, metric):
    """{config_key: [values]} of a metric (for completion: [1/0 per started run])."""
    out = {}
    kind = metric["kind"]
    if kind in ("e2e", "spread", "completion"):
        for r in runs:
            if r.get("task") != metric["task"] or r.get("series") != metric["series"] or r.get("kind") == "boot-cold":
                continue
            if r.get("mode") not in ("cold", "warm") or not r.get("config_key"):
                continue
            if kind == "completion":
                if (r.get("status") == "lost" or _mode(r) == metric["mode"]) and _config_match(r["config_key"], metric["config"]):
                    out.setdefault(r["config_key"], []).append(1.0 if r.get("status") == "ok" else 0.0)
            elif r.get("status") == "ok" and _mode(r) == metric["mode"] and _config_match(r["config_key"], metric["config"]):
                out.setdefault(r["config_key"], []).append(r["e2e_s"])
    elif kind == "rate":
        for r in benches:
            if r.get("status") != "ok" or "trials_per_s" not in r or str(r.get("kind", "")).endswith("probe"):
                continue
            if _point_of(r) == metric["point"] and _config_match(r["config_key"], metric["config"]):
                out.setdefault(r["config_key"], []).append(r["trials_per_s"])
    else:
        raise ValueError(f"unknown metric kind {kind}")
    return out


def _value(kind, v):
    if kind == "completion":
        return sum(v) / len(v)
    if kind == "spread":
        return max(v) / min(v) if len(v) >= 2 else None
    return statistics.median(v)


def observed(label, metric):
    """(value, config_key, n) of a metric on a label, or (None, None, 0). Raises ValueError if the records span
    sessions on different hardware and the metric does not name a session."""
    runs, benches = label_records(label)
    if metric.get("session") is not None:
        runs = [r for r in runs if r["_session"] == metric["session"]]
        benches = [r for r in benches if r["_session"] == metric["session"]]
    by = values_by_config(runs, benches, metric)
    if not by:
        return None, None, 0
    hws = {r["_hw"] for r in (runs if metric["kind"] != "rate" else benches)
           if r.get("config_key") in by and (metric["kind"] == "rate" or r.get("task") == metric.get("task"))}
    if len(hws) > 1:
        raise ValueError(f"{label}: records from {len(hws)} different machines (hardware fingerprints {sorted(hws)}); "
                         "name the session in the metric")
    val = {k: _value(metric["kind"], v) for k, v in by.items()}
    val = {k: x for k, x in val.items() if x is not None}
    if not val:
        return None, None, 0
    pick = (min if metric["kind"] in LOWER_IS_BETTER else max)(val, key=val.get)
    return val[pick], pick, len(by[pick])


def gain(metric, baseline, value):
    """> 1 means the target is better: baseline / observed for times and spread, observed / baseline otherwise."""
    if baseline is None or value is None or baseline <= 0 or value <= 0:
        return None
    return baseline / value if metric["kind"] in LOWER_IS_BETTER else value / baseline


def catalog(label):
    """Every metric the label has data for, with its value, per session (the menu a scaling record picks from).
    Session-level entries carry "session"; a label-wide entry exists only if all its sessions share one machine."""
    runs, benches = label_records(label)
    sessions = sorted({r["_session"] for r in runs + benches})
    out, seen = [], set()

    def add(m):
        key = tuple(sorted((k, str(v)) for k, v in m.items()))
        if key in seen:
            return
        seen.add(key)
        try:
            m["value"], m["value_config"], m["n"] = observed(label, m)
        except ValueError:
            return
        if m["value"] is not None:
            out.append(m)

    for sess in [None] + sessions:
        for r in runs:
            if r.get("status") != "ok" or r.get("task") not in ("T1-curve", "T2-policy-map") or r.get("mode") not in ("cold", "warm"):
                continue
            if sess is not None and r["_session"] != sess:
                continue
            for cfg in (r["config_key"], "best-cpu" if r["config_key"].startswith("cpu:") else "best-gpu"):
                for kind in ("e2e", "completion", "spread"):
                    m = {"kind": kind, "task": r["task"], "series": r["series"], "mode": _mode(r), "config": cfg}
                    add({**m, "session": sess} if sess is not None else m)
        for r in benches:
            if r.get("status") != "ok" or str(r.get("kind", "")).endswith("probe") or "trials_per_s" not in r:
                continue
            if sess is not None and r["_session"] != sess:
                continue
            m = {"kind": "rate", "point": _point_of(r), "config": r["config_key"]}
            add({**m, "session": sess} if sess is not None else m)
    return out


def rel(path):
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)

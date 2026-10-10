"""Observed values of a pilot or verification label, read from its raw records (runs.jsonl, bench.jsonl).

A metric names what to compare between a baseline machine and a target machine:

    {"name": "...", "kind": "e2e", "task": "T2-policy-map", "series": "1s-week", "mode": "cache-cold"|"process-cold"|"warm",
     "config": "<config_key>"|"best"|"best-cpu"|"best-gpu"}
    {"name": "...", "kind": "rate", "point": "calib/1s-week_P24_L32", "config": "<config_key>"|"best"|...}

E2E values are the median over all ok runs of all sessions (lower is better); rates are the median trials/s over
all ok bench runs at the point (higher is better). "best" is each machine's own best configuration for that metric.
"""

import statistics

from .common import RESULTS, ROOT, read_jsonl


def label_records(label):
    d = RESULTS / "pilot" / label
    runs, benches = [], []
    for sd in sorted(d.glob("session-*")):
        runs += read_jsonl(sd / "runs.jsonl")
        benches += read_jsonl(sd / "bench.jsonl")
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


def values_by_config(runs, benches, metric):
    """{config_key: [values]} of a metric."""
    out = {}
    if metric["kind"] == "e2e":
        for r in runs:
            if r.get("status") != "ok" or r.get("task") != metric["task"] or r.get("series") != metric["series"]:
                continue
            mode = r.get("cold_kind") if r["mode"] == "cold" else r["mode"]
            if mode == metric["mode"] and _config_match(r["config_key"], metric["config"]):
                out.setdefault(r["config_key"], []).append(r["e2e_s"])
    elif metric["kind"] == "rate":
        for r in benches:
            if r.get("status") != "ok" or "trials_per_s" not in r or r.get("kind") == "microbatch-probe":
                continue
            if _point_of(r) == metric["point"] and _config_match(r["config_key"], metric["config"]):
                out.setdefault(r["config_key"], []).append(r["trials_per_s"])
    else:
        raise ValueError(f"unknown metric kind {metric['kind']}")
    return out


def observed(label, metric):
    """(value, config_key, n) of a metric on a label, or (None, None, 0)."""
    runs, benches = label_records(label)
    by = values_by_config(runs, benches, metric)
    if not by:
        return None, None, 0
    med = {k: statistics.median(v) for k, v in by.items()}
    pick = (min if metric["kind"] == "e2e" else max)(med, key=med.get)
    return med[pick], pick, len(by[pick])


def gain(metric, baseline, value):
    """> 1 means the target is better: baseline / observed for times, observed / baseline for rates."""
    if baseline is None or value is None or baseline <= 0 or value <= 0:
        return None
    return baseline / value if metric["kind"] == "e2e" else value / baseline


def catalog(label):
    """Every metric the label has data for, with its value (the menu a scaling record picks from)."""
    runs, benches = label_records(label)
    out = []
    seen = set()
    for r in runs:
        if r.get("status") != "ok" or r.get("task") not in ("T1-curve", "T2-policy-map"):
            continue
        mode = r.get("cold_kind") if r["mode"] == "cold" else r["mode"]
        for cfg in (r["config_key"], "best-cpu" if r["config_key"].startswith("cpu:") else "best-gpu"):
            key = ("e2e", r["task"], r["series"], mode, cfg)
            if key not in seen:
                seen.add(key)
                m = {"kind": "e2e", "task": r["task"], "series": r["series"], "mode": mode, "config": cfg}
                m["value"], m["value_config"], m["n"] = observed(label, m)
                out.append(m)
    for r in benches:
        if r.get("status") != "ok" or r.get("kind") == "microbatch-probe":
            continue
        key = ("rate", _point_of(r), r["config_key"])
        if key not in seen:
            seen.add(key)
            m = {"kind": "rate", "point": _point_of(r), "config": r["config_key"]}
            m["value"], m["value_config"], m["n"] = observed(label, m)
            out.append(m)
    return out


def rel(path):
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)

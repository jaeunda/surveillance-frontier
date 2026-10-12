"""Observed values of a pilot or verification label, read from its raw records (runs.jsonl, bench.jsonl, attempts.jsonl).

A metric names what to compare between a baseline machine and a target machine:

    {"name": "...", "kind": "e2e", "task": "T2-policy-map", "series": "1s-week", "mode": "cache-cold"|"process-cold"|"warm",
     "config": "<config_key>"|"best"|"best-cpu"|"best-gpu", "session": K (optional)}
    {"name": "...", "kind": "rate", "point": "calib/1s-week_P24_L32", "config": "<config_key>"|"best"|...}
    {"name": "...", "kind": "completion" | "spread", "task": ..., "series": ..., "mode": ..., "config": ...}

A condition is (task, series, mode, configuration). Its runs are counted as planned (every record of the condition,
runs not started included), started, ok, censored (timeout, time box), lost (started without a terminal record) and
not started. E2E and spread are defined only for an uncensored condition, i.e. one whose planned runs all finished ok:
the median of the ok runs of a condition with any timeout, time box, lost, failed or unstarted run is a conditional
statistic (ok runs only), is reported as such, and is never a metric value nor enters a "best" choice. Rates are the
median trials/s over ok bench runs at the point (higher is better). Completion is ok runs / planned runs of the
condition (higher is better; defined, as 0, also when no run finished); completion over started runs is reported next
to it. "best" is each machine's own best configuration among the eligible ones. Records of a GPU path that failed an
output check (identity or spot mismatch, or a later mismatch with the CPU reference) and every record of a session
whose CPU output failed a check are excluded. Records of sessions with different hardware fingerprints are never
pooled: such a label needs "session" in the metric.
"""

import statistics

from .common import RESULTS, ROOT, read_json, read_jsonl

LOWER_IS_BETTER = ("e2e", "spread")
FIXED = ("T1-curve", "T2-policy-map")
OUTPUT_FAILURES = ("identity_mismatch", "spot_mismatch")
CENSORED = ("timeout", "time box")


def label_records(label):
    """(runs, benches) of a label, each record tagged with its session and the session's hardware fingerprint; lost
    runs (started without a terminal record) are added with status 'lost': cold, warm and deadline runs and warm
    priming (mode 'warm-priming') to runs, benches to benches."""
    from .machine import hardware_fingerprint
    from colab_ops import manifest_errors, record_output
    d = RESULTS / "pilot" / label
    runs, benches = [], []
    for sd in sorted(d.glob("session-*")):
        k = int(sd.name.split("-")[1])
        env = read_json(sd / "env.json") if (sd / "env.json").exists() else {}
        hw = env.get("hardware_fingerprint") or hardware_fingerprint(env)
        tag = {"_session": k, "_hw": hw}
        r = [{**x, **tag} for x in read_jsonl(sd / "runs.jsonl")]
        for x in r:
            if x.get("out_manifest"):
                try:
                    missing = manifest_errors(record_output(d, x["out"]), x["out_manifest"])
                except (KeyError, ValueError) as exc:
                    missing = [str(exc)]
                if missing:
                    x.update({"status_at_run": x.get("status"), "status": "artifact_missing",
                              "artifact_missing": missing})
        b = [{**x, **tag} for x in read_jsonl(sd / "bench.jsonl")]
        done = {x.get("run_id") for x in r + b}
        for st in read_jsonl(sd / "attempts.jsonl"):
            if st.get("event") != "started" or st["run_id"] in done:
                continue
            lost = {**(st.get("condition") or {}), "status": "lost", "run_id": st["run_id"],
                    "config_key": st.get("config_key"), "started_at": st.get("time"), **tag}
            if st.get("kind") in ("cold", "warm", "deadline"):
                r.append({**lost, "mode": st["kind"]})
            elif st.get("kind") == "priming":
                r.append({**lost, "mode": "warm-priming"})
            elif st.get("kind") == "bench":
                b.append({**lost, "kind": "lost-bench"})
        runs += r
        benches += b
    return runs, benches


def config_path(key):
    """GPU path of a configuration key (cuda:inc:b64:... -> 'inc'), None for a CPU key."""
    parts = str(key or "").split(":")
    return parts[1] if len(parts) > 1 and parts[0] != "cpu" else None


def invalid(label, runs=None):
    """{"paths": GPU paths with an output failure, "sessions": sessions whose CPU output failed a check} of a label,
    including mismatches found later against the CPU reference (reverify.jsonl)."""
    if runs is None:
        runs, _ = label_records(label)
    paths, sessions = set(), set()
    for r in runs:
        if r.get("status_at_run", r.get("status")) in OUTPUT_FAILURES:
            p = config_path(r.get("config_key"))
            if p:
                paths.add(p)
            else:
                sessions.add(r["_session"])
    for v in read_jsonl(RESULTS / "pilot" / label / "reverify.jsonl"):
        if v.get("match") is False:
            p = config_path(v.get("config_key"))
            if p:
                paths.add(p)
            else:
                sessions.add(v.get("session"))
    return {"paths": paths, "sessions": sessions}


def excluded(r, bad):
    return r.get("_session") in bad["sessions"] or config_path(r.get("config_key")) in bad["paths"]


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


ANY_COLD = "cold (kind unknown)"


def _mode(r):
    """cache-cold / process-cold / warm; a lost or not-started cold run without a known cold kind is ANY_COLD and
    matches either cold mode."""
    if r["mode"] != "cold":
        return r["mode"]
    kind = r.get("cold_kind")
    if kind in (None, "unknown") and (r.get("status") == "lost" or str(r.get("status", "")).startswith("not started")):
        return ANY_COLD
    return kind


def _mode_match(r, mode):
    m = _mode(r)
    return m == mode or (m == ANY_COLD and mode in ("cache-cold", "process-cold"))


def counts(recs):
    """planned / started / ok / censored / lost / failed / not started of the records of one condition."""
    c = {"planned": 0, "started": 0, "ok": 0, "censored": 0, "lost": 0, "failed": 0, "not_started": 0}
    for r in recs:
        st = str(r.get("status", ""))
        n = int(r.get("runs_not_started", 1)) if st.startswith("not started") else 1
        c["planned"] += n
        if st.startswith("not started"):
            c["not_started"] += n
            continue
        c["started"] += 1
        if st == "ok":
            c["ok"] += 1
        elif st in CENSORED:
            c["censored"] += 1
        elif st == "lost":
            c["lost"] += 1
        else:
            c["failed"] += 1
    c["uncensored"] = c["planned"] > 0 and c["ok"] == c["planned"]
    c["completion"] = c["ok"] / c["planned"] if c["planned"] else None
    c["completion_started"] = c["ok"] / c["started"] if c["started"] else None
    return c


def conditions(runs, metric, bad=None):
    """{config_key: [records]} of the fixed-task runs (cold or warm, priming excluded) of a metric's condition."""
    out = {}
    for r in runs:
        if r.get("task") != metric["task"] or r.get("series") != metric["series"] or r.get("kind") == "boot-cold":
            continue
        if r.get("mode") not in ("cold", "warm") or not r.get("config_key"):
            continue
        if bad and excluded(r, bad):
            continue
        if _mode_match(r, metric["mode"]) and _config_match(r["config_key"], metric["config"]):
            out.setdefault(r["config_key"], []).append(r)
    return out


def values_by_config(runs, benches, metric, bad=None):
    """{config_key: [values]} of a metric: e2e/spread the ok times of uncensored conditions only; completion 1/0 per
    planned run; rate the ok bench rates."""
    out = {}
    kind = metric["kind"]
    if kind in ("e2e", "spread", "completion"):
        for key, recs in conditions(runs, metric, bad).items():
            c = counts(recs)
            if kind == "completion":
                out[key] = [1.0] * c["ok"] + [0.0] * (c["planned"] - c["ok"])
            elif c["uncensored"]:
                out[key] = [r["e2e_s"] for r in recs if r.get("status") == "ok"]
    elif kind == "rate":
        for r in benches:
            if r.get("status") != "ok" or "trials_per_s" not in r or str(r.get("kind", "")).endswith("probe"):
                continue
            if bad and excluded(r, bad):
                continue
            if _point_of(r) == metric["point"] and _config_match(r["config_key"], metric["config"]):
                out.setdefault(r["config_key"], []).append(r["trials_per_s"])
    else:
        raise ValueError(f"unknown metric kind {kind}")
    return {k: v for k, v in out.items() if v}


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
    bad = invalid(label, runs)
    if metric.get("session") is not None:
        runs = [r for r in runs if r["_session"] == metric["session"]]
        benches = [r for r in benches if r["_session"] == metric["session"]]
    by = values_by_config(runs, benches, metric, bad)
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
    Session-level entries carry "session"; a label-wide entry exists only if all its sessions share one machine.
    Every planned fixed-task condition gets a completion entry (also at 0); e2e and spread entries exist only for
    uncensored conditions; excluded paths and sessions (output failures) give no entries."""
    runs, benches = label_records(label)
    bad = invalid(label, runs)
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
            if r.get("task") not in FIXED or r.get("mode") not in ("cold", "warm") or not r.get("config_key"):
                continue
            if excluded(r, bad) or (sess is not None and r["_session"] != sess):
                continue
            mode = _mode(r)
            for md in (("cache-cold", "process-cold") if mode == ANY_COLD else (mode,)):
                for cfg in (r["config_key"], "best-cpu" if r["config_key"].startswith("cpu:") else "best-gpu"):
                    for kind in ("e2e", "completion", "spread"):
                        m = {"kind": kind, "task": r["task"], "series": r["series"], "mode": md, "config": cfg}
                        add({**m, "session": sess} if sess is not None else m)
        for r in benches:
            if r.get("status") != "ok" or str(r.get("kind", "")).endswith("probe") or "trials_per_s" not in r:
                continue
            if excluded(r, bad) or (sess is not None and r["_session"] != sess):
                continue
            m = {"kind": "rate", "point": _point_of(r), "config": r["config_key"]}
            add({**m, "session": sess} if sess is not None else m)
    return out


def rel(path):
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)

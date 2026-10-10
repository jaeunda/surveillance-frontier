"""Measurement designs shared by the pilot and the formal comparison: steady-state points, the short + sustained
choice among candidates (threads, microbatch), the CPU parity rule, GPU path selection, and drain calibration.
Every choice is made by the registered rule from these measurements and logged with all its inputs."""

import json
import math
import statistics
from pathlib import Path

from . import harness
from .common import FORMAL, ROOT, append_jsonl, geomean, read_json, run_logged
from .tasks import calib_task, point

TC = FORMAL["thread_calibration"]


def meta(task_path):
    t = read_json(task_path)
    return {k: t[k] for k in ("calibration", "tdiag", "tref_cell", "parity_point") if k in t}


def bench_point(task_path, config, seconds, log, *, build, label, sample=False, diag=False, **extra):
    """One bench run, recorded with the point's metadata; returns the record."""
    rec = harness.bench(task_path, config, seconds, build=build, sample=sample, diag=diag)
    rec.update({"kind": label, "meta": meta(task_path), **extra})
    append_jsonl(log, rec)
    return rec


def median_rate(records):
    ok = [r["trials_per_s"] for r in records if r.get("status") == "ok"]
    return statistics.median(ok) if len(ok) == len(records) and ok else None


def choose(series, grid, candidates, log, *, build, label):
    """Short + sustained design for one (series, grid) among candidate configurations.

    Short: q = 16, L in {2, 32, 256}, 3 x 5 s each; the candidate with the highest geometric mean of the three median
    throughputs wins. Sustained: the two best once for 60 s at L = 32; if their order disagrees with the short one,
    the sustained winner is used and the flip recorded. A candidate with any failed run is out (recorded).
    Returns the decision record; its "choice" is the candidate's config_key.
    """
    short, sus_spec = TC["short"], TC["sustained"]
    scores, failed = {}, {}
    by_key = {harness.config_key(c): c for c in candidates}
    for key, cfg in by_key.items():
        meds = []
        for L in short["lengths"]:
            recs = [bench_point(calib_task(series, grid, L), cfg, short["seconds"], log, build=build,
                                label=f"{label}-short", rep=rep, series=series, grid=grid, L=L)
                    for rep in range(short["repetitions"])]
            meds.append(median_rate(recs))
        if any(m is None for m in meds):
            failed[key] = "a short run failed"
            continue
        scores[key] = geomean(meds)
    if not scores:
        rec = {"kind": f"{label}-decision", "series": series, "grid": grid, "choice": None, "failed": failed}
        append_jsonl(log, rec)
        return rec
    ranked = sorted(scores, key=lambda k: -scores[k])
    sustained = {}
    for key in ranked[:2]:
        r = bench_point(calib_task(series, grid, sus_spec["length"]), by_key[key], sus_spec["seconds"], log,
                        build=build, label=f"{label}-sustained", series=series, grid=grid, L=sus_spec["length"])
        if r.get("status") == "ok":
            sustained[key] = r["trials_per_s"]
    sus_pick = max(sustained, key=sustained.get) if sustained else ranked[0]
    flip = sus_pick != ranked[0]
    rec = {"kind": f"{label}-decision", "series": series, "grid": grid, "short_geomean": scores,
           "short_choice": ranked[0], "sustained": sustained, "sustained_choice": sus_pick, "flip": flip,
           "choice": sus_pick if flip else ranked[0], "config": by_key[sus_pick if flip else ranked[0]],
           "failed": failed}
    append_jsonl(log, rec)
    return rec


PARITY_POINTS = ["week_P24_L32", "week_P1_L2", "quarter_P24_L256"]


def parity(cpu_cfg, log, md_path, *, build, min_gain):
    """CPU parity rule: PC1..PC6 in order; a candidate is adopted if the CPU gate passes with it and the geometric
    mean of the three points' median throughput (3 x 5 s each) rises by >= min_gain. Returns the adopted set."""
    adopted = []
    lines = ["# CPU parity log", "", "| Candidate | Gate | Geo-mean gain | Decision | Reason |", "|---|---|---|---|---|"]

    def throughput(pcs):
        cfg = {**cpu_cfg, "parity": ",".join(pcs) or "none"}
        meds = []
        for pt in PARITY_POINTS:
            recs = [bench_point(point(f"parity/{pt}"), cfg, 5, log, build=build, label="parity", rep=k)
                    for k in range(3)]
            meds.append(median_rate(recs))
        return geomean(meds) if all(meds) else None

    for pc in ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6"]:
        if pc == "PC5":
            lines.append("| PC5 | - | - | in the baseline | one Evaluator per worker reuses its scratch across trials "
                         "and cells; there is no variant without it |")
            continue
        trial = adopted + [pc]
        gate = run_logged([Path(build) / "sf_gate", "--device", "cpu", "--threads", cpu_cfg["threads"], "--parity",
                           ",".join(trial), "--items", "1,5,7", "--out", Path(log).with_suffix(f".gate_{pc}.json")],
                          Path(log).with_suffix(".gate.log")).returncode == 0
        before, after = throughput(adopted), throughput(trial)
        gain = after / before - 1 if before and after else None
        adopt = bool(gate and gain is not None and gain >= min_gain)
        reason = ("gate failed" if not gate else "a throughput run failed" if gain is None else
                  f"gain {gain:+.2%} {'>=' if adopt else '<'} {min_gain:.0%}")
        append_jsonl(log, {"kind": "parity-decision", "candidate": pc, "gate": gate, "before": before,
                           "after": after, "gain": gain, "adopt": adopt})
        lines.append(f"| {pc} | {'pass' if gate else 'FAIL'} | {'-' if gain is None else f'{gain:+.2%}'} | "
                     f"{'adopted' if adopt else 'not adopted'} | {reason} |")
        if adopt:
            adopted = trial
    Path(md_path).write_text("\n".join(lines) + f"\n\nAdopted set: {','.join(adopted) or 'none'}\n")
    return ",".join(adopted) or "none"


def select_path(gpu_cfgs, log, *, build, sample=True):
    """GPU path selection at the representative point: higher steady-state rate (each path at its best batch);
    within the tie band, G-inc."""
    ps = FORMAL["path_selection"]
    rates = {}
    for path, cfg in gpu_cfgs.items():
        recs = [bench_point(point(ps["point"]), cfg, 5, log, build=build, label="path", rep=k, sample=sample)
                for k in range(3)]
        rates[path] = median_rate(recs)
    ok = {p: r for p, r in rates.items() if r}
    if not ok:
        choice = None
    elif len(ok) == 2 and abs(ok["inc"] / ok["hyb"] - 1) <= ps["tie_band"]:
        choice = ps["tie_choice"]
    else:
        choice = max(ok, key=ok.get)
    rec = {"kind": "path-decision", "rates": rates, "choice": choice, "tie_band": ps["tie_band"]}
    append_jsonl(log, rec)
    return rec


def drain(task, config, log, workdir, *, build):
    """Drain margin: 1.5 x the largest (exit - stop) over 20 deadline runs (10 at B = 10 s, 10 at 300 s, stream
    purpose calibration), rounded up to 10 ms."""
    worst, runs = 0.0, []
    for B in (10, 300):
        for k in range(10):
            out = Path(workdir) / f"drain_B{B}_{k}"
            rec = harness.run_cold(task, out, config, {"kind": "drain", "budget_s": B, "rep": k}, build=build,
                                   budget_s=B, drain_ms=0, replicate=k + 1, drop=False)
            try:
                stop_ns = json.loads((out / "prefix.json").read_text())["stop_ns"]
                gap = (rec["t1_ns"] - stop_ns) / 1e6
            except (OSError, ValueError, KeyError):
                gap = None
            runs.append({"B": B, "rep": k, "exit_minus_stop_ms": gap, "status": rec["status"]})
            append_jsonl(log, {"kind": "drain", **runs[-1], "run": rec})
            if gap is not None:
                worst = max(worst, gap)
    complete = all(r["exit_minus_stop_ms"] is not None for r in runs)
    return {"device": config["device"], "max_exit_minus_stop_ms": worst, "complete": complete,
            "drain_ms": math.ceil(1.5 * worst / 10) * 10 if complete else None, "runs": runs}


PERF_EVENTS = ["cycles", "instructions", "LLC-loads", "LLC-load-misses", "stalled-cycles-frontend",
               "stalled-cycles-backend"]


def profile(config, log, *, build):
    """Exploratory hardware counters (never timed): canonical point and the three calibration lengths."""
    import subprocess
    points = [point("tdiag/count_24_n604800")] + [calib_task("1s-week", "P24", L) for L in TC["short"]["lengths"]]
    out = []
    for task in points:
        prefix, env = harness.launcher(config)
        cmd = ["perf", "stat", "-x", ",", "-e", ",".join(PERF_EVENTS), *prefix, str(Path(build) / "sf_run"),
               str(task), "--bench", "5", *harness.sf_args(config)]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=env)
        except OSError as e:
            out.append({"task": task.name, "perf": f"unavailable: {e}"})
            continue
        counters = {}
        for line in r.stderr.splitlines():
            f = line.split(",")
            if len(f) > 2 and f[2] in PERF_EVENTS:
                counters[f[2]] = None if f[0].startswith("<not") else float(f[0])
        rec = {"kind": "profile", "task": task.name, "returncode": r.returncode, "counters": counters,
               "unavailable": [e for e in PERF_EVENTS if counters.get(e) is None]}
        append_jsonl(log, rec)
        out.append(rec)
    return out

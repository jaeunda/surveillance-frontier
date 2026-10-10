"""Formal verdicts and summaries from stored records only (Part III, "Verdict rules"; SPEC §10). Nothing is
re-measured.

    analyze.py stage-a LABEL                       R1, R2, sigma and m (formal CPU launches only)
    analyze.py stage-d LABEL [--skip-identity]     C, H3, H4, D1, cost
    analyze.py cross-env LABEL --pilot PILOT       D3 (exploratory): per-point GPU rate ratio pilot / formal machine
    analyze.py spend LABEL [--verification L ...]  instance-hours x official price against the scaling budget

LABEL is results/formal/<LABEL>/. Writes verdicts_<what>.json and tables_<what>.md there.

Rules (fixed before any measurement):
  launch value   median of the repetitions of a condition in one launch; a condition counts only if every repetition
                 is valid (ok, or timeout = censored at exactly the timeout) and every ok output has the reference hash
  pairs          launch-i-cpu with launch-i-gpu, both complete; fewer than m complete pairs -> "incomplete"
  interval       rho = exp(mean l), l_i = log(T_cpu,i / T_gpu,i), 90% paired t; categories with Delta = 1.25
  censoring      a censored run cannot give its own device an advantage, nor equivalence (p2.stats.category)
  H4             per run MSE over the 360 rates (N_c = 0 -> 1/2; late or invalid -> all N_c = 0); RMSE per device and
                 pair over its seeds; eps = RMSE_cpu / RMSE_gpu with the same interval and categories
  R2             per launch and task x series: |median warm E2E / projection - 1| <= 0.15; projection = sum of
                 n / r over cells with this launch's rates at every task length (base scenario, no setup)
  D1             NNLS of t = b0 + b1 C + b2 Cw + b3 W + b4 P (+ b5 / b on the GPU) on rows scaled by 1/t_obs;
                 fit on the calibration split, judged on the 13 held-out points (median <= 0.20, max <= 0.50)
"""

import argparse
import io
import json
import math
import statistics
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stream  # noqa: E402
from p2 import formal, projection, records  # noqa: E402
from p2.common import FORMAL, PROTOCOL, RESULTS, read_json, read_jsonl, write_json  # noqa: E402
from p2.stats import category, paired_interval  # noqa: E402

V = FORMAL["verdicts"]
LADDER = PROTOCOL["window_ladder"]
FIXED = [(t, s) for t in ("T1-curve", "T2-policy-map") for s in PROTOCOL["series"]]


# ---- reading ----

def launches(fd, role):
    return {int(p.name.split("-")[1]): p for p in sorted(Path(fd).glob(f"launch-*-{role}"))}


def complete(d):
    env = read_json(Path(d) / "env.json")
    return env.get("status") == "complete" and env.get("identity", {}).get("match") is True, env.get("status")


class Outputs:
    """Files of a launch's outputs, from the directory or from outputs_bits.tar.gz."""

    def __init__(self, d):
        self.d = Path(d)
        tar = self.d / "outputs_bits.tar.gz"
        self.tar = tarfile.open(tar) if tar.exists() else None
        self.members = {m.name: m for m in self.tar.getmembers()} if self.tar else {}

    def read(self, rec, name):
        local = self.d / "outputs" / Path(rec["out"]).name / name
        if local.exists():
            return local.read_bytes()
        key = f"outputs/{Path(rec['out']).name}/{name}"
        if key in self.members:
            return self.tar.extractfile(self.members[key]).read()
        raise FileNotFoundError(key)


def condition(rec):
    if rec.get("mode") == "deadline":
        return f"T-ref-est/B{rec['budget_s']}"
    return f"{rec['task']}/{rec['series']}/{'warm' if rec['mode'] == 'warm' else 'cold'}"


# ---- Stage A ----

def recount(d, rec):
    """k per (cell, policy) from the stored per-trial bits equals cells.csv."""
    outs = Outputs(d)
    cells = pd.read_csv(io.BytesIO(outs.read(rec, "cells.csv")))
    bits = np.unpackbits(np.frombuffer(outs.read(rec, "hits.bin"), np.uint8), bitorder="little")
    task = read_json(HERE / "tasks" / "main" / f"{rec['task']}_{rec['series']}.json")
    P, n, C = cells.policy.nunique(), int(task["trials"]["per_cell"]), len(task["cells"])
    h = bits[: C * n * P].reshape(C, n, P).sum(axis=1)
    want = cells.sort_values(["cell", "policy"]).k.to_numpy().reshape(C, P)
    return bool(np.array_equal(h, want))


def stage_a(fd):
    out = {"launches": {}, "R2": {"rows": []}}
    r1_ok, t2_week = True, {}
    cpu = launches(fd, "cpu")
    for i, d in cpu.items():
        ok, status = complete(d)
        runs = [r for r in read_jsonl(d / "runs.jsonl") if r.get("task") in ("T1-curve", "T2-policy-map")]
        problems = [] if ok else [f"launch not complete ({status})"]
        for r in runs:
            good = r.get("status") == "ok" and (r.get("runner") or {}).get("width_ok") and r.get("spot_mismatches") == 0 \
                and (r["mode"] == "warm" or r.get("cold_kind") == "cache-cold")
            try:
                same = recount(d, r) if r.get("status") == "ok" else False
            except (OSError, KeyError, ValueError) as e:
                same = f"unreadable: {e}"
            if not (good and same is True):
                problems.append({k: r.get(k) for k in ("task", "series", "mode", "rep", "status", "cold_kind",
                                                       "spot_mismatches")} | {"recount": same})
        seen = {(r["task"], r["series"], "warm" if r["mode"] == "warm" else "cold") for r in runs}
        want = {(t, s, m) for t, s in FIXED for m in ("cold", "warm")}
        if seen != want:
            problems.append(f"missing conditions: {sorted(want - seen)}")
        r1_ok &= not problems
        out["launches"][i] = {"status": status, "runs": len(runs), "problems": problems}
        benches = [r for r in read_jsonl(d / "rates.jsonl") if r.get("kind") == "rate"]
        rates, _ = projection.rate_table(benches)
        for t, s in FIXED:
            warm = [r["e2e_s"] for r in runs if (r["task"], r["series"], r["mode"]) == (t, s, "warm") and r["status"] == "ok"]
            p = projection.project_fixed(t, s, rates, warm=True)
            base = p["seconds"]["base"] if p["status"] == "ok" else None
            ratio = statistics.median(warm) / base if warm and base else None
            out["R2"]["rows"].append({"launch": i, "task": t, "series": s, "median_warm_s": statistics.median(warm) if warm else None,
                                      "projection_s": base, "rate_flags": p.get("rate_flags"), "ratio": ratio,
                                      "pass": ratio is not None and abs(ratio - 1) <= V["r2_tolerance"]})
        cold = [r["e2e_s"] for r in runs if (r["task"], r["series"], r["mode"]) == ("T2-policy-map", "1s-week", "cold")
                and r["status"] == "ok"]
        if ok and cold:
            t2_week[i] = statistics.median(cold)
    n_ok = sum(1 for d in cpu.values() if complete(d)[0])
    out["R1"] = {"status": "judged", "pass": r1_ok} if n_ok >= 2 else {"status": "incomplete", "pass": None,
                                                                         "reason": f"{n_ok} of 2 complete launches"}
    rows = out["R2"]["rows"]
    out["R2"].update({"status": "judged" if rows and all(r["ratio"] is not None for r in rows) else "incomplete",
                      "pass": bool(rows) and all(r["pass"] for r in rows)})
    out["sigma"], out["m"] = None, None
    if len(t2_week) >= 2:
        sigma = float(np.std(np.log(list(t2_week.values())), ddof=1))
        out["sigma"] = sigma
        out["m"] = V["launch_pairs"] if sigma <= V["sigma_threshold"] else V["launch_pairs_if_sigma_high"]
    return out


# ---- Stage D ----

def launch_values(d):
    by = {}
    for r in read_jsonl(Path(d) / "runs.jsonl"):
        if r.get("mode") in ("cold", "warm") and r.get("task") in ("T1-curve", "T2-policy-map", "T-ref") \
                and r.get("kind") != "boot-cold":
            by.setdefault(condition(r), []).append(r)
    out = {}
    for key, runs in by.items():
        valid = all(r["status"] in ("ok", "timeout") for r in runs) and \
            all(r.get("identity_ok", False) for r in runs if r["status"] == "ok") and \
            all(r.get("cold_kind") == "cache-cold" for r in runs if r["mode"] == "cold")
        out[key] = {"value": statistics.median(r["e2e_s"] for r in runs) if valid else None, "valid": valid,
                    "censored": any(r["status"] == "timeout" for r in runs), "statuses": [r["status"] for r in runs]}
    return out


def pairs(fd):
    cpu, gpu = launches(fd, "cpu"), launches(fd, "gpu")
    return [i for i in sorted(cpu) if i in gpu and complete(cpu[i])[0] and complete(gpu[i])[0]], cpu, gpu


def h3(fd, m):
    ok_pairs, cpu, gpu = pairs(fd)
    vals = {i: (launch_values(cpu[i]), launch_values(gpu[i])) for i in ok_pairs}
    rows = []
    for role, conds in (("primary", V["h3_primary"]), ("secondary", V["h3_secondary"])):
        for task, series, mode in conds:
            key = f"{task}/{series}/{mode}"
            ratios, used, cc, cg = [], [], False, False
            for i in ok_pairs:
                c, g = vals[i][0].get(key), vals[i][1].get(key)
                if c and g and c["valid"] and g["valid"]:
                    ratios.append(c["value"] / g["value"])
                    used.append(i)
                    cc |= c["censored"]
                    cg |= g["censored"]
            row = {"condition": key, "role": role, "pairs": used, "per_pair_ratio": ratios}
            if len(ratios) < m:
                row.update({"status": "incomplete", "category": None, "reason": f"{len(ratios)} of {m} complete pairs"})
            else:
                rho, lo, hi = paired_interval(ratios[:m])
                cat, note = category(lo, hi, cc, cg)
                row.update({"status": "judged", "rho": rho, "lo": lo, "hi": hi, "category": cat, "note": note,
                            "censored_cpu": cc, "censored_gpu": cg})
            rows.append(row)
    return rows


def identity_check(prefix, bits, task, P=24):
    """(N_c, k_c) of a live run against the reference bits of the same trial ids (stream replay)."""
    n = PROTOCOL["series"]["1s-week"]["bars"]
    offset = 0
    for c, cell in enumerate(task["cells"]):
        M = n - cell["L"] + 1
        info = prefix["cells"][c]
        if info["N"]:
            starts = stream.starts_np(task["task_code"], task["series"]["code"], cell["stream_cell"], prefix["replicate"],
                                      stream.PURPOSES["measured"], np.arange(info["N"]), M)
            k = bits[(offset + starts)[:, None] * P + np.arange(P)[None, :]].sum(axis=0)
            if not np.array_equal(k, np.asarray(info["k"])):
                return False
        offset += M
    return True


def h4(fd, m, check_identity=True):
    ok_pairs, cpu, gpu = pairs(fd)
    ref = Path(fd) / "reference" / "T-ref_1s-week"
    p_exact = pd.read_csv(ref / "rates.csv").sort_values(["cell", "policy"]).rate.to_numpy().reshape(15, -1)
    bits = None
    if check_identity and (ref / "ref_bits.bin").exists():
        bits = np.unpackbits(np.frombuffer((ref / "ref_bits.bin").read_bytes(), np.uint8), bitorder="little")
    task = read_json(HERE / "tasks" / "main" / "T-ref-est_1s-week.json")
    lo_t, hi_t = PROTOCOL["tasks"]["T-ref-est"]["transition_set"]
    transition = (p_exact >= lo_t) & (p_exact <= hi_t)
    rows, id_fail = [], []
    for B in PROTOCOL["tasks"]["T-ref-est"]["budgets_s"]:
        per = {}
        for i in ok_pairs:
            for dev, d in (("cpu", cpu[i]), ("gpu", gpu[i])):
                mses, tr, maxabs, late, ckpt = [], [], [], 0, []
                outs = Outputs(d)
                for r in read_jsonl(d / "runs.jsonl"):
                    if r.get("mode") != "deadline" or r.get("budget_s") != B:
                        continue
                    est = np.full_like(p_exact, 0.5)
                    prefix = None
                    if r["status"] == "ok":
                        try:
                            prefix = json.loads(outs.read(r, "prefix.json"))
                        except (OSError, KeyError, ValueError):
                            prefix = None
                    if prefix is None:
                        late += 1
                        ckpt.append(0)
                    else:
                        if bits is not None and not identity_check(prefix, bits, task):
                            id_fail.append({"pair": i, "device": dev, "budget": B, "replicate": r["replicate"]})
                        for c, info in enumerate(prefix["cells"]):
                            if info["N"]:
                                est[c] = np.asarray(info["k"]) / info["N"]
                        ckpt.append(prefix.get("checkpoint", 0))
                    err = est - p_exact
                    mses.append(float(np.mean(err ** 2)))
                    tr.append(float(np.mean(err[transition] ** 2)) if transition.any() else None)
                    maxabs.append(float(np.max(np.abs(err))))
                per[(i, dev)] = {"rmse": math.sqrt(np.mean(mses)) if mses else None, "seeds": len(mses),
                                 "late_or_invalid": late, "max_abs_error": max(maxabs) if maxabs else None,
                                 "rmse_transition": math.sqrt(np.mean(tr)) if tr and None not in tr else "undefined",
                                 "checkpoint_median": statistics.median(ckpt) if ckpt else None}
        eps, used, both_zero, one_zero = [], [], [], []
        for i in ok_pairs:
            c, g = per.get((i, "cpu")), per.get((i, "gpu"))
            if not c or not g or c["rmse"] is None or g["rmse"] is None:
                continue
            used.append(i)
            if c["rmse"] == 0 and g["rmse"] == 0:
                both_zero.append(i)
            elif c["rmse"] == 0 or g["rmse"] == 0:
                one_zero.append(i)
            else:
                eps.append(c["rmse"] / g["rmse"])
        row = {"budget_s": B, "pairs": used, "per_pair_eps": eps, "per_device": {f"{i}-{dv}": v for (i, dv), v in per.items()}}
        if len(used) < m:
            row.update({"status": "incomplete", "category": None, "reason": f"{len(used)} of {m} complete pairs"})
        elif len(both_zero) == len(used):
            row.update({"status": "judged", "category": "equal (both exact)"})
        elif one_zero:
            row.update({"status": "judged", "category": None, "note": f"RMSE 0 in pairs {one_zero}: no interval"})
        else:
            e, lo, hi = paired_interval(eps[:m])
            cat, note = category(lo, hi)
            row.update({"status": "judged", "eps": e, "lo": lo, "hi": hi, "category": cat, "note": note})
        rows.append(row)
    return rows, id_fail


def model_features(policies, L, n, batch=None):
    s_floor = min(p["s_min"] for p in policies)
    kmax = max(p["K"] for p in policies)
    C = Cw = W = 0.0
    for w in LADDER[:kmax]:
        cap = math.floor((n - w + 1) * 10 ** (-s_floor))
        C += cap + 1
        Cw += (cap + 1) * max(1, w // 2)
        W += L + w - 1
    row = [1.0, C, Cw, W, float(len(policies))]
    if batch is not None:
        row.append(1.0 / batch)
    return row


def d1_fit(rows, gpu):
    """rows: [{name, n, batch, rate, L, policies}] -> fit on the calibration split, errors on the held-out points."""
    if not rows:
        return {"status": "not measured"}
    df = pd.DataFrame(rows)
    agg = df.groupby(["name", "batch"], dropna=False).agg(rate=("rate", "median"), n=("n", "first"),
                                                          L=("L", "first"), policies=("policies", "first")).reset_index()
    agg["t_obs"] = 1.0 / agg.rate
    X = np.array([model_features(r.policies, r.L, r.n, r.batch if gpu else None) for r in agg.itertuples()])
    held = (agg.n == PROTOCOL["t_diag"]["heldout_n"][0]).to_numpy()
    A = X[~held] / agg.t_obs[~held].to_numpy()[:, None]
    beta, _ = nnls(A, np.ones((~held).sum()))
    pred = X @ beta
    rel = (pred - agg.t_obs.to_numpy()) / agg.t_obs.to_numpy()
    err = np.abs(rel[held])
    crit = PROTOCOL["t_diag"]["criterion"]
    full = len(err) == PROTOCOL["t_diag"]["heldout_points"]
    return {"status": "judged" if full else "incomplete", "beta": beta.tolist(), "heldout_points": int(held.sum()),
            "median_abs_rel_err": float(np.median(err)) if len(err) else None,
            "max_abs_rel_err": float(np.max(err)) if len(err) else None,
            "pass": bool(full and np.median(err) <= crit["median_abs_rel_err"] and np.max(err) <= crit["max_abs_rel_err"]),
            "points": [{"name": r.name, "batch": None if pd.isna(r.batch) else r.batch, "t_obs": r.t_obs, "t_pred": float(p),
                        "rel_err": float(e), "heldout": bool(h)} for r, p, e, h in zip(agg.itertuples(), pred, rel, held)]}


def d1(fd, role):
    rows = []
    for d in launches(fd, role).values():
        for r in read_jsonl(d / "tdiag.jsonl"):
            if r.get("status") != "ok":
                continue
            task = read_json(HERE / "tasks" / "tdiag" / Path(r["task"]).name)
            rows.append({"name": Path(r["task"]).stem, "n": task["tdiag"]["n"], "rate": r["trials_per_s"],
                         "batch": r["device"].get("microbatch") if role == "gpu" else None,
                         "L": task["cells"][0]["L"], "policies": task["policies"]})
    return d1_fit(rows, role == "gpu")


def costs(fd):
    out = []
    for d in sorted(Path(fd).glob("launch-*")) + sorted(Path(fd).glob("size-control")):
        env = read_json(d / "env.json")
        price = (env.get("on_demand_price") or {}).get("price_usd_per_hour")
        ec2 = read_json(d / "ec2.json") if (d / "ec2.json").exists() else {}
        hours = ec2.get("instance_hours")
        per_task = {}
        for r in read_jsonl(d / "runs.jsonl"):
            if r.get("mode") in ("cold", "warm") and r.get("status") in ("ok", "timeout") and price:
                per_task.setdefault(condition(r), []).append(r["e2e_s"] / 3600 * price)
        out.append({"launch": d.name, "instance_type": env.get("ec2", {}).get("instance_type"),
                    "purchase": env.get("ec2", {}).get("purchase_option"), "on_demand_usd_per_h": price,
                    "instance_hours": hours, "compute_usd_on_demand": price * hours if price and hours else None,
                    "spot_price_at_launch": (ec2.get("spot_price_at_launch") or {}).get("price_usd_per_hour"),
                    "median_task_usd": {k: statistics.median(v) for k, v in per_task.items()}})
    return out


def stage_d(fd, check_identity):
    freeze = read_json(Path(fd) / "protocol-freeze.json")
    m = freeze["m"]
    d0 = Path(fd) / "d0" / "gate_cuda.json"
    c_pass = d0.exists() and read_json(d0).get("pass") is True
    out = {"m": m, "machines": freeze["machines"], "gpu_status": freeze["gpu"]["status"],
           "C": {"status": "judged" if d0.exists() else "not measured", "pass": c_pass}}
    if not c_pass:
        out["H3"] = out["H4"] = {"status": "not measured", "reason": "gate C (D0) not passed or not run"}
    else:
        out["H3"] = h3(fd, m)
        out["H4"], out["H4_identity_failures"] = h4(fd, m, check_identity)
    out["D1"] = {"cpu": d1(fd, "cpu"), "gpu": d1(fd, "gpu") if c_pass else {"status": "not measured"}}
    out["cost"] = costs(fd)
    return out


# ---- exploratory ----

def cross_env(fd, pilot):
    """D3: GPU rates of the same calibration-split points on the pilot machine and the formal GPU (pair 1)."""
    _, pilot_b = records.label_records(pilot)
    best = {}
    for r in pilot_b:
        if r.get("kind") == "diag" and r.get("status") == "ok" and not r["config_key"].startswith("cpu:"):
            best.setdefault(Path(r["task"]).stem, []).append(r["trials_per_s"])
    formal_rates = {}
    for d in launches(fd, "gpu").values():
        for r in read_jsonl(d / "tdiag.jsonl"):
            if r.get("status") == "ok":
                formal_rates.setdefault(Path(r["task"]).stem, []).append(r["trials_per_s"])
    rows = [{"point": k, "pilot_rate": statistics.median(v), "formal_rate": statistics.median(formal_rates[k]),
             "ratio_formal_over_pilot": statistics.median(formal_rates[k]) / statistics.median(v)}
            for k, v in sorted(best.items()) if k in formal_rates]
    ratios = [r["ratio_formal_over_pilot"] for r in rows]
    return {"status": "exploratory", "points": rows,
            "ratio_summary": {"median": statistics.median(ratios), "min": min(ratios), "max": max(ratios)} if ratios else None,
            "note": "consistency across environments; the pilot data were seen before the formal design choices, so this "
                    "is not a held-out prediction test"}


def spend(fd, verification):
    rows = costs(fd)
    for label in verification:
        for ec2 in sorted((RESULTS / "pilot" / label).glob("session-*/ec2.json")):
            e = read_json(ec2)
            env = read_json(ec2.parent / "env.json")
            price = (env.get("on_demand_price") or {}).get("price_usd_per_hour")
            hours = e.get("instance_hours")
            rows.append({"launch": f"{label}/{ec2.parent.name}", "instance_type": e.get("instance_type"),
                         "on_demand_usd_per_h": price, "instance_hours": hours,
                         "compute_usd_on_demand": price * hours if price and hours else None})
    dec = read_json(formal.DECISION)
    total = sum(r["compute_usd_on_demand"] or 0 for r in rows)
    missing = [r["launch"] for r in rows if r["compute_usd_on_demand"] is None]
    return {"rows": rows, "total_usd_on_demand": total, "rows_without_cost": missing,
            "cap_usd": dec["budget"]["cap_usd"], "spent_before_usd": dec["budget"]["spent_usd"],
            "planned_usd": dec["budget"]["planned_usd"],
            "remaining_usd": dec["budget"]["cap_usd"] - dec["budget"]["spent_usd"] - total}


# ---- tables ----

def tables(v, what):
    f = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
    lines = [f"# Phase 2 formal results ({what})", ""]
    if what == "stage-a":
        lines += [f"- R1: {v['R1']}", f"- R2: {v['R2']['status']}, pass = {v['R2']['pass']}",
                  f"- sigma (T2 week cold, log launch medians) = {v['sigma']}, m = {v['m']}", "",
                  "| launch | task | series | median warm s | projection s | ratio | pass |", "|---|---|---|---|---|---|---|"]
        lines += [f"| {r['launch']} | {r['task']} | {r['series']} | {f(r['median_warm_s'])} | {f(r['projection_s'])} | "
                  f"{f(r['ratio'])} | {r['pass']} |" for r in v["R2"]["rows"]]
    elif what == "stage-d":
        lines += [f"m = {v['m']}; machines {v['machines']}; GPU build {v['gpu_status']}; C: {v['C']}", "", "## H3", ""]
        if isinstance(v["H3"], list):
            lines += ["| condition | role | per-pair ratio | rho [90% CI] | category | note |", "|---|---|---|---|---|---|"]
            for r in v["H3"]:
                ci = f"{r['rho']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]" if r.get("rho") else r.get("reason", "")
                lines.append(f"| {r['condition']} | {r['role']} | {', '.join(f'{x:.3f}' for x in r['per_pair_ratio'])} | "
                             f"{ci} | {r['category'] or r['status']} | {r.get('note') or ''} |")
        else:
            lines.append(str(v["H3"]))
        lines += ["", "## H4", ""]
        if isinstance(v["H4"], list):
            lines += ["| budget s | per-pair eps | eps [90% CI] | category | note |", "|---|---|---|---|---|"]
            for r in v["H4"]:
                ci = f"{r['eps']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]" if r.get("eps") else r.get("reason", "")
                lines.append(f"| {r['budget_s']} | {', '.join(f'{x:.3f}' for x in r['per_pair_eps'])} | {ci} | "
                             f"{r['category'] or r['status']} | {r.get('note') or ''} |")
        else:
            lines.append(str(v["H4"]))
        lines += ["", "## D1", ""] + [f"- {k}: {d.get('status')}, median |rel err| {d.get('median_abs_rel_err')}, "
                                      f"max {d.get('max_abs_rel_err')}, pass {d.get('pass')}" for k, d in v["D1"].items()]
    else:
        lines.append("```json\n" + json.dumps(v, indent=1, default=str)[:20000] + "\n```")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["stage-a", "stage-d", "cross-env", "spend"])
    ap.add_argument("label")
    ap.add_argument("--skip-identity", action="store_true")
    ap.add_argument("--pilot")
    ap.add_argument("--verification", nargs="*", default=[])
    a = ap.parse_args()
    fd = formal.formal_dir(a.label)
    if a.what == "stage-a":
        v = stage_a(fd)
    elif a.what == "stage-d":
        v = stage_d(fd, not a.skip_identity)
    elif a.what == "cross-env":
        if not a.pilot:
            sys.exit("--pilot is required")
        v = cross_env(fd, a.pilot)
    else:
        v = spend(fd, a.verification)
    write_json(fd / f"verdicts_{a.what}.json", v)
    (fd / f"tables_{a.what}.md").write_text(tables(v, a.what))
    print(tables(v, a.what))


if __name__ == "__main__":
    main()

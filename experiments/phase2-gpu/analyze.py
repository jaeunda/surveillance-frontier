"""Formal verdicts and summaries from stored records only (Part III, "Verdict rules"; SPEC §10). Nothing is
re-measured.

    analyze.py stage-a LABEL                       R1, R2, sigma and m (formal Stage A CPU launches only)
    analyze.py stage-d LABEL                       C, H3, H4, D1, cost
    analyze.py cross-env LABEL --pilot PILOT       D3 (exploratory): per-point GPU rate ratio pilot / formal machine
    analyze.py spend LABEL [--verification L ...]  instance-hours x official price against the scaling budget

LABEL is results/formal/<LABEL>/. Writes verdicts_<what>.json and tables_<what>.md there.

Rules (fixed before any measurement; r2.1 amendments marked):
  records        every launch attempt is validated first (p2/validate.py); per slot the first valid attempt is used,
                 an invalid attempt is excluded as a whole and listed with its reasons (r2.1)
  launch value   median of the repetitions of a condition in one launch (timeout = censored at exactly the timeout)
  pairs          pairs 1..m of the freeze; a pair counts only if both launches are valid, on distinct instances, in one
                 region on one UTC day, in the registered order, and every CPU gate has the reference golden hashes
                 (r2.1); fewer than m valid pairs -> "incomplete"
  interval       rho = exp(mean l), l_i = log(T_cpu,i / T_gpu,i), 90% paired t; categories with Delta = 1.25
  censoring      a condition with a censored run gets no category (r2.1, p2.stats.category)
  H4             the reference bits and rates must have the frozen hashes; exact rates are recounted from the bits.
                 Per run MSE over the 360 rates (N_c = 0 -> 1/2; late, timed-out or failed -> all N_c = 0); RMSE per
                 device and pair over exactly its 10 registered seeds; eps = RMSE_cpu / RMSE_gpu with the same interval
                 and categories. Any run whose (G, N_c, k_c) differ from the reference bits for its trial ids violates
                 the identity contract: that budget gets no category ("invalid") (r2.1)
  R2             per launch and task x series: |median warm E2E / projection - 1| <= 0.15; projection = sum of
                 n / r over cells with this launch's rates at every task length (base scenario, no setup)
  D1             NNLS of t = b0 + b1 C + b2 Cw + b3 W + b4 P (+ b5 / b on the GPU) on rows scaled by 1/t_obs;
                 fit on the calibration split, judged on the held-out points whose workload never ran before D1
                 (t_diag.heldout_judged, r2.1; median <= 0.20, max <= 0.50); the other held-out points are reported
                 apart as repeat predictions
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
from p2 import cost, formal, projection, records, validate  # noqa: E402
from p2.common import FORMAL, PROTOCOL, RESULTS, read_json, read_jsonl, sha256_file, write_json  # noqa: E402
from p2.stats import category, paired_interval  # noqa: E402

V = FORMAL["verdicts"]
LADDER = PROTOCOL["window_ladder"]
FIXED = [(t, s) for t in ("T1-curve", "T2-policy-map") for s in PROTOCOL["series"]]
EST = PROTOCOL["tasks"]["T-ref-est"]


# ---- reading ----

def label_of(fd):
    return Path(fd).name


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


def decision_sha():
    return sha256_file(formal.DECISION) if formal.DECISION.exists() else None


def load_freeze(fd):
    fz = Path(fd) / "protocol-freeze.json"
    freeze = read_json(fz)
    freeze["_sha256"] = sha256_file(fz)
    return freeze


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
    label = label_of(fd)
    out = {"launches": {}, "used": {}, "R2": {"rows": []}}
    r1_ok, t2_week = True, {}
    dsha = decision_sha()
    for i in (1, 2):
        pick, report = validate.chosen(formal.slot_dir(label, "A", "cpu", i), "A", "cpu", i, None, dsha)
        out["launches"][i] = {"attempts": report}
        if pick is None:
            r1_ok = False
            continue
        out["used"][i] = {"dir": str(pick)}
        runs = [r for r in read_jsonl(pick / "runs.jsonl") if r.get("task") in ("T1-curve", "T2-policy-map")]
        problems = []
        for r in runs:
            good = r.get("status") == "ok" and (r.get("runner") or {}).get("width_ok") and r.get("spot_mismatches") == 0 \
                and (r["mode"] == "warm" or r.get("cold_kind") == "cache-cold")
            try:
                same = recount(pick, r) if r.get("status") == "ok" else False
            except (OSError, KeyError, ValueError) as e:
                same = f"unreadable: {e}"
            if not (good and same is True):
                problems.append({k: r.get(k) for k in ("task", "series", "mode", "rep", "status", "cold_kind",
                                                       "spot_mismatches")} | {"recount": same})
        seen = {}
        for r in runs:
            k = (r["task"], r["series"], "warm" if r["mode"] == "warm" else "cold")
            seen[k] = seen.get(k, 0) + 1
        reps = V["reps_per_launch"]
        for t, s in FIXED:
            for m, n in (("cold", reps["T1/T2 cold"]), ("warm", reps["T1/T2 warm"])):
                if seen.get((t, s, m), 0) != n:
                    problems.append(f"{t} {s} {m}: {seen.get((t, s, m), 0)} runs, {n} required")
        r1_ok &= not problems
        out["launches"][i]["R1_problems"] = problems
        benches = [r for r in read_jsonl(pick / "rates.jsonl") if r.get("kind") == "rate"]
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
        if cold:
            t2_week[i] = statistics.median(cold)
    dup = validate.distinct_instances([x["dir"] for x in out["used"].values()])
    if dup:
        r1_ok = False
        out["instance_problems"] = f"both Stage A launches on instance {dup}"
    golds = [validate.goldens(Path(x["dir"]) / "gate" / "gate_cpu.json") for x in out["used"].values()
             if (Path(x["dir"]) / "gate" / "gate_cpu.json").exists()]
    if len(golds) == 2 and golds[0] != golds[1]:
        r1_ok = False
        out["golden_problems"] = "the two Stage A CPU gates give different golden hashes"
    n_ok = len(out["used"])
    out["R1"] = {"status": "judged", "pass": r1_ok} if n_ok == 2 else {"status": "incomplete", "pass": None,
                                                                       "reason": f"{n_ok} of 2 valid launches"}
    rows = out["R2"]["rows"]
    out["R2"].update({"status": "judged" if n_ok == 2 and rows and all(r["ratio"] is not None for r in rows)
                      else "incomplete", "pass": bool(rows) and all(r["pass"] for r in rows)})
    out["sigma"], out["m"] = None, None
    if len(t2_week) == 2 and n_ok == 2 and not dup:
        sigma = float(np.std(np.log(list(t2_week.values())), ddof=1))
        out["sigma"] = sigma
        out["m"] = V["launch_pairs"] if sigma <= V["sigma_threshold"] else V["launch_pairs_if_sigma_high"]
    return out


# ---- Stage D ----

def c_status(fd, freeze):
    pr = validate.gate_problems(Path(fd) / "d0" / "gate_cuda.json", device="cuda", full=True,
                                path_name=freeze["gpu"]["path"], code=freeze["code_sha256"])
    return {"status": "judged" if (Path(fd) / "d0" / "gate_cuda.json").exists() else "not measured",
            "pass": not pr, "problems": pr}


def valid_pairs(fd, freeze):
    """{pair: (cpu_dir, gpu_dir)} of the registered pairs that count, and the report of every slot and pair."""
    label, dsha = label_of(fd), decision_sha()
    golden = freeze.get("golden_sha256") or {}
    used, report = {}, {}
    for i in freeze["pairs"]:
        rep = {}
        picks = {}
        for role in ("cpu", "gpu"):
            pick, att = validate.chosen(formal.slot_dir(label, "D", role, i), "D", role, i, freeze, dsha)
            rep[role] = att
            picks[role] = pick
        problems = []
        if picks["cpu"] is None or picks["gpu"] is None:
            problems.append("no valid attempt for " + ", ".join(r for r in picks if picks[r] is None))
        else:
            problems += validate.pair_problems(picks["cpu"], picks["gpu"], i)
            gate = picks["cpu"] / "gate" / "gate_cpu.json"
            other = validate.goldens(gate) if gate.exists() else {}
            if any(other.get(k) != v for k, v in golden.items() if k in other) or not other:
                problems.append("CPU gate golden hashes differ from the reference (or no CPU gate)")
        rep["pair_problems"] = problems
        report[i] = rep
        if not problems:
            used[i] = (picks["cpu"], picks["gpu"])
    dup = validate.distinct_instances([d for pair in used.values() for d in pair])
    for i in list(used):
        if any((read_json(Path(d) / "env.json").get("ec2") or {}).get("instance_id") in dup for d in used[i]):
            report[i]["pair_problems"].append(f"instance reused across launches: {dup}")
            del used[i]
    return used, report


def launch_values(d):
    by = {}
    for r in read_jsonl(Path(d) / "runs.jsonl"):
        if r.get("mode") in ("cold", "warm") and r.get("task") in ("T1-curve", "T2-policy-map", "T-ref") \
                and r.get("kind") != "boot-cold":
            by.setdefault(condition(r), []).append(r)
    return {key: {"value": statistics.median(r["e2e_s"] for r in runs),
                  "censored": any(r["status"] == "timeout" for r in runs), "n": len(runs)} for key, runs in by.items()}


def h3(used, m):
    vals = {i: (launch_values(c), launch_values(g)) for i, (c, g) in used.items()}
    rows = []
    for role, conds in (("primary", V["h3_primary"]), ("secondary", V["h3_secondary"])):
        for task, series, mode in conds:
            key = f"{task}/{series}/{mode}"
            ratios, pairs_used, cc, cg = [], [], False, False
            for i in sorted(used):
                c, g = vals[i][0].get(key), vals[i][1].get(key)
                if c and g:
                    ratios.append(c["value"] / g["value"])
                    pairs_used.append(i)
                    cc |= c["censored"]
                    cg |= g["censored"]
            row = {"condition": key, "role": role, "pairs": pairs_used, "per_pair_ratio": ratios}
            if len(ratios) < m:
                row.update({"status": "incomplete", "category": None, "reason": f"{len(ratios)} of {m} valid pairs"})
            else:
                rho, lo, hi = paired_interval(ratios)
                cat, note = category(lo, hi, cc, cg)
                row.update({"status": "judged", "rho": rho, "lo": lo, "hi": hi, "category": cat, "note": note,
                            "censored_cpu": cc, "censored_gpu": cg})
            rows.append(row)
    return rows


def identity_check(prefix, bits, task, replicate, P=24):
    """Problems of one live deadline run against the reference bits of the same trial ids (empty list = identical):
    the replicate; the round-robin structure N_c = G // C (+1 for c < G mod C); the cells; 0 <= k <= N; k = 0 at N = 0;
    and every k_c equal to the reference bits at the starts the stream gives for trials 0..N_c - 1."""
    n = PROTOCOL["series"]["1s-week"]["bars"]
    C = len(task["cells"])
    out = []
    if prefix.get("replicate") != replicate:
        out.append(f"replicate {prefix.get('replicate')}, {replicate} registered")
    cells = prefix.get("cells") or []
    G = prefix.get("G")
    if len(cells) != C or not isinstance(G, int) or G < 0:
        return out + [f"{len(cells)} cells, G = {G}"]
    offset = 0
    for c, cell in enumerate(task["cells"]):
        info = cells[c]
        M = n - cell["L"] + 1
        want_n = G // C + (1 if c < G % C else 0)
        k = np.asarray(info.get("k", []), dtype=np.int64)
        if info.get("N") != want_n:
            out.append(f"cell {c}: N = {info.get('N')}, round-robin prefix gives {want_n}")
        elif info.get("L") != cell["L"] or float(info.get("q")) != float(cell["q"]):
            out.append(f"cell {c}: (L, q) = ({info.get('L')}, {info.get('q')})")
        elif k.shape != (P,) or (k < 0).any() or (k > want_n).any():
            out.append(f"cell {c}: k out of range")
        elif want_n:
            starts = stream.starts_np(task["task_code"], task["series"]["code"], cell["stream_cell"], replicate,
                                      stream.PURPOSES["measured"], np.arange(want_n), M)
            ref = bits[(offset + starts)[:, None] * P + np.arange(P)[None, :]].sum(axis=0)
            if not np.array_equal(ref, k):
                out.append(f"cell {c}: k differs from the reference bits")
        offset += M
    return out


def reference_rates(fd, freeze):
    """(exact rates [15, P] as float, bits) from the reference bits with the frozen hashes, or (None, reason)."""
    ref = Path(fd) / "reference" / "T-ref_1s-week"
    want = freeze.get("reference_files_sha256") or {}
    for name in ("ref_bits.bin", "rates.csv"):
        if not (ref / name).exists():
            return None, f"reference {name} missing"
        if want.get(name) != sha256_file(ref / name):
            return None, f"reference {name} does not have the frozen hash"
    bits = np.unpackbits(np.frombuffer((ref / "ref_bits.bin").read_bytes(), np.uint8), bitorder="little")
    task = read_json(HERE / "tasks" / "main" / "T-ref-est_1s-week.json")
    n, P = PROTOCOL["series"]["1s-week"]["bars"], len(task["policies"])
    rates, offset = np.zeros((len(task["cells"]), P), dtype=np.float64), 0
    for c, cell in enumerate(task["cells"]):
        M = n - cell["L"] + 1
        rates[c] = bits[offset * P:(offset + M) * P].reshape(M, P).sum(axis=0) / M
        offset += M
    csv = pd.read_csv(ref / "rates.csv").sort_values(["cell", "policy"])
    if not np.array_equal(csv.k.to_numpy(np.int64).reshape(rates.shape),
                          np.rint(rates * np.array([n - c["L"] + 1 for c in task["cells"]])[:, None]).astype(np.int64)):
        return None, "rates.csv counts differ from the reference bits"
    return (rates, bits), None


def h4(fd, used, m, freeze):
    loaded, why = reference_rates(fd, freeze)
    if loaded is None:
        return {"status": "not measured", "reason": why}, []
    p_exact, bits = loaded
    task = read_json(HERE / "tasks" / "main" / "T-ref-est_1s-week.json")
    P = len(task["policies"])
    lo_t, hi_t = EST["transition_set"]
    transition = (p_exact >= lo_t) & (p_exact <= hi_t)
    k_seeds = EST["seeds_per_launch_pair"]
    rows, id_fail = [], []
    for B in EST["budgets_s"]:
        per, problems = {}, []
        for i in sorted(used):
            want = set(range(k_seeds * (i - 1) + 1, k_seeds * i + 1))
            for dev, d in zip(("cpu", "gpu"), used[i]):
                runs = [r for r in read_jsonl(Path(d) / "runs.jsonl") if r.get("mode") == "deadline"
                        and r.get("budget_s") == B]
                got = sorted(r.get("replicate") for r in runs)
                if got != sorted(want):
                    problems.append(f"pair {i} {dev}: replicates {got}, {sorted(want)} registered")
                mses, tr, maxabs, late, ckpt = [], [], [], 0, []
                outs = Outputs(d)
                for r in runs:
                    est = np.full(p_exact.shape, EST["estimate_if_n0"], dtype=np.float64)
                    prefix = None
                    if r.get("status") == "ok":
                        try:
                            prefix = json.loads(outs.read(r, "prefix.json"))
                        except (OSError, KeyError, ValueError):
                            prefix = None
                    if prefix is None:  # late, timed out, failed, or no output: N_c = 0 everywhere
                        late += 1
                        ckpt.append(0)
                    else:
                        bad = identity_check(prefix, bits, task, r.get("replicate"), P)
                        if bad:
                            id_fail.append({"pair": i, "device": dev, "budget": B, "replicate": r.get("replicate"),
                                            "problems": bad[:5]})
                        for c, info in enumerate(prefix["cells"]):
                            if info["N"]:
                                est[c] = np.asarray(info["k"], dtype=np.float64) / info["N"]
                        ckpt.append(prefix.get("checkpoint", 0))
                    err = est - p_exact
                    mses.append(float(np.mean(err ** 2)))
                    tr.append(float(np.mean(err[transition] ** 2)) if transition.any() else None)
                    maxabs.append(float(np.max(np.abs(err))))
                per[(i, dev)] = {"rmse": math.sqrt(np.mean(mses)) if mses else None, "seeds": len(mses),
                                 "late_or_invalid": late, "max_abs_error": max(maxabs) if maxabs else None,
                                 "rmse_transition": math.sqrt(np.mean(tr)) if tr and None not in tr else "undefined",
                                 "checkpoint_median": statistics.median(ckpt) if ckpt else None}
        fails = [f for f in id_fail if f["budget"] == B]
        eps, both_zero, one_zero = [], [], []
        for i in sorted(used):
            c, g = per[(i, "cpu")], per[(i, "gpu")]
            if c["rmse"] == 0 and g["rmse"] == 0:
                both_zero.append(i)
            elif c["rmse"] == 0 or g["rmse"] == 0:
                one_zero.append(i)
            else:
                eps.append(c["rmse"] / g["rmse"])
        row = {"budget_s": B, "pairs": sorted(used), "per_pair_eps": eps, "identity_failures": len(fails),
               "seed_problems": problems, "per_device": {f"{i}-{dv}": v for (i, dv), v in per.items()}}
        if fails or problems:
            row.update({"status": "invalid", "category": None,
                        "reason": f"{len(fails)} runs differ from the reference bits; seed problems: {len(problems)}"})
        elif len(used) < m:
            row.update({"status": "incomplete", "category": None, "reason": f"{len(used)} of {m} valid pairs"})
        elif len(both_zero) == len(used):
            row.update({"status": "judged", "category": "equal (both exact)"})
        elif one_zero:
            row.update({"status": "judged", "category": None, "note": f"RMSE 0 in pairs {one_zero}: no interval"})
        else:
            e, lo, hi = paired_interval(eps)
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
    """rows: [{name, n, batch, rate, L, policies}] -> fit on the calibration split; errors on the held-out points,
    judged on t_diag.heldout_judged (workloads never run before D1), the rest reported as repeat predictions."""
    if not rows:
        return {"status": "not measured"}
    td = PROTOCOL["t_diag"]
    judged_names = set(td["heldout_judged"])
    df = pd.DataFrame(rows)
    agg = df.groupby(["name", "batch"], dropna=False).agg(rate=("rate", "median"), n=("n", "first"),
                                                          L=("L", "first"), policies=("policies", "first")).reset_index()
    agg["t_obs"] = 1.0 / agg.rate
    X = np.array([model_features(r.policies, r.L, r.n, r.batch if gpu else None) for r in agg.itertuples()])
    held = (agg.n == td["heldout_n"][0]).to_numpy()
    judged = held & agg.name.isin(judged_names).to_numpy()
    A = X[~held] / agg.t_obs[~held].to_numpy()[:, None]
    beta, _ = nnls(A, np.ones((~held).sum()))
    pred = X @ beta
    rel = (pred - agg.t_obs.to_numpy()) / agg.t_obs.to_numpy()
    err = np.abs(rel[judged])
    seen = np.abs(rel[held & ~judged])
    crit = td["criterion"]
    full = int(judged.sum()) == len(judged_names)
    return {"status": "judged" if full else "incomplete", "beta": beta.tolist(), "heldout_judged": int(judged.sum()),
            "median_abs_rel_err": float(np.median(err)) if len(err) else None,
            "max_abs_rel_err": float(np.max(err)) if len(err) else None,
            "pass": bool(full and np.median(err) <= crit["median_abs_rel_err"] and np.max(err) <= crit["max_abs_rel_err"]),
            "repeat_predictions": {"points": int(len(seen)), "median_abs_rel_err": float(np.median(seen)) if len(seen) else None,
                                   "max_abs_rel_err": float(np.max(seen)) if len(seen) else None,
                                   "note": "held-out points whose workload ran before D1; not part of the verdict"},
            "points": [{"name": r.name, "batch": None if pd.isna(r.batch) else r.batch, "t_obs": r.t_obs, "t_pred": float(p),
                        "rel_err": float(e), "heldout": bool(h), "judged": bool(j)}
                       for r, p, e, h, j in zip(agg.itertuples(), pred, rel, held, judged)]}


def d1(d, role):
    rows = []
    for r in read_jsonl(Path(d) / "tdiag.jsonl"):
        if r.get("status") != "ok":
            continue
        task = read_json(HERE / "tasks" / "tdiag" / Path(r["task"]).name)
        rows.append({"name": Path(r["task"]).stem, "n": task["tdiag"]["n"], "rate": r["trials_per_s"],
                     "batch": r["device"].get("microbatch") if role == "gpu" else None,
                     "L": task["cells"][0]["L"], "policies": task["policies"]})
    return d1_fit(rows, role == "gpu")


def stage_d(fd):
    freeze = load_freeze(fd)
    m = freeze["m"]
    used, report = valid_pairs(fd, freeze)
    out = {"m": m, "machines": freeze["machines"], "gpu_status": freeze["gpu"]["status"], "C": c_status(fd, freeze),
           "pairs_used": sorted(used), "slots": report}
    if not out["C"]["pass"]:
        out["H3"] = out["H4"] = {"status": "not measured", "reason": "gate C (D0) not passed or not run"}
    else:
        out["H3"] = h3(used, m)
        out["H4"], out["H4_identity_failures"] = h4(fd, used, m, freeze)
    out["D1"] = {"cpu": d1(used[1][0], "cpu") if 1 in used else {"status": "not measured", "reason": "pair 1 not valid"},
                 "gpu": d1(used[1][1], "gpu") if 1 in used and out["C"]["pass"] else {"status": "not measured"}}
    out["cost"] = cost.ec2_rows([fd])
    return out


# ---- exploratory ----

def cross_env(fd, pilot):
    """D3: GPU rates of the same calibration-split points on the pilot machine and the formal GPU (pair 1)."""
    _, pilot_b = records.label_records(pilot)
    best = {}
    for r in pilot_b:
        if r.get("kind") == "diag" and r.get("status") == "ok" and not r["config_key"].startswith("cpu:"):
            best.setdefault(Path(r["task"]).stem, []).append(r["trials_per_s"])
    used, _ = valid_pairs(fd, load_freeze(fd))
    formal_rates = {}
    if 1 in used:
        for r in read_jsonl(Path(used[1][1]) / "tdiag.jsonl"):
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
    rows = cost.ec2_rows([fd] + [RESULTS / "pilot" / label for label in verification])
    dec = read_json(formal.DECISION)
    total = sum(r["compute_usd_on_demand"] or 0 for r in rows)
    return {"rows": rows, "total_usd_on_demand": total,
            "rows_without_cost": [r["record"] for r in rows if r["compute_usd_on_demand"] is None],
            "not_included": "storage, transfer, and any instance without an ec2*.json record (check the bill)",
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
        lines += [f"m = {v['m']}; machines {v['machines']}; GPU build {v['gpu_status']}; C: {v['C']}",
                  f"pairs used: {v['pairs_used']}", "", "## H3", ""]
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
            lines += ["| budget s | per-pair eps | eps [90% CI] | identity failures | late or invalid runs | category | note |",
                      "|---|---|---|---|---|---|---|"]
            for r in v["H4"]:
                ci = f"{r['eps']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]" if r.get("eps") else r.get("reason", "")
                late = sum(x["late_or_invalid"] for x in r["per_device"].values())
                lines.append(f"| {r['budget_s']} | {', '.join(f'{x:.3f}' for x in r['per_pair_eps'])} | {ci} | "
                             f"{r['identity_failures']} | {late} | {r['category'] or r['status']} | {r.get('note') or ''} |")
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
    ap.add_argument("--pilot")
    ap.add_argument("--verification", nargs="*", default=[])
    a = ap.parse_args()
    fd = formal.formal_dir(a.label)
    if a.what == "stage-a":
        v = stage_a(fd)
    elif a.what == "stage-d":
        v = stage_d(fd)
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

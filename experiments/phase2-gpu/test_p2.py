"""Checks of the experiment code on synthetic records with known answers (no measurement involved).

    python experiments/phase2-gpu/test_p2.py
"""

import copy
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze  # noqa: E402
import scaling  # noqa: E402
from p2 import harness, machine, projection, records, stats  # noqa: E402
from p2.common import FORMAL, PROTOCOL  # noqa: E402
from p2.tasks import generate  # noqa: E402

failures = 0


def check(ok, name):
    global failures
    print(f"{name:<88} {'pass' if ok else 'FAIL'}")
    failures += not ok


def write_launch(fd, i, role, runs, status="complete", match=True):
    d = fd / f"launch-{i}-{role}"
    (d / "outputs").mkdir(parents=True, exist_ok=True)
    (d / "env.json").write_text(json.dumps({"status": status, "identity": {"match": match},
                                            "ec2": {"instance_type": "x"}, "on_demand_price": {"price_usd_per_hour": 2.0}}))
    (d / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in runs))
    return d


def h3_runs(t, cold_kind="cache-cold"):
    out = []
    for task, series, mode in FORMAL["verdicts"]["h3_primary"] + FORMAL["verdicts"]["h3_secondary"]:
        for rep, x in enumerate((t, t * 1.01, t * 0.99)):
            out.append({"mode": mode, "task": task, "series": series, "rep": rep, "e2e_s": x, "status": "ok",
                        "identity_ok": True, "cold_kind": cold_kind})
    return out


def test_stats():
    rho, lo, hi = stats.paired_interval([2.0, 2.1, 1.9])
    check(abs(rho - math.exp(np.mean(np.log([2.0, 2.1, 1.9])))) < 1e-12 and lo < rho < hi, "interval contains rho")
    check(stats.category(1.3, 2.0)[0] == "GPU advantage", "lo >= 1.25 -> GPU advantage")
    check(stats.category(0.5, 0.79)[0] == "CPU advantage", "hi <= 0.80 -> CPU advantage")
    check(stats.category(0.85, 1.2)[0] == "equivalent", "inside (0.80, 1.25) -> equivalent")
    check(stats.category(1.05, 1.6) == ("inconclusive", "direction GPU, size unresolved"), "direction note")
    check(stats.category(1.3, 2.0, censored_gpu=True)[0] == "inconclusive", "censored GPU: no GPU advantage")
    check(stats.category(0.5, 0.7, censored_gpu=True)[0] == "CPU advantage", "censored GPU: CPU advantage stands")
    check(stats.category(0.85, 1.2, censored_cpu=True)[0] == "inconclusive", "censored: no equivalence")


def test_formal(tmp):
    fd = tmp / "formal"
    for i, (c, g) in enumerate([(10.0, 5.0), (10.5, 5.1), (9.8, 5.0)], start=1):
        write_launch(fd, i, "cpu", h3_runs(c))
        write_launch(fd, i, "gpu", h3_runs(g))
    rows = analyze.h3(fd, 3)
    check(len(rows) == 9 and all(r["category"] == "GPU advantage" for r in rows), "H3: GPU 2x on 9 conditions")
    write_launch(fd, 3, "gpu", h3_runs(5.0), status="incomplete: spot interruption")
    check(all(r["status"] == "incomplete" for r in analyze.h3(fd, 3)), "H3: interrupted launch -> incomplete")
    write_launch(fd, 3, "gpu", h3_runs(5.0), match=False)
    check(all(r["status"] == "incomplete" for r in analyze.h3(fd, 3)), "H3: hardware mismatch excluded")
    write_launch(fd, 3, "gpu", h3_runs(5.0, cold_kind="process-cold"))
    rows = analyze.h3(fd, 3)
    check(all(r["status"] == ("incomplete" if r["condition"].endswith("cold") else "judged") for r in rows),
          "H3: process-cold runs never enter a formal cold verdict")

    # H4 on synthetic reference rates and prefix files
    ref = fd / "reference" / "T-ref_1s-week"
    ref.mkdir(parents=True)
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=(15, 24))
    lines = ["cell,policy,rate"] + [f"{c},{q},{float(p[c, q])!r}" for c in range(15) for q in range(24)]
    (ref / "rates.csv").write_text("\n".join(lines) + "\n")
    for i in (1, 2, 3):
        for role, N in (("cpu", 100), ("gpu", 400)):
            runs = []
            for B in PROTOCOL["tasks"]["T-ref-est"]["budgets_s"]:
                for k in range(10):
                    name = f"{role}_{B}_{k}"
                    od = fd / f"launch-{i}-{role}" / "outputs" / name
                    od.mkdir(parents=True, exist_ok=True)
                    counts = rng.binomial(N, p)
                    (od / "prefix.json").write_text(json.dumps({"replicate": k + 1, "checkpoint": 256 if N >= 256 else 0,
                                                                "cells": [{"N": N, "k": counts[c].tolist()} for c in range(15)]}))
                    runs.append({"mode": "deadline", "budget_s": B, "status": "ok", "out": f"/x/{name}", "replicate": k + 1})
            write_launch(fd, i, role, runs)
    rows, _ = analyze.h4(fd, 3, check_identity=False)
    eps = [r["eps"] for r in rows]
    check(all(r["category"] == "GPU advantage" for r in rows) and all(1.6 < e < 2.4 for e in eps),
          f"H4: 4x trials -> eps about 2 ({', '.join(f'{e:.2f}' for e in eps)})")

    # D1: an exactly linear cost is recovered and judged on 13 held-out points
    beta = np.array([2e-4, 1e-8, 1e-10, 1e-7, 5e-6])
    rows = []
    for f in sorted((HERE / "tasks" / "tdiag").glob("*.json")):
        if f.name.startswith("cross_"):
            continue
        t = json.loads(f.read_text())
        x = np.array(analyze.model_features(t["policies"], t["cells"][0]["L"], t["tdiag"]["n"]))
        rows.append({"name": f.stem, "n": t["tdiag"]["n"], "batch": None, "rate": 1 / float(x @ beta),
                     "L": t["cells"][0]["L"], "policies": t["policies"]})
    r = analyze.d1_fit(rows, gpu=False)
    check(r["status"] == "judged" and r["heldout_points"] == 13 and r["max_abs_rel_err"] < 1e-6 and r["pass"],
          "D1: linear cost recovered; 13 held-out points")


def test_projection():
    rates = {(s, g, L): [100.0, 200.0, 300.0] for s in PROTOCOL["series"] for g in ("P1", "P24")
             for L in PROTOCOL["tasks"]["T1-curve"]["lengths"]}
    p = projection.project_fixed("T2-policy-map", "1s-week", rates, warm=True)
    n = 8 * 17 * 11726
    check(p["status"] == "ok" and abs(p["seconds"]["base"] - n / 200) < 1e-6 and
          abs(p["seconds"]["conservative"] - n / 100) < 1e-6 and p["rate_flags"] == ["measured"],
          "projection: sum n_c / r_c; scenarios from slowest/median/fastest")
    sparse = {k: v for k, v in rates.items() if k[2] in (2, 32, 256)}
    check(set(projection.project_fixed("T1-curve", "1s-week", sparse, warm=True)["rate_flags"]) ==
          {"measured", "interpolated"}, "projection: unmeasured lengths flagged")
    t = projection.project_tref({c: [1000.0] for c in range(15)})
    check(t["status"] == "ok" and t["trials"] == 9071175 and abs(t["seconds"]["base"] - 9071.175) < 1e-6,
          "T-ref projection: 9,071,175 trials")
    m = projection.memory_check({"shared_bytes": 100, "working_set_bytes_per_trial": 10, "microbatch": 5,
                                 "free_at_context_bytes": 1000, "free_after_alloc_bytes": 840, "b_max": 70})
    check(m["planned_bytes"] == 150 and m["reported_bytes"] == 160 and m["difference_bytes"] == 10,
          "memory: planned shared + b x trial vs reported allocation")


def test_harness_and_tasks():
    c = harness.cpu_config(6, "0-5")
    check(harness.config_key(c) == "cpu:t6@0-5:none" and harness.launcher(c)[0] == ["taskset", "-c", "0-5"],
          "CPU configuration: key and pinning")
    g = harness.gpu_config("inc", "auto", 8)
    check(harness.config_key(g) == "cuda:inc:bauto" and "--microbatch" in harness.sf_args(g), "GPU configuration")
    info = harness.parse_device_info("NVIDIA L40S path=inc microbatch=64 b_max=812 shared_bytes=1024")
    check(info["name"] == "NVIDIA L40S" and info["b_max"] == 812, "device_info parsing")
    topo = {"logical": 18, "physical_cores": 18, "smt": False, "first_thread_per_core": "0-17", "p_cores": None}
    names = [x["name"] for x in machine.thread_candidates(topo, "0-5")]
    check(names == ["p-cores", "all"], "thread candidates: P-core set and all CPUs")
    files = generate()
    check(len([k for k in files if k.startswith("pilot/tref-cell_")]) == 15, "15 T-ref cell tasks")
    held = [k for k, t in files.items() if k.startswith("tdiag/") and t["tdiag"]["n"] in PROTOCOL["t_diag"]["heldout_n"]]
    check(len(held) == 13, "13 held-out T-diag points (never run by the pilot)")


def test_scaling():
    base = {"status": "final", "pilot": [], "outputs_and_schedule": {"required_outputs": "x", "schedule": "y"},
            "alternatives_without_scaling": [{"what": "a", "range_checked": "b", "result": "c"}],
            "actions": [{"action": "gpu", "target": "g6e.2xlarge", "constraint_observed": "c", "hypothesis": "h",
                         "metrics": [{"name": "m", "kind": "rate", "point": "calib/1s-week_P24_L32", "config": "best-gpu",
                                      "baseline": 100.0, "predicted": {"conservative": 110, "base": 150, "optimistic": 200},
                                      "outside_observed_range": "no", "threshold_gain": 1.1}],
                         "falsification": "f", "cost": {"usd_on_demand": 20.0}, "stop": "s"}],
            "formal": {"proceed": True, "cpu": "c7i.16xlarge", "gpu": "g6e.2xlarge", "cpu_size_control": None, "reason": "r"},
            "budget": {"cap_usd": 250, "spent_usd": 0.0, "planned_usd": 180.0}}
    check(scaling.problems(base) == [], "scaling record: complete record passes")
    r = copy.deepcopy(base)
    r["actions"][0]["hypothesis"] = "TODO"
    check(any("TODO" in p for p in scaling.problems(r)), "scaling record: TODO refused")
    r = copy.deepcopy(base)
    r["budget"]["planned_usd"] = 260.0
    check(any("exceeds the cap" in p for p in scaling.problems(r)), "scaling record: budget above cap refused")
    r = copy.deepcopy(base)
    r["formal"]["gpu"] = "colab"
    check(any("identity" in p for p in scaling.problems(r)), "scaling record: formal machine needs identity")
    r = copy.deepcopy(base)
    del r["actions"][0]["metrics"][0]["threshold_gain"]
    check(any("threshold_gain" in p for p in scaling.problems(r)), "scaling record: falsification threshold required")
    m = {"kind": "e2e"}
    check(records.gain(m, 100.0, 50.0) == 2.0 and records.gain({"kind": "rate"}, 100.0, 50.0) == 0.5,
          "gain: baseline/observed for times, observed/baseline for rates")


def main():
    test_stats()
    with tempfile.TemporaryDirectory() as tmp:
        test_formal(Path(tmp))
    test_projection()
    test_harness_and_tasks()
    test_scaling()
    print(f"{'tests passed' if failures == 0 else 'TESTS FAILED'} ({failures} failures)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

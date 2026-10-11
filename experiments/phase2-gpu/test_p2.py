"""Checks of the experiment code on synthetic records with known answers (no measurement involved). Besides the
ordinary cases, every counterexample of the r2 preflight audit is rebuilt here and must be refused or left without a
verdict.

    python experiments/phase2-gpu/test_p2.py
"""

import copy
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze  # noqa: E402
import launch  # noqa: E402
import scaling  # noqa: E402
import stream  # noqa: E402
from p2 import common, formal, harness, machine, projection, records, stats, validate  # noqa: E402
from p2.common import FORMAL, PROTOCOL, read_json  # noqa: E402
from p2.tasks import generate, heldout_split  # noqa: E402

failures = 0
CODE = "c0de"
GOLD = {1: "g1", 3: "g3", 5: "g5", 7: "g7"}


def check(ok, name):
    global failures
    print(f"{name:<96} {'pass' if ok else 'FAIL'}")
    failures += not ok


def jdump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def jsonl(path, recs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))


# ---- synthetic formal records ----

def env_of(stage, role, i, *, iid, started, freeze_sha=None, status="complete", match=True, region="us-east-1"):
    return {"stage": stage, "role": role, "launch_index": i, "status": status, "identity": {"match": match},
            "code_sha256": CODE, "freeze_sha256": freeze_sha, "started": started,
            "ec2": {"instance_id": iid, "region": region, "instance_type": "x"}}


def write_attempt(label_dir, stage, role, i, times, *, iid, started, freeze_sha=None, cold_kind="cache-cold",
                  one_run=False, deadline=None, **env_kw):
    """An attempt whose runs follow its order exactly (cold 1 record, warm block reps records, deadline 1 record).
    times: {condition key: seconds}; deadline: {(B, replicate): prefix dict or None}."""
    slot = (label_dir / "stage-a" / f"launch-{i}") if stage == "A" else (label_dir / "stage-d" / f"pair-{i}" / role)
    d = common.new_attempt_dir(slot)
    order = formal.conditions(role, stage, i)
    jdump(d / "order.json", {"seed": 0, "order": order})
    jdump(d / "env.json", env_of(stage, role, i, iid=iid, started=started, freeze_sha=freeze_sha, **env_kw))
    jdump(d / "gate" / "gate_cpu.json", {"items": [{"item": k, "golden_sha256": v, "pass": True} for k, v in GOLD.items()]})
    runs = []
    for pos, it in enumerate(order):
        key = formal.condition_key(it)
        base = {"task": it["task"], "series": it["series"], "position": pos, "status": "ok", "identity_ok": True,
                "spot_mismatches": 0, "out": f"/x/{pos}"}
        if it["kind"] == "cold":
            if one_run and it["rep"] > 0:
                continue
            runs.append({**base, "mode": "cold", "rep": it["rep"], "e2e_s": times[key], "cold_kind": cold_kind})
        elif it["kind"] == "warm-block":
            for r in range(1 if one_run else it["reps"]):
                runs.append({**base, "mode": "warm", "rep": r, "e2e_s": times[key]})
        else:
            pre = (deadline or {}).get((it["budget_s"], it["replicate"]))
            name = f"dl_{pos}"
            if pre is not None:
                jdump(d / "outputs" / name / "prefix.json", pre)
            runs.append({"mode": "deadline", "task": "T-ref-est", "series": "1s-week", "position": pos,
                         "budget_s": it["budget_s"], "replicate": it["replicate"], "out": f"/x/{name}",
                         "status": "ok" if pre is not None else "late"})
    jsonl(d / "runs.jsonl", runs)
    return d


def times_for(scale):
    keys = [f"{t}/{s}/{m}" for t, s, m in FORMAL["verdicts"]["h3_primary"] + FORMAL["verdicts"]["h3_secondary"]]
    return {k: scale for k in keys}


def freeze_file(label_dir, m=3, ref_files=None):
    fz = {"m": m, "pairs": list(range(1, m + 1)), "machines": {}, "code_sha256": CODE, "golden_sha256": GOLD,
          "gpu": {"path": "inc", "status": "prototype"}, "reference_files_sha256": ref_files or {}}
    p = label_dir / "protocol-freeze.json"
    jdump(p, fz)
    return common.sha256_file(p)


def d0_gate(label_dir, device="cuda", partial=False, items=range(1, 9), starts=64, code=CODE):
    g = label_dir / "d0" / "gate_cuda.json"
    jdump(g, {"device": device, "partial": partial, "starts_per_combination": starts, "config": "path=inc",
              "items": [{"item": k, "pass": True} for k in items], "pass": True})
    jdump(g.with_suffix(".meta.json"), {"code_sha256": code})


def stage_d_pairs(label_dir, fsha, cpu_s, gpu_s, *, same_instance=False, deadline=None, one_run=False):
    for i, (c, g) in enumerate(zip(cpu_s, gpu_s), start=1):
        first, second = ("cpu", "gpu") if i % 2 else ("gpu", "cpu")
        st = {first: f"2026-11-0{i}T01:00:00+00:00", second: f"2026-11-0{i}T05:00:00+00:00"}
        for role, scale in (("cpu", c), ("gpu", g)):
            iid = "i-same" if same_instance else f"i-{role}-{i}"
            write_attempt(label_dir, "D", role, i, times_for(scale), iid=iid, started=st[role], freeze_sha=fsha,
                          one_run=one_run, deadline=(deadline or {}).get((i, role)))


# ---- tests ----

def test_stats():
    rho, lo, hi = stats.paired_interval([2.0, 2.1, 1.9])
    check(abs(rho - math.exp(np.mean(np.log([2.0, 2.1, 1.9])))) < 1e-12 and lo < rho < hi, "interval contains rho")
    check(stats.category(1.3, 2.0)[0] == "GPU advantage", "lo >= 1.25 -> GPU advantage")
    check(stats.category(0.5, 0.79)[0] == "CPU advantage", "hi <= 0.80 -> CPU advantage")
    check(stats.category(0.85, 1.2)[0] == "equivalent", "inside (0.80, 1.25) -> equivalent")
    check(stats.category(1.05, 1.6) == ("inconclusive", "direction GPU, size unresolved"), "direction note")
    _, lo, hi = stats.paired_interval([2, 2, 2])
    check(stats.category(lo, hi, censored_cpu=True)[0] == "censored", "audit F07: censored [2,2,2] gets no verdict")
    _, lo2, hi2 = stats.paired_interval([2, 2, 200])
    check(abs(lo2 - 0.104964) < 1e-5 and abs(hi2 - 821.019961) < 1e-3, "audit F07: [2,2,200] CI reproduced")
    check(stats.category(0.5, 0.7, censored_gpu=True)[0] == "censored", "censored GPU: no verdict either way")


def test_layout(tmp):
    slot = tmp / "slot"
    a1, a2 = common.new_attempt_dir(slot), common.new_attempt_dir(slot)
    check((a1.name, a2.name) == ("attempt-1", "attempt-2"), "audit F01: attempts are new directories, never reused")
    (a1 / "x").write_text("1")
    try:
        common.fresh_dir(a1)
        ok = False
    except FileExistsError:
        ok = True
    check(ok, "audit F01: an existing output directory is never overwritten")
    paths = {formal.slot_dir("L", "A", "cpu", 1), formal.slot_dir("L", "D", "cpu", 1), formal.slot_dir("L", "D", "gpu", 1)}
    check(len(paths) == 3, "audit F01: Stage A launch 1 and Stage D pair 1 have different slots")
    rec = harness.run_cold(HERE / "tasks" / "main" / "T1-curve_1m-quarter.json", a1, harness.cpu_config(1), {},
                           build=tmp)
    check(rec["status"] == "harness_error", "audit F11: a harness failure becomes a record, not a lost run")


def test_stage_a(tmp):
    lab = tmp / "A1"
    for i in (1, 2):
        write_attempt(lab, "A", "cpu", i, times_for(10.0), iid=f"i-a{i}", started="2026-11-01T00:00:00+00:00")
    d1 = formal.slot_dir("A1", "A", "cpu", 1)
    common.new_attempt_dir(d1)  # an empty, crashed second attempt: reported, never used
    pick, rep = validate.chosen(d1, "A", "cpu", 1)
    check(pick is not None and pick.name == "attempt-1" and rep[1]["problems"], "Stage A: first valid attempt used")
    # a Stage D record in the Stage A slot is not a Stage A launch
    lab2 = tmp / "A2"
    d = write_attempt(lab2, "A", "cpu", 1, times_for(10.0), iid="i-x", started="2026-11-01T00:00:00+00:00")
    e = read_json(d / "env.json")
    e["stage"] = "D"
    jdump(d / "env.json", e)
    check(validate.chosen(formal.slot_dir("A2", "A", "cpu", 1), "A", "cpu", 1)[0] is None,
          "audit F01: a D-tagged record never sets m")


def test_h3(tmp):
    lab = tmp / "H3"
    fsha = freeze_file(lab)
    d0_gate(lab)
    stage_d_pairs(lab, fsha, (10.0, 10.5, 9.8), (5.0, 5.1, 5.0))
    v = analyze.stage_d(lab)
    check(v["C"]["pass"] and len(v["H3"]) == 9 and all(r["category"] == "GPU advantage" for r in v["H3"]),
          "H3: GPU 2x on 9 conditions, 3 valid pairs")
    lab = tmp / "H3same"
    fsha = freeze_file(lab)
    d0_gate(lab)
    stage_d_pairs(lab, fsha, (10.0, 10.5, 9.8), (5.0, 5.1, 5.0), same_instance=True, one_run=True)
    v = analyze.stage_d(lab)
    check(all(r["status"] == "incomplete" for r in v["H3"]),
          "audit F03: same instance + one run per condition -> no verdict (was 9 x GPU advantage)")
    lab = tmp / "H3cold"
    fsha = freeze_file(lab)
    d0_gate(lab)
    stage_d_pairs(lab, fsha, (10.0, 10.5, 9.8), (5.0, 5.1, 5.0))
    g3 = formal.attempts(lab / "stage-d" / "pair-3" / "gpu")[0]
    runs = [json.loads(x) for x in (g3 / "runs.jsonl").read_text().splitlines()]
    for r in runs:
        if r["mode"] == "cold":
            r["cold_kind"] = "process-cold"
    jsonl(g3 / "runs.jsonl", runs)
    v = analyze.stage_d(lab)
    check(all(r["status"] == "incomplete" for r in v["H3"]) and v["slots"][3]["gpu"][0]["problems"],
          "H3: a launch with process-cold runs is excluded whole")
    e = read_json(g3 / "env.json")
    check(e["status"] == "complete", "(fixture) excluded launch was complete")
    # wrong order of a pair
    lab = tmp / "H3order"
    fsha = freeze_file(lab)
    d0_gate(lab)
    stage_d_pairs(lab, fsha, (10.0, 10.5, 9.8), (5.0, 5.1, 5.0))
    c1 = formal.attempts(lab / "stage-d" / "pair-1" / "cpu")[0]
    e = read_json(c1 / "env.json")
    e["started"] = "2026-11-01T09:00:00+00:00"
    jdump(c1 / "env.json", e)
    v = analyze.stage_d(lab)
    check(1 not in v["pairs_used"] and any("registered to start" in p for p in v["slots"][1]["pair_problems"]),
          "pair order: a pair that started with the wrong role is excluded")


def test_gates(tmp):
    lab = tmp / "G"
    freeze_file(lab)
    d0_gate(lab, device="emul", partial=True, items=[])
    v = analyze.c_status(lab, read_json(lab / "protocol-freeze.json"))
    check(not v["pass"], "audit F04: emul/partial/empty gate does not pass C (was C judged pass)")
    d0_gate(lab, code="other")
    check(not analyze.c_status(lab, read_json(lab / "protocol-freeze.json"))["pass"], "audit F04: D0 of other code refused")
    d0_gate(lab, starts=2)
    check(not analyze.c_status(lab, read_json(lab / "protocol-freeze.json"))["pass"], "D0 with 2 starts refused")
    d0_gate(lab)
    check(analyze.c_status(lab, read_json(lab / "protocol-freeze.json"))["pass"], "D0 full gate of the frozen code passes")


def reference_fixture(lab, p_of):
    """T-ref bits drawn with per-(cell, policy) probability p_of[c, q]; returns (bits, frozen file hashes)."""
    task = read_json(HERE / "tasks" / "main" / "T-ref-est_1s-week.json")
    n, P = PROTOCOL["series"]["1s-week"]["bars"], 24
    rng = np.random.default_rng(1)
    parts, rows = [], ["cell,policy,starts,k,rate"]
    for c, cell in enumerate(task["cells"]):
        M = n - cell["L"] + 1
        b = (rng.random((M, P)) < p_of[c][None, :]).astype(np.uint8)
        parts.append(b.reshape(-1))
        k = b.sum(axis=0)
        rows += [f"{c},{q},{M},{int(k[q])},{k[q] / M!r}" for q in range(P)]
    bits = np.concatenate(parts)
    ref = lab / "reference" / "T-ref_1s-week"
    ref.mkdir(parents=True)
    (ref / "ref_bits.bin").write_bytes(np.packbits(bits, bitorder="little").tobytes())
    (ref / "rates.csv").write_text("\n".join(rows) + "\n")
    return bits, {x: common.sha256_file(ref / x) for x in ("ref_bits.bin", "rates.csv")}


def honest_prefix(bits, task, replicate, N):
    n, P, offset, cells = PROTOCOL["series"]["1s-week"]["bars"], 24, 0, []
    for c, cell in enumerate(task["cells"]):
        M = n - cell["L"] + 1
        starts = stream.starts_np(task["task_code"], task["series"]["code"], cell["stream_cell"], replicate,
                                  stream.PURPOSES["measured"], np.arange(N), M)
        k = bits[(offset + starts)[:, None] * P + np.arange(P)[None, :]].sum(axis=0)
        cells.append({"cell": c, "L": cell["L"], "q": cell["q"], "N": N, "k": k.tolist()})
        offset += M
    return {"G": N * len(task["cells"]), "replicate": replicate, "checkpoint": 256 if N >= 256 else 0, "cells": cells}


def test_h4(tmp):
    task = read_json(HERE / "tasks" / "main" / "T-ref-est_1s-week.json")
    lab = tmp / "H4"
    p = np.random.default_rng(0).uniform(0.05, 0.95, size=(15, 24))
    bits, files = reference_fixture(lab, p)
    fsha = freeze_file(lab, ref_files=files)
    d0_gate(lab)
    k = EST_SEEDS = PROTOCOL["tasks"]["T-ref-est"]["seeds_per_launch_pair"]
    dl = {}
    for i in (1, 2, 3):
        for role, N in (("cpu", 100), ("gpu", 400)):
            dl[(i, role)] = {(B, k * (i - 1) + j + 1): honest_prefix(bits, task, k * (i - 1) + j + 1, N)
                             for B in PROTOCOL["tasks"]["T-ref-est"]["budgets_s"] for j in range(EST_SEEDS)}
    stage_d_pairs(lab, fsha, (10.0, 10.5, 9.8), (5.0, 5.1, 5.0), deadline=dl)
    v = analyze.stage_d(lab)
    eps = [r.get("eps") for r in v["H4"]]
    check(all(r["category"] == "GPU advantage" for r in v["H4"]) and all(1.6 < e < 2.4 for e in eps),
          f"H4: 4x trials -> eps about 2 ({', '.join(f'{e:.2f}' for e in eps)})")

    # audit F02: all-zero reference, CPU k = 40/100, GPU k = 10/100, one seed, seeds reused across pairs
    lab = tmp / "H4bad"
    bits0, files0 = reference_fixture(lab, np.zeros((15, 24)))
    fsha = freeze_file(lab, ref_files=files0)
    d0_gate(lab)

    def fake(N, kk, rep):
        return {"G": N * 15, "replicate": rep, "checkpoint": 0,
                "cells": [{"cell": c, "L": cl["L"], "q": cl["q"], "N": N, "k": [kk] * 24} for c, cl in enumerate(task["cells"])]}
    dl = {(i, role): {(B, 1): fake(100, kk, 1) for B in PROTOCOL["tasks"]["T-ref-est"]["budgets_s"]}
          for i in (1, 2, 3) for role, kk in (("cpu", 40), ("gpu", 10))}
    stage_d_pairs(lab, fsha, (10.0, 10.5, 9.8), (5.0, 5.1, 5.0), deadline=dl)
    v = analyze.stage_d(lab)
    check(all(r["category"] is None and r["status"] in ("invalid", "incomplete") for r in v["H4"]),
          "audit F02: identity failures and one reused seed -> no H4 category (was GPU advantage, eps 4)")
    check(all(r["identity_failures"] > 0 for r in v["H4"]), "audit F02: identity failures are counted")
    (lab / "reference" / "T-ref_1s-week" / "ref_bits.bin").unlink()
    v = analyze.stage_d(lab)
    check(v["H4"]["status"] == "not measured", "audit F02: missing reference bits -> H4 not measured")
    bad = fake(0, 999, 1)
    check(analyze.identity_check(bad, bits0, task, 1) != [], "audit F02: N = 0 with k = 999 fails the identity check")
    ok = honest_prefix(bits, task, 7, 5)
    check(analyze.identity_check(ok, bits, task, 7) == [], "identity check: an honest prefix passes")
    check(analyze.identity_check(ok, bits, task, 8) != [], "identity check: wrong replicate fails")
    sw = copy.deepcopy(ok)
    sw["cells"][0]["N"] = 6
    check(analyze.identity_check(sw, bits, task, 7) != [], "identity check: N inconsistent with G fails")
    rates, _ = analyze.reference_rates(tmp / "H4", read_json(tmp / "H4" / "protocol-freeze.json"))
    check(rates[0].dtype == np.float64, "audit F02: exact rates are float (0/1 text never truncates estimates)")


def test_freeze_refuses(tmp):
    """audit F04: R1 false, no parity log, no partial device gate -> no freeze (was protocol-freeze.json written)."""
    import freeze
    lab = tmp / "FZ"
    for i in (1, 2):
        d = write_attempt(lab, "A", "cpu", i, times_for(10.0), iid=f"i-f{i}", started="2026-11-01T00:00:00+00:00")
        runs = [json.loads(x) for x in (d / "runs.jsonl").read_text().splitlines()]
        runs[0]["spot_mismatches"] = 3  # R1 fails
        jsonl(d / "runs.jsonl", runs)
    old_dec, old_argv = formal.decision, sys.argv
    formal.decision = lambda require_commit=True: {"formal": {"cpu": "c7i.16xlarge", "gpu": "g6e.2xlarge"}}
    sys.argv = ["freeze.py", "--label", "FZ", "--gpu-status", "prototype", "--warmup-cpu", "0", "--warmup-gpu", "0",
                "--pair-cost-usd", "30"]
    try:
        freeze.main()
        msg = ""
    except SystemExit as e:
        msg = str(e)
    finally:
        formal.decision, sys.argv = old_dec, old_argv
    check("R1 not passed" in msg and "parity" in msg and "GPU path selection" in msg and not (lab / "protocol-freeze.json").exists(),
          "audit F04: freeze refused with R1 failed, no parity log, no device gate")


def test_d1():
    split = heldout_split()
    check(split["unseen"] == PROTOCOL["t_diag"]["heldout_judged"] and len(split["unseen"]) == 6 and len(split["seen"]) == 7,
          "audit F06: 6 held-out points unseen, 7 share an exposed workload (registered)")
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
    check(r["status"] == "judged" and r["heldout_judged"] == 6 and r["max_abs_rel_err"] < 1e-6 and r["pass"]
          and r["repeat_predictions"]["points"] == 7, "D1: linear cost recovered; judged on 6, 7 reported apart")


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
    g = harness.gpu_config("inc", 256, 8, submission=2, warmup=100)
    check(harness.config_key(g) == "cuda:inc:b256:h8:s2:w100" and "--microbatch" in harness.sf_args(g),
          "audit F08: GPU key includes host threads, submission, warm-up")
    try:
        harness.gpu_config("inc", 64, {"1s-week/P24": 8})
        ok = False
    except TypeError:
        ok = True
    check(ok, "audit F08: a dict of host threads is refused at the configuration")
    fz = {"cpu": {"parity": "none", "threads": {}}, "gpu": {"path": "inc", "submission": 1, "warmup_trials": 0,
          "microbatch": {"1s-week/P24": 256, "1m-quarter/P1": 64}, "host_threads": {"1s-week/P24": 8, "1m-quarter/P1": 4}}}
    a, b = launch.config_for("gpu", "T2-policy-map", "1s-week", fz), launch.config_for("gpu", "T1-curve", "1m-quarter", fz)
    check((a["threads"], a["microbatch"], b["threads"], b["microbatch"]) == (8, 256, 4, 64),
          "audit F08: per (series, grid) host threads and batch reach the run (was TypeError)")
    info = harness.parse_device_info("NVIDIA L40S path=inc microbatch=64 b_max=812 shared_bytes=1024")
    check(info["name"] == "NVIDIA L40S" and info["b_max"] == 812, "device_info parsing")
    topo = {"logical": 18, "physical_cores": 18, "smt": False, "first_thread_per_core": "0-17", "p_cores": None}
    names = [x["name"] for x in machine.thread_candidates(topo, "0-5")]
    check(names == ["p-cores", "all"], "thread candidates: P-core set and all CPUs")
    files = generate()
    check(len([k for k in files if k.startswith("pilot/tref-cell_")]) == 15, "15 T-ref cell tasks")
    held = [k for k, t in files.items() if k.startswith("tdiag/") and t["tdiag"]["n"] in PROTOCOL["t_diag"]["heldout_n"]]
    check(len(held) == 13, "13 held-out T-diag points (never run by the pilot)")
    if len(os.sched_getaffinity(0)) >= 4:
        code = ("import os,sys,json; os.sched_setaffinity(0,{0,1}); sys.path.insert(0,%r); from p2 import machine; "
                "print(json.dumps(machine.topology()))" % str(HERE))
        t = json.loads(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout)
        check(t["logical"] == 2 and t["physical_cores"] <= 2 and set(machine._cpulist(t["first_thread_per_core"])) <= {0, 1},
              "audit F10: topology restricted to the allowed CPUs (was physical 9 with 2 logical)")
    harness.TIME_BOX_NS = 0
    try:
        harness.bounded(10)
        ok = False
    except harness.TimeBox:
        ok = True
    harness.TIME_BOX_NS = None
    check(ok, "audit F11: no run starts after the time box")


def scaling_base():
    return {"status": "final", "pilot": [{"label": "p", "observations": {"e2e": []}, "metric_catalog": [{}],
                                          "summary_sha256": "x"}],
            "outputs_and_schedule": {"required_outputs": "x", "schedule": "y"},
            "alternatives_without_scaling": [{"what": "a", "range_checked": "b", "result": "c"}],
            "actions": [{"action": "gpu", "target": "g6e.2xlarge", "constraint_observed": "c", "hypothesis": "h",
                         "metrics": [{"name": "m", "kind": "rate", "point": "calib/1s-week_P24_L32", "config": "best-gpu",
                                      "baseline_label": "p", "baseline": 100.0,
                                      "predicted": {"conservative": 110, "base": 150, "optimistic": 200},
                                      "outside_observed_range": "no", "threshold_gain": 1.1}],
                         "falsification": "f", "cost": {"usd_on_demand": 20.0, "instance_hours": 10}, "stop": "s"}],
            "formal": {"proceed": True, "cpu": "c7i.16xlarge", "gpu": "g6e.2xlarge", "cpu_size_control": None, "reason": "r"},
            "budget": {"cap_usd": 250, "spent_usd": 0.0, "planned_usd": 180.0, "formal_items": []}}


def test_scaling():
    base = scaling_base()
    pr = [x for x in scaling.problems(base, recompute=False) if "summary changed" not in x]
    check(pr == [], "scaling record: complete record passes (summary hash aside)")
    r = copy.deepcopy(base)
    r["actions"][0]["hypothesis"] = "TODO"
    check(any("TODO" in p for p in scaling.problems(r, recompute=False)), "scaling record: TODO refused")
    r = copy.deepcopy(base)
    r["budget"]["planned_usd"] = 260.0
    check(any("exceeds the cap" in p for p in scaling.problems(r, recompute=False)), "scaling record: budget above cap")
    r = copy.deepcopy(base)
    r["formal"]["gpu"] = "colab"
    check(any("identity" in p for p in scaling.problems(r, recompute=False)), "scaling record: formal machine identity")
    r = copy.deepcopy(base)
    del r["actions"][0]["metrics"][0]["threshold_gain"]
    check(any("threshold_gain" in p for p in scaling.problems(r, recompute=False)), "scaling record: threshold required")
    r = copy.deepcopy(base)
    r["pilot"] = []
    del r["alternatives_without_scaling"]
    del r["actions"][0]["falsification"]
    del r["actions"][0]["stop"]
    del r["outputs_and_schedule"]["schedule"]
    pr = scaling.problems(r, recompute=False)
    check(all(any(w in p for p in pr) for w in ("pilot", "alternatives", "falsification", "stop", "schedule")),
          "audit F15: deleted observations/schedule/alternatives/falsification/stop refused (was [])")
    r = copy.deepcopy(base)
    r["actions"][0]["metrics"][0]["predicted"] = {"conservative": 200, "base": 150, "optimistic": float("inf")}
    check(any("predicted" in p for p in scaling.problems(r, recompute=False)), "audit F15: non-finite/unordered forecast")
    r = copy.deepcopy(base)
    r["actions"][0]["metrics"][0].update({"kind": "completion", "task": "T1-curve", "series": "1s-week", "mode": "warm"})
    check(any("environment" in p for p in scaling.problems(r, recompute=False)),
          "completion metrics verify an environment action only")
    m = {"kind": "e2e"}
    check(records.gain(m, 100.0, 50.0) == 2.0 and records.gain({"kind": "rate"}, 100.0, 50.0) == 0.5 and
          records.gain({"kind": "spread"}, 1.5, 1.2) == 1.25, "gain: lower-is-better kinds invert")


def test_records(tmp):
    root = tmp / "results"
    old = records.RESULTS
    records.RESULTS = root
    try:
        for k, gpu in ((1, "Tesla T4"), (2, "NVIDIA L4")):
            sd = root / "pilot" / "lab" / f"session-{k}"
            jdump(sd / "env.json", {"gpu": {"name": gpu}, "hardware_fingerprint": f"hw{k}"})
            jsonl(sd / "bench.jsonl", [{"task": "experiments/phase2-gpu/tasks/calib/1s-week_P24_L32.json",
                                        "status": "ok", "trials_per_s": 100.0 * k, "config_key": "cuda:inc:b64:h2:s1:w0",
                                        "kind": "rate", "run_id": f"r{k}"}])
        m = {"kind": "rate", "point": "calib/1s-week_P24_L32", "config": "best-gpu"}
        try:
            records.observed("lab", m)
            ok = False
        except ValueError:
            ok = True
        check(ok, "audit F10: two sessions on different GPUs are never pooled into one baseline")
        check(records.observed("lab", {**m, "session": 2})[0] == 200.0, "a session-specific baseline is allowed")
    finally:
        records.RESULTS = old


def main():
    test_stats()
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        old = formal.formal_dir
        formal.formal_dir = lambda label: tmp / label
        try:
            test_layout(tmp)
            test_stage_a(tmp)
            test_h3(tmp)
            test_gates(tmp)
            test_h4(tmp)
            test_records(tmp)
            test_freeze_refuses(tmp)
        finally:
            formal.formal_dir = old
    test_d1()
    test_projection()
    test_harness_and_tasks()
    test_scaling()
    print(f"{'tests passed' if failures == 0 else 'TESTS FAILED'} ({failures} failures)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

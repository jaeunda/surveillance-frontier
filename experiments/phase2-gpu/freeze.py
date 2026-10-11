"""Assemble protocol-freeze.json (Part III, "What is fixed when") from the committed scaling decision, Stage A, the
formal calibration, and the reference. Refuses an incomplete or inconsistent freeze; every refusal names the missing
or failed prerequisite.

    freeze.py --label L --gpu-status optimised|prototype --warmup-cpu W --warmup-gpu W [--submission S]
              --pair-cost-usd X [--stage-a-code-note "..."]

Prerequisites: Stage A judged with R1 passed and m set (recomputed here from the stored records); CPU threads, parity,
rates, drain and drain check on the formal CPU; GPU partial device gate of the chosen path with this code, host
threads, batch, path, drain and drain check on the formal GPU, calibrated with the warm-up and submission given here;
the Stage B reference (fixed-task hashes, T-ref bits and rates, oracle passed, golden CPU gate and item 9 passed) and
identical golden hashes in the Stage A gates. m comes only from the formal Stage A CPU launches. Timeouts are 3 x the
conservative projection of each condition from the formal CPU rates of that condition's grid.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze  # noqa: E402
from p2 import formal, harness, machine, projection, validate  # noqa: E402
from p2.common import (FORMAL, PROTOCOL, RESULTS, code_fingerprint, git_state, now, read_json,  # noqa: E402
                       read_jsonl, sha256_file, write_json)

REP = "1s-week/P24"
FIXED = ("T1-curve/1m-quarter", "T1-curve/1s-week", "T2-policy-map/1m-quarter", "T2-policy-map/1s-week", "T-ref/1s-week")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--gpu-status", choices=["optimised", "prototype"], required=True)
    ap.add_argument("--warmup-cpu", type=int, required=True)
    ap.add_argument("--warmup-gpu", type=int, required=True)
    ap.add_argument("--submission", type=int, default=1)
    ap.add_argument("--pair-cost-usd", type=float, required=True,
                    help="estimated on-demand cost of one complete launch pair (both launches, all conditions)")
    ap.add_argument("--stage-a-code-note", help="required if Stage A ran on other code than the code frozen now")
    a = ap.parse_args()
    rec = formal.decision()
    fd = formal.formal_dir(a.label)
    code = code_fingerprint()
    cpu_name, _ = formal.profile_of(rec, "cpu")
    gpu_name, _ = formal.profile_of(rec, "gpu")
    cd, gd, rd = fd / "calibration" / cpu_name, fd / "calibration" / gpu_name, fd / "reference"
    problems = []

    def load(path, what):
        if not Path(path).exists():
            problems.append(f"missing {what} ({Path(path).relative_to(fd)})")
            return None
        return read_json(path)

    # Stage A, recomputed from the records
    sa = analyze.stage_a(fd)
    if sa["R1"].get("status") != "judged" or sa["R1"].get("pass") is not True:
        problems.append(f"R1 not passed ({sa['R1']}): no speed verdicts until fixed")
    if sa.get("m") is None:
        problems.append("Stage A did not set m (fewer than two valid formal CPU launches)")
    stage_a_codes = sorted({read_json(Path(x["dir"]) / "env.json").get("code_sha256") for x in sa["used"].values()})
    if stage_a_codes and stage_a_codes != [code] and not a.stage_a_code_note:
        problems.append("Stage A ran on other code than the code frozen now: give --stage-a-code-note (what changed and "
                        "why the CPU timings still apply) or repeat Stage A")

    # CPU calibration
    threads = load(cd / "threads.json", "CPU threads")
    parity = load(cd / "parity.json", "CPU parity log")
    if threads and parity and threads.get("parity", "none") != parity["parity"]:
        problems.append(f"threads calibrated with parity {threads.get('parity')}, CPU-best uses {parity['parity']}")
    rates_cpu = read_jsonl(cd / "rates_cpu.jsonl")
    if not rates_cpu:
        problems.append("missing CPU rates (calibrate.py rates --config cpu)")
    drains = {}
    for dev, d, warm in (("cpu", cd, a.warmup_cpu), ("gpu", gd, a.warmup_gpu)):
        dr = load(d / f"drain_{dev}.json", f"{dev} drain")
        ck = load(d / f"drain_check_{dev}.json", f"{dev} drain check")
        if dr:
            if dr.get("drain_ms") is None:
                problems.append(f"drain {dev} incomplete")
            if dr["config"].get("warmup_trials") != warm:
                problems.append(f"drain {dev} calibrated with warm-up {dr['config'].get('warmup_trials')}, not {warm}")
            if dev == "gpu" and dr["config"].get("submission") != a.submission:
                problems.append(f"drain gpu calibrated with submission {dr['config'].get('submission')}")
        if dr and ck and ck.get("config_key") != dr.get("config_key"):
            problems.append(f"drain check {dev} used another configuration than the drain calibration")
        drains[dev] = {"drain": dr, "check": ck}

    # GPU calibration
    path_rec = load(gd / "path.json", "GPU path selection")
    path = path_rec.get("choice") if path_rec else None
    if path_rec and not path:
        problems.append("no GPU path chosen")
    host = load(gd / "threads_host.json", "GPU host threads")
    batch = load(gd / f"batch_{path}.json", f"batch of path {path}") if path else None
    if path:
        gp = validate.gate_problems(gd / f"gate_partial_{path}.json", device="cuda", full=False, path_name=path,
                                    code=code)
        problems += [f"partial device gate {path}: {x}" for x in gp]
    for k in (f"{s}/{g}" for s in PROTOCOL["series"] for g in ("P1", "P24")):
        if threads and k not in threads["threads"]:
            problems.append(f"threads {k}")
        if host and k not in host["threads"]:
            problems.append(f"host threads {k}")
        if batch and k not in batch["microbatch"]:
            problems.append(f"batch {k}")

    # Stage B reference
    reference = load(rd / "hashes.json", "reference hashes") or {}
    problems += [f"reference {k}" for k in FIXED if k not in reference]
    oracle = load(rd / "oracle.json", "oracle")
    if oracle and oracle.get("pass") is not True:
        problems.append("oracle did not pass")
    o = FORMAL["oracle"]
    if oracle and (oracle.get("shared_per_cell"), oracle.get("full_per_cell")) != (o["starts_per_cell_shared"],
                                                                                  o["starts_per_cell_full"]):
        problems.append("oracle ran with other start counts than registered")
    problems += [f"reference CPU gate: {x}" for x in validate.gate_problems(rd / "gate_cpu.json", device="cpu", full=True)]
    item9 = load(rd / "item9.json", "item 9 (Phase 1 hit files)")
    if item9 and item9.get("pass") is not True:
        problems.append("item 9 did not pass")
    tref = rd / "T-ref_1s-week"
    ref_files = {n: sha256_file(tref / n) for n in ("ref_bits.bin", "rates.csv") if (tref / n).exists()}
    if len(ref_files) != 2:
        problems.append("T-ref reference bits or rates missing")
    elif reference.get("T-ref/1s-week") != ref_files["ref_bits.bin"]:
        problems.append("T-ref bits do not match the recorded reference hash")
    elif oracle and oracle.get("reference_sha256") != ref_files["ref_bits.bin"]:
        problems.append("the oracle checked other T-ref bits than the reference")
    if (rd / "gate_cpu.json").exists():
        gold = validate.goldens(rd / "gate_cpu.json")
        for i, x in sa["used"].items():
            g2 = Path(x["dir"]) / "gate" / "gate_cpu.json"
            if not g2.exists():
                problems.append(f"Stage A launch {i}: no CPU gate")
                continue
            other = validate.goldens(g2)
            diff = [k for k in gold if k in other and other[k] != gold[k]]
            if diff:
                problems.append(f"Stage A launch {i}: golden hashes differ from the reference in items {diff}")

    if problems:
        sys.exit("incomplete freeze:\n- " + "\n- ".join(problems))

    # timeouts from the formal CPU rates of each condition's grid
    rates, setups = projection.rate_table([r for r in rates_cpu if r.get("kind") == "rate"])
    cells = {}
    for r in rates_cpu:
        if r.get("kind") == "tref-rate" and r.get("status") == "ok":
            cells.setdefault(r["meta"]["tref_cell"]["cell"], []).append(r["trials_per_s"])
    proj = {}
    for task in ("T1-curve", "T2-policy-map"):
        for series in PROTOCOL["tasks"][task]["series"]:
            for mode in ("cold", "warm"):
                proj[f"{task}/{series}/{mode}"] = projection.project_fixed(task, series, rates, setups, warm=mode == "warm")
    setup = max((max(v) for v in setups.values()), default=0.0)
    proj["T-ref/1s-week/cold"] = projection.project_tref(cells, setup_s=setup)
    factor = FORMAL["verdicts"]["timeout_factor"]
    timeouts = {k: factor * p["seconds"]["conservative"] for k, p in proj.items() if p.get("status") == "ok"}
    missing = [k for k in proj if k not in timeouts]
    if missing:
        sys.exit(f"incomplete freeze: no projection for {missing}")

    ver = sorted(p.name.split("_", 1)[1].rsplit("_action", 1)[0] for p in (RESULTS / "scaling").glob("verification_*.json"))
    freeze = {
        "_about": "Values fixed by the protocol's rules before Stage D. Written by freeze.py; commit with the code.",
        "frozen": now(),
        "git": git_state(),
        "code_sha256": code,
        "stage_a_code_sha256": stage_a_codes,
        "stage_a_code_note": a.stage_a_code_note,
        "scaling_decision_sha256": sha256_file(formal.DECISION),
        "verification_labels": ver,
        "machines": {"cpu": cpu_name, "gpu": gpu_name, "size_control": rec["formal"].get("cpu_size_control")},
        "toolchain": machine.build_info(Path(__file__).resolve().parents[2] / "build")["cmake_cache"],
        "m": sa["m"], "sigma_stage_a": sa["sigma"], "pairs": list(range(1, sa["m"] + 1)),
        "pair_first_role": {str(i): formal.first_role(i) for i in range(1, sa["m"] + 1)},
        "pair_cost_usd": a.pair_cost_usd,
        "cpu": {"parity": parity["parity"], "threads": threads["threads"], "warmup_trials": a.warmup_cpu,
                "drain_ms": drains["cpu"]["drain"]["drain_ms"], "drain_check": drains["cpu"]["check"]},
        "gpu": {"path": path, "status": a.gpu_status, "microbatch": batch["microbatch"], "b_max": batch["b_max"][REP],
                "submission": a.submission, "host_threads": host["threads"], "warmup_trials": a.warmup_gpu,
                "drain_ms": drains["gpu"]["drain"]["drain_ms"], "drain_check": drains["gpu"]["check"]},
        "config_keys": {"cpu": {k: harness.config_key(harness.cpu_config(v, parity=parity["parity"], warmup=a.warmup_cpu))
                                for k, v in threads["threads"].items()},
                        "gpu": {k: harness.config_key(harness.gpu_config(path, batch["microbatch"][k], host["threads"][k],
                                                                         a.submission, a.warmup_gpu))
                                for k in batch["microbatch"]}},
        "projections": proj,
        "timeouts_s": timeouts,
        "reference_sha256": {k: reference[k] for k in FIXED},
        "reference_files_sha256": ref_files,
        "golden_sha256": validate.goldens(rd / "gate_cpu.json"),
        "inputs_sha256": {str(p.relative_to(fd)): sha256_file(p) for p in sorted(
            [cd / "threads.json", cd / "parity.json", cd / "rates_cpu.jsonl", gd / "path.json", gd / "threads_host.json",
             gd / f"batch_{path}.json", gd / f"gate_partial_{path}.json", rd / "hashes.json", rd / "oracle.json",
             rd / "gate_cpu.json", rd / "item9.json", cd / "drain_cpu.json", gd / "drain_gpu.json"])},
    }
    out = fd / "protocol-freeze.json"
    write_json(out, freeze)
    print(f"wrote {out}; commit it with the frozen code, then calibrate.py d0 on the GPU machine, commit d0/, Stage D")


if __name__ == "__main__":
    main()

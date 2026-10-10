"""Assemble protocol-freeze.json (Part III, "What is fixed when") from the committed scaling decision, Stage A
verdicts, the formal calibration, and the reference. Refuses an incomplete freeze.

    freeze.py --label L --gpu-status optimised|prototype --warmup-cpu W --warmup-gpu W --nvcc VERSION --driver VERSION

m comes only from analyze.py stage-a (formal CPU launches). Timeouts are 3 x the conservative projection of each
condition from the formal CPU rates (calibrate.py rates --config cpu): T = T_setup + sum_c n_c / r_c, without setup
for warm conditions.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from p2 import formal, projection  # noqa: E402
from p2.common import FORMAL, PROTOCOL, git_state, now, read_json, read_jsonl, sh, sha256_file, write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--gpu-status", choices=["optimised", "prototype"], required=True)
    ap.add_argument("--warmup-cpu", type=int, required=True)
    ap.add_argument("--warmup-gpu", type=int, required=True)
    ap.add_argument("--submission", type=int, default=1)
    ap.add_argument("--nvcc", required=True)
    ap.add_argument("--driver", required=True)
    a = ap.parse_args()
    rec = formal.decision()
    fd = formal.formal_dir(a.label)
    cpu_name, _ = formal.profile_of(rec, "cpu")
    gpu_name, _ = formal.profile_of(rec, "gpu")
    cd, gd = fd / "calibration" / cpu_name, fd / "calibration" / gpu_name
    need = {"stage-a verdicts": fd / "verdicts_stage-a.json", "threads": cd / "threads.json",
            "cpu rates": cd / "rates_cpu.jsonl", "drain cpu": cd / "drain_cpu.json", "path": gd / "path.json",
            "drain gpu": gd / "drain_gpu.json", "reference": fd / "reference" / "hashes.json",
            "oracle": fd / "reference" / "oracle.json"}
    missing = [k for k, p in need.items() if not p.exists()]
    if missing:
        sys.exit("incomplete freeze, missing: " + ", ".join(missing))
    stage_a = read_json(need["stage-a verdicts"])
    if stage_a.get("m") is None:
        sys.exit("Stage A did not set m (fewer than two complete formal CPU launches)")
    path = read_json(need["path"])["choice"]
    batch = read_json(gd / f"batch_{path}.json")["microbatch"]
    threads = read_json(need["threads"])["threads"]
    parity = read_json(cd / "parity.json")["parity"] if (cd / "parity.json").exists() else "none"
    host = read_json(gd / "threads_host.json")["threads"] if (gd / "threads_host.json").exists() else None
    reference = read_json(need["reference"])

    benches = read_jsonl(need["cpu rates"])
    rates, setups = projection.rate_table([r for r in benches if r.get("kind") == "rate"])
    cells = {}
    for r in benches:
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

    problems = [f"timeout {k}" for k in proj if k not in timeouts]
    problems += [f"threads {s}/{g}" for s in PROTOCOL["series"] for g in ("P1", "P24") if f"{s}/{g}" not in threads]
    problems += [f"batch {s}/{g}" for s in PROTOCOL["series"] for g in ("P1", "P24") if f"{s}/{g}" not in batch]
    problems += [f"reference {k}" for k in ("T1-curve/1m-quarter", "T1-curve/1s-week", "T2-policy-map/1m-quarter",
                                            "T2-policy-map/1s-week", "T-ref/1s-week") if k not in reference]
    if not read_json(need["oracle"]).get("pass", False):
        problems.append("oracle did not pass")
    for dev in ("cpu", "gpu"):
        if read_json(need[f"drain {dev}"]).get("drain_ms") is None:
            problems.append(f"drain {dev} incomplete")
    if problems:
        sys.exit("incomplete freeze: " + ", ".join(problems))

    freeze = {
        "_about": "Values fixed by the protocol's rules before Stage D. Written by freeze.py; commit with the code.",
        "frozen": now(),
        "git": git_state(),
        "scaling_decision_sha256": sha256_file(formal.DECISION),
        "machines": {"cpu": cpu_name, "gpu": gpu_name, "size_control": rec["formal"].get("cpu_size_control")},
        "toolchain": {"cxx": (sh(["c++", "--version"]) or "").splitlines()[:1], "nvcc": a.nvcc, "driver": a.driver},
        "m": stage_a["m"], "sigma_stage_a": stage_a["sigma"],
        "cpu": {"parity": parity, "threads": threads, "warmup_trials": a.warmup_cpu,
                "drain_ms": read_json(need["drain cpu"])["drain_ms"]},
        "gpu": {"path": path, "status": a.gpu_status, "microbatch": batch, "submission": a.submission,
                "host_threads": host, "warmup_trials": a.warmup_gpu, "drain_ms": read_json(need["drain gpu"])["drain_ms"]},
        "projections": proj,
        "timeouts_s": timeouts,
        "reference_sha256": reference,
        "inputs_sha256": {k: sha256_file(p) for k, p in need.items()},
    }
    out = fd / "protocol-freeze.json"
    write_json(out, freeze)
    print(f"wrote {out}; commit it with the frozen code before Stage D")


if __name__ == "__main__":
    main()

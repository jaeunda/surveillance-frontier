"""Pre-freeze calibration on the formal machines (Part III). Every choice follows the registered rule; inputs and
decisions go to results/formal/<label>/calibration/<machine>/. The formal machines are those of the committed
scaling decision; this script refuses to run on any other hardware. Pilot choices are never carried over.

CPU (formal CPU machine), in this order:
    calibrate.py threads --label L [--parity SET]        threads per (series, grid): short + sustained design
    calibrate.py parity  --label L                       CPU parity rule (PC1..PC6) at the 1s-week/P24 threads -> parity.md
    calibrate.py threads --label L --parity SET          again, if the adopted set is not 'none'
    calibrate.py rates   --label L --config cpu          rates per (series, grid, L) and per T-ref cell (timeouts)
    calibrate.py drain   --label L --config cpu --warmup W
    calibrate.py drain-check --label L --config cpu --warmup W
GPU (formal GPU machine), in this order:
    calibrate.py gate    --label L --path inc|hyb        partial gate on this device (required before any GPU timing)
    calibrate.py threads --label L --gpu-host            host threads per (series, grid) for the GPU paths
    calibrate.py batch   --label L --path inc|hyb        microbatch among {64, 256, 1024, b_max} per (series, grid)
    calibrate.py path    --label L                       G-inc vs G-hyb at the representative point
    calibrate.py drain   --label L --config gpu --warmup W --submission S
    calibrate.py drain-check --label L --config gpu --warmup W --submission S
After the freeze is committed:
    calibrate.py d0      --label L                       full device gate of the frozen build (D0)
Exploratory: calibrate.py profile --label L --config cpu|gpu (perf stat counters, never timed).
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from p2 import design, formal, harness, machine, validate  # noqa: E402
from p2.common import (FORMAL, PROTOCOL, ROOT, code_fingerprint, committed_unchanged, read_json,  # noqa: E402
                       run_logged, write_json)
from p2.tasks import calib_task, point  # noqa: E402

TC = FORMAL["thread_calibration"]
GRIDS = TC["grids"]["set_task_threads"] + TC["grids"]["report_only"]
KEYS = [f"{s}/{g}" for s in PROTOCOL["series"] for g in TC["grids"]["set_task_threads"]]
SEED = PROTOCOL["stream"]["seed"]
REP = "1s-week/P24"


def setup(a, role):
    rec = formal.decision()
    name, prof = formal.profile_of(rec, role)
    ident = machine.identity(name)
    if ident["match"] is not True:
        sys.exit(f"this machine is not the formal {role} machine {name}: {ident}")
    d = formal.formal_dir(a.label) / "calibration" / name
    d.mkdir(parents=True, exist_ok=True)
    write_json(d / f"env_{a.what}{'_' + a.path if a.path else ''}.json", machine.environment(name, a.build))
    harness.ATTEMPT_LOG = d / "attempts.jsonl"
    return d, name, prof


def need(path, what):
    if not Path(path).exists():
        sys.exit(f"{path.name} missing: run {what} first")
    return read_json(path)


def cpu_cfg(d, key, warmup=0):
    t = need(d / "threads.json", "calibrate.py threads")
    par = need(d / "parity.json", "calibrate.py parity")["parity"]
    if t.get("parity", "none") != par:
        sys.exit(f"threads.json was calibrated with parity {t.get('parity')}, CPU-best uses {par}: repeat threads")
    return harness.cpu_config(t["threads"][key], parity=par, warmup=warmup)


def gpu_cfg(d, key, warmup=0, submission=1, path=None):
    p = path or need(d / "path.json", "calibrate.py path")["choice"]
    b = need(d / f"batch_{p}.json", f"calibrate.py batch --path {p}")["microbatch"][key]
    h = need(d / "threads_host.json", "calibrate.py threads --gpu-host")["threads"][key]
    return harness.gpu_config(p, b, h, submission, warmup)


def require_device_gate(d, path):
    pr = validate.gate_problems(d / f"gate_partial_{path}.json", device="cuda", full=False, path_name=path,
                                code=code_fingerprint(), gpu_name=machine.gpu_state().get("name"))
    if pr:
        sys.exit(f"path {path}: no passed partial device gate of this code on this GPU ({'; '.join(pr)}); "
                 f"run calibrate.py gate --path {path}")


def gate_meta(out, build):
    write_json(Path(out).with_suffix(".meta.json"), {"code_sha256": code_fingerprint(), "gpu_name": machine.gpu_state()
                                                     .get("name"), "binaries": machine.build_info(build)})


def cmd_gate(a):
    d, name, _ = setup(a, "gpu")
    host = harness.cpu_config(machine.topology()["logical"])["threads"]
    out = d / f"gate_partial_{a.path}.json"
    r = run_logged([a.build / "sf_gate", "--device", "cuda", "--path", a.path, "--microbatch", 64, "--threads", host,
                    "--partial", "--out", out], d / "gate.log")
    gate_meta(out, a.build)
    print(f"partial device gate {a.path}: {'pass' if r.returncode == 0 else 'FAIL'}")
    sys.exit(r.returncode)


def cmd_threads(a):
    role = "gpu" if a.gpu_host else "cpu"
    d, name, prof = setup(a, role)
    topo = machine.topology()
    cands = formal.thread_candidates({} if a.gpu_host else prof, topo)
    cfgs = [harness.cpu_config(n, parity=a.parity) for n in cands]
    out, recs = {}, []
    for i, (series, grid) in enumerate((s, g) for s in PROTOCOL["series"] for g in GRIDS):
        r = design.choose(series, grid, cfgs, d / "calibration.jsonl", build=a.build, label="threads", seed=SEED + i)
        recs.append(r)
        if grid in TC["grids"]["set_task_threads"]:
            if not r.get("config"):
                sys.exit(f"threads {series} {grid}: every candidate failed ({r.get('failed')})")
            out[f"{series}/{grid}"] = r["config"]["threads"]
        print(f"threads {series} {grid}: {r.get('choice')} (flip {r.get('flip')})", flush=True)
    write_json(d / ("threads_host.json" if a.gpu_host else "threads.json"),
               {"machine": name, "candidates": cands, "parity": a.parity, "threads": out, "records": recs,
                "choice_differs_across_grids_or_series": len({r.get("choice") for r in recs}) > 1})


def cmd_parity(a):
    d, name, _ = setup(a, "cpu")
    t = need(d / "threads.json", "calibrate.py threads")["threads"][REP]
    adopted = design.parity(harness.cpu_config(t), d / "calibration.jsonl", d / "parity.md", build=a.build,
                            min_gain=FORMAL["verdicts"]["parity_min_gain"], seed=SEED)
    write_json(d / "parity.json", {"machine": name, "threads": t, "parity": adopted})
    print(f"CPU-best parity set: {adopted}; repeat calibrate.py threads with --parity {adopted} if it is not 'none'")


def cmd_rates(a):
    d, name, _ = setup(a, a.config)
    if a.config == "gpu":
        require_device_gate(d, need(d / "path.json", "calibrate.py path")["choice"])
    cfg_of = (lambda k: cpu_cfg(d, k)) if a.config == "cpu" else (lambda k: gpu_cfg(d, k))
    log = d / f"rates_{a.config}.jsonl"
    for series in PROTOCOL["series"]:
        for grid in TC["grids"]["set_task_threads"]:
            cfg = cfg_of(f"{series}/{grid}")
            for L in PROTOCOL["tasks"]["T1-curve"]["lengths"]:
                for rep in range(3):
                    design.bench_point(calib_task(series, grid, L), cfg, 5, log, build=a.build, label="rate", rep=rep)
    cfg = cfg_of(REP)
    for c in range(15):
        for rep in range(3):
            design.bench_point(point(f"pilot/tref-cell_c{c}"), cfg, 5, log, build=a.build, label="tref-rate", rep=rep)
    print(f"rates written: {log}")


def cmd_batch(a):
    d, name, _ = setup(a, "gpu")
    require_device_gate(d, a.path)
    host = need(d / "threads_host.json", "calibrate.py threads --gpu-host")["threads"]
    out, recs, bmax = {}, [], {}
    for i, (series, grid) in enumerate((s, g) for s in PROTOCOL["series"] for g in GRIDS):
        h = host.get(f"{series}/{grid}", host[REP])
        # b_max of this point on this device, then integer candidates only (the frozen batch never says 'auto')
        probe = design.bench_point(calib_task(series, grid, 32), harness.gpu_config(a.path, "auto", h), 1,
                                   d / "calibration.jsonl", build=a.build, label=f"batch-{a.path}-probe")
        b_max = (probe.get("device") or {}).get("b_max")
        if not b_max:
            sys.exit(f"batch {a.path} {series} {grid}: b_max probe failed ({probe.get('status')})")
        bmax[f"{series}/{grid}"] = b_max
        cands = sorted({b for b in (64, 256, 1024) if b <= b_max} | {b_max})
        cfgs = [harness.gpu_config(a.path, b, h) for b in cands]
        r = design.choose(series, grid, cfgs, d / "calibration.jsonl", build=a.build, label=f"batch-{a.path}",
                          seed=SEED + 100 + i)
        recs.append(r)
        if grid in TC["grids"]["set_task_threads"]:
            if not r.get("config"):
                sys.exit(f"batch {a.path} {series} {grid}: every candidate failed")
            out[f"{series}/{grid}"] = r["config"]["microbatch"]
        print(f"batch {a.path} {series} {grid}: {r.get('choice')} (b_max {b_max})", flush=True)
    write_json(d / f"batch_{a.path}.json", {"machine": name, "path": a.path, "microbatch": out, "b_max": bmax,
                                             "host_threads": host, "records": recs})


def cmd_path(a):
    d, name, _ = setup(a, "gpu")
    cfgs = {}
    for p in ("inc", "hyb"):
        try:
            require_device_gate(d, p)
        except SystemExit as e:
            print(f"path {p} not a candidate: {e}")
            continue
        cfgs[p] = gpu_cfg(d, REP, path=p)
    if not cfgs:
        sys.exit("no GPU path passed the partial device gate: GPU side 'not measured'")
    rec = design.select_path(cfgs, d / "calibration.jsonl", build=a.build)
    write_json(d / "path.json", {"machine": name, **rec, "candidates": sorted(cfgs)})
    print(f"path: {rec['choice']} (rates {rec['rates']})")


def drain_cfg(d, a):
    if a.config == "cpu":
        return cpu_cfg(d, REP, a.warmup)
    require_device_gate(d, need(d / "path.json", "calibrate.py path")["choice"])
    return gpu_cfg(d, REP, a.warmup, a.submission)


def cmd_drain(a):
    d, name, _ = setup(a, a.config)
    cfg = drain_cfg(d, a)
    rec = design.drain(point("calib/T-ref-est_drain"), cfg, d / "calibration.jsonl", HERE / "results" / "_drain_runs",
                       build=a.build)
    write_json(d / f"drain_{a.config}.json", {"machine": name, "config": cfg, **rec})
    print(f"drain margin {a.config}: {rec['drain_ms']} ms")


def cmd_drain_check(a):
    """With the calibrated margin, 2 runs at every H4 budget (stream purpose calibration): on time or late, and the
    prefix structure (G, N_c). Reported in the freeze; the margin itself is not changed by it."""
    d, name, _ = setup(a, a.config)
    cfg = drain_cfg(d, a)
    dr = need(d / f"drain_{a.config}.json", "calibrate.py drain")
    if dr["config_key"] != harness.config_key(cfg):
        sys.exit(f"drain was calibrated with {dr['config_key']}, not {harness.config_key(cfg)}")
    from p2.common import new_attempt_dir
    work = new_attempt_dir(HERE / "results" / "_drain_runs")
    rows = []
    for B in PROTOCOL["tasks"]["T-ref-est"]["budgets_s"]:
        for k in range(2):
            out = work / f"check_B{B}_{k}"
            r = harness.run_cold(point("calib/T-ref-est_drain"), out, cfg, {"kind": "drain-check", "budget_s": B,
                                                                            "rep": k}, build=a.build, budget_s=B,
                                 drain_ms=dr["drain_ms"], replicate=100 + k, timeout_s=2 * B + 120)
            row = {"B": B, "rep": k, "status": r["status"], "e2e_s": r.get("e2e_s")}
            try:
                pre = json.loads((out / "prefix.json").read_text())
                C = len(pre["cells"])
                row["structure_ok"] = all(c["N"] == pre["G"] // C + (1 if i < pre["G"] % C else 0)
                                          for i, c in enumerate(pre["cells"]))
                row["G"] = pre["G"]
            except (OSError, ValueError, KeyError):
                row["structure_ok"] = False
            rows.append(row)
            print(f"drain check B={B}: {row['status']} G={row.get('G')} structure {row['structure_ok']}", flush=True)
    write_json(d / f"drain_check_{a.config}.json", {"machine": name, "drain_ms": dr["drain_ms"],
                                                     "config_key": harness.config_key(cfg), "runs": rows,
                                                     "late": sum(r["status"] != "ok" for r in rows),
                                                     "structure_ok": all(r["structure_ok"] for r in rows)})


def cmd_profile(a):
    d, name, _ = setup(a, a.config)
    cfg = cpu_cfg(d, REP) if a.config == "cpu" else gpu_cfg(d, REP)
    write_json(d / f"profile_{a.config}.json", {"machine": name, "profile": design.profile(cfg, d / "profile.jsonl",
                                                                                            build=a.build)})


def cmd_d0(a):
    fd = formal.formal_dir(a.label)
    fz = fd / "protocol-freeze.json"
    if not fz.exists() or not committed_unchanged(fz):
        sys.exit("D0 runs on the committed freeze")
    freeze = read_json(fz)
    if freeze["code_sha256"] != code_fingerprint():
        sys.exit("the code differs from the frozen code")
    d, name, _ = setup(a, "gpu")
    g = freeze["gpu"]
    out = fd / "d0" / "gate_cuda.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(fd / "d0" / "env.json", machine.environment(name, a.build))
    r = run_logged([a.build / "sf_gate", "--device", "cuda", "--path", g["path"], "--microbatch", g["microbatch"][REP],
                    "--threads", g["host_threads"][REP], "--submission", g["submission"], "--starts",
                    FORMAL["gate"]["starts_per_combination"], "--out", out], fd / "d0" / "gate.log")
    gate_meta(out, a.build)
    pr = validate.gate_problems(out, device="cuda", full=True, path_name=g["path"], code=freeze["code_sha256"])
    print(f"D0 full gate: {'pass' if r.returncode == 0 and not pr else 'FAIL'} {pr}; commit d0/ before Stage D")
    sys.exit(0 if r.returncode == 0 and not pr else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["gate", "threads", "parity", "rates", "batch", "path", "drain", "drain-check",
                                     "profile", "d0"])
    ap.add_argument("--label", required=True)
    ap.add_argument("--build", type=Path, default=ROOT / "build")
    ap.add_argument("--parity", default="none")
    ap.add_argument("--path", choices=["inc", "hyb"])
    ap.add_argument("--config", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--submission", type=int, default=1)
    ap.add_argument("--gpu-host", action="store_true", help="threads: calibrate the GPU machine's host")
    a = ap.parse_args()
    a.build = a.build.resolve()
    if a.what in ("batch", "gate") and not a.path:
        sys.exit("--path is required")
    {"gate": cmd_gate, "threads": cmd_threads, "parity": cmd_parity, "rates": cmd_rates, "batch": cmd_batch,
     "path": cmd_path, "drain": cmd_drain, "drain-check": cmd_drain_check, "profile": cmd_profile,
     "d0": cmd_d0}[a.what](a)


if __name__ == "__main__":
    main()

"""Task files (SPEC §2) generated from protocol.json. The runner reads only these; nothing about a task is decided in
C++. Generation is deterministic; the files are committed under tasks/.

    main/      T1-curve, T2-policy-map (both series), T-ref, T-ref-est
    calib/     <series>_<grid>_L<L>: steady-state points at q = 16 (threads, batch, rates); T-ref-est drain calibration
    parity/    the three parity calibration points
    tdiag/     T-diag points; calibration split at N in calibration_n, held out at N = 604,800
    pilot/     tref-cell_c<c>: one T-ref cell each (reference-build projection); T-ref-est on a 75,600-bar prefix
"""

import itertools
from pathlib import Path

from .common import PROTOCOL, ROOT, TASKS, sha256_file

CODES = PROTOCOL["stream"]["task_codes"]
TD = PROTOCOL["t_diag"]
WEEK_BARS = PROTOCOL["series"]["1s-week"]["bars"]


def main_task(task, series):
    return TASKS / "main" / f"{task}_{series}.json"


def calib_task(series, grid, L):
    return TASKS / "calib" / f"{series}_{grid}_L{L}.json"


def point(name):
    """A task by its short name relative to tasks/, e.g. 'calib/1s-week_P24_L32'."""
    return TASKS / f"{name}.json"


def is_heldout(task_json):
    td = task_json.get("tdiag")
    return bool(td) and td["n"] in TD["heldout_n"]


def series_entry(name, prefix=0):
    s = PROTOCOL["series"][name]
    path = ROOT / s["file"]
    entry = {"name": name, "path": s["file"], "sha256": sha256_file(path) if path.exists() else "",
             "code": PROTOCOL["stream"]["series_codes"][name], "bars": s["bars"], "prefix": 0}
    if prefix and prefix != s["bars"]:
        code = PROTOCOL["stream"]["series_codes"][f"week-prefix-{prefix}"]
        entry.update({"name": f"week-prefix-{prefix}", "code": code, "bars": prefix, "prefix": prefix})
    return entry


def policies(spec):
    """A grid name or an explicit list -> policy dicts (K x s_min x gap nesting, as sf::policy_grid)."""
    if isinstance(spec, str):
        g = PROTOCOL["policy_grids"][spec]
        spec = [{"K": k, "s_min": s, "episode_gap": e}
                for k, s, e in itertools.product(g["K"], g["s_min"], g["episode_gap"])]
    return [{"K": p["K"], "s_min": p["s_min"], "episode_gap": p["episode_gap"], "top_k": 20} for p in spec]


def task(name, code, series, pols, cells, trials, purpose="measured", max_length=None, **extra):
    t = {"task": name, "task_code": code, "series": series, "window_ladder": PROTOCOL["window_ladder"],
         "policies": pols, "max_length": max_length or max(c["L"] for c in cells), "cells": cells,
         "trials": trials, "stream": {"purpose": purpose, "replicate": 0}}
    t.update(extra)
    return t


def grid_cells(spec):
    return [{"L": L, "q": q, "kind": "both", "stream_cell": iL * len(spec["strengths"]) + iq}
            for iL, L in enumerate(spec["lengths"]) for iq, q in enumerate(spec["strengths"])]


def one_cell(L, q):
    return [{"L": L, "q": q, "kind": "both", "stream_cell": 0}]


def bench_task(series, pols, L, q, **extra):
    """A single-cell steady-state point (stream purpose calibration, never a measured trial)."""
    return task("T-diag", CODES["T-diag"], series, pols, one_cell(L, q), {"order": "round-robin"},
                purpose="calibration", max_length=256, **extra)


def generate():
    out = {}
    T = PROTOCOL["tasks"]
    for name in ("T1-curve", "T2-policy-map"):
        spec = T[name]
        for s in spec["series"]:
            out[f"main/{name}_{s}.json"] = task(name, CODES[name], series_entry(s), policies(spec["policies"]),
                                                grid_cells(spec), {"order": "cell-major", "per_cell": spec["trials_per_cell"]},
                                                alpha=spec["alpha"], half_width=spec["half_width"],
                                                coverage=spec["coverage"])
    ref, est = T["T-ref"], T["T-ref-est"]
    ref_cells = grid_cells(ref)
    # T-ref draws nothing from the stream; the T-ref-est task code only keys its spot and oracle ids
    out["main/T-ref_1s-week.json"] = task("T-ref", CODES["T-ref-est"], series_entry("1s-week"),
                                          policies(ref["policies"]), ref_cells, {"order": "enumerate"},
                                          trials_total=ref["trials_total"])
    est_fields = dict(checkpoints=est["checkpoints"], alpha_per_interval=est["alpha_per_interval"],
                      estimate_if_n0=est["estimate_if_n0"], budgets_s=est["budgets_s"])
    out["main/T-ref-est_1s-week.json"] = task("T-ref-est", CODES["T-ref-est"], series_entry("1s-week"),
                                              policies(est["policies"]), ref_cells, {"order": "round-robin"},
                                              **est_fields)
    out["calib/T-ref-est_drain.json"] = {**out["main/T-ref-est_1s-week.json"],
                                         "stream": {"purpose": "calibration", "replicate": 0}}

    # T-diag: one cell per point; nested week prefixes (calibration) and the full week (held out)
    canon = TD["canonical"]
    for cname, cfg in TD["configs"].items():
        for n in TD["calibration_n"] + TD["heldout_n"]:
            out[f"tdiag/{cname}_n{n}.json"] = bench_task(series_entry("1s-week", n), policies(cfg["policies"]),
                                                         cfg.get("L", canon["L"]), cfg.get("q", canon["q"]),
                                                         tdiag={"config": cname, "n": n})
    for extra in TD["heldout_extra"]:
        for n in TD["heldout_n"]:
            out[f"tdiag/L_{extra['L']}_n{n}.json"] = bench_task(series_entry("1s-week", n), policies(extra["policies"]),
                                                                extra["L"], canon["q"],
                                                                tdiag={"config": f"L_{extra['L']}", "n": n})
    cross = TD["gpu_cross"]
    for sf in cross["sfloor"]:
        for L in cross["L"]:
            out[f"tdiag/cross_sfloor_{sf}_L{L}_n{cross['n']}.json"] = bench_task(
                series_entry("1s-week", cross["n"]), policies(TD["configs"][f"sfloor_{sf}"]["policies"]), L,
                canon["q"], tdiag={"config": f"cross_sfloor_{sf}_L{L}", "n": cross["n"]})

    # steady-state points per (series, grid, L) at q = 16: threads, batch, and projection rates
    tc = PROTOCOL["formal"]["thread_calibration"]
    for s in PROTOCOL["series"]:
        for grid in tc["grids"]["set_task_threads"] + tc["grids"]["report_only"]:
            for L in sorted(set(tc["short"]["lengths"]) | set(T["T1-curve"]["lengths"])):
                out[f"calib/{s}_{grid}_L{L}.json"] = bench_task(series_entry(s), policies(grid), L, tc["short"]["q"],
                                                                calibration={"series": s, "grid": grid, "L": L})
    for name, s, grid, L in [("week_P24_L32", "1s-week", "P24", 32), ("week_P1_L2", "1s-week", "P1", 2),
                             ("quarter_P24_L256", "1m-quarter", "P24", 256)]:
        out[f"parity/{name}.json"] = bench_task(series_entry(s), policies(grid), L, 16.0, parity_point=name)

    # pilot: one task per T-ref cell (rate of the reference build), and a small deadline problem
    for c, cell in enumerate(ref_cells):
        out[f"pilot/tref-cell_c{c}.json"] = bench_task(series_entry("1s-week"), policies(ref["policies"]), cell["L"],
                                                       cell["q"], tref_cell={"cell": c, "L": cell["L"], "q": cell["q"]})
    out["pilot/T-ref-est_prefix75600.json"] = task("T-ref-est", CODES["T-ref-est"], series_entry("1s-week", 75600),
                                                   policies(est["policies"]), ref_cells, {"order": "round-robin"},
                                                   purpose="calibration", **est_fields)
    return out

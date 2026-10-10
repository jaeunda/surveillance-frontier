"""Time and memory projections (README Part II, SPEC §11).

Time:    T = T_setup + sum_c n_c / r_c + T_output. r_c is a steady-state rate measured on the same machine, code,
         configuration, policy grid and length; T1/T2 points are measured at q = 16 for every task length, T-ref cells
         are measured cell by cell. Three scenarios use the slowest, median, and fastest repetition of each rate.
         A length without its own measurement takes the nearest measured length in log L and is flagged
         "extrapolated" if it lies outside the measured range, "interpolated" otherwise.
Memory:  M = M_shared + b * M_trial, against the allocation the device reported (free at context creation minus
         free after our allocations). The difference is M_temporary plus allocator granularity.
"""

import math

from .common import GRID_OF_TASK, PROTOCOL

SCENARIOS = ("conservative", "base", "optimistic")


def rate_table(bench_records):
    """{(series, grid, L): [rates]} and {(series, grid, L): [setup_s]} from bench records of calib/ points."""
    rates, setups = {}, {}
    for r in bench_records:
        if r.get("status") != "ok":
            continue
        cal = (r.get("meta") or {}).get("calibration")
        if cal is None:
            continue
        key = (cal["series"], cal["grid"], cal["L"])
        rates.setdefault(key, []).append(r["trials_per_s"])
        setups.setdefault(key, []).append(r["setup_s"])
    return rates, setups


def _pick(values, scenario):
    v = sorted(values)
    if scenario == "conservative":
        return v[0]
    if scenario == "optimistic":
        return v[-1]
    mid = len(v) // 2
    return v[mid] if len(v) % 2 else (v[mid - 1] + v[mid]) / 2


def _rate_for(rates, series, grid, L, scenario):
    lengths = sorted(Lm for (s, g, Lm) in rates if (s, g) == (series, grid))
    if not lengths:
        return None, "missing"
    if L in lengths:
        return _pick(rates[(series, grid, L)], scenario), "measured"
    near = min(lengths, key=lambda m: abs(math.log(m / L)))
    flag = "extrapolated" if L < lengths[0] or L > lengths[-1] else "interpolated"
    return _pick(rates[(series, grid, near)], scenario), flag


def project_fixed(task, series, rates, setups=None, output_s=0.0, warm=False):
    """Projected E2E of T1/T2 per scenario; setup from the bench setup times unless warm."""
    spec = PROTOCOL["tasks"][task]
    grid = GRID_OF_TASK[task]
    out, flags = {}, set()
    for sc in SCENARIOS:
        loop = 0.0
        for L in spec["lengths"]:
            r, flag = _rate_for(rates, series, grid, L, sc)
            if r is None:
                return {"status": "missing rates", "series": series, "grid": grid}
            flags.add(flag)
            loop += len(spec["strengths"]) * spec["trials_per_cell"] / r
        setup = 0.0
        if not warm and setups:
            vals = [max(v) for (s, g, _), v in setups.items() if (s, g) == (series, grid)]
            setup = max(vals) if vals else 0.0
        out[sc] = setup + loop + output_s
    return {"status": "ok", "seconds": out, "rate_flags": sorted(flags), "warm": warm}


def project_tref(cell_rates, setup_s=0.0, output_s=0.0):
    """T-ref build time from per-cell rates {cell: [rates]}: n_c = N - L + 1 for every cell."""
    spec = PROTOCOL["tasks"]["T-ref"]
    bars = PROTOCOL["series"]["1s-week"]["bars"]
    cells = [(L, q) for L in spec["lengths"] for q in spec["strengths"]]
    missing = [c for c in range(len(cells)) if not cell_rates.get(c)]
    if missing:
        return {"status": "missing cells", "missing": missing}
    out = {}
    per_cell = {}
    for sc in SCENARIOS:
        total = 0.0
        for c, (L, _) in enumerate(cells):
            t = (bars - L + 1) / _pick(cell_rates[c], sc)
            total += t
            if sc == "base":
                per_cell[c] = t
        out[sc] = setup_s + total + output_s
    n = sum(bars - L + 1 for L, _ in cells)
    assert n == spec["trials_total"], (n, spec["trials_total"])
    return {"status": "ok", "seconds": out, "per_cell_base_s": per_cell, "trials": n}


def memory_check(device):
    """Planned vs reported device memory of one GPU configuration (parsed device_info of a run or bench)."""
    need = ("shared_bytes", "working_set_bytes_per_trial", "microbatch", "free_at_context_bytes",
            "free_after_alloc_bytes")
    if not device or any(k not in device for k in need):
        return None
    planned = device["shared_bytes"] + device["microbatch"] * device["working_set_bytes_per_trial"]
    reported = device["free_at_context_bytes"] - device["free_after_alloc_bytes"]
    return {"microbatch": device["microbatch"], "b_max": device.get("b_max"), "planned_bytes": planned,
            "reported_bytes": reported, "difference_bytes": reported - planned,
            "free_at_context_bytes": device["free_at_context_bytes"], "total_bytes": device.get("total_bytes"),
            "headroom_fraction": 1 - reported / device["free_at_context_bytes"] if device["free_at_context_bytes"] else None}

"""Interval and category rules of the formal comparison (README "Verdict rules"), and descriptive summaries."""

import math
import statistics

import numpy as np
from scipy.stats import t as student_t

from .common import FORMAL

DELTA = FORMAL["verdicts"]["margin"]


def paired_interval(ratios, level=0.90):
    """Geometric mean of the ratios and exp(mean log r +- t(1 - (1 - level)/2, m - 1) SD / sqrt(m))."""
    logs = np.log(np.asarray(ratios, float))
    m = len(logs)
    mean = float(np.mean(logs))
    if m < 2:
        return math.exp(mean), float("nan"), float("nan")
    sd = float(np.std(logs, ddof=1))
    half = float(student_t.ppf(1 - (1 - level) / 2, m - 1)) * sd / math.sqrt(m)
    return math.exp(mean), math.exp(mean - half), math.exp(mean + half)


def category(lo, hi, censored_cpu=False, censored_gpu=False):
    """Category of a ratio interval, > 1 favouring the GPU (H3: T_cpu / T_gpu; H4: RMSE_cpu / RMSE_gpu).

    A censored (timed-out) run is a lower bound on its own device's time, which can only flatter that device: an
    advantage of the censored device, or equivalence, is not concluded; the other device's advantage stands.
    """
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return "inconclusive", None
    if lo >= DELTA:
        cat = "GPU advantage"
    elif hi <= 1 / DELTA:
        cat = "CPU advantage"
    elif lo > 1 / DELTA and hi < DELTA:
        cat = "equivalent"
    else:
        cat = "inconclusive"
    note = None
    if cat == "inconclusive" and (lo > 1 or hi < 1):
        note = f"direction {'GPU' if lo > 1 else 'CPU'}, size unresolved"
    if (cat == "GPU advantage" and censored_gpu) or (cat == "CPU advantage" and censored_cpu) or \
            (cat == "equivalent" and (censored_cpu or censored_gpu)):
        note = f"{cat} not concluded: censored {'GPU' if censored_gpu else 'CPU'} runs"
        cat = "inconclusive"
    return cat, note


def describe(values):
    """Median and range of a list (pilot sessions are described, never turned into an interval)."""
    v = [x for x in values if x is not None]
    if not v:
        return None
    return {"n": len(v), "median": statistics.median(v), "min": min(v), "max": max(v)}

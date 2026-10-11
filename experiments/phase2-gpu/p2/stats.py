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

    Censoring (amendment 2026-10-11, before any measurement): a run stopped at its timeout enters the ratio at exactly
    the timeout, but the paired-t interval computed from such substituted values is not a confidence bound for the
    true ratio (the true value can move the mean and the SD in either direction: ratios [2, 2, 2] give [2, 2], while
    a true third ratio of 200 gives [0.10, 821]). A condition with any censored run in the used pairs therefore gets
    no category; the substituted interval is reported as description only.
    """
    if censored_cpu or censored_gpu:
        who = " and ".join(d for d, c in (("CPU", censored_cpu), ("GPU", censored_gpu)) if c)
        return "censored", f"timed-out {who} runs: the substituted-timeout interval is descriptive, no verdict"
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
    return cat, note


def describe(values):
    """Median and range of a list (pilot sessions are described, never turned into an interval)."""
    v = [x for x in values if x is not None]
    if not v:
        return None
    return {"n": len(v), "median": statistics.median(v), "min": min(v), "max": max(v)}

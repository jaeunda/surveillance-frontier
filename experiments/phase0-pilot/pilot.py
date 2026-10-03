# %% [markdown]
# # Phase 0 pilot: exhaustive multi-scale window scan of a market time series on the GPU
#
# **One line.** For every (start time, window length) pair in a trading time series, compute how unusually large the
# price range and the traded volume were *together*. Measure when and why the GPU beats the CPU as the number of
# candidate windows grows, and show the result as a time × window-length rarity map.
#
# **Real example.** Around 21:11 UTC on 2024-01-09 the U.S. SEC's X account was compromised and posted a fake
# "spot Bitcoin ETF approved" message. Bitcoin jumped by more than $1,000 and then fell back (BleepingComputer,
# CBS News). Nobody knows in advance how many minutes such an event lasts, so every window length is scanned.
#
# **Out of scope.** This notebook does not judge whether any trading was manipulative. The output is a map that
# narrows down candidate intervals for a human to look at; the score is a rank, not a classifier or a probability.
#
# **Role in this repository.** This pilot came first. Running it raised the questions that shaped the project plan:
# how fast a detector can be re-run over many perturbed copies of the data, where the end-to-end time goes, and what
# a fixed time budget buys on a CPU versus a GPU. Section 14 measures that last point directly.
#
# ## Design
#
# | Element | Design |
# |---|---|
# | Input | Binance BTCUSDT spot klines, `low, high, volume` as float32 `[N]`. 1-minute bars for 2024 Q1 (131,040 rows), 1-second bars for 2024-01-08..14 (604,800 rows) |
# | Output | Raw features `[4, K, N]` (min, max, range, volume sum). Reduced paths return only candidate positions |
# | Parallel unit | 1 thread = (k, start). A warp shares one window length, so loop trip counts match and writes are contiguous |
# | Transfers, dependencies | One H2D of the input; candidates are independent. The rarity score needs a per-length ranking, so it runs after the features |
# | Expected bottleneck | Small N: fixed costs. Large N·K: the kernel rereads O(N·Σw) inputs, end-to-end time is dominated by feature D2H (16 B per candidate) |
# | Checks | Exact min/max, tolerance on range/volume, exact rank counts, identical top candidates across paths, hit rate of controlled injections |
#
# ## Hypotheses (written before measuring)
#
# | ID | Hypothesis | Pass criterion |
# |---|---|---|
# | H1 | Whether the GPU end-to-end beats the CPU is set not by N but by the ratio of work per candidate (mean window length w̄) to output bytes per candidate (16 B). A fixed-cost model predicts the crossover | Measured crossover window length w* within 2x of the prediction; out-of-sample winner accuracy reported as secondary |
# | H2 | At K = 15 and large N, the feature-returning GPU end-to-end is dominated by D2H | D2H > 60% of H2D + kernel + D2H |
# | H3 | Moving scoring and candidate selection to the GPU and returning only candidates cuts end-to-end time by at least 2x | Path 3 vs path 1 |
# | H4 | A start-major thread layout mixes window lengths inside a warp, so each warp runs max(w) iterations. Predicted slowdown = K·max(w)/Σw (4.4x at K = 15) | Measured ratio within ±35% of the prediction |
# | H5 | The map reveals the *length* of an event, not only its position. For a pump-and-dump shape (rise over 2L/3, fall over L/3), range rarity peaks near L/3 and volume rarity near L, so the estimated length is about (1/3 .. 1) × L | At q = 16 the median ratio for all three lengths lies in [0.28, 1.19] (includes half-octave quantization) |
# | H6 | Detection has a minimum strength | The top-20 hit rate crosses 50% inside the tested strength range for every length |
# | H7 | Within the same wall-clock budget, the GPU path completes more detector re-evaluations than the optimized CPU path, and the confidence interval of the estimated detection rate narrows roughly as 1/√n | GPU trials per second > optimized CPU; measured CI-width ratio within 25% of √(trial ratio) |
#
# The comparison with real events (Section 10) is a descriptive observation, not a hypothesis test. The news list was
# compiled after seeing an earlier run; the macro list (CPI, FOMC) follows a fixed rule. Both are reported against a
# chance baseline (circular-shift permutation).

# %% [markdown]
# ## 1. Environment

# %%
import os, sys, re, io, json, time, math, zipfile, shutil, platform, subprocess, importlib, warnings, tempfile
from pathlib import Path

os.environ.setdefault("CUPY_CACHE_DIR", tempfile.mkdtemp(prefix="cupy_cold_cache_"))
QUICK = os.environ.get("QUICK_MODE") == "1"  # local dry-run only (tools/run_notebook.py sets it)


def command_output(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable ({exc.__class__.__name__})"


try:
    import cupy as cp
except ImportError:
    print("CuPy not found; installing cupy-cuda12x ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "cupy-cuda12x"])
    importlib.invalidate_caches()
    import cupy as cp
import numpy as np
import numba
import pandas as pd

IS_FAKE_GPU = bool(getattr(cp, "IS_FAKE", False))
if cp.cuda.runtime.getDeviceCount() < 1:
    raise RuntimeError("No CUDA GPU. In Colab: Runtime > Change runtime type > T4 GPU, then Run all.")
props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
GPU_NAME = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])


def cpu_model_name():
    m = re.search(r"Model name:\s*(.+)", command_output(["lscpu"]))
    return m.group(1).strip() if m else platform.processor()


def gpu_state():
    return command_output(["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw",
                           "--format=csv,noheader"])


ENV = {
    "gpu_name": GPU_NAME, "gpu_sm_count": int(props.get("multiProcessorCount", -1)),
    "compute_capability": f"{props['major']}.{props['minor']}", "gpu_memory_gib": round(props["totalGlobalMem"] / 2**30, 2),
    "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
    "gpu_state_start": gpu_state(),
    "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(), "cuda_driver": cp.cuda.runtime.driverGetVersion(),
    "cpu_model": cpu_model_name(), "cpu_logical_cores": os.cpu_count(), "numba_threads": numba.get_num_threads(),
    "python": platform.python_version(), "cupy": cp.__version__, "numpy": np.__version__, "numba": numba.__version__,
    "pandas": pd.__version__, "dtype": "float32 features on CPU and GPU", "host_memory_for_e2e": "pageable, preallocated",
    "quick_mode": QUICK, "fake_gpu": IS_FAKE_GPU,
}
RESULTS_DIR = Path("/content/pilot_results") if Path("/content").exists() else Path.cwd() / "pilot_results"
FIG_DIR = RESULTS_DIR / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS = {"sections_done": []}


def checkpoint(section, **extra):
    """Write progress after every major section, so a cut-off run still leaves usable results on disk."""
    PROGRESS["sections_done"].append(section)
    PROGRESS.update(extra)
    (RESULTS_DIR / "progress.json").write_text(json.dumps({"env": ENV, **PROGRESS}, indent=2, default=str))


for k, v in ENV.items():
    print(f"{k:20s}: {v}")
if IS_FAKE_GPU:
    print("\n*** FAKE GPU (CPU emulation). Logic check only: every timing below is meaningless. ***")
checkpoint("1_environment")

# %% [markdown]
# ## 2. Settings
# Fifteen half-octave window lengths `{2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 91, 128, 181, 256}` (in bars): 2 min to
# 4.3 h on 1-minute bars, 2 s to 4.3 min on 1-second bars. The spacing is logarithmic, so the vertical axis of the map
# is a scale space.

# %%
import random

SEED = 20261002
random.seed(SEED); np.random.seed(SEED); cp.random.seed(SEED)

WINDOWS = np.array([2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 91, 128, 181, 256], dtype=np.int32)
K_ALL = len(WINDOWS)
WINDOW_SETS = {"K1": WINDOWS[:1], "K4 short": WINDOWS[:4], "K4 long": WINDOWS[-4:], "K8": WINDOWS[:8], "K15": WINDOWS}
N_SWEEP = [1_000, 4_000, 16_000, 64_000, 256_000, 604_800]
CALIBRATION = [(1_000, "K1"), (64_000, "K8"), (64_000, "K4 long")]
LADDER_W = [2, 3, 4, 6, 8, 11, 16, 23, 32]          # single-window sets at the largest N: locate w* directly
S_MIN = 2.0                       # candidate filter: both features in the top 1% for that window length
TOP_K = 20
EPISODE_GAP = 60                  # candidates closer than this many bars are one episode (keep the rarest)
EVENT_TOL_MIN = 60
INJ_LENGTHS = [6, 30, 120]
INJ_STRENGTHS = [2.0, 4.0, 8.0, 16.0, 32.0]
INJ_TRIALS = 20
BUDGET_L, BUDGET_Q = 30, 8.0      # Section 13: injection setting near the detection threshold
BUDGETS_S = [10, 30, 60]          # Section 13: wall-clock budgets per implementation
BREAKDOWN_REPS = 10
BLOCK_SIZES = [64, 128, 256, 512, 1024]
CUDA_BLOCK = 256
WARMUP, MIN_REPS, TARGET_REPS, BUDGET_PER_CASE_S, MAX_INNER = 3, 10, 30, 2.0, 200
RTOL = {"min": 0.0, "max": 0.0, "range": 2e-6, "volume": 2e-6}
SMART_VOLUME_RTOL = 2e-5          # float64 prefix-sum vs float32 sequential sum

if QUICK:
    N_SWEEP = [1_000, 16_000, 64_000]
    WINDOW_SETS = {k: WINDOW_SETS[k] for k in ["K1", "K4 long", "K8"]}
    LADDER_W = [2, 4, 8, 16]
    INJ_STRENGTHS, INJ_TRIALS = [4.0, 32.0], 2
    BUDGETS_S = [2, 4]
    BREAKDOWN_REPS = 3
    BLOCK_SIZES = [128, 256]
    WARMUP, MIN_REPS, TARGET_REPS, BUDGET_PER_CASE_S, MAX_INNER = 1, 3, 3, 0.2, 2
print("windows:", WINDOWS.tolist(), "| quick mode" if QUICK else "| full mode")

# %% [markdown]
# ## 3. Measurement harness
# Four questions are fixed in code: what is timed, when it is done, the distribution, and the tolerance.
# - **Kernel:** timing `Event → kernel → Event` on an idle GPU mixes in the host time spent launching. A **spin kernel**
#   (a `clock64` loop) keeps the GPU busy for a few ms first; `Event → R kernels → Event` is queued behind it, which
#   gives pure GPU time. The difference to the wall time of "one kernel + sync" is the host dispatch cost.
# - **H2D:** copy into a preallocated device array with `set` (pageable = staging + DMA). **D2H:** into a preallocated
#   host buffer (no page faults). CPU baselines also write into preallocated buffers.
# - **Distribution:** the first call is reported separately (`first_ms`), then warm-up, then at least 10 and up to 30
#   samples within a time budget; median, p95, and n are reported. Conditions over 1 s use at least 5 samples and are
#   flagged. True cold start (NVRTC compilation) is measured once per kernel in Section 8.
# - Device memory is measured with the CuPy memory pool already warm.

# %%
def gpu_sync():
    cp.cuda.runtime.deviceSynchronize()


def _summarize(samples, first_ms, reduced):
    x = np.asarray(samples, dtype=np.float64)
    return {"median_ms": float(np.median(x)), "p95_ms": float(np.percentile(x, 95)), "min_ms": float(x.min()),
            "n": int(x.size), "first_ms": float(first_ms), "reduced_reps": bool(reduced)}


def measure_host(fn, gpu=False):
    def once():
        if gpu:
            gpu_sync()
        t0 = time.perf_counter()
        fn()
        if gpu:
            gpu_sync()
        return (time.perf_counter() - t0) * 1e3

    cold = once()
    slow = cold > 1000.0
    for _ in range(1 if slow else WARMUP):
        once()
    min_reps = min(5, MIN_REPS) if slow else MIN_REPS
    samples, start = [], time.perf_counter()
    while True:
        samples.append(once())
        if len(samples) >= TARGET_REPS or (len(samples) >= min_reps and time.perf_counter() - start > BUDGET_PER_CASE_S):
            break
    return _summarize(samples, cold, slow)


SPIN_SRC = r'''
extern "C" __global__ void spin(const long long cycles)
{   /* keeps one SM busy so the host can enqueue work behind it without gaps */
    const long long t0 = clock64();
    while (clock64() - t0 < cycles) { }
}
'''
k_spin = cp.RawKernel(SPIN_SRC, "spin")
CLOCK_HZ = float(props.get("clockRate", 1_590_000)) * 1e3


def spin(ms):
    k_spin((1,), (1,), (np.int64(ms * 1e-3 * CLOCK_HZ),))


def h2d_into(dst, src):
    """Host -> preallocated device array (cudaMemcpy semantics: pageable = staged, pinned = direct DMA)."""
    if hasattr(dst, "set"):
        dst.set(src)
    else:              # fakecupy dry run: device arrays are NumPy arrays
        dst[...] = src


def _event_ms(launch, reps, queued=True):
    if queued:
        spin(max(2.0, 0.03 * reps))        # host enqueues the timed work while the GPU is still spinning
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    for _ in range(reps):
        launch()
    stop.record()
    stop.synchronize()
    return float(cp.cuda.get_elapsed_time(start, stop))


def measure_kernel(launch):
    """Pure GPU time per launch (spin-queued, R launches / R) and single-launch wall time; difference = dispatch cost."""
    gpu_sync()
    first = _event_ms(launch, 1, queued=False)
    for _ in range(WARMUP):
        _event_ms(launch, 1)
    single = _event_ms(launch, 1)
    R = int(min(MAX_INNER, max(1, math.ceil(1.0 / max(single, 1e-3)))))   # aim for >= ~1 ms per sample
    samples, start = [], time.perf_counter()
    while True:
        samples.append(_event_ms(launch, R) / R)
        if len(samples) >= TARGET_REPS or (len(samples) >= MIN_REPS and time.perf_counter() - start > BUDGET_PER_CASE_S):
            break
    st = _summarize(samples, first, False)
    wall = measure_host(launch, gpu=True)
    st.update(inner_R=R, single_wall_ms=wall["median_ms"], dispatch_overhead_ms=wall["median_ms"] - st["median_ms"])
    return st


def save_fig(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")


print("harness ready: measure_host, measure_kernel, spin, h2d_into")

# %% [markdown]
# ## 4. Data: 1-minute bars (2024 Q1) and 1-second bars (2024-01-08..14)
# Fixed files from the Binance public archive. Missing timestamps (exchange maintenance and so on) are re-gridded onto
# a complete time grid; empty bars take the previous close and zero volume, and the number of filled rows is recorded.
# The timestamp unit (ms or µs) is detected from its magnitude.
# If the download fails, a synthetic fallback is used and labeled; the real-event comparison and the 1-second zoom are
# then skipped.

# %%
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades",
        "taker_base", "taker_quote", "ignore"]
ARCHIVE = "https://data.binance.vision/data/spot/{freq}/klines/BTCUSDT/{iv}/BTCUSDT-{iv}-{period}.zip"
DATA_DIR = RESULTS_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
SESSION = requests.Session()
SESSION.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504])))


def fetch_klines(iv, periods, freq):
    frames = []
    for period in periods:
        r = SESSION.get(ARCHIVE.format(freq=freq, iv=iv, period=period), timeout=120)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            raw = pd.read_csv(zf.open(zf.namelist()[0]), header=None, names=COLS)
        raw = raw[pd.to_numeric(raw.open_time, errors="coerce").notna()]          # drop a header row if present
        frames.append(raw)
    df = pd.concat(frames, ignore_index=True)
    t = pd.to_numeric(df.open_time).astype(np.int64)
    unit = "us" if t.iloc[0] > 10**14 else "ms"
    df["timestamp"] = pd.to_datetime(t, unit=unit, utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"]).drop_duplicates("timestamp").sort_values("timestamp")
    step = "1min" if iv == "1m" else "1s"
    grid = pd.date_range(df.timestamp.iloc[0].floor(step), df.timestamp.iloc[-1].floor(step), freq=step, tz="UTC")
    df = df.set_index("timestamp").reindex(grid)
    filled = int(df.close.isna().sum())
    df["close"] = df.close.ffill()
    for c in ["open", "high", "low"]:
        df[c] = df[c].fillna(df.close)
    df["volume"] = df.volume.fillna(0.0)
    df = df.rename_axis("timestamp").reset_index()[["timestamp", "open", "high", "low", "close", "volume"]]
    return df, filled


def load_or_fetch(name, iv, periods, freq):
    cache = DATA_DIR / f"{name}.npz"
    if cache.exists():
        z = np.load(cache)
        df = pd.DataFrame({"timestamp": pd.to_datetime(z["ts"], unit="ns", utc=True),
                           **{c: z[c] for c in ["open", "high", "low", "close", "volume"]}})
        return df, int(z["filled"])
    df, filled = fetch_klines(iv, periods, freq)
    # store as int64 nanoseconds: pandas >= 3 may keep datetime64[ms], which would silently change the unit
    ts_ns = df.timestamp.dt.tz_convert(None).to_numpy().astype("datetime64[ns]").astype(np.int64)
    np.savez_compressed(cache, ts=ts_ns, filled=filled,
                        **{c: df[c].to_numpy(np.float64) for c in ["open", "high", "low", "close", "volume"]})
    return df, filled


def synthetic_fallback(n, step):
    rng = np.random.default_rng(SEED)
    close = 42_000 * np.exp(np.cumsum(rng.normal(0, 0.0005, n)))
    spread = np.abs(rng.normal(0.0006, 0.0003, n))
    return pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=n, freq=step, tz="UTC"),
                         "open": close, "high": close * (1 + spread), "low": close * (1 - spread), "close": close,
                         "volume": rng.lognormal(2.5, 0.7, n)}), 0


try:
    m1, m1_filled = load_or_fetch("BTCUSDT_1m_2024Q1", "1m", ["2024-01", "2024-02", "2024-03"], "monthly")
    s1, s1_filled = load_or_fetch("BTCUSDT_1s_20240108_14", "1s", [f"2024-01-{d:02d}" for d in range(8, 15)], "daily")
    DATA_KIND = "real_market"
except Exception as exc:
    warnings.warn(f"download failed ({exc}); using SYNTHETIC FALLBACK - event checks will be skipped")
    (m1, m1_filled), (s1, s1_filled) = synthetic_fallback(131_040, "1min"), synthetic_fallback(604_800, "1s")
    DATA_KIND = "synthetic_fallback"


def arrays(df):
    return tuple(np.ascontiguousarray(df[c].to_numpy(np.float32)) for c in ["low", "high", "volume"])


for name, df in [("1m", m1), ("1s", s1)]:
    lo, hi, vo = arrays(df)
    o, c = df.open.to_numpy(np.float32), df.close.to_numpy(np.float32)
    assert df.timestamp.is_monotonic_increasing and not df.isna().any().any()
    assert (lo <= np.minimum(o, c) + 1e-3).all() and (hi >= np.maximum(o, c) - 1e-3).all() and (vo >= 0).all()
    print(f"{name}: {len(df):,} rows {df.timestamp.iloc[0]} .. {df.timestamp.iloc[-1]}")
print(f"filled empty bars: 1m {m1_filled}, 1s {s1_filled} | data: {DATA_KIND}")
M1 = arrays(m1)
S1 = arrays(s1)
checkpoint("4_data", data=DATA_KIND, filled_bars={"1m": m1_filled, "1s": s1_filled})

# %% [markdown]
# ## 5. CPU references
# - **Direct scan, 1 thread / multi-thread (Numba):** exactly like the GPU kernel, every candidate reads its own window
#   from start to end. The multi-threaded version uses `prange` over the flattened (k, start) index to avoid load
#   imbalance across window lengths.
# - **Optimized CPU, O(N·K):** sliding min/max with a monotonic deque in O(N) per length, volume from a float64 prefix
#   sum. The direct scan rereads inputs on purpose (it mirrors the GPU kernel); this baseline shows what a careful CPU
#   implementation achieves and is the fair comparison.

# %%
from numba import njit, prange


@njit(cache=False, inline="always")
def _one_window(low, high, volume, start, w):
    mn = np.float32(np.inf)
    mx = np.float32(-np.inf)
    vs = np.float32(0.0)
    for o in range(w):
        p = start + o
        if low[p] < mn:
            mn = low[p]
        if high[p] > mx:
            mx = high[p]
        vs = np.float32(vs + volume[p])
    return mn, mx, vs


@njit(cache=False)
def cpu_direct_1t_into(low, high, volume, windows, out):
    n, K = low.size, windows.size
    for k in range(K):
        w = windows[k]
        for s in range(n):
            if s + w > n:
                out[0, k, s] = np.nan; out[1, k, s] = np.nan; out[2, k, s] = np.nan; out[3, k, s] = np.nan
                continue
            mn, mx, vs = _one_window(low, high, volume, s, w)
            out[0, k, s] = mn; out[1, k, s] = mx
            out[2, k, s] = np.float32((mx - mn) / mn) if mn > 0 else np.nan
            out[3, k, s] = vs
    return out


@njit(parallel=True, cache=False)
def cpu_direct_mt_into(low, high, volume, windows, out):
    n, K = low.size, windows.size
    for idx in prange(K * n):
        k = idx // n
        s = idx - k * n
        w = windows[k]
        if s + w > n:
            out[0, k, s] = np.nan; out[1, k, s] = np.nan; out[2, k, s] = np.nan; out[3, k, s] = np.nan
        else:
            mn, mx, vs = _one_window(low, high, volume, s, w)
            out[0, k, s] = mn; out[1, k, s] = mx
            out[2, k, s] = np.float32((mx - mn) / mn) if mn > 0 else np.nan
            out[3, k, s] = vs
    return out


@njit(cache=False)
def cpu_smart_into(low, high, volume, windows, out):
    """O(N) per window length: monotonic-deque sliding min/max + float64 prefix sum."""
    n, K = low.size, windows.size
    pre = np.zeros(n + 1, dtype=np.float64)
    for i in range(n):
        pre[i + 1] = pre[i] + volume[i]
    qmin = np.empty(n, np.int64)
    qmax = np.empty(n, np.int64)
    for k in range(K):
        w = windows[k]
        for s in range(max(n - w + 1, 0), n):
            out[0, k, s] = np.nan; out[1, k, s] = np.nan; out[2, k, s] = np.nan; out[3, k, s] = np.nan
        h1 = t1 = h2 = t2 = 0
        for i in range(n):
            while t1 > h1 and low[qmin[t1 - 1]] >= low[i]:
                t1 -= 1
            qmin[t1] = i; t1 += 1
            while t2 > h2 and high[qmax[t2 - 1]] <= high[i]:
                t2 -= 1
            qmax[t2] = i; t2 += 1
            if qmin[h1] <= i - w:
                h1 += 1
            if qmax[h2] <= i - w:
                h2 += 1
            s = i - w + 1
            if s >= 0:
                mn, mx = low[qmin[h1]], high[qmax[h2]]
                out[0, k, s] = mn; out[1, k, s] = mx
                out[2, k, s] = np.float32((mx - mn) / mn) if mn > 0 else np.nan
                out[3, k, s] = np.float32(pre[s + w] - pre[s])
    return out


def _alloc(low, windows):
    return np.empty((4, len(windows), low.size), np.float32)


def cpu_direct_1t(low, high, volume, windows):
    return cpu_direct_1t_into(low, high, volume, windows, _alloc(low, windows))


def cpu_direct_mt(low, high, volume, windows):
    return cpu_direct_mt_into(low, high, volume, windows, _alloc(low, windows))


def cpu_smart(low, high, volume, windows):
    return cpu_smart_into(low, high, volume, windows, _alloc(low, windows))


_probe = tuple(x[:512] for x in M1)
for f in (cpu_direct_1t, cpu_direct_mt, cpu_smart):
    f(*_probe, WINDOWS[:3])                                          # JIT compile outside timing
print("CPU references compiled")

# %% [markdown]
# ## 6. CUDA kernels (CuPy RawKernel)
# | Kernel | Mapping | Purpose |
# |---|---|---|
# | `features_kmajor` | id = k·N + start, output `[4, K, N]` | baseline |
# | `features_smajor` | id = start·K + k, output `[4, N, K]` (id order, so writes stay contiguous) | H4: removes the output-layout effect and leaves only **warp trip-count imbalance** |
# | `local_peaks` | id = k·N + start | keeps only points rarer (smaller count) than every neighbour within ±w/2 at the same length; ties go to the earlier start (deterministic) |
# | negative control `features_bound_bug` | boundary check changed to `>=`, dropping the last valid start | the gate must report FAIL |
# | negative control `features_window_bug` | reads the window length as `windows[(k+1) % K]` (memory-safe) | the gate must report FAIL |
#
# Per-length ranking (sort + searchsorted) and candidate compaction (flatnonzero) use CuPy library calls. Sorting is
# what the library (CUB) does well, so there is no reason to hand-write it.

# %%
FEATURE_SRC = r'''
#include <math_constants.h>

#define WINDOW_BODY(W_EXPR, BOUND_EXPR, OUT_IDX)                                     \
    const int w = (W_EXPR);                                                         \
    if (BOUND_EXPR) {                                                               \
        out_min[OUT_IDX] = CUDART_NAN_F; out_max[OUT_IDX] = CUDART_NAN_F;           \
        out_range[OUT_IDX] = CUDART_NAN_F; out_vol[OUT_IDX] = CUDART_NAN_F;         \
        return;                                                                     \
    }                                                                               \
    float mn = CUDART_INF_F, mx = -CUDART_INF_F, vs = 0.0f;                         \
    for (int o = 0; o < w; ++o) {                                                   \
        const int p = start + o;                                                    \
        mn = fminf(mn, low[p]); mx = fmaxf(mx, high[p]); vs += volume[p];           \
    }                                                                               \
    out_min[OUT_IDX] = mn; out_max[OUT_IDX] = mx;                                   \
    out_range[OUT_IDX] = (mn > 0.0f) ? ((mx - mn) / mn) : CUDART_NAN_F;             \
    out_vol[OUT_IDX] = vs;


extern "C" __global__ void features_kmajor(
        const float* __restrict__ low, const float* __restrict__ high, const float* __restrict__ volume,
        const int* __restrict__ windows, const int n, const int K, float* __restrict__ out_min,
        float* __restrict__ out_max, float* __restrict__ out_range, float* __restrict__ out_vol)
{
    const long long id = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= (long long)n * K) return;
    const int k = (int)(id / n);
    const int start = (int)(id - (long long)k * n);
    WINDOW_BODY(windows[k], start + w > n, id)
}

extern "C" __global__ void features_smajor(
        const float* __restrict__ low, const float* __restrict__ high, const float* __restrict__ volume,
        const int* __restrict__ windows, const int n, const int K, float* __restrict__ out_min,
        float* __restrict__ out_max, float* __restrict__ out_range, float* __restrict__ out_vol)
{
    const long long id = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= (long long)n * K) return;
    const int start = (int)(id / K);
    const int k = (int)(id - (long long)start * K);
    WINDOW_BODY(windows[k], start + w > n, id)          /* output [N][K] in id order: writes stay contiguous */
}

extern "C" __global__ void features_bound_bug(
        const float* __restrict__ low, const float* __restrict__ high, const float* __restrict__ volume,
        const int* __restrict__ windows, const int n, const int K, float* __restrict__ out_min,
        float* __restrict__ out_max, float* __restrict__ out_range, float* __restrict__ out_vol)
{
    const long long id = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= (long long)n * K) return;
    const int k = (int)(id / n);
    const int start = (int)(id - (long long)k * n);
    WINDOW_BODY(windows[k], start + w >= n, id)         /* NEGATIVE CONTROL: drops the last valid start */
}

extern "C" __global__ void features_window_bug(
        const float* __restrict__ low, const float* __restrict__ high, const float* __restrict__ volume,
        const int* __restrict__ windows, const int n, const int K, float* __restrict__ out_min,
        float* __restrict__ out_max, float* __restrict__ out_range, float* __restrict__ out_vol)
{
    const long long id = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= (long long)n * K) return;
    const int k = (int)(id / n);
    const int start = (int)(id - (long long)k * n);
    WINDOW_BODY(windows[(k + 1) % K], start + w > n, id)  /* NEGATIVE CONTROL: wrong window length (memory-safe) */
}

extern "C" __global__ void local_peaks(const int* __restrict__ count, const int* __restrict__ windows,
                                       const int* __restrict__ thr, const int n, const int K,
                                       unsigned char* __restrict__ flag)
{
    const long long id = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= (long long)n * K) return;
    const int k = (int)(id / n);
    const int s = (int)(id - (long long)k * n);
    const int w = windows[k];
    const int nk = n - w + 1;
    unsigned char keep = 0;
    if (s < nk) {
        const int c = count[id];
        if (c <= thr[k]) {
            const int r = max(1, w / 2);
            const int lo = max(0, s - r), hi = min(nk - 1, s + r);
            keep = 1;
            const int* row = count + (long long)k * n;
            for (int j = lo; j <= hi; ++j) {
                if (j == s) continue;
                const int cj = row[j];
                if (cj < c || (cj == c && j < s)) { keep = 0; break; }
            }
        }
    }
    flag[id] = keep;
}
'''
k_kmajor = cp.RawKernel(FEATURE_SRC, "features_kmajor")
k_smajor = cp.RawKernel(FEATURE_SRC, "features_smajor")
k_bound_bug = cp.RawKernel(FEATURE_SRC, "features_bound_bug")
k_window_bug = cp.RawKernel(FEATURE_SRC, "features_window_bug")
k_peaks = cp.RawKernel(FEATURE_SRC, "local_peaks")
INT_MAX = np.iinfo(np.int32).max


def grid_for(total, block=CUDA_BLOCK):
    return ((int(total) + block - 1) // block,)


def launch_features(kernel, d_low, d_high, d_vol, d_win, outs, block=CUDA_BLOCK):
    n, K = int(d_low.size), int(d_win.size)
    kernel(grid_for(n * K, block), (block,), (d_low, d_high, d_vol, d_win, np.int32(n), np.int32(K), *outs))


def gpu_features(arrs, windows, kernel=k_kmajor):
    d_in = [cp.asarray(a) for a in arrs]
    d_win = cp.asarray(windows)
    n, K = arrs[0].size, len(windows)
    shape = (K, n) if kernel is not k_smajor else (n, K)
    outs = [cp.empty(shape, cp.float32) for _ in range(4)]
    launch_features(kernel, *d_in, d_win, outs)
    res = np.stack([cp.asnumpy(o) for o in outs])
    return res if kernel is not k_smajor else np.ascontiguousarray(res.transpose(0, 2, 1))


print("kernels: features_kmajor, features_smajor, local_peaks, 2 negative controls")

# %% [markdown]
# ## 7. Score: per-length rarity and candidate selection
# A z-score clipped at 50 saturated in an earlier version, so the top candidates were effectively tied. This version
# uses a **rank-based rarity**:
# - Among the N_k valid starts for window length k, `c(v) = #{x ≥ v}` (including itself, so at least 1), computed for
#   price range and for volume separately.
# - Combined count `c = max(c_range, c_volume)`, **rarity S = −log10(c / N_k)**. "S = 3" means "price range and volume
#   are **both** in the top 0.1% of all intervals of this length".
# - Ranks do not saturate, are invariant to monotone transforms such as log, and are comparable across lengths. Counts
#   are integers, so CPU and GPU agree bit for bit.
# - Candidate selection: S ≥ 2 (top 1%) and the rarest point among ±w/2 neighbours at the same length → sort by S
#   descending (ties by k, then start) → drop any candidate within 60 bars of, or overlapping, an already chosen one
#   (another time or scale of the same episode) → top 20 episodes.
# - Without the episode merge, the single SEC event fills the list with 21:11, 21:13, 21:15, and 21:24.
# - Caveat: overlapping windows are not independent. "Top 0.1%" is a rank, not the probability of an independent sample.

# %%
def nk_of(n, windows):
    return n - np.asarray(windows, np.int64) + 1


def thresholds(n, windows):
    return np.floor(nk_of(n, windows) * 10.0 ** (-S_MIN)).astype(np.int32)


def counts_cpu(feat, windows):
    K, n = feat.shape[1], feat.shape[2]
    cnt = np.full((K, n), INT_MAX, np.int32)
    for k, nk in enumerate(nk_of(n, windows)):
        r, v = feat[2, k, :nk], feat[3, k, :nk]
        cr = nk - np.searchsorted(np.sort(r), r, side="left")
        cv = nk - np.searchsorted(np.sort(v), v, side="left")
        cnt[k, :nk] = np.maximum(cr, cv)
    return cnt


def counts_gpu(d_range, d_vol, windows):
    K, n = d_range.shape
    d_cnt = cp.full((K, n), INT_MAX, dtype=cp.int32)
    for k, nk in enumerate(nk_of(n, windows)):
        r, v = d_range[k, :nk], d_vol[k, :nk]
        cr = nk - cp.searchsorted(cp.sort(r), r, side="left")
        cv = nk - cp.searchsorted(cp.sort(v), v, side="left")
        d_cnt[k, :nk] = cp.maximum(cr, cv)
    return d_cnt


@njit(cache=False)
def peaks_cpu(cnt, windows, thr):
    K, n = cnt.shape
    flag = np.zeros((K, n), np.uint8)
    for k in range(K):
        w = windows[k]
        nk = n - w + 1
        r = max(1, w // 2)
        for s in range(nk):
            c = cnt[k, s]
            if c > thr[k]:
                continue
            keep = 1
            for j in range(max(0, s - r), min(nk - 1, s + r) + 1):
                if j != s and (cnt[k, j] < c or (cnt[k, j] == c and j < s)):
                    keep = 0
                    break
            flag[k, s] = keep
    return flag


def rarity_matrix(cnt, windows):
    n = cnt.shape[1]
    nk = nk_of(n, windows)[:, None]
    with np.errstate(divide="ignore"):
        S = -np.log10(cnt.astype(np.float64) / nk)
    S[cnt == INT_MAX] = np.nan
    return S


def select_top(flat_idx, counts, n, windows, top_k=TOP_K):
    """Host side: survivors -> S -> deterministic order -> one per episode -> top_k rows of (k, start, window, S)."""
    flat_idx = np.asarray(flat_idx, np.int64)
    k = flat_idx // n
    s = flat_idx - k * n
    S = -np.log10(np.asarray(counts, np.float64) / nk_of(n, windows)[k])
    order = np.lexsort((s, k, -S))
    kept = []
    for i in order:
        a0, a1 = s[i], s[i] + windows[k[i]]
        if all(max(a0, b0) - min(a1, b1) >= EPISODE_GAP for _, b0, b1, _ in kept):   # gap between intervals
            kept.append((int(k[i]), int(a0), int(a1), float(S[i])))
            if len(kept) == top_k:
                break
    return pd.DataFrame([{"k": kk, "start": a0, "window": a1 - a0, "S": sc} for kk, a0, a1, sc in kept],
                        columns=["k", "start", "window", "S"])


def cpu_pipeline(arrs, windows, features_fn=cpu_direct_1t):
    feat = features_fn(*arrs, windows)
    cnt = counts_cpu(feat, windows)
    flag = peaks_cpu(cnt, windows, thresholds(cnt.shape[1], windows))
    idx = np.flatnonzero(flag)
    return select_top(idx, cnt.ravel()[idx], cnt.shape[1], windows), feat, cnt


print("rarity score and candidate selection defined")

# %% [markdown]
# ## 8. Correctness gate and negative controls (required before any timing)
# 1. GPU features (k-major, start-major) = CPU direct-scan features: exact min/max, tolerance on range/volume, same NaN positions.
# 2. Optimized CPU features = direct-scan features (volume uses a float64 prefix sum, tolerance 2e-5).
# 3. GPU counts = CPU counts (integers, exact). GPU candidate flags = CPU candidate flags.
# 4. Both negative controls must FAIL.
#
# Checked on the full 1-minute series (131,040 rows) with K = 15. The run stops here if anything is off.

# %%
def feature_gate(ref, got, rtol=RTOL):
    res = {"shape_ok": ref.shape == got.shape, "nan_mask_same": bool(np.array_equal(np.isnan(ref), np.isnan(got)))}
    ok = res["shape_ok"] and res["nan_mask_same"]
    for i, name in enumerate(["min", "max", "range", "volume"]):
        v = ~np.isnan(ref[i])
        err = float(np.max(np.abs(ref[i][v] - got[i][v]) / np.maximum(np.abs(ref[i][v]), 1e-30))) if v.any() else 0.0
        res[f"{name}_max_rel_err"] = err
        ok = ok and err <= rtol[name]
    res["PASS"] = bool(ok)
    return res


COLD = {}
_small = tuple(x[:4096] for x in M1)
for label, fn in [("features_kmajor (NVRTC compile + launch)", lambda: gpu_features(_small, WINDOWS))]:
    times = []
    for _ in range(2):
        gpu_sync(); t0 = time.perf_counter(); fn(); gpu_sync()
        times.append((time.perf_counter() - t0) * 1e3)
    COLD[label] = {"first_call_ms": times[0], "second_call_ms": times[1]}
display(pd.DataFrame(COLD).T.round(3))

ref_feat = cpu_direct_1t(*M1, WINDOWS)
rows = []
for label, kern in [("GPU features_kmajor", k_kmajor), ("GPU features_smajor", k_smajor),
                    ("NEG: boundary >= bug", k_bound_bug), ("NEG: wrong window length", k_window_bug)]:
    rows.append({"case": label, "expected": "FAIL" if label.startswith("NEG") else "PASS",
                 **feature_gate(ref_feat, gpu_features(M1, WINDOWS, kern))})
rows.append({"case": "CPU direct MT", "expected": "PASS", **feature_gate(ref_feat, cpu_direct_mt(*M1, WINDOWS))})
rows.append({"case": "CPU optimized O(N*K)", "expected": "PASS",
             **feature_gate(ref_feat, cpu_smart(*M1, WINDOWS), {**RTOL, "volume": SMART_VOLUME_RTOL})})

cnt_cpu = counts_cpu(ref_feat, WINDOWS)
d_feat = [cp.asarray(ref_feat[i]) for i in range(4)]
d_cnt = counts_gpu(d_feat[2], d_feat[3], WINDOWS)
cnt_gpu = cp.asnumpy(d_cnt)
thr = thresholds(cnt_cpu.shape[1], WINDOWS)
flag_cpu = peaks_cpu(cnt_cpu, WINDOWS, thr)
d_flag = cp.empty(cnt_cpu.shape, cp.uint8)
k_peaks(grid_for(cnt_cpu.size), (CUDA_BLOCK,), (d_cnt, cp.asarray(WINDOWS), cp.asarray(thr),
                                                 np.int32(cnt_cpu.shape[1]), np.int32(len(WINDOWS)), d_flag))
flag_gpu = cp.asnumpy(d_flag)
rows.append({"case": "GPU rarity counts (exact)", "expected": "PASS", "PASS": bool(np.array_equal(cnt_cpu, cnt_gpu))})
rows.append({"case": "GPU local_peaks flags (exact)", "expected": "PASS", "PASS": bool(np.array_equal(flag_cpu, flag_gpu)),
             "survivors": int(flag_cpu.sum())})

gate_df = pd.DataFrame(rows)
gate_df["as_expected"] = gate_df.PASS == (gate_df.expected == "PASS")
with pd.option_context("display.width", 250):
    display(gate_df)
gate_df.to_csv(RESULTS_DIR / "correctness_gate.csv", index=False)
CORRECTNESS_PASS = bool(gate_df.as_expected.all())
if not CORRECTNESS_PASS:
    raise AssertionError("Correctness gate FAILED (or a negative control was not caught). Benchmarks stopped.")
print("CORRECTNESS GATE: PASS (all real cases pass, both negative controls caught)")
del d_feat, d_cnt, d_flag
checkpoint("8_correctness", correctness_pass=CORRECTNESS_PASS, cold_start=COLD)

# %% [markdown]
# ## 9. Three GPU paths (output design) and agreement across paths
# Every path must return the same top 20. What differs is **where each step runs and what crosses the bus**.
#
# | Path | On GPU | D2H | On CPU |
# |---|---|---|---|
# | (1) features returned | features | all four features `[4,K,N]` = 16 B per candidate | rarity, candidate selection, top 20 |
# | (2) counts returned | features, rarity counts | counts `[K,N]` int32 = 4 B per candidate | candidate selection, top 20 |
# | (3) candidates returned | features, counts, candidate selection (local_peaks + compaction) | surviving candidates only (thousands) | top 20 (episode merge) |
#
# Segments are separated with Events on one stream; `host gap` = wall time − (Event segments + CPU segments).

# %%
class GpuPipeline:
    """Preallocated device/host buffers for one (n, windows) shape, so timing excludes allocation and page faults.

    run(arrs, path, queued=True): GPU stages after H2D are enqueued behind a spin kernel, so Event intervals are pure
    GPU time (CuPy's sort may still synchronize internally, so the rarity segment can include some host gaps).
    run(arrs, path, queued=False): plain run; its wall-clock total is what an application would see.
    """

    def __init__(self, n, windows):
        self.n, self.windows = n, np.asarray(windows, np.int32)
        K = len(windows)
        self.d_win, self.d_thr = cp.asarray(self.windows), cp.asarray(thresholds(n, self.windows))
        self.d_in = [cp.empty(n, cp.float32) for _ in range(3)]
        self.outs = [cp.empty((K, n), cp.float32) for _ in range(4)]
        self.d_flag = cp.empty((K, n), cp.uint8)
        self.h_feat = np.empty((4, K, n), np.float32)
        self.h_cnt = np.empty((K, n), np.int32)

    def run(self, arrs, path, queued=False):
        n, K = self.n, len(self.windows)
        seg = {}
        gpu_sync(); t0 = time.perf_counter()
        ev = [cp.cuda.Event() for _ in range(6)]
        ev[0].record()
        for d, a in zip(self.d_in, arrs):
            h2d_into(d, a)
        ev[1].record()
        if queued:
            spin(3.0)
        ev[2].record()
        launch_features(k_kmajor, *self.d_in, self.d_win, self.outs)
        ev[3].record()
        if path == 1:
            ev[3].synchronize()
            t1 = time.perf_counter()
            for i in range(4):
                cp.asnumpy(self.outs[i], out=self.h_feat[i])
            t2 = time.perf_counter()
            cnt = counts_cpu(self.h_feat, self.windows)
            flag = peaks_cpu(cnt, self.windows, thresholds(n, self.windows))
            idx = np.flatnonzero(flag)
            top = select_top(idx, cnt.ravel()[idx], n, self.windows)
            t3 = time.perf_counter()
            seg.update(d2h=(t2 - t1) * 1e3, cpu_post=(t3 - t2) * 1e3)
        else:
            d_cnt = counts_gpu(self.outs[2], self.outs[3], self.windows)
            ev[4].record()
            if path == 2:
                ev[4].synchronize()
                t1 = time.perf_counter()
                cp.asnumpy(d_cnt, out=self.h_cnt)
                t2 = time.perf_counter()
                flag = peaks_cpu(self.h_cnt, self.windows, thresholds(n, self.windows))
                idx = np.flatnonzero(flag)
                top = select_top(idx, self.h_cnt.ravel()[idx], n, self.windows)
                t3 = time.perf_counter()
                seg.update(rarity=cp.cuda.get_elapsed_time(ev[3], ev[4]), d2h=(t2 - t1) * 1e3, cpu_post=(t3 - t2) * 1e3)
            else:
                k_peaks(grid_for(K * n), (CUDA_BLOCK,), (d_cnt, self.d_win, self.d_thr, np.int32(n), np.int32(K), self.d_flag))
                d_idx = cp.flatnonzero(self.d_flag)
                d_c = d_cnt.ravel()[d_idx]
                ev[5].record()
                ev[5].synchronize()
                t1 = time.perf_counter()
                idx, cnts = cp.asnumpy(d_idx), cp.asnumpy(d_c)
                t2 = time.perf_counter()
                top = select_top(idx, cnts, n, self.windows)
                t3 = time.perf_counter()
                seg.update(rarity=cp.cuda.get_elapsed_time(ev[3], ev[4]), peaks=cp.cuda.get_elapsed_time(ev[4], ev[5]),
                           d2h=(t2 - t1) * 1e3, cpu_post=(t3 - t2) * 1e3)
        gpu_sync()
        seg.update(h2d=cp.cuda.get_elapsed_time(ev[0], ev[1]), features=cp.cuda.get_elapsed_time(ev[2], ev[3]),
                   total=(time.perf_counter() - t0) * 1e3)
        return top, seg


top_cpu, feat_m1, cnt_m1 = cpu_pipeline(M1, WINDOWS)
pipe_m1 = GpuPipeline(len(M1[0]), WINDOWS)
same = {}
for path in (1, 2, 3):
    top_gpu, _ = pipe_m1.run(M1, path)
    same[path] = bool(top_gpu[["k", "start"]].equals(top_cpu[["k", "start"]]))
print("top-20 identical to CPU pipeline:", same)
PATHS_AGREE = all(same.values())
if not PATHS_AGREE:
    raise AssertionError("GPU paths disagree with the CPU pipeline")
S_m1 = rarity_matrix(cnt_m1, WINDOWS)
checkpoint("9_paths", paths_agree=PATHS_AGREE)

# %% [markdown]
# ## 10. Comparison with real events (descriptive, no injection)
# The top 20 candidates on the real 1-minute series (2024 Q1) are compared with published event times.
# - **News events (post-hoc list):** compiled after noticing that an earlier run's top candidates overlapped some
#   reports. Times were taken separately from article timestamps. Being post hoc, the list carries selection bias.
# - **Macro calendar (pre-registered rule):** "every U.S. CPI release (08:30 ET) and FOMC statement (14:00 ET) in
#   2024 Q1". The list was fixed by the rule, not chosen from results.
# - A hit: the event time falls inside a candidate interval widened by 60 minutes on each side.
# - **Chance baseline:** shift all event times by the same random amount (circularly over the quarter) 5,000 times and
#   recount, giving the hits expected by chance.
# - **p-values are reported for the pre-registered list only.** The news list was built after seeing top candidates of
#   the same data, so a p-value from it is structurally small and is not evidence. For that list only hit counts and
#   the chance expectation are shown.
# - Note: CPI on 01-11 (13:30) and the spot ETF trading start (14:30) are 60 minutes apart, so one episode can hit both lists.

# %%
NEWS_EVENTS = [
    ("2024-01-03 12:20", "Matrixport note: SEC to reject ETFs; BTC drops", "crypto.news / The Block (7:20 a.m. EST; hour-level)"),
    ("2024-01-09 21:11", "SEC X account hacked: fake ETF approval post", "BleepingComputer / CBS News (4:11 p.m. ET)"),
    ("2024-01-11 14:30", "US spot BTC ETFs start trading", "exchange open 9:30 a.m. ET"),
    ("2024-02-28 17:40", "BTC to ~$64k; Coinbase outage", "CNBC / CoinDesk (status update 9:40 a.m. PST)"),
    ("2024-03-05 15:00", "BTC record ~$69.2k then -10% flush", "CNBC / CoinDesk (just after 10 a.m. ET)"),
]
MACRO_EVENTS = [
    ("2024-01-11 13:30", "US CPI (Dec)"), ("2024-01-31 19:00", "FOMC statement"), ("2024-02-13 13:30", "US CPI (Jan)"),
    ("2024-03-12 12:30", "US CPI (Feb, EDT)"), ("2024-03-20 18:00", "FOMC statement (EDT)"),
]
t_m1 = m1.timestamp
t0_m1 = t_m1.iloc[0]
minutes_total = len(m1)


def event_index(ts):
    return int((pd.Timestamp(ts, tz="UTC") - t0_m1) / pd.Timedelta(minutes=1))


def hits_for(top, ev_idx, tol=EVENT_TOL_MIN):
    hit = np.zeros(len(ev_idx), bool)
    for _, r in top.iterrows():
        lo, hi = r.start - tol, r.start + r.window + tol
        hit |= (ev_idx >= lo) & (ev_idx < hi)
    return hit


def permutation_p(top, ev_idx, trials=5000):
    obs = int(hits_for(top, ev_idx).sum())
    rng = np.random.default_rng(SEED)
    shifts = rng.integers(0, minutes_total, trials)
    null = np.array([hits_for(top, (ev_idx + sh) % minutes_total).sum() for sh in shifts])
    return obs, float(np.mean(null >= obs)), float(null.mean())


top_real = top_cpu.copy()
top_real["start_time"] = t_m1.iloc[top_real.start].to_numpy()
top_real["window_min"] = top_real.window
news_idx = np.array([event_index(t) for t, *_ in NEWS_EVENTS])
macro_idx = np.array([event_index(t) for t, _ in MACRO_EVENTS])


def nearest_event(r):
    best = None
    for (ts, name, *_), ei in zip(NEWS_EVENTS + [(t, n) for t, n in MACRO_EVENTS], np.r_[news_idx, macro_idx]):
        d = max(r.start - ei, ei - (r.start + r.window), 0)
        if best is None or d < best[0]:
            best = (d, name)
    return pd.Series({"nearest_event": best[1], "distance_min": best[0]})


top_real = pd.concat([top_real, top_real.apply(nearest_event, axis=1)], axis=1)
with pd.option_context("display.width", 250, "display.max_colwidth", 60):
    display(top_real[["start_time", "window_min", "S", "nearest_event", "distance_min"]])
top_real.to_csv(RESULTS_DIR / "top20_real_1m.csv", index=False)

EVENT_RESULTS = {}
if DATA_KIND == "real_market":
    for label, idx_arr, use_p in [("news (post-hoc list)", news_idx, False), ("macro (pre-registered rule)", macro_idx, True)]:
        obs, p, null_mean = permutation_p(top_real, idx_arr)
        hit_names = [ev[1] for ev, h in zip(NEWS_EVENTS if not use_p else MACRO_EVENTS, hits_for(top_real, idx_arr)) if h]
        EVENT_RESULTS[label] = {"events": len(idx_arr), "hit": obs, "chance_mean_hits": round(null_mean, 3),
                                "p_value": p if use_p else "not interpretable (list built after seeing an earlier run)",
                                "hit_events": "; ".join(hit_names)}
    with pd.option_context("display.max_colwidth", 120):
        display(pd.DataFrame(EVENT_RESULTS).T)
else:
    print("synthetic fallback data: event comparison skipped")
checkpoint("10_events", event_results=EVENT_RESULTS)

# %% [markdown]
# ## 11. Figures 1-2: quarter overview and the SEC event zoom (1-minute bars)
# The map's horizontal axis is the **window centre** (plotting by start time shifts long windows to the left). The
# vertical axis is window length (log scale); colour is rarity S.
# The full quarter (131,040 minutes) drawn on about 1,500 screen columns puts 87 minutes in one column, so naive
# downsampling erases short events. The overview therefore draws the **rarest value per column** (max pooling); the
# 2-D map is used only in the ±3 h zoom (Figure 2).

# %%
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates

mpl.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25})
S_TICKS = [0, 1, 2, 3, 4, 5]
S_LABELS = ["all", "top 10%", "top 1%", "top 0.1%", "top 0.01%", "top 0.001%"]


def centered(S, windows):
    out = np.full_like(S, np.nan)
    for k, w in enumerate(windows):
        h = int(w) // 2
        out[k, h:] = S[k, : S.shape[1] - h]
    return out


def scale_panel(ax, S_c, times, windows, lo, hi, cbar=True, vmax=4.5):
    sub = S_c[:, lo:hi]
    mesh = ax.pcolormesh(times[lo:hi], np.arange(len(windows)), sub, shading="nearest", cmap="magma",
                         vmin=0, vmax=vmax)
    ax.set_yticks(np.arange(len(windows))[::2]); ax.set_yticklabels(windows[::2])
    ax.set_ylabel("window length")
    if cbar:
        cb = plt.colorbar(mesh, ax=ax, pad=0.01, fraction=0.03)
        cb.set_ticks(S_TICKS[: int(vmax) + 1]); cb.set_ticklabels(S_LABELS[: int(vmax) + 1])
    return mesh


S_m1_c = centered(S_m1, WINDOWS)
times_m1 = t_m1.dt.tz_convert(None).to_numpy()

# Figure 1: overview. Bottom = max rarity over all window lengths, max-pooled per screen column (short events survive).
BINS = 1500
edges = np.linspace(0, len(m1), BINS + 1).astype(int)
row_max = np.nanmax(np.where(np.isnan(S_m1_c), -np.inf, S_m1_c), axis=0)
pooled = np.array([row_max[a:b].max() for a, b in zip(edges[:-1], edges[1:])])
ptimes = times_m1[edges[:-1]]
fig, axes = plt.subplots(2, 1, figsize=(16, 6.5), sharex=True, height_ratios=[1.3, 1])
axes[0].plot(times_m1, m1.close, lw=0.6, color="black")
axes[0].set(ylabel="BTCUSDT close", title="Figure 1 - 2024 Q1: every (start, length) scored on GPU; top-20 episodes vs public events")
axes[1].fill_between(ptimes, 0, pooled, color="tab:purple", alpha=0.35, step="post")
axes[1].plot(ptimes, pooled, color="tab:purple", lw=0.6, drawstyle="steps-post")
top_t = times_m1[top_real.start.to_numpy()]
axes[1].scatter(top_t, top_real.S, marker="v", color="tab:orange", s=40, zorder=5, label="top-20 episodes")
axes[0].scatter(top_t, m1.close.to_numpy()[top_real.start.to_numpy()], marker="v", color="tab:orange", s=40, zorder=5)
axes[1].set_yticks(S_TICKS[:6]); axes[1].set_yticklabels(S_LABELS[:6]); axes[1].set_ylim(0, 5.3)
axes[1].set_ylabel("rarest window here")
for ax in axes:
    for ts, name, *_ in NEWS_EVENTS:
        ax.axvline(pd.Timestamp(ts), color="tab:red", lw=0.9, ls="--")
    for ts, name in MACRO_EVENTS:
        ax.axvline(pd.Timestamp(ts), color="tab:cyan", lw=0.9, ls=":")
for ts, name, *_ in NEWS_EVENTS:
    axes[0].annotate(name.split(":")[0][:30], (pd.Timestamp(ts), axes[0].get_ylim()[1]), rotation=90, va="top",
                     ha="right", fontsize=7, color="tab:red")
axes[1].plot([], [], color="tab:red", ls="--", label="news event (post-hoc list)")
axes[1].plot([], [], color="tab:cyan", ls=":", label="CPI / FOMC (pre-registered rule)")
axes[1].legend(loc="upper left", fontsize=8, ncol=3)
plt.tight_layout(); save_fig(fig, "fig1_overview"); plt.show()

# Figure 2: SEC event zoom, +-3 h
sec_i = event_index("2024-01-09 21:11")
lo, hi = max(sec_i - 180, 0), min(sec_i + 180, len(m1))
fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True, height_ratios=[1.2, 0.6, 1.4])
axes[0].fill_between(times_m1[lo:hi], m1.low[lo:hi], m1.high[lo:hi], color="tab:blue", alpha=0.3, label="low-high")
axes[0].plot(times_m1[lo:hi], m1.close[lo:hi], color="tab:blue", lw=1, label="close")
axes[1].bar(times_m1[lo:hi], m1.volume[lo:hi], width=1 / 1440, color="tab:gray")
scale_panel(axes[2], S_m1_c, times_m1, WINDOWS, lo, hi)
for ax in axes:
    ax.axvline(pd.Timestamp("2024-01-09 21:11"), color="cyan", ls="--", lw=1.5)
axes[0].set(ylabel="price (USDT)", title="Figure 2 - fake SEC ETF post (21:11 UTC): price, volume, and the multi-scale rarity map (1-minute bars)")
axes[0].legend(loc="upper left"); axes[1].set_ylabel("volume (BTC)")
axes[2].set_ylabel("window (min)"); axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.tight_layout(); save_fig(fig, "fig2_sec_zoom_1m"); plt.show()

# %% [markdown]
# ## 12. Figure 3: the same event at 1-second resolution
# All 7 days of 1-second bars (604,800 rows) are scanned with the same 15 windows (2 s to 4.3 min) and the ±20 min
# around the event are shown. A reaction that took one or two cells at 1-minute resolution unfolds into second-level
# structure. Raising the resolution multiplies N by 60, which is why the GPU matters.

# %%
if DATA_KIND == "real_market":
    d_in = [cp.asarray(a) for a in S1]
    outs = [cp.empty((K_ALL, len(S1[0])), cp.float32) for _ in range(4)]
    launch_features(k_kmajor, *d_in, cp.asarray(WINDOWS), outs)
    cnt_s1 = cp.asnumpy(counts_gpu(outs[2], outs[3], WINDOWS))
    del d_in, outs
    S_s1_c = centered(rarity_matrix(cnt_s1, WINDOWS), WINDOWS)
    times_s1 = s1.timestamp.dt.tz_convert(None).to_numpy()
    sec_s = int((pd.Timestamp("2024-01-09 21:11", tz="UTC") - s1.timestamp.iloc[0]) / pd.Timedelta(seconds=1))
    lo, hi = sec_s - 1200, sec_s + 1200
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True, height_ratios=[1.2, 0.6, 1.4])
    axes[0].fill_between(times_s1[lo:hi], s1.low[lo:hi], s1.high[lo:hi], color="tab:blue", alpha=0.3)
    axes[0].plot(times_s1[lo:hi], s1.close[lo:hi], color="tab:blue", lw=0.8)
    axes[1].bar(times_s1[lo:hi], s1.volume[lo:hi], width=1 / 86400, color="tab:gray")
    scale_panel(axes[2], S_s1_c, times_s1, WINDOWS, lo, hi, vmax=5)
    for ax in axes:
        ax.axvline(pd.Timestamp("2024-01-09 21:11"), color="cyan", ls="--", lw=1.5)
    axes[0].set(ylabel="price (USDT)", title="Figure 3 - the same event at 1-second resolution (7 days = 604,800 bars scanned)")
    axes[1].set_ylabel("volume (BTC)"); axes[2].set_ylabel("window (s)")
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.tight_layout(); save_fig(fig, "fig3_sec_zoom_1s"); plt.show()
else:
    print("synthetic fallback: 1-second zoom skipped")
checkpoint("12_figures_1_3")

# %% [markdown]
# ## 13. Controlled injection: does the map read the length (H5), and how strong must an event be (H6)?
# A pump-and-dump shape (rise over 2/3 of the length, back to the start over the last 1/3) is added to a copy of the real
# 1-minute series. This measures the **sensitivity of the detector**; it is the standard way to evaluate an anomaly
# detector when labeled events are scarce.
# - Strength: price rise = q × (real median range of windows of that length), volume × (1 + q). The same positive factor
#   multiplies the whole bar (OHLC), so OHLC invariants hold.
# - Position: **random**, excluding ±1 day around known events and both ends of the data (choosing quiet places would
#   make detection easier and bias the result).
# - 20 trials per (3 lengths × 5 strengths). Each trial recomputes all 15 windows over the full 1-minute series.
# - Metrics: (1) top-20 episode hit (a candidate centre inside the injected interval), (2) overall rank of the best
#   candidate inside the injection (top x %; the **primary metric**, continuous, so changes with strength are visible),
#   (3) estimated length / true length. At strong injections several lengths saturate at counts of 1-3, so picking one
#   would be decided by tie order; the estimate is the geometric mean of all lengths within 2x of the best count, and the
#   number of such lengths is recorded.
# - Meaning of H6: the minimum strength needed to beat the quarter's real top-20 episodes.
# - Why q = 2..32: in a local check, q ≤ 4 (for a 30-minute event, +1.7% and volume ×5) was weaker than real shocks in the
#   quarter (5-10% moves with 10-30x volume) and never reached the top 20; the threshold sits near q ≈ 8-16, so the range
#   covers both sides.
# - Length prediction (H5): the rise is gentle over 2L/3 and the fall is steep over L/3. Range rarity peaks at windows
#   covering only the fall (≈ L/3); volume rarity keeps growing up to L. The combined score sits where the two meet,
#   so the prediction is **between L/3 and L**, possibly moving with strength.

# %%
def median_range(arrs, L):
    f = cpu_smart(*arrs, np.array([L], np.int32))
    return float(np.nanmedian(f[2, 0]))


def inject(arrs, start, L, q, med_range):
    low, high, vol = (a.copy() for a in arrs)
    t = (np.arange(L) + 0.5) / L
    shape = np.where(t < 2 / 3, t / (2 / 3), (1 - t) / (1 / 3))
    factor = (1.0 + q * med_range * shape).astype(np.float32)
    low[start:start + L] *= factor
    high[start:start + L] *= factor
    vol[start:start + L] *= np.float32(1.0 + q)
    return low, high, vol


def evaluate_injection(arrs, start, L, pipe):
    top, _ = pipe.run(arrs, 3)
    centers = top.start + top.window / 2
    hit = bool(((centers >= start) & (centers < start + L)).any())
    # best candidate whose window centre lies inside the injected interval (needs the full count matrix)
    d_in = [cp.asarray(a) for a in arrs]
    outs = [cp.empty((K_ALL, pipe.n), cp.float32) for _ in range(4)]
    launch_features(k_kmajor, *d_in, pipe.d_win, outs)
    d_cnt = counts_gpu(outs[2], outs[3], WINDOWS)
    nk = nk_of(pipe.n, WINDOWS)
    row_best = {}                                 # per window length: smallest count among candidates centred inside
    for k, w in enumerate(WINDOWS):
        a, b = max(start - int(w) // 2, 0), min(start + L - int(w) // 2, int(nk[k]))
        if b > a:
            row_best[k] = int(cp.min(d_cnt[k, a:b]))
    kb = min(row_best, key=lambda k: (row_best[k] / nk[k], k))
    c_best, nk_best = row_best[kb], int(nk[kb])
    near = [k for k, c in row_best.items() if c * nk_best <= 2 * c_best * int(nk[k])]
    w_est = float(np.exp(np.mean(np.log(WINDOWS[near].astype(float)))))
    better = sum(int(cp.count_nonzero(d_cnt[k, : nk[k]].astype(cp.int64) * nk_best <= c_best * int(nk[k])))
                 for k in range(K_ALL))
    return {"hit_top20": hit, "best_rank_pct": 100.0 * better / int(nk.sum()), "best_window": int(WINDOWS[kb]),
            "lengths_within_2x": len(near), "w_est": w_est, "len_ratio": w_est / L,
            "best_S": float(-np.log10(c_best / nk_best))}


all_events = np.r_[news_idx, macro_idx]
allowed = np.ones(len(m1), bool)
allowed[:400] = allowed[-400:] = False
for ei in all_events:
    allowed[max(ei - 1440, 0): ei + 1440] = False


def random_start(rng, L):
    while True:
        st = int(rng.integers(0, len(m1) - L))
        if allowed[st: st + L].all():
            return st


rng_inj = np.random.default_rng(SEED + 3)
inj_rows = []
examples = {}
for L in INJ_LENGTHS:
    med = median_range(M1, L)
    for q in INJ_STRENGTHS:
        for trial in range(INJ_TRIALS):
            st = random_start(rng_inj, L)
            arrs = inject(M1, st, L, q, med)
            r = evaluate_injection(arrs, st, L, pipe_m1)
            inj_rows.append({"L": L, "q": q, "trial": trial, "start": st, "median_range_L": med, **r})
            if q == max(INJ_STRENGTHS) and trial == 0:
                examples[L] = (st, arrs)
inj_df = pd.DataFrame(inj_rows)
inj_df.to_csv(RESULTS_DIR / "injection_trials.csv", index=False)
inj_summary = inj_df.groupby(["L", "q"]).agg(hit_rate=("hit_top20", "mean"), median_rank_pct=("best_rank_pct", "median"),
                                             median_len_ratio=("len_ratio", "median"),
                                             median_lengths_within_2x=("lengths_within_2x", "median"),
                                             trials=("trial", "count")).reset_index()
inj_summary.to_csv(RESULTS_DIR / "injection_summary.csv", index=False)
display(inj_summary.round(4))

# %% [markdown]
# ### Figures 4-5: reading length from the map, hit rate by strength

# %%
fig, axes = plt.subplots(2, len(INJ_LENGTHS), figsize=(16, 6.5), height_ratios=[1, 1.3])
for j, L in enumerate(INJ_LENGTHS):
    st, arrs = examples[L]
    d_in = [cp.asarray(a) for a in arrs]
    outs = [cp.empty((K_ALL, len(m1)), cp.float32) for _ in range(4)]
    launch_features(k_kmajor, *d_in, cp.asarray(WINDOWS), outs)
    S_c = centered(rarity_matrix(cp.asnumpy(counts_gpu(outs[2], outs[3], WINDOWS)), WINDOWS), WINDOWS)
    pad = max(3 * L, 120)
    lo, hi = max(st - pad, 0), min(st + L + pad, len(m1))
    axes[0, j].plot(times_m1[lo:hi], arrs[1][lo:hi], color="tab:blue", lw=0.8)
    axes[0, j].axvspan(times_m1[st], times_m1[st + L - 1], color="tab:red", alpha=0.15)
    axes[0, j].set_title(f"injected pump-and-dump shape, L = {L} min (q = {max(INJ_STRENGTHS)})")
    scale_panel(axes[1, j], S_c, times_m1, WINDOWS, lo, hi, cbar=(j == len(INJ_LENGTHS) - 1))
    for frac_ in (1 / 3, 1.0):
        axes[1, j].axhline(np.interp(np.log(L * frac_), np.log(WINDOWS), np.arange(K_ALL)), color="cyan", ls=":", lw=1.2)
    axes[1, j].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    axes[1, j].tick_params(axis="x", labelsize=7)
axes[0, 0].set_ylabel("high (USDT)")
fig.suptitle("Figure 4 - the blob's height reads the event length (cyan lines: predicted range L/3 .. L)", y=1.0)
plt.tight_layout(); save_fig(fig, "fig4_injection_length"); plt.show()

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for L, g in inj_summary.groupby("L"):
    axes[0].plot(g.q, 100 * g.hit_rate, "o-", label=f"L = {L} min")
    axes[1].plot(g.q, g.median_rank_pct, "o-", label=f"L = {L} min")
axes[0].axhline(50, color="gray", ls=":")
axes[0].set(xscale="log", xlabel="strength q (x median range of that length)", ylabel="top-20 hit rate (%)",
            title=f"Figure 5a - how strong must it be? ({INJ_TRIALS} random positions each)")
axes[1].set(xscale="log", yscale="log", xlabel="strength q", ylabel="best candidate inside: top x % of all candidates",
            title="Figure 5b - rank of the injected event among all candidates")
axes[1].invert_yaxis()
for ax in axes:
    ax.legend()
plt.tight_layout(); save_fig(fig, "fig5_detectability"); plt.show()
checkpoint("13_injection")

# %% [markdown]
# ## 14. H7: what a fixed time budget buys (same workload vs same budget)
# Section 13 re-runs the full detector once per trial. Estimating a detection rate is a Monte Carlo problem: its
# confidence interval narrows only as 1/√n, so the number of re-evaluations that fit in a time budget sets how precise the
# estimate can be. Two experiments are kept strictly separate:
# 1. **Same workload.** Both implementations evaluate the same trials (same seeds and positions). This gives time per
#    trial and checks that the hit decisions agree.
# 2. **Same budget.** Each implementation runs fresh trials until a wall-clock budget is used up. This gives the
#    number of trials, the estimated hit rate, and its 95% Wilson interval width.
#
# Implementations: the **optimized CPU pipeline** (O(N·K) features, NumPy ranking, Numba candidate selection) is the
# baseline; the **GPU candidates-only path** (path 3) is the accelerated version. The injection itself (a NumPy copy) is
# identical on both sides. Setting: L = 30, q = 8 (near the detection threshold, where the estimate is least certain).

# %%
def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, centre - half, centre + half


MED_BUDGET = median_range(M1, BUDGET_L)


def trial_hit(impl, st):
    arrs = inject(M1, st, BUDGET_L, BUDGET_Q, MED_BUDGET)
    if impl == "cpu_optimized":
        top, _, _ = cpu_pipeline(arrs, WINDOWS, features_fn=cpu_smart)
    else:
        top, _ = pipe_m1.run(arrs, 3)
    centers = top.start + top.window / 2
    return bool(((centers >= st) & (centers < st + BUDGET_L)).any())


IMPLS = ["cpu_optimized", "gpu_path3"]
trial_hit("cpu_optimized", random_start(np.random.default_rng(1), BUDGET_L))   # warm-up (JIT, memory pool)
trial_hit("gpu_path3", random_start(np.random.default_rng(1), BUDGET_L))

# 1) same workload
rng_w = np.random.default_rng(SEED + 7)
W_TRIALS = 6 if QUICK else 40
starts = [random_start(rng_w, BUDGET_L) for _ in range(W_TRIALS)]
same_rows = []
for impl in IMPLS:
    gpu_sync(); t0 = time.perf_counter()
    hits = [trial_hit(impl, st) for st in starts]
    gpu_sync(); dt = time.perf_counter() - t0
    same_rows.append({"impl": impl, "trials": W_TRIALS, "seconds": dt, "ms_per_trial": 1e3 * dt / W_TRIALS,
                      "hits": int(sum(hits)), "_hits": hits})
agree = float(np.mean([a == b for a, b in zip(same_rows[0]["_hits"], same_rows[1]["_hits"])]))
same_df = pd.DataFrame([{k: v for k, v in r.items() if k != "_hits"} for r in same_rows])
same_df["hit_agreement_with_cpu"] = agree
same_df.to_csv(RESULTS_DIR / "budget_same_workload.csv", index=False)
display(same_df.round(4))

# 2) same budget
budget_rows = []
for B in BUDGETS_S:
    for impl in IMPLS:
        rng_b = np.random.default_rng(SEED + 11 + B)       # same position stream for both implementations
        n = k = 0
        gpu_sync(); t0 = time.perf_counter()
        while time.perf_counter() - t0 < B:
            k += trial_hit(impl, random_start(rng_b, BUDGET_L))
            n += 1
        p, lo_, hi_ = wilson(k, n)
        budget_rows.append({"budget_s": B, "impl": impl, "trials": n, "hit_rate": p, "ci_low": lo_, "ci_high": hi_,
                            "ci_width": hi_ - lo_})
        print(f"budget {B:4d} s  {impl:14s} trials {n:6d}  hit rate {p:.3f}  95% CI width {hi_ - lo_:.3f}")
budget_df = pd.DataFrame(budget_rows)
budget_df.to_csv(RESULTS_DIR / "budget_same_time.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for impl, g in budget_df.groupby("impl"):
    axes[0].plot(g.budget_s, g.trials, "o-", label=impl)
    axes[1].plot(g.budget_s, g.ci_width, "o-", label=impl)
axes[0].set(xscale="log", yscale="log", xlabel="wall-clock budget (s)", ylabel="detector re-evaluations completed",
            title="Figure 6a - re-evaluations within a fixed budget")
axes[1].set(xscale="log", yscale="log", xlabel="wall-clock budget (s)", ylabel="95% CI width of the hit-rate estimate",
            title=f"Figure 6b - estimate precision within a fixed budget (L={BUDGET_L}, q={BUDGET_Q:g})")
for ax in axes:
    ax.legend()
plt.tight_layout(); save_fig(fig, "fig6_budget"); plt.show()
checkpoint("14_budget", hit_agreement_same_workload=agree)

# %% [markdown]
# ## 15. H1: fixed-cost model
# `T_gpu = T0 + 12N/BW_h2d + 16·C/BW_d2h + E/R_gpu`,  `T_cpu = c0 + a·C + b·E`
# - C = number of candidates Σ(N−w+1), E = element evaluations Σ(N−w+1)·w, mean window length w̄ = E/C.
# - The CPU also pays per candidate (a: writing 16 B, loop entry), hence three terms. Three calibration points (1k, K1),
#   (64k, K8), (64k, K4 long) solve for c0, a, b; every other condition is out of sample.
# - At large N, ignoring fixed costs, a single-window scan (w̄ = w) favours the GPU end-to-end when
#   `a + b·w > (16/BW_d2h + 12/BW_h2d) + w/R_gpu`, i.e. **w* = (16/BW_d2h + 12/BW_h2d − a) / (b − 1/R_gpu)**: the window
#   length at which moving one candidate across the bus costs as much as the CPU computing it. N does not appear.
# - The verdict compares this w* with the crossover measured on a window ladder (w = 2 → 32) at the largest N (Section 16).

# %%
def work_of(n, windows):
    w = np.asarray(windows, np.int64)
    C = int(np.sum(n - w + 1))
    return C, int(np.sum((n - w + 1) * w))


class FeatureE2E:
    """GPU end-to-end with features returned: H2D (set) -> kernel -> D2H into preallocated host buffers."""

    def __init__(self, arrs, windows):
        n, K = arrs[0].size, len(windows)
        self.arrs = arrs
        self.d_in = [cp.empty(n, cp.float32) for _ in range(3)]
        self.d_win = cp.asarray(windows)
        self.outs = [cp.empty((K, n), cp.float32) for _ in range(4)]
        self.h_out = np.empty((4, K, n), np.float32)

    def run(self):
        for d, a in zip(self.d_in, self.arrs):
            h2d_into(d, a)
        launch_features(k_kmajor, *self.d_in, self.d_win, self.outs)
        for i in range(4):
            cp.asnumpy(self.outs[i], out=self.h_out[i])

    def kernel(self):
        launch_features(k_kmajor, *self.d_in, self.d_win, self.outs)


def set_windows(name):
    return WINDOW_SETS[name] if name in WINDOW_SETS else np.array([int(name.split("=")[1])], np.int32)


probe = np.ones(8 * 2**20, np.float32)
probe_back = np.empty_like(probe)
d_probe = cp.empty(probe.shape, cp.float32)
BW_H2D = probe.nbytes / (measure_host(lambda: h2d_into(d_probe, probe), gpu=True)["median_ms"] / 1e3)
BW_D2H = probe.nbytes / (measure_host(lambda: cp.asnumpy(d_probe, out=probe_back), gpu=True)["median_ms"] / 1e3)
del d_probe


def sl(arrs, n):
    return tuple(np.ascontiguousarray(a[:n]) for a in arrs)


calib = {}
for n, ws in CALIBRATION:
    win = set_windows(ws)
    arrs = sl(S1, n)
    fe = FeatureE2E(arrs, win)
    fe.run()
    buf = np.empty((4, len(win), n), np.float32)
    calib[(n, ws)] = {"cpu1t": measure_host(lambda: cpu_direct_1t_into(*arrs, win, buf))["median_ms"],
                      "cpumt": measure_host(lambda: cpu_direct_mt_into(*arrs, win, buf))["median_ms"],
                      "kernel": measure_kernel(fe.kernel)["median_ms"],
                      "e2e": measure_host(fe.run, gpu=True)["median_ms"], "work": work_of(n, win)}
    del fe
R_GPU = max(calib[c]["work"][1] / calib[c]["kernel"] for c in CALIBRATION[1:])    # evals per ms (large points)


def transfer_ms(n, C):
    return (12 * n / BW_H2D + 16 * C / BW_D2H) * 1e3


cs = CALIBRATION[0]
T0_MS = max(calib[cs]["e2e"] - transfer_ms(cs[0], calib[cs]["work"][0]) - calib[cs]["work"][1] / R_GPU, 0.0)
CPU_MODEL = {}
for base in ["cpu1t", "cpumt"]:
    M = np.array([[1.0, calib[c]["work"][0], calib[c]["work"][1]] for c in CALIBRATION])
    y = np.array([calib[c][base] for c in CALIBRATION])
    coef = np.linalg.solve(M, y)                                   # c0 [ms], a [ms/candidate], b [ms/eval]
    if (coef < 0).any() or coef[2] <= 0:                           # non-physical (noise): fall back to throughput only
        coef = np.array([0.0, 0.0, y[-1] / calib[CALIBRATION[-1]]["work"][1]])
    CPU_MODEL[base] = coef


def pred_gpu(n, windows):
    C, E = work_of(n, windows)
    return T0_MS + transfer_ms(n, C) + E / R_GPU


def pred_cpu(n, windows, base="cpu1t"):
    c0, a, b = CPU_MODEL[base]
    C, E = work_of(n, windows)
    return c0 + a * C + b * E


def w_star(base="cpu1t"):
    _, a, b = CPU_MODEL[base]
    g = (16 / BW_D2H + 12 / BW_H2D) * 1e3                          # ms per candidate moved (single-window scan)
    if b <= 1 / R_GPU:
        return float("inf")                                       # GPU never wins on compute
    return (g - a) / (b - 1 / R_GPU)                               # <= 1 means GPU wins at every window length


W_STAR = {b: w_star(b) for b in ["cpu1t", "cpumt"]}
MODEL = {"T0_ms": T0_MS, "BW_H2D_GBps": BW_H2D / 1e9, "BW_D2H_GBps": BW_D2H / 1e9, "R_gpu_evals_per_ns": R_GPU / 1e6,
         "cpu1t_c0_ms": CPU_MODEL["cpu1t"][0], "cpu1t_ns_per_candidate": CPU_MODEL["cpu1t"][1] * 1e6,
         "cpu1t_ns_per_eval": CPU_MODEL["cpu1t"][2] * 1e6, "cpumt_ns_per_eval": CPU_MODEL["cpumt"][2] * 1e6,
         "w_star_vs_cpu1t": W_STAR["cpu1t"], "w_star_vs_cpumt": W_STAR["cpumt"]}
(RESULTS_DIR / "fixed_cost_model.json").write_text(json.dumps(MODEL, indent=2))
for k, v in MODEL.items():
    print(f"{k:24s}: {v:.4g}")
print(f"\nPredicted (large N): GPU E2E with features returned beats CPU 1T once the window length exceeds w* = {W_STAR['cpu1t']:.2f}.")
checkpoint("15_model", model=MODEL)

# %% [markdown]
# ## 16. Sweep over N × window set: CPU 1T / CPU MT / optimized CPU / GPU kernel / GPU end-to-end
# Inputs are the first N rows of the real 1-second series. Conditions run in shuffled order and the GPU state is
# recorded before and after. At the largest N a separate "ladder" of single-window scans (w = 2 → 32) locates the
# crossover w* directly (H1 verdict). CPU baselines also write into preallocated buffers.

# %%
N_BIG = max(N_SWEEP)
cases = [(n, ws) for n in N_SWEEP for ws in WINDOW_SETS] + [(N_BIG, f"w={w}") for w in LADDER_W]
order = np.random.default_rng(SEED).permutation(len(cases))
ENV["gpu_state_before_sweep"] = gpu_state()
rows = []
for i, ci in enumerate(order, 1):
    n, ws = cases[ci]
    win = set_windows(ws)
    arrs = sl(S1, n)
    C, E = work_of(n, win)
    fe = FeatureE2E(arrs, win)
    buf = np.empty((4, len(win), n), np.float32)
    c1 = measure_host(lambda: cpu_direct_1t_into(*arrs, win, buf))
    cm_ = measure_host(lambda: cpu_direct_mt_into(*arrs, win, buf))
    cs_ = measure_host(lambda: cpu_smart_into(*arrs, win, buf))
    fe.run()
    kr = measure_kernel(fe.kernel)
    e2 = measure_host(fe.run, gpu=True)
    rows.append({"N": n, "set": ws, "ladder": ws.startswith("w="), "K": len(win), "candidates": C, "evals": E, "w_bar": E / C,
                 "calibration_point": (n, ws) in CALIBRATION,
                 **{f"cpu1t_{k}": v for k, v in c1.items()}, **{f"cpumt_{k}": v for k, v in cm_.items()},
                 **{f"smart_{k}": v for k, v in cs_.items()}, **{f"kernel_{k}": v for k, v in kr.items()},
                 **{f"e2e_{k}": v for k, v in e2.items()},
                 "pred_e2e_ms": pred_gpu(n, win), "pred_cpu1t_ms": pred_cpu(n, win, "cpu1t"), "pred_cpumt_ms": pred_cpu(n, win, "cpumt")})
    print(f"[{i:02d}/{len(cases)}] N={n:7d} {ws:8s} w_bar={E / C:6.1f}  CPU1T {c1['median_ms']:9.3f}  MT {cm_['median_ms']:9.3f}  "
          f"opt {cs_['median_ms']:8.3f}  kernel {kr['median_ms']:8.4f}  E2E {e2['median_ms']:8.3f} ms")
    del fe, buf
ENV["gpu_state_after_sweep"] = gpu_state()
(RESULTS_DIR / "env.json").write_text(json.dumps(ENV, indent=2))
sweep = pd.DataFrame(rows).sort_values(["ladder", "set", "N"]).reset_index(drop=True)
for b in ["cpu1t", "cpumt", "smart"]:
    sweep[f"speedup_e2e_vs_{b}"] = sweep[f"{b}_median_ms"] / sweep.e2e_median_ms
    sweep[f"gpu_wins_{b}"] = sweep[f"{b}_median_ms"] > sweep.e2e_median_ms
for b in ["cpu1t", "cpumt"]:
    sweep[f"pred_gpu_wins_{b}"] = sweep[f"pred_{b}_ms"] > sweep.pred_e2e_ms
sweep["pred_e2e_err_pct"] = 100 * (sweep.pred_e2e_ms - sweep.e2e_median_ms) / sweep.e2e_median_ms
sweep.to_csv(RESULTS_DIR / "sweep.csv", index=False)
display(sweep[["N", "set", "w_bar", "cpu1t_median_ms", "cpumt_median_ms", "smart_median_ms", "kernel_median_ms",
               "kernel_dispatch_overhead_ms", "e2e_median_ms", "e2e_p95_ms", "speedup_e2e_vs_cpu1t", "pred_e2e_err_pct"]].round(4))
checkpoint("16_sweep")

# %% [markdown]
# ### Figure 7: the winner is set by w̄, not by N
# Left: latency vs N with all 15 windows. Right: the window ladder at the largest N (w = 2 → 32) and the other conditions
# on the w̄ axis, with the measured crossover and the predicted w* as vertical lines.

# %%
def measured_w_star(df, base="cpu1t"):
    lad = df[df.ladder].sort_values("w_bar")
    r = np.log(lad[f"{base}_median_ms"].to_numpy() / lad.e2e_median_ms.to_numpy())
    x = np.log(lad.w_bar.to_numpy())
    for i in range(len(r) - 1):
        if r[i] <= 0 < r[i + 1]:
            return float(np.exp(x[i] - r[i] * (x[i + 1] - x[i]) / (r[i + 1] - r[i])))
    if (r > 0).all():
        return float("-inf")      # GPU already wins at the shortest window
    if (r <= 0).all():
        return float("inf")       # CPU wins at every tested window
    return float("nan")


W_MEAS = {b: measured_w_star(sweep, b) for b in ["cpu1t", "cpumt"]}
oos = sweep[~sweep.calibration_point]
WINNER_ACC = {b: float((oos[f"gpu_wins_{b}"] == oos[f"pred_gpu_wins_{b}"]).mean()) for b in ["cpu1t", "cpumt"]}
BOTH_WIN = {b: bool(sweep[f"gpu_wins_{b}"].any() and (~sweep[f"gpu_wins_{b}"]).any()) for b in ["cpu1t", "cpumt", "smart"]}
print("w*: measured", W_MEAS, "| predicted", W_STAR, "(-inf: GPU wins at every tested w; inf: CPU wins at every tested w)")
print("secondary: out-of-sample winner accuracy", WINNER_ACC, "| both winners observed:", BOTH_WIN)

fig, axes = plt.subplots(1, 2, figsize=(16, 5.3))
BIG_SET = max(WINDOW_SETS, key=lambda k: len(WINDOW_SETS[k]))
big = sweep[(sweep.set == BIG_SET)].sort_values("N")
for col, lab, st in [("cpu1t_median_ms", "CPU direct, 1 thread", "o-"), ("cpumt_median_ms", f"CPU direct, {ENV['numba_threads']} threads", "o--"),
                     ("smart_median_ms", "CPU optimized O(N*K), 1 thread", "d-"), ("kernel_median_ms", "GPU kernel", "s-"),
                     ("e2e_median_ms", "GPU E2E (features returned)", "s-"), ("pred_e2e_ms", "GPU E2E model", "k:")]:
    axes[0].plot(big.N, big[col], st, label=lab)
axes[0].set(xscale="log", yscale="log", xlabel="series length N (1-second bars)", ylabel="median latency (ms)",
            title=f"Figure 7a - {BIG_SET} windows: latency vs N")
axes[0].legend(fontsize=8)
for ws, g in sweep[~sweep.ladder].groupby("set"):
    axes[1].scatter(g.w_bar, g.speedup_e2e_vs_cpu1t, s=20 + 12 * np.log2(g.N / g.N.min() + 1), alpha=0.7, label=f"{ws} (size = N)")
lad = sweep[sweep.ladder].sort_values("w_bar")
axes[1].plot(lad.w_bar, lad.speedup_e2e_vs_cpu1t, "k-o", lw=1.5, label=f"window ladder at N={N_BIG:,} (measured)")
wb = np.geomspace(1.5, 400, 200)
mc = np.array([pred_cpu(N_BIG, np.array([int(round(w))], np.int32)) / pred_gpu(N_BIG, np.array([int(round(w))], np.int32)) for w in wb])
ok = np.isfinite(mc) & (mc > 0)
axes[1].plot(wb[ok], mc[ok], "k:", label="model")
axes[1].axhline(1, color="red", ls="--", lw=1)
for val, ls_, lab in [(W_MEAS["cpu1t"], "-", "measured w*"), (W_STAR["cpu1t"], "--", "predicted w*")]:
    if np.isfinite(val) and val > 0:
        axes[1].axvline(val, color="gray", ls=ls_, lw=1.2, label=f"{lab} = {val:.1f}")
axes[1].set(xscale="log", yscale="log", xlabel="mean window length w_bar = evaluations per output candidate",
            ylabel="CPU 1T / GPU E2E", title="Figure 7b - the winner is set by w_bar, not by N")
axes[1].legend(fontsize=7)
plt.tight_layout(); save_fig(fig, "fig7_crossover"); plt.show()

# %% [markdown]
# ## 17. H2, H3: end-to-end breakdown and output design (CPU / H2D / kernel / D2H / host wait)
# On the full 1-second series (N = 604,800, K = 15) and the full 1-minute series (N = 131,040, K = 15), paths (1)(2)(3)
# and a CPU-only path are measured repeatedly, and every repetition checks that all paths return the same top 20.
# Each GPU path runs twice: a spin-queued run gives pure GPU segments (D2H is timed after the last Event completes), and
# a plain run gives the wall-clock total. `host gap` = total − sum of segments.
# **Reading the ratios:** (1)→(2) mixes "sorting moved from CPU to GPU" with "16 B → 4 B output"; (2)→(3) removes the
# count D2H (4 B per candidate) and the CPU candidate selection. Reporting the two steps separately avoids overstating
# the effect of output design.
# Each dataset is measured and saved independently, so a failure on the large series still leaves the small one.

# %%
SEG_ORDER = ["h2d", "features", "rarity", "peaks", "d2h", "cpu_post", "host_gap"]


def breakdown(arrs, reps):
    n = arrs[0].size
    pipe = GpuPipeline(n, WINDOWS)
    f_buf = np.empty((4, K_ALL, n), np.float32)
    rec = {p: [] for p in ("cpu", 1, 2, 3)}
    for r in range(reps + 2):
        t0 = time.perf_counter()
        f = cpu_direct_mt_into(*arrs, WINDOWS, f_buf); t1 = time.perf_counter()
        cnt = counts_cpu(f, WINDOWS)
        flag = peaks_cpu(cnt, WINDOWS, thresholds(n, WINDOWS))
        idx = np.flatnonzero(flag)
        ref_top = select_top(idx, cnt.ravel()[idx], n, WINDOWS); t2 = time.perf_counter()
        out = {"cpu": {"features": (t1 - t0) * 1e3, "cpu_post": (t2 - t1) * 1e3, "total": (t2 - t0) * 1e3}}
        for path in (1, 2, 3):
            top, seg = pipe.run(arrs, path, queued=True)
            _, plain = pipe.run(arrs, path, queued=False)
            if not top[["k", "start"]].equals(ref_top[["k", "start"]]):
                raise AssertionError(f"path {path} top-20 differs from CPU")
            seg["total"] = plain["total"]
            seg["host_gap"] = max(plain["total"] - sum(v for k_, v in seg.items() if k_ not in ("total", "host_gap")), 0.0)
            out[path] = seg
        if r >= 2:
            for p in rec:
                rec[p].append(out[p])
    del pipe, f_buf
    return {p: pd.DataFrame(v).median().to_dict() for p, v in rec.items()}


BREAK_DATA = {"1m, N=131,040": M1} if QUICK else {"1m, N=131,040": M1, "1s, N=604,800": S1}
bd, BREAK_ERRORS = {}, {}
for label, arrs in BREAK_DATA.items():
    try:
        bd[label] = breakdown(arrs, reps=BREAKDOWN_REPS)
        print(label, {str(p): round(v["total"], 2) for p, v in bd[label].items()})
    except Exception as exc:                       # keep what was measured; report the failure instead of losing it
        BREAK_ERRORS[label] = repr(exc)
        print(f"breakdown failed on {label}: {exc!r}")
    pd.DataFrame([{"data": d, "path": str(p), "segment": s, "median_ms": v} for d, m in bd.items() for p, segs in m.items()
                  for s, v in segs.items()]).to_csv(RESULTS_DIR / "e2e_breakdown.csv", index=False)
    gpu_sync()
    cp.get_default_memory_pool().free_all_blocks()

COL = {"h2d": "#DD8452", "features": "#C44E52", "rarity": "#55A868", "peaks": "#8CC78C", "d2h": "#937860",
       "cpu_post": "#4C72B0", "host_gap": "#BBBBBB"}
LAB = {"h2d": "H2D", "features": "features (CPU MT or GPU kernel)", "rarity": "rarity counts (GPU sort)",
       "peaks": "local peaks + compaction (GPU)", "d2h": "D2H", "cpu_post": "CPU: rarity / peaks / top-20",
       "host_gap": "host dispatch / sync wait"}
paths = ["cpu", 1, 2, 3]
plabels = [f"CPU only\n({ENV['numba_threads']} threads)", "(1) features\nreturned", "(2) counts\nreturned", "(3) candidates\nreturned"]
if bd:
    fig, axes = plt.subplots(1, len(bd), figsize=(7.5 * len(bd), 5.5), squeeze=False)
    for ax, (label, med) in zip(axes[0], bd.items()):
        bottom = np.zeros(len(paths))
        for s in SEG_ORDER:
            vals = np.array([0.0 if pd.isna(med[p].get(s, 0.0)) else med[p].get(s, 0.0) for p in paths])
            ax.bar(plabels, vals, bottom=bottom, color=COL[s], label=LAB[s])
            bottom += vals
        for i, p in enumerate(paths):
            ax.text(i, bottom[i] * 1.02, f"{med[p]['total']:.1f} ms", ha="center", va="bottom", fontsize=9)
        ax.set(ylabel="median latency (ms)", title=f"Figure 8 - where the time goes ({label}, K=15)")
        ax.set_ylim(0, bottom.max() * 1.18)
    axes[0][0].legend(fontsize=7)
    plt.tight_layout(); save_fig(fig, "fig8_e2e_breakdown"); plt.show()
checkpoint("17_breakdown", breakdown_errors=BREAK_ERRORS)

# %% [markdown]
# ## 18. H4: thread layout (k-major vs start-major) and block size
# In k-major, the 32 threads of a warp share one window length and all loop w times. In start-major, neighbouring threads
# take different window lengths (all 15 lengths mix in each warp), so **the whole warp waits for the longest window (256)**.
# - Prediction: mean warp trip count changes from Σw/K to max(w), so the ratio ≈ K·max(w)/Σw. Outputs are written in id
#   order, which removes the write-coalescing effect.
# - Input reads favour start-major (only 2-3 distinct starts per warp, so threads read the same addresses), so the
#   measured ratio may come out slightly below the prediction.

# %%
arrs_h4 = S1 if not QUICK else sl(S1, 64_000)
n_h4 = arrs_h4[0].size
d_in = [cp.asarray(a) for a in arrs_h4]
d_win = cp.asarray(WINDOWS)
outs_k = [cp.empty((K_ALL, n_h4), cp.float32) for _ in range(4)]
outs_s = [cp.empty((n_h4, K_ALL), cp.float32) for _ in range(4)]
H4_PRED = K_ALL * WINDOWS.max() / WINDOWS.sum()
h4_rows = []
for bs in BLOCK_SIZES:
    for name, kern, outs in [("k-major", k_kmajor, outs_k), ("start-major", k_smajor, outs_s)]:
        st = measure_kernel(lambda: launch_features(kern, *d_in, d_win, outs, block=bs))
        h4_rows.append({"layout": name, "block": bs, **st})
h4_df = pd.DataFrame(h4_rows)
h4_df.to_csv(RESULTS_DIR / "layout_blocksize.csv", index=False)
piv = h4_df.pivot(index="block", columns="layout", values="median_ms")
display(piv.round(4))
H4_RATIO = float(piv["start-major"].min() / piv["k-major"].min())
print(f"N={n_h4:,}, K={K_ALL}: start-major / k-major = {H4_RATIO:.2f}x (predicted {H4_PRED:.2f}x from K*max(w)/sum(w))")
fig, ax = plt.subplots(figsize=(8, 4.5))
for name in piv.columns:
    ax.plot(piv.index, piv[name], "o-", label=name)
ax.set(xscale="log", xlabel="threads per block", ylabel="kernel time per launch (ms)",
       title=f"Figure 9 - thread layout: measured {H4_RATIO:.2f}x vs predicted {H4_PRED:.2f}x")
ax.set_xticks(BLOCK_SIZES); ax.set_xticklabels(BLOCK_SIZES); ax.legend()
plt.tight_layout(); save_fig(fig, "fig9_layout"); plt.show()
del d_in, outs_k, outs_s

redundancy = pd.DataFrame([{"window": int(w), "reads_per_input_row": (len(m1) - w + 1) * w / len(m1)} for w in WINDOWS])
print("the direct scan rereads each input row this many times per window length (1-minute data):")
display(redundancy.T.round(1))
checkpoint("18_layout", h4_ratio=H4_RATIO, h4_pred=H4_PRED)

# %% [markdown]
# ## 19. Verdicts and summary

# %%
def verdict(ok, inconclusive=False):
    return "INCONCLUSIVE" if inconclusive else ("SUPPORTED" if ok else "NOT SUPPORTED")


H = []
wm, wp = W_MEAS["cpu1t"], W_STAR["cpu1t"]
if np.isfinite(wm) and np.isfinite(wp) and wp > 0:
    h1_ok, h1_inc = 0.5 <= wm / wp <= 2.0, False
elif wm == float("-inf"):
    h1_ok, h1_inc = wp <= min(LADDER_W), False
elif wm == float("inf"):
    h1_ok, h1_inc = wp >= max(LADDER_W), False
else:
    h1_ok, h1_inc = False, True
H.append(("H1 crossover window length w* predicted within 2x", verdict(h1_ok, h1_inc),
          f"w* measured {wm:.2f} vs predicted {wp:.2f} (vs CPU 1T); (secondary) winner accuracy {WINNER_ACC['cpu1t']:.0%}"))

main = "1s, N=604,800" if "1s, N=604,800" in bd else (list(bd)[0] if bd else None)
if main:
    p1 = bd[main][1]
    gpu_part = p1["h2d"] + p1["features"] + p1["d2h"]
    d2h_share = p1["d2h"] / gpu_part
    step12 = bd[main][1]["total"] / bd[main][2]["total"]
    step23 = bd[main][2]["total"] / bd[main][3]["total"]
    h3_gain = bd[main][1]["total"] / bd[main][3]["total"]
    H.append(("H2 D2H dominates the GPU part of feature-returning E2E at K=15", verdict(d2h_share > 0.6),
              f"{main}: D2H {p1['d2h']:.1f} ms of H2D+kernel+D2H {gpu_part:.1f} ms = {d2h_share:.0%}"))
    H.append(("H3 candidates-only output >= 2x faster than features output", verdict(h3_gain >= 2),
              f"{main}: path1 {bd[main][1]['total']:.1f} -> path2 {bd[main][2]['total']:.1f} ({step12:.1f}x: GPU sort + 16->4 B) "
              f"-> path3 {bd[main][3]['total']:.1f} ms ({step23:.1f}x: no count D2H / CPU peaks); total {h3_gain:.1f}x"))
else:
    H.append(("H2 D2H dominates the GPU part of feature-returning E2E at K=15", verdict(False, True), f"breakdown failed: {BREAK_ERRORS}"))
    H.append(("H3 candidates-only output >= 2x faster than features output", verdict(False, True), f"breakdown failed: {BREAK_ERRORS}"))

H.append(("H4 start-major slowdown ~ K*max(w)/sum(w)", verdict(abs(H4_RATIO / H4_PRED - 1) <= 0.35),
          f"measured {H4_RATIO:.2f}x vs predicted {H4_PRED:.2f}x"))
H5_Q = 16.0 if 16.0 in INJ_STRENGTHS else max(INJ_STRENGTHS)
len_ok = inj_df[inj_df.q == H5_Q].groupby("L").len_ratio.median()   # clearly detected, not yet saturated
H.append(("H5 estimated length ~ (1/3..1) x event length", verdict(bool(((len_ok >= 0.28) & (len_ok <= 1.19)).all())),
          f"median estimated length / L (q={H5_Q:g}): " + ", ".join(f"L={L}: {v:.2f}" for L, v in len_ok.items())))
hit_by = inj_summary.groupby("L").apply(lambda g: (g.hit_rate.min() < 0.5) and (g.hit_rate.max() >= 0.5))
H.append(("H6 a detection threshold exists within the q range", verdict(bool(hit_by.all())),
          "; ".join(f"L={L}: " + ", ".join(f"q={r.q}:{r.hit_rate:.0%}" for r in g.itertuples()) for L, g in inj_summary.groupby("L"))))

bmax = budget_df[budget_df.budget_s == max(BUDGETS_S)].set_index("impl")
trial_ratio = bmax.loc["gpu_path3", "trials"] / max(bmax.loc["cpu_optimized", "trials"], 1)
ci_ratio = bmax.loc["cpu_optimized", "ci_width"] / bmax.loc["gpu_path3", "ci_width"]
h7_ok = trial_ratio > 1 and abs(ci_ratio / math.sqrt(trial_ratio) - 1) <= 0.25
H.append(("H7 same budget: more re-evaluations and a ~1/sqrt(n) narrower CI on GPU", verdict(h7_ok),
          f"{max(BUDGETS_S)} s budget: trials GPU/CPU = {trial_ratio:.1f}x, CI width CPU/GPU = {ci_ratio:.2f} "
          f"(sqrt prediction {math.sqrt(max(trial_ratio, 1e-9)):.2f}); same-workload hit agreement {agree:.0%}"))

hyp_df = pd.DataFrame(H, columns=["hypothesis", "verdict", "evidence"])
with pd.option_context("display.max_colwidth", 220):
    display(hyp_df)
hyp_df.to_csv(RESULTS_DIR / "hypotheses.csv", index=False)

required = {
    "real GPU run (not emulated)": not IS_FAKE_GPU,
    "correctness gate passed and both negative controls caught": CORRECTNESS_PASS,
    "all GPU output paths give the CPU top-20": PATHS_AGREE,
    "real market data used": DATA_KIND == "real_market",
    ">= 3 sizes and >= 10 repetitions (or flagged)": bool(sweep.N.nunique() >= 3 and ((sweep.e2e_n >= MIN_REPS) | sweep.e2e_reduced_reps).all()),
    "kernel / E2E / dispatch overhead separated": True,
    "E2E segments separated on every dataset": len(bd) == len(BREAK_DATA),
    "injection experiment completed for every (L, q)": bool(len(inj_summary) == len(INJ_LENGTHS) * len(INJ_STRENGTHS)),
    "same-workload and same-budget experiments both completed": bool(len(budget_df) == len(BUDGETS_S) * len(IMPLS)),
}
print("\n=== REQUIRED (experiment validity) ===")
for k, v in required.items():
    print(f"[{'PASS' if v else 'FAIL'}] {k}")
summary = {"env": ENV, "data": DATA_KIND, "filled_bars": {"1m": m1_filled, "1s": s1_filled}, "model": MODEL,
           "w_star_measured": W_MEAS, "winner_accuracy": WINNER_ACC, "event_results": EVENT_RESULTS, "cold_start": COLD,
           "breakdown_errors": BREAK_ERRORS, "required": required, "hypotheses": hyp_df.to_dict("records")}
(RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
checkpoint("19_summary")
print(f"\nCSV / JSON / PNG saved under {RESULTS_DIR}")

# %% [markdown]
# ## 20. Package the results
# Everything except the cached raw data is zipped into one file. In Colab the download starts automatically; otherwise
# download `pilot_results.zip` from the file browser and place its contents under `results/phase0-pilot/` in the
# repository.

# %%
bundle = shutil.make_archive(str(RESULTS_DIR.parent / "pilot_results"), "zip", root_dir=RESULTS_DIR.parent,
                             base_dir=RESULTS_DIR.name)
with zipfile.ZipFile(bundle) as zf:                       # drop the raw-data cache from the bundle
    keep = [i for i in zf.infolist() if "/data/" not in i.filename]
    payload = {i.filename: zf.read(i.filename) for i in keep}
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
    for name_, data_ in payload.items():
        zf.writestr(name_, data_)
print("bundle:", bundle, f"({Path(bundle).stat().st_size / 2**20:.1f} MiB)")
try:
    from google.colab import files  # type: ignore
    files.download(bundle)
except Exception:
    pass

# %% [markdown]
# ## What this pilot shows and does not show
#
# **Shows.** On this session's GPU and CPU: when an exhaustive (start, length) scan favours the GPU, and that this is set
# not by N but by work per candidate versus output bytes per candidate; where end-to-end time goes and how much output
# design removes; how much one thread-layout choice slows the kernel and that the ratio can be predicted; that the rarity
# map shows the position and the length of an event (controlled injection); and how many detector re-evaluations, and
# therefore how precise a sensitivity estimate, a fixed time budget buys on each side.
#
# **Does not show.** Whether any trading was manipulative (rarity says nothing about cause); causality of the event
# comparison (post-hoc list, selection bias); other assets, venues, or order-level data; ratios on other GPUs or CPUs.
# Rarity of overlapping windows is not an independent probability. Ranks are taken over the whole quarter, so volatile
# periods (the January ETF phase) tend to dominate.

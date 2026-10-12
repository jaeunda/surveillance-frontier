# %% [markdown]
# # Phase 2 GPU pilot on Google Colab
#
# Select **Runtime → Change runtime type → GPU**, check the settings cell below, then **Run all**.
# The notebook calls the repository's `pilot.py`; the tasks, accuracy targets and repetitions of the protocol are used
# unchanged. What a Colab session cannot do is cut **before the first measurement** and recorded in the plan (`--cut`).
#
# - Default plan: **2 sessions** at different times of day; per session T1/T2 cold 3 and warm 3 on every path.
# - A Colab VM has few CPUs (often 2 vCPUs). On it a single CPU run of T2 on the 1-second week is projected at several
#   hours, so fixed-task runs are capped (`RUN_CAP_MINUTES`): a run over the cap is killed and recorded as `timeout`
#   (censored at the cap), and the other runs of that condition are recorded as `not started (run cap)`. A censored
#   condition has no E2E; the cap is only a lower bound, and its time is an (unverified unless checked) projection.
# - `STEP_HOURS` gives the long steps a share of a session's hours, so the CPU baseline cannot use up a session before
#   the GPU steps start.
# - Measurements and inputs live on the VM's `/content`. Every record line is mirrored to Drive as it is written, the
#   output files of every finished run right after it (never inside a timed region), and the whole result folder after
#   each step. A restored run whose output files are missing or changed is listed as `artifact_missing`.
# - Session 2: same `LABEL`, `SESSION = 2`, run again at a different time of day.
# - Resuming after an interruption on the same hardware: run with the same session number; complete steps and a
#   passed P0 check of the same hardware and build are not repeated. A failed, incomplete or interrupted step is not
#   repeated automatically: check it, then list it in `RETRY_STEPS`.
# - If the GPU or CPU changed (a new VM is usually a new assignment), the session is kept as it is and the notebook
#   stops. A further session is a new machine: set `AMEND = {"reason": ..., "set": {"sessions": N}}` (recorded once,
#   before any run of it; the plan's total hours do not grow), then run with `SESSION = N`. It repeats P0 and every step from the start; it is
#   not a checkpoint of the earlier session. Sessions are never pooled.
# - This notebook is for free Colab. A paid runtime is a plan change: record it (amendment with the expected compute
#   units and their price) before the run, and keep it within the project's cost cap.
#
# [Protocol](https://github.com/jaeunda/surveillance-frontier/blob/main/experiments/phase2-gpu/README.md)
# · [Colab resource limits](https://research.google.com/colaboratory/faq.html#resource-limits)
#
# Setup, compilation, downloads and backups are outside every timed region and outside the plan's hours, but they use
# VM time: `_colab/operations` records command times, logs and boot uptime separately. The available hours bound the checks, steps, runs and
# the reference oracle over all sessions (and per session, `SESSION_HOURS`); they do not guarantee how long Colab keeps
# the VM. How long one session needs on a Colab VM has not been measured; the first session's records give it.

# %% [markdown]
# ## 1. Settings
#
# `LABEL` names one pilot; use a new label for a separate experiment. `MODE = "short"` is a 1-session plan without
# `tref-build`. `PLANNED_SESSIONS`, `STEP_HOURS` and the hours are used only when the plan is written; an existing plan
# is never rewritten, and later changes are recorded with `pilot.py amend` before the run they affect. A new plan needs
# `REPO_REF`, the full commit hash of the pushed source to measure (the remote default branch is not assumed to hold
# it); a resumed label is pinned to the commit of its plan. `CUT_REASON` is recorded in the plan whenever the plan is
# reduced (run cap, step shares, short mode).

# %%
LABEL = "colab_phase2_01"  # @param {type:"string"}
SESSION = 1  # @param {type:"integer"}
MODE = "full"  # @param ["full", "short"]
PLANNED_SESSIONS = 2  # @param {type:"integer"}
AVAILABLE_HOURS = 16.0  # @param {type:"number"}
SESSION_HOURS = 8.0  # @param {type:"number"}
RUN_CAP_MINUTES = 15.0  # @param {type:"number"}
STEP_HOURS = {"check": 1.0, "cpu-baseline": 1.5, "gpu-tasks": 2.0, "parity": 1.5}  # includes bounded P0
CUT_REASON = ("Colab VM with few vCPUs: CPU T2 runs are projected at hours each, so fixed-task runs are capped and "
              "a condition over the cap is recorded as censored; the CPU baseline, GPU tasks and parity get a share "
              "of each session so the GPU steps are reached")  # @param {type:"string"}
REPO_URL = "https://github.com/jaeunda/surveillance-frontier.git"
REPO_REF = ""  # @param {type:"string"}
SOURCE_BUNDLE = ""  # optional: /content/phase2-1-source.bundle, uploaded with this notebook; REPO_REF still required
USE_DRIVE = True  # @param {type:"boolean"}
DRIVE_FOLDER = "/content/drive/MyDrive/phase2-pilot"
RETRY_STEPS = []  # e.g. ["gpu-tasks"]: repeat a failed or interrupted step as a new attempt
RERUN_CHECKS = False  # repeat P0 although this session already passed it on the same hardware and build
AMEND = None  # e.g. {"reason": "VM reassigned: session 3 on new hardware", "set": {"sessions": 3}}; recorded once
RUN_MEASUREMENTS = False  # first run ends after P0; inspect evidence, then set True and run section 6

# %% [markdown]
# ## 2. Drive, repository, and restoring an earlier run
#
# Approve the Drive connection in Colab. For a private repository, clone it to `/content/surveillance-frontier`
# first with an authenticated method; never put a token in a notebook cell. Research runs require Drive; local-only
# setup is available with `USE_DRIVE = False`, but section 6 will refuse to collect measurements.

# %%
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", LABEL):
    raise ValueError("LABEL may contain letters, digits, underscore, hyphen and dot.")
if MODE not in ("full", "short"):
    raise ValueError("MODE is full or short.")
if type(SESSION) is not int or SESSION < 1:
    raise ValueError("SESSION is a positive integer (checked against the plan below).")
if MODE == "short":
    PLANNED_SESSIONS = 1
if type(PLANNED_SESSIONS) is not int or PLANNED_SESSIONS < 1:
    raise ValueError("PLANNED_SESSIONS is a positive integer.")
if AMEND is not None and not (isinstance(AMEND, dict) and str(AMEND.get("reason", "")).strip()
                              and isinstance(AMEND.get("set"), dict) and AMEND["set"]):
    raise ValueError('AMEND is None or {"reason": "...", "set": {"key": value}}.')
if not isinstance(STEP_HOURS, dict) or any(not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0)
                                         for v in STEP_HOURS.values()):
    raise ValueError("STEP_HOURS maps step names to positive hours.")
for name, value in (("AVAILABLE_HOURS", AVAILABLE_HOURS), ("SESSION_HOURS", SESSION_HOURS)):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive.")
if not math.isfinite(RUN_CAP_MINUTES) or RUN_CAP_MINUTES < 0:
    raise ValueError("RUN_CAP_MINUTES must be >= 0 (0 = no cap).")
if (MODE == "short" or RUN_CAP_MINUTES > 0 or STEP_HOURS or PLANNED_SESSIONS < 2) and not CUT_REASON.strip():
    raise ValueError("A reduced plan (short mode, a run cap, step shares, one session) needs its reason in CUT_REASON.")

ROOT = Path("/content/surveillance-frontier")
H = ROOT / "experiments/phase2-gpu"
RESULT = H / "results/pilot" / LABEL
BACKUP = Path(DRIVE_FOLDER) / LABEL if USE_DRIVE else None
PY = ROOT / ".venv/bin/python"
BUILD = ROOT / "build"
PLAN = RESULT / "pilot-plan.json"
BUNDLE = RESULT / "_colab/plan.bundle"
STATE = RESULT / "_colab/notebook-state.json"
OP_ID = f"session-{SESSION}-{time.time_ns()}-{uuid.uuid4().hex[:8]}"
OPS = ROOT.parent / "phase2-operations" / LABEL / OP_ID
OPS.mkdir(parents=True)
OP_REMOTE = Path(DRIVE_FOLDER) / "_operations" / LABEL / OP_ID if USE_DRIVE else None
BOOT_ID = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
OPS.joinpath("started.json").write_text(json.dumps({"boot_id": BOOT_ID, "started_unix": time.time(),
    "uptime_s": float(Path("/proc/uptime").read_text().split()[0]), "session": SESSION,
    "python": sys.version, "label": LABEL}, indent=1) + "\n")
# GPU-relevant steps first; on a few-vCPU VM the CPU parity gates take about an hour, so parity runs late and the
# rates, diag and reference rates before it use the threads choice without parity (their config key says so).
STEP_ORDER = ["cpu-baseline", "gpu-rates", "gpu-tasks", "threads", "microbatch", "rates", "diag",
              "tref-rates", "deadline", "parity", "tref-build"]
GPU_ONLY = {"gpu-rates", "gpu-tasks", "microbatch"}
if not isinstance(RETRY_STEPS, list) or any(s not in STEP_ORDER for s in RETRY_STEPS):
    raise ValueError("RETRY_STEPS may only contain step names of STEP_ORDER.")
if any(s not in STEP_ORDER + ["check"] for s in STEP_HOURS):
    raise ValueError("STEP_HOURS may only contain check and step names of STEP_ORDER.")


def kill_runners():
    """Runners start in their own sessions; make sure none survives an interrupted step into the next one."""
    subprocess.run(["pkill", "-KILL", "-f", f"^{ROOT}/build[^ ]*/sf_"], check=False)


def operation(event):
    event = {"wall": time.time(), **event}
    for folder in (OPS, OP_REMOTE):
        if folder is not None:
            folder.mkdir(parents=True, exist_ok=True)
            with open(folder / "commands.jsonl", "a") as f:
                f.write(json.dumps(event) + "\n")
                f.flush()
                os.fsync(f.fileno())


def command(*args, cwd=ROOT, timeout=1800):
    """Stream child output; on an interrupt stop its process group and any runner before saving results."""
    args = [str(a) for a in args]
    print("$", " ".join(args), flush=True)
    cid, started = str(time.time_ns()), time.monotonic()
    operation({"id": cid, "event": "started", "args": args, "timeout_s": timeout})
    log_path = OPS / f"{cid}.log"
    proc = subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, start_new_session=True)
    expired = threading.Event()
    def expire():
        expired.set()
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    timer = threading.Timer(timeout, expire) if timeout else None
    if timer:
        timer.start()
    output = []
    try:
        with open(log_path, "w") as log:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                output.append(line)
                print(line, end="", flush=True)
        rc = proc.wait()
    except BaseException:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
        kill_runners()
        raise
    finally:
        if timer:
            timer.cancel()
        proc.stdout.close()
        operation({"id": cid, "event": "finished", "elapsed_s": time.monotonic() - started,
                   "returncode": proc.poll(), "timeout": expired.is_set(), "log": log_path.name})
        if OP_REMOTE is not None and log_path.exists():
            shutil.copy2(log_path, OP_REMOTE / log_path.name)
    if expired.is_set():
        kill_runners()
        raise subprocess.TimeoutExpired(args, timeout)
    if rc:
        raise subprocess.CalledProcessError(rc, args)
    return "".join(output)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_lines(path):
    """Never discard damaged measurement records to make a resume appear valid."""
    if not Path(path).exists():
        return []
    rows = []
    for i, line in enumerate(Path(path).read_text().splitlines()):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Damaged record: {path}, line {i + 1}. Keep the original and inspect it.") from exc
    return rows


def save_backup():
    if RESULT.exists():
        shutil.copytree(OPS, RESULT / "_colab/operations" / OP_ID, dirs_exist_ok=True)
        write_at = RESULT / "_colab/operations" / OP_ID / "occupancy.json"
        write_at.write_text(json.dumps({"observed_unix": time.time(), "boot_id": BOOT_ID,
            "vm_uptime_s": float(Path("/proc/uptime").read_text().split()[0]),
            "meaning": "VM age includes setup, measurements, backup and idle; not performance E2E or a bill"}) + "\n")
    if BACKUP is not None and RESULT.exists():
        started = time.monotonic()
        receipt = backup_tree(RESULT, BACKUP)
        operation({"event": "backup verified", "elapsed_s": time.monotonic() - started, "receipt": str(receipt)})
        print(f"Drive copy verified: {receipt.name}", flush=True)


if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    operation({"event": "Drive writable", "boot_id": BOOT_ID})
    # p2/common.append_jsonl mirrors every record line under RESULT to the same path under BACKUP
    os.environ["SF_MIRROR_FROM"] = str(RESULT)
    os.environ["SF_MIRROR_TO"] = str(BACKUP)
    os.environ["SF_REQUIRE_MIRROR"] = "1"
else:
    os.environ.pop("SF_MIRROR_FROM", None)
    os.environ.pop("SF_MIRROR_TO", None)
    os.environ.pop("SF_REQUIRE_MIRROR", None)

if not ROOT.exists():
    ROOT.parent.mkdir(parents=True, exist_ok=True)
    restore_bundle = BACKUP / "_colab/plan.bundle" if BACKUP is not None else None
    clone_source = str(restore_bundle) if restore_bundle and restore_bundle.exists() else (SOURCE_BUNDLE or REPO_URL)
    # A plan bundle contains a tracked pilot-plan.json. Delay checkout so RESULT does not appear before restore.
    command("git", "clone", "--no-checkout", clone_source, ROOT, cwd=ROOT.parent)
if not (ROOT / ".git").exists():
    raise RuntimeError(f"{ROOT} must be a git checkout.")

if BACKUP is not None and BACKUP.exists() and not RESULT.exists():
    backup_bundle = BACKUP / "_colab/plan.bundle"
    backup_state = BACKUP / "_colab/notebook-state.json"
    if not backup_bundle.exists() or not backup_state.exists():
        raise RuntimeError("The Drive backup of this LABEL has no commit bundle. Check the backup.")
    # Restore the tracked plan via git first, so an untracked restored plan cannot block checkout.
    command("git", "bundle", "verify", backup_bundle)
    command("git", "fetch", backup_bundle, "HEAD")
    command("git", "checkout", "--detach", read_json(backup_state)["plan_commit"])
    shutil.copytree(BACKUP, RESULT, dirs_exist_ok=True)
    print("Restored earlier results from Drive.")
if PLAN.exists() and BUNDLE.exists() and STATE.exists():
    state = read_json(STATE)
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if current != state["plan_commit"]:
        command("git", "fetch", BUNDLE, "HEAD")
        command("git", "checkout", "--detach", state["plan_commit"])
elif not PLAN.exists():
    if not re.fullmatch(r"[0-9a-f]{40}", REPO_REF):
        raise RuntimeError("A new plan needs REPO_REF = the full source commit hash (pushed or in SOURCE_BUNDLE).")
    if SOURCE_BUNDLE:
        command("git", "bundle", "verify", SOURCE_BUNDLE)
        command("git", "fetch", SOURCE_BUNDLE, "HEAD")
    else:
        command("git", "fetch", "origin", REPO_REF)
    command("git", "checkout", "--detach", REPO_REF)
HEAD = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
print(f"Source revision: {HEAD}")

if not (H / "pilot.py").exists():
    raise RuntimeError("This source version has no Phase 2 runner. Check REPO_REF.")
command("git", "diff", "--exit-code", "HEAD", "--", "CMakeLists.txt", "engine", "experiments/phase2-gpu",
        "cases/finance/fetch_klines.py", ":(exclude)experiments/phase2-gpu/results")
sys.path.insert(0, str(H))
from colab_ops import atomic_copy, audit_results, backup_tree, restore_inputs, seal_archive, verify_archive
problems = audit_results(RESULT)
if problems:
    raise RuntimeError("Restore audit failed; preserve originals and inspect: " + "; ".join(problems[:5]))
allocation = RESULT / f"session-{SESSION}/allocation.json"
if allocation.exists() and read_json(allocation)["boot_id"] != BOOT_ID:
    raise RuntimeError("A new VM (even with the same GPU model) needs a new SESSION and P0. Keep the old records.")
print(f"Pilot {LABEL}, session {SESSION}, local results {RESULT}")

# %% [markdown]
# ## 3. GPU, Python environment, CUDA build, input data
#
# The build targets only the compute capability of the assigned GPU, on this VM. The pinned dependencies go into a
# separate `.venv`, apart from the notebook kernel. A successful build is not a GPU check: the device gates run in P0.

# %%
if sys.version_info < (3, 12):
    raise RuntimeError("Pinned NumPy/SciPy require Python >=3.12. Select a compatible runtime; keep the pins.")
for executable in ("nvidia-smi", "nvcc", "cmake", "c++"):
    if shutil.which(executable) is None:
        raise RuntimeError(f"{executable} not found. Check the GPU runtime and the build tools.")
command("nvidia-smi")
command("nvcc", "--version")
command("cmake", "--version")
command("c++", "--version")
cap = subprocess.check_output([
    "nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"
], text=True).splitlines()[0].strip()
if not re.fullmatch(r"\d+\.\d+", cap):
    raise RuntimeError(f"Could not read the GPU compute capability: {cap!r}")
ARCH = cap.replace(".", "") + "-real"
print(f"CUDA architecture: {ARCH}; CPUs available: {len(os.sched_getaffinity(0))}")

if not PY.exists():
    try:
        command(sys.executable, "-m", "venv", ROOT / ".venv")
    except subprocess.CalledProcessError:
        # some images ship Python without ensurepip; virtualenv brings its own pip
        shutil.rmtree(ROOT / ".venv", ignore_errors=True)
        command(sys.executable, "-m", "pip", "install", "-q", "virtualenv")
        command(sys.executable, "-m", "virtualenv", ROOT / ".venv")
command(PY, "-c", "import sys; print(sys.version); assert sys.version_info >= (3, 12)")
command(PY, "-m", "pip", "install", "--only-binary=:all:", "--report", OPS / "pip-install.json",
        "-r", H / "requirements-colab.lock")
command(PY, "-m", "pip", "check")
OPS.joinpath("pip-freeze.txt").write_text(command(PY, "-m", "pip", "freeze"))

cache = BUILD / "CMakeCache.txt"
if cache.exists() and f"CMAKE_CUDA_ARCHITECTURES:STRING={ARCH}" not in cache.read_text():
    BUILD = ROOT / f"build-colab-{ARCH}"
command("cmake", "-S", ROOT, "-B", BUILD,
        "-DCMAKE_BUILD_TYPE=Release", "-DSF_CUDA=ON",
        f"-DCMAKE_CUDA_ARCHITECTURES={ARCH}")
command("cmake", "--build", BUILD, "-j", min(4, len(os.sched_getaffinity(0))))
if PLAN.exists():
    restore_inputs(ROOT, RESULT, read_json(PLAN), read_json(H / "protocol.json"))
command(PY, ROOT / "cases/finance/fetch_klines.py", "1m-2024q1")
command(PY, ROOT / "cases/finance/fetch_klines.py", "1s-2024-01-08")
command(PY, H / "make_tasks.py", "--check")
command(PY, H / "test_p2.py")
command(PY, H / "test_colab.py")

# Preserve inputs and the actual compiled binaries, not just their paths on an ephemeral VM.
REPRO = RESULT / "_colab/reproduction" / OP_ID
for src in [*sorted((ROOT / "data").glob("BTCUSDT_*.bin")), *sorted(BUILD.glob("sf_*")),
            BUILD / "CMakeCache.txt", H / "requirements-colab.lock"]:
    if src.is_file():
        atomic_copy(src, REPRO / src.name)

# %% [markdown]
# ## 4. P0: write and commit the plan
#
# A new plan is written and **committed locally and backed up to Drive before the first check or measurement**.
# Nothing is pushed to GitHub; the commit is kept as a bundle so that a new VM restores the same code and plan.
# The cold kind is `process-cold` (new process, page cache kept). A restored plan that differs from the settings stops
# here, before any measurement.

# %%
run_cap_s = RUN_CAP_MINUTES * 60 if RUN_CAP_MINUTES > 0 else None
if not PLAN.exists():
    if SESSION != 1:
        raise RuntimeError("Create a new pilot's plan with SESSION = 1.")
    args = [PY, H / "pilot.py", "plan", "--label", LABEL, "--machine", "colab",
            "--hours", AVAILABLE_HOURS, "--session-hours", SESSION_HOURS, "--sessions", PLANNED_SESSIONS,
            "--cold-kind", "process-cold", "--build", BUILD]
    if run_cap_s:
        args += ["--run-cap-min", RUN_CAP_MINUTES]
    if STEP_HOURS:
        args += ["--step-hours", *[f"{k}={v}" for k, v in STEP_HOURS.items()]]
    if MODE == "short":
        args += ["--skip-step", "tref-build"]
    if MODE == "short" or run_cap_s or STEP_HOURS or PLANNED_SESSIONS < 2:
        args += ["--cut", CUT_REASON]
    command(*args)
    command("git", "add", PLAN)
    command("git", "-c", "user.name=Colab Pilot", "-c", "user.email=pilot@localhost",
            "commit", "--only", "-m", f"Record Phase2 Colab pilot plan: {LABEL}", PLAN)

if not BUNDLE.exists() or not STATE.exists():
    # Recover an interrupted plan-creation cell only if the plan is already committed unchanged.
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(PLAN.relative_to(ROOT))],
                             cwd=ROOT, capture_output=True).returncode == 0
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", str(PLAN)],
                                    cwd=ROOT, text=True).strip()
    if not tracked or dirty:
        raise RuntimeError("The plan is not committed. Commit pilot-plan.json and run again.")
    BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if not BUNDLE.exists():
        command("git", "bundle", "create", BUNDLE, "HEAD")
    command("git", "bundle", "verify", BUNDLE)
    STATE.write_text(json.dumps({"label": LABEL, "mode": MODE, "plan_commit": commit}, indent=2) + "\n")

if AMEND is not None and not any(x.get("reason") == AMEND["reason"] and x.get("set") == AMEND["set"]
                                 for x in read_lines(RESULT / "amendments.jsonl")):
    command(PY, H / "pilot.py", "amend", "--label", LABEL, "--reason", AMEND["reason"],
            "--set", *[f"{k}={json.dumps(v)}" for k, v in AMEND["set"].items()])
plan = read_json(PLAN)
for amendment in read_lines(RESULT / "amendments.jsonl"):
    for key, value in amendment["set"].items():
        node = plan
        *parts, last = key.split(".")
        for part in parts:
            node = node.setdefault(part, {})
        node[last] = value
# The settings of a resumed label must match its effective plan (amendments applied); the plan, not the cell, decides.
expected = {"machine": "colab", "hours_available": AVAILABLE_HOURS, "hours_per_session": SESSION_HOURS,
            "run_cap_s": run_cap_s, "step_hours": STEP_HOURS, "sessions": PLANNED_SESSIONS,
            "cold_kind": "process-cold",
            "steps": [s for s in read_json(H / "protocol.json")["pilot"]["steps"]
                      if not (MODE == "short" and s == "tref-build")]}
differ = {k: (plan.get(k), v) for k, v in expected.items() if plan.get(k) != v}
if differ:
    raise RuntimeError(f"The existing plan differs from the settings (plan, settings): {differ}. "
                       "Match the settings to the plan, or record an amendment before the run it affects.")
if SESSION > plan["sessions"]:
    raise RuntimeError(f"SESSION {SESSION} is not in the plan (1..{plan['sessions']}). A further session is a new "
                       'machine: set AMEND = {"reason": "...", "set": {"sessions": N}} first (the total hours stay).')
save_backup()
print(json.dumps({k: plan.get(k) for k in ("label", "sessions", "hours_available", "hours_per_session", "run_cap_s",
                                           "step_hours", "steps", "cold_kind", "cut")}, indent=2))

# %% [markdown]
# ## 5. P0 checks of this session
#
# CPU regression, the Phase 1 hit files, stream, interval and CPU gate checks, and the partial gate of both GPU paths on
# this session's device, each bounded by the plan's remaining hours. The session's CPU baseline and thread candidates
# are derived from this VM's CPUs and recorded before any measurement. If this session already ran on other hardware,
# nothing is overwritten and the notebook stops. A failed GPU gate is recorded; only paths that passed are measured. A
# passed check of this session on the same hardware, code and build is reused unless `RERUN_CHECKS`.

# %%
SD = RESULT / f"session-{SESSION}"
if not allocation.exists():
    allocation.parent.mkdir(parents=True, exist_ok=True)
    allocation.write_text(json.dumps({"boot_id": BOOT_ID, "time_unix": time.time(), "session": SESSION}) + "\n")
env_path = RESULT / f"_colab/operations/{OP_ID}/env-session-{SESSION}.json"
command(PY, H / "collect_env.py", "--profile", "colab", "--build", BUILD, "--out", env_path)
env = read_json(env_path)
if (SD / "env.json").exists():
    old_env = read_json(SD / "env.json")
    if old_env["hardware_fingerprint"] != env["hardware_fingerprint"]:
        raise RuntimeError("The hardware changed. The session is kept; use the next SESSION or a new LABEL.")
if env["code_sha256"] != plan["code_sha256"]:
    raise RuntimeError("The code differs from the plan. Restore the backed-up commit or record an amendment first.")
if env["inputs_sha256"] != plan["inputs_sha256"]:
    raise RuntimeError("The input data differ from the plan's hashes. Check the inputs.")
print(f"GPU: {env['gpu'].get('name')}; CPU: {env['identity'].get('cpu_model')}; "
      f"logical CPUs: {env['topology']['logical']}; MemTotal: {env['meminfo'].get('MemTotal')}")

def latest_checks():
    found = sorted(SD.glob("checks/attempt-*/checks.json"), key=lambda p: int(p.parent.name.split("-")[-1]))
    return (found[-1], read_json(found[-1])) if found else (None, None)


checks_path, checks = latest_checks()
reuse = (not RERUN_CHECKS and checks is not None and checks.get("cpu_pass")
         and checks["hardware_fingerprint"] == env["hardware_fingerprint"]
         and checks["code_sha256"] == env["code_sha256"] and checks["binaries"] == env["build"]["binary_sha256"])
if reuse:
    print(f"P0 checks of this session already passed on this hardware and build: {checks_path.parent}")
else:
    try:
        command(PY, H / "pilot.py", "check", "--label", LABEL, "--session", SESSION, "--build", BUILD, timeout=None)
    except subprocess.CalledProcessError:
        last = (read_lines(SD / "steps.jsonl") or [{}])[-1]
        print(f"P0 checks: {last.get('status')} {last.get('error') or ''}")
        raise
    finally:
        save_backup()
    checks_path, checks = latest_checks()
if not checks["cpu_pass"]:
    raise RuntimeError(f"CPU checks failed: {checks_path.parent / 'checks.log'}")
print("GPU paths validated on this session:", checks["validated_paths"])
if not checks["validated_paths"]:
    print("No GPU path passed. CPU steps run; GPU steps are recorded as not run.")
command(PY, H / "colab_ops.py", "audit", RESULT)
save_backup()
print("P0 recorded. Review checks.json, device gates and the Drive receipt before section 6.")

# %% [markdown]
# ## 6. P1/P2 measurements, backed up after each step
#
# Order: CPU baseline → GPU rates and tasks → threads, microbatch → rates, diag, reference rates → deadline → CPU parity
# → reference build. Parity is late on purpose (its gates are slow on few vCPUs); rates measured before it use the
# threads choice without parity, as their configuration key records. `tref-build` starts only if its conservative
# projection, checked against an observed warm E2E of this session, fits the remaining hours; otherwise its projection
# is the result and the notebook continues. Complete steps are not repeated. `RETRY_STEPS` repeats a
# failed, incomplete or interrupted step **as a whole, as a new attempt**. Each record line and each finished run's
# outputs are mirrored to Drive when written, so a lost VM leaves its started runs visible and the outputs of its
# finished runs; a run in progress at the loss is `lost`, and its time is bounded by the last heartbeat.

# %%
if RUN_MEASUREMENTS and not USE_DRIVE:
    raise RuntimeError("Phase 2-1 measurements require a writable, verified Drive backup. Enable USE_DRIVE.")
if not RUN_MEASUREMENTS:
    print("P0-only run: inspect section 5, then set RUN_MEASUREMENTS = True and run this cell.")
command(PY, H / "colab_ops.py", "audit", RESULT)

def step_state(rows, step):
    relevant = [r for r in rows if r.get("step") == step]
    if not relevant:
        return "new"
    last = relevant[-1]
    if last.get("event") == "started":
        return "unfinished"
    return last.get("status", "unknown")


def time_used_up(status):
    return status == "not started (time)" or status.startswith("stopped (time)")


for step in STEP_ORDER if RUN_MEASUREMENTS else []:
    if step not in plan["steps"]:
        print(f"Not in the plan: {step}")
        continue
    before = step_state(read_lines(SD / "steps.jsonl"), step)
    budget = command(PY, H / "colab_ops.py", "time-action", RESULT, "--session", SESSION, "--step", step).strip()
    if budget == "stop session":
        print("Session or label hours exhausted. Run sections 7 and 8 to preserve the partial results.")
        break
    if budget == "skip step" and before != "new":
        print(f"{step}: its step share is exhausted; continuing to the next planned step.")
        continue
    if before == "complete":
        print(f"Already complete: {step}")
        continue
    if step in GPU_ONLY and not checks["validated_paths"]:
        if before == "new":
            try:
                command(PY, H / "pilot.py", "run", "--label", LABEL,
                        "--session", SESSION, "--step", step, "--build", BUILD, timeout=None)
            except subprocess.CalledProcessError:
                recorded = step_state(read_lines(SD / "steps.jsonl"), step)
                if recorded != "not run (no GPU path passed the device gate)":
                    raise
            finally:
                save_backup()
        print(f"No validated GPU path: {step}")
        continue
    if before != "new" and step not in RETRY_STEPS:
        if before.startswith(("not started", "not run (")) and not time_used_up(before):
            print(f"Recorded earlier as '{before}': {step}")
            continue
        raise RuntimeError(f"{step}: {before}. Check the cause, then add the step to RETRY_STEPS and run again.")
    print(f"\nSession {SESSION}: {step}", flush=True)
    failure = None
    try:
        command(PY, H / "pilot.py", "run", "--label", LABEL,
                "--session", SESSION, "--step", step, "--build", BUILD, timeout=None)
    except subprocess.CalledProcessError as exc:
        failure = exc
    finally:
        save_backup()
    after = step_state(read_lines(SD / "steps.jsonl"), step)
    if time_used_up(after):
        budget = command(PY, H / "colab_ops.py", "time-action", RESULT, "--session", SESSION, "--step", step).strip()
        if budget == "stop session":
            print("Session or label hours exhausted. Summarising the partial results.")
            break
        print(f"{step}: its time box ended; the session has time for the next planned step.")
        continue
    if after.startswith(("not started", "not run (")):
        print(f"{step}: {after} (recorded; continuing)")  # e.g. tref-build whose projection does not fit
        continue
    if failure is not None:
        raise failure
    if after != "complete":
        raise RuntimeError(f"{step}: {after}. Check steps.jsonl and the step's logs.")

# %% [markdown]
# ## 7. Summary and figures
#
# This cell can be run on its own to see the results so far, also after a stop. Per session: median and range, CPU and
# GPU times, memory, failures, output identity, and the projections (including the reference build). The pilot never
# gives a formal H3/H4 verdict.

# %%
from IPython.display import Markdown, Image, display

try:
    command(PY, H / "pilot.py", "summary", "--label", LABEL, "--build", BUILD)
    command(PY, H / "make_figures.py", "pilot", LABEL)
finally:
    save_backup()
display(Markdown((RESULT / "pilot-summary.md").read_text()))
figure = RESULT / "figs/pilot_microbatch.png"
if figure.exists():
    display(Image(filename=str(figure)))
print("Local results:", RESULT)
print("Drive results:", BACKUP if BACKUP else "Drive backup not used")

# %% [markdown]
# ## 8. Seal and verify the results; download a second copy
#
# Run this cell also after a stop, before deleting the VM. It creates an immutable archive and SHA-256 sidecar,
# verifies the Drive copy, and permits a second copy to be downloaded. The archive
# includes the plan's commit bundle, input bytes, binaries, setup/P0 logs and raw measurement outputs. The next session uses the same `LABEL` and
# `SESSION = 2`, at a different time of day, from section 1.

# %%
DOWNLOAD_RESULTS = False  # @param {type:"boolean"}
try:
    save_backup()
except (OSError, ValueError) as exc:
    print(f"Backup failed: {exc}. Sealing local evidence for download; no verified Drive backup is claimed.")
archive = seal_archive(RESULT, ROOT.parent / "phase2-archives")
checksum = archive.with_name(archive.name + ".sha256")
if BACKUP is not None:
    saved = BACKUP.parent / "_archives" / LABEL / archive.name
    try:
        atomic_copy(archive, saved)
        atomic_copy(checksum, saved.with_name(saved.name + ".sha256"))
        verify_archive(saved)
        print("Verified archive on Drive:", saved)
    except (OSError, ValueError) as exc:
        print(f"Archive not verified on Drive: {exc}. Download the local archive and checksum before VM deletion.")
print("Archive SHA-256:", verify_archive(archive))
if DOWNLOAD_RESULTS:
    from google.colab import files
    files.download(str(archive))
    files.download(str(checksum))

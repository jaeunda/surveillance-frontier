"""Paths, the protocol, hashing, append-only records, git state, and command logging."""

import datetime as dt
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]  # experiments/phase2-gpu
ROOT = HERE.parents[1]
TASKS = HERE / "tasks"
RESULTS = HERE / "results"
PROTOCOL = json.loads((HERE / "protocol.json").read_text())
FORMAL = PROTOCOL["formal"]
PILOT = PROTOCOL["pilot"]
SCALING = PROTOCOL["scaling"]
MACHINES = PROTOCOL["machines"]
GRID_OF_TASK = {"T1-curve": "P1", "T2-policy-map": "P24", "T-ref": "P24", "T-ref-est": "P24"}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str) + "\n")


def append_jsonl(path, rec):
    """One record per line, appended and fsynced at once (callers write a 'started' record before a run and the
    terminal record after it, so a run cut off by an interruption still leaves its start)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def sh(cmd, timeout=120):
    """stdout of a probe command, or None if it is unavailable or fails."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def run_logged(cmd, log, **kw):
    """Run a command from the repository root and append its command line, output and exit code to log."""
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, cwd=ROOT, **kw)
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(f"$ {' '.join(map(str, cmd))}\n{r.stdout}\n{r.stderr}\nexit {r.returncode}\n")
    return r


def git_state():
    commit = sh(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    status = sh(["git", "-C", str(ROOT), "status", "--porcelain"])
    return {"commit": commit, "dirty": bool(status)}


def committed_unchanged(path):
    """True if path is tracked and has no uncommitted changes (records that gate later work must be committed)."""
    rel = str(Path(path).resolve().relative_to(ROOT))
    tracked = sh(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", rel]) is not None
    changed = sh(["git", "-C", str(ROOT), "status", "--porcelain", "--", rel])
    return tracked and not changed


# Files whose content defines what is measured: the engine, the build, the experiment code, the protocol, the tasks.
# Documentation and results are not part of it.
CODE_PATHS = ["CMakeLists.txt", "engine", "experiments/phase2-gpu"]
CODE_SUFFIXES = (".cpp", ".hpp", ".cu", ".inl", ".h", ".txt", ".py", ".sh", ".json", ".cmake")


def code_fingerprint():
    """SHA-256 over (path, content) of every tracked code file (CODE_PATHS, CODE_SUFFIXES; results/ excluded), as
    read from the working tree. Gates, calibrations, the freeze and every launch record it; later steps compare it."""
    listed = sh(["git", "-C", str(ROOT), "ls-files", "--", *CODE_PATHS]) or ""
    h = hashlib.sha256()
    for rel in sorted(listed.splitlines()):
        if "/results/" in rel or not rel.endswith(CODE_SUFFIXES):
            continue
        p = ROOT / rel
        h.update(rel.encode() + b"\0" + (sha256_file(p).encode() if p.exists() else b"missing") + b"\n")
    return h.hexdigest()


def new_attempt_dir(parent):
    """parent/attempt-<k> for the next unused k, created empty. Attempts are never reused or overwritten."""
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    k = 1
    while (parent / f"attempt-{k}").exists():
        k += 1
    d = parent / f"attempt-{k}"
    d.mkdir()
    return d


def fresh_dir(path):
    """Create path; refuse an existing non-empty directory (an output is never overwritten)."""
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"{path} exists and is not empty")
    path.mkdir(parents=True, exist_ok=True)
    return path


def geomean(values):
    values = list(values)
    return math.exp(sum(math.log(v) for v in values) / len(values))


def binary_hashes(build):
    return {b: sha256_file(Path(build) / b) for b in ("sf_run", "sf_gate", "sf_test", "sf_test2", "sf_bench")
            if (Path(build) / b).exists()}


def input_hashes():
    return {s: sha256_file(ROOT / v["file"]) if (ROOT / v["file"]).exists() else None
            for s, v in PROTOCOL["series"].items()}


def task_hashes():
    return {str(p.relative_to(TASKS)): sha256_file(p) for p in sorted(TASKS.rglob("*.json"))}

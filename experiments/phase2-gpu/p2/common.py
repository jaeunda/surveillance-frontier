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
    """One record per line, fsynced: a run that was started is never lost from the record."""
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

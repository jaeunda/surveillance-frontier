"""Paths, the protocol, hashing, append-only records, git state, and command logging."""

import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
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
    terminal record after it, so a run cut off by an interruption still leaves its start). If SF_MIRROR_FROM and
    SF_MIRROR_TO are set, a record of a file under SF_MIRROR_FROM is also appended to the same relative path under
    SF_MIRROR_TO (e.g. Drive on Colab, so a lost VM still leaves its records); appends never happen inside a timed
    region. SF_REQUIRE_MIRROR=1 makes a mirror failure fatal after preserving the local line (Colab research runs);
    otherwise it is reported as a warning."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(rec, default=str) + "\n"
    with open(path, "a") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())
    src, dst = os.environ.get("SF_MIRROR_FROM"), os.environ.get("SF_MIRROR_TO")
    if src and dst:
        try:
            rel = path.resolve().relative_to(Path(src).resolve())
        except ValueError:
            return
        try:
            m = Path(dst) / rel
            m.parent.mkdir(parents=True, exist_ok=True)
            with open(m, "a") as f:
                f.write(line)
        except OSError as e:
            if os.environ.get("SF_REQUIRE_MIRROR") == "1":
                raise OSError(f"Required record mirror failed: {dst}") from e
            print(f"warning: record mirror {dst} failed: {e}", file=sys.stderr)


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


def run_logged(cmd, log, timeout=None):
    """Run a command from the repository root in its own process group and append its command line, output and exit
    code to log. With a timeout the whole group (children included) is killed when it expires and
    subprocess.TimeoutExpired is raised after the log is written; an interrupt kills the group too."""
    cmd = [str(c) for c in cmd]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=ROOT,
                            start_new_session=True)
    expired, interrupted = False, None
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        expired = True
        _killpg(proc)
        out, err = proc.communicate()
    except BaseException as exc:
        interrupted = exc
        _killpg(proc)
        out, err = proc.communicate()
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(f"$ {' '.join(cmd)}\n{out}\n{err}\n{'killed at timeout ' + str(timeout) + ' s' if expired else ''}"
                f"\nexit {proc.returncode}; interrupted={type(interrupted).__name__ if interrupted else None}\n")
        f.flush()
        os.fsync(f.fileno())
    if interrupted is not None:
        raise interrupted
    if expired:
        raise subprocess.TimeoutExpired(cmd, timeout, out, err)
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


def _killpg(proc):
    try:
        os.killpg(proc.pid, 9)
    except OSError:
        pass


def manifest(path):
    """{relative file: SHA-256} of every file under a run's output directory (the artifact a record refers to)."""
    path = Path(path)
    if not path.is_dir():
        return {}
    return {str(f.relative_to(path)): sha256_file(f) for f in sorted(path.rglob("*")) if f.is_file()}


def mirror_tree(path):
    """Copy a finished directory under SF_MIRROR_FROM to the same relative path under SF_MIRROR_TO (outside any
    timed region), so a lost VM keeps the outputs of every finished run, not only its records. Returns True if
    copied; SF_REQUIRE_MIRROR=1 propagates failure, otherwise a failure is reported as a warning."""
    import shutil
    src, dst = os.environ.get("SF_MIRROR_FROM"), os.environ.get("SF_MIRROR_TO")
    if not (src and dst) or not Path(path).is_dir():
        return False
    try:
        rel = Path(path).resolve().relative_to(Path(src).resolve())
    except ValueError:
        return False
    try:
        shutil.copytree(path, Path(dst) / rel, dirs_exist_ok=True)
        return True
    except OSError as e:
        if os.environ.get("SF_REQUIRE_MIRROR") == "1":
            raise OSError(f"Required output mirror failed: {dst}") from e
        print(f"warning: output mirror {dst} failed: {e}", file=sys.stderr)
        return False


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
CODE_SUFFIXES = (".cpp", ".hpp", ".cu", ".inl", ".h", ".txt", ".py", ".sh", ".json", ".cmake", ".lock")


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

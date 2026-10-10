"""External harness for sf_run (SPEC §8): back-end configurations, the end-to-end clock, cold / warm / deadline runs,
steady-state benches, and resource records.

The harness owns the timing. t0 = time.monotonic_ns() (CLOCK_MONOTONIC, the clock sf_run stamps its stages with) is
taken immediately before spawn (cold, deadline) or before the request is sent (warm); t1 at waitpid or when the
acknowledgement (sent after the output fsync) arrives. Everything else happens outside [t0, t1]:

  cold      sync and drop the page cache if permitted ("cache-cold"); otherwise the run is labelled "process-cold"
            and never aggregated with cache-cold runs. Each run gets its own cgroup v2 group if permitted
  warm      a server (sf_run --serve) primed with a task of the same base-state key, outside the timing
  deadline  t0 + B is passed to sf_run as an absolute deadline; a run is late if it exits after it
  sampler   1 Hz on the harness side: MemAvailable, swap, major faults, CPU steal, per-core MHz, and on GPU machines
            nvidia-smi clocks, memory and power
"""

import json
import os
import signal
import socket
import statistics
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

from .common import ROOT, append_jsonl

CGROUP_ROOT = Path("/sys/fs/cgroup")
OMP_ENV = {"OMP_PLACES": "cores", "OMP_PROC_BIND": "spread"}


# ---- back-end configurations ----

def cpu_config(threads, cpus=None, parity="none", warmup=0, name=None):
    return {"device": "cpu", "threads": int(threads), "cpus": cpus, "parity": parity or "none",
            "warmup_trials": int(warmup), "name": name}


def gpu_config(path, microbatch, host_threads, submission=1, warmup=0, device="cuda"):
    return {"device": device, "path": path, "microbatch": microbatch, "submission": int(submission),
            "threads": int(host_threads), "warmup_trials": int(warmup)}


def config_key(c):
    """A stable short name of a configuration, e.g. cpu:t6@0-5:none or cuda:inc:b256."""
    if c["device"] == "cpu":
        pin = f"@{c['cpus']}" if c.get("cpus") else ""
        return f"cpu:t{c['threads']}{pin}:{c.get('parity') or 'none'}"
    return f"{c['device']}:{c['path']}:b{c['microbatch']}"


def sf_args(c):
    """sf_run options of a configuration."""
    a = ["--device", c["device"], "--threads", str(c["threads"]), "--warmup-trials", str(c.get("warmup_trials", 0))]
    if c["device"] == "cpu":
        a += ["--parity", c.get("parity") or "none"]
    else:
        mb = c["microbatch"]
        a += ["--path", c["path"], "--microbatch", "auto" if mb in (None, "auto", 0) else str(mb),
              "--submission", str(c.get("submission", 1))]
    return a


def launcher(c):
    """Command prefix and environment: CPU pinning by taskset (pilot topology sets), OpenMP placement."""
    prefix = ["taskset", "-c", c["cpus"]] if c.get("cpus") else []
    return prefix, {**os.environ, **OMP_ENV}


def parse_device_info(text):
    """'NAME path=inc microbatch=64 b_max=..' -> {'name': NAME, 'path': 'inc', 'microbatch': 64, ...}."""
    out, name = {}, []
    for tok in (text or "").split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
        elif not out:
            name.append(tok)
    out["name"] = " ".join(name)
    return out


# ---- system probes ----

def drop_caches():
    os.sync()
    try:
        Path("/proc/sys/vm/drop_caches").write_text("3\n")
        return "cache-cold"
    except OSError:
        r = subprocess.run(["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"], capture_output=True)
        return "cache-cold" if r.returncode == 0 else "process-cold"


class Cgroup:
    """A cgroup v2 group for one run; memory.peak is read when the run ends (needs a writable cgroup tree)."""

    def __init__(self):
        self.path = None
        if not ((CGROUP_ROOT / "cgroup.controllers").exists() and os.access(CGROUP_ROOT, os.W_OK)):
            return
        parent = CGROUP_ROOT / "sf-runs"
        try:
            parent.mkdir(exist_ok=True)
            for ctl in (CGROUP_ROOT / "cgroup.subtree_control", parent / "cgroup.subtree_control"):
                if "memory" not in ctl.read_text():
                    ctl.write_text("+memory")
            self.path = parent / f"run-{uuid.uuid4().hex[:12]}"
            self.path.mkdir()
        except OSError:
            self.path = None

    def preexec(self):
        procs = None if self.path is None else str(self.path / "cgroup.procs")

        def join():
            os.setsid()  # own process group, so a timeout kills every child
            if procs:
                with open(procs, "w") as f:
                    f.write(str(os.getpid()))
        return join

    def peak_bytes(self):
        try:
            return int((self.path / "memory.peak").read_text()) if self.path else None
        except (OSError, ValueError):
            return None

    def remove(self):
        if self.path is not None:
            try:
                self.path.rmdir()
            except OSError:
                pass


def _vmstat():
    out = {}
    for line in Path("/proc/vmstat").read_text().splitlines():
        k, v = line.split()
        if k in ("pswpin", "pswpout", "pgmajfault"):
            out[k] = int(v)
    return out


def _steal():
    f = Path("/proc/stat").read_text().splitlines()[0].split()
    return int(f[8]) if len(f) > 8 else 0


def _mem_available_kib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    return None


def _core_mhz():
    return [float(line.split(":")[1]) for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("cpu MHz")]


def _summary(v):
    return {"min": min(v), "median": statistics.median(v), "max": max(v)} if v else None


class Sampler:
    """1 Hz samples on the harness side."""

    def __init__(self, gpu=False):
        self.gpu = gpu
        self.stop_flag = threading.Event()
        self.mem, self.mhz, self.smi_lines = [], [], []
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.smi = None

    def _loop(self):
        while not self.stop_flag.is_set():
            self.mem.append(_mem_available_kib())
            mhz = _core_mhz()
            if mhz:
                self.mhz.append(statistics.median(mhz))
            self.stop_flag.wait(1.0)

    def __enter__(self):
        self.vm0, self.steal0 = _vmstat(), _steal()
        if self.gpu:
            try:
                self.smi = subprocess.Popen(
                    ["nvidia-smi", "--query-gpu=timestamp,clocks.sm,clocks.mem,memory.used,power.draw,utilization.gpu",
                     "--format=csv,noheader,nounits", "-lms", "1000"], stdout=subprocess.PIPE, text=True)
                threading.Thread(target=lambda: self.smi_lines.extend(self.smi.stdout), daemon=True).start()
            except OSError:
                self.smi = None
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop_flag.set()
        self.thread.join()
        self.vm1, self.steal1 = _vmstat(), _steal()
        if self.smi:
            self.smi.terminate()
            self.smi.wait()

    def record(self):
        mem = [m for m in self.mem if m is not None]
        rec = {"min_mem_available_mib": min(mem) / 1024 if mem else None,
               "pswpin": self.vm1.get("pswpin", 0) - self.vm0.get("pswpin", 0),
               "pswpout": self.vm1.get("pswpout", 0) - self.vm0.get("pswpout", 0),
               "major_faults_system": self.vm1.get("pgmajfault", 0) - self.vm0.get("pgmajfault", 0),
               "steal_jiffies": self.steal1 - self.steal0, "core_mhz_median": _summary(self.mhz),
               "samples": len(self.mem)}
        if self.gpu:
            rows = [line.strip().split(", ") for line in self.smi_lines if line.count(",") >= 5]

            def col(i):
                vals = []
                for r in rows:
                    try:
                        vals.append(float(r[i]))
                    except (ValueError, IndexError):
                        pass
                return vals
            rec["gpu"] = {"sm_mhz": _summary(col(1)), "mem_mhz": _summary(col(2)),
                          "memory_used_mib_max": max(col(3)) if col(3) else None, "power_w": _summary(col(4)),
                          "utilization": _summary(col(5)), "samples": len(rows)}
        return rec


# ---- runs ----

def _stages(out_dir):
    try:
        return json.loads((Path(out_dir) / "stages.json").read_text())
    except (OSError, ValueError):
        return None


def _runner_record(st):
    if not st:
        return None
    b = st["backend"]
    rec = {k: st.get(k) for k in ("device_info", "config", "warm", "trials", "completed", "width_ok",
                                   "worst_half_width", "rusage", "warmup_trials", "base_key_sha256", "stages_ns")}
    rec["device"] = parse_device_info(st.get("device_info"))
    rec["affinity"] = [(w["cpu_start"], w["affinity_start"], w["cpu_end"], w["affinity_end"]) for w in b["workers"]]
    rec["per_worker_trials"] = [w["trials"] for w in b["workers"]]
    rec["stage_ms_sum"] = b["stage_ms_sum"]
    rec.update({"bytes_h2d": b["bytes_h2d"], "bytes_d2h": b["bytes_d2h"], "peak_device_bytes": b["peak_device_bytes"]})
    return rec


def _record(mode, condition, config, cmd, t0, t1, status, out_dir, extra, sampler, cg_peak, expected_sha):
    st = _stages(out_dir)
    rec = {"mode": mode, **condition, "config": config, "config_key": config_key(config), "cmd": cmd,
           "t0_ns": t0, "t1_ns": t1, "e2e_s": (t1 - t0) / 1e9, "status": status, "out": str(out_dir),
           "result_sha256": st["result_sha256"] if st else None, "runner": _runner_record(st),
           "cgroup_memory_peak_bytes": cg_peak, **sampler.record(), **extra}
    if expected_sha:
        rec["identity_ok"] = rec["result_sha256"] == expected_sha
        if status == "ok" and not rec["identity_ok"]:
            rec["status"] = "identity_mismatch"
    return rec


def run_cold(task, out_dir, config, condition=None, *, build, timeout_s=None, expected_sha=None, budget_s=None,
             drain_ms=0, replicate=None, drop=True):
    """One cold run (deadline run with budget_s). Returns the run record; never raises for a failed run."""
    out_dir, task = Path(out_dir).resolve(), Path(task).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cold = drop_caches() if drop else "process-cold"
    cg = Cgroup()
    prefix, env = launcher(config)
    base = [*prefix, str(Path(build) / "sf_run"), str(task), "--out", str(out_dir), *sf_args(config)]
    if replicate is not None:
        base += ["--replicate", str(replicate)]
    gpu = config["device"] == "cuda"
    with Sampler(gpu=gpu) as smp:
        t0 = time.monotonic_ns()
        deadline = t0 + int(budget_s * 1e9) if budget_s is not None else None
        cmd = base + (["--deadline-ns", str(deadline), "--drain-ms", str(drain_ms)] if deadline else [])
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=ROOT, env=env,
                                preexec_fn=cg.preexec())
        try:
            _, err = proc.communicate(timeout=timeout_s)
            t1 = time.monotonic_ns()
            status = "ok" if proc.returncode == 0 else ("width_fail" if proc.returncode == 3 else "error")
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            _, err = proc.communicate()
            t1 = t0 + int(timeout_s * 1e9)  # censored at exactly the timeout
            status = "timeout"
    if deadline is not None and status == "ok" and t1 > deadline:
        status = "late"
    rec = _record("deadline" if deadline else "cold", condition or {}, config, cmd, t0, t1, status, out_dir,
                  {"cold_kind": cold, "returncode": proc.returncode, "stderr_tail": err[-2000:],
                   "deadline_ns": deadline, "budget_s": budget_s, "drain_ms": drain_ms if deadline else None,
                   "replicate": replicate}, smp, cg.peak_bytes(), expected_sha)
    cg.remove()
    return rec


class WarmServer:
    """sf_run --serve in its own cgroup; each timed request is one warm run."""

    def __init__(self, config, *, build):
        self.config = config
        self.sock = str(Path(tempfile.gettempdir()) / f"sf_run_{uuid.uuid4().hex[:8]}.sock")
        self.cg = Cgroup()
        prefix, env = launcher(config)
        self.proc = subprocess.Popen([*prefix, str(Path(build) / "sf_run"), "--serve", self.sock, *sf_args(config)],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=ROOT, env=env,
                                     preexec_fn=self.cg.preexec())
        line = self.proc.stdout.readline()
        if "serving" not in line:
            raise RuntimeError(f"warm server did not start: {line} {self.proc.stderr.read()[-2000:]}")

    def request(self, payload):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(self.sock)
        s.sendall((json.dumps(payload) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
        s.close()
        return json.loads(buf.decode())

    def prime(self, task, out_dir):
        """A first task outside the timing; its base-state key matches the timed task."""
        r = self.request({"task": str(Path(task).resolve()), "out": str(Path(out_dir).resolve())})
        if r.get("status") != "ok":
            raise RuntimeError(f"warm priming failed: {r}")
        return r

    def timed(self, task, out_dir, condition=None, expected_sha=None, timeout_s=None):
        out_dir, task = Path(out_dir).resolve(), Path(task).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        with Sampler(gpu=self.config["device"] == "cuda") as smp:
            t0 = time.monotonic_ns()
            r = self.request({"task": str(task), "out": str(out_dir)})
            t1 = time.monotonic_ns()
        status = r.get("status", "error")
        if timeout_s is not None and (t1 - t0) / 1e9 > timeout_s:
            status, t1 = "timeout", t0 + int(timeout_s * 1e9)
        if status == "ok" and not r.get("base_cached"):
            status = "not_warm"  # the server rebuilt the base state: not a warm run
        return _record("warm", condition or {}, self.config, ["warm", str(task)], t0, t1, status, out_dir,
                       {"base_cached": r.get("base_cached"), "reply": r}, smp, self.cg.peak_bytes(), expected_sha)

    def close(self):
        try:
            self.request({"cmd": "quit"})
            self.proc.wait(timeout=60)
        except (OSError, subprocess.TimeoutExpired, ValueError):
            self.proc.kill()
        self.cg.remove()


def bench(task, config, seconds, *, build, sample=False, diag=False):
    """Steady-state throughput of one task point (sf_run --bench). Returns the bench record or, on failure, a record
    with status 'error' and the tail of stderr (an allocation failure is a result, not an exception)."""
    prefix, env = launcher(config)
    cmd = [*prefix, str(Path(build) / "sf_run"), str(task), "--bench", str(seconds), *sf_args(config)]
    if diag:
        cmd.append("--diag")
    with Sampler(gpu=sample and config["device"] == "cuda") as smp:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=env)
    rec = {"task": str(Path(task).relative_to(ROOT)) if Path(task).is_absolute() else str(task), "config": config,
           "config_key": config_key(config), "seconds": seconds, "returncode": r.returncode, "sampler": smp.record()}
    if r.returncode == 0:
        b = json.loads(r.stdout.strip().splitlines()[-1])
        rec.update({"status": "ok", "trials_per_s": b["trials_per_s"], "setup_s": b["setup_s"], "bench": b,
                    "device": parse_device_info(b.get("device_info"))})
    else:
        err = r.stderr[-2000:]
        oom = any(s in err for s in ("out of memory", "cudaMalloc", "no microbatch fits", "allocation failed"))
        rec.update({"status": "allocation_failure" if oom else "error", "stderr_tail": err, "trials_per_s": None})
    return rec


def append(path, rec):
    append_jsonl(path, rec)

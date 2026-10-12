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

Records: before a run starts, a 'started' line (run id, kind, condition) goes to the attempt log (ATTEMPT_LOG, set
by the caller); the terminal record carries the same run id. A run that never got a terminal record (lost connection,
OOM kill, spot interruption) is therefore visible as started-but-unfinished. Every run is bounded: by its timeout, by
the caller's time box (TIME_BOX_NS, CLOCK_MONOTONIC), or both; warm requests, priming and benches included. A failure
of the harness itself becomes a record with status 'harness_error', never a lost run.
"""

import datetime as dt
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

from .common import ROOT, append_jsonl, fresh_dir, mirror_tree

CGROUP_ROOT = Path("/sys/fs/cgroup")
OMP_ENV = {"OMP_PLACES": "cores", "OMP_PROC_BIND": "spread"}
ATTEMPT_LOG = None  # path of the 'started' records; set by the pilot session or the launch
TIME_BOX_NS = None  # CLOCK_MONOTONIC ns after which no run starts and every running one is stopped
BENCH_SETUP_ALLOWANCE_S = 900  # a bench is killed after its seconds plus this setup allowance
WARM_CONTROL_TIMEOUT_S = 120  # quit request of a warm server


class TimeBox(Exception):
    """The caller's time box is used up: no further run starts."""


def time_left_s():
    return None if TIME_BOX_NS is None else (TIME_BOX_NS - time.monotonic_ns()) / 1e9


def bounded(timeout_s):
    """(effective timeout, whether it is the time box) for a run that may start now; raises TimeBox if none is left."""
    left = time_left_s()
    if left is not None and left <= 0:
        raise TimeBox("time box used up")
    if left is not None and (timeout_s is None or left < timeout_s):
        return left, True
    return timeout_s, False


def started(kind, condition, config):
    run_id = uuid.uuid4().hex
    if ATTEMPT_LOG:
        append_jsonl(ATTEMPT_LOG, {"event": "started", "run_id": run_id, "time": dt.datetime.now(dt.timezone.utc)
                                   .isoformat(), "kind": kind, "condition": condition, "config_key": config_key(config)})
    return run_id


# ---- back-end configurations ----

def cpu_config(threads, cpus=None, parity="none", warmup=0, name=None):
    return {"device": "cpu", "threads": int(threads), "cpus": cpus, "parity": parity or "none",
            "warmup_trials": int(warmup), "name": name}


def gpu_config(path, microbatch, host_threads, submission=1, warmup=0, device="cuda", cpus=None):
    if isinstance(host_threads, dict) or isinstance(host_threads, bool):
        raise TypeError(f"host_threads must be one thread count, not {host_threads!r}")
    return {"device": device, "path": path, "microbatch": microbatch, "submission": int(submission),
            "threads": int(host_threads), "warmup_trials": int(warmup), "cpus": cpus}


def config_key(c):
    """A stable short name of everything that can change a timing, e.g. cpu:t6@0-5:none or cuda:inc:b256:h8:s1:w0.
    Two runs with the same key ran the same configuration."""
    pin = f"@{c['cpus']}" if c.get("cpus") else ""
    if c["device"] == "cpu":
        w = f":w{c['warmup_trials']}" if c.get("warmup_trials") else ""
        return f"cpu:t{c['threads']}{pin}:{c.get('parity') or 'none'}{w}"
    return (f"{c['device']}:{c['path']}:b{c['microbatch']}:h{c['threads']}{pin}:s{c.get('submission', 1)}"
            f":w{c.get('warmup_trials', 0)}")


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
    """'cache-cold' only if the write to drop_caches actually succeeded (as root or through sudo -n)."""
    os.sync()
    try:
        Path("/proc/sys/vm/drop_caches").write_text("3\n")
        return "cache-cold"
    except OSError:
        try:
            r = subprocess.run(["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
                               capture_output=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            return "process-cold"
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
        self.mem, self.mhz, self.smi_lines, self.per_core = [], [], [], []
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.smi = None

    def _loop(self):
        while not self.stop_flag.is_set():
            self.mem.append(_mem_available_kib())
            mhz = _core_mhz()
            if mhz:
                self.mhz.append(statistics.median(mhz))
                self.per_core.append(mhz)
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
               "core_mhz_per_core": [_summary([s[i] for s in self.per_core if i < len(s)])
                                     for i in range(max((len(s) for s in self.per_core), default=0))],
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

def _kill_group(proc):
    """SIGKILL a runner started in its own session (setsid), then reap it."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        pass
    proc.wait()


def save_console(run_id, command, out, err):
    """Full console evidence, after process exit and outside measurement timing (also on malformed replies)."""
    if ATTEMPT_LOG is None:
        return None
    d = Path(ATTEMPT_LOG).parent / "logs" / str(run_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "command.json").write_text(json.dumps([str(x) for x in command]) + "\n")
    (d / "stdout.log").write_text(out)
    (d / "stderr.log").write_text(err)
    mirror_tree(d)
    return str(d)


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


def _record(mode, condition, config, cmd, t0, t1, status, out_dir, extra, sampler, cg_peak, expected_sha, run_id=None):
    st = _stages(out_dir)
    rec = {"mode": mode, **condition, "run_id": run_id, "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
           "config": config, "config_key": config_key(config), "cmd": cmd,
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
    """One cold run (deadline run with budget_s). Returns the run record; never raises for a failed run (TimeBox is
    raised before the start if the caller's time box is used up)."""
    condition = condition or {}
    kind = "deadline" if budget_s is not None else "cold"
    eff_timeout, boxed = bounded(timeout_s)
    run_id = started(kind, condition, config)
    try:
        out_dir, task = Path(out_dir).resolve(), Path(task).resolve()
        fresh_dir(out_dir)
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
                out, err = proc.communicate(timeout=eff_timeout)
                t1 = time.monotonic_ns()
                status = "ok" if proc.returncode == 0 else ("width_fail" if proc.returncode == 3 else "error")
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                out, err = proc.communicate()
                t1 = t0 + int(eff_timeout * 1e9)  # censored at exactly the timeout
                status = "time box" if boxed else "timeout"
            except BaseException:
                _kill_group(proc)  # an interrupt must not leave the runner (own session) running into later runs
                out, err = proc.communicate()
                (out_dir / "stdout.log").write_text(out)
                (out_dir / "stderr.log").write_text(err)
                raise
        (out_dir / "stdout.log").write_text(out)
        (out_dir / "stderr.log").write_text(err)
        if deadline is not None and status == "ok" and t1 > deadline:
            status = "late"
        rec = _record(kind, condition, config, cmd, t0, t1, status, out_dir,
                      {"cold_kind": cold, "returncode": proc.returncode, "stderr_tail": err[-2000:],
                       "deadline_ns": deadline, "budget_s": budget_s, "drain_ms": drain_ms if deadline else None,
                       "replicate": replicate, "timeout_s": eff_timeout}, smp, cg.peak_bytes(), expected_sha, run_id)
        cg.remove()
        return rec
    except Exception as e:  # noqa: BLE001  (a harness failure is a recorded result, not a lost run)
        return {"mode": kind, **condition, "run_id": run_id, "config": config, "config_key": config_key(config),
                "status": "harness_error", "error": repr(e), "out": str(out_dir), "budget_s": budget_s,
                "replicate": replicate, "finished_at": dt.datetime.now(dt.timezone.utc).isoformat()}


class WarmServer:
    """sf_run --serve in its own cgroup and process group; each timed request is one warm run. Every request has a
    timeout; a request that exceeds it kills the server (the run is recorded as timed out, the block ends)."""

    def __init__(self, config, *, build):
        self.config = config
        self.sock = str(Path(tempfile.gettempdir()) / f"sf_run_{uuid.uuid4().hex[:8]}.sock")
        self.cg = Cgroup()
        self.console_id = "warm-server-" + uuid.uuid4().hex
        prefix, env = launcher(config)
        self.proc = subprocess.Popen([*prefix, str(Path(build) / "sf_run"), "--serve", self.sock, *sf_args(config)],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=ROOT, env=env,
                                     preexec_fn=self.cg.preexec())
        line = self.proc.stdout.readline()
        self.greeting = line
        if "serving" not in line:
            self.kill()
            out, err = self.proc.stdout.read(), self.proc.stderr.read()
            save_console(self.console_id, self.proc.args, line + out, err)
            raise RuntimeError(f"warm server did not start: {line} {err[-2000:]}")

    def kill(self):
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except OSError:
            pass
        self.proc.wait()

    def request(self, payload, timeout_s=None):
        """The server's JSON reply; socket.timeout (an OSError) if none arrives within timeout_s."""
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(timeout_s)
        try:
            s.connect(self.sock)
            s.sendall((json.dumps(payload) + "\n").encode())
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf += chunk
        finally:
            s.close()
        return json.loads(buf.decode())

    def prime(self, task, out_dir, condition=None, timeout_s=None):
        """A first task outside the timing; its base-state key matches the timed task. Like every run it has a
        'started' record (kind 'priming') and returns its terminal record: status ok, 'timeout'/'time box' (the server
        is then killed) or 'error'."""
        condition = condition or {}
        eff, boxed = bounded(timeout_s)
        run_id = started("priming", condition, self.config)
        rec = {"mode": "warm-priming", **condition, "run_id": run_id, "config": self.config,
               "config_key": config_key(self.config), "timeout_s": eff, "out": str(Path(out_dir).resolve())}
        t0 = time.monotonic_ns()
        try:
            r = self.request({"task": str(Path(task).resolve()), "out": str(fresh_dir(Path(out_dir).resolve()))}, eff)
            rec.update({"status": "ok" if r.get("status") == "ok" else "error", "reply": r})
        except socket.timeout:
            self.kill()
            rec.update({"status": "time box" if boxed else "timeout", "error": f"priming exceeded {eff} s; server stopped"})
        except (OSError, ValueError) as e:
            rec.update({"status": "error", "error": repr(e)})
        rec.update({"e2e_s": (time.monotonic_ns() - t0) / 1e9,
                    "finished_at": dt.datetime.now(dt.timezone.utc).isoformat()})
        return rec

    def timed(self, task, out_dir, condition=None, expected_sha=None, timeout_s=None):
        condition = condition or {}
        eff, boxed = bounded(timeout_s)
        run_id = started("warm", condition, self.config)
        out_dir, task = Path(out_dir).resolve(), Path(task).resolve()
        fresh_dir(out_dir)
        killed = False
        with Sampler(gpu=self.config["device"] == "cuda") as smp:
            t0 = time.monotonic_ns()
            try:
                r = self.request({"task": str(task), "out": str(out_dir)}, eff)
                t1 = time.monotonic_ns()
            except socket.timeout:
                self.kill()
                killed, r, t1 = True, {"status": "timeout"}, t0 + int(eff * 1e9)
            except (OSError, ValueError) as e:
                r, t1 = {"status": "error", "error": repr(e)}, time.monotonic_ns()
        status = r.get("status", "error")
        if killed:
            status = "time box" if boxed else "timeout"
        elif status == "ok" and eff is not None and (t1 - t0) / 1e9 > eff:
            status, t1 = ("time box" if boxed else "timeout"), t0 + int(eff * 1e9)
        if status == "ok" and not r.get("base_cached"):
            status = "not_warm"  # the server rebuilt the base state: not a warm run
        return _record("warm", condition, self.config, ["warm", str(task)], t0, t1, status, out_dir,
                       {"base_cached": r.get("base_cached"), "reply": r, "timeout_s": eff,
                        "cgroup_scope": "server lifetime (priming and earlier requests included)"},
                       smp, self.cg.peak_bytes(), expected_sha, run_id)

    @property
    def alive(self):
        return self.proc.poll() is None

    def close(self):
        if self.alive:
            try:
                self.request({"cmd": "quit"}, WARM_CONTROL_TIMEOUT_S)
                self.proc.wait(timeout=WARM_CONTROL_TIMEOUT_S)
            except (OSError, subprocess.TimeoutExpired, ValueError):
                self.kill()
        save_console(self.console_id, self.proc.args, self.greeting + self.proc.stdout.read(), self.proc.stderr.read())
        self.proc.stdout.close()
        self.proc.stderr.close()
        self.cg.remove()


def warm_block(config, task, out, reps, condition, *, build, expected_sha=None, timeout_s=None, prime_timeout_s=None,
               on_record):
    """One warm block: priming outside the timing, then reps timed requests. on_record(rec) is called for each record
    as soon as it exists (so a block cut off later keeps its earlier runs). The priming has its own record (mode
    'warm-priming'). If the server does not start, the priming fails, or a request ends the server (timeout), every
    timed request not sent is recorded as 'not started (...)' with its rep, so the block's planned requests are all
    accounted for."""

    def not_started(first, why):
        for r in range(first, reps):
            on_record({"mode": "warm", **condition, "rep": r, "config": config, "config_key": config_key(config),
                       "status": f"not started ({why})", "runs_not_started": 1,
                       "finished_at": dt.datetime.now(dt.timezone.utc).isoformat()})

    try:
        srv = WarmServer(config, build=build)
    except (RuntimeError, OSError) as e:
        on_record({"mode": "warm-priming", **condition, "config": config, "config_key": config_key(config),
                   "status": "error", "error": f"server start: {e!r}"})
        not_started(0, "warm server did not start")
        return
    try:
        p = srv.prime(task, Path(f"{out}_prime"), condition, prime_timeout_s)
        p["priming_timed_out"] = p["status"] in ("timeout", "time box")
        on_record(p)
        if p["status"] != "ok":
            not_started(0, f"priming {p['status']}")
            return
        for r in range(reps):
            rec = srv.timed(task, Path(f"{out}_w{r}"), {**condition, "rep": r}, expected_sha=expected_sha,
                            timeout_s=timeout_s)
            on_record(rec)
            if not srv.alive:
                not_started(r + 1, f"warm server ended after {rec.get('status')}")
                return
    except BaseException:
        srv.kill()  # an interrupt or a time box: do not wait for the server to finish its request
        raise
    finally:
        srv.close()


def bench(task, config, seconds, *, build, sample=False, diag=False):
    """Steady-state throughput of one task point (sf_run --bench). Returns the bench record or, on failure, a record
    with status 'error' and the tail of stderr (an allocation failure is a result, not an exception). A bench is killed
    after seconds + BENCH_SETUP_ALLOWANCE_S, or at the time box."""
    eff, boxed = bounded(seconds + BENCH_SETUP_ALLOWANCE_S)
    run_id = started("bench", {"task": str(task)}, config)
    prefix, env = launcher(config)
    cmd = [*prefix, str(Path(build) / "sf_run"), str(task), "--bench", str(seconds), *sf_args(config)]
    if diag:
        cmd.append("--diag")
    timed_out = False
    with Sampler(gpu=sample and config["device"] == "cuda") as smp:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=ROOT, env=env,
                                start_new_session=True)
        try:
            out, err = proc.communicate(timeout=eff)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            out, err = proc.communicate()
            timed_out = True
        except BaseException:
            _kill_group(proc)
            out, err = proc.communicate()
            save_console(run_id, cmd, out, err)
            raise
    console_dir = save_console(run_id, cmd, out, err)
    rec = {"task": str(Path(task).relative_to(ROOT)) if Path(task).is_absolute() else str(task), "config": config,
           "config_key": config_key(config), "seconds": seconds, "returncode": proc.returncode, "sampler": smp.record(),
           "run_id": run_id, "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
           "stdout": out, "stderr": err, "console_dir": console_dir}
    if timed_out:
        rec.update({"status": "time box" if boxed else "timeout", "timeout_s": eff, "trials_per_s": None,
                    "stderr_tail": err[-2000:]})
    elif proc.returncode == 0:
        b = json.loads(out.strip().splitlines()[-1])
        rec.update({"status": "ok", "trials_per_s": b["trials_per_s"], "setup_s": b["setup_s"], "bench": b,
                    "device": parse_device_info(b.get("device_info"))})
    else:
        oom = any(x in err for x in ("out of memory", "cudaMalloc", "no microbatch fits", "allocation failed"))
        rec.update({"status": "allocation_failure" if oom else "error", "stderr_tail": err[-2000:], "trials_per_s": None})
    return rec


def append(path, rec):
    append_jsonl(path, rec)

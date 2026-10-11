"""What machine a run is on: environment manifest, CPU topology, GPU state, the identity check against a machine
profile of protocol.json, and (on AWS) instance metadata and official prices.

Everything is read from the machine itself; a value that cannot be read is recorded as null with the reason.
"""

import datetime as dt
import hashlib
import json
import os
import platform
import re
import sys
import urllib.request
from pathlib import Path

from .common import MACHINES, PROTOCOL, ROOT, binary_hashes, code_fingerprint, git_state, input_hashes, sh, sha256_file

LOCATION = "US East (N. Virginia)"


# ---- CPU topology ----

def _cpulist(text):
    out = []
    for part in (text or "").strip().split(","):
        if "-" in part:
            a, b = part.split("-")
            out += range(int(a), int(b) + 1)
        elif part:
            out.append(int(part))
    return out


def cpulist_text(cpus):
    cpus = sorted(cpus)
    runs, start = [], None
    for i, c in enumerate(cpus):
        if start is None:
            start = c
        if i + 1 == len(cpus) or cpus[i + 1] != c + 1:
            runs.append(f"{start}-{c}" if c != start else f"{c}")
            start = None
    return ",".join(runs)


def topology():
    """Logical CPUs, physical cores, SMT, NUMA, and the P/E core mapping if the kernel exposes it.

    Hybrid Intel parts expose /sys/devices/cpu_core/cpus (P) and /sys/devices/cpu_atom/cpus (E) on bare metal; a
    WSL2 or VM kernel usually does not. Then the per-CPU maximum frequency (lscpu MAXMHZ) is tried; if it is flat
    the mapping is 'unknown' and a pilot plan must state it explicitly (--p-cores).
    """
    allowed = sorted(os.sched_getaffinity(0))
    allowed_set = set(allowed)
    rows = sh(["lscpu", "-p=CPU,CORE,SOCKET,NODE"]) or ""
    cores, nodes = {}, set()
    for line in rows.splitlines():
        if line.startswith("#"):
            continue
        cpu, core, sock, node = (int(x) if x else 0 for x in line.split(","))
        if cpu not in allowed_set:  # only CPUs this process may run on (cpuset / affinity)
            continue
        cores.setdefault((sock, core), []).append(cpu)
        nodes.add(node)
    p = Path("/sys/devices/cpu_core/cpus")
    e = Path("/sys/devices/cpu_atom/cpus")
    pe = {"source": None, "p_cores": None, "e_cores": None}
    if p.exists() and e.exists():
        pe = {"source": "sysfs cpu_core/cpu_atom", "p_cores": [c for c in _cpulist(p.read_text()) if c in allowed_set],
              "e_cores": [c for c in _cpulist(e.read_text()) if c in allowed_set]}
    else:
        mhz = sh(["lscpu", "-e=CPU,MAXMHZ"]) or ""
        vals = {}
        for line in mhz.splitlines()[1:]:
            f = line.split()
            if len(f) == 2 and f[1] not in ("-", "") and int(f[0]) in allowed_set:
                vals[int(f[0])] = float(f[1])
        if vals and len(set(vals.values())) == 2:
            hi = max(vals.values())
            pe = {"source": "lscpu MAXMHZ (two levels)", "p_cores": [c for c, v in vals.items() if v == hi],
                  "e_cores": [c for c, v in vals.items() if v != hi]}
    physical = [min(v) for v in cores.values()] if cores else allowed
    return {"logical": len(allowed), "allowed_cpus": cpulist_text(allowed), "physical_cores": len(physical),
            "cgroup_cpu_max": cgroup_cpu_max(), "online_cpus": os.cpu_count(),
            "smt": bool(cores) and any(len(v) > 1 for v in cores.values()),
            "first_thread_per_core": cpulist_text(physical), "numa_nodes": len(nodes) or 1,
            "p_cores": cpulist_text(pe["p_cores"]) if pe["p_cores"] else None,
            "e_cores": cpulist_text(pe["e_cores"]) if pe["e_cores"] else None, "pe_source": pe["source"]}


def cgroup_cpu_max():
    """The cgroup v2 CPU quota of this process ('max 100000' = none), or None if not readable."""
    try:
        rel = next(line.split("::", 1)[1] for line in Path("/proc/self/cgroup").read_text().splitlines()
                   if line.startswith("0::"))
        return (Path("/sys/fs/cgroup") / rel.lstrip("/") / "cpu.max").read_text().strip()
    except (OSError, StopIteration):
        return None


def thread_candidates(topo, p_cores=None):
    """CPU configurations to compare on a pilot machine: P-core set, physical cores, all logical CPUs.

    Returns [{name, threads, cpus}] with cpus a CPU list for taskset (None = no pinning). Duplicates are dropped.
    """
    out = []
    pc = p_cores or topo.get("p_cores")
    if pc:
        out.append({"name": "p-cores", "threads": len(_cpulist(pc)), "cpus": pc})
    if topo["smt"]:
        out.append({"name": "physical", "threads": topo["physical_cores"], "cpus": topo["first_thread_per_core"]})
    out.append({"name": "all", "threads": topo["logical"], "cpus": None})
    seen, uniq = set(), []
    for c in out:
        key = (c["threads"], c["cpus"])
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


# ---- GPU and runtime environment ----

def gpu_state():
    q = sh(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,persistence_mode,"
            "compute_cap,clocks.max.sm,power.limit", "--format=csv,noheader,nounits"])
    if not q:
        return {"present": False}
    f = [x.strip() for x in q.splitlines()[0].split(",")]
    keys = ["name", "driver", "memory_total_mib", "memory_used_mib", "memory_free_mib", "persistence_mode",
            "compute_cap", "max_sm_mhz", "power_limit_w"]
    rec = dict(zip(keys, f))
    rec["present"] = True
    rec["processes"] = sh(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])
    return rec


def runtime_kind():
    rel = platform.release().lower()
    if "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ or Path("/content").is_dir() and Path("/opt/google").exists():
        return "colab"
    if "microsoft" in rel or "wsl" in rel:
        return "wsl2"
    if ec2_metadata().get("on_ec2"):
        return "ec2"
    return "linux"


def capabilities():
    """What the harness can control here; missing ones are recorded with the substitute measure (SPEC §8)."""
    from .harness import drop_caches  # an actual drop, not a guess from root/sudo
    cg = Path("/sys/fs/cgroup/cgroup.controllers").exists() and os.access("/sys/fs/cgroup", os.W_OK)
    g = gpu_state()
    return {"drop_page_cache": drop_caches() == "cache-cold", "cgroup_memory_peak": cg,
            "gpu_persistence_mode": (g.get("persistence_mode") == "Enabled") if g.get("present") else None,
            "perf": sh(["perf", "--version"]) is not None, "taskset": sh(["taskset", "-V"]) is not None}


# ---- AWS (IMDSv2, EC2 API, Price List API) ----

def _imds(path, token=None, method="GET", headers=None):
    try:
        h = dict(headers or {})
        if token:
            h["X-aws-ec2-metadata-token"] = token
        req = urllib.request.Request(f"http://169.254.169.254/latest/{path}", method=method, headers=h)
        with urllib.request.urlopen(req, timeout=1) as r:
            return r.read().decode()
    except OSError:
        return None


def _imds_token():
    return _imds("api/token", method="PUT", headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"})


def ec2_metadata():
    tok = _imds_token()
    if tok is None:
        return {"on_ec2": False}
    get = lambda p: _imds(f"meta-data/{p}", tok)  # noqa: E731
    md = {"on_ec2": True, "instance_type": get("instance-type"), "instance_id": get("instance-id"),
          "availability_zone": get("placement/availability-zone"), "region": get("placement/region"),
          "purchase_option": get("instance-life-cycle"), "ami_id": get("ami-id")}
    desc = sh(["aws", "ec2", "describe-instances", "--region", md["region"] or "us-east-1", "--instance-ids",
               md["instance_id"] or "", "--output", "json"])
    if desc:
        inst = json.loads(desc)["Reservations"][0]["Instances"][0]
        md.update({"launch_time": inst.get("LaunchTime"), "tenancy": inst.get("Placement", {}).get("Tenancy"),
                   "spot_request_id": inst.get("SpotInstanceRequestId")})
    else:
        md["ec2_api"] = "unavailable (no credentials or aws CLI); run collect_env.py --ec2-times from the operator machine"
    return md


def spot_interruption_notice():
    tok = _imds_token()
    return _imds("meta-data/spot/instance-action", tok) if tok else None


def on_demand_price(instance_type, region="us-east-1"):
    """Official on-demand Linux price with the SKU and the hash of the Price List API response."""
    filters = [("instanceType", instance_type), ("location", LOCATION), ("operatingSystem", "Linux"),
               ("tenancy", "Shared"), ("preInstalledSw", "NA"), ("capacitystatus", "Used"),
               ("licenseModel", "No License required")]
    raw = sh(["aws", "pricing", "get-products", "--region", region, "--service-code", "AmazonEC2", "--output", "json",
              "--filters", *[f"Type=TERM_MATCH,Field={k},Value={v}" for k, v in filters]])
    if raw is None:
        return {"instance_type": instance_type, "price_usd_per_hour": None, "reason": "Price List API unavailable"}
    items = [json.loads(p) for p in json.loads(raw)["PriceList"]]
    digest = hashlib.sha256(raw.encode()).hexdigest()
    if len(items) != 1:
        return {"instance_type": instance_type, "price_usd_per_hour": None, "response_sha256": digest,
                "response": raw, "reason": f"{len(items)} price list entries matched"}
    term = next(iter(items[0]["terms"]["OnDemand"].values()))
    dim = next(iter(term["priceDimensions"].values()))
    return {"instance_type": instance_type, "region": region, "sku": items[0]["product"]["sku"],
            "rate_code": dim["rateCode"], "price_usd_per_hour": float(dim["pricePerUnit"]["USD"]),
            "publication_date": items[0].get("publicationDate"), "effective_date": term.get("effectiveDate"),
            "retrieved": dt.datetime.now(dt.timezone.utc).isoformat(), "response_sha256": digest, "response": raw}


def spot_price(instance_type, zone, when_iso, region="us-east-1"):
    raw = sh(["aws", "ec2", "describe-spot-price-history", "--region", region, "--instance-types", instance_type,
              "--availability-zone", zone, "--product-descriptions", "Linux/UNIX", "--start-time", when_iso,
              "--end-time", when_iso, "--output", "json"])
    if raw is None:
        return None
    h = json.loads(raw)["SpotPriceHistory"]
    return {"zone": zone, "at": when_iso, "price_usd_per_hour": float(h[0]["SpotPrice"]) if h else None}


def ec2_times(instance_id, region="us-east-1"):
    """Launch and termination times from the EC2 API (after termination, from the operator machine)."""
    raw = sh(["aws", "ec2", "describe-instances", "--region", region, "--instance-ids", instance_id, "--output", "json"])
    if raw is None:
        return {"instance_id": instance_id, "reason": "EC2 API unavailable"}
    inst = json.loads(raw)["Reservations"][0]["Instances"][0]
    reason = inst.get("StateTransitionReason", "")
    m = re.search(r"\((\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) GMT\)", reason)
    terminated = inst["State"]["Name"] == "terminated"
    rec = {"instance_id": instance_id, "instance_type": inst["InstanceType"], "launch_time": inst["LaunchTime"],
           "state": inst["State"]["Name"], "state_transition_reason": reason,
           "termination_time": f"{m.group(1).replace(' ', 'T')}+00:00" if m and terminated else None,
           "purchase_option": inst.get("InstanceLifecycle", "on-demand"),
           "availability_zone": inst["Placement"]["AvailabilityZone"]}
    if rec["termination_time"]:
        t0 = dt.datetime.fromisoformat(rec["launch_time"].replace("Z", "+00:00"))
        rec["instance_hours"] = (dt.datetime.fromisoformat(rec["termination_time"]) - t0).total_seconds() / 3600
    if rec["purchase_option"] == "spot":
        rec["spot_price_at_launch"] = spot_price(rec["instance_type"], rec["availability_zone"], rec["launch_time"])
    rec["on_demand_price"] = on_demand_price(rec["instance_type"], region)  # cost basis of p2/cost.py
    return rec


# ---- identity against a machine profile ----

def identity(profile):
    """Check this machine against machines[profile].identity. match is None for profiles without an identity
    (Colab: every session is its own hardware and is only recorded)."""
    lscpu = {k.strip(): v.strip() for k, v in
             (line.split(":", 1) for line in (sh(["lscpu"]) or "").splitlines() if ":" in line)}
    topo = topology()
    g = gpu_state()
    rec = {"profile": profile, "cpu_model": lscpu.get("Model name", ""), "cores": topo["physical_cores"],
           "threads": topo["logical"], "numa_nodes": topo["numa_nodes"], "gpu_name": g.get("name")}
    want = (MACHINES.get(profile) or {}).get("identity")
    if not want:
        rec.update({"expected": None, "match": None})
        return rec
    ok = True
    if "cpu_model_contains" in want:
        ok &= want["cpu_model_contains"] in rec["cpu_model"]
    for k in ("cores", "threads", "numa_nodes"):
        if k in want:
            ok &= rec[k] == want[k]
    if "gpu_name_contains" in want:
        ok &= want["gpu_name_contains"] in (rec["gpu_name"] or "")
    prof = MACHINES.get(profile) or {}
    if prof.get("kind") == "cloud":  # the instance type and region themselves, not only the CPU/GPU model
        md = ec2_metadata()
        rec.update({"instance_type": md.get("instance_type"), "region": md.get("region")})
        ok &= md.get("instance_type") == profile and md.get("region") == prof.get("region")
    rec.update({"expected": want, "match": bool(ok)})
    return rec


def build_info(build):
    cache = Path(build) / "CMakeCache.txt"
    flags = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            for key in ("CMAKE_CXX_COMPILER:", "CMAKE_CXX_FLAGS_RELEASE:", "CMAKE_BUILD_TYPE:", "SF_CUDA:",
                        "CMAKE_CUDA_COMPILER:", "CMAKE_CUDA_ARCHITECTURES:", "CMAKE_CUDA_FLAGS:"):
                if line.startswith(key):
                    flags[key.split(":")[0]] = line.split("=", 1)[1]
    return {"cmake_cache": flags, "binary_sha256": binary_hashes(build)}


def hardware_fingerprint(env):
    """Short hash of what makes a session's hardware: runtime, CPU model, allowed CPUs, quota, RAM, GPU and driver.
    Pilot sessions with different fingerprints are different machines and are never pooled."""
    g = env.get("gpu") or {}
    topo = env.get("topology") or {}
    key = {"runtime": env.get("runtime"), "cpu": (env.get("identity") or {}).get("cpu_model"),
           "cpus": topo.get("allowed_cpus"), "quota": topo.get("cgroup_cpu_max"),
           "mem_total": (env.get("meminfo") or {}).get("MemTotal"), "gpu": g.get("name"),
           "gpu_mem": g.get("memory_total_mib"), "driver": g.get("driver"),
           "instance_type": (env.get("ec2") or {}).get("instance_type")}
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]


def environment(profile, build=ROOT / "build"):
    """The manifest of one session or launch (SPEC §9)."""
    read = lambda p: Path(p).read_text().strip() if Path(p).exists() else None  # noqa: E731
    pkgs = {}
    for name in ("numpy", "pandas", "scipy", "matplotlib"):
        try:
            pkgs[name] = __import__(name).__version__
        except ImportError:
            pkgs[name] = None
    meminfo = {k: v.strip() for k, v in (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
               if k in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")}
    env = {
        "collected": dt.datetime.now(dt.timezone.utc).isoformat(),
        "profile": profile,
        "runtime": runtime_kind(),
        "git": git_state(),
        "build": build_info(build),
        "compiler": sh(["c++", "--version"]),
        "nvcc": sh(["nvcc", "--version"]),
        "gpu": gpu_state(),
        "nvidia_smi_q": sh(["nvidia-smi", "-q"]),
        "lscpu": sh(["lscpu"]),
        "topology": topology(),
        "numactl": sh(["numactl", "--hardware"]),
        "meminfo": meminfo,
        "uptime_s": float(read("/proc/uptime").split()[0]) if read("/proc/uptime") else None,
        "governor": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "thp": read("/sys/kernel/mm/transparent_hugepage/enabled"),
        "kernel": platform.release(),
        "os": platform.platform(),
        "omp": {k: v for k, v in os.environ.items() if k.startswith("OMP_")},
        "capabilities": capabilities(),
        "identity": identity(profile),
        "ec2": ec2_metadata(),
        "inputs_sha256": input_hashes(),
        "protocol_sha256": sha256_file(Path(__file__).resolve().parents[1] / "protocol.json"),
        "stream": PROTOCOL["stream"]["name"],
        "python": sys.version,
        "python_packages": pkgs,
        "pip_freeze": sh([sys.executable, "-m", "pip", "freeze"]),
        "code_sha256": code_fingerprint(),
    }
    env["hardware_fingerprint"] = hardware_fingerprint(env)
    if env["ec2"].get("on_ec2") and env["ec2"].get("instance_type"):
        env["on_demand_price"] = on_demand_price(env["ec2"]["instance_type"])
    return env

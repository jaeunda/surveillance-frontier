"""Colab storage and resume checks. Standard library only; also usable on a downloaded result folder.

    python colab_ops.py audit /path/to/LABEL
    python colab_ops.py verify /path/to/LABEL-....tar.gz

No measurement, repair, deletion or rewriting of raw records occurs here. Drive receipts mean successful readback
through the mounted filesystem, not a guarantee against provider loss. Keep a second copy of the sealed archive.
"""

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import time
import uuid
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rows(path):
    p = Path(path)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def atomic_copy(src, dst):
    """Publish only a fully copied, readback-verified file; preserve the old destination on failure."""
    src, dst = Path(src), Path(dst)
    expected = digest(src)
    if dst.is_file() and digest(dst) == expected:
        return expected
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial-" + uuid.uuid4().hex)
    try:
        with open(src, "rb") as inp, open(tmp, "xb") as out:
            shutil.copyfileobj(inp, out)
            out.flush()
            os.fsync(out.fileno())
        if digest(tmp) != expected:
            raise OSError(f"Backup checksum mismatch: {tmp}")
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)
    return expected


def backup_tree(source, destination):
    """Run between commands only. A receipt is published after every source file has passed readback."""
    source, destination = Path(source), Path(destination)
    files = {}
    for p in sorted(source.rglob("*")):
        rel = p.relative_to(source)
        if p.is_file() and "checkpoints" not in rel.parts:
            old = destination / rel
            if old.is_file() and p.suffix == ".jsonl" and not p.read_bytes().startswith(old.read_bytes()):
                raise ValueError(f"Backup record is not a prefix of the local record; preserve both: {rel}")
            if old.is_file() and "outputs" in rel.parts and digest(old) != digest(p):
                raise ValueError(f"Existing raw output differs; preserve both copies: {rel}")
            files[str(rel)] = atomic_copy(p, destination / rel)
    receipt = {"time_unix": time.time(), "files_sha256": files,
               "meaning": "all listed files copied and read back; live mirrors may append after this receipt"}
    p = source / "_colab/checkpoints" / f"{time.time_ns()}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(receipt, indent=1) + "\n")
    atomic_copy(p, destination / p.relative_to(source))
    return p


def record_output(root, value):
    """Resolve original Colab output paths under an offline copy, never outside the result root."""
    root = Path(root).resolve()
    p = Path(value)
    try:
        rel = p.relative_to(root)
    except ValueError:
        marker = f"/results/pilot/{root.name}/"
        if marker not in str(p):
            raise ValueError(f"output path outside label: {p}")
        rel = Path(str(p).split(marker, 1)[1])
    out = (root / rel).resolve()
    out.relative_to(root)
    return out


def restore_inputs(repo, result, plan, protocol):
    """Reuse archived input bytes on a new VM; refuse any change to the plan's input identities."""
    repo, result = Path(repo), Path(result)
    for name, spec in protocol["series"].items():
        target = repo / spec["file"]
        expected = plan["inputs_sha256"][name]
        if target.exists():
            if digest(target) != expected:
                raise ValueError(f"Input differs from plan: {target}")
            continue
        copies = sorted((result / "_colab/reproduction").glob(f"*/{target.name}"))
        saved = next((p for p in copies if digest(p) == expected), None)
        if saved is None:
            raise ValueError(f"Original input unavailable: {name}; restore the matching archived bytes")
        atomic_copy(saved, target)


def manifest_errors(base, manifest):
    errors = []
    base = Path(base).resolve()
    for rel, expected in manifest.items():
        p = (base / rel).resolve()
        try:
            p.relative_to(base)
            valid = p.is_file() and digest(p) == expected
        except (ValueError, OSError):
            valid = False
        if not valid:
            errors.append(f"artifact_missing: {p}")
    return errors


def step_state(root, session, step):
    found = [r for r in rows(Path(root) / f"session-{session}/steps.jsonl") if r.get("step") == step]
    if not found:
        return "new"
    return "unfinished" if found[-1].get("event") == "started" else found[-1].get("status", "unknown")


def audit_results(root):
    """Strictly read records, verify listed outputs and P0 evidence, and check fixed-task completed attempts.
    Missing manifests on historical runs are not invented: an ok record must carry its evidence to be reusable.
    """
    root, errors = Path(root), []
    for p in root.rglob("*.jsonl"):
        try:
            rows(p)
        except (ValueError, OSError) as exc:
            errors.append(f"damaged record {p}: {exc}")
    if errors:
        return errors
    for p in root.glob("reference/attempt-*/reference.json"):
        ref = json.loads(p.read_text())
        if ref.get("artifacts"):
            errors += manifest_errors(p.parent, ref["artifacts"])
    for sd in sorted(root.glob("session-*")):
        runs = rows(sd / "runs.jsonl")
        for r in runs:
            man = r.get("out_manifest")
            if not man:
                if r.get("status") == "ok" and r.get("out"):
                    errors.append(f"no output manifest: {r.get('run_id')} ({r['out']})")
                continue
            try:
                errors += manifest_errors(record_output(root, r["out"]), man)
            except (KeyError, ValueError) as exc:
                errors.append(str(exc))
        checks = sorted(sd.glob("checks/attempt-*/checks.json"), key=lambda p: int(p.parent.name.split("-")[-1]))
        if checks:
            p = checks[-1]
            c = json.loads(p.read_text())
            if not c.get("artifacts"):
                errors.append(f"no P0 artifact manifest: {p}; retain it and run a new check attempt")
            else:
                errors += manifest_errors(p.parent, c["artifacts"])
        for step in ("cpu-baseline", "gpu-tasks"):
            if step_state(root, int(sd.name.split("-")[-1]), step) != "complete":
                continue
            orders = sorted((sd / "outputs" / step).glob("attempt-*/order.json"),
                            key=lambda p: int(p.parent.name.split("-")[-1]))
            if not orders:
                errors.append(f"complete {sd.name}/{step} has no order.json")
                continue
            order = orders[-1]
            recs = [r for r in runs if r.get("step") == step and r.get("outputs") == order.parent.name]
            for pos, it in enumerate(json.loads(order.read_text())["order"]):
                want = [("cold", it["rep"])] if it["kind"] == "cold" else [
                    ("warm-priming", None), *[("warm", n) for n in range(it["reps"])]]
                got = [(r.get("mode"), r.get("rep")) for r in recs if r.get("position") == pos]
                if sorted(got, key=str) != sorted(want, key=str):
                    errors.append(f"complete {sd.name}/{step}/{order.parent.name} position {pos} lacks exact records")
    return errors


def time_action(plan, root, session, step):
    """Distinguish a step share from the whole-session/label budget using the harness's accounting."""
    import pilot
    sd = Path(root) / f"session-{session}"
    if pilot.hours_left(plan, root, sd) <= 0:
        return "stop session"
    if pilot.hours_left(plan, root, sd, step) <= 0:
        return "skip step"
    return "run"


def seal_archive(root, destination):
    """Create an immutable archive outside root; copy verified bytes and checksum, then verify them again."""
    root, destination = Path(root), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    p = destination / f"{root.name}-{time.time_ns()}.tar.gz"
    tmp = p.with_suffix(p.suffix + ".partial")
    with tarfile.open(tmp, "w:gz") as tar:
        tar.add(root, arcname=root.name)
    os.replace(tmp, p)
    p.with_name(p.name + ".sha256").write_text(f"{digest(p)}  {p.name}\n")
    verify_archive(p)
    return p


def verify_archive(path):
    path = Path(path)
    want = path.with_name(path.name + ".sha256").read_text().split()[0]
    if digest(path) != want:
        raise ValueError(f"Archive checksum mismatch: {path}")
    with tarfile.open(path, "r:gz") as tar:
        for member in tar:
            p = Path(member.name)
            if p.is_absolute() or ".." in p.parts or not (member.isfile() or member.isdir()):
                raise ValueError(f"Unexpected archive entry: {member.name}")
    return want


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("audit", "verify", "time-action"))
    ap.add_argument("path", type=Path)
    ap.add_argument("--session", type=int)
    ap.add_argument("--step")
    a = ap.parse_args()
    if a.action == "verify":
        print(verify_archive(a.path))
    elif a.action == "time-action":
        import pilot
        print(time_action(pilot.effective_plan(a.path), a.path, a.session, a.step))
    else:
        errors = audit_results(a.path)
        print(json.dumps({"ok": not errors, "problems": errors}, indent=1))
        raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()

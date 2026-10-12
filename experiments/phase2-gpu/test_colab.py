"""Storage, interruption and notebook orchestration regression tests; no CUDA or measurements required."""

import ast
import contextlib
import importlib.util
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import colab_ops as ops
import pilot
from p2 import common, harness, records

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "pilot_colab.py"


def dump(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


def lines(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(x) + "\n" for x in data))


class ColabTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_backup_and_seal_readback(self):
        src, dst = self.root / "label", self.root / "drive/label"
        dump(src / "pilot-plan.json", {"label": "label"})
        (src / "raw.bin").write_bytes(bytes(range(256)))
        receipt = ops.backup_tree(src, dst)
        man = json.loads(receipt.read_text())["files_sha256"]
        self.assertEqual(ops.manifest_errors(dst, man), [])
        archive = ops.seal_archive(src, self.root / "archives")
        self.assertEqual(ops.verify_archive(archive), ops.digest(archive))
        with open(archive, "ab") as f:
            f.write(b"broken")
        with self.assertRaisesRegex(ValueError, "checksum"):
            ops.verify_archive(archive)

    def test_failed_copy_preserves_previous_backup(self):
        src, dst = self.root / "new", self.root / "backup"
        src.write_text("new data")
        dst.write_text("old data")
        with patch.object(shutil, "copyfileobj", side_effect=OSError("quota")):
            with self.assertRaises(OSError):
                ops.atomic_copy(src, dst)
        self.assertEqual(dst.read_text(), "old data")
        self.assertFalse(list(self.root.glob("*.partial-*")))

    def test_backup_refuses_to_replace_history(self):
        src, dst = self.root / "local", self.root / "drive"
        lines(src / "runs.jsonl", [{"status": "error"}])
        ops.backup_tree(src, dst)
        lines(src / "runs.jsonl", [{"status": "ok"}])
        with self.assertRaisesRegex(ValueError, "preserve both"):
            ops.backup_tree(src, dst)
        self.assertEqual(ops.rows(dst / "runs.jsonl"), [{"status": "error"}])

    def test_copy_detects_corruption(self):
        src, dst = self.root / "new", self.root / "backup"
        src.write_text("original")
        with patch.object(shutil, "copyfileobj", side_effect=lambda i, o: o.write(b"bad")):
            with self.assertRaisesRegex(OSError, "checksum"):
                ops.atomic_copy(src, dst)
        self.assertFalse(dst.exists())

    def test_required_mirror_failure_stops_after_preserving_local_record(self):
        src, dst = self.root / "source", self.root / "drive"
        src.mkdir()
        dst.write_text("unavailable directory")
        with patch.dict(os.environ, SF_MIRROR_FROM=str(src), SF_MIRROR_TO=str(dst), SF_REQUIRE_MIRROR="1"):
            with self.assertRaisesRegex(OSError, "Required record mirror"):
                common.append_jsonl(src / "runs.jsonl", {"status": "error", "evidence": "kept"})
        self.assertEqual(ops.rows(src / "runs.jsonl"), [{"status": "error", "evidence": "kept"}])

    def test_audit_preserves_damaged_records(self):
        p = self.root / "session-1/runs.jsonl"
        p.parent.mkdir()
        p.write_text('{"unfinished":')
        before = p.read_bytes()
        self.assertIn("damaged record", ops.audit_results(self.root)[0])
        self.assertEqual(p.read_bytes(), before)

    def test_restored_gate_and_output_are_checked(self):
        out = self.root / "session-1/outputs/r1"
        dump(out / "cells.json", {"value": 1})
        lines(self.root / "session-1/runs.jsonl", [{"status": "ok", "out": str(out),
                                                   "out_manifest": common.manifest(out)}])
        c = self.root / "session-1/checks/attempt-1"
        dump(c / "gate_cuda_inc.json", {"pass": True, "device": "cuda"})
        dump(c / "checks.json", {"artifacts": common.manifest(c)})
        self.assertEqual(ops.audit_results(self.root), [])
        (c / "gate_cuda_inc.json").unlink()
        (out / "cells.json").write_text("changed")
        self.assertEqual(len(ops.audit_results(self.root)), 2)

    def test_complete_step_needs_all_planned_records(self):
        sd = self.root / "session-1"
        lines(sd / "steps.jsonl", [{"step": "cpu-baseline", "status": "complete"}])
        dump(sd / "outputs/cpu-baseline/attempt-1/order.json", {"order": [{"kind": "cold", "rep": 0}]})
        self.assertTrue(ops.audit_results(self.root))
        lines(sd / "runs.jsonl", [{"step": "cpu-baseline", "outputs": "attempt-1", "position": 0,
                                   "mode": "cold", "rep": 0, "status": "timeout"}])
        self.assertEqual(ops.audit_results(self.root), [])  # accounted for, not an E2E success

    def test_offline_paths_cannot_escape_label(self):
        root = self.root / "label"
        value = "/content/surveillance-frontier/experiments/phase2-gpu/results/pilot/label/session-1/outputs/x"
        self.assertEqual(ops.record_output(root, value), root / "session-1/outputs/x")
        with self.assertRaises(ValueError):
            ops.record_output(root, value + "/../../../../../outside")

    def test_restored_inputs_use_archived_bytes(self):
        label = self.root / "label"
        saved = label / "_colab/reproduction/session-1/input.bin"
        saved.parent.mkdir(parents=True)
        saved.write_bytes(b"original series")
        plan = {"inputs_sha256": {"series": ops.digest(saved)}}
        protocol = {"series": {"series": {"file": "data/input.bin"}}}
        repo = self.root / "repo"
        ops.restore_inputs(repo, label, plan, protocol)
        self.assertEqual((repo / "data/input.bin").read_bytes(), saved.read_bytes())
        (repo / "data/input.bin").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "differs"):
            ops.restore_inputs(repo, label, plan, protocol)

    def test_missing_artifact_cannot_supply_metric(self):
        d = self.root / "pilot/label"
        out = d / "session-1/outputs/one"
        dump(out / "cells.json", {"x": 1})
        rec = {"status": "ok", "out": str(out), "out_manifest": common.manifest(out), "e2e_s": 1.0,
               "task": "T1-curve", "series": "1m-quarter", "mode": "warm", "config_key": "cpu:t2:none"}
        lines(d / "session-1/runs.jsonl", [rec])
        (out / "cells.json").unlink()
        with patch.object(records, "RESULTS", self.root):
            got, _ = records.label_records("label")
            self.assertEqual(got[0]["status"], "artifact_missing")
            result = records.observed("label", {"kind": "e2e", "task": "T1-curve", "series": "1m-quarter",
                                                 "mode": "warm", "config": "cpu:t2:none"})
            self.assertIsNone(result[0])
        self.assertEqual(ops.rows(d / "session-1/runs.jsonl")[0]["status"], "ok")

    def test_step_share_is_not_session_exhaustion(self):
        plan = {"hours_available": 2, "hours_per_session": 1, "step_hours": {"cpu-baseline": 0.1}}
        lines(self.root / "session-1/steps.jsonl", [{"step": "cpu-baseline", "elapsed_s": 360}])
        self.assertEqual(ops.time_action(plan, self.root, 1, "cpu-baseline"), "skip step")
        self.assertEqual(ops.time_action(plan, self.root, 1, "gpu-rates"), "run")
        lines(self.root / "session-1/steps.jsonl", [{"step": "cpu-baseline", "elapsed_s": 3600}])
        self.assertEqual(ops.time_action(plan, self.root, 1, "gpu-rates"), "stop session")

    def test_third_session_amendment_does_not_increase_hours(self):
        dump(self.root / "pilot-plan.json", {"sessions": 2, "hours_available": 16})
        lines(self.root / "amendments.jsonl", [{"set": {"sessions": 3}}])
        self.assertEqual(pilot.effective_plan(self.root), {"sessions": 3, "hours_available": 16, "amendments": 1})

    def test_notebook_continues_after_step_time_box(self):
        cell = next(x for x in SOURCE.read_text().split("# %%\n") if x.startswith("if RUN_MEASUREMENTS"))
        cell = cell.split("# %% [markdown]")[0]
        sd = self.root / "session-1"
        executed = []
        def command(*args, **kwargs):
            if "time-action" in args:
                step = args[-1]
                return "skip step" if step == "cpu-baseline" and step in executed else "run"
            if "run" in args:
                step = args[args.index("--step") + 1]
                executed.append(step)
                state = "stopped (time): time box" if step == "cpu-baseline" else "complete"
                common.append_jsonl(sd / "steps.jsonl", {"step": step, "status": state})
            return ""
        ns = dict(RUN_MEASUREMENTS=True, USE_DRIVE=True, command=command, PY="python", H=HERE, RESULT=self.root,
                  SD=sd, SESSION=1, LABEL="label", STEP_ORDER=["cpu-baseline", "gpu-rates"],
                  plan={"steps": ["cpu-baseline", "gpu-rates"]}, read_lines=ops.rows,
                  GPU_ONLY={"gpu-rates"}, checks={"validated_paths": ["inc"]}, RETRY_STEPS=[],
                  BUILD=self.root, save_backup=lambda: None, subprocess=subprocess)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(cell, str(SOURCE), "exec"), ns)
        self.assertEqual(executed, ["cpu-baseline", "gpu-rates"])

    def test_command_timeout_preserves_console_and_status(self):
        tree = ast.parse(SOURCE.read_text())
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("command", "operation")]
        ops_dir = self.root / "ops"
        ops_dir.mkdir()
        ns = dict(ROOT=self.root, OPS=ops_dir, OP_REMOTE=None, subprocess=subprocess, os=os, signal=signal,
                  time=time, threading=threading, json=json, kill_runners=lambda: None)
        exec(compile(ast.Module(body=funcs, type_ignores=[]), str(SOURCE), "exec"), ns)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(subprocess.TimeoutExpired):
            ns["command"](sys.executable, "-c", "import time; print('evidence', flush=True); time.sleep(10)", timeout=0.2)
        self.assertIn("evidence", next(ops_dir.glob("*.log")).read_text())
        self.assertTrue(ops.rows(ops_dir / "commands.jsonl")[-1]["timeout"])

    def test_notebook_is_generated_and_code_cells_parse(self):
        spec = importlib.util.spec_from_file_location("py2nb", HERE.parents[1] / "tools/py2nb.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        nb = self.root / "notebook.ipynb"
        with contextlib.redirect_stdout(io.StringIO()):
            mod.convert(SOURCE, nb)
        self.assertEqual(nb.read_bytes(), SOURCE.with_suffix(".ipynb").read_bytes())
        for cell in json.loads(nb.read_text())["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
                self.assertEqual(cell["outputs"], [])

    def test_raw_runner_console_is_not_truncated(self):
        build = self.root / "build"
        build.mkdir()
        runner = build / "sf_run"
        runner.write_text(f"#!{sys.executable}\nimport sys\nprint('original stdout')\n"
                          "sys.stderr.write('FIRST' + 'x' * 4000 + 'LAST')\nsys.exit(1)\n")
        runner.chmod(0o755)
        with patch.object(harness, "ATTEMPT_LOG", self.root / "session-1/attempts.jsonl"), \
             patch.object(harness, "TIME_BOX_NS", None):
            r = harness.run_cold("synthetic-task.json", self.root / "out", harness.cpu_config(1), build=build,
                                 timeout_s=5, drop=False)
            self.assertEqual(r["status"], "error")
            self.assertTrue((self.root / "out/stderr.log").read_text().startswith("FIRST"))
            b = harness.bench("synthetic-task.json", harness.cpu_config(1), 0.01, build=build)
            self.assertTrue(b["stderr"].startswith("FIRST"))
            self.assertEqual((Path(b["console_dir"]) / "stdout.log").read_text(), "original stdout\n")

    def test_lock_retains_direct_versions_and_matches_environment(self):
        from importlib.metadata import version
        def pins(path):
            return dict(x.split("==") for x in path.read_text().splitlines() if x and not x.startswith("#"))
        direct, locked = pins(HERE / "requirements.txt"), pins(HERE / "requirements-colab.lock")
        self.assertTrue(direct.items() <= locked.items())
        self.assertEqual({p: version(p) for p in locked}, locked)


if __name__ == "__main__":
    unittest.main(verbosity=2)

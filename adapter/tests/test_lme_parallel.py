import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from aml_adapter.lme_parallel import run_parallel, stop_worker


def state(count=12):
    return {"selection": {"variant": "s"}, "cases": [
        {"subset_index": i, "question_id": f"q-{i}", "status": "pending", "attempts": []}
        for i in range(count)]}


class ParallelTests(unittest.TestCase):
    def test_six_real_processes_failure_drains_then_resume_skips_completed(self):
        self.check_failure_and_resume(stop_on_error=True)

    def test_six_real_processes_continue_after_failure_and_resume_only_failed(self):
        self.check_failure_and_resume(stop_on_error=False)

    def check_failure_and_resume(self, stop_on_error):
        popen = subprocess.Popen
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            batch = state()
            args = SimpleNamespace(concurrency=6, answer_cutoff=50, allow_http_models=True,
                                   memory_profile="schema-entity", stop_on_error=stop_on_error)
            started, processes, peaks, snapshots = [], [], [], []
            fail_first = True

            def launch(command, **kwargs):
                index = int(command[command.index("--sample-index") + 1])
                path = command[command.index("--output-dir") + 1]
                self.assertIn("--no-oracle-sessions", command)
                self.assertIn("--allow-http-models", command)
                self.assertNotIn("--oracle-sessions", command)
                self.assertEqual(command[command.index("--memory-profile") + 1], "schema-entity")
                self.assertTrue(kwargs["env"]["AML_LME_REPOSITORY"].endswith("agent-memory-schema-pr337"))
                started.append(index)
                kwargs["env"] = {**kwargs["env"], "TEST_WORKER_MARKER": str(index)}
                worker = Path(__file__).with_name("parallel_worker_stub.py")
                process = popen([command[0], str(worker), path, str(root / "markers"), str(index),
                                 "1" if index == 0 and fail_first else "0"], **kwargs)
                processes.append(process)
                peaks.append(sum(p.poll() is None for p in processes))
                return process

            def checkpoint():
                snapshots.append(json.loads(json.dumps(batch)))
                self.assertLessEqual(sum(c["status"] == "running" for c in batch["cases"]), 6)

            with patch("aml_adapter.lme_parallel.subprocess.Popen", side_effect=launch):
                self.assertEqual(run_parallel(root, batch, args, root / "env", root, checkpoint), 1)
                self.assertEqual(started, list(range(6 if stop_on_error else 12)))
                self.assertEqual([c["status"] for c in batch["cases"]],
                                 ["failed"] + ["completed"] * (5 if stop_on_error else 11)
                                 + (["pending"] * 6 if stop_on_error else []))
                args.skip_failed = True
                self.assertEqual(run_parallel(root, batch, args, root / "env", root, checkpoint), 1)
                self.assertEqual(started.count(0), 1)
                self.assertEqual([c["status"] for c in batch["cases"]], ["failed"] + ["completed"] * 11)
                args.skip_failed = False
                fail_first = False
                self.assertEqual(run_parallel(root, batch, args, root / "env", root, checkpoint), 0)
            self.assertEqual(max(peaks), 6)
            self.assertEqual(started.count(0), 2)
            self.assertTrue(all(started.count(i) == 1 for i in range(1, 12)))
            self.assertTrue(all(c["status"] == "completed" for c in batch["cases"]))
            pids = set()
            for case in batch["cases"]:
                result = json.loads(Path(case["result"]).read_text())
                self.assertEqual(result["worker_marker"], str(case["subset_index"]))
                pids.add(result["worker_pid"])
            self.assertEqual(len(pids), 12)
            self.assertTrue(all(p.poll() is not None for p in processes))
            self.assertTrue(snapshots)

    def test_interrupt_stops_active_trees_and_preserves_pending(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            batch = state(3)
            args = SimpleNamespace(concurrency=2, answer_cutoff=50, allow_http_models=False)
            processes = [SimpleNamespace(pid=101, poll=lambda: None),
                         SimpleNamespace(pid=102, poll=lambda: None)]
            with patch("aml_adapter.lme_parallel.subprocess.Popen", side_effect=processes), \
                 patch("aml_adapter.lme_parallel.time.sleep", side_effect=KeyboardInterrupt), \
                 patch("aml_adapter.lme_parallel.stop_worker") as stop:
                self.assertEqual(run_parallel(root, batch, args, root / "env", root, lambda: None), 130)
            self.assertEqual(stop.call_count, 2)
            self.assertEqual([c["status"] for c in batch["cases"]],
                             ["interrupted", "interrupted", "pending"])

    def test_windows_cleanup_targets_owned_process_tree(self):
        process = SimpleNamespace(pid=12345, poll=lambda: None, wait=lambda timeout: None)
        with patch("aml_adapter.lme_parallel.os.name", "nt"), \
             patch("aml_adapter.lme_parallel.subprocess.run", return_value=SimpleNamespace(returncode=0)) as run:
            stop_worker(process)
        self.assertEqual(run.call_args.args[0], ["taskkill", "/PID", "12345", "/T", "/F"])


if __name__ == "__main__":
    unittest.main()

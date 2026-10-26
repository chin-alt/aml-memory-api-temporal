import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aml_adapter.lme_batch import apply_compatible_upgrade, load_selection, main as batch_main, summarize
from aml_adapter.lme_sampling import BUCKETS, bucket, has_standard_answer, prepare, select


def rows():
    result = []
    for name in BUCKETS:
        for i in range(4):
            result.append({"question_id": f"{name}-{i}" + ("_abs" if name == "abstention" else ""),
                           "question_type": "multi-session" if name == "abstention" else name,
                           "question": "synthetic question", "answer": "gold-hidden",
                           "question_date": "2024/01/03 (Wed) 00:00",
                           "haystack_session_ids": ["evidence", "distractor"],
                           "haystack_dates": ["2024/01/01 (Mon) 00:00", "2024/01/02 (Tue) 00:00"],
                           "answer_session_ids": ["evidence"],
                           "haystack_sessions": [[{"role": "user", "content": "fact", "has_answer": True}],
                                                 [{"role": "assistant", "content": "unrelated text"}]]})
    return result


class SamplingTests(unittest.TestCase):
    def test_balanced_deterministic_nested_selection_retains_full_history(self):
        source = rows()
        chosen = select(source, 3, 42)
        ids = [row["question_id"] for _, row in chosen]
        self.assertEqual(len(set(ids)), 21)
        self.assertEqual([bucket(row) for _, row in chosen], list(BUCKETS) * 3)
        self.assertEqual(ids, [row["question_id"] for _, row in select(source[::-1], 3, 42)])
        self.assertEqual(ids[:7], [row["question_id"] for _, row in select(source, 1, 42)])
        self.assertTrue(all(len(row["haystack_sessions"]) == 2 for _, row in chosen))
        with self.assertRaisesRegex(ValueError, "available"):
            select(source[:1], 3)

    def test_export_does_not_modify_rows_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = rows()
            (root / "source.json").write_text(json.dumps(source), encoding="utf-8")
            prepare(root / "source.json", root / "selection")
            manifest, data = load_selection(root / "selection/manifest.json")
            selected = json.loads(data.read_text(encoding="utf-8"))
            for case, row in zip(manifest["cases"], selected):
                self.assertEqual(row, source[case["source_index"]])
            data.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                load_selection(root / "selection/manifest.json")

    def test_batch_stops_on_failure_resumes_and_keeps_wrong_answers_completed(self):
        self.check_batch_failure_resume(stop_on_error=True)

    def test_batch_continues_on_failure_and_skip_failed_preserves_results(self):
        self.check_batch_failure_resume(stop_on_error=False)

    def check_batch_failure_resume(self, stop_on_error):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            (root / "source.json").write_text(json.dumps(rows()), encoding="utf-8")
            prepare(root / "source.json", root / "selection")
            (root / "env").write_text("", encoding="utf-8")
            args = ["--manifest", str(root / "selection/manifest.json"), "--env-file", str(root / "env")]
            if stop_on_error:
                args.append("--stop-on-error")
            calls = []

            def fake_run(arguments):
                self.assertIn("--no-oracle-sessions", arguments)
                self.assertNotIn("--oracle-sessions", arguments)
                index = int(arguments[arguments.index("--sample-index") + 1])
                path = Path(arguments[arguments.index("--output-dir") + 1])
                path.mkdir(parents=True)
                fail = index == 1 and calls.count(1) == 0
                calls.append(index)
                result = {"protocol_success": not fail}
                if not fail:
                    result["qa"] = {"verdict": {"correct": index != 0}}
                    result["qa_cutoffs"] = {"50": index != 0, "10": False, "20": True}
                (path / "result.json").write_text(json.dumps(result), encoding="utf-8")
                return 1 if fail else 0

            with patch("aml_adapter.lme_batch.ROOT", root), \
                 patch("aml_adapter.lme_batch.fingerprint", return_value="stable"), \
                 patch("aml_adapter.lme_batch.run_case", side_effect=fake_run) as run:
                self.assertEqual(batch_main(args), 0)
                run.assert_not_called()
                self.assertEqual(batch_main([*args, "--run"]), 1)
                batch = next((root / "outputs/longmemeval-batch").iterdir())
                report = json.loads((batch / "summary.json").read_text())
                self.assertEqual(report["completed"], 1 if stop_on_error else 20)
                self.assertIsNone(report["subset_accuracy"])
                self.assertEqual(batch_main([*args, "--run", "--resume", str(batch), "--skip-failed"]), 1)
                self.assertEqual(calls.count(1), 1)
                report = json.loads((batch / "summary.json").read_text())
                self.assertEqual(report["completed"], 20)
                self.assertEqual(report["pending"], 0)
                self.assertEqual(report["failed_or_interrupted"], 1)
                self.assertIsNone(report["subset_accuracy"])
                self.assertEqual(batch_main([*args, "--run", "--resume", str(batch)]), 0)
                report = json.loads((batch / "summary.json").read_text())
                self.assertEqual(report["completed"], 21)
                self.assertEqual(report["correct"], 20)
                self.assertEqual(report["qa_cutoffs"]["10"]["correct"], 0)
                self.assertEqual(report["qa_cutoffs"]["20"]["subset_accuracy"], 1)
                self.assertEqual(calls.count(0), 1)
                self.assertEqual(calls.count(1), 2)
                self.assertEqual(batch_main([*args, "--run", "--resume", str(batch)]), 0)
                self.assertEqual(len(calls), 22)

    def test_resume_upgrades_require_exact_fingerprints_and_matching_profile(self):
        upgrades = [{"from": "old", "to": "new", "memory_profile": "schema-entity"}]
        for profile, identity, expected in [("ordinary", "new", "old"),
                                             ("schema-entity", "changed", "old"),
                                             ("schema-entity", "new", "new")]:
            with self.subTest(profile=profile, identity=identity):
                state = {"fingerprint": "old"}
                apply_compatible_upgrade(state, identity, profile, upgrades)
                self.assertEqual(state["fingerprint"], expected)
                self.assertEqual(len(state.get("runner_upgrades", [])), int(expected == "new"))

    def test_retrieval_only_never_reports_qa_accuracy(self):
        result = summarize([{"bucket": "multi-session", "status": "completed"}], retrieval_only=True)
        self.assertIsNone(result["subset_accuracy"])
        self.assertIsNone(result["accuracy_on_graded"])

    def test_unsupported_manifest_is_rejected_before_running_models(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stderr(io.StringIO()):
            root = Path(temp)
            (root / "source.json").write_text(json.dumps(rows()), encoding="utf-8")
            prepare(root / "source.json", root / "selection", variant="custom")
            with patch("aml_adapter.lme_batch.run_case") as run, self.assertRaises(SystemExit):
                batch_main(["--manifest", str(root / "selection/manifest.json"), "--run"])
            run.assert_not_called()

    def test_answerable_selection_filters_before_sampling_and_keeps_nested_ids(self):
        source = rows()
        for name, blank in zip(BUCKETS[:-1], (None, "", "  ", [], {}, False)):
            row = next(row for row in source if bucket(row) == name)
            row["answer"] = blank
        chosen = select(source, 3, 42, answerable_only=True)
        self.assertEqual(len(chosen), 18)
        self.assertEqual([bucket(row) for _, row in chosen], list(BUCKETS[:-1]) * 3)
        self.assertTrue(all(has_standard_answer(row) for _, row in chosen))
        self.assertEqual([row["question_id"] for _, row in chosen[:6]],
                         [row["question_id"] for _, row in select(source, 1, 42, answerable_only=True)])
        self.assertEqual([row["question_id"] for _, row in chosen],
                         [row["question_id"] for _, row in select(source[::-1], 3, 42, answerable_only=True)])

    def test_oracle_batch_selects_oracle_mode_and_reports_separately(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            (root / "source.json").write_text(json.dumps(rows()), encoding="utf-8")
            prepare(root / "source.json", root / "selection", per_type=1,
                    variant="oracle", answerable_only=True)
            (root / "env").write_text("", encoding="utf-8")

            def fake_run(arguments):
                self.assertIn("--oracle-sessions", arguments)
                self.assertNotIn("--no-oracle-sessions", arguments)
                path = Path(arguments[arguments.index("--output-dir") + 1])
                path.mkdir(parents=True)
                (path / "result.json").write_text(json.dumps({"protocol_success": True,
                    "qa": {"verdict": {"correct": True}},
                    "qa_cutoffs": {"50": True, "10": True, "20": True}}), encoding="utf-8")
                return 0

            with patch("aml_adapter.lme_batch.ROOT", root), \
                 patch("aml_adapter.lme_batch.fingerprint", return_value="stable"), \
                 patch("aml_adapter.lme_batch.run_case", side_effect=fake_run) as run:
                self.assertEqual(batch_main(["--manifest", str(root / "selection/manifest.json"),
                                             "--env-file", str(root / "env"), "--run"]), 0)
                self.assertEqual(run.call_count, 6)
            batch = next((root / "outputs/longmemeval-batch").iterdir())
            self.assertTrue(batch.name.startswith("oracle-6-"))
            report = json.loads((batch / "summary.json").read_text())
            self.assertEqual(report["context_mode"], "oracle_sessions")
            self.assertTrue(report["answerable_only"])
            self.assertEqual(report["planned"], 6)


if __name__ == "__main__":
    unittest.main()

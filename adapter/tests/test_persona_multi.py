import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from aml_adapter import persona_batch as base
from aml_adapter import persona_multi as multi
from aml_adapter.diagnostics import DiagnosticLog
from aml_adapter.persona_data import official, save


class MultiCutoffTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.pipeline = official()
        self.row = {"id": "q", "persona_id": "p", "pref_type": "neutral", "history_file": "history.json",
                    "user_query": "{'role':'user','content':'Dinner?'}", "preference": "SECRET_PREF",
                    "option_mapping": {"A": "soup", "B": "salad"}, "correct_letter": "A"}
        save(self.root / "history.json", [{"role": "user", "content": "I like soup."}])
        self.args = argparse.Namespace(qa_cutoffs=[10, 20, 50, 100], mode="both", top_k=100,
                                       mcq_format="original", skip_failed=False)
        self.state = {"profile": "schema-entity", "top_k": 100, "qa_cutoffs": self.args.qa_cutoffs,
                      "mcq_format": "original", "histories": {}, "questions": {"q": {
                          "status": "pending", "persona_id": "p", "pref_type": "neutral"}}}
        self.requests = []

    def transport(self, request):
        self.requests.append(request.url.path)
        payload = json.loads(request.content)
        self.assertNotIn("SECRET_PREF", str(payload))
        if request.url.path == "/add":
            return httpx.Response(200, json={"success": True})
        self.assertEqual(payload["top_k"], 100)
        return httpx.Response(200, json={"data": [{"content": f"memory-{i:03}"} for i in range(1, 101)]})

    def run_questions(self, complete):
        with httpx.Client(base_url="http://test", transport=httpx.MockTransport(self.transport)) as client:
            with patch.object(base, "DATA", self.root), patch.object(base, "complete", complete):
                base.run_questions(self.state, [self.row], client, self.pipeline, self.args, {}, self.root,
                                   DiagnosticLog(), lambda: save(self.root / "state.json", self.state))

    def test_one_ingestion_one_search_four_cutoffs_and_cached_resume(self):
        calls = []
        async def complete(pipeline, env, role, prompt, diagnostic):
            calls.append(role)
            if role == "answer":
                self.assertNotIn("SECRET_PREF", str(prompt))
            return "\\boxed{0.8}" if role == "judge" else "A"
        self.run_questions(complete)
        self.run_questions(complete)
        self.assertEqual(self.requests, ["/add", "/search"])
        self.assertEqual(len(calls), 12)
        self.assertEqual(self.state["questions"]["q"]["status"], "completed")
        for k in self.args.qa_cutoffs:
            result = self.state["questions"]["q"]["cutoffs"][str(k)]
            self.assertFalse(result["mcq_correct"])
            self.assertTrue(result["mcq_normalized_correct"])
            prompt = json.loads((self.root / f"questions/q/top-{k}/generative-prompt.json").read_text())
            self.assertIn(f"memory-{k:03}", prompt[0]["content"])
            self.assertNotIn(f"memory-{k + 1:03}", prompt[0]["content"])
        multi.write_summary(self.root, self.state)
        report = json.loads((self.root / "summary.json").read_text())
        self.assertEqual(report["completed"], 1)
        self.assertEqual(report["paired_ids"], ["q"])

    def test_one_cutoff_failure_continues_then_only_missing_stage_retried(self):
        calls = []
        async def complete(pipeline, env, role, prompt, diagnostic):
            calls.append(role)
            if len(calls) == 3:
                raise TimeoutError("judge unavailable")
            return "\\boxed{0.8}" if role == "judge" else "Final Answer: A"
        self.run_questions(complete)
        self.assertEqual(len(calls), 12)
        self.assertEqual(self.state["questions"]["q"]["status"], "failed")
        report = multi.build_report(self.state)
        self.assertEqual(report["qa_cutoffs"]["10"]["failed"], 1)
        self.assertEqual(report["qa_cutoffs"]["100"]["completed"], 1)
        self.assertEqual(report["paired_ids"], [])
        self.args.skip_failed = True
        self.run_questions(complete)
        self.assertEqual(len(calls), 12)
        self.args.skip_failed = False
        self.run_questions(complete)
        self.assertEqual(calls[12:], ["judge"])
        self.assertEqual(self.requests, ["/add", "/search"])
        self.assertEqual(self.state["questions"]["q"]["status"], "completed")

    def test_compare_rejects_mismatched_cutoffs(self):
        left = copy.deepcopy(self.state)
        right = copy.deepcopy(self.state)
        right["qa_cutoffs"] = [10, 20]
        with self.assertRaisesRegex(ValueError, "qa_cutoffs"):
            multi.compare([self.root, self.root], left, right)

    def test_history_failure_propagates_to_all_cutoff_counts(self):
        self.state["questions"]["q"].update(status="failed", failure_stage="history")
        report = multi.build_report(self.state)
        self.assertEqual(len(report["failures"]), 4)
        self.assertTrue(all(r["failed"] == 1 and r["mcq_accuracy"] is None
                            for r in report["qa_cutoffs"].values()))


if __name__ == "__main__":
    unittest.main()

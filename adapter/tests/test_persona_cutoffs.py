import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aml_adapter import persona_cutoffs as runner
from aml_adapter.diagnostics import DiagnosticLog
from aml_adapter.persona_data import digest, official, save


class CutoffTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source, self.output = self.root / "source", self.root / "output"
        self.row = {"id": "q1", "persona_id": "p1", "pref_type": "neutral",
                    "history_file": "history.json", "user_query": "{'role': 'user', 'content': 'Lunch?'}",
                    "preference": "PRIVATE_GOLD", "option_mapping": {"A": "soup", "B": "salad", "C": "bread"},
                    "correct_letter": "B"}
        self.pipeline = official()
        self.args = argparse.Namespace(mode="both", mcq_format="reinforced", skip_failed=False)
        self.state = {"profile": "ordinary", "selection": {"rows": [self.row]},
                      "cutoffs": [10, 20, 50], "source_top_k": 100, "mcq_format": "reinforced",
                      "questions": {"q1": {"persona_id": "p1", "pref_type": "neutral", "cutoffs": {
                          str(k): {"status": "pending_source"} for k in [10, 20, 50]}}}}
        save(self.source / "state.json", {"histories": {"p1:history.json": {"status": "completed"}},
             "questions": {"q1": {"status": "completed", "mcq_correct": False, "narrow_score": .8}}})
        self.memories = [{"content": f"memory-{i:03}"} for i in range(1, 101)]
        save(self.source / "questions/q1/search.json", {"data": self.memories})
        save(self.source / "questions/q1/mcq-answer.json", {"answer": "**B.**\nIt fits."})

    def test_normalization_is_conservative_and_gold_independent(self):
        cases = {"B": "B", "**B.**\nIt fits.": "B", "**B** is the best answer.": "B",
                 "The best answer is **B**.": "B", "B or C": "", "B. Maybe.\nC. Also.": "",
                 "**B.**\nThe best answer is **C**.": "", "I might choose B": "", "Z": "",
                 "Final Answer: C": "C", "B. fits, but A is an alternative.": "B",
                 "**B.**\nThis works for a person who wants more choices.": "B"}
        for answer, expected in cases.items():
            with self.subTest(answer=answer):
                self.assertEqual(runner.normalized_choice(answer, self.row["option_mapping"], self.pipeline), expected)
        artifact = self.output / "scores"
        artifact.mkdir(parents=True)
        scores = runner.score_mcq(self.pipeline, self.row, "B", artifact)
        self.assertFalse(scores["mcq_correct"])
        self.assertTrue(scores["mcq_normalized_correct"])
        different_gold = dict(self.row, correct_letter="A")
        scores = runner.score_mcq(self.pipeline, different_gold, "B", artifact)
        self.assertFalse(scores["mcq_normalized_correct"])
        self.assertEqual(runner.read(artifact / "mcq-normalization.json")["normalized_answer"], "Final Answer: B")

    def test_cutoffs_reuse_search_and_preserve_source(self):
        before = {str(p): digest(p) for p in self.source.rglob("*") if p.is_file()}
        calls = []
        async def complete(pipeline, env, role, prompt, diagnostic):
            calls.append((role, prompt))
            if role == "answer":
                self.assertNotIn("PRIVATE_GOLD", json.dumps(prompt))
                if prompt[-1]["role"] == "system":
                    self.assertIn(runner.FORMAT_INSTRUCTION, prompt[-1]["content"])
            return "\\boxed{0.8}" if role == "judge" else "Final Answer: B"
        with patch.object(runner.base, "complete", complete):
            runner.run(self.state, self.source, self.output, self.args, {}, self.pipeline, DiagnosticLog())
            runner.run(self.state, self.source, self.output, self.args, {}, self.pipeline, DiagnosticLog())
        self.assertEqual(len(calls), 9)
        for k in [10, 20, 50]:
            folder = self.output / f"questions/q1/top-{k}"
            prompt = runner.read(folder / "generative-prompt.json")
            self.assertIn(f"memory-{k:03}", prompt[0]["content"])
            self.assertNotIn(f"memory-{k + 1:03}", prompt[0]["content"])
            self.assertEqual(self.state["questions"]["q1"]["cutoffs"][str(k)]["status"], "completed")
        self.assertEqual(before, {str(p): digest(p) for p in self.source.rglob("*") if p.is_file()})
        report = runner.read(self.output / "summary.json")
        self.assertEqual(report["paired_ids"], ["q1"])
        self.assertEqual(report["all"]["original-top-100"]["mcq_accuracy"], 0)
        self.assertEqual(report["all"]["original-top-100"]["mcq_normalized_accuracy"], 1)

    def test_cutoff_failure_continues_and_resume_only_missing_judge(self):
        calls = []
        async def complete(pipeline, env, role, prompt, diagnostic):
            calls.append(role)
            if len(calls) == 3:
                raise TimeoutError("test")
            return "\\boxed{0.7}" if role == "judge" else "Final Answer: B"
        with patch.object(runner.base, "complete", complete):
            runner.run(self.state, self.source, self.output, self.args, {}, self.pipeline, DiagnosticLog())
            self.assertEqual(self.state["questions"]["q1"]["cutoffs"]["10"]["status"], "failed")
            self.assertEqual(self.state["questions"]["q1"]["cutoffs"]["50"]["status"], "completed")
            self.args.skip_failed = True
            runner.run(self.state, self.source, self.output, self.args, {}, self.pipeline, DiagnosticLog())
            self.assertEqual(len(calls), 9)
            self.args.skip_failed = False
            runner.run(self.state, self.source, self.output, self.args, {}, self.pipeline, DiagnosticLog())
        self.assertEqual(calls[9:], ["judge"])
        self.assertEqual(self.state["questions"]["q1"]["cutoffs"]["10"]["status"], "completed")

    def test_wait_for_source_then_resume_and_reject_changed_search(self):
        original = runner.read(self.source / "state.json")
        original["histories"]["p1:history.json"]["status"] = "running"
        save(self.source / "state.json", original)
        with patch.object(runner.base, "complete") as model:
            runner.run(self.state, self.source, self.output, self.args, {}, self.pipeline, DiagnosticLog())
            model.assert_not_called()
        self.assertEqual(self.state["questions"]["q1"]["cutoffs"]["10"]["status"], "pending_source")
        original["histories"]["p1:history.json"]["status"] = "completed"
        save(self.source / "state.json", original)
        async def complete(*args):
            return "Final Answer: B"
        with patch.object(runner.base, "complete", complete):
            runner.run(self.state, self.source, self.output, self.args, {}, self.pipeline, DiagnosticLog())
        save(self.source / "questions/q1/search.json", {"data": []})
        with self.assertRaisesRegex(ValueError, "Source Search changed"):
            runner.run(self.state, self.source, self.output, self.args, {}, self.pipeline, DiagnosticLog())

    def test_negative_judge_unchanged(self):
        self.row["preference"] = "Do not use soup"
        self.args.mcq_format = "original"
        async def complete(pipeline, env, role, prompt, diagnostic):
            if role == "judge":
                self.assertEqual(prompt[0]["content"], pipeline.JUDGE_PROMPT_NARROW_NEGATIVE.format(
                    user_query="Lunch?", preference=self.row["preference"], model_response="Final Answer: B"))
                return "\\boxed{1}"
            if prompt[-1]["role"] == "system":
                self.assertNotIn(runner.FORMAT_INSTRUCTION, prompt[-1]["content"])
            return "Final Answer: B"
        with patch.object(runner.base, "complete", complete):
            runner.run(self.state, self.source, self.output, self.args, {}, self.pipeline, DiagnosticLog())

    def test_failure_denominators_and_paired_reports(self):
        records = [{"status": "completed", "mcq_correct": False, "mcq_normalized_correct": True},
                   {"status": "failed", "mcq_correct": True, "mcq_normalized_correct": True},
                   {"status": "pending_source"}]
        stats = runner.stats(records)
        self.assertEqual(stats["mcq_accuracy"], 0)
        self.assertEqual(stats["mcq_normalized_accuracy"], 1)
        self.assertEqual(stats["failed"], 1)
        self.assertEqual(stats["pending_source"], 1)


if __name__ == "__main__":
    unittest.main()

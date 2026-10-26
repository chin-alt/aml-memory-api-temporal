import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path

from aml_adapter.persona_data import DATA, save
from aml_adapter.persona_expansion import expand_state, extend_selection


class ExpansionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.selection = json.loads((DATA / "selection.json").read_text(encoding="utf-8"))
        self.args = argparse.Namespace(top_k=100, mode="both", qa_cutoffs=[10, 20, 50, 100], mcq_format="original")

    def test_frozen_70_keeps_25_and_same_histories(self):
        expanded = extend_selection(self.selection, 14)
        self.assertEqual(expanded, extend_selection(self.selection, 14))
        self.assertEqual(len(expanded["rows"]), 70)
        self.assertEqual(len({r["id"] for r in expanded["rows"]}), 70)
        self.assertEqual(expanded["files"], self.selection["files"])
        new = {r["id"]: r for r in expanded["rows"]}
        for row in self.selection["rows"]:
            self.assertEqual(new[row["id"]], row)
        for person in {r["persona_id"] for r in expanded["rows"]}:
            self.assertEqual(sum(r["persona_id"] == person for r in expanded["rows"]), 14)
        with self.assertRaisesRegex(ValueError, "fewer than 15"):
            extend_selection(self.selection, 15)

    def make_state(self):
        return {"profile": "ordinary", "fingerprint": "old", "selection": self.selection,
                "mode": "both", "top_k": 100, "histories": {"p": {"status": "completed", "next_add": 15}},
                "questions": {r["id"]: {"status": "completed", "persona_id": r["persona_id"],
                    "pref_type": r["pref_type"], "mcq_correct": True, "narrow_score": .8}
                              for r in self.selection["rows"]}}

    def test_promotes_cached_top100_without_modifying_original_artifacts(self):
        state = self.make_state()
        old = copy.deepcopy(state)
        row = self.selection["rows"][0]
        artifact = self.root / "questions" / row["id"]
        save(artifact / "mcq-answer.json", {"answer": "Final Answer: " + row["correct_letter"]})
        save(artifact / "generative-answer.json", {"answer": "original answer"})
        save(artifact / "judge.json", {"response": "\\boxed{0.8}", "score": .8})
        before = (artifact / "generative-answer.json").read_bytes()
        expanded = extend_selection(self.selection, 14)
        result = expand_state(state, expanded, self.args, "new", self.root)
        self.assertEqual(state, old)
        self.assertEqual(result["histories"], old["histories"])
        self.assertEqual(len(result["questions"]), 70)
        self.assertEqual((artifact / "generative-answer.json").read_bytes(), before)
        self.assertEqual((artifact / "top-100/generative-answer.json").read_bytes(), before)
        for r in self.selection["rows"]:
            q = result["questions"][r["id"]]
            self.assertEqual(q["cutoffs"]["100"]["status"], "completed")
            self.assertEqual(q["cutoffs"]["100"]["narrow_score"], .8)
            self.assertEqual(q["cutoffs"]["10"]["status"], "pending")
        again = expand_state(result, expanded, self.args, "new", self.root)
        self.assertEqual(result, again)

    def test_rejects_changed_existing_question_or_prompt(self):
        expanded = extend_selection(self.selection, 14)
        bad = copy.deepcopy(expanded)
        bad["rows"][0]["correct_letter"] = "Z"
        with self.assertRaisesRegex(ValueError, "preserve every previous"):
            expand_state(self.make_state(), bad, self.args, "new", self.root)
        self.args.mcq_format = "reinforced"
        with self.assertRaisesRegex(ValueError, "original MCQ prompt"):
            expand_state(self.make_state(), expanded, self.args, "new", self.root)

    def test_schema_multicutoff_statuses_are_retained(self):
        state = self.make_state()
        state.update(profile="schema-entity", qa_cutoffs=[10,20,50,100], mcq_format="original")
        first = next(iter(state["questions"].values()))
        first.update(status="failed", cutoffs={"10": {"status": "completed", "narrow_score": .7},
                                             "20": {"status": "failed"}})
        result = expand_state(state, extend_selection(self.selection, 14), self.args, "new", self.root)
        self.assertEqual(result["questions"][self.selection["rows"][0]["id"]], first)
        self.assertEqual(len(result["questions"]), 70)


if __name__ == "__main__":
    unittest.main()

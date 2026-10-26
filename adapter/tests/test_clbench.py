import argparse
import asyncio
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import httpx

from aml_adapter import clbench_batch as runner
from aml_adapter.clbench_data import CATEGORIES, convert_row, official, payloads, sample_rows, sample_stratified
from aml_adapter.diagnostics import DiagnosticLog
from aml_adapter.lme_profiles import memory_config
from aml_adapter.models import AddRequest, MemoryItem, SearchResponse
from pydantic import ValidationError


def raw_row(category=CATEGORIES[0], suffix="1"):
    return {"messages": [{"role": "system", "content": "Original system instruction"},
                         {"role": "user", "content": "PRIVATE_HISTORY document with facts"},
                         {"role": "assistant", "content": "Previous answer"},
                         {"role": "user", "content": "CURRENT_QUERY"}],
            "rubrics": ["SECRET_RUBRIC"],
            "metadata": {"task_id": category + suffix, "context_id": "ctx" + suffix,
                         "context_category": category, "sub_category": "sub"}}


class CLBenchTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.diagnostic = DiagnosticLog(self.root / "diagnostics.jsonl")
        self.pipeline = official()
        self.row = convert_row(raw_row())
        self.args = argparse.Namespace(skip_failed=False, top_k=100)

    def test_stratified_40_unique_contexts_length_bands_and_determinism(self):
        rows = []
        for category in CATEGORIES:
            for context in range(36):
                for question in range(2):
                    row = raw_row(category, f"-{category}-{context}")
                    row["metadata"]["task_id"] += f"-{question}"
                    row["messages"][0]["content"] = "system"
                    row["messages"][1]["content"] = "fact " * (10 + context * 100 + question)
                    rows.append(row)
        selected, pools = sample_stratified(rows)
        self.assertEqual((selected, pools), sample_stratified(list(reversed(rows))))
        self.assertEqual(len(selected), 40)
        self.assertEqual(len({r["context_id"] for r in selected}), 40)
        self.assertNotEqual(selected, sample_stratified(rows, seed=43)[0])
        for category in CATEGORIES:
            group = [r for r in selected if r["category"] == category]
            self.assertEqual(len(group), 10)
            bands = {b: [r for r in group if r["history_band"] == b] for b in ("short", "medium", "long")}
            self.assertEqual([len(v) for v in bands.values()], [3, 4, 3])
            self.assertLess(max(r["history_words"] for r in bands["short"]),
                            min(r["history_words"] for r in bands["medium"]))
            self.assertLess(max(r["history_words"] for r in bands["medium"]),
                            min(r["history_words"] for r in bands["long"]))
            self.assertEqual(pools[category]["eligible_contexts"], 36)
        with self.assertRaises(ValueError):
            sample_stratified(rows[:4])

    def test_deterministic_balanced_sample_and_exact_input_boundary(self):
        rows = [raw_row(c, str(i)) for c in CATEGORIES for i in range(8)]
        selected, _ = sample_rows(rows)
        self.assertEqual(selected, sample_rows(rows)[0])
        self.assertEqual({r["category"] for r in selected}, set(CATEGORIES))
        self.assertEqual(len(selected), 4)
        for row in selected:
            chunks = payloads(row, "scope")
            for chunk in chunks:
                AddRequest.model_validate(chunk)
                self.assertNotIn("SECRET_RUBRIC", json.dumps(chunk))
                self.assertNotIn("CURRENT_QUERY", json.dumps(chunk))
            self.assertEqual([m for c in chunks for m in c["messages"]], row["history"])
            item = runner.answer_item(row, [{"content": "RETRIEVED_FACT"}])
            prompt = self.pipeline.build_answer_prompt(item)
            for expected in ("CURRENT_QUERY", "Original system instruction", "RETRIEVED_FACT"):
                self.assertIn(expected, prompt)
            self.assertNotIn("SECRET_RUBRIC", prompt)
            self.assertNotIn("PRIVATE_HISTORY", prompt)

    def test_official_judge_and_no_thinking_on_wire(self):
        env = {"AML_LLM_BASE_URL": "https://model.test/v1", "AML_LLM_API_KEY": "test",
               "AML_LLM_MODEL": "Qwen3.8-Flash"}
        seen = []

        def transport(request):
            body = json.loads(request.content)
            seen.append(body)
            self.assertIs(body["enable_thinking"], False)
            self.assertEqual(body["temperature"], 0)
            content = "answer" if len(seen) == 1 else json.dumps({
                "Overall Score": 0, "Grading Rationale": "One requirement not met",
                "List of Requirement Satisfaction Status": ["Satisfied", "Not Satisfied"]})
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

        original = httpx.AsyncClient
        with patch.object(runner.httpx, "AsyncClient", side_effect=lambda **kw: original(
                transport=httpx.MockTransport(transport), **kw)):
            answer = asyncio.run(runner.chat_stage(self.pipeline, "answer", self.row, "PROMPT", None,
                                                   env, self.root, self.diagnostic))
            score = asyncio.run(runner.chat_stage(self.pipeline, "judge", self.row, None, answer,
                                                  env, self.root, self.diagnostic))
        self.assertEqual(seen[0]["messages"], [{"role": "user", "content": "PROMPT"}])
        self.assertIn("SECRET_RUBRIC", json.dumps(seen[1]))
        self.assertEqual(score["rubric_clbench_score"], 0)
        self.assertFalse(runner.judge_failed(score))
        self.assertEqual(len(list(self.root.glob("*-response-*.json"))), 2)

    def test_fail_continue_resume_reuses_add_search_answer(self):
        rows = [self.row, convert_row(raw_row(suffix="2"))]
        state = {"selection": {"rows": rows}, "questions": {
            r["id"]: {"status": "pending", "stage": "pending", "next_add": 0,
                      "user_id": "scope:" + r["id"], "category": r["category"]} for r in rows}}
        calls, chat_calls = [], []

        def post(client, endpoint, body, diagnostic):
            calls.append((endpoint, copy.deepcopy(body)))
            return (SearchResponse(data=[MemoryItem(id="m1", content="fact")]).model_dump()
                    if endpoint == "/search" else {"success": True})

        async def chat(pipeline, role, row, prompt, answer, *rest):
            chat_calls.append((row["id"], role))
            if role == "answer":
                return "answer"
            if row["id"] == rows[0]["id"] and chat_calls.count((row["id"], role)) == 1:
                return {"rubric_clbench_score": 0, "rubric_clbench_rationale": "API call failed (counted as score 0)"}
            return {"rubric_clbench_score": 1, "rubric_clbench_requirement_ratio": 1}

        with patch.object(runner, "http_post", side_effect=post), patch.object(runner, "chat_stage", side_effect=chat):
            def run():
                runner.run_cases(state, Mock(), self.pipeline, self.args, {}, self.root,
                                 self.diagnostic, lambda: None)
            run()
            self.assertEqual([r["status"] for r in state["questions"].values()], ["failed", "completed"])
            self.assertIsNone(runner.stats(state["questions"].values())["accuracy_full_sample"])
            self.args.skip_failed = True
            run()
            self.assertEqual(len(calls), 4)
            self.args.skip_failed = False
            run()
            self.assertEqual(len(calls), 4)
            self.assertEqual(chat_calls.count((rows[0]["id"], "answer")), 1)
            self.assertEqual(chat_calls.count((rows[0]["id"], "judge")), 2)
            self.assertTrue(all(r["status"] == "completed" for r in state["questions"].values()))
            self.assertNotEqual(calls[0][1]["user_id"], calls[2][1]["user_id"])

    def test_search_contract_empty_allowed_malformed_rejected(self):
        self.assertEqual(SearchResponse.model_validate({"data": []}).data, [])
        for result in ({"success": True}, {"data": "wrong"}, {"data": [{"id": "m1"}]},
                       {"success": False, "data": []}):
            with self.assertRaises(ValidationError):
                SearchResponse.model_validate(result)

    def test_exact_resume_upgrade_preserves_checkpoints_rejects_other_changes(self):
        prior = {"profile": "ordinary", "fingerprint": "old", "namespace": "original-scope",
                 "questions": {"q": {"next_add": 2, "status": "failed", "stage": "search"}}}
        original = copy.deepcopy(prior)
        upgrade = {"profile": "ordinary", "from": "old", "to": "new", "reason": "Search contract"}
        for signature, profile in (("unrelated", "ordinary"), ("new", "schema-entity")):
            with self.assertRaises(ValueError):
                runner.validate_resume(prior, signature, profile, [upgrade])
            self.assertEqual(prior, original)
        runner.validate_resume(prior, "new", "ordinary", [upgrade])
        self.assertEqual(prior["questions"], original["questions"])
        self.assertEqual(prior["namespace"], original["namespace"])
        self.assertEqual(prior["fingerprint"], "new")

    def test_completed_ingestion_restarts_at_search(self):
        state = {"selection": {"rows": [self.row]}, "questions": {
            self.row["id"]: {"status": "failed", "stage": "search", "next_add": 1,
                              "user_id": "original-scope", "category": self.row["category"]}}}
        endpoints = []

        def post(client, endpoint, body, diagnostic):
            endpoints.append(endpoint)
            self.assertEqual(body["user_id"], "original-scope")
            return SearchResponse(data=[MemoryItem(id="m1", content="fact")]).model_dump()

        async def chat(pipeline, role, *args):
            return "answer" if role == "answer" else {
                "rubric_clbench_score": 1, "rubric_clbench_requirement_ratio": 1}

        with patch.object(runner, "http_post", side_effect=post), patch.object(runner, "chat_stage", side_effect=chat):
            runner.run_cases(state, Mock(), self.pipeline, self.args, {}, self.root, self.diagnostic, lambda: None)
        self.assertEqual(endpoints, ["/search"])
        self.assertEqual(state["questions"][self.row["id"]]["status"], "completed")

    def test_profiles_close_extraction_thinking_and_schema_switches(self):
        ordinary = memory_config("ordinary")
        schema = memory_config("schema-entity")
        for config in (ordinary, schema):
            self.assertIs(config["llm"]["default"]["params"]["enable_thinking"], False)
        self.assertFalse(ordinary["globals"]["entity_enabled"])
        self.assertTrue(schema["globals"]["entity_enabled"])
        self.assertEqual(schema["extractor"]["default"]["target"], "entity_schema")
        self.assertEqual(schema["evolver"]["default"]["target"], "schema_orchestrating")


if __name__ == "__main__":
    unittest.main()

"""原 Runner + HTTP ASGI + 真内存内核；仅模型响应使用离线桩。"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml
from fastapi.testclient import TestClient

from aml_adapter.app import create_app
from aml_adapter.backend import Mem2Backend
from aml_adapter.repo_lme import normalize_result
from aml_adapter.lme_profiles import memory_config, repository_root
from evaluation.longmemeval import run
from evaluation.longmemeval.metrics.mem0_longmemeval_prompt import (
    get_answer_generation_prompt, get_judge_prompt,
)

ROOT = Path(__file__).resolve().parents[1]


class RepositoryHTTPTests(unittest.TestCase):
    def test_original_runner_prompts_dates_scores_and_http_memory(self):
        self.check_original_runner(enhanced=False)

    def test_bm25_fidelity_original_runner_http_metadata_and_fuser(self):
        self.check_original_runner(enhanced=True)

    def test_score_max_fidelity_original_runner_http_metadata_and_fuser(self):
        self.check_original_runner(enhanced=True, profile="score-max-fidelity")

    def test_score_max_profile_only_changes_fuser_from_bm25(self):
        bm25, score_max = memory_config("bm25-fidelity"), memory_config("score-max-fidelity")
        self.assertEqual(repository_root("bm25-fidelity"), repository_root("score-max-fidelity"))
        self.assertEqual(score_max["fuser"]["default"]["target"], "score_max")
        self.assertEqual(score_max["fuser"]["default"]["params"]["fusion_channel_weights"],
                         bm25["fuser"]["default"]["params"]["fusion_channel_weights"])
        score_max["fuser"] = bm25["fuser"]
        self.assertEqual(score_max, bm25)

    def test_bm25_profile_changes_only_fusion_and_fidelity(self):
        ordinary, enhanced = memory_config("ordinary"), memory_config("bm25-fidelity")
        self.assertEqual(repository_root("ordinary"), repository_root("bm25-fidelity"))
        self.assertTrue(enhanced["globals"].pop("extraction_fidelity_mode"))
        self.assertEqual(enhanced["fuser"]["default"]["target"], "BM25_scored_fuser")
        enhanced["fuser"] = ordinary["fuser"]
        self.assertEqual(enhanced, ordinary)

    def test_fidelity_metadata_changes_only_opted_in_extraction_prompt(self):
        from jiuwen_memory.common.type_def import MemoryUnit, Segment
        from jiuwen_memory.construction.extractor_impl.llm_extractor import ExtractorImpl
        prompts = []

        def chat(messages, **kwargs):
            prompts.append(messages[0].content)
            return "[]"

        extractor = ExtractorImpl(llm=SimpleNamespace(chat=chat))
        for enabled in (False, True, False):
            extractor.extract([MemoryUnit(segments=[Segment(content="Alice owns three bicycles.")], system_metadata={
                "extraction_fidelity_mode": enabled, "observation_date": "2024-01-01"})])
        self.assertEqual(len(prompts), 3)
        self.assertEqual(prompts[0], prompts[2])
        self.assertNotIn("Additional fidelity contract for this write only:", prompts[0])
        self.assertIn("Additional fidelity contract for this write only:", prompts[1])

    def check_original_runner(self, enhanced, profile="bm25-fidelity"):
        row = {"question_id": "offline", "question_type": "single-session-user",
               "question": "What is my favorite fruit?", "answer": "GOLD_ONLY_SENTINEL",
               "question_date": "2024/01/03 (Wed) 00:00",
               "haystack_session_ids": ["s1", "distractor"], "answer_session_ids": ["s1"],
               "haystack_dates": ["2024/01/01 (Mon) 00:00", "2024/01/01 (Mon) 01:00"],
               "haystack_sessions": [[{"role": "user", "content": "My favorite fruit is mango.",
                                       "has_answer": True},
                                      {"role": "assistant", "content": "You like mango."}],
                                     [{"role": "user", "content": "Mango shop also sells bicycles."}]]}
        calls, posts = [], []
        verdicts = iter(("yes", "no", "unparseable"))

        def fake_chat(base, model, key, options):
            def chat(system, user):
                calls.append((options, system, user))
                return "mango" if options.audit_category == "answer" else next(verdicts)
            return chat

        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {
            "AML_REPO_EVAL": "true", "LME_MEMORY_API_KEY": "test-only",
        }):
            temp = Path(temporary)
            dataset = temp / "input.json"
            dataset.write_text(json.dumps([row]), encoding="utf-8")
            config = ROOT / "config/local-smoke.yml"
            if enhanced:
                offline = yaml.safe_load(config.read_text(encoding="utf-8"))
                offline["fuser"] = memory_config(profile)["fuser"]
                offline["globals"].update(extraction_fidelity_mode=True, vector_enabled=True)
                offline["embedder"] = {"default": {"target": "hashing"}}
                config = temp / "memory.yml"
                config.write_text(yaml.safe_dump(offline), encoding="utf-8")
            backend = Mem2Backend(config, "offline-repo-test", fidelity=enhanced, smoke=True)
            if enhanced:
                self.assertEqual(backend.api._engine._retriever._fuser.explain()["strategy"],
                                 memory_config(profile)["fuser"]["default"]["target"])
            app = create_app(backend=backend, state_dir=temp / "state", api_key="test-only", concurrency=1)
            with TestClient(app, headers={"Authorization": "Bearer test-only"}) as client:
                def post(path, **kwargs):
                    posts.append((path, kwargs["json"]))
                    return client.post(path, **kwargs)

                transport = SimpleNamespace(post=post, close=lambda: None)
                with patch("evaluation.longmemeval.http_harness.httpx.Client", return_value=transport), \
                     patch.object(run, "openai_chat", side_effect=fake_chat), \
                     patch("evaluation.longmemeval.harness.no_source_extraction",
                           side_effect=AssertionError("Must use ordinary extraction")), \
                     patch.object(backend.api, "add", wraps=backend.api.add) as add:
                    code = run.main(["--data", str(dataset), "--config", str(config),
                                     "--http-memory-url", "http://testserver", "--run-id", "offline",
                                     "--no-oracle-sessions",
                                     "--output-dir", str(temp / "output"),
                                     "--judge-model", "offline", "--judge-base-url", "http://unused.invalid",
                                     "--judge-api-key", "unused"])
                    self.assertEqual(code, 0)
                    self.assertEqual(add.call_count, 2)  # 证据轮与干扰会话都写入。
                    memory_args, memory_kwargs = add.call_args_list[0]
                    self.assertEqual(memory_args[0], posts[0][1]["messages"][0]["content"])
                    self.assertIs(memory_kwargs["system_metadata"]["retain_source"], True)
                    self.assertIs(memory_kwargs["system_metadata"]["infer"], True)
                    self.assertIs(memory_kwargs["system_metadata"]["extraction_fidelity_mode"], enhanced)
                    first = client.post("/add", json=posts[0][1]).json()
                    second = client.post("/add", json=posts[0][1]).json()
                    self.assertEqual(first, second)
                    self.assertTrue(first["evaluation"]["unit_ids"])
                    self.assertEqual(add.call_count, 2)
                self.assertEqual([path for path, _ in posts], ["/add", "/add", "/search"])
                self.assertEqual(posts[1][1]["session_id"], "distractor")
                self.assertIn("bicycles", posts[1][1]["messages"][0]["content"])
                self.assertEqual(posts[-1][1]["top_k"], 200)
                self.assertNotIn("GOLD_ONLY_SENTINEL", json.dumps(posts))
                self.assertNotIn("has_answer", json.dumps(posts))
                self.assertNotIn(row["question"], json.dumps(posts[0]))
                self.assertEqual(client.post("/search", json={**posts[-1][1], "top_k": 201}).status_code, 422)
                self.assertEqual(client.post("/search", json=posts[-1][1],
                                             headers={"Authorization": "Bearer wrong"}).status_code, 401)
            result = json.loads((temp / "output/result.json").read_text(encoding="utf-8"))
            self.assertIs(result["evaluation_protocol"]["extraction_fidelity_mode"], enhanced)
            report = normalize_result(result)
            self.assertEqual(report["qa_cutoffs"], {"50": True, "10": False, "20": False})
            protocol = result["evaluation_protocol"]
            self.assertEqual(protocol["extraction_path"], "ordinary")
            self.assertEqual(protocol["context_mode"], "full_haystack")
            self.assertEqual(protocol["answer_cutoffs"], [50, 10, 20])
            self.assertEqual([call[0].audit_category for call in calls], ["answer", "judge"] * 3)
            for index, cutoff in enumerate(("50", "10", "20")):
                record = result["qa_records"][0]["cutoffs"][cutoff]
                self.assertTrue(record["context_contents"])
                self.assertTrue(all(value.startswith("2024-01-01") for value in record["context_message_dates"]))
                memories = [{"memory": content, "created_at": date, "message_at": message,
                             "event_at": event, "retrieval_rank": rank}
                            for content, date, message, event, rank in zip(
                                record["context_contents"], record["context_dates"],
                                record["context_message_dates"], record["context_event_dates"],
                                record["context_retrieval_ranks"])]
                question_date = row["question_date"]
                self.assertEqual(calls[index * 2][2], get_answer_generation_prompt(
                    row["question"], memories, question_date=question_date))
                self.assertEqual(calls[index * 2 + 1][2], get_judge_prompt(
                    row["question"], row["answer"], "mango", question_date=question_date))
                self.assertEqual(calls[index * 2][0].max_tokens, 16384)
                self.assertEqual(calls[index * 2 + 1][0].max_tokens, 4096)
                self.assertEqual(calls[index * 2][0].temperature, 0)
            self.assertIsNone(result["qa_records"][0]["cutoffs"]["20"]["judge_parsed"])
            observations = json.loads((temp / "output/artifacts/http_memory_observations.json").read_text())
            self.assertTrue(observations[0]["seed_unit_ids"])
            self.assertGreater(observations[0]["response"]["evaluation"]["storage_recall_wall_ms"], 0)

    def test_normal_competition_mode_rejects_evaluation_extensions(self):
        from test_api import FakeBackend, payload
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"AML_REPO_EVAL": "false"}):
            with TestClient(create_app(backend=FakeBackend(), state_dir=temporary, api_key="test"),
                            headers={"Authorization": "Bearer test"}) as client:
                self.assertEqual(client.post("/search", json={"query": "x", "user_id": "u", "top_k": 200}).status_code, 422)
                self.assertEqual(client.post("/add", json={**payload(), "evaluation_seed": {}}).status_code, 422)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from aml_adapter.app import create_app
from aml_adapter.lme_smoke import add_requests, evaluate_http, local_qa, model_transport_overrides, timestamp_ms
from test_api import FakeBackend, KEY


def sample():
    return {"question_id": "synthetic", "question": "favorite fruit?", "answer": "gold-hidden",
            "answer_session_ids": ["session-A"], "haystack_session_ids": ["session-A", "session-B"],
            "haystack_dates": ["2024/01/01 (Mon) 00:00", "2024/01/02 (Tue) 00:00"],
            "haystack_sessions": [[{"role": "user", "content": "mango", "has_answer": True}],
                                  [{"role": "assistant", "content": "bicycle"}]]}


class LongMemEvalSmokeTests(unittest.TestCase):
    def test_http_models_require_explicit_opt_in_and_only_override_http_component(self):
        env = {"AML_LLM_BASE_URL": "http://192.0.2.1/v1",
               "AML_EMBED_BASE_URL": "https://api.siliconflow.cn/v1"}
        with self.assertRaisesRegex(ValueError, "AllowHttpModels"):
            model_transport_overrides(env)
        self.assertEqual(model_transport_overrides(env, True), {"llm_ssl_verify": False})
        env["AML_LLM_BASE_URL"] = "https://example.test/v1"
        self.assertEqual(model_transport_overrides(env, True), {})
        env["LME_QA_BASE_URL"] = "http://localhost:8080/v1"
        with self.assertRaisesRegex(ValueError, "LME_QA_BASE_URL"):
            model_transport_overrides(env)
        self.assertEqual(model_transport_overrides(env, include_qa=False), {})
        self.assertEqual(model_transport_overrides(env, True), {})

    def test_invalid_model_url_error_does_not_echo_credentials(self):
        env = {"AML_LLM_BASE_URL": "ftp://user:private-key@example.test/v1"}
        with self.assertRaises(ValueError) as caught:
            model_transport_overrides(env, True)
        self.assertNotIn("private-key", str(caught.exception))

    def test_qwen_answer_and_judge_disable_thinking_without_leaking_gold(self):
        env = {"AML_LLM_MODEL": "qwen3.8-flash", "AML_LLM_BASE_URL": "https://example.test/v1",
               "AML_LLM_API_KEY": "test-only"}
        for model in ("qwen3.8-flash", "gpt-4o-mini"):
            with self.subTest(model=model), patch("openai.OpenAI") as factory:
                env["AML_LLM_MODEL"] = model
                create = factory.return_value.__enter__.return_value.chat.completions.create
                create.side_effect = [SimpleNamespace(choices=[SimpleNamespace(
                    message=SimpleNamespace(content=content))]) for content in
                    ("mango", '{"correct": true, "reason": "match"}')]
                result = local_qa(sample(), [{"content": "mango"}], env)
                self.assertTrue(result["verdict"]["correct"])
                self.assertNotIn("gold-hidden", str(create.call_args_list[0]))
                self.assertIn("gold-hidden", str(create.call_args_list[1]))
                for call in create.call_args_list:
                    if model.startswith("qwen"):
                        self.assertEqual(call.kwargs["extra_body"], {"enable_thinking": False})
                    else:
                        self.assertNotIn("extra_body", call.kwargs)

    def test_time_and_no_gold_in_add_payload(self):
        requests = list(add_requests(sample(), "one-user"))
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0]["messages"], [
            {"role": "user", "content": "mango", "timestamp": 1704067200000}])
        self.assertNotIn("gold-hidden", str(requests))
        self.assertNotIn("has_answer", str(requests))
        self.assertEqual(timestamp_ms("2024/01/02 (Tue) 00:00"), 1704153600000)

    def test_chunking_preserves_order_and_content(self):
        data = sample()
        data["haystack_sessions"][0] = [{"role": "user", "content": f"message {i}"} for i in range(21)]
        requests = list(add_requests(data, "u"))
        self.assertEqual([len(row["messages"]) for row in requests], [20, 1, 1])
        joined = [m["content"] for row in requests for m in row["messages"]]
        self.assertEqual(joined, [f"message {i}" for i in range(21)] + ["bicycle"])
        self.assertEqual(len({row["request_id"] for row in requests}), 3)

    def test_http_flow_retries_and_cross_session_scope(self):
        backend = FakeBackend()
        with tempfile.TemporaryDirectory() as temp:
            with TestClient(create_app(backend=backend, state_dir=temp, api_key=KEY),
                            headers={"Authorization": "Bearer " + KEY}) as client:
                report = evaluate_http(client, sample(), "lme-test")
        self.assertEqual(report["retrieved_count"], 2)
        self.assertTrue(report["user_isolation"])
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual([c[0].session_id for c in backend.calls], ["session-A", "session-B"])


if __name__ == "__main__":
    unittest.main()

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from openai import APIConnectionError

from aml_adapter.app import create_app
from aml_adapter.diagnostics import DiagnosticLog, endpoint_info, exception_info, wrap_model
from aml_adapter.lme_smoke import evaluate_http
from test_api import FakeBackend, KEY
from test_lme_smoke import sample


class DiagnosticsTests(unittest.TestCase):
    def test_extraction_validation_codes_exclude_sensitive_values(self):
        from jiuwen_memory.construction.extractor_impl.llm_extractor import (
            InvalidExtractionCandidateError,
        )

        error = InvalidExtractionCandidateError("secret-original-output", reason_counts={
            "missing_confidence": 2, "source_id_not_in_batch": 1,
            "secret-unrecognized-key": 3, "empty_content": "secret-value",
        })
        info = exception_info(error)
        self.assertEqual(info["chain"][0]["validation_reasons"], {
            "missing_confidence": 2, "source_id_not_in_batch": 1,
        })
        self.assertNotIn("secret", json.dumps(info))
        self.assertIn("validation_reasons", info["hint"])

    def test_kernel_assembly_instruments_the_shared_model_without_network(self):
        import yaml
        from aml_adapter.backend import Mem2Backend
        from jiuwen_memory.common.llm.base import LlmProducer
        from jiuwen_memory.common.llm.llm_impl.openai_llm import OpenAILLM
        from jiuwen_memory.config import Config

        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load((root / "config/local-smoke.yml").read_text(encoding="utf-8"))
        config["globals"].update(llm_model="qwen3.8-flash", llm_base_url="http://example.test/v1",
                                 llm_api_key="unused", embedder_model="BAAI/bge-m3",
                                 embedder_base_url="https://example.test/v1", embedder_api_key="unused")
        config["llm"] = {"default": {"target": "dashscope"}}
        config["embedder"] = {"default": {"target": "openai"}}
        with tempfile.TemporaryDirectory() as temp:
            path, log = Path(temp) / "config.yml", Path(temp) / "api.jsonl"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            # 仅用内存存储验证真实装配；不连接公司网关或数据库。
            with patch.dict(os.environ, {"AML_DIAGNOSTIC_LOG": str(log)}), \
                 patch.object(Mem2Backend, "_validate"), \
                 patch.object(OpenAILLM, "chat", return_value="mock-answer"):
                backend = Mem2Backend(path, "diagnostic-test")
                try:
                    model = LlmProducer.build_named("default", Config.from_yaml(str(path)).context())
                    self.assertEqual(model.chat([]), "mock-answer")
                    self.assertEqual(json.loads(log.read_text().splitlines()[-1])["stage"], "llm.chat")
                finally:
                    backend.close()

    def test_failure_preserves_http_status_and_root_cause_without_body_or_key(self):
        request = httpx.Request("POST", "http://gateway.test/v1/chat/completions?key=hidden-query")

        class FailingBackend(FakeBackend):
            def add(self, *args):
                try:
                    raise httpx.ConnectTimeout("secret-request-text secret-api-key", request=request)
                except httpx.ConnectTimeout as cause:
                    raise APIConnectionError(request=request) from cause

        with tempfile.TemporaryDirectory() as temp:
            api_log, runner_log = Path(temp) / "api.jsonl", Path(temp) / "runner.jsonl"
            with patch.dict(os.environ, {"AML_DIAGNOSTIC_LOG": str(api_log),
                                         "AML_LLM_API_KEY": "secret-api-key"}):
                with TestClient(create_app(backend=FailingBackend(), state_dir=Path(temp) / "state", api_key=KEY),
                                headers={"Authorization": "Bearer " + KEY}) as client:
                    # 新版 Starlette 的 TestClient 使用 httpx2；生产 runner 使用 httpx。
                    with self.assertRaises(Exception) as caught:
                        evaluate_http(client, sample(), "u", DiagnosticLog(runner_log))
                    self.assertEqual(type(caught.exception).__name__, "HTTPStatusError")
                    self.assertEqual(caught.exception.response.status_code, 503)
            text = api_log.read_text() + runner_log.read_text()
            for secret in ("secret-request-text", "secret-api-key", "hidden-query", "gold-hidden"):
                self.assertNotIn(secret, text)
            api_error = json.loads(api_log.read_text().splitlines()[-1])
            self.assertEqual([e["type"] for e in api_error["error"]["chain"]],
                             ["APIConnectionError", "ConnectTimeout"])
            self.assertTrue(api_error["error"]["chain"][-1]["frames"])
            self.assertEqual(json.loads(runner_log.read_text().splitlines()[-1])["status_code"], 503)

    def test_model_call_logs_timing_and_effective_endpoint_on_each_attempt(self):
        from jiuwen_memory.common.llm.llm_impl.dashscope_llm import DashScopeLLM
        from jiuwen_memory.common.embedder.embedder_impl.openai_embedder import OpenAIEmbedder

        llm = DashScopeLLM(model_name="qwen3.8-flash", base_url="http://example.test/v1", api_key="unused")
        embedder = OpenAIEmbedder(model_name="BAAI/bge-m3", base_url="https://example.test/v1", api_key="unused")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "calls.jsonl"
            with patch.object(llm, "chat", return_value="private-answer") as chat, \
                 patch.object(embedder, "embed", return_value=[[0.0]]) as embed:
                wrap_model(llm, "chat", "llm.chat", DiagnosticLog(path))
                wrap_model(embedder, "embed", "embedding.embed", DiagnosticLog(path))
                self.assertEqual(llm.chat(["private-prompt"]), "private-answer")
                self.assertEqual(embedder.embed(["private-input"]), [[0.0]])
                chat.assert_called_once()
                embed.assert_called_once()
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row["event"] for row in rows], ["start", "ok", "start", "ok"])
            self.assertEqual(rows[0]["connect_timeout_seconds"], 5)
            self.assertEqual(rows[2]["model"], "BAAI/bge-m3")
            self.assertNotIn("private-", path.read_text())

    def test_http_status_and_endpoint_sanitization(self):
        request = httpx.Request("POST", "https://user:password@example.test/v1?token=secret")
        error = httpx.HTTPStatusError("private-response", request=request, response=httpx.Response(401))
        detail = exception_info(error)
        self.assertEqual(detail["chain"][0]["status_code"], 401)
        self.assertNotIn("password", json.dumps(detail))
        self.assertNotIn("secret", json.dumps(detail))
        self.assertEqual(endpoint_info(request.url)["host"], "example.test")


if __name__ == "__main__":
    unittest.main()

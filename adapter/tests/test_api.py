import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from aml_adapter.app import create_app
from aml_adapter.backend import Mem2Backend
from aml_adapter.models import AddRequest, SearchRequest
from aml_adapter.service import MemoryService
from aml_adapter.state import Ledger, ProcessLock, ServiceError

KEY = "test-only-memory-key"
AUTH = {"Authorization": "Bearer " + KEY}


def payload(request_id="r1", user="u1", session="s1", content="mango"):
    return {"request_id": request_id, "user_id": user, "session_id": session,
            "messages": [{"role": "user", "content": content, "timestamp": 1704067200123}]}


class FakeBackend:
    fingerprint = "fake-v1"
    persistent = True

    def __init__(self):
        self.memories = {}
        self.calls = []
        self.fail = False
        self.extra = None

    def add(self, request, generation):
        self.calls.append((request, generation))
        target = self.memories.setdefault((request.user_id, generation), [])
        for message in request.messages:
            target.append({"id": f"{generation}-{len(target)}", "content": message.content,
                           "score": 1.0})
            if self.fail:
                self.fail = False
                raise RuntimeError("secret-request-body-must-not-be-returned")

    def search(self, request, generation):
        if self.extra is not None:
            return self.extra
        return self.memories.get((request.user_id, generation), [])

    def close(self):
        pass


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.backend = FakeBackend()
        self.client = TestClient(create_app(backend=self.backend, state_dir=self.temp.name,
                                           api_key=KEY))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def add(self, value=None):
        return self.client.post("/add", json=value or payload(), headers=AUTH)

    def search(self, user="u1", top_k=100, **extra):
        return self.client.post("/search", json={"query": "mango", "user_id": user,
                                                 "top_k": top_k, **extra}, headers=AUTH)

    def test_health_and_all_auth_schemes(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertEqual(self.client.post("/add", json=payload()).status_code, 401)
        for header in ({"Authorization": "Bearer " + KEY},
                       {"Authorization": "Token " + KEY}, {"X-Api-Key": KEY}):
            self.assertEqual(self.client.post("/add", json=payload(), headers=header).status_code, 200)
        self.assertEqual(len(self.backend.calls), 1)

    def test_exact_echo_retry_cross_session_and_isolation(self):
        response = self.add()
        self.assertEqual(response.json(), {"success": True, "request_id": "r1",
                                          "user_id": "u1", "session_id": "s1"})
        self.assertEqual(response.json(), self.add().json())
        self.add(payload("r2", session="s2", content="bicycle"))
        data = self.search(options=["A. mango", "B. pear"]).json()["data"]
        self.assertEqual([item["content"] for item in data], ["mango", "bicycle"])
        self.assertEqual(len(self.backend.calls), 2)
        self.assertEqual(self.search("other").json(), {"data": []})
        self.assertEqual(len(self.search(top_k=1).json()["data"]), 1)
        # 同 request_id 在不同 user 下不能命中其他用户的幂等缓存。
        self.assertEqual(self.add(payload(user="other", content="pear")).status_code, 200)
        self.assertEqual(self.search("other").json()["data"][0]["content"], "pear")

    def test_payload_conflict(self):
        self.add()
        result = self.add(payload(content="different"))
        self.assertEqual(result.status_code, 409)
        self.assertEqual(len(self.backend.calls), 1)

    def test_validation_and_no_error_input_leak(self):
        bad = payload(content="secret-validation-payload")
        bad["messages"][0]["role"] = "system"
        response = self.add(bad)
        self.assertEqual(response.status_code, 422)
        self.assertNotIn("secret-validation-payload", response.text)
        for value in (0, 101, True, "100"):
            self.assertEqual(self.search(top_k=value).status_code, 422)
        for value in (True, "1704067200000", 10**30):
            bad = payload()
            bad["messages"][0]["timestamp"] = value
            self.assertEqual(self.add(bad).status_code, 422)
        bad = payload()
        bad["messages"][0]["content"] = [{"type": "text", "text": "not text track"}]
        self.assertEqual(self.add(bad).status_code, 422)
        self.assertEqual(self.search(session_id="s1").status_code, 422)
        self.assertEqual(self.backend.calls, [])

    def test_partial_write_rebuilds_new_scope_without_duplicates(self):
        self.add()
        self.backend.fail = True
        second = payload("r2", content="bicycle")
        failed = self.add(second)
        self.assertEqual(failed.status_code, 503)
        self.assertNotIn("secret-request-body", failed.text)
        old_scope = self.backend.calls[-1][1]
        self.assertEqual(self.add(second).status_code, 200)
        new_scope = self.backend.calls[-1][1]
        self.assertNotEqual(old_scope, new_scope)
        contents = [item["content"] for item in self.search().json()["data"]]
        self.assertEqual(contents, ["mango", "bicycle"])
        count = len(self.backend.calls)
        self.assertEqual(self.add(second).status_code, 200)
        self.assertEqual(len(self.backend.calls), count)

    def test_restart_success_cache_and_failed_write_recovery(self):
        self.add()
        self.backend.fail = True
        second = payload("r2", content="bicycle")
        self.assertEqual(self.add(second).status_code, 503)
        self.client.__exit__(None, None, None)
        self.client = TestClient(create_app(backend=self.backend, state_dir=self.temp.name, api_key=KEY))
        self.client.__enter__()
        # 搜索同样先恢复，不泄露旧 scope 的半成品。
        self.assertEqual(len(self.search().json()["data"]), 2)
        count = len(self.backend.calls)
        self.assertEqual(self.add().status_code, 200)
        self.assertEqual(len(self.backend.calls), count)

    def test_invalid_backend_result_is_not_success(self):
        self.add()
        self.backend.extra = [{"id": "x", "content": ""}]
        self.assertEqual(self.search().status_code, 503)
        self.backend.extra = [{"id": "x", "content": "mango", "score": float("nan")}]
        self.assertNotIn("score", self.search().json()["data"][0])

    def test_preserves_ids_and_message_order(self):
        body = payload(request_id=" r:原始 ", user=" u ", session=" s ")
        body["messages"].append({"role": "assistant", "content": "  acknowledged  "})
        self.assertEqual(self.add(body).json()["request_id"], body["request_id"])
        received = self.backend.calls[0][0]
        self.assertEqual(received.messages[1].content, "  acknowledged  ")
        self.assertIsNone(received.messages[1].timestamp)

    def test_failed_recovery_stays_unpublished(self):
        self.backend.fail = True
        self.assertEqual(self.add().status_code, 503)
        self.backend.fail = True
        self.assertEqual(self.search().status_code, 503)
        self.assertEqual(len(self.search().json()["data"]), 1)


class DurabilityTests(unittest.TestCase):
    def test_payload_limit_rejects_instead_of_truncating(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict("os.environ", {"AML_MAX_BODY_BYTES": "128"}):
            backend = FakeBackend()
            with TestClient(create_app(backend=backend, state_dir=temp, api_key=KEY)) as client:
                response = client.post("/add", json=payload(content="x" * 500), headers=AUTH)
                self.assertEqual(response.status_code, 413)
                self.assertEqual(backend.calls, [])

    def test_concurrent_same_user_and_capacity(self):
        with tempfile.TemporaryDirectory() as temp:
            ledger = Ledger(Path(temp) / "state.db", "test")
            service = MemoryService(FakeBackend(), ledger, 2)
            with service.admission("u1", "add"):
                with self.assertRaises(ServiceError) as ctx:
                    service.add(AddRequest.model_validate(payload()))
                self.assertEqual(ctx.exception.status, 409)
                with self.assertRaises(ServiceError) as ctx:
                    service.search(SearchRequest(query="x", user_id="u1", top_k=1))
                self.assertEqual(ctx.exception.status, 425)
                with service.admission("another-user", "add"):
                    with self.assertRaises(ServiceError) as ctx:
                        service.search(SearchRequest(query="x", user_id="u3", top_k=1))
                    self.assertEqual(ctx.exception.status, 429)
            ledger.close()

    def test_process_lock_and_configuration_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            lock = ProcessLock(Path(temp) / "lock")
            try:
                with self.assertRaises(RuntimeError):
                    ProcessLock(Path(temp) / "lock")
            finally:
                lock.close()
            Ledger(Path(temp) / "state.db", "v1").close()
            with self.assertRaises(RuntimeError):
                Ledger(Path(temp) / "state.db", "v2")

    def test_dirty_intent_survives_crash_before_response(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "state.db"
            ledger = Ledger(path, "test")
            ledger.prepare(AddRequest.model_validate(payload()), "any-digest", "old-generation")
            ledger.close()  # 模拟在后端变更与完成记录之间进程退出。
            reopened = Ledger(path, "test")
            service = MemoryService(FakeBackend(), reopened)
            result = service.search(SearchRequest(query="mango", user_id="u1", top_k=100))
            self.assertEqual(result["data"][0]["content"], "mango")
            self.assertNotEqual(reopened.user("u1")["generation"], "old-generation")
            reopened.close()


class RealKernelTests(unittest.TestCase):
    def test_production_config_requires_visible_indexes(self):
        from jiuwen_memory.config import Config
        config = Path(__file__).resolve().parents[1] / "config" / "memory.yml"
        values = {key: "test-only" for key in ("AML_LLM_BASE_URL", "AML_LLM_API_KEY",
                  "AML_EMBED_BASE_URL", "AML_EMBED_API_KEY")}
        ctx = Config.from_yaml_str(config.read_text(encoding="utf-8"), **values).context()
        Mem2Backend._validate(ctx, False)
        ctx.namespaces["fulltext_store"]["default"].params["refresh"] = "false"
        with self.assertRaises(ValueError):
            Mem2Backend._validate(ctx, False)

    def test_http_to_real_mem2_and_restart_replay(self):
        config = Path(__file__).resolve().parents[1] / "config" / "local-smoke.yml"
        with tempfile.TemporaryDirectory() as temp:
            backend = Mem2Backend(config, "test-kernel", smoke=True)
            with TestClient(create_app(backend=backend, state_dir=temp, api_key=KEY)) as client:
                for i, text in enumerate(("mango favorite fruit", "purple bicycle")):
                    response = client.post("/add", json=payload(f"r{i}", session=f"s{i}", content=text), headers=AUTH)
                    self.assertEqual(response.status_code, 200, response.text)
                response = client.post("/search", json={"query": "mango bicycle", "user_id": "u1", "top_k": 100}, headers=AUTH)
                self.assertEqual(response.status_code, 200, response.text)
                joined = " ".join(item["content"] for item in response.json()["data"])
                self.assertIn("mango", joined)
                self.assertIn("bicycle", joined)
            # 本地 smoke 用进程内索引，重启要恢复，不可只重放成功 HTTP 缓存。
            backend = Mem2Backend(config, "test-kernel", smoke=True)
            with TestClient(create_app(backend=backend, state_dir=temp, api_key=KEY)) as client:
                response = client.post("/search", json={"query": "mango", "user_id": "u1", "top_k": 100}, headers=AUTH)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertTrue(response.json()["data"])

    def test_real_adapter_time_scope_and_l2_mapping(self):
        from types import SimpleNamespace
        backend = Mem2Backend.__new__(Mem2Backend)
        backend.namespace, backend.fidelity = "test", True
        calls = []
        backend.api = SimpleNamespace(add=lambda *args, **kwargs: calls.append((args, kwargs)))
        request = AddRequest.model_validate(payload())
        backend.add(request, "gen")
        args, kwargs = calls[0]
        self.assertEqual(args[1].session, "")
        self.assertEqual(kwargs["occurred_at"].isoformat(), "2024-01-01T00:00:00.123000+00:00")
        self.assertEqual(kwargs["system_metadata"]["session_id"], "s1")
        self.assertEqual(kwargs["system_metadata"]["observation_date"], "2024-01-01")
        self.assertIs(kwargs["system_metadata"]["extraction_fidelity_mode"], True)
        self.assertNotEqual(backend.scope("u1", "gen"), backend.scope("u2", "gen"))


if __name__ == "__main__":
    unittest.main()

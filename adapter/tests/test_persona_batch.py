import argparse
import asyncio
import copy
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import httpx

from aml_adapter import persona_batch as runner
from aml_adapter.diagnostics import DiagnosticLog
from aml_adapter.models import AddRequest, SearchRequest
from aml_adapter.persona_data import add_chunks, messages, official, sample_rows, save, validate_selection


class PersonaTests(unittest.TestCase):
    def test_answer_and_judge_send_thinking_disabled_on_wire(self):
        env = {"AML_LLM_BASE_URL": "https://model.test/v1", "AML_LLM_API_KEY": "test-key",
               "AML_LLM_MODEL": "Qwen3.8-Flash"}
        seen = []

        def transport(request):
            payload = json.loads(request.content)
            seen.append(payload)
            self.assertIs(payload["enable_thinking"], False)
            self.assertNotIn("extra_body", payload)
            self.assertEqual(payload["temperature"], 0)
            self.assertEqual(str(request.url), "https://model.test/v1/chat/completions")
            return httpx.Response(200, json={"choices": [{"message": {"content": " response "}}]})

        client_type = httpx.AsyncClient
        def make_client(**kwargs):
            return client_type(transport=httpx.MockTransport(transport), **kwargs)

        diagnostic = DiagnosticLog(self.root / "diagnostics.jsonl")
        with patch.object(runner.httpx, "AsyncClient", side_effect=make_client):
            for role in ("answer", "judge"):
                prompt = [{"role": "user", "content": role}]
                result = asyncio.run(runner.complete(self.pipeline, env, role, prompt, diagnostic))
                self.assertEqual(result, "response")
                self.assertEqual(seen[-1]["messages"], prompt)
        events = [json.loads(line) for line in diagnostic.path.read_text().splitlines()]
        self.assertEqual(len(events), 4)
        self.assertTrue(all(event["enable_thinking"] is False for event in events))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.pipeline = official()
        self.row = {"id": "q1", "persona_id": "p1", "pref_type": "neutral", "history_file": "history.json",
                    "user_query": "{'role': 'user', 'content': 'Choose a meal'}", "preference": "SECRET_GOLD_PREF",
                    "correct_answer": "vegetables", "option_mapping": {"A": "meat", "B": "vegetables"},
                    "correct_letter": "B"}
        save(self.root / "history.json", {"chat_history": [
            {"role": "system", "content": "original profile"},
            {"role": "user", "content": "I enjoyed that soup."},
            {"role": "assistant", "content": "Glad to hear it."}]})
        self.args = argparse.Namespace(skip_failed=False, mode="both", top_k=100)

    def test_deterministic_sampling_balances_types(self):
        rows = [{"persona_id": str(p), "pref_type": str(i % 3), "preference": "p", "correct_answer": "a"}
                for p in range(10) for i in range(8)]
        first = sample_rows(rows)
        self.assertEqual(first, sample_rows(rows))
        self.assertEqual(len(first), 25)
        self.assertEqual(len({r["persona_id"] for r in first}), 5)
        for person in {r["persona_id"] for r in first}:
            group = [r for r in first if r["persona_id"] == person]
            self.assertEqual(len(group), 5)
            self.assertEqual(len({r["pref_type"] for r in group}), 3)

    def test_messages_and_chunks_preserve_system_and_long_content(self):
        history = messages(self.root / "history.json")
        long_text = " leading " + " word\n" * 5000
        history += [{"role": "user", "content": long_text}]
        requests = list(add_chunks(history, "test"))
        for request in requests:
            AddRequest.model_validate(request)
            self.assertLessEqual(len(request["messages"]), 20)
            self.assertLessEqual(sum(len(m["content"].split()) for m in request["messages"]), 2000)
            self.assertNotIn("SECRET_GOLD", json.dumps(request))
        flattened = [m for r in requests for m in r["messages"]]
        self.assertIn("Original history system", flattened[0]["content"])
        self.assertEqual("".join(m["content"] for m in flattened[3:]), long_text)

    def test_official_prompts_and_mcq_mapping(self):
        retrieved = [{"id": "m1", "content": "soup memory"}]
        generative = runner.answer_messages(self.pipeline, self.row, retrieved, "generative")
        self.assertNotIn("SECRET_GOLD_PREF", json.dumps(generative))
        self.assertNotIn("vegetables", json.dumps(generative))
        self.assertEqual(generative[-1]["content"], "Choose a meal" + self.pipeline.RECALL_SUFFIX)
        mcq = runner.answer_messages(self.pipeline, self.row, retrieved, "mcq")
        self.assertEqual(mcq[-1]["content"], self.pipeline.MCQ_PROMPT_TEMPLATE.format(options="A. meat\nB. vegetables"))

    def run_case(self, rows=None, fail_first_answer=False, fail_add=False):
        rows = rows or [self.row]
        state = {"questions": {r["id"]: {"status": "pending", "persona_id": r["persona_id"],
                                        "pref_type": r["pref_type"]} for r in rows}, "histories": {}}
        requests, calls = [], []
        def transport(request):
            payload = json.loads(request.content)
            requests.append((request.url.path, payload))
            if request.url.path == "/add":
                AddRequest.model_validate(payload)
                if fail_add and payload["messages"][0]["content"].startswith("[Original"):
                    return httpx.Response(503, json={"error": "failed"})
                return httpx.Response(200, json={"success": True})
            SearchRequest.model_validate(payload)
            return httpx.Response(200, json={"data": [{"id": "m1", "content": "soup memory"}]})
        async def complete(pipeline, env, role, prompt, diagnostic):
            calls.append((role, prompt))
            if fail_first_answer and len(calls) == 1:
                raise TimeoutError()
            return "\\boxed{0.75}" if role == "judge" else "Final Answer: B"
        with httpx.Client(base_url="http://test", transport=httpx.MockTransport(transport)) as client:
            with patch.object(runner, "DATA", self.root), patch.object(runner, "complete", complete):
                runner.run_questions(state, rows, client, self.pipeline, self.args, {}, self.root,
                                     DiagnosticLog(), lambda: save(self.root / "state.json", state))
                # Re-running the same state must skip completed work and reuse durable artifacts.
                if not fail_add and not fail_first_answer:
                    runner.run_questions(state, rows, client, self.pipeline, self.args, {}, self.root,
                                         DiagnosticLog(), lambda: None)
        return state, requests, calls

    def test_http_flow_official_scores_no_gold_leak_and_resume(self):
        second = dict(self.row, id="q2")
        state, requests, calls = self.run_case([self.row, second])
        self.assertEqual(sum(path == "/add" for path, _ in requests), 1)
        self.assertEqual(sum(path == "/search" for path, _ in requests), 2)
        self.assertEqual(len(calls), 6)
        for record in state["questions"].values():
            self.assertEqual(record["status"], "completed")
            self.assertTrue(record["mcq_correct"])
            self.assertEqual(record["narrow_score"], .75)
        for _, payload in requests:
            self.assertNotIn("SECRET_GOLD_PREF", json.dumps(payload))
        self.assertIn("SECRET_GOLD_PREF", str([p for role, p in calls if role == "judge"]))

    def test_question_failure_continues_same_history(self):
        state, requests, _ = self.run_case([self.row, dict(self.row, id="q2")], fail_first_answer=True)
        self.assertEqual(state["questions"]["q1"]["status"], "failed")
        self.assertEqual(state["questions"]["q2"]["status"], "completed")
        self.assertEqual(sum(path == "/add" for path, _ in requests), 1)
        self.assertFalse(runner.should_run(state["questions"]["q1"], True))
        self.assertTrue(runner.should_run(state["questions"]["q1"], False))

    def test_failed_add_marks_dependents_without_using_partial_memory(self):
        state, requests, calls = self.run_case([self.row, dict(self.row, id="q2")], fail_add=True)
        self.assertTrue(all(r["status"] == "failed" for r in state["questions"].values()))
        self.assertEqual(len(requests), 1)
        self.assertEqual(calls, [])

    def test_negative_judge_uses_official_template(self):
        self.row["preference"] = "Do not personalize using soup"
        _, _, calls = self.run_case()
        prompt = next(p[0]["content"] for role, p in calls if role == "judge")
        self.assertEqual(prompt, self.pipeline.JUDGE_PROMPT_NARROW_NEGATIVE.format(
            user_query="Choose a meal", preference=self.row["preference"], model_response="Final Answer: B"))

    def test_failed_judge_resume_reuses_search_and_both_answers(self):
        requests, model_roles = [], []
        def transport(request):
            requests.append(request.url.path)
            return httpx.Response(200, json={"data": [{"id": "m", "content": "memory"}]})
        async def fail_judge(pipeline, env, role, prompt, diagnostic):
            model_roles.append(role)
            if role == "judge":
                raise TimeoutError()
            return "Final Answer: B"
        async def succeed_judge(pipeline, env, role, prompt, diagnostic):
            model_roles.append(role)
            self.assertEqual(role, "judge")
            return "\\boxed{1.0}"
        record = {"status": "running"}
        with httpx.Client(base_url="http://test", transport=httpx.MockTransport(transport)) as client:
            with patch.object(runner, "complete", fail_judge), self.assertRaises(TimeoutError):
                runner.evaluate_question(self.row, record, client, "u", self.pipeline, self.args, {},
                                         self.root, DiagnosticLog(), lambda: None)
            with patch.object(runner, "complete", succeed_judge):
                runner.evaluate_question(self.row, record, client, "u", self.pipeline, self.args, {},
                                         self.root, DiagnosticLog(), lambda: None)
        self.assertEqual(requests, ["/search"])
        self.assertEqual(model_roles, ["answer", "answer", "judge", "judge"])
        self.assertTrue(record["mcq_correct"])
        self.assertEqual(record["narrow_score"], 1.0)

    def test_interrupted_ingest_resumes_from_next_checkpoint(self):
        history = [{"role": "user", "content": f"message {i}"} for i in range(21)]
        save(self.root / "history.json", history)
        state = {"questions": {"q1": {"status": "interrupted"}},
                 "histories": {"p1:history.json": {"status": "pending", "user_id": "stable", "next_add": 1}}}
        added = []
        def transport(request):
            payload = json.loads(request.content)
            added.append(payload)
            return httpx.Response(200, json={"success": True})
        with httpx.Client(base_url="http://test", transport=httpx.MockTransport(transport)) as client:
            with patch.object(runner, "DATA", self.root), patch.object(runner, "evaluate_question"):
                runner.run_questions(state, [self.row], client, self.pipeline, self.args, {}, self.root,
                                     DiagnosticLog(), lambda: None)
        self.assertEqual(len(added), 1)
        self.assertEqual(added[0]["request_id"], "stable:1")
        self.assertEqual(added[0]["messages"], history[20:])
        self.assertEqual(state["questions"]["q1"]["status"], "completed")

    def test_corrupted_history_rejected(self):
        from aml_adapter.persona_data import digest
        path = "data/chat_history_32k/history.json"
        save(self.root / path, [{"role": "user", "content": "one"}])
        save(self.root / "selection.json", {"files": {path: digest(self.root / path)}})
        save(self.root / path, [{"role": "user", "content": "changed"}])
        with patch("aml_adapter.persona_data.DATA", self.root), self.assertRaises(ValueError):
            validate_selection(self.root / "selection.json")

    def test_statistics_do_not_count_failures_as_wrong_answers(self):
        values = [{"status": "completed", "mcq_correct": True, "narrow_score": .5},
                  {"status": "completed", "mcq_correct": False, "narrow_score": 1},
                  {"status": "failed", "mcq_correct": True}]
        self.assertEqual(runner.stats(values), {"total": 3, "completed": 2, "failed": 1,
                                               "mcq_accuracy": .5, "narrow_mean": .75})

    def test_resume_upgrade_is_exact_audited_and_preserves_checkpoints(self):
        original = {"profile": "ordinary", "fingerprint": "old", "histories": {
            "h1": {"status": "completed", "next_add": 15},
            "h2": {"status": "failed", "next_add": 12}}, "questions": {
                "q1": {"status": "completed", "mcq_correct": False, "narrow_score": .4},
                "q2": {"status": "failed", "mcq_correct": True}}}
        upgrades = [{"memory_profile": "ordinary", "from": "old", "to": "new", "reason": "JSON repair"}]
        state = copy.deepcopy(original)
        runner.validate_resume(state, "new", "ordinary", upgrades)
        self.assertEqual(state["questions"], original["questions"])
        self.assertEqual(state["histories"], original["histories"])
        self.assertEqual(state["fingerprint"], "new")
        self.assertEqual(state["runner_upgrades"][0]["preserved_completed_ids"], ["q1"])
        runner.validate_resume(state, "new", "ordinary", upgrades)
        self.assertEqual(len(state["runner_upgrades"]), 1)
        for signature, profile in [("other", "ordinary"), ("new", "schema-entity")]:
            with self.assertRaises(ValueError):
                runner.validate_resume(copy.deepcopy(original), signature, profile, upgrades)

    def test_resume_after_generative_timeout_reuses_mcq_and_search(self):
        artifact = self.root / "questions/q1"
        save(artifact / "search.json", {"data": [{"id": "m1", "content": "soup memory"}]})
        save(artifact / "mcq-answer.json", {"answer": "Final Answer: B"})
        roles = []
        async def complete(pipeline, env, role, prompt, diagnostic):
            roles.append(role)
            return "\\boxed{0.75}" if role == "judge" else "Soup fits."
        def no_request(request):
            raise AssertionError("cached Search must not be fetched again")
        record = {"status": "running"}
        with httpx.Client(base_url="http://test", transport=httpx.MockTransport(no_request)) as client:
            with patch.object(runner, "complete", complete):
                runner.evaluate_question(self.row, record, client, "u", self.pipeline, self.args, {},
                                         self.root, DiagnosticLog(), lambda: None)
        self.assertEqual(roles, ["answer", "judge"])
        self.assertTrue(record["mcq_correct"])
        self.assertEqual(record["narrow_score"], .75)

    def test_memory_http_timeout_is_separate_and_configurable(self):
        self.assertEqual(runner.memory_http_timeout({}).read, 14400)
        self.assertEqual(runner.memory_http_timeout({"PM_MEMORY_HTTP_TIMEOUT_SECONDS": "18000"}).read, 18000)
        self.assertEqual(runner.memory_http_timeout({}).connect, 5)
        with self.assertRaises(ValueError):
            runner.memory_http_timeout({"PM_MEMORY_HTTP_TIMEOUT_SECONDS": "0"})

    def test_interrupted_wait_does_not_wait_for_http_response(self):
        release = threading.Event()
        done = threading.Event()
        def blocking_post(*args, **kwargs):
            release.wait(5)
            done.set()
        try:
            with patch.object(runner.queue.Queue, "get", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    runner.interruptible_post(Mock(post=blocking_post), "/add", {})
        finally:
            release.set()
            self.assertTrue(done.wait(2))

    def test_history_timeout_restarts_worker_before_next_user(self):
        second = dict(self.row, id="q2", persona_id="p2")
        state = {"histories": {}, "questions": {q: {"status": "pending"} for q in ["q1", "q2"]}}
        events = []
        def transport(request):
            events.append("add")
            if events == ["add"]:
                raise httpx.ReadTimeout("local read timeout", request=request)
            self.assertEqual(events, ["add", "restart", "add"])
            return httpx.Response(200, json={"success": True})
        with httpx.Client(base_url="http://test", transport=httpx.MockTransport(transport)) as client:
            client.restart_owned_server = lambda: events.append("restart")
            with patch.object(runner, "DATA", self.root), patch.object(runner, "evaluate_question"):
                runner.run_questions(state, [self.row, second], client, self.pipeline, self.args, {},
                                     self.root, DiagnosticLog(), lambda: None)
        self.assertEqual(state["questions"]["q1"]["status"], "failed")
        self.assertEqual(state["questions"]["q2"]["status"], "completed")

    def test_failed_worker_restart_does_not_dispatch_other_users(self):
        second = dict(self.row, id="q2", persona_id="p2")
        state = {"histories": {}, "questions": {q: {"status": "pending"} for q in ["q1", "q2"]}}
        def transport(request):
            raise httpx.ReadTimeout("local read timeout", request=request)
        with httpx.Client(base_url="http://test", transport=httpx.MockTransport(transport)) as client:
            client.restart_owned_server = Mock(side_effect=RuntimeError("restart failed"))
            with patch.object(runner, "DATA", self.root), self.assertRaisesRegex(RuntimeError, "restart failed"):
                runner.run_questions(state, [self.row, second], client, self.pipeline, self.args, {},
                                     self.root, DiagnosticLog(), lambda: None)
        self.assertEqual(state["questions"]["q1"]["status"], "failed")
        self.assertEqual(state["questions"]["q2"]["status"], "pending")

    def test_owned_worker_stop_waits_or_kills(self):
        process = Mock()
        process.poll.return_value = None
        process.wait.side_effect = [runner.subprocess.TimeoutExpired("worker", 10), None]
        runner.stop_owned_process(process)
        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        self.assertEqual(process.wait.call_count, 2)

    def test_memory_server_restart_and_interrupt_clean_up_owned_workers(self):
        first, second = Mock(), Mock()
        first.poll.return_value = second.poll.return_value = None
        probe = MagicMock()
        probe.__enter__.return_value = probe
        probe.get.return_value.status_code = 200
        client = Mock()
        with patch.object(runner, "memory_config", return_value={"globals": {}}), \
                patch.object(runner, "model_transport_overrides", return_value={}), \
                patch.object(runner.subprocess, "Popen", side_effect=[first, second]), \
                patch.object(runner.httpx, "Client", side_effect=[probe, client, probe]):
            with self.assertRaises(KeyboardInterrupt):
                with runner.memory_server(self.root, "ordinary", "test", {}, True) as active:
                    self.assertIs(active, client)
                    active.restart_owned_server()
                    first.terminate.assert_called_once()
                    first.wait.assert_called_once()
                    raise KeyboardInterrupt()
        second.terminate.assert_called_once()
        second.wait.assert_called_once()
        client.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()

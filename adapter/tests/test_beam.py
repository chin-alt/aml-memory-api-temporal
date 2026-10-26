import tempfile
import unittest
from pathlib import Path

from aml_adapter import beam_batch as runner
from aml_adapter.beam_data import (QUESTION_TYPES, _select_conversations, add_payloads,
                                   append_conversations, flatten_chat, official, prepare,
                                   probing_questions)
from aml_adapter.models import AddRequest


def source_row(ident="1", words=20):
    questions = {question_type: [
        {"question": f"{question_type} question A", "rubric": [f"{question_type} fact A"]},
        {"question": f"{question_type} question B", "rubric": [f"{question_type} fact B"]},
    ] for question_type in QUESTION_TYPES}
    content = "fact " * words
    return {"conversation_id": ident, "conversation_seed": {"category": "test"},
            "chat": [[{"role": "user", "content": content, "time_anchor": "March-15-2024"},
                      {"role": "assistant", "content": "answer", "time_anchor": "March-15-2024"}]],
            "probing_questions": repr(questions)}


class BeamTests(unittest.TestCase):
    def test_logged_client_leaves_429_for_official_retry_loop(self):
        class Response:
            status_code = 429
            is_success = False

        class Client:
            async def post(self, url, **kwargs):
                return Response()

        with tempfile.TemporaryDirectory() as directory:
            wrapped = runner.LoggedChatClient(
                Client(), "answer", Path(directory), runner.DiagnosticLog())
            call = wrapped.post(
                "https://models.example/v1/chat/completions", json={"messages": []})
            with self.assertRaises(StopIteration) as completed:
                call.send(None)
            response = completed.exception.value
        self.assertEqual(response.status_code, 429)

    def test_beam_storage_is_isolated_by_profile(self):
        ordinary = {}
        runner.isolate_beam_storage(ordinary, "ordinary")
        self.assertEqual(ordinary["AML_ES_INDEX"], "aml_beam_ordinary_text_v1")
        self.assertEqual(ordinary["AML_MILVUS_COLLECTION"], "aml_beam_ordinary_vector_v1")
        self.assertEqual(ordinary["AML_MILVUS_URI"], "http://127.0.0.1:21530")
        self.assertEqual(ordinary["LME_MILVUS_HEALTH_URL"], "http://127.0.0.1:9191/healthz")
        schema = {"BEAM_ES_INDEX": "custom_text", "BEAM_MILVUS_COLLECTION": "custom_vector"}
        runner.isolate_beam_storage(schema, "schema-entity")
        self.assertEqual(schema["AML_ES_INDEX"], "custom_text")
        self.assertEqual(schema["AML_MILVUS_COLLECTION"], "custom_vector")

    def test_question_types_and_length_sampling(self):
        row = source_row()
        self.assertEqual(set(probing_questions(row)), set(QUESTION_TYPES))
        rows = [source_row(str(index), index * 10) for index in range(1, 10)]
        import random
        smoke = _select_conversations(rows, 2, random.Random(42), "smoke")
        self.assertEqual([item["conversation_id"] for item in smoke], ["1", "2"])
        stratified = _select_conversations(rows, 3, random.Random(42), "stratified")
        lengths = [sum(len(turn["content"].split()) for turn in flatten_chat(item)) for item in stratified]
        self.assertLess(lengths[0], lengths[1])
        self.assertLess(lengths[1], lengths[2])

    def test_selection_extension_requires_exact_subset(self):
        base = {"dataset": "d", "revision": "r", "source_sha256": "s", "size": "100K",
                "seed": 42, "per_type": 2, "sampling_mode": "smoke",
                "input_contract": "i", "official_competition_bundle": False,
                "conversations": [{"id": "4", "words": 10}],
                "questions": [{"id": "q4", "question": "same"}]}
        expanded = {**base,
                    "conversations": base["conversations"] + [{"id": "5", "words": 20}],
                    "questions": base["questions"] + [{"id": "q5", "question": "new"}]}
        self.assertTrue(runner.selection_is_extension(base, expanded))
        changed = {**expanded, "questions": [{"id": "q4", "question": "changed"},
                                               {"id": "q5", "question": "new"}]}
        self.assertFalse(runner.selection_is_extension(base, changed))

    def test_explicit_diverse_conversation_append_is_frozen_extension(self):
        base, _ = prepare(5, 2, 42, "100K", "smoke")
        expanded, path = append_conversations(base, ["8"])
        self.assertTrue(path.is_file())
        self.assertTrue(runner.selection_is_extension(base, expanded))
        self.assertEqual([row["id"] for row in expanded["conversations"]][-1], "8")
        self.assertEqual(expanded["conversations"][-1]["topic"]["category"],
                         "Writing Assistant & Learning")
        added = [row for row in expanded["questions"] if row["conversation_id"] == "8"]
        self.assertEqual(len(added), 20)
        self.assertEqual({row["question_type"] for row in added}, set(QUESTION_TYPES))

    def test_add_payloads_are_lossless_bounded_and_keep_time(self):
        row = source_row(words=4500)
        payloads = list(add_payloads(row, "scope"))
        self.assertEqual(len(payloads), 3)
        for payload in payloads:
            AddRequest.model_validate(payload)
            self.assertLessEqual(sum(len(message["content"].split()) for message in payload["messages"]), 2000)
            self.assertEqual(payload["session_id"], "beam-batch-000")
            self.assertTrue(all("timestamp" in message for message in payload["messages"]))
        source = "".join(turn["content"] for turn in row["chat"][0])
        rebuilt = "".join(message["content"] for payload in payloads for message in payload["messages"])
        self.assertEqual(rebuilt, source)

    def test_official_prompt_judge_and_no_thinking(self):
        pipeline = official()
        prompt = runner.answer_prompt(pipeline, "QUESTION", [{"content": "MEMORY"}])
        self.assertIn("QUESTION", prompt)
        self.assertIn("MEMORY", prompt)
        self.assertNotIn("SECRET_RUBRIC", prompt)
        payload = runner.model_request_payload({"temperature": 0, "enable_thinking": True,
                                                "response_format": {"type": "json_object"}})
        self.assertEqual(payload["temperature"], 0)
        self.assertIs(payload["enable_thinking"], False)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        scores = pipeline.parse_rubric_scores(
            '{"scores":[{"index":0,"score":1,"reason":"supported"}]}', 1)
        self.assertEqual(scores[0]["score"], 1)


if __name__ == "__main__":
    unittest.main()

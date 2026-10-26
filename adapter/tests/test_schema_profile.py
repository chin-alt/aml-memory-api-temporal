import json
import os
import re
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml
from fastapi.testclient import TestClient

from aml_adapter.app import create_app
from aml_adapter.backend import Mem2Backend
from aml_adapter.lme_profiles import memory_config
from aml_adapter.schema_eval import SchemaEvaluation
from jiuwen_memory.common.llm.llm_impl.echo_llm import EchoLLM
from jiuwen_memory.common.type_def.entity import EntityBatchResult, EntityOpType
from jiuwen_memory.storage.entity_impl.elasticsearch_entity_store import ElasticsearchEntityStore


class OfflineEntityStore(ElasticsearchEntityStore):
    fail_lookup = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.records = {}

    def ensure_index(self):
        self._index_ready = True

    def health(self):
        return None

    def find_by_entity_text_hash(self, space_id, entity_text_hashes, *, filters, limit=500):
        if self.fail_lookup:
            raise RuntimeError("offline lookup failure")
        return [r for r in self.records.values() if r.space_id == space_id
                and r.filters == filters and r.entity_text_hash in entity_text_hashes][:limit]

    def find_by_linked_memory_id(self, space_id, memory_id, *, filters):
        return [r for r in self.records.values() if r.space_id == space_id
                and r.filters == filters and memory_id in r.linked_memory_ids]

    def execute_operations(self, space_id, operations):
        ids = []
        for op in operations:
            rid = op.record.id if op.record else op.record_id
            if op.type in (EntityOpType.INSERT, EntityOpType.UNLINK_UPDATE):
                self.records[rid] = op.record
            elif op.type == EntityOpType.LINK:
                record = self.records[rid]
                self.records[rid] = replace(record, linked_memory_ids=tuple(
                    dict.fromkeys((*record.linked_memory_ids, *op.link_memory_ids))))
            elif op.type == EntityOpType.DELETE:
                self.records.pop(rid, None)
            ids.append(rid)
        return EntityBatchResult(successful_ids=ids, failed_ids=[])


class SchemaProfileTests(unittest.TestCase):
    def test_schema_validation_miss_keeps_source_first_write(self):
        events = []
        evaluation = SchemaEvaluation.__new__(SchemaEvaluation)
        evaluation.evolver = SimpleNamespace(last_schema_error="")
        evaluation.log = SimpleNamespace(emit=lambda stage, event, **fields:
                                         events.append((stage, event, fields)))
        with evaluation.writing():
            evaluation.evolver.last_schema_error = "invalid schema candidate"
        self.assertTrue(evaluation.last_write_degraded)
        self.assertTrue(any(fields.get("source_only_fallback")
                            for _, _, fields in events))

    def test_profile_keeps_baseline_settings_and_enables_complete_chain(self):
        ordinary, schema = memory_config("ordinary"), memory_config("schema-entity")
        self.assertFalse(ordinary["globals"]["schema_enabled"])
        self.assertTrue(schema["globals"]["schema_enabled"])
        self.assertTrue(schema["globals"]["entity_enabled"])
        self.assertEqual(schema["extractor"]["default"]["target"], "entity_schema")
        self.assertEqual(schema["evolver"]["default"]["target"], "schema_orchestrating")
        for key in ("query_parser", "fuser", "retriever", "llm", "embedder"):
            self.assertEqual(ordinary[key], schema[key])
        self.assertTrue(Path(schema["extractor"]["default"]["params"]["schema_path"]).is_file())

    def test_http_schema_extraction_entity_expansion_and_source_first_degradation(self):
        config = memory_config("schema-entity")
        # Keep production's vector + keyword combination: both recallers use L2.
        self.assertTrue(config["globals"]["vector_enabled"])
        config["llm"] = {"default": {"target": "echo"}}
        config["embedder"] = {"default": {"target": "hashing"}}
        config["tokenizer"] = {"default": {"target": "whitespace"}}
        config["kv_store"] = {"default": {"target": "memory"}}
        config["fulltext_store"] = {"default": {"target": "memory"}}
        config["vector_store"] = {"default": {"target": "memory"}}
        config["dedup"] = {"default": {"target": "keyword"}}
        config["entity_store"]["default"]["params"]["hosts"] = "http://unused.invalid"
        broken = False

        def chat(messages, **kwargs):
            prompt = messages[-1].content
            if "unit_id=" not in prompt:
                return json.dumps({"selected_entities": [{"entity_type": "person",
                                                         "relevant_properties": ["default_property"]}]})
            if broken:
                return "not valid JSON"
            match = re.search(r"\[unit_id=([^\]]+)\].*?\[speaker=[^\]]+\]\n([^\n]+)", prompt, re.S)
            if match is None:
                match = re.search(r"\[message_index=0, unit_id=([^\]]+)\]\n([^\n]+)", prompt)
            self.assertIsNotNone(match)
            return json.dumps({"entities": [{"name": "Alice", "entity_type": "person",
                "properties": [{"property_name": "default_property", "value": match[2],
                                "time": "", "source_unit_ids": [match[1]]}]}]})

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "memory.yml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            log = root / "diagnostics.jsonl"
            with patch.dict(os.environ, {"AML_REPO_EVAL": "true", "AML_DIAGNOSTIC_LOG": str(log)}), \
                 patch("jiuwen_memory.storage.entity_impl.elasticsearch_entity_store.ElasticsearchEntityStore",
                       OfflineEntityStore), patch.object(EchoLLM, "chat", side_effect=chat):
                backend = Mem2Backend(path, "schema-test", smoke=True)
                recallers = backend.api._engine._retriever.storage.recallers
                self.assertGreaterEqual(sum(r.layer == "l2" for r in recallers), 2)
                with TestClient(create_app(backend=backend, api_key="test-only", state_dir=root / "state",
                                           concurrency=1), headers={"Authorization": "Bearer test-only"}) as client:
                    for index, text in enumerate(("Alice owns three bicycles.", "Alice prefers dark coffee.")):
                        response = client.post("/add", json={"request_id": f"r{index}", "user_id": "u1",
                            "session_id": "s1", "messages": [{"role": "user", "content": text}],
                            "evaluation_seed": {"occurred_at": "2024-01-01T00:00:00"}})
                        self.assertEqual(response.status_code, 200, response.text)
                    response = client.post("/search", json={"query": "bicycles", "user_id": "u1", "top_k": 200})
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertTrue(any("coffee" in item["content"] for item in response.json()["data"]))
                    with patch.object(OfflineEntityStore, "fail_lookup", True):
                        response = client.post("/search", json={"query": "bicycles", "user_id": "u1", "top_k": 200})
                        self.assertEqual(response.status_code, 503)
                    response = client.post("/search", json={"query": "bicycles", "user_id": "other", "top_k": 200})
                    self.assertEqual(response.json()["data"], [])
                    broken = True
                    response = client.post("/add", json={"request_id": "broken", "user_id": "u2",
                        "session_id": "s1", "messages": [{"role": "user", "content": "Alice runs."}],
                        "evaluation_seed": {}})
                    self.assertEqual(response.status_code, 200, response.text)
                events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
                self.assertTrue(any(e.get("property_units", 0) > 0 for e in events))
                self.assertTrue(any(e.get("added_candidates", 0) > 0 for e in events))
                self.assertTrue(any(e.get("source_only_fallback") for e in events))

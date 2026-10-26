import tempfile
import unittest
from pathlib import Path

import yaml

from aml_adapter.deploy_config import main, render


class DeploymentConfigTest(unittest.TestCase):
    def test_ordinary_profile_stays_on_main_chain(self):
        config = render("ordinary")
        self.assertFalse(config["globals"]["schema_enabled"])
        self.assertFalse(config["globals"]["entity_enabled"])
        self.assertEqual(config["extractor"]["default"]["target"], "llm")

    def test_schema_profile_uses_baked_schema_and_isolated_entity_index(self):
        path = "/app/agent-memory/examples/entity_schema_locomo_person.json"
        config = render("schema-entity", path)
        self.assertTrue(config["globals"]["schema_enabled"])
        self.assertTrue(config["globals"]["entity_enabled"])
        self.assertEqual(config["extractor"]["default"]["target"], "entity_schema")
        self.assertEqual(config["extractor"]["default"]["params"]["schema_path"], path)
        self.assertEqual(
            config["entity_store"]["default"]["params"]["index"],
            "${AML_ENTITY_ES_INDEX:-aml_schema_entities_v1}",
        )

    def test_cli_writes_loadable_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            schema = Path(directory) / "schema.json"
            schema.write_text("[]", encoding="utf-8")
            output = Path(directory) / "memory.yml"
            self.assertEqual(
                main(["--profile", "schema-entity", "--schema-path", str(schema),
                      "--output", str(output)]),
                0,
            )
            self.assertTrue(yaml.safe_load(output.read_text(encoding="utf-8"))["globals"]["schema_enabled"])

    def test_schema_temporal_profile_uses_persistent_dual_recall_ports(self):
        path = "/app/agent-memory/examples/entity_schema_locomo_person.json"
        config = render("schema-temporal", path)
        self.assertTrue(config["globals"]["schema_enabled"])
        self.assertTrue(config["globals"]["entity_enabled"])
        self.assertTrue(config["globals"]["schema_temporal_enabled"])
        self.assertTrue(config["globals"]["schema_temporal_force_enabled"])
        self.assertTrue(
            config["retriever"]["default"]["params"]["schema_temporal_auto_enabled"]
        )
        self.assertEqual(config["fulltext_store"]["schema_entities"]["target"], "elasticsearch")
        self.assertEqual(config["vector_store"]["schema_entities"]["target"], "milvus")


if __name__ == "__main__":
    unittest.main()

"""PR #337's temporal contracts, executed in the isolated experiment checkout."""

import json
import os
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from aml_adapter.lme_profiles import profile_files
from jiuwen_memory.common.type_def import MemoryUnit, Segment, Temporal
from jiuwen_memory.construction.entity_schema import EntitySchemaCatalog
from jiuwen_memory.construction.extractor_impl.entity_schema_extractor import (
    EntitySchemaExtractor, InvalidSchemaExtractionError,
)


@unittest.skipUnless("schema-pr337" in os.environ.get("AML_LME_REPOSITORY", ""),
                     "Run in the PR337 experiment checkout")
class PR337Tests(unittest.TestCase):
    def extract(self, value, time, sources):
        response = json.dumps({"entities": [{"name": "Alice", "entity_type": "person",
            "properties": [{"property_name": "default_property", "value": value, "time": time,
                            "source_unit_ids": [s.id for s in sources]}]}]})
        extractor = EntitySchemaExtractor(llm=SimpleNamespace(chat=lambda *a, **kw: response),
            schema=str(profile_files("schema-entity")[-1]), validation_attempts=1)
        return extractor.extract(sources)[0]

    def source(self, name, text, day):
        return MemoryUnit(id=name, segments=[Segment(content=text)],
                          temporal=Temporal(t_message=datetime(2023, 1, day, tzinfo=timezone.utc)))

    def test_selected_schema_is_locomo_person_with_42_properties(self):
        catalog = EntitySchemaCatalog.from_file(profile_files("schema-entity")[-1])
        self.assertEqual(catalog.list_types(), ["person"])
        self.assertEqual(len(catalog.get("person").dynamic_property), 42)

    def test_undated_fact_keeps_message_date_without_inventing_event_date(self):
        source = self.source("s1", "speaker=Alice: I am an engineer", 20)
        unit = self.extract("Alice is an engineer", "", [source])
        self.assertIn("Source message date: 2023-01-20", unit.content)
        self.assertIsNone(unit.temporal.t_event)

    def test_partial_event_time_keeps_month_interval(self):
        source = self.source("s1", "speaker=Alice: I will perform in February 2023", 20)
        unit = self.extract("Alice will perform in 2023-02", "2023-02", [source])
        self.assertIsNone(unit.temporal.t_event)
        self.assertEqual(unit.system_metadata["schema_event_precision"], "month")
        self.assertEqual(unit.system_metadata["schema_event_start"], "2023-02-01T00:00:00+00:00")
        self.assertEqual(unit.system_metadata["schema_event_end"], "2023-03-01T00:00:00+00:00")

    def test_relative_time_uses_supporting_message_not_first_message(self):
        early = self.source("early", "speaker=Alice: I am rehearsing after work", 1)
        supporting = self.source("support", "speaker=Alice: I will perform at the festival next month", 20)
        unit = self.extract("In 2023-02 (next month from 2023-01-20), Alice will perform at the festival",
                            "2023-02", [early, supporting])
        self.assertEqual(unit.source_ref, supporting.id)
        self.assertEqual(unit.temporal.t_message, supporting.temporal.t_message)

    def test_conflicting_event_dates_are_rejected(self):
        source = self.source("s1", "speaker=Alice: I started a new job", 20)
        with self.assertRaises(InvalidSchemaExtractionError):
            self.extract("On 2023-08-03, Alice started a new job", "2022-08", [source])

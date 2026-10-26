import unittest

from jiuwen_memory.api import Scope

from aml_adapter.backend import Mem2Backend


class ForcedTemporalSearchTest(unittest.TestCase):
    def test_general_question_defaults_to_history(self):
        self.assertEqual(
            Mem2Backend._temporal_search_extension(
                "When did Caroline go to the LGBTQ support group?"
            ),
            {"mode": "history", "include_undated": True},
        )

    def test_explicit_date_uses_range_but_remains_temporal(self):
        value = Mem2Backend._temporal_search_extension(
            "What did Caroline do on May 7, 2023?"
        )
        self.assertEqual(value["mode"], "range")
        self.assertEqual(value["event_precision"], "day")
        self.assertEqual(value["event_from"], "2023-05-07T00:00:00+00:00")
        self.assertEqual(value["event_to"], "2023-05-08T00:00:00+00:00")

    def test_search_context_forces_explicit_temporal_extension(self):
        backend = Mem2Backend.__new__(Mem2Backend)
        backend.schema_temporal_force = True
        context = backend._search_context(
            Scope(org="test", user="test"), "What does Caroline prefer?"
        )
        self.assertEqual(
            context.extensions["schema_temporal"],
            {"mode": "history", "include_undated": True},
        )


if __name__ == "__main__":
    unittest.main()

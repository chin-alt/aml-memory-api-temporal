import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from aml_adapter import beam_cutoffs as cutoffs


class BeamCutoffTests(unittest.TestCase):
    def test_evaluate_uses_exact_prefix_and_official_judge(self):
        question = {"id": "q", "question": "Question?", "question_type": "information_extraction",
                    "rubrics": ["fact"]}
        retrieved = [{"content": f"memory {index}"} for index in range(20)]
        record = {"status": "running"}

        class Pipeline:
            @staticmethod
            def render_answer_prompt(row):
                self.assertEqual(row["context"], [f"memory {index}" for index in range(10)])
                return "answer prompt"

            @staticmethod
            def render_batch_judge_prompt(question, answer, rubrics):
                return "judge prompt"

            @staticmethod
            def parse_rubric_scores(raw, count):
                return [{"index": 0, "score": 1, "reason": "ok"}]

        async def complete(pipeline, role, artifact, diagnostic, config, messages, max_tokens, json_mode=False):
            return ('answer' if role == "answer" else
                    '{"scores":[{"index":0,"score":1,"reason":"ok"}]}')

        files = {}

        def save(path, value):
            files[str(path)] = value

        def read(path):
            return files[str(path)]

        def run(coroutine):
            try:
                coroutine.send(None)
            except StopIteration as completed:
                return completed.value
            raise AssertionError("test coroutine unexpectedly suspended")

        test_temp_root = Path(__file__).resolve().parents[1] / "outputs" / "test-temp"
        test_temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_temp_root) as directory, \
                patch.object(cutoffs, "model_stage", complete), \
                patch.object(cutoffs.asyncio, "run", run), \
                patch.object(cutoffs, "save", save), patch.object(cutoffs, "read", read):
            artifact = Path(directory)
            # The in-memory save makes the resumable artifacts appear present
            # without exercising Windows antivirus/file-lock behavior.
            with patch.object(Path, "exists", lambda path: str(path) in files):
                cutoffs.evaluate(question, record, retrieved, 10, artifact, Pipeline(),
                                 {"AML_LLM_BASE_URL": "https://example", "AML_LLM_API_KEY": "x",
                                  "AML_LLM_MODEL": "m"}, SimpleNamespace(), lambda: None)
        self.assertEqual(record["retrieved"], 10)
        self.assertEqual(record["score"], 1)
        self.assertEqual(record["status"], "completed")


if __name__ == "__main__":
    unittest.main()

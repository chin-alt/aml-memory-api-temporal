"""Pinned BEAM 100K sampling and lossless Add payload construction."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import random
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .persona_data import digest, save

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/beam"
VENDOR = ROOT / "vendor/aml_beam"
HF_REPO = "Mohammadta/BEAM"
HF_REVISION = "3205395e897e7318c7b094ef4e6047b9b82dbb03"
AML_REVISION = "1b8142bfe0f20f1c5218d6b554aa0012de34e504"
SOURCE_SHA = {"100K": "6942de2dcb9a723c9b88d7a2e4936620e90b5b536de7b23f285fd2136cafc73a"}
VENDOR_SHA = {
    "data/beam/pipeline.py": "3889e26ae696abc67711daab0ebf24e3082ed50d5f85df9d9f95540a0a9b8763",
    "api_config.py": "04ccace501f29a3ce808286bfbe5a9a34ed5ed30e42143e6289772b02bbb84f9",
}
QUESTION_TYPES = (
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
)


def object_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def download(url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print("DOWNLOAD: " + str(path.name), flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "AML-BEAM-local-evaluation"})
    with urllib.request.urlopen(request, timeout=300) as response:
        content = response.read()
    temporary = path.with_suffix(path.suffix + ".download")
    temporary.write_bytes(content)
    temporary.replace(path)


def official():
    for relative, sha in VENDOR_SHA.items():
        path = VENDOR / relative
        if not path.exists():
            download(
                f"https://raw.githubusercontent.com/AML-memory/agent-memory-leaderboard/"
                f"{AML_REVISION}/{relative}",
                path,
            )
        if digest(path) != sha:
            raise ValueError("Official BEAM code changed: " + relative)
    previous = sys.modules.pop("api_config", None)
    previous_path = list(sys.path)
    sys.path.insert(0, str(VENDOR))
    try:
        source = VENDOR / "data/beam/pipeline.py"
        spec = importlib.util.spec_from_file_location("aml_official_beam", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = previous_path
        sys.modules.pop("api_config", None)
        if previous is not None:
            sys.modules["api_config"] = previous


def source_path(size: str) -> Path:
    if size not in SOURCE_SHA:
        raise ValueError("The local BEAM runner currently supports the public 100K split")
    return DATA / f"rows-{size}.json"


def ensure_source(size: str) -> Path:
    path = source_path(size)
    if not path.exists():
        url = (
            "https://datasets-server.huggingface.co/rows?"
            f"dataset=Mohammadta%2FBEAM&config=default&split={size}&offset=0&length=100"
        )
        download(url, path)
    if digest(path) != SOURCE_SHA[size]:
        raise ValueError("BEAM source checksum mismatch; do not silently change the frozen sample")
    return path


def source_rows(size: str) -> list[dict]:
    payload = json.loads(ensure_source(size).read_text(encoding="utf-8-sig"))
    rows = [item["row"] for item in payload.get("rows", [])]
    if not rows or len({str(row["conversation_id"]) for row in rows}) != len(rows):
        raise ValueError("BEAM source must contain unique conversations")
    return rows


def probing_questions(row: dict) -> dict[str, list[dict]]:
    value = row.get("probing_questions")
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError) as error:
            raise ValueError("Invalid BEAM probing_questions") from error
    if not isinstance(value, dict):
        raise ValueError("BEAM probing_questions must be a mapping")
    result = {}
    for question_type in QUESTION_TYPES:
        questions = value.get(question_type)
        if not isinstance(questions, list) or not questions:
            raise ValueError(f"Missing BEAM question type: {question_type}")
        result[question_type] = questions
    return result


def flatten_chat(row: dict) -> list[dict]:
    output = []
    for batch_index, batch in enumerate(row.get("chat") or []):
        if not isinstance(batch, list):
            raise ValueError("BEAM chat batch must be a list")
        for turn in batch:
            role = turn.get("role")
            content = turn.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str) or not content.strip():
                raise ValueError("BEAM chat contains an invalid turn")
            output.append({"role": role, "content": content, "time_anchor": turn.get("time_anchor"),
                           "batch_index": batch_index, "index": turn.get("index")})
    if not output:
        raise ValueError("BEAM chat is empty")
    return output


def _timestamp(value) -> int | None:
    if not isinstance(value, str) or not value.strip():
        return None
    for pattern in ("%B-%d-%Y", "%B %d, %Y", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(value.strip(), pattern).replace(tzinfo=timezone.utc)
            return int(parsed.timestamp() * 1000)
        except ValueError:
            pass
    return None


def add_payloads(row: dict, user_id: str):
    """Yield lossless <=2,000-word Add chunks while preserving source batch and time."""
    request_number = 0
    for batch_index, batch in enumerate(row.get("chat") or []):
        messages, words = [], 0
        for turn in batch:
            content = turn["content"]
            boundaries = list(re.finditer(r"\S+", content))
            starts = [0] + [boundaries[i].start() for i in range(2000, len(boundaries), 2000)] + [len(content)]
            for start, end in zip(starts, starts[1:]):
                part = content[start:end]
                count = len(part.split())
                if messages and (len(messages) == 20 or words + count > 2000):
                    yield {"request_id": f"{user_id}:{request_number}", "user_id": user_id,
                           "session_id": f"beam-batch-{batch_index:03d}", "messages": messages}
                    request_number += 1
                    messages, words = [], 0
                message = {"role": turn["role"], "content": part}
                timestamp = _timestamp(turn.get("time_anchor"))
                if timestamp is not None:
                    message["timestamp"] = timestamp
                messages.append(message)
                words += count
        if messages:
            yield {"request_id": f"{user_id}:{request_number}", "user_id": user_id,
                   "session_id": f"beam-batch-{batch_index:03d}", "messages": messages}
            request_number += 1


def _select_conversations(rows: list[dict], count: int, rng: random.Random, sampling: str) -> list[dict]:
    ranked = sorted(rows, key=lambda row: (
        sum(len(turn["content"].split()) for turn in flatten_chat(row)), str(row["conversation_id"])))
    if sampling == "smoke":
        return ranked[:count]
    if sampling != "stratified":
        raise ValueError("sampling must be smoke or stratified")
    bands = [ranked[:len(ranked)//3], ranked[len(ranked)//3:2*len(ranked)//3], ranked[2*len(ranked)//3:]]
    quotas = [count // 3] * 3
    for index in range(count % 3):
        quotas[index] += 1
    selected = []
    for band, quota in zip(bands, quotas):
        if quota > len(band):
            raise ValueError("Not enough conversations in a BEAM length band")
        selected.extend(rng.sample(band, quota))
    return selected


def prepare(conversations: int = 1, per_type: int = 1, seed: int = 42, size: str = "100K",
            sampling: str = "smoke"):
    rows = source_rows(size)
    if not 1 <= conversations <= len(rows):
        raise ValueError(f"conversations must be 1..{len(rows)} for {size}")
    if not 1 <= per_type <= 2:
        raise ValueError("per_type must be 1 or 2 for the public BEAM data")
    rng = random.Random(seed)
    selected_source = _select_conversations(rows, conversations, rng, sampling)
    selected_conversations, questions = [], []
    for row in selected_source:
        conversation_id = str(row["conversation_id"])
        turns = flatten_chat(row)
        selected_conversations.append({
            "id": conversation_id,
            "topic": row.get("conversation_seed", {}),
            "turns": len(turns),
            "words": sum(len(turn["content"].split()) for turn in turns),
            "source_sha256": object_digest(row),
        })
        typed = probing_questions(row)
        for question_type in QUESTION_TYPES:
            pool = list(enumerate(typed[question_type]))
            for source_index, question in rng.sample(pool, per_type):
                rubrics = question.get("rubric")
                if not isinstance(rubrics, list) or not rubrics or not all(isinstance(x, str) and x.strip() for x in rubrics):
                    raise ValueError("BEAM question has invalid rubric nuggets")
                ident = f"{size}-{conversation_id}-{question_type}-{source_index}"
                questions.append({"id": ident, "conversation_id": conversation_id,
                                  "question_type": question_type, "question": question["question"],
                                  "rubrics": rubrics, "difficulty": question.get("difficulty"),
                                  "source_index": source_index})
    selection = {
        "dataset": HF_REPO,
        "revision": HF_REVISION,
        "source_sha256": SOURCE_SHA[size],
        "size": size,
        "seed": seed,
        "conversation_count": conversations,
        "per_type": per_type,
        "sampling": (("shortest conversations for an ingestion smoke" if sampling == "smoke"
                      else "seeded short/medium/long conversation strata")
                     + "; equal questions across all ten BEAM types"),
        "sampling_mode": sampling,
        "conversations": selected_conversations,
        "questions": questions,
        "input_contract": "complete chat -> shared Add scope per conversation; probing question -> Search",
        "official_competition_bundle": False,
    }
    path = DATA / f"selection-{size}-{sampling}-{conversations}c-{per_type}pt-seed{seed}.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != selection:
        raise ValueError("Frozen BEAM selection differs; do not overwrite")
    save(path, selection)
    return selection, path


def append_conversations(selection: dict, conversation_ids: list[str]):
    """Append explicit diverse conversations without perturbing the frozen base sample."""
    result = json.loads(json.dumps(selection, ensure_ascii=False))
    existing = {row["id"] for row in result["conversations"]}
    rows = {str(row["conversation_id"]): row for row in source_rows(result["size"])}
    appended = []
    for raw_ident in conversation_ids:
        ident = str(raw_ident).strip()
        if not re.fullmatch(r"\d+", ident) or ident not in rows:
            raise ValueError("Unknown BEAM conversation ID: " + ident)
        if ident in existing:
            raise ValueError("BEAM conversation is already selected: " + ident)
        row = rows[ident]
        turns = flatten_chat(row)
        result["conversations"].append({
            "id": ident,
            "topic": row.get("conversation_seed", {}),
            "turns": len(turns),
            "words": sum(len(turn["content"].split()) for turn in turns),
            "source_sha256": object_digest(row),
        })
        typed = probing_questions(row)
        # Per-conversation seed keeps an explicit append reproducible and
        # independent of the order/size of the original sample.
        rng = random.Random(f"{result['seed']}:{ident}:{result['per_type']}")
        for question_type in QUESTION_TYPES:
            pool = list(enumerate(typed[question_type]))
            for source_index, question in rng.sample(pool, result["per_type"]):
                rubrics = question.get("rubric")
                if (not isinstance(rubrics, list) or not rubrics
                        or not all(isinstance(x, str) and x.strip() for x in rubrics)):
                    raise ValueError("BEAM question has invalid rubric nuggets")
                qid = f"{result['size']}-{ident}-{question_type}-{source_index}"
                result["questions"].append({
                    "id": qid, "conversation_id": ident, "question_type": question_type,
                    "question": question["question"], "rubrics": rubrics,
                    "difficulty": question.get("difficulty"), "source_index": source_index,
                })
        existing.add(ident)
        appended.append(ident)
    result["conversation_count"] = len(result["conversations"])
    result["appended_conversation_ids"] = appended
    result["sampling"] += "; explicitly appended conversation IDs: " + ",".join(appended)
    label = "-".join(appended)
    path = DATA / (f"selection-{result['size']}-{result['sampling_mode']}-"
                   f"{selection['conversation_count']}c-plus-{label}-"
                   f"{result['per_type']}pt-seed{result['seed']}.json")
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != result:
        raise ValueError("Frozen explicit BEAM selection differs; do not overwrite")
    save(path, result)
    return result, path


def selected_source_rows(selection: dict) -> dict[str, dict]:
    wanted = {row["id"]: row["source_sha256"] for row in selection["conversations"]}
    result = {}
    for row in source_rows(selection["size"]):
        ident = str(row["conversation_id"])
        if ident in wanted:
            if object_digest(row) != wanted[ident]:
                raise ValueError("Selected BEAM conversation changed: " + ident)
            result[ident] = row
    if set(result) != set(wanted):
        raise ValueError("Selected BEAM conversation is missing")
    return result

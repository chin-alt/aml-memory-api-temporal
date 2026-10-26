"""Pinned LoCoMo-Refined text-only data and lossless Add construction."""

from __future__ import annotations

import json
import random
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .persona_data import digest, save

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor/locomo_refined"
PUBLIC = VENDOR / "data/public"
DATA = ROOT / "data/locomo-refined"
REPOSITORY = "https://github.com/mem-eval-suite/LoCoMo_refined.git"
REVISION = "887091190789e8d6760e70b9edd696539923dc4f"
OFFICIAL_SHA = {
    "data/public/manifest.json": "e516da06c472f864151802dfbdf828885370e142229c234fa64a37e7c60e961f",
    "data/public/questions.jsonl": "a88902b3428b76c7c0d75765c844f42ca0a1b99b7b207c2d398b0d100fa579a6",
    "data/public/conversations.jsonl": "62aac9b61cb03ed83dc3123baa0b1b3e20d29aabb9ef9c59422af3377b794f23",
    "src/evaluate.py": "8f19fc984290b148faa478f3102510f4e693c521320edcdbe0b9e96bc9b9757b",
    "src/llm_judge.py": "4be1e27ec01ebfb6955bc3e8336eb8ea767aae217d781c7784f0dfce10de35e4",
    "src/bleu_f1.py": "d30ab7429ab6f2b0d2f28527f299b1240286e40f398fdf6268d7ffa0b0f76fe2",
}
CATEGORY_NAMES = {
    "1": "single-hop",
    "2": "temporal",
    "3": "multi-hop",
    "4": "open-domain",
}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ensure_official() -> None:
    """Clone once, then reject silent dataset drift."""
    if not VENDOR.exists():
        VENDOR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "--branch", "v1.0.0",
                        REPOSITORY, str(VENDOR)], check=True)
    required = [PUBLIC / name for name in ("manifest.json", "questions.jsonl", "conversations.jsonl")]
    if not all(path.is_file() for path in required):
        raise ValueError("LoCoMo-Refined checkout is incomplete")
    try:
        head = subprocess.check_output(["git", "-C", str(VENDOR), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("Cannot verify the LoCoMo-Refined source revision") from error
    if head != REVISION:
        raise ValueError(f"LoCoMo-Refined revision changed: expected {REVISION}, got {head}")
    for relative, expected in OFFICIAL_SHA.items():
        path = VENDOR / relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError("Official LoCoMo-Refined file changed: " + relative)


def official() -> tuple[list[dict], list[dict], dict]:
    ensure_official()
    manifest = json.loads((PUBLIC / "manifest.json").read_text(encoding="utf-8"))
    conversations = _read_jsonl(PUBLIC / "conversations.jsonl")
    questions = _read_jsonl(PUBLIC / "questions.jsonl")
    if (len(conversations), len(questions)) != (10, 1382):
        raise ValueError("Unexpected LoCoMo-Refined public dataset size")
    text = [row for row in questions if row.get("is_multi_modality") is False]
    if len(text) != 861:
        raise ValueError("Unexpected LoCoMo-Refined text-only question count")
    return conversations, questions, manifest


def _round_robin_sample(pool: list[dict], count: int, rng: random.Random) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in pool:
        groups[row["sample_id"]].append(row)
    for rows in groups.values():
        rng.shuffle(rows)
    conversation_ids = sorted(groups)
    rng.shuffle(conversation_ids)
    selected = []
    while len(selected) < count:
        progressed = False
        for ident in conversation_ids:
            if groups[ident] and len(selected) < count:
                selected.append(groups[ident].pop())
                progressed = True
        if not progressed:
            break
    if len(selected) != count:
        raise ValueError("Not enough text-only questions for the requested category quota")
    return selected


def prepare(experiment: str = "pilot120", seed: int = 42) -> tuple[dict, Path]:
    conversations, questions, manifest = official()
    text = [row for row in questions if row["is_multi_modality"] is False]
    if experiment == "full":
        selected = text
    else:
        quota = 4 if experiment == "smoke16" else 30 if experiment == "pilot120" else None
        if quota is None:
            raise ValueError("experiment must be smoke16, pilot120, or full")
        rng = random.Random(seed)
        selected = []
        for category in CATEGORY_NAMES:
            selected.extend(_round_robin_sample(
                [row for row in text if str(row["category"]) == category], quota, rng))
    selected = sorted(selected, key=lambda row: (row["sample_id"], row["qa_index"]))
    used = sorted({row["sample_id"] for row in selected})
    source = {row["sample_id"]: row for row in conversations}
    selection = {
        "dataset": "LoCoMo-Refined",
        "repository": REPOSITORY,
        "revision": REVISION,
        "experiment": experiment,
        "seed": seed,
        "text_only": True,
        "source_question_count": manifest["question_count"],
        "source_text_question_count": 861,
        "excluded_multimodal_questions": 521,
        "conversation_ids": used,
        "conversation_digests": {ident: object_digest(source[ident]) for ident in used},
        "category_names": CATEGORY_NAMES,
        "questions": selected,
        "input_contract": "message text only; image URLs, captions and image queries excluded",
    }
    path = DATA / f"selection-{experiment}-seed{seed}.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != selection:
        raise ValueError("Frozen LoCoMo-Refined selection differs; do not overwrite it")
    save(path, selection)
    questions_path = path.with_name(path.stem + "-questions.jsonl")
    questions_path.parent.mkdir(parents=True, exist_ok=True)
    questions_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
                              encoding="utf-8")
    return selection, path


def object_digest(value) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def selected_conversations(selection: dict) -> dict[str, dict]:
    conversations, _, _ = official()
    wanted = set(selection["conversation_ids"])
    return {row["sample_id"]: row for row in conversations if row["sample_id"] in wanted}


def _timestamp(value: str) -> int:
    parsed = datetime.strptime(value, "%I:%M %p on %d %B, %Y").replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def _parts(text: str, limit: int = 2000):
    boundaries = list(re.finditer(r"\S+", text))
    if len(boundaries) <= limit:
        yield text
        return
    starts = [0] + [boundaries[index].start() for index in range(limit, len(boundaries), limit)] + [len(text)]
    for start, end in zip(starts, starts[1:]):
        yield text[start:end]


def add_payloads(conversation: dict, user_id: str):
    """Keep sessions ordered and produce <=20-message, <=2,000-word text-only Adds."""
    request_number = 0
    previous_session = 0
    for session in conversation["sessions"]:
        session_index = int(session["session_index"])
        if session_index <= previous_session:
            raise ValueError("LoCoMo sessions are not strictly ordered")
        previous_session = session_index
        timestamp = _timestamp(session["date_time"])
        buffered, words = [], 0
        for message in session["messages"]:
            text = message.get("text")
            speaker = message.get("speaker")
            role = message.get("role")
            if role not in {"user", "assistant"} or not isinstance(text, str) or not text.strip() or not speaker:
                raise ValueError("Invalid LoCoMo text message")
            for part in _parts(text):
                content = f"[{speaker}]: {part}"
                count = len(content.split())
                if buffered and (len(buffered) == 20 or words + count > 2000):
                    yield {"request_id": f"{user_id}:{request_number}", "user_id": user_id,
                           "session_id": f"locomo-{session_index:03d}", "messages": buffered}
                    request_number += 1
                    buffered, words = [], 0
                buffered.append({"role": role, "content": content, "timestamp": timestamp})
                words += count
        if buffered:
            yield {"request_id": f"{user_id}:{request_number}", "user_id": user_id,
                   "session_id": f"locomo-{session_index:03d}", "messages": buffered}
            request_number += 1


def provenance() -> dict:
    return {
        "revision": REVISION,
        "manifest_sha256": digest(PUBLIC / "manifest.json"),
        "questions_sha256": digest(PUBLIC / "questions.jsonl"),
        "conversations_sha256": digest(PUBLIC / "conversations.jsonl"),
    }

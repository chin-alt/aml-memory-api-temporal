"""Frozen public PersonaMem-v2 text sample; no gold fields enter memory writes."""

import csv
import hashlib
import importlib.util
import json
import random
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/personamem-v2"
VENDOR = ROOT / "vendor/aml_personamem"
AML_REVISION = "1b8142bfe0f20f1c5218d6b554aa0012de34e504"
HF_REPO = "bowen-upenn/PersonaMem-v2"


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download(url, path):
    path = Path(path)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print("DOWNLOAD: " + str(path.name), flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "PersonaMem-local-evaluation"})
    with urllib.request.urlopen(request, timeout=120) as response:
        content = response.read()
    temporary = path.with_suffix(path.suffix + ".download")
    temporary.write_bytes(content)
    temporary.replace(path)


def official():
    source = VENDOR / "data/personamem/pipeline_v2.py"
    manifest = json.loads((VENDOR / "provenance.json").read_text(encoding="utf-8"))
    for relative, sha in manifest["files"].items():
        if digest(VENDOR / relative) != sha:
            raise ValueError("Official evaluation code changed: " + relative)
    sys.path.insert(0, str(VENDOR))
    spec = importlib.util.spec_from_file_location("aml_official_persona_v2", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def history_path(value):
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "data":
        raise ValueError("Unexpected public history path")
    if "chat_history_32k" not in path.parts:
        raise ValueError("Expected the text-only 32k history")
    return str(path)


def messages(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("chat_history", data.get("conversations"))
    if not isinstance(data, list) or not data:
        raise ValueError("Expected a nonempty history message list")
    result = []
    for row in data:
        if row.get("role") not in {"user", "assistant", "system"}:
            raise ValueError("History contains unsupported role; do not silently discard it")
        content = row.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("History contains empty or non-text content")
        result.append({"role": row["role"], "content": content})
    return result


def sample_rows(rows, seed=42, users=5, per_user=5):
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        if row.get("correct_answer", "").strip() and row.get("preference", "").strip():
            grouped[row["persona_id"]].append(dict(row, id=f"row-{index:05d}"))
    rng = random.Random(seed)
    eligible = sorted(key for key, values in grouped.items() if len(values) >= per_user)
    if len(eligible) < users:
        raise ValueError("Insufficient personas with enough answered questions")
    selected = []
    for person in sorted(rng.sample(eligible, users)):
        # Round-robin shuffled preference types, not answer-dependent cherry-picking.
        buckets = defaultdict(list)
        for row in grouped[person]:
            buckets[row.get("pref_type", "unknown")].append(row)
        keys = sorted(buckets)
        rng.shuffle(keys)
        for key in keys:
            rng.shuffle(buckets[key])
        chosen = []
        while len(chosen) < per_user:
            for key in keys:
                if buckets[key] and len(chosen) < per_user:
                    chosen.append(buckets[key].pop())
        selected.extend(chosen)
    return selected


def prepare():
    selection_path = DATA / "selection.json"
    if selection_path.exists():
        return validate_selection(selection_path)
    for relative in ("data/personamem/pipeline_v2.py", "api_config.py"):
        download(f"https://raw.githubusercontent.com/AML-memory/agent-memory-leaderboard/{AML_REVISION}/{relative}",
                 VENDOR / relative)
    save(VENDOR / "provenance.json", {
        "repository": "https://github.com/AML-memory/agent-memory-leaderboard",
        "revision": AML_REVISION,
        "files": {p: digest(VENDOR / p) for p in ("data/personamem/pipeline_v2.py", "api_config.py")}})
    metadata = DATA / "dataset-revision.json"
    download(f"https://huggingface.co/api/datasets/{HF_REPO}/revision/main", metadata)
    revision = json.loads(metadata.read_text(encoding="utf-8"))["sha"]
    csv_path = DATA / "benchmark/text/benchmark.csv"
    download(f"https://huggingface.co/datasets/{HF_REPO}/resolve/{revision}/benchmark/text/benchmark.csv", csv_path)
    csv.field_size_limit(10_000_000)
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        selected = sample_rows(list(csv.DictReader(handle)))
    pipeline = official()
    files = {}
    for row in selected:
        relative = history_path(row["chat_history_32k_link"])
        path = DATA / relative
        download(f"https://huggingface.co/datasets/{HF_REPO}/resolve/{revision}/{relative}", path)
        history = messages(path)
        files[relative] = digest(path)
        # Freeze options once. Launcher pins PYTHONHASHSEED as upstream uses hash().
        _, mapping, correct = pipeline.official_mcq_options(row)
        row.update(history_file=relative, history_messages=len(history),
                   option_mapping=mapping, correct_letter=correct)
    selection = {"dataset": HF_REPO, "revision": revision, "size": "32k", "seed": 42,
                 "users": 5, "per_user": 5, "csv_sha256": digest(csv_path),
                 "sampling": "uniform personas, round-robin preference types", "files": files, "rows": selected}
    save(selection_path, selection)
    return selection


def validate_selection(path):
    selection = json.loads(Path(path).read_text(encoding="utf-8"))
    for relative, sha in selection["files"].items():
        if digest(DATA / history_path(relative)) != sha:
            raise ValueError("History changed: " + relative)
    return selection


def add_chunks(history, user_id):
    """20 messages / 2000 whitespace words; split oversized messages losslessly."""
    import re
    chunk, words, number = [], 0, 0
    for message in history:
        # The contest Add contract has no system role. Preserve its data explicitly.
        role = "user" if message["role"] == "system" else message["role"]
        content = ("[Original history system message; stored as reference data]\n" + message["content"]
                   if message["role"] == "system" else message["content"])
        boundaries = list(re.finditer(r"\S+", content))
        starts = [0] + [boundaries[i].start() for i in range(2000, len(boundaries), 2000)] + [len(content)]
        for start, end in zip(starts, starts[1:]):
            part = content[start:end]
            count = len(part.split())
            if chunk and (len(chunk) == 20 or words + count > 2000):
                yield {"request_id": f"{user_id}:{number}", "user_id": user_id,
                       "session_id": "history", "messages": chunk}
                chunk, words, number = [], 0, number + 1
            chunk.append({"role": role, "content": part})
            words += count
    if chunk:
        yield {"request_id": f"{user_id}:{number}", "user_id": user_id,
               "session_id": "history", "messages": chunk}

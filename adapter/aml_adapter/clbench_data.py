"""Pinned CL-bench follow-up smoke sample; gold stays outside Add/Search/Answer."""

import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

from .persona_data import add_chunks, digest, download, save

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/clbench"
VENDOR = ROOT / "vendor/aml_clbench"
HF_REVISION = "b28a5832a09b0d96c0cf4c22e90d7c60ede25b80"
AML_REVISION = "1b8142bfe0f20f1c5218d6b554aa0012de34e504"
DATA_SHA = "d5fc88d4b2eea75c61dd40862021b6ae2fba26bd21b58e8c5e18377a763943be"
VENDOR_SHA = {
    "data/clbench/pipeline.py": "65b9719dc22e055aad808f68d757c145b9f9115cb2275b1ee373610c6a8116c7",
    "api_config.py": "04ccace501f29a3ce808286bfbe5a9a34ed5ed30e42143e6289772b02bbb84f9",
}
CATEGORIES = ("Domain Knowledge Reasoning", "Empirical Discovery & Simulation",
              "Procedural Task Execution", "Rule System Application")
POLICY = "follow-up-only; query<=512 whitespace words; shortest history quartile per category; seed42"
STRATIFIED_POLICY = ("follow-up-only; query<=512 whitespace words; one seeded question per distinct context; "
                     "history-word-count rank thirds within category; balanced short/medium/long")


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def official():
    for name, sha in VENDOR_SHA.items():
        if digest(VENDOR / name) != sha:
            raise ValueError("Official CL-bench code changed: " + name)
    # The upstream module imports a generic api_config name. Avoid colliding with
    # other vendored benchmark modules in tests or combined offline analysis.
    previous = sys.modules.pop("api_config", None)
    previous_path = list(sys.path)
    sys.path.insert(0, str(VENDOR))
    try:
        spec = importlib.util.spec_from_file_location("aml_official_clbench", VENDOR / "data/clbench/pipeline.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = previous_path
        sys.modules.pop("api_config", None)
        if previous is not None:
            sys.modules["api_config"] = previous


def convert_row(row):
    messages = row["messages"]
    if (len(messages) < 4 or messages[0]["role"] != "system"
            or messages[-1]["role"] != "user"):
        raise ValueError("Smoke requires a system prompt and a separate follow-up query")
    if any(m["role"] != ("user" if i % 2 else "assistant")
           for i, m in enumerate(messages[1:], 1)):
        raise ValueError("Unexpected conversation ordering")
    if any(not isinstance(m["content"], str) or not m["content"].strip() for m in messages):
        raise ValueError("Empty/non-text message")
    if not row.get("rubrics"):
        raise ValueError("Missing official rubrics")
    meta = row["metadata"]
    history = [{"role": m["role"], "content": m["content"]} for m in messages[1:-1]]
    return {"id": meta["task_id"], "context_id": meta["context_id"],
            "category": meta["context_category"], "sub_category": meta["sub_category"],
            "system_prompt": messages[0]["content"], "history": history,
            "question": messages[-1]["content"], "rubrics": row["rubrics"],
            "source_sha256": object_digest(row),
            "history_words": sum(len(m["content"].split()) for m in history)}


def sample_rows(rows, per_category=1, seed=42):
    rng = random.Random(seed)
    selected, pools = [], {}
    for category in CATEGORIES:
        eligible = [convert_row(r) for r in rows
                    if r["metadata"]["context_category"] == category and len(r["messages"]) >= 4
                    and len(r["messages"][-1]["content"].split()) <= 512]
        eligible.sort(key=lambda r: (r["history_words"], r["id"]))
        pool = eligible[:max(1, len(eligible) // 4)]
        if len(pool) < per_category:
            raise ValueError("Not enough smoke candidates in " + category)
        pools[category] = {"eligible": len(eligible), "short_quartile": len(pool)}
        selected.extend(rng.sample(pool, per_category))
    return selected, pools


def sample_stratified(rows, per_category=10, seed=42):
    """Sample contexts first, then balance representative questions by history length.

    Taking one seeded question per context prevents multi-question documents from
    getting extra weight. Rank thirds are relative to each category, not token
    windows. For 10 questions the allocation is short=3, medium=4, long=3.
    """
    rng = random.Random(seed)
    selected, pools, used_contexts = [], {}, set()
    for category in CATEGORIES:
        by_context = {}
        for raw in rows:
            if (raw["metadata"]["context_category"] != category or len(raw["messages"]) < 4
                    or len(raw["messages"][-1]["content"].split()) > 512):
                continue
            row = convert_row(raw)
            if row["context_id"] not in used_contexts:
                by_context.setdefault(row["context_id"], []).append(row)
        representatives = [rng.choice(sorted(by_context[c], key=lambda r: r["id"]))
                           for c in sorted(by_context)]
        representatives.sort(key=lambda r: (r["history_words"], r["id"]))
        n = len(representatives)
        bands = {"short": representatives[:n // 3],
                 "medium": representatives[n // 3:2 * n // 3],
                 "long": representatives[2 * n // 3:]}
        quotas = dict.fromkeys(bands, per_category // 3)
        for band in ("medium", "long")[:per_category % 3]:
            quotas[band] += 1
        info = {"eligible_questions": sum(map(len, by_context.values())),
                "eligible_contexts": n, "bands": {}}
        for band, pool in bands.items():
            if len(pool) < quotas[band]:
                raise ValueError(f"Not enough distinct contexts for {category}/{band}")
            picked = rng.sample(pool, quotas[band])
            info["bands"][band] = {"contexts": len(pool), "selected": len(picked),
                                   "min_words": min((r["history_words"] for r in pool), default=None),
                                   "max_words": max((r["history_words"] for r in pool), default=None)}
            for row in picked:
                selected.append({**row, "history_band": band})
                used_contexts.add(row["context_id"])
        pools[category] = info
    if len(used_contexts) != len(selected):
        raise ValueError("Duplicate context in stratified sample")
    return selected, pools


def prepare(per_category=1, seed=42, sampling="smoke"):
    if sampling not in {"smoke", "stratified"}:
        raise ValueError("Unknown sampling policy")
    source = DATA / "CL-bench.jsonl"
    download(f"https://huggingface.co/datasets/tencent/CL-bench/resolve/{HF_REVISION}/CL-bench.jsonl", source)
    if digest(source) != DATA_SHA:
        raise ValueError("CL-bench dataset checksum mismatch")
    for name in VENDOR_SHA:
        download(f"https://raw.githubusercontent.com/AML-memory/agent-memory-leaderboard/{AML_REVISION}/{name}",
                 VENDOR / name)
    official()
    save(VENDOR / "provenance.json", {"repository": "https://github.com/AML-memory/agent-memory-leaderboard",
                                    "revision": AML_REVISION, "files": VENDOR_SHA})
    # JSON strings may contain literal U+2028/U+2029; str.splitlines would split
    # inside them. Only the physical JSONL newline is a record boundary.
    with source.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    sampler = sample_rows if sampling == "smoke" else sample_stratified
    chosen, pools = sampler(rows, per_category, seed)
    selection = {"dataset": "tencent/CL-bench", "revision": HF_REVISION, "dataset_sha256": DATA_SHA,
                 "aml_revision": AML_REVISION, "seed": seed, "per_category": per_category,
                 "policy": POLICY if sampling == "smoke" else STRATIFIED_POLICY,
                 "eligible_pools": pools, "rows": chosen,
                 "input_contract": "messages[1:-1] -> Add; last user -> Search; original system -> Answer",
                 "official_competition_bundle": False}
    suffix = "" if sampling == "smoke" else "-stratified"
    path = DATA / f"selection-{4 * per_category}-seed{seed}{suffix}.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != selection:
        raise ValueError("Frozen selection differs; do not overwrite")
    save(path, selection)
    return selection, path


def payloads(row, user_id):
    return list(add_chunks(row["history"], user_id))

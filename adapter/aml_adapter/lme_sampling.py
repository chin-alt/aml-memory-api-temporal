"""固定种子的 LongMemEval 分层抽题；仅处理文件，不连接模型或数据库。"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

BUCKETS = ("single-session-user", "single-session-assistant", "single-session-preference",
           "multi-session", "temporal-reasoning", "knowledge-update", "abstention")
ROOT = Path(__file__).resolve().parents[1]


def bucket(row):
    return "abstention" if str(row["question_id"]).endswith("_abs") else row["question_type"]


def counts(row):
    sessions = row["haystack_sessions"]
    texts = [turn["content"] for session in sessions for turn in session]
    return {"sessions": len(sessions), "messages": len(texts),
            "characters": sum(map(len, texts)), "words": sum(len(text.split()) for text in texts)}


def has_standard_answer(row):
    answer = row.get("answer")
    return (bucket(row) != "abstention" and isinstance(answer, (str, int, float))
            and not isinstance(answer, bool) and bool(str(answer).strip()))


def select(rows, per_type=3, seed=42, answerable_only=False):
    if per_type < 1:
        raise ValueError("per_type must be positive")
    groups = defaultdict(list)
    ids = set()
    for index, row in enumerate(rows):
        qid = str(row["question_id"])
        if qid in ids:
            raise ValueError("Duplicate question_id in dataset")
        ids.add(qid)
        if answerable_only and not has_standard_answer(row):
            continue
        groups[bucket(row)].append((index, row))
    unknown = set(groups) - set(BUCKETS)
    if unknown:
        raise ValueError(f"Unknown question types: {sorted(unknown)}")
    active_buckets = BUCKETS[:-1] if answerable_only else BUCKETS
    for name in active_buckets:
        if len(groups[name]) < per_type:
            raise ValueError(f"{name}: need {per_type}, available {len(groups[name])}; mini cannot cover all types")
        groups[name].sort(key=lambda pair: hashlib.sha256(
            f"{seed}:{pair[1]['question_id']}".encode()).hexdigest())
    # 每轮各类型一题，较早停止也能先覆盖类型；不按长度或答案筛选。
    return [groups[name][rank] for rank in range(per_type) for name in active_buckets]


def prepare(data, output, per_type=3, seed=42, variant="s", answerable_only=False):
    raw = Path(data).read_bytes()
    rows = json.loads(raw.decode("utf-8-sig"))
    selected = select(rows, per_type, seed, answerable_only)
    subset = [row for _, row in selected]
    # 沿用 HTTP Add 的合法输入检查；不删 session、turn 或正文。
    from .lme_smoke import add_requests
    cases = []
    for position, (index, row) in enumerate(selected):
        chunks = list(add_requests(row, "validate-only"))
        cases.append({"subset_index": position, "source_index": index,
                      "question_id": row["question_id"], "question_type": row["question_type"],
                      "bucket": bucket(row), "question": row["question"],
                      **counts(row), "add_chunks": len(chunks)})
    encoded = (json.dumps(subset, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    manifest = {"version": 1, "source": str(Path(data).resolve()),
                "source_sha256": hashlib.sha256(raw).hexdigest(), "source_count": len(rows),
                "variant": variant, "per_type": per_type, "seed": seed,
                "answerable_only": answerable_only,
                "answer_policy": "exclude abstention and missing/blank answers" if answerable_only else "all buckets",
                "eligible_by_bucket": dict(Counter(bucket(row) for row in rows
                                                     if not answerable_only or has_standard_answer(row))),
                "excluded_count": sum(not has_standard_answer(row) for row in rows) if answerable_only else 0,
                "selection": "sha256(seed:question_id) ascending within disjoint buckets; round-robin order",
                "history_policy": "All original sessions and turns retained in dataset order; no gold filtering",
                "dataset": "selected.json", "dataset_sha256": hashlib.sha256(encoded).hexdigest(),
                "available_by_bucket": dict(Counter(bucket(row) for row in rows)),
                "selected_by_bucket": dict(Counter(case["bucket"] for case in cases)),
                "totals": {key: sum(case[key] for case in cases)
                           for key in ("sessions", "messages", "characters", "words", "add_chunks")},
                "cases": cases}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "selected.json").write_bytes(encoded)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [f"# LongMemEval {variant} 分层抽样", "",
             f"每类 {per_type} 题，共 {len(cases)} 题；seed={seed}。", "",
             "排除拒答题与缺失/空白标准答案的题。" if answerable_only else "拒答题单列，不重复计入原 question_type。", "",
             f"来源 SHA-256：`{manifest['source_sha256']}`", "",
             f"输入总量：{manifest['totals']}。words 是空白分词数，不是模型 token 数。", "",
             "仅减少题目数，保留选中题目的全部原始历史。Oracle 数据本身只包含证据会话，不能代表 S/M 的干扰检索难度。", "",
             "|序号|类别|原下标|题号|会话|消息|词数|问题|", "|---|---|---:|---|---:|---:|---:|---|"]
    for case in cases:
        question = case["question"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"|{case['subset_index'] + 1}|{case['bucket']}|{case['source_index']}|"
                     f"{case['question_id']}|{case['sessions']}|{case['messages']}|{case['words']}|{question}|")
    (output / "selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ROOT / "data/longmemeval_s_cleaned.json"))
    parser.add_argument("--output", default=str(ROOT / "data/lme-s-21-seed42"))
    parser.add_argument("--per-type", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant", choices=("oracle", "s", "m", "custom"), default="s")
    parser.add_argument("--answerable-only", action="store_true")
    args = parser.parse_args(argv)
    manifest = prepare(args.data, args.output, args.per_type, args.seed, args.variant, args.answerable_only)
    print(json.dumps({"selected_by_bucket": manifest["selected_by_bucket"],
                      "totals": manifest["totals"]}, indent=2))
    print("Selection: " + str(Path(args.output).resolve() / "selection.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

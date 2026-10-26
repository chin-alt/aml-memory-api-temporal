"""Deterministic PersonaMem question expansion using the same frozen user histories."""

import copy
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone

from .persona_data import DATA, digest, history_path, official, save


def extend_selection(selection, per_user):
    if per_user < selection["per_user"]:
        raise ValueError("Expansion cannot remove previously selected questions")
    if per_user == selection["per_user"]:
        return copy.deepcopy(selection)
    path = DATA / "benchmark/text/benchmark.csv"
    if digest(path) != selection["csv_sha256"]:
        raise ValueError("Frozen benchmark CSV changed")
    csv.field_size_limit(10_000_000)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row, id=f"row-{index:05d}") for index, row in enumerate(csv.DictReader(handle))]
    rng = random.Random(selection["seed"])
    pipeline = official()
    selected = []
    for person in sorted({r["persona_id"] for r in selection["rows"]}):
        existing = [copy.deepcopy(r) for r in selection["rows"] if r["persona_id"] == person]
        ids = {r["id"] for r in existing}
        histories = {r["history_file"] for r in existing}
        buckets = defaultdict(list)
        for row in rows:
            if (row["persona_id"] == person and row["id"] not in ids
                    and row.get("correct_answer", "").strip() and row.get("preference", "").strip()):
                if history_path(row["chat_history_32k_link"]) not in histories:
                    raise ValueError("New question refers to a different history: " + row["id"])
                buckets[row["pref_type"]].append(row)
        if len(existing) + sum(map(len, buckets.values())) < per_user:
            raise ValueError(f"Persona {person} has fewer than {per_user} eligible questions")
        keys = sorted(buckets)
        rng.shuffle(keys)
        for key in keys:
            rng.shuffle(buckets[key])
        counts = Counter(r["pref_type"] for r in existing)
        while len(existing) < per_user:
            # Fill underrepresented types first; shuffled key order breaks ties reproducibly.
            key = min((k for k in keys if buckets[k]), key=lambda k: counts[k])
            row = copy.deepcopy(buckets[key].pop())
            _, mapping, correct = pipeline.official_mcq_options(row)
            row.update(history_file=history_path(row["chat_history_32k_link"]),
                       history_messages=existing[0]["history_messages"], option_mapping=mapping,
                       correct_letter=correct)
            existing.append(row)
            counts[key] += 1
        selected.extend(existing)
    result = copy.deepcopy(selection)
    result.update(per_user=per_user, rows=selected,
                  sampling="same frozen personas; preserve old questions; fill least-represented preference types",
                  expansion={"original_ids": [r["id"] for r in selection["rows"]],
                             "original_per_user": selection["per_user"]})
    return result


def save_selection(selection):
    path = DATA / f"selection-{selection['users']}x{selection['per_user']}-seed{selection['seed']}.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != selection:
        raise ValueError("Expanded selection already exists with different content: " + str(path))
    if not path.exists():
        save(path, selection)
    return path


def expand_state(state, selection, args, signature, output):
    """Called only under the batch runner lock, after verifying the predecessor fingerprint."""
    if state["top_k"] != args.top_k or state["mode"] != args.mode:
        raise ValueError("Expansion cannot change retrieval top_k or QA mode")
    old_cutoffs = state.get("qa_cutoffs")
    new_cutoffs = args.qa_cutoffs
    if old_cutoffs and (old_cutoffs != new_cutoffs or state["mcq_format"] != args.mcq_format):
        raise ValueError("Expansion must preserve existing multi-cutoff settings")
    if not old_cutoffs and new_cutoffs:
        if state["top_k"] not in new_cutoffs or args.mcq_format != "original":
            raise ValueError("Reusing legacy QA requires its top_k cutoff and original MCQ prompt")
    old_rows = {r["id"]: r for r in state["selection"]["rows"]}
    new_rows = {r["id"]: r for r in selection["rows"]}
    if len(new_rows) != len(selection["rows"]):
        raise ValueError("Duplicate question IDs")
    if not old_rows.keys() <= new_rows.keys() or any(new_rows[k] != r for k, r in old_rows.items()):
        raise ValueError("Expansion must preserve every previous question and frozen option mapping")
    if selection["files"] != state["selection"]["files"]:
        raise ValueError("Expansion must reuse exactly the same frozen histories")
    if state["selection"] == selection and old_cutoffs == new_cutoffs:
        result = copy.deepcopy(state)
        result["fingerprint"] = signature
        return result
    backup = output / f"state.before-expansion-{state['fingerprint'][:16]}.json"
    if not backup.exists():
        save(backup, state)
    result = copy.deepcopy(state)
    for qid, row in new_rows.items():
        if qid not in result["questions"]:
            result["questions"][qid] = {"status": "pending", "persona_id": row["persona_id"],
                                        "pref_type": row["pref_type"]}
    reused = []
    if not old_cutoffs and new_cutoffs:
        from .persona_cutoffs import score_mcq
        pipeline = official()
        for qid in old_rows:
            record = result["questions"][qid]
            artifact = output / "questions" / qid
            target = artifact / f"top-{state['top_k']}"
            for path in artifact.glob("*.json*"):
                if path.name == "search.json":
                    continue
                target.mkdir(parents=True, exist_ok=True)
                dest = target / path.name
                if dest.exists() and digest(dest) != digest(path):
                    raise ValueError("Conflicting migrated QA artifact: " + str(dest))
                if not dest.exists():
                    shutil.copyfile(path, dest)
            record["cutoffs"] = {str(k): {"status": "pending"} for k in new_cutoffs}
            old_result = {k: copy.deepcopy(v) for k, v in record.items()
                          if k not in ("cutoffs", "persona_id", "pref_type")}
            answer_path = target / "mcq-answer.json"
            if answer_path.exists():
                answer = json.loads(answer_path.read_text(encoding="utf-8"))["answer"]
                old_result.update(score_mcq(pipeline, old_rows[qid], answer, target))
            record["cutoffs"][str(state["top_k"])] = old_result
            if old_result["status"] == "completed":
                reused.append(qid)
                record["status"] = "pending" if len(new_cutoffs) > 1 else "completed"
    result.update(selection=selection, fingerprint=signature)
    if new_cutoffs:
        result.update(qa_cutoffs=new_cutoffs, mcq_format=args.mcq_format)
    result.setdefault("expansions", []).append({
        "time": datetime.now(timezone.utc).isoformat(), "previous_question_count": len(old_rows),
        "question_count": len(new_rows), "new_ids": sorted(new_rows.keys() - old_rows.keys()),
        "reused_legacy_top_k_ids": reused, "state_backup": str(backup),
        "histories_reused": True,
    })
    return result

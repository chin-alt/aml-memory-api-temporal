"""Supplement PersonaMem QA using cached HTTP Search results, without re-ingestion."""

import argparse
import asyncio
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from dotenv import load_dotenv

from . import persona_batch as base
from .diagnostics import DiagnosticLog, exception_info
from .persona_data import ROOT, VENDOR, digest, official, save
from .state import ProcessLock


FORMAT_INSTRUCTION = (
    "Output format is mandatory: after your reasoning, end with exactly one plain-text line "
    "'Final Answer: X', replacing X with your chosen option letter. "
    "Do not use Markdown on that line. Do not put any text after it."
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalized_choice(answer, options, pipeline):
    """Conservative syntax repair only; never inspect a gold label or option text."""
    parsed = pipeline.extract_final_letter(answer)
    if parsed:
        return parsed if parsed in options else ""
    # Only an explicit leading choice is accepted. Multiple explicit choices are ambiguous.
    plain = re.sub(r"[*_`]", "", answer).strip()
    markers = re.findall(
        r"(?:final\s+answer\s*(?:is|:)\s*|(?:the\s+)?(?:best\s+)?(?:answer|choice|option)\s+is\s+)([A-Z])\b",
        plain, re.I,
    )
    lead = re.match(r"^([A-Z])(?:\s*[.)](?:\s|$)|\s+is\s+the\s+best\s+answer\b|\s*$)", plain, re.I)
    if lead:
        markers.append(lead.group(1))
    # A leading 'The best answer is B' is also an explicit choice, not a mention in reasoning.
    phrase = re.match(r"^(?:The\s+)?(?:best\s+)?(?:answer|choice|option)\s+is\s+([A-Z])\b", plain, re.I)
    if not lead and not phrase:
        return ""
    choices = {letter.upper() for letter in markers}
    # Refuse lists of alternatives and revisions rather than guessing their intent.
    alternatives = re.findall(r"\b([A-Z])(?:\s+(?:or|and)\s+|\s*/\s*)([A-Z])\b", plain, re.I)
    choices.update(letter.upper() for pair in alternatives for letter in pair)
    choices.update(re.findall(r"^([A-Z])[.)]\s", plain, re.M))
    return next(iter(choices)) if len(choices) == 1 and choices <= set(options) else ""


def score_mcq(pipeline, row, answer, artifact):
    def score(text, name):
        source, target = artifact / (name + "-input.jsonl"), artifact / (name + "-score.jsonl")
        pipeline.write_jsonl(source, [{"id": row["id"], "generated_answer": text,
            "option_mapping": row["option_mapping"], "correct_letter": row["correct_letter"]}])
        pipeline.evaluate_mcq(argparse.Namespace(answers=str(source), output=str(target)))
        return pipeline.read_jsonl(target)[0]
    strict = score(answer, "mcq")
    letter = normalized_choice(answer, row["option_mapping"], pipeline)
    normalized = "Final Answer: " + letter if letter else answer
    save(artifact / "mcq-normalization.json", {
        "original_answer": answer, "normalized_answer": normalized,
        "repaired": not strict["predicted_letter"] and bool(letter),
        "policy": "explicit choice only; no gold label used; supplementary metric",
    })
    repaired = score(normalized, "mcq-normalized")
    return {"mcq_correct": strict["is_correct"], "mcq_normalized_correct": repaired["is_correct"],
            "mcq_format_invalid": strict["predicted_letter"] not in row["option_mapping"],
            "mcq_format_repaired": not strict["predicted_letter"] and bool(letter)}


def evaluate(row, record, retrieved, cutoff, artifact, pipeline, args, env, diagnostic, checkpoint):
    artifact.mkdir(parents=True, exist_ok=True)
    selected = retrieved[:cutoff]
    record["retrieved"] = len(selected)
    modes = ("mcq", "generative") if args.mode == "both" else (args.mode,)
    for mode in modes:
        answer_path = artifact / (mode + "-answer.json")
        if not answer_path.exists():
            record["stage"] = mode + ".answer"
            checkpoint()
            prompt = base.answer_messages(pipeline, row, selected, mode)
            if mode == "mcq" and args.mcq_format == "reinforced":
                prompt[-1]["content"] += "\n\n" + FORMAT_INSTRUCTION
            save(artifact / (mode + "-prompt.json"), prompt)
            answer = asyncio.run(base.complete(pipeline, env, "answer", prompt, diagnostic))
            save(answer_path, {"answer": answer})
        answer = read(answer_path)["answer"]
        if mode == "mcq":
            record.update(score_mcq(pipeline, row, answer, artifact))
        else:
            judge_path = artifact / "judge.json"
            if not judge_path.exists():
                record["stage"] = "generative.judge"
                checkpoint()
                preference = row["preference"]
                template = (pipeline.JUDGE_PROMPT_NARROW_NEGATIVE if preference.lower().startswith("do not")
                            else pipeline.JUDGE_PROMPT_NARROW_POSITIVE)
                prompt = [{"role": "user", "content": template.format(
                    user_query=pipeline.user_query_text(row), preference=preference, model_response=answer)}]
                save(artifact / "judge-prompt.json", prompt)
                text = asyncio.run(base.complete(pipeline, env, "judge", prompt, diagnostic))
                save(judge_path, {"response": text, "score": pipeline.extract_boxed_score(text)})
            record["narrow_score"] = read(judge_path)["score"]
        checkpoint()


def stats(records):
    rows = list(records)
    result = base.stats(rows)
    completed = [r for r in rows if r["status"] == "completed"]
    for key, name in [("mcq_normalized_correct", "mcq_normalized_accuracy"),
                      ("mcq_format_invalid", "mcq_format_invalid_rate")]:
        values = [r[key] for r in completed if key in r]
        result[name] = sum(values) / len(values) if values else None
    result["pending_source"] = sum(r["status"] == "pending_source" for r in rows)
    return result


def summarize(state, output):
    groups = {"top-" + str(k): {qid: q["cutoffs"][str(k)] for qid, q in state["questions"].items()}
              for k in state["cutoffs"]}
    groups["original-top-" + str(state["source_top_k"])] = {
        qid: q.get("original", {"status": "pending_source"}) for qid, q in state["questions"].items()}
    def grouped(ids):
        return {name: stats(records[qid] for qid in ids) for name, records in groups.items()}
    common = [qid for qid in state["questions"]
              if all(records[qid]["status"] == "completed" for records in groups.values())]
    summary = {"profile": state["profile"], "mcq_format": state["mcq_format"],
               "original_mcq_format": "unchanged source prompt", "all": grouped(state["questions"]),
               "paired_ids": common, "paired": grouped(common),
               "by_preference_type": {}, "by_persona": {}}
    for field, dest in [("pref_type", "by_preference_type"), ("persona_id", "by_persona")]:
        for value in sorted({q[field] for q in state["questions"].values()}):
            ids = [qid for qid, q in state["questions"].items() if q[field] == value]
            summary[dest][value] = {"all": grouped(ids), "paired": grouped([q for q in common if q in ids])}
    summary["failures"] = [{"question": qid, "group": group, "stage": rec.get("stage", rec.get("failure_stage"))}
                           for group, records in groups.items() for qid, rec in records.items()
                           if rec["status"] == "failed"]
    save(output / "summary.json", summary)
    lines = ["# PersonaMem cached cutoff evaluation", "", "Local diagnostic, not an official competition score.",
             "Original source artifacts and strict scores are unchanged. Normalized MCQ is supplementary.",
             f"New MCQ prompt: {state['mcq_format']}; original: unchanged source prompt.",
             "Each row uses completed questions only. Use paired results to compare equal denominators.", "",
             "| Context | Completed | Failed | Strict MCQ | Normalized MCQ | Narrow mean |",
             "| --- | --- | --- | --- | --- | --- |"]
    for name, item in summary["all"].items():
        lines.append(f"| {name} | {item['completed']}/{item['total']} | {item['failed']} | "
                     f"{item['mcq_accuracy']} | {item['mcq_normalized_accuracy']} | {item['narrow_mean']} |")
    lines += ["", f"Common completed questions across all rows: {len(common)}.",
              "Paired scores and per-user/per-type breakdowns: summary.json.", "", "Failed question/cutoff pairs:"]
    lines += [f"- {f['question']} / {f['group']} / {f['stage']}" for f in summary["failures"]] or ["- none"]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(state, source, output, args, env, pipeline, diagnostic):
    def checkpoint():
        save(output / "state.json", state)
    # Read the latest source status, but never write to the original run.
    original = read(source / "state.json")
    for row in state["selection"]["rows"]:
        qid = row["id"]
        question = state["questions"][qid]
        source_artifact = source / "questions" / qid
        artifact = output / "questions" / qid
        original_record = dict(original["questions"][qid])
        original_dir = artifact / "original"
        original_dir.mkdir(parents=True, exist_ok=True)
        answer_path = source_artifact / "mcq-answer.json"
        if answer_path.exists():
            original_record.update(score_mcq(pipeline, row, read(answer_path)["answer"], original_dir))
        question["original"] = original_record
        snapshot = artifact / "search.json"
        if not snapshot.exists() or "snapshot_sha256" not in question:
            history = original["histories"].get(row["persona_id"] + ":" + row["history_file"], {})
            search = source_artifact / "search.json"
            if history.get("status") != "completed" or not search.exists():
                print(f"WAIT SOURCE: {qid}; complete history/Search required", flush=True)
                checkpoint()
                continue
            save(snapshot, read(search))
            question["source_search_sha256"] = digest(search)
            question["snapshot_sha256"] = digest(snapshot)
        if digest(snapshot) != question["snapshot_sha256"]:
            raise ValueError("Cached Search snapshot changed: " + qid)
        if (source_artifact / "search.json").exists() and digest(source_artifact / "search.json") != question["source_search_sha256"]:
            raise ValueError("Source Search changed: " + qid)
        retrieved = read(snapshot)["data"]
        for cutoff in state["cutoffs"]:
            record = question["cutoffs"][str(cutoff)]
            if not base.should_run(record, args.skip_failed):
                continue
            print(f"QA {qid} top-{cutoff}; memories={min(cutoff, len(retrieved))}", flush=True)
            record["status"] = "running"
            checkpoint()
            try:
                evaluate(row, record, retrieved, cutoff, artifact / f"top-{cutoff}", pipeline,
                         args, env, diagnostic, checkpoint)
                record.update(status="completed", stage="done")
                record.pop("error", None)
            except Exception as exc:
                record.update(status="failed", error=exception_info(exc))
                diagnostic.emit(record.get("stage", "qa"), "failed", question_id=qid, cutoff=cutoff,
                                error=record["error"])
                print(f"FAILED {qid} top-{cutoff}: {type(exc).__name__}; continuing", flush=True)
            checkpoint()
        summarize(state, output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-batch", type=Path, required=True)
    parser.add_argument("--cutoffs", default="10,20,50")
    parser.add_argument("--mcq-format", choices=("reinforced", "original"), default="reinforced")
    parser.add_argument("--mode", choices=("both", "mcq", "generative"), default="both")
    parser.add_argument("--env-file", default=str(ROOT / "config/longmemeval.local.env"))
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--skip-failed", action="store_true")
    parser.add_argument("--allow-http-models", action="store_true")
    args = parser.parse_args(argv)
    source = args.source_batch.resolve()
    original = read(source / "state.json")
    try:
        cutoffs = sorted(set(int(k) for k in args.cutoffs.split(",")))
    except ValueError:
        parser.error("cutoffs must be comma-separated integers")
    if not cutoffs or any(k < 1 or k >= original["top_k"] for k in cutoffs):
        parser.error("each cutoff must be >=1 and smaller than the source top_k")
    if args.skip_failed and not args.resume:
        parser.error("skip-failed requires resume")
    if original["mode"] != args.mode:
        parser.error("mode must match source batch for comparable results")
    pipeline = official()
    count = len(original["selection"]["rows"])
    available = sum((source / "questions" / r["id"] / "search.json").exists()
                    for r in original["selection"]["rows"])
    print(f"PLAN: {count} questions; cutoffs={cutoffs}; cached Search={available}/{count}; "
          f"original top-{original['top_k']} reused; no Add/Search/embedding calls")
    calls = 3 if args.mode == "both" else 2 if args.mode == "generative" else 1
    print(f"At most {count * len(cutoffs) * calls} new Answer/Judge calls before resume; "
          f"MCQ format={args.mcq_format}; strict and normalized scores reported separately")
    if not args.run:
        return 0
    if not Path(args.env_file).is_file():
        raise ValueError("Environment file does not exist")
    load_dotenv(args.env_file, override=True)
    env = dict(os.environ)
    if base.public_models(env) != original["models"]:
        raise ValueError("Answer/Judge models must match the source batch")
    for role in ("answer", "judge"):
        url = urlsplit(base.model_config(env, role)["base_url"])
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query:
            raise ValueError("Invalid model endpoint for " + role)
        if url.scheme == "http" and not args.allow_http_models:
            raise ValueError("HTTP model requires -AllowHttpModels")
    contract = {"source": str(source), "source_fingerprint": original["fingerprint"],
                "selection": original["selection"], "models": original["models"], "cutoffs": cutoffs,
                "mode": args.mode, "mcq_format": args.mcq_format,
                "code": {str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__), Path(base.__file__),
                         ROOT / "aml_adapter/persona_data.py", VENDOR / "data/personamem/pipeline_v2.py",
                         VENDOR / "api_config.py"]}}
    signature = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output = (args.resume or ROOT / f"outputs/personamem-cutoffs/{original['profile']}-{stamp}-{uuid4().hex[:8]}").resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Supplement output must be separate from the source batch")
    if args.resume and not (output / "state.json").exists():
        raise ValueError("Resume directory has no state.json")
    output.mkdir(parents=True, exist_ok=True)
    lock = ProcessLock(output / "runner.lock")
    state = None
    try:
        if args.resume:
            previous = read(output / "state.json")
            if previous["fingerprint"] != signature:
                raise ValueError("Source/config/code changed; cannot mix evaluation settings")
            state = previous
        else:
            state = {"fingerprint": signature, "contract": contract, "selection": original["selection"],
                     "profile": original["profile"], "cutoffs": cutoffs, "source_top_k": original["top_k"],
                     "mcq_format": args.mcq_format, "questions": {
                         r["id"]: {"persona_id": r["persona_id"], "pref_type": r["pref_type"],
                                   "cutoffs": {str(k): {"status": "pending_source"} for k in cutoffs}}
                         for r in original["selection"]["rows"]}}
        print("Batch directory: " + str(output), flush=True)
        save(output / "state.json", state)
        run(state, source, output, args, env, pipeline, DiagnosticLog(output / "diagnostics.jsonl"))
        return int(any(r["status"] != "completed" for q in state["questions"].values() for r in q["cutoffs"].values()))
    except KeyboardInterrupt:
        print("Interrupted; resume reuses completed cutoffs and cached answers.", flush=True)
        return 130
    finally:
        if state:
            for question in state["questions"].values():
                for record in question["cutoffs"].values():
                    if record["status"] == "running":
                        record["status"] = "interrupted"
            save(output / "state.json", state)
            summarize(state, output)
            print("Summary: " + str(output / "summary.md"), flush=True)
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())

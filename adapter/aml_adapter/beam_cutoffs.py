"""Evaluate BEAM Top-K answer quality from immutable cached Top-100 Search results."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import httpx
from dotenv import load_dotenv

from .beam_batch import (answer_prompt, event_metrics, model_config, model_stage,
                         public_models, stats, validate_models)
from .beam_data import ROOT, VENDOR_SHA, object_digest, official
from .diagnostics import DiagnosticLog, exception_info
from .lme_smoke import model_transport_overrides
from .models import SearchResponse
from .persona_data import digest, save
from .state import ProcessLock


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def has_status(error, status):
    return any(item.get("status_code") == status for item in exception_info(error)["chain"])


def judge(pipeline, question, answer, artifact, env, diagnostic):
    prompt = pipeline.render_batch_judge_prompt(question["question"], answer, question["rubrics"])
    save(artifact / "judge-prompt.json", {"prompt": prompt})
    for attempt in range(2):
        print(("JUDGE: " if attempt == 0 else "JUDGE FORMAT RETRY: ")
              + question["id"], flush=True)
        raw = asyncio.run(model_stage(
            pipeline, "judge", artifact, diagnostic, model_config(env, "judge"),
            [{"role": "user", "content": prompt}],
            1024 if attempt == 0 else max(2048, 512 * len(question["rubrics"])), True))
        try:
            scores = pipeline.parse_rubric_scores(raw, len(question["rubrics"]))
            break
        except (json.JSONDecodeError, ValueError):
            save(artifact / f"judge-unparsed-{attempt+1}.json", {"response": raw})
            if attempt == 1:
                raise
    result = {
        "question_type": question["question_type"],
        "rubric_scores": [{"rubric": rubric, **score}
                          for rubric, score in zip(question["rubrics"], scores)],
        "llm_judge_score": sum(score["score"] for score in scores) / len(scores),
        "judge_response": raw,
    }
    if question["question_type"] == "event_ordering":
        result["event_ordering"] = asyncio.run(event_metrics(
            pipeline, artifact, diagnostic, model_config(env, "judge"),
            question["rubrics"], answer))
    return result


def evaluate(question, record, retrieved, cutoff, artifact, pipeline, env, diagnostic, checkpoint):
    artifact.mkdir(parents=True, exist_ok=True)
    selected = retrieved[:cutoff]
    record.update(retrieved=len(selected), context_words=sum(len(row["content"].split()) for row in selected))
    answer_file = artifact / "answer.json"
    if not answer_file.exists():
        record["stage"] = "answer"
        checkpoint()
        prompt = answer_prompt(pipeline, question["question"], selected)
        save(artifact / "answer-prompt.json", {"prompt": prompt})
        print(f"ANSWER top-{cutoff}: {question['id']}", flush=True)
        answer = asyncio.run(model_stage(
            pipeline, "answer", artifact, diagnostic, model_config(env, "answer"),
            [{"role": "user", "content": prompt}], 512))
        save(answer_file, {"generated_answer": answer})
    answer = read(answer_file)["generated_answer"]
    judge_file = artifact / "judge.json"
    if not judge_file.exists():
        record["stage"] = "judge"
        checkpoint()
        save(judge_file, judge(pipeline, question, answer, artifact, env, diagnostic))
    result = read(judge_file)
    record.update(score=result["llm_judge_score"], status="completed", stage="done")
    if "event_ordering" in result:
        record["event_ordering_score"] = result["event_ordering"]["final_score"]
    record.pop("error", None)


def grouped_stats(state, ids):
    output = {}
    for cutoff in state["cutoffs"]:
        output["top-" + str(cutoff)] = stats(
            state["questions"][qid]["cutoffs"][str(cutoff)] for qid in ids)
    output["original-top-" + str(state["source_top_k"])] = stats(
        state["questions"][qid]["original"] for qid in ids)
    return output


def summarize(state, output):
    ids = list(state["questions"])
    groups = ["top-" + str(cutoff) for cutoff in state["cutoffs"]]
    complete = [qid for qid in ids if all(
        state["questions"][qid]["cutoffs"][str(cutoff)]["status"] == "completed"
        for cutoff in state["cutoffs"])]
    report = {
        "profile": state["profile"], "source_batch": state["source_batch"],
        "cutoff_method": "prefix of immutable cached Top-100 Search response",
        "all": grouped_stats(state, ids), "paired_cutoff_ids": complete,
        "paired_cutoffs": grouped_stats(state, complete), "by_type": {}, "by_conversation": {},
    }
    for question_type in sorted({q["question_type"] for q in state["questions"].values()}):
        selected = [qid for qid, q in state["questions"].items() if q["question_type"] == question_type]
        report["by_type"][question_type] = grouped_stats(state, selected)
    for conversation_id in sorted({q["conversation_id"] for q in state["questions"].values()}):
        selected = [qid for qid, q in state["questions"].items()
                    if q["conversation_id"] == conversation_id]
        report["by_conversation"][conversation_id] = grouped_stats(state, selected)
    report["failures"] = [
        {"id": qid, "cutoff": cutoff, "stage": q["cutoffs"][str(cutoff)].get("stage")}
        for qid, q in state["questions"].items() for cutoff in state["cutoffs"]
        if q["cutoffs"][str(cutoff)]["status"] == "failed"
    ]
    save(output / "summary.json", report)
    lines = [f"# BEAM cached cutoff evaluation — {state['profile']}", "",
             "Local cutoff diagnostic; not an official competition score.",
             "Top-10/20/50 are prefixes of the preserved Top-100 Search response; no Add/Search was rerun.", "",
             "| Context | Completed | Failed | Mean nugget | Perfect | Event ordering | Avg context words |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, values in report["all"].items():
        records = ([state["questions"][qid]["original"] for qid in ids]
                   if name.startswith("original") else
                   [state["questions"][qid]["cutoffs"][name.removeprefix("top-")] for qid in ids])
        completed = [row for row in records if row["status"] == "completed"]
        average_words = (sum(row.get("context_words", 0) for row in completed) / len(completed)
                         if completed else None)
        lines.append(f"| {name} | {values['completed']}/{values['total']} | {values['failed']} | "
                     f"{values['mean_nugget_score']} | {values['perfect_questions']} | "
                     f"{values['event_ordering_mean']} | {average_words} |")
    lines += ["", f"Questions complete at every new cutoff: {len(complete)}/{len(ids)}.",
              "Per-type and per-conversation details: summary.json.", "",
              "Failed question/cutoff pairs: " + (", ".join(
                  f"{row['id']}@{row['cutoff']}" for row in report["failures"]) or "none")]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(state, source, output, args, env, pipeline, diagnostic):
    def checkpoint():
        save(output / "state.json", state)

    selection = {row["id"]: row for row in state["selection"]["questions"]}
    for qid, question_state in state["questions"].items():
        search_file = source / "questions" / qid / "search.json"
        if not search_file.is_file() or digest(search_file) != question_state["source_search_sha256"]:
            raise ValueError("Cached source Search is missing or changed: " + qid)
        search = read(search_file)
        SearchResponse.model_validate(search)
        retrieved = search["data"]
        if len(retrieved) < max(state["cutoffs"]):
            raise ValueError("Cached Search has fewer rows than the largest cutoff: " + qid)
        for cutoff in state["cutoffs"]:
            record = question_state["cutoffs"][str(cutoff)]
            if record["status"] == "completed" or (args.skip_failed and record["status"] == "failed"):
                continue
            record["status"] = "running"
            checkpoint()
            artifact = output / "questions" / qid / f"top-{cutoff}"
            try:
                evaluate(selection[qid], record, retrieved, cutoff, artifact,
                         pipeline, env, diagnostic, checkpoint)
                print(f"DONE top-{cutoff}: {qid}; score={record['score']:.4f}", flush=True)
            except Exception as error:
                details = exception_info(error)
                if has_status(error, 429):
                    record.update(status="interrupted", error=details)
                    checkpoint()
                    print(f"PAUSED top-{cutoff}: {qid}; persistent model HTTP 429", flush=True)
                    raise
                record.update(status="failed", error=details)
                diagnostic.emit(record.get("stage", "qa"), "failed", question_id=qid,
                                cutoff=cutoff, error=details)
                print(f"FAILED top-{cutoff}: {qid}; {type(error).__name__}; continuing", flush=True)
            checkpoint()
        summarize(state, output)


def compare(left_path, right_path):
    left_path, right_path = Path(left_path).resolve(), Path(right_path).resolve()
    left, right = read(left_path / "state.json"), read(right_path / "state.json")
    for field in ("selection", "cutoffs", "source_top_k", "models"):
        if left[field] != right[field]:
            raise ValueError("Cannot compare different " + field)
    report = {"profiles": [left["profile"], right["profile"]], "cutoffs": {}}
    for cutoff in left["cutoffs"]:
        key = str(cutoff)
        ids = [qid for qid in left["questions"] if
               left["questions"][qid]["cutoffs"][key]["status"] == "completed" and
               right["questions"][qid]["cutoffs"][key]["status"] == "completed"]
        pairs = [(left["questions"][qid]["cutoffs"][key]["score"],
                  right["questions"][qid]["cutoffs"][key]["score"]) for qid in ids]
        report["cutoffs"][key] = {
            "paired": len(ids),
            "ordinary_mean": sum(a for a, _ in pairs) / len(pairs) if pairs else None,
            "schema_mean": sum(b for _, b in pairs) / len(pairs) if pairs else None,
            "ordinary_wins": sum(a > b for a, b in pairs),
            "schema_wins": sum(b > a for a, b in pairs),
            "ties": sum(a == b for a, b in pairs),
        }
    save(right_path / "cutoff-comparison.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-batch", type=Path)
    parser.add_argument("--cutoffs", default="10,20,50")
    parser.add_argument("--env-file", default=str(ROOT / "config/longmemeval.local.env"))
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--skip-failed", action="store_true")
    parser.add_argument("--allow-http-models", action="store_true")
    parser.add_argument("--compare", nargs=2)
    args = parser.parse_args(argv)
    if args.compare:
        compare(*args.compare)
        return 0
    if not args.source_batch:
        parser.error("source-batch is required")
    source = args.source_batch.resolve()
    original = read(source / "state.json")
    try:
        cutoffs = sorted(set(int(value) for value in args.cutoffs.split(",")))
    except ValueError:
        parser.error("cutoffs must be comma-separated integers")
    if not cutoffs or any(value < 1 or value >= original["top_k"] for value in cutoffs):
        parser.error("each cutoff must be >=1 and smaller than source top_k")
    if args.skip_failed and not args.resume:
        parser.error("skip-failed requires resume")
    available = sum((source / "questions" / row["id"] / "search.json").is_file()
                    for row in original["selection"]["questions"])
    print(f"PLAN: BEAM cached cutoffs {cutoffs}; questions={len(original['questions'])}; "
          f"Search files={available}/{len(original['questions'])}; no Add/Search/embedding calls")
    print(f"At least {len(original['questions']) * len(cutoffs) * 2} Answer/Judge calls, "
          "plus official event-ordering alignment calls; enable_thinking=false")
    if not args.run:
        return 0
    if not Path(args.env_file).is_file():
        raise ValueError("Environment file does not exist")
    load_dotenv(args.env_file, override=True)
    env = dict(os.environ)
    model_transport_overrides(env, args.allow_http_models)
    validate_models(env, args.allow_http_models)
    if public_models(env) != original["models"]:
        raise ValueError("Answer/Judge models must match the source batch")
    code_paths = [Path(__file__), ROOT / "aml_adapter/beam_batch.py",
                  ROOT / "aml_adapter/beam_data.py", ROOT / "vendor/aml_beam/data/beam/pipeline.py"]
    contract = {"source": str(source), "source_fingerprint": original["fingerprint"],
                "selection": original["selection"], "models": original["models"],
                "cutoffs": cutoffs, "vendor": VENDOR_SHA,
                "code": {str(path.relative_to(ROOT)): digest(path) for path in code_paths}}
    signature = object_digest(contract)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output = (args.resume or ROOT / f"outputs/beam-cutoffs/{original['profile']}-{stamp}-{uuid4().hex[:8]}").resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Cutoff output must be separate from the source batch")
    if args.resume and not (output / "state.json").is_file():
        raise ValueError("Resume directory has no state.json")
    output.mkdir(parents=True, exist_ok=True)
    lock = ProcessLock(output / "runner.lock")
    state = None
    try:
        if args.resume:
            state = read(output / "state.json")
            if state["fingerprint"] != signature:
                raise ValueError("Source/config/code changed; cannot mix cutoff results")
        else:
            state = {"fingerprint": signature, "profile": original["profile"],
                     "source_batch": str(source), "selection": original["selection"],
                     "models": original["models"], "cutoffs": cutoffs,
                     "source_top_k": original["top_k"], "questions": {}}
            for question in original["selection"]["questions"]:
                qid = question["id"]
                search_file = source / "questions" / qid / "search.json"
                if not search_file.is_file():
                    raise ValueError("Missing cached Search: " + qid)
                state["questions"][qid] = {
                    "question_type": question["question_type"],
                    "conversation_id": question["conversation_id"],
                    "source_search_sha256": digest(search_file),
                    "original": dict(original["questions"][qid]),
                    "cutoffs": {str(value): {"status": "pending", "stage": "pending"}
                                for value in cutoffs},
                }
        save(output / "state.json", state)
        print("Batch directory: " + str(output), flush=True)
        run(state, source, output, args, env, official(), DiagnosticLog(output / "diagnostics.jsonl"))
        return int(any(record["status"] != "completed" for question in state["questions"].values()
                       for record in question["cutoffs"].values()))
    except httpx.HTTPStatusError as error:
        if error.response.status_code == 429:
            print("Batch paused by persistent model HTTP 429; resume later.", flush=True)
            return 75
        raise
    except KeyboardInterrupt:
        print("Interrupted; resume keeps all completed cutoff Answer/Judge artifacts.", flush=True)
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

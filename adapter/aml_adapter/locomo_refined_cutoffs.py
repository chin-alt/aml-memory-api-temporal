"""Evaluate LoCoMo-Refined Top-K answer quality from cached Top-100 Search results."""

from __future__ import annotations

import argparse
import json
import os
import threading
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import httpx
from dotenv import load_dotenv

from .diagnostics import DiagnosticLog, exception_info
from .locomo_refined_batch import (
    RunCancelled,
    _load_official_evaluator,
    _official_refined_prompt,
    _score_with_official_logic,
    answer_prompt,
    model_config,
    public_models,
    validate_models,
)
from .locomo_refined_data import CATEGORY_NAMES, ROOT, object_digest, provenance
from .lme_smoke import model_transport_overrides
from .models import SearchResponse
from .persona_data import digest, save
from .state import ProcessLock


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_model_name(value: str) -> str:
    return "".join(character for character in value.rsplit("/", 1)[-1].lower()
                   if character.isalnum())


def validate_source_model_compatibility(runtime: dict, original: dict) -> None:
    # The answer generator must be identical. The official Qwen3-14B judge was
    # moved from the company gateway to SiliconFlow during the long source run;
    # provider prefixes and endpoints may differ while the model itself remains
    # the same.
    if runtime["answer"] != original["answer"]:
        raise ValueError("Answer model/endpoint must match the source batch")
    if normalized_model_name(runtime["judge"]["model"]) != normalized_model_name(
            original["judge"]["model"]):
        raise ValueError("Judge model must match the source batch")


def completed_source_ids(source: Path, state: dict) -> set[str]:
    return {
        qid for qid, record in state["questions"].items()
        if record.get("status") == "completed"
        and (source / "questions" / qid / "search.json").is_file()
        and (source / "questions" / qid / "official-score.json").is_file()
    }


def stats(records) -> dict:
    rows = list(records)
    completed = [row for row in rows if row.get("status") == "completed"]
    count = len(completed)
    return {
        "total": len(rows),
        "completed": count,
        "failed": sum(row.get("status") == "failed" for row in rows),
        "accuracy": sum(float(row["llm_score"]) for row in completed) / count if count else None,
        "correct": sum(float(row["llm_score"]) >= .5 for row in completed),
        "mean_f1": sum(float(row["f1_score"]) for row in completed) / count if count else None,
        "mean_bleu": sum(float(row["bleu_score"]) for row in completed) / count if count else None,
        "mean_context_words": (
            sum(int(row.get("context_words", 0)) for row in completed) / count if count else None
        ),
    }


def grouped_stats(state: dict, qids: list[str]) -> dict:
    output = {}
    for cutoff in state["cutoffs"]:
        output[f"top-{cutoff}"] = stats(
            state["questions"][qid]["cutoffs"][str(cutoff)] for qid in qids
        )
    output[f"original-top-{state['source_top_k']}"] = stats(
        state["questions"][qid]["original"] for qid in qids
    )
    return output


def summarize(state: dict, output: Path) -> None:
    qids = list(state["questions"])
    paired = [qid for qid in qids if all(
        state["questions"][qid]["cutoffs"][str(cutoff)].get("status") == "completed"
        for cutoff in state["cutoffs"]
    )]
    report = {
        "profile": state["profile"],
        "source_batch": state["source_batch"],
        "matched_source_batch": state.get("matched_source_batch"),
        "cutoff_method": "prefix of immutable cached Top-100 Search response",
        "all": grouped_stats(state, qids),
        "paired_cutoff_ids": paired,
        "paired_cutoffs": grouped_stats(state, paired),
        "by_category": {},
        "by_conversation": {},
    }
    for category in sorted({row["category"] for row in state["questions"].values()}):
        selected = [qid for qid, row in state["questions"].items() if row["category"] == category]
        report["by_category"][category] = grouped_stats(state, selected)
    for conversation in sorted({row["sample_id"] for row in state["questions"].values()}):
        selected = [qid for qid, row in state["questions"].items()
                    if row["sample_id"] == conversation]
        report["by_conversation"][conversation] = grouped_stats(state, selected)
    report["failures"] = [
        {"qa_id": qid, "cutoff": cutoff, "stage": row["cutoffs"][str(cutoff)].get("stage")}
        for qid, row in state["questions"].items() for cutoff in state["cutoffs"]
        if row["cutoffs"][str(cutoff)].get("status") == "failed"
    ]
    save(output / "summary.json", report)

    lines = [
        f"# LoCoMo-Refined cached cutoff evaluation — {state['profile']}", "",
        "Top-10/20/50 are prefixes of the preserved Top-100 Search response; no Add, Search, extraction, or embedding was rerun.", "",
        "| Context | Completed | Failed | Correct | Accuracy | F1 | BLEU | Avg context words |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in report["all"].items():
        def fmt(value):
            return "—" if value is None else f"{value:.4f}"
        lines.append(
            f"| {name} | {values['completed']}/{values['total']} | {values['failed']} | "
            f"{values['correct']} | {fmt(values['accuracy'])} | {fmt(values['mean_f1'])} | "
            f"{fmt(values['mean_bleu'])} | {fmt(values['mean_context_words'])} |"
        )
    lines += [
        "", f"Questions complete at every new cutoff: {len(paired)}/{len(qids)}.",
        "Per-category, per-conversation, and paired-only results are in summary.json.", "",
        "Failed question/cutoff pairs: " + (", ".join(
            f"{row['qa_id']}@{row['cutoff']}" for row in report["failures"]
        ) or "none"),
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for cutoff in state["cutoffs"]:
        rows = []
        for qid in qids:
            record = state["questions"][qid]["cutoffs"][str(cutoff)]
            score_file = output / "questions" / qid / f"top-{cutoff}" / "official-score.json"
            if record.get("status") == "completed" and score_file.is_file():
                rows.append(score_file.read_text(encoding="utf-8").strip())
        (output / f"official-scored-top-{cutoff}.jsonl").write_text(
            ("\n".join(rows) + "\n") if rows else "", encoding="utf-8"
        )


def evaluate(question: dict, record: dict, retrieved: list[dict], cutoff: int,
             artifact: Path, evaluator, prompt_template: str, env: dict,
             diagnostic: DiagnosticLog, checkpoint, cancelled: threading.Event) -> None:
    artifact.mkdir(parents=True, exist_ok=True)
    selected = retrieved[:cutoff]
    record.update(
        status="running", stage="answer", retrieved=len(selected),
        context_words=sum(len(row["content"].split()) for row in selected), error=None,
    )
    checkpoint()
    answer_file = artifact / "answer.json"
    if not answer_file.is_file():
        from .locomo_refined_batch import _chat
        messages = answer_prompt(question["question"], selected)
        save(artifact / "answer-prompt.json", messages)
        print(f"ANSWER top-{cutoff}: {question['qa_id']}", flush=True)
        answer, body = _chat(model_config(env, "answer"), messages, 512, 6,
                             diagnostic, "answer.chat", cancelled)
        save(answer_file, {"generated_answer": answer, "response": body})
    answer = read(answer_file)["generated_answer"]
    record["stage"] = "judge"
    checkpoint()
    score_file = artifact / "official-score.json"
    if not score_file.is_file():
        scored = _score_with_official_logic(
            question, answer, evaluator, prompt_template, env, artifact, diagnostic, cancelled
        )
        save(score_file, scored)
    scored = read(score_file)
    record.update(
        status="completed", stage="done", llm_score=scored["llm_score"],
        f1_score=scored["f1_score"], bleu_score=scored["bleu_score"], error=None,
    )


def run(state: dict, source: Path, output: Path, args, env: dict,
        diagnostic: DiagnosticLog) -> None:
    questions = {row["qa_id"]: row for row in state["selection"]}
    state_lock = threading.RLock()

    def checkpoint():
        with state_lock:
            save(output / "state.json", state)

    evaluator = _load_official_evaluator()
    prompt_template = _official_refined_prompt()
    cancelled = threading.Event()

    def worker(item):
        qid, cutoff = item
        if cancelled.is_set():
            return
        row = state["questions"][qid]
        record = row["cutoffs"][str(cutoff)]
        if record.get("status") == "completed" or (args.skip_failed and record.get("status") == "failed"):
            return
        search_file = source / "questions" / qid / "search.json"
        if digest(search_file) != row["source_search_sha256"]:
            raise ValueError("Cached Search changed: " + qid)
        response = read(search_file)
        SearchResponse.model_validate(response)
        retrieved = response["data"]
        if len(retrieved) < cutoff:
            raise ValueError(f"Cached Search has fewer than {cutoff} rows: {qid}")
        artifact = output / "questions" / qid / f"top-{cutoff}"
        try:
            evaluate(questions[qid], record, retrieved, cutoff, artifact, evaluator,
                     prompt_template, env, diagnostic, checkpoint, cancelled)
            print(f"DONE top-{cutoff}: {qid}; llm={record['llm_score']}", flush=True)
        except RunCancelled:
            record.update(status="interrupted", error=None)
            checkpoint()
            return
        except Exception as error:
            details = exception_info(error)
            record.update(status="failed", error=details)
            diagnostic.emit(record.get("stage", "qa"), "failed", qa_id=qid,
                            cutoff=cutoff, error=details)
            print(f"FAILED top-{cutoff}: {qid}; {type(error).__name__}; continuing", flush=True)
        checkpoint()

    tasks = [
        (qid, cutoff) for qid, row in state["questions"].items()
        for cutoff in state["cutoffs"]
        if row["cutoffs"][str(cutoff)].get("status") != "completed"
        and not (args.skip_failed and row["cutoffs"][str(cutoff)].get("status") == "failed")
    ]
    pool = ThreadPoolExecutor(max_workers=args.qa_concurrency)
    futures = [pool.submit(worker, item) for item in tasks]
    interrupted = False
    try:
        pending = set(futures)
        while pending:
            done, pending = wait(pending, timeout=.2, return_when=FIRST_COMPLETED)
            for future in done:
                future.result()
    except KeyboardInterrupt:
        interrupted = True
        cancelled.set()
        for future in futures:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    finally:
        if not interrupted:
            pool.shutdown(wait=True, cancel_futures=True)
    summarize(state, output)


def compare(left_path: Path, right_path: Path) -> None:
    left_path, right_path = left_path.resolve(), right_path.resolve()
    left, right = read(left_path / "state.json"), read(right_path / "state.json")
    if left["cutoffs"] != right["cutoffs"]:
        raise ValueError("Cutoffs differ")
    report = {"profiles": [left["profile"], right["profile"]], "cutoffs": {}}
    for cutoff in left["cutoffs"]:
        key = str(cutoff)
        qids = sorted(set(left["questions"]) & set(right["questions"]))
        qids = [qid for qid in qids
                if left["questions"][qid]["cutoffs"][key].get("status") == "completed"
                and right["questions"][qid]["cutoffs"][key].get("status") == "completed"]
        pairs = [(float(left["questions"][qid]["cutoffs"][key]["llm_score"]),
                  float(right["questions"][qid]["cutoffs"][key]["llm_score"])) for qid in qids]
        report["cutoffs"][key] = {
            "paired": len(pairs),
            "ordinary_accuracy": sum(a for a, _ in pairs) / len(pairs) if pairs else None,
            "schema_accuracy": sum(b for _, b in pairs) / len(pairs) if pairs else None,
            "ordinary_to_schema": sum(a < b for a, b in pairs),
            "schema_to_ordinary": sum(a > b for a, b in pairs),
            "ties": sum(a == b for a, b in pairs),
        }
    save(right_path / "cutoff-comparison.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-batch", type=Path)
    parser.add_argument("--match-source-batch", type=Path)
    parser.add_argument("--cutoffs", default="10,20,50")
    parser.add_argument("--qa-concurrency", type=int, default=3)
    parser.add_argument("--env-file", default=str(ROOT / "config/longmemeval.local.env"))
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--skip-failed", action="store_true")
    parser.add_argument("--allow-http-models", action="store_true")
    parser.add_argument("--compare", nargs=2, type=Path)
    args = parser.parse_args(argv)
    if args.compare:
        compare(*args.compare)
        return 0
    if not args.source_batch:
        parser.error("source-batch is required")
    if not 1 <= args.qa_concurrency <= 32:
        parser.error("qa-concurrency must be 1..32")
    source = args.source_batch.resolve()
    original = read(source / "state.json")
    try:
        cutoffs = sorted(set(int(value) for value in args.cutoffs.split(",")))
    except ValueError:
        parser.error("cutoffs must be comma-separated integers")
    if not cutoffs or any(value < 1 or value >= int(original["top_k"]) for value in cutoffs):
        parser.error("each cutoff must be >=1 and smaller than source top_k")
    eligible = completed_source_ids(source, original)
    matched_source = None
    if args.match_source_batch:
        matched_source = args.match_source_batch.resolve()
        matched_state = read(matched_source / "state.json")
        eligible &= completed_source_ids(matched_source, matched_state)
    selection = [row for row in original["selection"]["questions"] if row["qa_id"] in eligible]
    print(f"PLAN: LoCoMo-Refined cached cutoffs {cutoffs}; questions={len(selection)}; "
          f"QA concurrency={args.qa_concurrency}")
    print("No Add/Search/extraction/embedding calls; each cutoff reuses a prefix of cached Top-100.")
    print(f"Up to {len(selection) * len(cutoffs) * 2} Answer/Judge calls before candidate/retry expansion; "
          "enable_thinking=false.")
    if matched_source:
        print("Paired question scope matched to: " + str(matched_source))
    if not args.run:
        return 0
    if not selection:
        raise ValueError("No completed cached questions are eligible")
    if not Path(args.env_file).is_file():
        raise ValueError("Environment file does not exist")
    load_dotenv(args.env_file, override=True)
    env = dict(os.environ)
    model_transport_overrides(env, args.allow_http_models)
    validate_models(env, args.allow_http_models)
    runtime_models = public_models(env)
    validate_source_model_compatibility(runtime_models, original["models"])
    search_digests = {
        row["qa_id"]: digest(source / "questions" / row["qa_id"] / "search.json")
        for row in selection
    }
    contract = {
        "source_batch": str(source), "source_fingerprint": original["fingerprint"],
        "matched_source": str(matched_source) if matched_source else None,
        "selection": selection, "models": runtime_models, "cutoffs": cutoffs,
        "search_digests": search_digests, "source_provenance": provenance(),
    }
    signature = object_digest(contract)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output = (args.resume or ROOT / f"outputs/locomo-refined-cutoffs/{original['profile']}-{stamp}-{uuid4().hex[:8]}").resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = ProcessLock(output / "runner.lock")
    state = None
    try:
        if args.resume:
            state = read(output / "state.json")
            if state["fingerprint"] != signature:
                raise ValueError("Source/config/code changed; cannot mix cutoff results")
        else:
            state = {
                "fingerprint": signature, "profile": original["profile"],
                "source_batch": str(source),
                "matched_source_batch": str(matched_source) if matched_source else None,
                "selection": selection, "models": runtime_models,
                "source_top_k": original["top_k"], "cutoffs": cutoffs, "questions": {},
            }
            for question in selection:
                qid = question["qa_id"]
                state["questions"][qid] = {
                    "category": str(question["category"]), "sample_id": question["sample_id"],
                    "source_search_sha256": search_digests[qid],
                    "original": dict(original["questions"][qid]),
                    "cutoffs": {str(value): {"status": "pending", "stage": "pending"}
                                for value in cutoffs},
                }
        for row in state["questions"].values():
            for record in row["cutoffs"].values():
                if record.get("status") == "running":
                    record["status"] = "interrupted"
        save(output / "state.json", state)
        print("Batch directory: " + str(output), flush=True)
        run(state, source, output, args, env, DiagnosticLog(output / "diagnostics.jsonl"))
        return int(any(record.get("status") != "completed"
                       for row in state["questions"].values() for record in row["cutoffs"].values()))
    except KeyboardInterrupt:
        print("Interrupted; completed cutoff artifacts remain resumable.", flush=True)
        return 130
    finally:
        if state:
            for row in state["questions"].values():
                for record in row["cutoffs"].values():
                    if record.get("status") == "running":
                        record["status"] = "interrupted"
            save(output / "state.json", state)
            summarize(state, output)
            print("Summary: " + str(output / "summary.md"), flush=True)
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())

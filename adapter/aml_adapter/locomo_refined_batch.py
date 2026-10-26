"""LoCoMo-Refined text-only Add/Search evaluation with the official scorer."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import queue
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from dotenv import load_dotenv

from .diagnostics import DiagnosticLog, exception_info
from .lme_profiles import profile_files, repository_root
from .lme_smoke import check_backends, model_transport_overrides
from .locomo_refined_data import (CATEGORY_NAMES, ROOT, VENDOR, add_payloads,
                                  object_digest, prepare, provenance, selected_conversations)
from .models import SearchResponse
from .persona_batch import http_post, memory_server
from .persona_data import digest, save
from .state import ProcessLock
from .vector_recovery import recover_vector_store


# These are the exact contracts produced before the interrupt-only runner upgrade below.
# Data selection, prompts, memory profile and scoring are unchanged.
INTERRUPT_UPGRADE_FINGERPRINTS = {
    "2ab1201b143925924e76b0df1dab7c4b52c579397f589c44a46c24cf7af9fdaf",
    "827b0aac998bb9a9ad3e326e935d3f41e0f969c66e2b02d47d595b83d035ad43",
    "432db77388078bca8e4192cf680cd4f6e592ed595ce1e79101a86054b90e4e1e",
    "2c6e42532cf32014e36d159840c194317b280ccc2e7ea9faaa0aefa476c6c57e",
    "bcf6b7739ab911d7e18dd263767c88b8faa3639f30081eff71da836bdf89e0db",
    "023e7664f2758eb85f8d3a3b10881251a0c0dbe81908636032afa84d037c8a3d",
    # Direct ES -> Milvus recovery was completed under this contract. The
    # follow-up only moves the same official Qwen3-14B judge to an authorized
    # endpoint; data, prompts, answer model and scoring remain unchanged.
    "3888ded25c8ae2d3ad239399b91357e91cd9397b08e5bbdba345d83d629c1f8f",
    # Add an execution-only conversation scope and scoped report. This does not
    # change the frozen selection, memory payloads, prompts, models or scoring.
    "92a62b213e4a61c5b387a9a37a8180c4d9cff61d019e337a01e1157cc71b86ae",
}


class RunCancelled(Exception):
    """Internal signal used to unwind worker threads after Ctrl+C."""


class MemoryTransportAbort(Exception):
    """Stop the batch when a local Add/Search request loses its response."""


def model_config(env: dict, role: str) -> dict:
    prefix = "LR_" + role.upper()
    if role == "judge":
        model = env.get(prefix + "_MODEL") or env.get("EVALUATOR_MODEL") or "qwen3-14b"
        base = (env.get(prefix + "_BASE_URL") or env.get("EVALUATOR_API_BASE")
                or env.get("LME_QA_BASE_URL") or env["AML_LLM_BASE_URL"])
        key = (env.get(prefix + "_API_KEY") or env.get("EVALUATOR_API_KEY")
               or env.get("LME_QA_API_KEY") or env["AML_LLM_API_KEY"])
    else:
        model = env.get(prefix + "_MODEL") or env.get("LME_QA_MODEL") or env["AML_LLM_MODEL"]
        base = env.get(prefix + "_BASE_URL") or env.get("LME_QA_BASE_URL") or env["AML_LLM_BASE_URL"]
        key = env.get(prefix + "_API_KEY") or env.get("LME_QA_API_KEY") or env["AML_LLM_API_KEY"]
    return {"model": model, "base_url": base, "api_key": key}


def public_models(env: dict) -> dict:
    return {role: {k: v for k, v in model_config(env, role).items() if k != "api_key"}
            for role in ("answer", "judge")}


def validate_models(env: dict, allow_http: bool) -> None:
    for role in ("answer", "judge"):
        config = model_config(env, role)
        url = urlsplit(config["base_url"])
        if (url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password
                or url.query or url.fragment or not config["model"] or not config["api_key"]):
            raise ValueError("Invalid model configuration: " + role)
        if url.scheme == "http" and not allow_http:
            raise ValueError("HTTP model endpoints require -AllowHttpModels")


def isolate_storage(env: dict, profile: str) -> None:
    suffix = profile.replace("-", "_")
    env["AML_ES_INDEX"] = env.get("LR_ES_INDEX") or f"aml_locomo_refined_{suffix}_text_v1"
    env["AML_MILVUS_COLLECTION"] = (env.get("LR_MILVUS_COLLECTION")
                                     or f"aml_locomo_refined_{suffix}_vector_v1")


def _load_official_evaluator():
    source = VENDOR / "src/evaluate.py"
    previous_path = list(sys.path)
    module_name = "locomo_refined_official_evaluate"
    sys.path.insert(0, str(source.parent))
    try:
        spec = importlib.util.spec_from_file_location(module_name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    finally:
        sys.path[:] = previous_path


def _official_refined_prompt() -> str:
    tree = ast.parse((VENDOR / "src/llm_judge.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "JUDGE_PROMPTS"
                                                 for t in node.targets):
            return ast.literal_eval(node.value)["refined"]
    raise ValueError("Official refined judge prompt was not found")


def _endpoint(base_url: str) -> str:
    value = base_url.rstrip("/")
    return value if value.endswith("/chat/completions") else value + "/chat/completions"


def _chat(config: dict, messages: list[dict], max_tokens: int, attempts: int,
          diagnostic: DiagnosticLog, stage: str, cancelled=None) -> tuple[str, dict]:
    payload = {"model": config["model"], "messages": messages, "temperature": 0,
               "max_tokens": max_tokens, "enable_thinking": False}
    last_error = None
    for attempt in range(1, attempts + 1):
        if cancelled is not None and cancelled.is_set():
            raise RunCancelled()
        results = queue.Queue(maxsize=1)

        def request():
            try:
                with httpx.Client(timeout=httpx.Timeout(900, connect=10), trust_env=False) as client:
                    with diagnostic.span(stage, attempt=attempt, enable_thinking=False) as event:
                        response = client.post(_endpoint(config["base_url"]), json=payload,
                                               headers={"Authorization": "Bearer " + config["api_key"]})
                        event["status_code"] = response.status_code
                        response.raise_for_status()
                        body = response.json()
                        text = str(body["choices"][0]["message"].get("content") or "").strip()
                        if not text:
                            raise ValueError("Model returned an empty answer")
                        results.put((True, (text, body)))
            except BaseException as error:
                results.put((False, error))

        threading.Thread(target=request, daemon=True).start()
        while True:
            try:
                ok, result = results.get(timeout=.2)
                break
            except queue.Empty:
                if cancelled is not None and cancelled.is_set():
                    raise RunCancelled()
        if ok:
            return result
        error = result
        if not isinstance(error, (httpx.HTTPError, KeyError, ValueError)):
            raise error
        last_error = error
        status = getattr(getattr(error, "response", None), "status_code", None)
        if attempt == attempts or (status is not None and status < 500 and status != 429):
            raise error
        delay = min(60, (20 if status == 429 else 2) * attempt)
        print(f"RETRY {stage}: attempt={attempt}/{attempts}; wait={delay}s; status={status}", flush=True)
        if cancelled is not None:
            if cancelled.wait(delay):
                raise RunCancelled()
        else:
            time.sleep(delay)
    raise last_error


def answer_prompt(question: str, retrieved: list[dict]) -> list[dict]:
    context = "\n\n".join(f"Memory {index + 1}: {item['content']}"
                            for index, item in enumerate(retrieved))
    return [
        {"role": "system", "content": (
            "Answer using only the retrieved conversation memories. Give only the concise final answer. "
            "Include every required fact, do not add unsupported details, and preserve the question's "
            "requested time granularity. If the memories do not support an answer, say that the information "
            "is unavailable.")},
        {"role": "user", "content": f"Retrieved conversation memories:\n{context}\n\nQuestion: {question}"},
    ]


def fingerprint(args, selection: dict, env: dict) -> str:
    paths = [Path(__file__), ROOT / "aml_adapter/locomo_refined_data.py",
             ROOT / "aml_adapter/persona_batch.py", ROOT / "aml_adapter/backend.py",
             ROOT / "aml_adapter/vector_recovery.py",
             VENDOR / "src/evaluate.py", VENDOR / "src/llm_judge.py", VENDOR / "src/bleu_f1.py"]
    paths += profile_files(args.memory_profile)
    paths += sorted((repository_root(args.memory_profile) / "jiuwen_memory").rglob("*.py"))
    settings = {key: env.get(key) for key in
                ("AML_LLM_MODEL", "AML_LLM_PROVIDER", "AML_LLM_BASE_URL", "AML_EMBED_MODEL",
                 "AML_EMBED_BASE_URL", "AML_REDIS_URL", "AML_ES_URL", "AML_ES_INDEX",
                 "AML_MILVUS_URI", "AML_MILVUS_COLLECTION")}
    return object_digest({"selection": selection, "profile": args.memory_profile, "top_k": args.top_k,
                          "models": public_models(env), "settings": settings,
                          "files": {str(path): digest(path) for path in paths}, "source": provenance()})


def _post_retry(client, path, payload, diagnostic, attempts=4, cancelled=None):
    last = None
    for attempt in range(1, attempts + 1):
        if cancelled is not None and cancelled.is_set():
            raise RunCancelled()
        try:
            return http_post(client, path, payload, diagnostic)
        except httpx.HTTPStatusError as error:
            last = error
            status = error.response.status_code
            if attempt == attempts or (status < 500 and status != 429):
                raise
            delay = min(60, attempt * (15 if status == 429 else 2))
            print(f"RETRY {path}: attempt={attempt}/{attempts}; wait={delay}s; status={status}", flush=True)
            if cancelled is not None:
                if cancelled.wait(delay):
                    raise RunCancelled()
            else:
                time.sleep(delay)
    raise last


def _run_parallel(items, worker, concurrency, client, cancelled):
    """Run worker jobs while making Ctrl+C able to break blocked local HTTP calls."""
    pool = ThreadPoolExecutor(max_workers=concurrency)
    futures = [pool.submit(worker, item) for item in items]
    interrupted = False
    try:
        pending = set(futures)
        while pending:
            # Python 3.11 on Windows can remain inside the condition wait used by
            # as_completed while every worker is blocked in HTTP. Polling gives
            # the main thread regular bytecode boundaries at which SIGINT can be
            # delivered as KeyboardInterrupt.
            done, pending = wait(pending, timeout=.2, return_when=FIRST_COMPLETED)
            for future in done:
                future.result()
    except KeyboardInterrupt:
        interrupted = True
        cancelled.set()
        for future in futures:
            future.cancel()
        # Requests run in executor threads. Stopping our local server closes their
        # sockets, allowing shutdown(wait=True) to finish instead of hanging.
        stop = getattr(client, "stop_owned_server", None)
        if stop is not None:
            stop()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    except MemoryTransportAbort:
        interrupted = True
        cancelled.set()
        for future in futures:
            future.cancel()
        # A client timeout does not cancel the synchronous server-side Add. Stop
        # the owned API so it cannot keep consuming model calls after the batch
        # has already recorded a transport failure.
        stop = getattr(client, "stop_owned_server", None)
        if stop is not None:
            stop()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    finally:
        if not interrupted:
            pool.shutdown(wait=True, cancel_futures=True)


def _question_groups(selection: dict, state: dict,
                     conversation_ids: list[str] | None = None) -> list[list[dict]]:
    """Keep one Search active per conversation while retaining cross-conversation concurrency."""
    groups = []
    for ident in conversation_ids or selection["conversation_ids"]:
        questions = [question for question in selection["questions"]
                     if question["sample_id"] == ident
                     and state["conversations"][ident]["status"] == "completed"
                     and state["questions"][question["qa_id"]]["status"] != "completed"]
        if questions:
            groups.append(questions)
    return groups


def _judge_label(text: str) -> int:
    stripped = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.IGNORECASE).strip()
    match = re.search(r"\{.*?\}", stripped, flags=re.DOTALL)
    if not match:
        raise ValueError("Judge response did not contain JSON")
    label = str(json.loads(match.group(0)).get("label", "")).strip().upper()
    if label not in {"CORRECT", "WRONG"}:
        raise ValueError("Judge response label must be CORRECT or WRONG")
    return int(label == "CORRECT")


def _score_with_official_logic(question, answer, evaluator, prompt_template, env, artifact, diagnostic,
                               cancelled=None):
    """Mirror evaluate.py candidate selection while checkpointing every question locally."""
    prediction = evaluator._normalize_prediction_text(answer)
    candidates = evaluator._normalize_answer_candidates(question.get("answer"), "")
    exact = next((candidate for candidate in candidates if prediction == candidate), None)
    if exact is not None:
        return {**question, "predicted_answer": answer, "response": prediction,
                "matched_answer": exact, "llm_judge": "refined", "success": True,
                "errors": [], "llm_score": 1.0, "llm_reason": "Predicted answer exactly matches the reference answer.",
                "f1_score": 1.0, "bleu_score": 1.0}
    if prediction == "" and any(candidate != "" for candidate in candidates):
        matched = next(candidate for candidate in candidates if candidate != "")
        return {**question, "predicted_answer": answer, "response": prediction,
                "matched_answer": matched, "llm_judge": "refined", "success": True,
                "errors": [], "llm_score": 0.0,
                "llm_reason": "Predicted answer is empty while the reference answer is not.",
                "f1_score": 0.0, "bleu_score": 0.0}
    if not candidates or all(candidate == "" for candidate in candidates):
        return {**question, "predicted_answer": answer, "response": prediction,
                "matched_answer": "", "llm_judge": "refined", "success": True,
                "errors": [], "llm_score": 0.0, "llm_reason": "",
                "f1_score": 0.0, "bleu_score": 0.0}
    candidate_results = []
    for candidate in candidates:
        if cancelled is not None and cancelled.is_set():
            raise RunCancelled()
        prompt = prompt_template.format(question=question["question"], gold_answer=candidate,
                                        generated_answer=prediction)
        response, body = _chat(model_config(env, "judge"), [{"role": "user", "content": prompt}],
                               512, 10, diagnostic, "judge.chat", cancelled)
        save(artifact / ("judge-response-" + uuid4().hex[:10] + ".json"), body)
        candidate_results.append({"answer": candidate, "llm": float(_judge_label(response)),
                                  "reason": response,
                                  "f1": evaluator.compute_token_f1(prediction, candidate),
                                  "bleu": evaluator.compute_bleu1(prediction, candidate)})
    best = max(candidate_results, key=lambda row: (row["llm"], row["f1"], row["bleu"]))
    return {**question, "predicted_answer": answer, "response": prediction,
            "matched_answer": best["answer"], "llm_judge": "refined", "success": True,
            "errors": [], "llm_score": best["llm"], "llm_reason": best["reason"],
            "f1_score": best["f1"], "bleu_score": best["bleu"]}


def run_question(question, record, client, args, env, output, diagnostic, checkpoint, evaluator,
                 prompt_template, cancelled=None):
    artifact = output / "questions" / question["qa_id"]
    artifact.mkdir(parents=True, exist_ok=True)
    record.update(status="running", error=None)
    checkpoint()
    try:
        if cancelled is not None and cancelled.is_set():
            record.update(status="interrupted", error=None)
            checkpoint()
            return
        search_path = artifact / "search.json"
        record["stage"] = "search"
        if not search_path.exists():
            print("SEARCH: " + question["qa_id"], flush=True)
            response = _post_retry(client, "/search", {"query": question["question"],
                                   "user_id": record["user_id"], "top_k": args.top_k}, diagnostic,
                                   cancelled=cancelled)
            SearchResponse.model_validate(response)
            save(search_path, response)
        retrieved = json.loads(search_path.read_text(encoding="utf-8"))["data"]
        record.update(retrieved=len(retrieved),
                      context_words=sum(len(item["content"].split()) for item in retrieved))

        answer_path = artifact / "answer.json"
        record["stage"] = "answer"
        checkpoint()
        if not answer_path.exists():
            messages = answer_prompt(question["question"], retrieved)
            save(artifact / "answer-prompt.json", messages)
            print("ANSWER: " + question["qa_id"], flush=True)
            answer, body = _chat(model_config(env, "answer"), messages, 512, 6,
                                 diagnostic, "answer.chat", cancelled)
            save(answer_path, {"generated_answer": answer, "response": body})
        answer = json.loads(answer_path.read_text(encoding="utf-8"))["generated_answer"]

        record["stage"] = "judge"
        checkpoint()
        scored_path = artifact / "official-score.json"
        if not scored_path.exists():
            scored = _score_with_official_logic(question, answer, evaluator, prompt_template,
                                                env, artifact, diagnostic, cancelled)
            save(scored_path, scored)
        scored = json.loads(scored_path.read_text(encoding="utf-8"))
        record.update(status="completed", stage="done", llm_score=scored["llm_score"],
                      f1_score=scored["f1_score"], bleu_score=scored["bleu_score"])
        record["error"] = None
        print(f"DONE {question['qa_id']}: llm={record['llm_score']}; retrieved={record['retrieved']}", flush=True)
    except Exception as error:
        if cancelled is not None and cancelled.is_set():
            record.update(status="interrupted", error=None)
            checkpoint()
            return
        record.update(status="failed", error=exception_info(error))
        diagnostic.emit(record.get("stage", "unknown"), "failed", question_id=question["qa_id"],
                        error=record["error"])
        print(f"FAILED {question['qa_id']}: stage={record.get('stage')}; {type(error).__name__}; continuing",
              flush=True)
    checkpoint()


def _stats(records) -> dict:
    rows = list(records)
    completed = [row for row in rows if row["status"] == "completed"]
    def mean(field, denominator):
        return sum(float(row.get(field, 0)) for row in completed) / denominator if denominator else None
    return {"total": len(rows), "completed": len(completed),
            "failed": sum(row["status"] == "failed" for row in rows),
            "llm_accuracy_completed": mean("llm_score", len(completed)),
            "llm_accuracy_full_sample": mean("llm_score", len(rows)),
            "f1_completed": mean("f1_score", len(completed)),
            "bleu_completed": mean("bleu_score", len(completed))}


def write_summary(output: Path, state: dict, conversation_ids: list[str] | None = None,
                  stem: str = "summary") -> None:
    conversation_ids = conversation_ids or state["selection"]["conversation_ids"]
    selected = set(conversation_ids)
    records = [row for row in state["questions"].values() if row["sample_id"] in selected]
    summary = _stats(records)
    summary["by_category"] = {
        category: {"name": CATEGORY_NAMES[category],
                   **_stats(row for row in records if row["category"] == category)}
        for category in CATEGORY_NAMES}
    summary["by_conversation"] = {
        ident: _stats(row for row in records if row["sample_id"] == ident)
        for ident in conversation_ids}
    summary["conversation_ids"] = conversation_ids
    summary["text_only"] = True
    summary["official_refined_prompt"] = True
    summary["official_judge_model"] = state["models"]["judge"]["model"].lower().replace("_", "-").endswith("qwen3-14b")
    predictions, scored = [], []
    for question in state["selection"]["questions"]:
        if question["sample_id"] not in selected:
            continue
        artifact = output / "questions" / question["qa_id"]
        answer_path = artifact / "answer.json"
        score_path = artifact / "official-score.json"
        answer = ""
        if answer_path.exists():
            answer = json.loads(answer_path.read_text(encoding="utf-8"))["generated_answer"]
        predictions.append({"qa_id": question["qa_id"], "predicted_answer": answer})
        if score_path.exists():
            scored.append(json.loads(score_path.read_text(encoding="utf-8")))
        else:
            scored.append({**question, "predicted_answer": answer, "response": answer,
                           "llm_judge": "refined", "success": False,
                           "errors": [{"stage": state["questions"][question["qa_id"]].get("stage", "pending"),
                                       "error": "question did not complete"}],
                           "llm_score": 0.0, "f1_score": 0.0, "bleu_score": 0.0})
    suffix = "" if stem == "summary" else "-selected"
    (output / f"predictions{suffix}.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions), encoding="utf-8")
    (output / f"official-scored{suffix}.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in scored), encoding="utf-8")
    save(output / f"{stem}.json", summary)
    lines = [f"# LoCoMo-Refined text-only — {state['profile']}", "",
             f"Completed: {summary['completed']}/{summary['total']}; failed: {summary['failed']}", "",
             f"LLM accuracy (completed): {summary['llm_accuracy_completed']}",
             f"LLM accuracy (failed as zero): {summary['llm_accuracy_full_sample']}", "",
             "| Category | Completed/Total | LLM accuracy (full sample) |", "| --- | ---: | ---: |"]
    for category, item in summary["by_category"].items():
        lines.append(f"| {category} {item['name']} | {item['completed']}/{item['total']} | "
                     f"{item['llm_accuracy_full_sample']} |")
    failed = [key for key, value in state["questions"].items()
              if value["sample_id"] in selected and value["status"] == "failed"]
    lines += ["", "Failed IDs: " + (", ".join(failed) if failed else "none"), ""]
    (output / f"{stem}.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-profile", choices=("ordinary", "schema-entity"), default="ordinary")
    parser.add_argument("--experiment", choices=("smoke16", "pilot120", "full"), default="pilot120")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--write-concurrency", type=int, default=10)
    parser.add_argument("--qa-concurrency", type=int, default=3)
    parser.add_argument("--memory-http-timeout-seconds", type=int, default=3600)
    parser.add_argument("--env-file", default=str(ROOT / "config/longmemeval.local.env"))
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--only-conversations", default="")
    parser.add_argument("--recover-vector-store", action="store_true")
    parser.add_argument("--skip-failed", action="store_true")
    parser.add_argument("--allow-http-models", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.top_k <= 100:
        parser.error("top-k must be 1..100")
    if not 1 <= args.write_concurrency <= 10 or not 1 <= args.qa_concurrency <= 32:
        parser.error("write-concurrency must be 1..10 and qa-concurrency must be 1..32")
    if not 60 <= args.memory_http_timeout_seconds <= 14400:
        parser.error("memory-http-timeout-seconds must be 60..14400")
    if args.skip_failed and not args.resume:
        parser.error("skip-failed requires resume")
    if args.recover_vector_store and not args.resume:
        parser.error("recover-vector-store requires resume")

    selection, selection_path = prepare(args.experiment, args.seed)
    execution_ids = selection["conversation_ids"]
    if args.only_conversations:
        requested = [value.strip() for value in args.only_conversations.split(",") if value.strip()]
        unknown = sorted(set(requested) - set(selection["conversation_ids"]))
        if unknown:
            parser.error("unknown conversation IDs: " + ", ".join(unknown))
        execution_ids = list(dict.fromkeys(requested))
        if not execution_ids:
            parser.error("only-conversations must contain at least one conversation ID")
    sources = selected_conversations(selection)
    add_count = sum(len(list(add_payloads(row, "plan:" + ident))) for ident, row in sources.items())
    categories = {category: sum(str(q["category"]) == category for q in selection["questions"])
                  for category in CATEGORY_NAMES}
    print(f"PLAN: LoCoMo-Refined text-only; {len(selection['conversation_ids'])} conversations; "
          f"{len(selection['questions'])} questions; {add_count} Add requests; top_k={args.top_k}")
    print("Selection: " + str(selection_path))
    print("Categories: " + json.dumps(categories, ensure_ascii=False))
    print("Excluded multimodal questions: 521; image URLs/captions/queries are not written.")
    if args.only_conversations:
        scoped_questions = sum(row["sample_id"] in set(execution_ids)
                               for row in selection["questions"])
        print(f"Execution scope: {','.join(execution_ids)}; {scoped_questions} questions; "
              "other conversations are left untouched.")
    print(f"Concurrency: conversations={args.write_concurrency} (sequential Add within each); "
          f"QA conversations={args.qa_concurrency} (sequential questions within each)")
    print(f"Failure control: local memory HTTP timeout={args.memory_http_timeout_seconds}s; "
          "transport loss stops the batch and preserves resume checkpoints.")
    if not args.run:
        return 0
    if not Path(args.env_file).is_file():
        raise ValueError("Environment file does not exist")
    load_dotenv(args.env_file, override=True)
    env = dict(os.environ)
    env["PM_MEMORY_HTTP_TIMEOUT_SECONDS"] = str(args.memory_http_timeout_seconds)
    model_transport_overrides(env, args.allow_http_models)
    isolate_storage(env, args.memory_profile)
    validate_models(env, args.allow_http_models)
    official_model = model_config(env, "judge")["model"].lower().replace("_", "-").endswith("qwen3-14b")
    print("Judge: official refined prompt; model=" + model_config(env, "judge")["model"]
          + (" (official)" if official_model else " (NON-OFFICIAL model override)"), flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output = (args.resume or ROOT / f"outputs/locomo-refined/{args.memory_profile}-{args.experiment}-{stamp}-{uuid4().hex[:8]}").resolve()
    if args.resume and not (output / "state.json").is_file():
        raise ValueError("Resume directory has no state.json")
    output.mkdir(parents=True, exist_ok=True)
    lock = ProcessLock(output / "runner.lock")
    state = None
    state_lock = threading.RLock()
    try:
        signature = fingerprint(args, selection, env)
        if args.resume:
            state = json.loads((output / "state.json").read_text(encoding="utf-8"))
            if state["profile"] != args.memory_profile:
                raise ValueError("Resume contract changed; use the original profile/configuration")
            if state["fingerprint"] != signature:
                if state["fingerprint"] not in INTERRUPT_UPGRADE_FINGERPRINTS:
                    raise ValueError("Resume contract changed; use the original profile/configuration")
                state.setdefault("runner_upgrades", []).append({
                    "from": state["fingerprint"], "to": signature,
                    "change": "runner recovery/concurrency upgrade; evaluation contract unchanged",
                })
                state["fingerprint"] = signature
                print("RESUME UPGRADE: runner recovery/concurrency support; data, prompts, memory "
                      "and scoring are unchanged.", flush=True)
            if not args.skip_failed:
                for record in state["conversations"].values():
                    if record["status"] == "failed":
                        record["status"] = "pending"
                for record in state["questions"].values():
                    if record["status"] == "failed":
                        record["status"] = "pending"
        else:
            namespace = "locomo-refined-" + uuid4().hex
            state = {"profile": args.memory_profile, "fingerprint": signature, "selection": selection,
                     "namespace": namespace, "top_k": args.top_k, "models": public_models(env),
                     "conversations": {ident: {"status": "pending", "next_add": 0,
                         "total_adds": 0, "error": None,
                         "user_id": namespace + ":" + ident} for ident in selection["conversation_ids"]},
                     "questions": {row["qa_id"]: {"status": "pending", "stage": "pending",
                         "error": None, "retrieved": 0, "context_words": 0,
                         "llm_score": 0.0, "f1_score": 0.0, "bleu_score": 0.0,
                         "category": str(row["category"]), "sample_id": row["sample_id"],
                         "user_id": namespace + ":" + row["sample_id"]}
                                   for row in selection["questions"]}}

        def checkpoint():
            with state_lock:
                save(output / "state.json", state)

        checkpoint()
        print("Batch directory: " + str(output), flush=True)
        check_backends(env)
        if args.recover_vector_store:
            report = recover_vector_store(output, state, args.memory_profile, env)
            state.setdefault("recoveries", []).append(report)
            save(output / "vector-recovery.json", report)
            checkpoint()
            print("VECTOR RECOVERY COMPLETE: "
                  f"clean_users={report['clean_users']}; rows={report['verified_rows']}; "
                  f"dirty_users_skipped={len(report['dirty_users_skipped'])}", flush=True)
        diagnostic = DiagnosticLog(output / "diagnostics.jsonl")
        evaluator = _load_official_evaluator()
        prompt_template = _official_refined_prompt()
        questions_by_conversation = defaultdict(list)
        cancelled = threading.Event()
        for question in selection["questions"]:
            questions_by_conversation[question["sample_id"]].append(question)

        with memory_server(output, args.memory_profile, state["namespace"], env,
                           args.allow_http_models, max_concurrency=max(args.write_concurrency,
                                                                       args.qa_concurrency)) as client:
            def ingest(ident):
                record = state["conversations"][ident]
                if record["status"] == "completed" or (args.skip_failed and record["status"] == "failed"):
                    return
                if cancelled.is_set():
                    record.update(status="interrupted", error=None)
                    checkpoint()
                    return
                record.update(status="running")
                checkpoint()
                try:
                    payloads = list(add_payloads(sources[ident], record["user_id"]))
                    record["total_adds"] = len(payloads)
                    for index in range(record["next_add"], len(payloads)):
                        if cancelled.is_set():
                            record.update(status="interrupted", error=None)
                            checkpoint()
                            return
                        print(f"ADD {ident} {index + 1}/{len(payloads)}", flush=True)
                        response = _post_retry(client, "/add", payloads[index], diagnostic,
                                               cancelled=cancelled)
                        if response.get("success") is not True:
                            raise RuntimeError("Add did not report success")
                        save(output / "conversations" / ident / f"add-{index + 1:03d}.json", response)
                        record["next_add"] = index + 1
                        checkpoint()
                    record.update(status="completed")
                    record["error"] = None
                except Exception as error:
                    if cancelled.is_set():
                        record.update(status="interrupted", error=None)
                        checkpoint()
                        return
                    if isinstance(error, httpx.TransportError):
                        record.update(status="interrupted", error=exception_info(error))
                        diagnostic.emit("add", "transport_abort", conversation_id=ident,
                                        error=record["error"])
                        checkpoint()
                        raise MemoryTransportAbort(
                            f"{ident} lost the local /add response; resume will recover it"
                        ) from error
                    record.update(status="failed", error=exception_info(error))
                    diagnostic.emit("add", "failed", conversation_id=ident, error=record["error"])
                    print(f"FAILED conversation {ident}: {type(error).__name__}; continuing", flush=True)
                    for question in questions_by_conversation[ident]:
                        qrecord = state["questions"][question["qa_id"]]
                        if qrecord["status"] != "completed":
                            qrecord.update(status="failed", stage="add", error=record["error"])
                checkpoint()

            _run_parallel(execution_ids, ingest, min(args.write_concurrency, len(execution_ids)),
                          client, cancelled)

            def qa(question):
                run_question(question, state["questions"][question["qa_id"]], client, args, env, output,
                             diagnostic, checkpoint, evaluator, prompt_template, cancelled)
            groups = _question_groups(selection, state, execution_ids)
            if args.skip_failed:
                groups = [[question for question in group
                           if state["questions"][question["qa_id"]]["status"] != "failed"]
                          for group in groups]
                groups = [group for group in groups if group]
            def qa_group(group):
                for question in group:
                    if cancelled.is_set():
                        return
                    qa(question)
            _run_parallel(groups, qa_group, min(args.qa_concurrency, len(groups)) if groups else 1,
                          client, cancelled)
        scoped_records = (row for row in state["questions"].values()
                          if row["sample_id"] in set(execution_ids))
        return 1 if any(row["status"] != "completed" for row in scoped_records) else 0
    except MemoryTransportAbort as error:
        print(f"STOPPED: {error}. Resume with lower WriteConcurrency; checkpoints are preserved.",
              flush=True)
        return 2
    except KeyboardInterrupt:
        print("Interrupted; resume uses persisted conversation and question checkpoints.", flush=True)
        return 130
    finally:
        if state:
            for record in state["conversations"].values():
                if record["status"] == "running":
                    record["status"] = "interrupted"
            for record in state["questions"].values():
                if record["status"] == "running":
                    record["status"] = "interrupted"
            save(output / "state.json", state)
            write_summary(output, state)
            if args.only_conversations:
                write_summary(output, state, execution_ids, "summary-selected")
                print("Selected summary: " + str(output / "summary-selected.md"), flush=True)
            else:
                print("Summary: " + str(output / "summary.md"), flush=True)
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())

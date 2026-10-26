"""Local BEAM 100K Add/Search evaluation using the AML official Answer/Judge."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from dotenv import load_dotenv

from .beam_data import (QUESTION_TYPES, ROOT, VENDOR_SHA, add_payloads, append_conversations,
                        object_digest, official, prepare, selected_source_rows)
from .diagnostics import DiagnosticLog, exception_info
from .lme_profiles import profile_files, repository_root
from .lme_smoke import check_backends, model_transport_overrides
from .models import SearchResponse
from .persona_batch import http_post, memory_server
from .persona_data import digest, save
from .state import ProcessLock


# Exact predecessor that raised inside LoggedChatClient before the official BEAM
# retry loop could inspect HTTP 429.  This upgrade changes transport handling
# only; selection, prompts, models, scoring and stored artifacts are unchanged.
RESUME_COMPATIBLE_FINGERPRINTS = {
    "686c74b76f3a0f07314c4be9414df32e183cabc903be9f34c8bd9cb1f8ba2236",
    "cc6206ff2692a5e9fa4f589dbb10417bbc34a0b1453893ef500533deb0c9724f",
    "1aee56ff71142990a785d510720afe83db4f934481d5aa5b87271786ef789a27",
    "95b8bddce1e5e4352cfb8813de316eab393ae993bdaeb852af5c720fa68ed5a6",
    "f3ba8d33509d532b34dfc768e1d337f516bb2a215d4b97bb9cb1080b8b053005",
    "4292a0e3f7ff0b312c30901bea1b71b6e57c01f1d6d0c72f43a510f856fa676c",
    "335be20ba8d40f08e1224661ec5d30c903fa29edc9b56fd7dff6ee232751a861",
    "bdf22b0a34b2f3098e0b0a464143fa786ae4e06aa7fcba30d7b25326a37d51f3",
}


def model_config(env, role):
    prefix = "BEAM_" + role.upper()
    return {"base_url": env.get(prefix + "_BASE_URL") or env.get("LME_QA_BASE_URL") or env["AML_LLM_BASE_URL"],
            "api_key": env.get(prefix + "_API_KEY") or env.get("LME_QA_API_KEY") or env["AML_LLM_API_KEY"],
            "model": env.get(prefix + "_MODEL") or env.get("LME_QA_MODEL") or env["AML_LLM_MODEL"]}


def public_models(env):
    return {role: {key: value for key, value in model_config(env, role).items() if key != "api_key"}
            for role in ("answer", "judge")}


def isolate_beam_storage(env, profile):
    """Keep BEAM away from long-running benchmark collections and their load state."""
    suffix = profile.replace("-", "_")
    env["AML_ES_INDEX"] = env.get("BEAM_ES_INDEX") or f"aml_beam_{suffix}_text_v1"
    env["AML_MILVUS_COLLECTION"] = (env.get("BEAM_MILVUS_COLLECTION")
                                    or f"aml_beam_{suffix}_vector_v1")
    env["AML_MILVUS_URI"] = env.get("BEAM_MILVUS_URI") or "http://127.0.0.1:21530"
    env["LME_MILVUS_HEALTH_URL"] = (env.get("BEAM_MILVUS_HEALTH_URL")
                                    or "http://127.0.0.1:9191/healthz")


def fingerprint(args, selection, env):
    paths = [ROOT / "aml_adapter" / name for name in
             ("beam_batch.py", "beam_data.py", "persona_batch.py", "persona_data.py",
              "lme_profiles.py", "lme_smoke.py", "backend.py", "service.py", "app.py",
              "models.py", "state.py", "schema_eval.py", "diagnostics.py")]
    paths += profile_files(args.memory_profile)
    paths += sorted((repository_root(args.memory_profile) / "jiuwen_memory").rglob("*.py"))
    keys = ("AML_LLM_MODEL", "AML_LLM_PROVIDER", "AML_LLM_BASE_URL", "AML_EMBED_MODEL",
            "AML_EMBED_BASE_URL", "AML_REDIS_URL", "AML_ES_URL", "AML_ES_INDEX",
            "AML_MILVUS_URI", "AML_MILVUS_COLLECTION")
    return object_digest({"selection": selection, "profile": args.memory_profile, "top_k": args.top_k,
                          "models": public_models(env), "vendor": VENDOR_SHA,
                          "settings": {key: env.get(key) for key in keys},
                          "files": {str(path.relative_to(ROOT.parent)): digest(path) for path in paths}})


def selection_is_extension(previous, candidate):
    """Accept only the same frozen sample with additional conversations/questions."""
    fields = ("dataset", "revision", "source_sha256", "size", "seed", "per_type",
              "sampling_mode", "input_contract", "official_competition_bundle")
    if any(previous.get(field) != candidate.get(field) for field in fields):
        return False
    old_conversations = {row["id"]: row for row in previous["conversations"]}
    new_conversations = {row["id"]: row for row in candidate["conversations"]}
    old_questions = {row["id"]: row for row in previous["questions"]}
    new_questions = {row["id"]: row for row in candidate["questions"]}
    return (len(new_conversations) > len(old_conversations)
            and all(new_conversations.get(ident) == row
                    for ident, row in old_conversations.items())
            and all(new_questions.get(ident) == row for ident, row in old_questions.items()))


def extend_state(state, selection):
    """Merge a validated selection extension while preserving every checkpoint."""
    for conversation in selection["conversations"]:
        ident = conversation["id"]
        user_id = state["namespace"] + ":conversation:" + ident
        state["conversations"].setdefault(
            ident, {"status": "pending", "stage": "pending", "next_add": 0,
                    "user_id": user_id})
    for question in selection["questions"]:
        state["questions"].setdefault(
            question["id"], {"status": "pending", "stage": "pending",
                             "question_type": question["question_type"],
                             "conversation_id": question["conversation_id"],
                             "user_id": state["namespace"] + ":conversation:"
                                        + question["conversation_id"]})
    state["selection"] = selection


class LoggedChatClient:
    def __init__(self, client, role, artifact, diagnostic):
        self.client, self.role = client, role
        self.artifact, self.diagnostic = artifact, diagnostic

    async def post(self, url, *, json, **kwargs):
        with self.diagnostic.span(self.role + ".chat", enable_thinking=False) as event:
            response = await self.client.post(url, json=model_request_payload(json), **kwargs)
            event["status_code"] = response.status_code
            # The official BEAM call_model must receive 429 responses so its
            # six-attempt backoff policy can run.  Raising here bypassed that
            # policy and turned one provider throttle into many failed cases.
            if response.is_success:
                body = response.json()
                save(self.artifact / f"{self.role}-response-{uuid4().hex[:12]}.json", body)
            return response


def model_request_payload(payload):
    """Force the public Answer/Judge model calls to use non-thinking mode."""
    return {**payload, "enable_thinking": False}


async def model_stage(pipeline, role, artifact, diagnostic, config, messages, max_tokens, json_mode=False):
    async with httpx.AsyncClient(timeout=900) as client:
        wrapped = LoggedChatClient(client, role, artifact, diagnostic)
        namespace = SimpleNamespace(**config)
        task = asyncio.create_task(pipeline.call_model(wrapped, namespace, messages, max_tokens, json_mode))
        started = time.monotonic()
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=60)
                if done:
                    return task.result()
                print(f"WAIT {role}: elapsed={time.monotonic()-started:.0f}s", flush=True)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)


async def event_metrics(pipeline, artifact, diagnostic, config, rubrics, answer):
    async with httpx.AsyncClient(timeout=900) as client:
        wrapped = LoggedChatClient(client, "alignment", artifact, diagnostic)
        namespace = SimpleNamespace(**config)
        task = asyncio.create_task(pipeline.align_with_llm(wrapped, namespace, rubrics, answer.split("\n")))
        started = time.monotonic()
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=60)
                if done:
                    reference, system = task.result()
                    return {**pipeline.event_ordering_metrics(reference, system),
                            "aligned_reference": reference, "aligned_system": system}
                print(f"WAIT event alignment: elapsed={time.monotonic()-started:.0f}s", flush=True)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)


def validate_models(env, allow_http):
    for role in ("answer", "judge"):
        config = model_config(env, role)
        url = urlsplit(config["base_url"])
        if (url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password
                or url.query or url.fragment or not config["api_key"] or not config["model"]):
            raise ValueError("Invalid model configuration: " + role)
        if url.scheme == "http" and not allow_http:
            raise ValueError("HTTP models require -AllowHttpModels")


def answer_prompt(pipeline, question, retrieved):
    return pipeline.render_answer_prompt({"question": question,
                                          "context": [item["content"] for item in retrieved]})


def run_question(state, record, question, client, pipeline, args, env, output, diagnostic, checkpoint):
    artifact = output / "questions" / question["id"]
    artifact.mkdir(parents=True, exist_ok=True)
    record.update(status="running")
    record.pop("error", None)
    checkpoint()
    try:
        record["stage"] = "search"
        checkpoint()
        search_file = artifact / "search.json"
        if not search_file.exists():
            print(f"SEARCH {question['question_type']}: {question['id']}", flush=True)
            result = http_post(client, "/search", {"query": question["question"],
                               "user_id": record["user_id"], "top_k": args.top_k}, diagnostic)
            SearchResponse.model_validate(result)
            save(search_file, result)
        search = json.loads(search_file.read_text(encoding="utf-8"))
        SearchResponse.model_validate(search)
        retrieved = search["data"]
        record.update(retrieved=len(retrieved),
                      context_words=sum(len(item["content"].split()) for item in retrieved))

        record["stage"] = "answer"
        checkpoint()
        answer_file = artifact / "answer.json"
        if not answer_file.exists():
            prompt = answer_prompt(pipeline, question["question"], retrieved)
            save(artifact / "answer-prompt.json", {"prompt": prompt})
            print("ANSWER: " + question["id"], flush=True)
            answer = asyncio.run(model_stage(pipeline, "answer", artifact, diagnostic,
                                             model_config(env, "answer"),
                                             [{"role": "user", "content": prompt}], 512))
            save(answer_file, {"generated_answer": answer})
        answer = json.loads(answer_file.read_text(encoding="utf-8"))["generated_answer"]

        record["stage"] = "judge"
        checkpoint()
        judge_file = artifact / "judge.json"
        if not judge_file.exists():
            prompt = pipeline.render_batch_judge_prompt(question["question"], answer, question["rubrics"])
            save(artifact / "judge-prompt.json", {"prompt": prompt})
            for judge_attempt in range(2):
                print(("JUDGE: " if judge_attempt == 0 else "JUDGE FORMAT RETRY: ")
                      + question["id"], flush=True)
                raw = asyncio.run(model_stage(pipeline, "judge", artifact, diagnostic,
                                              model_config(env, "judge"),
                                              [{"role": "user", "content": prompt}],
                                              (1024 if judge_attempt == 0 else
                                               max(2048, 512 * len(question["rubrics"]))),
                                              True))
                try:
                    scores = pipeline.parse_rubric_scores(raw, len(question["rubrics"]))
                    break
                except (json.JSONDecodeError, ValueError):
                    save(artifact / f"judge-unparsed-{judge_attempt+1}.json", {"response": raw})
                    if judge_attempt == 1:
                        raise
            result = {"question_type": question["question_type"],
                      "rubric_scores": [{"rubric": rubric, **score}
                                        for rubric, score in zip(question["rubrics"], scores)],
                      "llm_judge_score": sum(score["score"] for score in scores) / len(scores),
                      "judge_response": raw}
            if question["question_type"] == "event_ordering":
                result["event_ordering"] = asyncio.run(event_metrics(
                    pipeline, artifact, diagnostic, model_config(env, "judge"), question["rubrics"], answer))
            save(judge_file, result)
        result = json.loads(judge_file.read_text(encoding="utf-8"))
        record.update(score=result["llm_judge_score"], status="completed", stage="done")
        if "event_ordering" in result:
            record["event_ordering_score"] = result["event_ordering"]["final_score"]
        print(f"DONE {question['id']}: nugget_score={record['score']:.4f}; retrieved={record['retrieved']}", flush=True)
    except Exception as error:
        details = exception_info(error)
        statuses = {item.get("status_code") for item in details["chain"]}
        if 429 in statuses:
            record.update(status="interrupted", error=details)
            diagnostic.emit(record["stage"], "rate_limited", question_id=question["id"],
                            error=details)
            checkpoint()
            print(f"PAUSED {question['id']}: upstream still returns HTTP 429 after retries; "
                  "resume this batch later", flush=True)
            raise
        record.update(status="failed", error=details)
        diagnostic.emit(record["stage"], "failed", question_id=question["id"], error=record["error"])
        print(f"FAILED {question['id']}: stage={record['stage']}; {type(error).__name__}; continuing", flush=True)
        if record["stage"] == "search" and isinstance(error, httpx.TransportError):
            client.restart_owned_server()
    checkpoint()


def run_cases(state, source, client, pipeline, args, env, output, diagnostic, checkpoint):
    questions_by_conversation = {}
    for question in state["selection"]["questions"]:
        questions_by_conversation.setdefault(question["conversation_id"], []).append(question)
    # Finish ingestion first.  Answer/Judge (especially event alignment) can
    # consume a large model quota; it must not prevent later conversations
    # from reaching their Add checkpoints.
    for conversation in state["selection"]["conversations"]:
        ident = conversation["id"]
        ingestion = state["conversations"][ident]
        questions = questions_by_conversation[ident]
        if ingestion["status"] != "completed":
            if args.skip_failed and ingestion["status"] == "failed":
                continue
            ingestion.update(status="running")
            ingestion.pop("error", None)
            checkpoint()
            try:
                chunks = list(add_payloads(source[ident], ingestion["user_id"]))
                ingestion["total_adds"] = len(chunks)
                for index in range(ingestion["next_add"], len(chunks)):
                    print(f"ADD conversation {ident} {index+1}/{len(chunks)}", flush=True)
                    started = time.monotonic()
                    result = http_post(client, "/add", chunks[index], diagnostic)
                    if result.get("success") is not True:
                        raise RuntimeError("Add did not report success")
                    artifact = output / "conversations" / ident
                    save(artifact / f"add-{index+1:03d}.json",
                         {"response": result, "elapsed_seconds": time.monotonic()-started})
                    ingestion["next_add"] = index + 1
                    checkpoint()
                ingestion.update(status="completed", stage="done")
            except Exception as error:
                ingestion.update(status="failed", stage="add", error=exception_info(error))
                diagnostic.emit("add", "failed", conversation_id=ident, error=ingestion["error"])
                print(f"FAILED conversation {ident}: {type(error).__name__}; continuing", flush=True)
                for question in questions:
                    record = state["questions"][question["id"]]
                    if record["status"] != "completed":
                        record.update(status="failed", stage="add", error=ingestion["error"])
                if isinstance(error, httpx.TransportError):
                    client.restart_owned_server()
                checkpoint()
                continue
            checkpoint()

    for conversation in state["selection"]["conversations"]:
        ident = conversation["id"]
        ingestion = state["conversations"][ident]
        if ingestion["status"] != "completed":
            continue
        questions = questions_by_conversation[ident]
        for question in questions:
            record = state["questions"][question["id"]]
            if record["status"] == "completed" or (args.skip_failed and record["status"] == "failed"):
                continue
            run_question(state, record, question, client, pipeline, args, env, output, diagnostic, checkpoint)


def stats(records):
    rows = list(records)
    completed = [row for row in rows if row["status"] == "completed"]
    total_score = sum(row["score"] for row in completed)
    event = [row["event_ordering_score"] for row in completed if "event_ordering_score" in row]
    return {"total": len(rows), "completed": len(completed),
            "failed": sum(row["status"] == "failed" for row in rows),
            "mean_nugget_score": total_score / len(completed) if completed else None,
            "full_sample_nugget_score": total_score / len(rows) if rows else None,
            "perfect_questions": sum(row["score"] == 1.0 for row in completed),
            "event_ordering_mean": sum(event) / len(event) if event else None}


def summarize(output, state):
    report = {"profile": state["profile"], "official_score": False,
              "sampling": state["selection"]["sampling"], **stats(state["questions"].values())}
    report["by_type"] = {question_type: stats(row for row in state["questions"].values()
                                                   if row["question_type"] == question_type)
                         for question_type in QUESTION_TYPES}
    report["failures"] = [{"id": ident, "stage": row["stage"]}
                          for ident, row in state["questions"].items() if row["status"] == "failed"]
    save(output / "summary.json", report)
    lines = [f"# BEAM {state['selection']['size']} — {state['profile']}", "",
             "Local fixed sample; AML official Answer/Judge; not an official competition score.",
             f"Conversations: {state['selection']['conversation_count']}; questions: {report['total']}; Top-K: {state['top_k']}.",
             f"Completed {report['completed']}/{report['total']}; failed {report['failed']}.",
             f"Mean rubric-nugget score: {report['mean_nugget_score']}; full-sample score: {report['full_sample_nugget_score']}.",
             f"Perfect questions: {report['perfect_questions']}; event-ordering mean: {report['event_ordering_mean']}.", "",
             "| Question type | Completed | Mean nugget | Perfect | Event ordering | Failed |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for question_type, values in report["by_type"].items():
        lines.append(f"| {question_type} | {values['completed']}/{values['total']} | "
                     f"{values['mean_nugget_score']} | {values['perfect_questions']} | "
                     f"{values['event_ordering_mean']} | {values['failed']} |")
    lines += ["", "Failed IDs: " + (", ".join(item["id"] for item in report["failures"]) or "none")]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Completed={report['completed']}/{report['total']}; failed={report['failed']}", flush=True)


def compare(paths):
    states = [json.loads((Path(path) / "state.json").read_text(encoding="utf-8")) for path in paths]
    for field in ("selection", "models", "top_k"):
        if states[0][field] != states[1][field]:
            raise ValueError("Cannot compare different " + field)
    ids = [ident for ident in states[0]["questions"]
           if all(state["questions"][ident]["status"] == "completed" for state in states)]
    cases = []
    for ident in ids:
        left, right = (state["questions"][ident]["score"] for state in states)
        cases.append({"id": ident, "question_type": states[0]["questions"][ident]["question_type"],
                      "scores": [left, right], "delta": right-left})
    report = {"profiles": [state["profile"] for state in states], "paired_ids": ids,
              "all": [stats(state["questions"].values()) for state in states],
              "paired": [stats(state["questions"][ident] for ident in ids) for state in states],
              "wins": {"left": sum(case["delta"] < 0 for case in cases),
                       "tie": sum(case["delta"] == 0 for case in cases),
                       "right": sum(case["delta"] > 0 for case in cases)},
              "cases": cases}
    save(Path(paths[1]) / "comparison.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-profile", choices=("ordinary", "schema-entity"), default="ordinary")
    parser.add_argument("--size", choices=("100K",), default="100K")
    parser.add_argument("--conversations", type=int, default=1)
    parser.add_argument("--extend-conversations", type=int)
    parser.add_argument("--add-conversation", action="append", default=[])
    parser.add_argument("--per-type", type=int, default=1)
    parser.add_argument("--sampling", choices=("smoke", "stratified"), default="smoke")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--env-file", default=str(ROOT / "config/longmemeval.local.env"))
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--skip-failed", action="store_true")
    parser.add_argument("--allow-http-models", action="store_true")
    parser.add_argument("--compare", nargs=2)
    args = parser.parse_args(argv)
    if args.compare:
        compare(args.compare)
        return 0
    if not 1 <= args.top_k <= 100 or not 1 <= args.conversations <= 20 or not 1 <= args.per_type <= 2:
        parser.error("top-k 1..100; conversations 1..20; per-type 1..2")
    if args.extend_conversations is not None and (
            not args.resume or not 2 <= args.extend_conversations <= 20):
        parser.error("extend-conversations requires resume and a value from 2..20")
    if args.add_conversation and not args.resume:
        parser.error("add-conversation requires resume")
    if args.skip_failed and not args.resume:
        parser.error("skip-failed requires resume")
    conversation_count = args.extend_conversations or args.conversations
    selection, selection_path = prepare(conversation_count, args.per_type, args.seed,
                                        args.size, args.sampling)
    if args.add_conversation:
        selection, selection_path = append_conversations(selection, args.add_conversation)
    pipeline = official()
    source = selected_source_rows(selection)
    adds = sum(len(list(add_payloads(source[row["id"]], "plan"))) for row in selection["conversations"])
    print(f"PLAN: BEAM {args.size}; {len(selection['conversations'])} conversations; "
          f"{len(selection['questions'])} questions; {adds} Add requests; top_k={args.top_k}")
    print("Selection: " + str(selection_path))
    print("Question types: " + ", ".join(QUESTION_TYPES))
    print("Each conversation is ingested once and reused by its probing questions; no history truncation.")
    print("AML official Answer prompt and rubric-nugget Judge; event ordering adds F1 x normalized Kendall tau.")
    for row in selection["conversations"]:
        print(f"  conversation {row['id']}: turns={row['turns']}; words={row['words']}; "
              f"topic={row['topic'].get('category', 'unknown')}")
    if not args.run:
        return 0
    if not Path(args.env_file).is_file():
        raise ValueError("Environment file does not exist")
    load_dotenv(args.env_file, override=True)
    env = dict(os.environ)
    isolate_beam_storage(env, args.memory_profile)
    model_transport_overrides(env, args.allow_http_models)
    validate_models(env, args.allow_http_models)
    env["PM_MEMORY_HTTP_TIMEOUT_SECONDS"] = env.get("BEAM_MEMORY_HTTP_TIMEOUT_SECONDS", "14400")
    print(f"Storage isolation: ES={env['AML_ES_INDEX']}; "
          f"Milvus={env['AML_MILVUS_COLLECTION']} @ {env['AML_MILVUS_URI']}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output = (args.resume or ROOT / f"outputs/beam/{args.memory_profile}-{args.size}-{stamp}-{uuid4().hex[:8]}").resolve()
    if args.resume and not (output / "state.json").is_file():
        raise ValueError("Resume requires a BEAM batch directory containing state.json")
    output.mkdir(parents=True, exist_ok=True)
    lock = ProcessLock(output / "runner.lock")
    state = None
    try:
        signature = fingerprint(args, selection, env)
        if args.resume:
            state = json.loads((output / "state.json").read_text(encoding="utf-8"))
            expanding = ((args.extend_conversations is not None or bool(args.add_conversation))
                         and selection_is_extension(state["selection"], selection))
            predecessor_signature = fingerprint(args, state["selection"], env)
            compatible_upgrade = (
                (state.get("fingerprint") in RESUME_COMPATIBLE_FINGERPRINTS
                 or state.get("fingerprint") == predecessor_signature)
                and state.get("models") == public_models(env)
                and state.get("top_k") == args.top_k
                and (state.get("selection") == selection or expanding)
            )
            if state["profile"] != args.memory_profile or (
                    state["fingerprint"] != signature and not compatible_upgrade):
                raise ValueError("Resume contract changed; use the original profile/configuration")
            if (args.extend_conversations is not None or args.add_conversation) and state["selection"] != selection:
                if not expanding:
                    raise ValueError("Resume selection is not a strict extension of the existing batch")
                added_conversations = len(selection["conversations"]) - len(state["selection"]["conversations"])
                added_questions = len(selection["questions"]) - len(state["selection"]["questions"])
                extend_state(state, selection)
                print(f"EXPANSION: preserve existing checkpoints; add {added_conversations} conversations "
                      f"and {added_questions} questions.", flush=True)
            if state["fingerprint"] != signature:
                print("RESUME UPGRADE: preserve Add/Search/QA artifacts; restore official BEAM "
                      "HTTP 429 retries and pause the batch if throttling persists.", flush=True)
                state["fingerprint"] = signature
        else:
            namespace = "beam-" + uuid4().hex
            user_ids = {row["id"]: namespace + ":conversation:" + row["id"]
                        for row in selection["conversations"]}
            state = {"profile": args.memory_profile, "fingerprint": signature, "selection": selection,
                     "namespace": namespace, "models": public_models(env), "top_k": args.top_k,
                     "conversations": {row["id"]: {"status": "pending", "stage": "pending", "next_add": 0,
                                                             "user_id": user_ids[row["id"]]}
                                       for row in selection["conversations"]},
                     "questions": {question["id"]: {"status": "pending", "stage": "pending",
                                                        "question_type": question["question_type"],
                                                        "conversation_id": question["conversation_id"],
                                                        "user_id": user_ids[question["conversation_id"]]}
                                   for question in selection["questions"]}}

        def checkpoint():
            save(output / "state.json", state)

        checkpoint()
        print("Batch directory: " + str(output), flush=True)
        todo = [row for row in state["questions"].values() if row["status"] != "completed"
                and not (args.skip_failed and row["status"] == "failed")]
        if todo:
            check_backends(env)
            diagnostic = DiagnosticLog(output / "diagnostics.jsonl")
            with memory_server(output, args.memory_profile, state["namespace"], env, args.allow_http_models) as client:
                run_cases(state, source, client, pipeline, args, env, output, diagnostic, checkpoint)
        return 1 if any(row["status"] != "completed" for row in state["questions"].values()) else 0
    except KeyboardInterrupt:
        print("Interrupted; owned HTTP worker stopped; resume keeps completed Add/Search/QA.", flush=True)
        return 130
    except httpx.HTTPStatusError as error:
        if error.response.status_code == 429:
            print("Batch paused by persistent model HTTP 429; resume later without re-ingesting "
                  "or re-running completed questions.", flush=True)
            return 75
        raise
    finally:
        try:
            if state is not None:
                for record in list(state["conversations"].values()) + list(state["questions"].values()):
                    if record["status"] == "running":
                        record["status"] = "interrupted"
                save(output / "state.json", state)
                summarize(output, state)
                print("Summary: " + str(output / "summary.md"), flush=True)
        finally:
            lock.close()


if __name__ == "__main__":
    raise SystemExit(main())

"""PersonaMem-v2 text A/B evaluation through the competition HTTP contract."""

import argparse
import asyncio
import copy
import hashlib
import json
import os
import queue
import secrets
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import httpx
import yaml
from dotenv import load_dotenv

from .diagnostics import DiagnosticLog, exception_info
from .lme_profiles import memory_config, repository_root
from .lme_smoke import check_backends, model_transport_overrides
from .persona_data import DATA, ROOT, VENDOR, add_chunks, digest, messages, official, prepare, save
from .state import ProcessLock


def model_config(env, role):
    prefix = "PM_" + role.upper()
    return {"base_url": env.get(prefix + "_BASE_URL") or env.get("LME_QA_BASE_URL") or env["AML_LLM_BASE_URL"],
            "api_key": env.get(prefix + "_API_KEY") or env.get("LME_QA_API_KEY") or env["AML_LLM_API_KEY"],
            "model": env.get(prefix + "_MODEL") or env.get("LME_QA_MODEL") or env["AML_LLM_MODEL"]}


def public_models(env):
    return {role: {k: v for k, v in model_config(env, role).items() if k != "api_key"}
            for role in ("answer", "judge")}


def identity(args, selection, env):
    from .lme_batch import fingerprint
    contract = {"selection": selection, "top_k": args.top_k, "mode": args.mode,
                "models": public_models(env), "adapter": fingerprint(args.env_file, args.top_k, False, args.memory_profile),
                "code": {str(p.relative_to(ROOT)): digest(p) for p in
                         [Path(__file__), ROOT / "aml_adapter/persona_data.py",
                          ROOT / "aml_adapter/persona_expansion.py",
                          VENDOR / "data/personamem/pipeline_v2.py", VENDOR / "api_config.py"]}}
    if getattr(args, "qa_cutoffs", None):
        contract["qa_cutoffs"] = args.qa_cutoffs
        contract["mcq_format"] = args.mcq_format
        for name in ("persona_multi.py", "persona_cutoffs.py"):
            path = ROOT / "aml_adapter" / name
            contract["code"][str(path.relative_to(ROOT))] = digest(path)
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


def memory_http_timeout(env):
    seconds = int(env.get("PM_MEMORY_HTTP_TIMEOUT_SECONDS", "14400"))
    if seconds < 60:
        raise ValueError("PM_MEMORY_HTTP_TIMEOUT_SECONDS must be >= 60")
    return httpx.Timeout(connect=5, read=seconds, write=60, pool=5)


def stop_owned_process(process):
    if process is None or process.poll() is not None:
        return
    if os.name == "nt":
        # A venv python.exe can be a redirector which launches the base Python
        # as its child. Terminating only the Popen PID may orphan Uvicorn and
        # leave service.lock held after Ctrl+C. Kill only this owned process
        # tree; unrelated experiments have different roots and are untouched.
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


@contextmanager
def memory_server(output, profile, namespace, env, allow_http, max_concurrency=1):
    config = memory_config(profile)
    config["globals"].update(model_transport_overrides(env, allow_http))
    if profile in ("schema-entity", "schema-temporal"):
        params = config["extractor"]["default"]["params"]
        frozen = output / "schema.json"
        frozen.write_bytes(Path(params["schema_path"]).read_bytes())
        params["schema_path"] = str(frozen)
    config_path = output / "memory.yml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    token = secrets.token_urlsafe(32)
    child = dict(env)
    child.update(AML_API_KEY=token, AML_NAMESPACE=namespace, AML_STATE_DIR=str(output / "http-state"),
                 AML_MEMORY_CONFIG=str(config_path), AML_REPO_EVAL="false",
                 AML_MAX_CONCURRENCY=str(max_concurrency),
                 AML_EXTRACTION_FIDELITY="false", AML_LOCAL_SMOKE="false",
                 AML_DIAGNOSTIC_LOG=str(output / "api-diagnostics.jsonl"),
                 PYTHONPATH=os.pathsep.join((str(repository_root(profile)), str(ROOT))),
                 AML_LME_REPOSITORY=str(repository_root(profile)), PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    endpoint = f"http://127.0.0.1:{port}"
    with (output / "api.log").open("a", encoding="utf-8") as log:
        process = None
        client = None

        def start():
            nonlocal process
            process = subprocess.Popen([sys.executable, "-m", "uvicorn", "aml_adapter.app:create_app",
                                        "--factory", "--host", "127.0.0.1", "--port", str(port), "--no-access-log"],
                                       cwd=ROOT, env=child, stdout=log, stderr=log,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            with httpx.Client(timeout=1, trust_env=False) as probe:
                for _ in range(300):
                    if process.poll() is not None:
                        raise RuntimeError("Memory API exited; see api.log")
                    try:
                        if probe.get(endpoint + "/health").status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(.2)
                else:
                    raise RuntimeError("Memory API startup timed out")

        def restart():
            print("RECOVERY: stopping the owned HTTP worker after transport failure; "
                  "restart before dispatching another request", flush=True)
            stop_owned_process(process)
            start()

        def stop():
            """Stop this context's worker so callers can cancel blocked local requests."""
            stop_owned_process(process)

        try:
            timeout = memory_http_timeout(env)
            start()
            client = httpx.Client(base_url=endpoint, headers={"Authorization": "Bearer " + token},
                                  timeout=timeout, trust_env=False)
            client.restart_owned_server = restart
            client.stop_owned_server = stop
            print(f"Memory HTTP read timeout: {timeout.read:g}s; WAIT progress every 60s", flush=True)
            yield client
        finally:
            try:
                stop_owned_process(process)
            finally:
                if client is not None:
                    client.close()


def interruptible_post(client, path, payload, progress_seconds=60):
    # Keep the main thread interruptible on Windows during long synchronous HTTP reads.
    results = queue.Queue()

    def request():
        try:
            results.put((True, client.post(path, json=payload)))
        except BaseException as exc:
            results.put((False, exc))

    threading.Thread(target=request, daemon=True).start()
    started = time.monotonic()
    next_report = started + progress_seconds
    while True:
        try:
            ok, result = results.get(timeout=.2)
            if ok:
                return result
            raise result
        except queue.Empty:
            now = time.monotonic()
            if now >= next_report:
                print(f"WAIT {path}: elapsed={now - started:.0f}s; waiting for response "
                      "(Add may replay earlier history; this is not a health check)", flush=True)
                next_report = now + progress_seconds


def http_post(client, path, payload, diagnostic):
    with diagnostic.span(path) as event:
        response = interruptible_post(client, path, payload)
        event["status_code"] = response.status_code
        response.raise_for_status()
        return response.json()


def answer_messages(pipeline, row, retrieved, mode):
    # Only returned memory content becomes history; no preference, gold, or source hints.
    history = [{"role": "user", "content": "Retrieved conversation memories:\n" +
                "\n\n".join(item["content"] for item in retrieved)}]
    result = pipeline.official_generative_messages({"chat_history": history, "user_query": row["user_query"]})
    if mode == "mcq":
        options = "\n".join(f"{letter}. {text}" for letter, text in row["option_mapping"].items())
        result.append({"role": "system", "content": pipeline.MCQ_PROMPT_TEMPLATE.format(options=options)})
    return result


class NonThinkingChatClient:
    """Inject the gateway option without changing the vendored prompts or scoring."""

    def __init__(self, client):
        self.client = client

    async def post(self, url, *, json, **kwargs):
        # This is raw HTTP JSON; extra_body is an SDK option, not a wire field.
        return await self.client.post(url, json={**json, "enable_thinking": False}, **kwargs)


async def complete(pipeline, env, role, prompt, diagnostic):
    with diagnostic.span(role + ".chat", enable_thinking=False):
        async with httpx.AsyncClient(timeout=900) as client:
            return await pipeline.post_chat(NonThinkingChatClient(client), **model_config(env, role),
                                            messages=prompt, temperature=0)


def evaluate_question(row, record, client, user_id, pipeline, args, env, output, diagnostic, checkpoint):
    artifact = output / "questions" / row["id"]
    artifact.mkdir(parents=True, exist_ok=True)
    search_path = artifact / "search.json"
    if not search_path.exists():
        response = http_post(client, "/search", {"query": pipeline.user_query_text(row),
                             "user_id": user_id, "top_k": args.top_k}, diagnostic)
        save(search_path, response)
    retrieved = json.loads(search_path.read_text(encoding="utf-8"))["data"]
    record["retrieved"] = len(retrieved)
    if getattr(args, "qa_cutoffs", None):
        from .persona_multi import evaluate_cutoffs
        evaluate_cutoffs(row, record, retrieved, artifact, pipeline, args, env, diagnostic, checkpoint)
        return
    modes = ("mcq", "generative") if args.mode == "both" else (args.mode,)
    for mode in modes:
        answer_path = artifact / (mode + "-answer.json")
        if not answer_path.exists():
            prompt = answer_messages(pipeline, row, retrieved, mode)
            save(artifact / (mode + "-prompt.json"), prompt)
            answer = asyncio.run(complete(pipeline, env, "answer", prompt, diagnostic))
            save(answer_path, {"answer": answer})
        answer = json.loads(answer_path.read_text(encoding="utf-8"))["answer"]
        if mode == "mcq":
            # Invoke the upstream scorer without rewriting its mapping/label logic.
            pipeline.write_jsonl(artifact / "mcq-input.jsonl", [{"id": row["id"], "generated_answer": answer,
                "option_mapping": row["option_mapping"], "correct_letter": row["correct_letter"]}])
            pipeline.evaluate_mcq(argparse.Namespace(answers=str(artifact / "mcq-input.jsonl"),
                                                    output=str(artifact / "mcq-score.jsonl")))
            record["mcq_correct"] = pipeline.read_jsonl(artifact / "mcq-score.jsonl")[0]["is_correct"]
        else:
            judge_path = artifact / "judge.json"
            if not judge_path.exists():
                preference = row["preference"]
                template = (pipeline.JUDGE_PROMPT_NARROW_NEGATIVE if preference.lower().startswith("do not")
                            else pipeline.JUDGE_PROMPT_NARROW_POSITIVE)
                prompt = [{"role": "user", "content": template.format(user_query=pipeline.user_query_text(row),
                           preference=preference, model_response=answer)}]
                save(artifact / "judge-prompt.json", prompt)
                text = asyncio.run(complete(pipeline, env, "judge", prompt, diagnostic))
                save(judge_path, {"response": text, "score": pipeline.extract_boxed_score(text)})
            record["narrow_score"] = json.loads(judge_path.read_text(encoding="utf-8"))["score"]
        checkpoint()


def should_run(record, skip_failed):
    return record["status"] != "completed" and not (skip_failed and record["status"] == "failed")


def validate_resume(state, signature, profile, upgrades):
    """Allow only an audited exact predecessor/successor pair; retain all checkpoints."""
    if state["profile"] != profile:
        raise ValueError("Resume memory profile changed")
    if state["fingerprint"] == signature:
        return
    for upgrade in upgrades:
        if (upgrade["memory_profile"] == profile and upgrade["from"] == state["fingerprint"]
                and upgrade["to"] == signature):
            state.setdefault("runner_upgrades", []).append({
                **upgrade, "applied_at": datetime.now(timezone.utc).isoformat(),
                "preserved_completed_ids": [key for key, value in state["questions"].items()
                                            if value["status"] == "completed"],
                "remaining_ids": [key for key, value in state["questions"].items()
                                  if value["status"] != "completed"],
            })
            state["fingerprint"] = signature
            print("RESUME UPGRADE: " + upgrade["reason"], flush=True)
            return
    raise ValueError("Resume configuration/code/data changed; use original settings or start a new run")


def run_questions(state, rows, client, pipeline, args, env, output, diagnostic, checkpoint):
    for row in rows:
        record = state["questions"][row["id"]]
        if not should_run(record, args.skip_failed):
            continue
        history_key = row["persona_id"] + ":" + row["history_file"]
        history_state = state["histories"].setdefault(history_key,
                         {"status": "pending", "user_id": "pm-" + uuid4().hex, "next_add": 0})
        if history_state["status"] == "failed":
            record.update(status="failed", failure_stage="history", error=history_state["error"])
            checkpoint()
            continue
        record["status"] = "running"
        checkpoint()
        stage = "history"
        try:
            if history_state["status"] != "completed":
                chunks = list(add_chunks(messages(DATA / row["history_file"]), history_state["user_id"]))
                for index in range(history_state["next_add"], len(chunks)):
                    print(f"ADD {row['persona_id']} {index + 1}/{len(chunks)}", flush=True)
                    response = http_post(client, "/add", chunks[index], diagnostic)
                    if response.get("success") is not True:
                        raise RuntimeError("Add did not report success")
                    history_state["next_add"] = index + 1
                    checkpoint()
                history_state["status"] = "completed"
                checkpoint()
            stage = "search/answer/judge"
            print(f"QA {row['id']}: {row['persona_id']}", flush=True)
            evaluate_question(row, record, client, history_state["user_id"], pipeline, args, env,
                              output, diagnostic, checkpoint)
            record.update(status="completed")
            record.pop("error", None)
        except Exception as exc:
            error = exception_info(exc)
            record.update(status="failed", failure_stage=stage, error=error)
            if stage == "history":
                history_state.update(status="failed", error=error)
            diagnostic.emit(stage, "failed", question_id=row["id"], error=error)
            checkpoint()
            if (stage == "history" and isinstance(exc, httpx.TransportError)
                    and hasattr(client, "restart_owned_server")):
                # A client timeout does not cancel the server thread or release its sole slot.
                # If restart fails, propagate and stop dispatch rather than fail unrelated users.
                client.restart_owned_server()
            print(f"FAILED {row['id']}: {type(exc).__name__}; continuing", flush=True)
        checkpoint()


def stats(records):
    rows = list(records)
    completed = [r for r in rows if r["status"] == "completed"]
    def mean(key):
        values = [r[key] for r in completed if key in r]
        return sum(values) / len(values) if values else None
    return {"total": len(rows), "completed": len(completed),
            "failed": sum(r["status"] == "failed" for r in rows),
            "mcq_accuracy": mean("mcq_correct"), "narrow_mean": mean("narrow_score")}


def write_summary(output, state):
    if state.get("qa_cutoffs"):
        from .persona_multi import write_summary as write_multi_summary
        write_multi_summary(output, state)
        return
    summary = stats(state["questions"].values())
    summary["by_persona"] = {key: stats(r for r in state["questions"].values() if r["persona_id"] == key)
                             for key in sorted({r["persona_id"] for r in state["questions"].values()})}
    summary["by_preference_type"] = {key: stats(r for r in state["questions"].values() if r["pref_type"] == key)
                                     for key in sorted({r["pref_type"] for r in state["questions"].values()})}
    events_path = output / "api-diagnostics.jsonl"
    if events_path.exists() and state["profile"] == "schema-entity":
        events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        adds = [e for e in events if e.get("stage") == "schema.add" and e.get("event") == "ok"]
        expansions = [e for e in events if e.get("stage") == "entity.expand"]
        summary["feature_usage_including_retries"] = {
            "successful_schema_adds": len(adds),
            "property_units_returned": sum(e.get("property_units", 0) for e in adds),
            "entity_expansion_calls": len(expansions),
            "entity_expansion_candidates": sum(e.get("added_candidates", 0) for e in expansions)}
    failed = [key for key, value in state["questions"].items() if value["status"] == "failed"]
    save(output / "summary.json", summary)
    text = (f"# PersonaMem-v2 — {state['profile']}\n\nLocal sample, not an official competition score.\n\n"
            f"Completed: {summary['completed']}/{summary['total']}; failed: {summary['failed']}\n\n"
            f"MCQ accuracy: {summary['mcq_accuracy']}; generative narrow mean: {summary['narrow_mean']}\n\n"
            "Metrics use completed questions only; failed questions remain failed.\n\n"
            "Failed IDs: " + ", ".join(failed) + "\n\nBreakdowns: summary.json\n")
    if state.get("runner_upgrades"):
        summary["runner_upgrades"] = state["runner_upgrades"]
        save(output / "summary.json", summary)
        text += "\nResumed with an audited extraction repair upgrade; completed results retained. "
        text += "This batch contains results from before/after the repair; see state.json runner_upgrades.\n"
    (output / "summary.md").write_text(text, encoding="utf-8")
    print(f"Completed={summary['completed']}/{summary['total']}; failed={summary['failed']}; "
          f"failed IDs={','.join(failed) or 'none'}", flush=True)


def compare(paths):
    left, right = [json.loads((Path(p) / "state.json").read_text(encoding="utf-8")) for p in paths]
    if left.get("qa_cutoffs") or right.get("qa_cutoffs"):
        from .persona_multi import compare as compare_multi
        compare_multi(paths, left, right)
        return
    for field in ("selection", "models", "mode", "top_k"):
        if left[field] != right[field]:
            raise ValueError("Cannot pair different " + field)
    common = [key for key in left["questions"] if left["questions"][key]["status"] == "completed"
              and right["questions"][key]["status"] == "completed"]
    report = {"paired_n": len(common), "all": [stats(s["questions"].values()) for s in (left, right)],
              "paired": [stats(s["questions"][key] for key in common) for s in (left, right)],
              "profiles": [left["profile"], right["profile"]]}
    report["by_preference_type_paired"] = {
        kind: [stats(s["questions"][key] for key in common if left["questions"][key]["pref_type"] == kind)
               for s in (left, right)]
        for kind in sorted({left["questions"][key]["pref_type"] for key in common})}
    save(Path(paths[1]) / "comparison.json", report)
    lines = ["# PersonaMem-v2 paired comparison", "", f"Common completed questions: {len(common)}", "",
             "| Profile | Completed (all) | Failed (all) | Paired MCQ | Paired narrow mean |",
             "| --- | --- | --- | --- | --- |"]
    for index, profile in enumerate(report["profiles"]):
        all_stats, paired = report["all"][index], report["paired"][index]
        lines.append(f"| {profile} | {all_stats['completed']} | {all_stats['failed']} | "
                     f"{paired['mcq_accuracy']} | {paired['narrow_mean']} |")
    (Path(paths[1]) / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-profile", choices=("ordinary", "schema-entity", "schema-temporal"),
                        default="ordinary")
    parser.add_argument("--env-file", default=str(ROOT / "config/longmemeval.local.env"))
    parser.add_argument("--mode", choices=("both", "mcq", "generative"), default="both")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--qa-cutoffs", help="Comma-separated QA memory cutoffs, e.g. 10,20,50,100")
    parser.add_argument("--extend-per-user", type=int, help="Expand the same five personas while preserving old questions")
    parser.add_argument("--mcq-format", choices=("original", "reinforced"), default="original")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--skip-failed", action="store_true")
    parser.add_argument("--allow-http-models", action="store_true")
    parser.add_argument("--compare", nargs=2)
    args = parser.parse_args(argv)
    if args.compare:
        compare(args.compare)
        return 0
    if not 1 <= args.top_k <= 100:
        parser.error("top-k must be 1..100")
    if args.qa_cutoffs:
        try:
            args.qa_cutoffs = sorted(set(int(k) for k in args.qa_cutoffs.split(",")))
        except ValueError:
            parser.error("qa-cutoffs must be comma-separated integers")
        if any(k < 1 or k > args.top_k for k in args.qa_cutoffs):
            parser.error("each QA cutoff must be 1..top-k")
    if args.skip_failed and not args.resume:
        parser.error("skip-failed requires resume")
    selection = prepare()
    selection_path = DATA / "selection.json"
    if args.extend_per_user:
        from .persona_expansion import extend_selection, save_selection
        selection = extend_selection(selection, args.extend_per_user)
        selection_path = save_selection(selection)
    elif args.resume:
        # Expanded batches resume from their own frozen selection, not the original 25-question file.
        selection = json.loads((args.resume / "state.json").read_text(encoding="utf-8"))["selection"]
        selection_path = args.resume / "state.json"
    pipeline = official()
    chunks = sum(len(list(add_chunks(messages(DATA / p), "plan"))) for p in selection["files"])
    print(f"PLAN: PersonaMem-v2 32k; {len(selection['rows'])} questions; {len(selection['files'])} histories; "
          f"{chunks} Add requests per profile; top_k={args.top_k}; mode={args.mode}")
    print("Selection: " + str(selection_path))
    if args.qa_cutoffs:
        calls = 3 if args.mode == "both" else 2 if args.mode == "generative" else 1
        print(f"QA cutoffs: {args.qa_cutoffs}; MCQ format={args.mcq_format}; "
              f"{len(selection['rows']) * len(args.qa_cutoffs) * calls} Answer/Judge calls before retries. "
              "One shared ingestion per history and one Search per question.")
    if not args.run:
        return 0
    if not Path(args.env_file).is_file():
        raise ValueError("Environment file does not exist")
    load_dotenv(args.env_file, override=True)
    env = dict(os.environ)
    model_transport_overrides(env, args.allow_http_models)
    for role in ("answer", "judge"):
        config = model_config(env, role)
        from urllib.parse import urlsplit
        url = urlsplit(config["base_url"])
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query:
            raise ValueError("Invalid model endpoint for " + role)
        if url.scheme == "http" and not args.allow_http_models:
            raise ValueError("HTTP model requires -AllowHttpModels")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output = (args.resume or ROOT / f"outputs/personamem-v2/{args.memory_profile}-{stamp}-{uuid4().hex[:8]}").resolve()
    if args.resume and not (output / "state.json").exists():
        raise ValueError("Resume directory has no state.json")
    output.mkdir(parents=True, exist_ok=True)
    lock = ProcessLock(output / "runner.lock")
    state = None
    try:
        signature = identity(args, selection, env)
        if args.resume:
            previous = json.loads((output / "state.json").read_text(encoding="utf-8"))
            upgrades_path = ROOT / "config/persona-compatible-upgrades.json"
            upgrades = json.loads(upgrades_path.read_text(encoding="utf-8")) if upgrades_path.exists() else []
            if args.extend_per_user:
                from .persona_expansion import expand_state
                previous_args = copy.copy(args)
                previous_args.qa_cutoffs = previous.get("qa_cutoffs")
                previous_args.mcq_format = previous.get("mcq_format", "original")
                previous_signature = identity(previous_args, previous["selection"], env)
                validate_resume(previous, previous_signature, args.memory_profile, upgrades)
                previous = expand_state(previous, selection, args, signature, output)
                print(f"EXPANSION: {len(selection['rows'])} questions; existing histories and QA caches retained", flush=True)
            else:
                validate_resume(previous, signature, args.memory_profile, upgrades)
            state = previous
            if not args.skip_failed:
                for history in state["histories"].values():
                    if history["status"] == "failed":
                        history["status"] = "pending"
        else:
            state = {"profile": args.memory_profile, "fingerprint": signature, "selection": selection,
                     "namespace": "persona-" + uuid4().hex, "models": public_models(env), "mode": args.mode,
                     "top_k": args.top_k, "official_score": False, "histories": {},
                     "questions": {r["id"]: {"status": "pending", "persona_id": r["persona_id"],
                                             "pref_type": r.get("pref_type", "unknown")} for r in selection["rows"]}}
            if args.qa_cutoffs:
                state.update(qa_cutoffs=args.qa_cutoffs, mcq_format=args.mcq_format)
        def checkpoint():
            save(output / "state.json", state)
        checkpoint()
        print("Batch directory: " + str(output), flush=True)
        print("QA request policy: enable_thinking=false for Answer and Judge; cached results retained.", flush=True)
        if any(should_run(r, args.skip_failed) for r in state["questions"].values()):
            check_backends(env)
            diagnostic = DiagnosticLog(output / "diagnostics.jsonl")
            with memory_server(output, args.memory_profile, state["namespace"], env, args.allow_http_models) as client:
                run_questions(state, selection["rows"], client, pipeline, args, env, output, diagnostic, checkpoint)
        return 1 if any(r["status"] != "completed" for r in state["questions"].values()) else 0
    except KeyboardInterrupt:
        print("Interrupted; owned HTTP process stopped. Resume uses persisted checkpoints.", flush=True)
        return 130
    finally:
        if state:
            for record in state["questions"].values():
                if record["status"] == "running":
                    record["status"] = "interrupted"
                for cutoff in record.get("cutoffs", {}).values():
                    if cutoff["status"] == "running":
                        cutoff["status"] = "interrupted"
            save(output / "state.json", state)
            write_summary(output, state)
            print("Summary: " + str(output / "summary.md"), flush=True)
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())

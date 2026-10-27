"""CL-bench HTTP smoke: frozen follow-up input, unmodified AML Answer/Judge."""

import argparse
import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from dotenv import load_dotenv

from .clbench_data import ROOT, VENDOR_SHA, object_digest, official, payloads, prepare
from .diagnostics import DiagnosticLog, exception_info
from .lme_profiles import profile_files, repository_root
from .lme_smoke import check_backends, model_transport_overrides
from .models import SearchResponse
from .persona_batch import http_post, memory_server
from .persona_data import digest, save
from .state import ProcessLock


def model_config(env, role):
    prefix = "CL_" + role.upper()
    return {"base_url": env.get(prefix + "_BASE_URL") or env.get("LME_QA_BASE_URL") or env["AML_LLM_BASE_URL"],
            "api_key": env.get(prefix + "_API_KEY") or env.get("LME_QA_API_KEY") or env["AML_LLM_API_KEY"],
            "model": env.get(prefix + "_MODEL") or env.get("LME_QA_MODEL") or env["AML_LLM_MODEL"]}


def public_models(env):
    return {role: {k: v for k, v in model_config(env, role).items() if k != "api_key"}
            for role in ("answer", "judge")}


def fingerprint(args, selection, env):
    # Hash only relevant code/config; additions to unrelated benchmark scripts do
    # not invalidate ingestion. Provider credentials are intentionally excluded.
    paths = [ROOT / "aml_adapter" / name for name in
             ("clbench_batch.py", "clbench_data.py", "persona_batch.py", "persona_data.py",
              "lme_profiles.py", "lme_smoke.py", "backend.py", "service.py", "app.py",
              "models.py", "state.py", "schema_eval.py", "diagnostics.py")]
    paths += profile_files(args.memory_profile)
    paths += sorted((repository_root(args.memory_profile) / "jiuwen_memory").rglob("*.py"))
    keys = ("AML_LLM_MODEL", "AML_LLM_PROVIDER", "AML_LLM_BASE_URL", "AML_EMBED_MODEL",
            "AML_EMBED_BASE_URL", "AML_REDIS_URL", "AML_ES_URL", "AML_ES_INDEX",
            "AML_MILVUS_URI", "AML_MILVUS_COLLECTION")
    return object_digest({"selection": selection, "profile": args.memory_profile, "top_k": args.top_k,
                          "models": public_models(env), "vendor": VENDOR_SHA,
                          "settings": {k: env.get(k) for k in keys},
                          "files": {str(p.relative_to(ROOT.parent)): digest(p) for p in paths}})


class LoggedChatClient:
    def __init__(self, client, role, artifact, diagnostic):
        self.client, self.role = client, role
        self.artifact, self.diagnostic = artifact, diagnostic

    async def post(self, url, *, json, **kwargs):
        with self.diagnostic.span(self.role + ".chat", enable_thinking=False) as event:
            response = await self.client.post(url, json={**json, "enable_thinking": False}, **kwargs)
            event["status_code"] = response.status_code
            response.raise_for_status()
            body = response.json()
            # Store successful provider output privately for judge parsing audits;
            # never store authorization headers or gateway error bodies.
            save(self.artifact / f"{self.role}-response-{uuid4().hex[:12]}.json", body)
            return response


async def chat_stage(pipeline, role, row, prompt, answer, env, artifact, diagnostic):
    async with httpx.AsyncClient(timeout=900) as client:
        wrapped = LoggedChatClient(client, role, artifact, diagnostic)
        if role == "answer":
            coro = pipeline.post_chat(wrapped, **model_config(env, role),
                                      messages=[{"role": "user", "content": prompt}], temperature=0)
        else:
            coro = pipeline.evaluate_rubric_clbench(wrapped, **model_config(env, role),
                                                   rubrics=row["rubrics"], predicted_answer=answer, max_retries=3)
        task = asyncio.create_task(coro)
        started = time.monotonic()
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=60)
                if done:
                    return task.result()
                print(f"WAIT {role}: {row['id']}; elapsed={time.monotonic()-started:.0f}s", flush=True)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)


def answer_item(row, retrieved):
    # Preserve provider content order. No raw history, rubrics or references are
    # passed around retrieval into the answer model.
    return {"system_prompt": row["system_prompt"], "question": row["question"],
            "retrieval": {"selected": [{"text": item["content"]} for item in retrieved]}}


def judge_failed(score):
    reason = score.get("rubric_clbench_rationale", "")
    return (reason.startswith(("API call failed", "JSON parse failed", "Unknown error"))
            or reason == "No rubrics provided")


def validate_resume(prior, signature, profile, upgrades):
    if prior["profile"] != profile:
        raise ValueError("Resume profile changed")
    if prior["fingerprint"] == signature:
        return
    for upgrade in upgrades:
        if (upgrade["profile"] == profile and upgrade["from"] == prior["fingerprint"]
                and upgrade["to"] == signature):
            prior.setdefault("resume_upgrades", []).append(dict(upgrade))
            prior["fingerprint"] = signature
            print("RESUME UPGRADE: " + upgrade["reason"] + " Existing Add checkpoints retained.", flush=True)
            return
    raise ValueError("Resume contract changed; use original configuration or start a new batch")


def run_cases(state, client, pipeline, args, env, output, diagnostic, checkpoint):
    for row in state["selection"]["rows"]:
        record = state["questions"][row["id"]]
        if record["status"] == "completed" or (args.skip_failed and record["status"] == "failed"):
            continue
        artifact = output / "questions" / row["id"]
        artifact.mkdir(parents=True, exist_ok=True)
        record["status"] = "running"
        record.pop("error", None)
        checkpoint()
        try:
            chunks = payloads(row, record["user_id"])
            record["stage"] = "add"
            for index in range(record["next_add"], len(chunks)):
                print(f"ADD {row['category']} {index+1}/{len(chunks)}: {row['id']}", flush=True)
                started = time.monotonic()
                result = http_post(client, "/add", chunks[index], diagnostic)
                if result.get("success") is not True:
                    raise RuntimeError("Add did not report success")
                save(artifact / f"add-{index+1:03d}.json", {"response": result,
                                                          "elapsed_seconds": time.monotonic()-started})
                record["next_add"] = index + 1
                checkpoint()
            record["stage"] = "search"
            checkpoint()
            search_file = artifact / "search.json"
            if not search_file.exists():
                print("SEARCH: " + row["id"], flush=True)
                result = http_post(client, "/search", {"query": row["question"], "user_id": record["user_id"],
                                                       "top_k": args.top_k}, diagnostic)
                # Add has success=true; Search's actual HTTP contract is {data: [...]}.
                SearchResponse.model_validate(result)
                save(search_file, result)
            cached_search = json.loads(search_file.read_text(encoding="utf-8"))
            SearchResponse.model_validate(cached_search)
            retrieved = cached_search["data"]
            record.update(retrieved=len(retrieved), context_words=sum(len(x["content"].split()) for x in retrieved))
            record["stage"] = "answer"
            checkpoint()
            answer_file = artifact / "answer.json"
            if not answer_file.exists():
                prompt = pipeline.build_answer_prompt(answer_item(row, retrieved))
                save(artifact / "answer-prompt.json", {"prompt": prompt})
                print("ANSWER: " + row["id"], flush=True)
                answer = asyncio.run(chat_stage(pipeline, "answer", row, prompt, None, env, artifact, diagnostic))
                save(answer_file, {"model_output": answer})
            answer = json.loads(answer_file.read_text(encoding="utf-8"))["model_output"]
            record["stage"] = "judge"
            checkpoint()
            judge_file = artifact / "judge.json"
            if not judge_file.exists():
                print("JUDGE: " + row["id"], flush=True)
                score = asyncio.run(chat_stage(pipeline, "judge", row, None, answer, env, artifact, diagnostic))
                if judge_failed(score):
                    save(artifact / f"judge-failed-{uuid4().hex[:8]}.json", score)
                    raise RuntimeError("Official Judge could not complete; see judge-failed artifact")
                save(judge_file, score)
            score = json.loads(judge_file.read_text(encoding="utf-8"))
            if judge_failed(score):
                raise RuntimeError("Cached judge is a failure")
            record.update(score=score["rubric_clbench_score"],
                          requirement_ratio=score["rubric_clbench_requirement_ratio"],
                          status="completed", stage="done")
            print(f"DONE {row['id']}: strict_score={record['score']}; retrieved={record['retrieved']}", flush=True)
        except Exception as exc:
            record.update(status="failed", error=exception_info(exc))
            diagnostic.emit(record["stage"], "failed", question_id=row["id"], error=record["error"])
            checkpoint()
            print(f"FAILED {row['id']}: stage={record['stage']}; {type(exc).__name__}; continuing", flush=True)
            if record["stage"] in {"add", "search"} and isinstance(exc, httpx.TransportError):
                client.restart_owned_server()
        checkpoint()


def stats(records):
    rows = list(records)
    completed = [r for r in rows if r["status"] == "completed"]
    correct = sum(r["score"] for r in completed)
    return {"total": len(rows), "completed": len(completed), "failed": sum(r["status"] == "failed" for r in rows),
            "correct": correct, "accuracy_on_completed": correct / len(completed) if completed else None,
            "accuracy_full_sample": correct / len(rows) if rows and len(completed) == len(rows) else None,
            "requirement_ratio_diagnostic": (sum(r["requirement_ratio"] for r in completed) / len(completed)
                                             if completed else None)}


def summarize(output, state):
    report = {"profile": state["profile"], "official_score": False,
              "sampling_policy": state["selection"]["policy"], **stats(state["questions"].values())}
    report["by_category"] = {c: stats(r for r in state["questions"].values() if r["category"] == c)
                             for c in sorted({r["category"] for r in state["questions"].values()})}
    report["failures"] = [{"id": k, "stage": r["stage"]} for k, r in state["questions"].items()
                          if r["status"] == "failed"]
    save(output / "summary.json", report)
    lines = [f"# CL-bench smoke — {state['profile']}", "",
             "Local follow-up sample; AML public Answer/Judge; not an official competition score.",
             "Sampling: " + state["selection"]["policy"],
             f"Completed {report['completed']}/{report['total']}; failed {report['failed']}.",
             f"Strict accuracy on completed: {report['accuracy_on_completed']}; full sample: {report['accuracy_full_sample']}.",
             "Requirement ratio is diagnostic only; strict all-or-nothing scoring is unchanged.", "",
             "| Category | Completed | Correct | Failed |", "| --- | --- | --- | --- |"]
    for category, values in report["by_category"].items():
        lines.append(f"| {category} | {values['completed']}/{values['total']} | {values['correct']} | {values['failed']} |")
    lines += ["", "Failed IDs: " + (", ".join(x["id"] for x in report["failures"]) or "none")]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Completed={report['completed']}/{report['total']}; failed={report['failed']}", flush=True)


def compare(paths):
    states = [json.loads((Path(p) / "state.json").read_text(encoding="utf-8")) for p in paths]
    for field in ("selection", "models", "top_k"):
        if states[0][field] != states[1][field]:
            raise ValueError("Cannot compare different " + field)
    ids = [qid for qid in states[0]["questions"]
           if all(s["questions"][qid]["status"] == "completed" for s in states)]
    report = {"profiles": [s["profile"] for s in states], "paired_ids": ids,
              "all": [stats(s["questions"].values()) for s in states],
              "paired": [stats(s["questions"][qid] for qid in ids) for s in states],
              "cases": [{"id": qid, "category": states[0]["questions"][qid]["category"],
                         "scores": [s["questions"][qid]["score"] for s in states]} for qid in ids]}
    save(Path(paths[1]) / "comparison.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-profile", choices=("ordinary", "schema-entity", "schema-temporal"),
                        default="ordinary")
    parser.add_argument("--per-category", type=int, default=1)
    parser.add_argument("--sampling", choices=("smoke", "stratified"), default="smoke")
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
    if not 1 <= args.top_k <= 100 or not 1 <= args.per_category <= 20:
        parser.error("top-k must be 1..100; per-category must be 1..20")
    if args.skip_failed and not args.resume:
        parser.error("skip-failed requires resume")
    selection, path = prepare(args.per_category, sampling=args.sampling)
    pipeline = official()
    adds = sum(len(payloads(row, "plan")) for row in selection["rows"])
    print(f"PLAN: CL-bench; {len(selection['rows'])} questions; {adds} Add requests; top_k={args.top_k}")
    print(f"Selection: {path}\nSampling: {selection['policy']}; no context truncation.")
    print(f"Distinct contexts: {len({r['context_id'] for r in selection['rows']})}; memory profile: {args.memory_profile}")
    print("AML official Answer/Judge functions; enable_thinking=false; 2 model calls/question before retries.")
    for row in selection["rows"]:
        band = row.get("history_band", "smoke")
        print(f"  {row['category']}: {row['id']}; history_words={row['history_words']}; band={band}")
    if not args.run:
        return 0
    if not Path(args.env_file).is_file():
        raise ValueError("Environment file does not exist")
    load_dotenv(args.env_file, override=True)
    env = dict(os.environ)
    model_transport_overrides(env, args.allow_http_models)
    for role in ("answer", "judge"):
        config = model_config(env, role)
        url = urlsplit(config["base_url"])
        if (url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password
                or url.query or url.fragment or not config["api_key"] or not config["model"]):
            raise ValueError("Invalid model configuration: " + role)
        if url.scheme == "http" and not args.allow_http_models:
            raise ValueError("HTTP models require -AllowHttpModels")
    # Reuse the established interruptible memory server, with a benchmark-specific timeout override.
    env["PM_MEMORY_HTTP_TIMEOUT_SECONDS"] = env.get("CL_MEMORY_HTTP_TIMEOUT_SECONDS", "14400")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output = (args.resume or ROOT / f"outputs/clbench/{args.memory_profile}-{stamp}-{uuid4().hex[:8]}").resolve()
    if args.resume and not (output / "state.json").is_file():
        raise ValueError("Resume requires a batch directory containing state.json")
    output.mkdir(parents=True, exist_ok=True)
    lock = ProcessLock(output / "runner.lock")
    state = None
    try:
        signature = fingerprint(args, selection, env)
        if args.resume:
            prior = json.loads((output / "state.json").read_text(encoding="utf-8"))
            upgrades_path = ROOT / "config/clbench-resume-upgrades.json"
            upgrades = json.loads(upgrades_path.read_text(encoding="utf-8")) if upgrades_path.exists() else []
            previous_fingerprint = prior["fingerprint"]
            validate_resume(prior, signature, args.memory_profile, upgrades)
            if previous_fingerprint != signature:
                backup = output / ("state-before-upgrade-" + previous_fingerprint + ".json")
                if not backup.exists():
                    save(backup, json.loads((output / "state.json").read_text(encoding="utf-8")))
            state = prior
        else:
            namespace = "clbench-" + uuid4().hex
            state = {"profile": args.memory_profile, "fingerprint": signature, "selection": selection,
                     "namespace": namespace, "models": public_models(env), "top_k": args.top_k,
                     "questions": {r["id"]: {"status": "pending", "stage": "pending", "next_add": 0,
                                             "category": r["category"], "user_id": namespace + ":" + r["id"]}
                                   for r in selection["rows"]}}

        def checkpoint():
            save(output / "state.json", state)

        checkpoint()
        print("Batch directory: " + str(output), flush=True)
        todo = [r for r in state["questions"].values() if r["status"] != "completed"
                and not (args.skip_failed and r["status"] == "failed")]
        if todo:
            check_backends(env)
            diagnostic = DiagnosticLog(output / "diagnostics.jsonl")
            with memory_server(output, args.memory_profile, state["namespace"], env, args.allow_http_models) as client:
                run_cases(state, client, pipeline, args, env, output, diagnostic, checkpoint)
        return 1 if any(r["status"] != "completed" for r in state["questions"].values()) else 0
    except KeyboardInterrupt:
        print("Interrupted; owned HTTP worker stopped; resume keeps completed Add/Search/QA.", flush=True)
        return 130
    finally:
        try:
            if state is not None:
                for record in state["questions"].values():
                    if record["status"] == "running":
                        record["status"] = "interrupted"
                save(output / "state.json", state)
                summarize(output, state)
                print("Summary: " + str(output / "summary.md"), flush=True)
        finally:
            lock.close()


if __name__ == "__main__":
    raise SystemExit(main())

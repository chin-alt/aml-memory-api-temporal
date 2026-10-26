"""启动本地 HTTP 记忆系统，调用仓库原 run.main / Runner / LLMJudge。"""

import argparse
import json
import os
import secrets
import socket
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import httpx
import yaml
from dotenv import load_dotenv

from .diagnostics import DiagnosticLog, environment_info, exception_info
from .lme_smoke import ROOT, REQUIRED, check_backends, model_transport_overrides
from .lme_profiles import PROFILES, memory_config, select_repository

# evaluation 是仓库脚本目录，不在已安装的 jiuwen_memory wheel 中。
REPOSITORY_ROOT = Path(os.environ.get("AML_LME_REPOSITORY", ROOT.parent / "agent-memory"))
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def normalize_result(result):
    records = result.get("qa_records", [])
    if len(records) != 1 or set(records[0].get("cutoffs", {})) != {"50", "10", "20"}:
        raise RuntimeError("Repository result must contain one QA record with cutoffs 50/10/20")
    record = records[0]
    return {"protocol_success": True, "official_score": False,
            "qa_profile": "repository_mem0_longmemeval",
            "qa": {"hypothesis": record["model_answer"], "reference": record["gold_answer"],
                   "verdict": {"correct": record["score"] == 1.0}, "primary_cutoff": 50},
            "qa_cutoffs": {key: value["score"] == 1.0 for key, value in record["cutoffs"].items()},
            "evaluation_protocol": result["evaluation_protocol"],
            "metrics": result["metrics"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--env-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--answer-cutoff", type=int, default=50)
    parser.add_argument("--allow-http-models", action="store_true")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--oracle-sessions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--memory-profile", choices=PROFILES, default="ordinary")
    args = parser.parse_args(argv)
    repository = select_repository(args.memory_profile)
    if args.answer_cutoff != 50 or args.retrieval_only:
        parser.error("Repository QA profile requires primary cutoff 50 and all three QA cutoffs")
    load_dotenv(args.env_file, override=True)
    env = dict(os.environ)
    missing = [key for key in REQUIRED if not env.get(key)]
    if missing:
        raise ValueError("Missing model settings: " + ", ".join(missing))
    overrides = model_transport_overrides(env, args.allow_http_models)
    check_backends(env)
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Case output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    diagnostic = DiagnosticLog(output / "diagnostics.jsonl")
    diagnostic.emit("repository.run", "start", environment=environment_info(env))
    config = memory_config(args.memory_profile)
    config["globals"].update(overrides)
    if args.memory_profile == "schema-entity":
        schema = config["extractor"]["default"]["params"]
        frozen_schema = output / "schema.json"
        frozen_schema.write_bytes(Path(schema["schema_path"]).read_bytes())
        schema["schema_path"] = str(frozen_schema)
    fidelity = bool(config["globals"].get("extraction_fidelity_mode", False))
    config_path = output / "memory.yml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    run_id = "repo-http-" + uuid4().hex
    token = secrets.token_urlsafe(32)
    env.update(AML_API_KEY=token, AML_STATE_DIR=str(output / "state"),
               AML_MEMORY_CONFIG=str(config_path), AML_NAMESPACE=run_id,
               AML_LOCAL_SMOKE="false", AML_MAX_CONCURRENCY="1", AML_REPO_EVAL="true",
               AML_EXTRACTION_FIDELITY=str(fidelity).lower(),
               AML_DIAGNOSTIC_LOG=str(output / "api-diagnostics.jsonl"),
               PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    env["PYTHONPATH"] = os.pathsep.join((str(repository), str(ROOT)))
    env["AML_LME_REPOSITORY"] = str(repository)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    endpoint = f"http://127.0.0.1:{port}"
    report = {"protocol_success": False, "official_score": False,
              "memory_profile": args.memory_profile, "memory_repository": str(repository),
              "extraction_fidelity_mode": fidelity, "fuser_backend": config["fuser"]["default"]["target"],
              "qa_profile": "repository_mem0_longmemeval"}
    process = None
    try:
        with (output / "api.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-m", "uvicorn", "aml_adapter.app:create_app",
                "--factory", "--host", "127.0.0.1", "--port", str(port), "--workers", "1", "--no-access-log"],
                cwd=ROOT, env=env, stdout=log, stderr=log,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            with httpx.Client(timeout=1, trust_env=False) as client:
                for _ in range(150):
                    if process.poll() is not None:
                        raise RuntimeError("Memory HTTP server exited; inspect api.log")
                    try:
                        if client.get(endpoint + "/health").status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(0.2)
                else:
                    raise RuntimeError("Memory HTTP server startup timed out")
            from evaluation.longmemeval import run as repository_run

            updates = {"LME_MEMORY_API_KEY": token,
                       "JUDGE_BASE_URL": env.get("LME_QA_BASE_URL") or env["AML_LLM_BASE_URL"],
                       "JUDGE_API_KEY": env.get("LME_QA_API_KEY") or env["AML_LLM_API_KEY"],
                       "JUDGE_MODEL": env.get("LME_QA_MODEL") or env["AML_LLM_MODEL"],
                       "LME_CLIENT_API_AUDIT_PATH": str(output / "client_api_audit.jsonl")}
            previous = {key: os.environ.get(key) for key in updates}
            os.environ.update(updates)
            try:
                arguments = ["--data", str(Path(args.data).resolve()), "--samples", str(args.sample_index),
                             "--max-questions", "1", "--config", str(config_path),
                             "--http-memory-url", endpoint, "--run-id", run_id,
                             "--output-dir", str(output), "--json", str(output / "repository-result.json"),
                             "--granularity", "dialogue_turn", "--dialogue-turn-max-chars", "4096",
                             "--oracle-sessions" if args.oracle_sessions else "--no-oracle-sessions",
                             "--infer", "--concurrency", "1",
                             "--recall-top-k", "200", "--answer-cutoff", "50", "--answer-cutoffs", "50,10,20",
                             "--answer-temperature", "0", "--answer-max-tokens", "16384",
                             "--judge-temperature", "0", "--judge-max-tokens", "4096", "--llm-timeout", "900"]
                print(f"REPOSITORY: {args.memory_profile} -> HTTP /add,/search -> original QA @50/10/20", flush=True)
                with diagnostic.span("repository.run"):
                    code = repository_run.main(arguments)
                if code:
                    raise RuntimeError("Repository evaluator failed")
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
            result = json.loads((output / "repository-result.json").read_text(encoding="utf-8"))
            expected_mode = "oracle_sessions" if args.oracle_sessions else "full_haystack"
            if result.get("evaluation_protocol", {}).get("context_mode") != expected_mode:
                raise RuntimeError("Repository result history mode does not match the requested experiment")
            if result.get("evaluation_protocol", {}).get("extraction_fidelity_mode") is not fidelity:
                raise RuntimeError("Repository result fidelity mode does not match the memory profile")
            report.update(normalize_result(result))
            report["repository_result"] = str(output / "repository-result.json")
            print("Repository QA: " + json.dumps(report["qa_cutoffs"]), flush=True)
            return 0
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        report["failure"] = exception_info(exc)
        diagnostic.emit("repository.run", "failed", error=report["failure"])
        print(f"FAILED: {type(exc).__name__}; inspect diagnostics.jsonl / api-diagnostics.jsonl")
        return 1
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if args.memory_profile == "schema-entity":
            events_path = output / "api-diagnostics.jsonl"
            events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()
                      if line.strip()] if events_path.exists() else []
            adds = [e for e in events if e.get("stage") == "schema.add" and e.get("event") == "ok"]
            expansions = [e for e in events if e.get("stage") == "entity.expand"]
            report["feature_usage"] = {
                "successful_schema_adds": len(adds),
                "property_units_returned": sum(e["property_units"] for e in adds),
                "units_with_entities_returned": sum(e["units_with_entities"] for e in adds),
                "entity_expansion_calls": len(expansions),
                "entity_expansion_candidates": sum(e.get("added_candidates", 0) for e in expansions),
            }
        (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Result: " + str(output / "result.json"), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

"""按已冻结清单跑 HTTP LongMemEval；支持题目并发，--run 才调用模型。"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import dotenv_values

from .lme_sampling import BUCKETS
from .repo_lme import main as run_case
from .lme_profiles import PROFILES, profile_files, repository_root, select_repository

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/lme-s-21-seed42/manifest.json"
EXPERIMENTS = {"oracle5": "data/lme-oracle-30-answerable-seed42/manifest.json",
               "s1": "data/lme-s-6-answerable-seed42/manifest.json"}


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_selection(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    data = (path.parent / manifest["dataset"]).resolve()
    raw = data.read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["dataset_sha256"]:
        raise ValueError("Selected dataset changed; prepare a new selection")
    rows = json.loads(raw)
    if [r["question_id"] for r in rows] != [c["question_id"] for c in manifest["cases"]]:
        raise ValueError("Manifest and selected dataset question order differ")
    return manifest, data


def fingerprint(env_file, cutoff, retrieval_only, memory_profile="ordinary"):
    env = {**os.environ, **dotenv_values(env_file)}
    names = ("AML_LLM_MODEL", "AML_LLM_PROVIDER", "AML_LLM_BASE_URL", "AML_EMBED_MODEL", "AML_EMBED_BASE_URL",
             "AML_REDIS_URL", "AML_ES_URL", "AML_ES_INDEX", "AML_MILVUS_URI", "AML_MILVUS_COLLECTION",
             "AML_EXTRACTION_FIDELITY", "LME_QA_MODEL", "LME_QA_BASE_URL")
    files = ("config/memory.yml", "aml_adapter/backend.py", "aml_adapter/repo_lme.py",
             "aml_adapter/service.py", "aml_adapter/app.py", "aml_adapter/models.py",
             "aml_adapter/lme_batch.py", "aml_adapter/lme_parallel.py", "aml_adapter/state.py",
             "aml_adapter/lme_profiles.py", "aml_adapter/schema_eval.py", "aml_adapter/diagnostics.py")
    config = {"settings": {name: env.get(name) for name in names}, "answer_cutoff": cutoff,
              "retrieval_only": retrieval_only,
              "code": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}}
    repository = repository_root(memory_profile) / "evaluation/longmemeval"
    config["repository_code"] = {str(path.relative_to(repository)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in repository.rglob("*.py")}
    config["memory_profile"] = memory_profile
    config["profile_files"] = {str(path.relative_to(ROOT.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in profile_files(memory_profile)}
    core = repository_root(memory_profile) / "jiuwen_memory"
    config["core_code"] = {str(path.relative_to(core)): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in core.rglob("*.py")}
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def summarize(cases, retrieval_only=False):
    def cutoff_scores(rows):
        scores = {}
        for cutoff in ("50", "10", "20"):
            values = [row["qa_cutoffs"][cutoff] for row in rows
                      if row.get("status") == "completed"
                      and isinstance(row.get("qa_cutoffs", {}).get(cutoff), bool)]
            correct = sum(values)
            scores[cutoff] = {"graded": len(values), "correct": correct,
                              "accuracy_on_graded": correct / len(values) if values else None,
                              "subset_accuracy": correct / len(rows) if rows and len(values) == len(rows) else None}
        return scores

    groups = {}
    for name in BUCKETS:
        rows = [case for case in cases if case["bucket"] == name]
        completed = [case for case in rows if case.get("status") == "completed"]
        graded = [case for case in completed if isinstance(case.get("correct"), bool)]
        correct = sum(case["correct"] for case in graded)
        groups[name] = {"planned": len(rows), "completed": len(completed), "graded": len(graded),
                        "correct": correct, "accuracy_on_graded": correct / len(graded) if graded else None,
                        "qa_cutoffs": cutoff_scores(rows)}
    planned = len(cases)
    completed = sum(group["completed"] for group in groups.values())
    graded = sum(group["graded"] for group in groups.values())
    correct = sum(group["correct"] for group in groups.values())
    return {"planned": planned, "completed": completed, "graded": graded, "correct": correct,
            "failed_or_interrupted": sum(case.get("status") in {"failed", "interrupted"} for case in cases),
            "pending": sum(case.get("status", "pending") == "pending" for case in cases),
            "running": sum(case.get("status") == "running" for case in cases),
            "accuracy_on_graded": correct / graded if graded else None,
            "all_cases_completed": completed == planned,
            "subset_accuracy": correct / planned if planned and graded == planned and not retrieval_only else None,
            "official_score": False, "primary_cutoff": 50,
            "qa_cutoffs": cutoff_scores(cases), "by_bucket": groups}


def apply_compatible_upgrade(state, identity, profile, upgrades):
    for upgrade in upgrades:
        if (upgrade.get("memory_profile", "ordinary") == profile
                and state["fingerprint"] == upgrade["from"] and identity == upgrade["to"]):
            state.setdefault("runner_upgrades", []).append(upgrade)
            state["fingerprint"] = identity
            break


def write_summary(output, state):
    summary = summarize(state["cases"], state["retrieval_only"])
    context_mode = "oracle_sessions" if state["selection"]["variant"] == "oracle" else "full_haystack"
    summary.update(dataset_variant=state["selection"]["variant"], context_mode=context_mode,
                   memory_profile=state.get("memory_profile", "ordinary"),
                   case_concurrency=state.get("case_concurrency", 1),
                   answerable_only=state["selection"].get("answerable_only", False))
    save(output / "summary.json", summary)
    text = ["# LongMemEval 抽样结果", "",
            f"数据：{summary['dataset_variant']}；历史模式：{context_mode}；仅有标准答案题：{summary['answerable_only']}。", "",
            f"完成 {summary['completed']}/{summary['planned']}；判分 {summary['graded']}；正确 {summary['correct']}。", "",
            "未完成整批时 subset_accuracy 为 null；accuracy_on_graded 只代表已判分题，不能忽略失败题当作整批成绩。", "",
            f"本结果调用仓库原评测脚本与 QA，记忆经 HTTP；配置 {summary['memory_profile']}；不是官方 AML 分数。", "",
            "主结果采用 Top-50；下列正确率均以已判分题为分母。三档结果也分别保存在 summary.json。", "",
            "|类别|计划|完成|判分|Top-50 正确|Top-50 正确率|Top-10 正确率|Top-20 正确率|",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in summary["by_bucket"].items():
        accuracy = "—" if row["accuracy_on_graded"] is None else f"{row['accuracy_on_graded']:.1%}"
        other = ["—" if row["qa_cutoffs"][key]["accuracy_on_graded"] is None
                 else f"{row['qa_cutoffs'][key]['accuracy_on_graded']:.1%}" for key in ("10", "20")]
        text.append(f"|{name}|{row['planned']}|{row['completed']}|{row['graded']}|{row['correct']}|{accuracy}|{'|'.join(other)}|")
    text.extend(["", "|证据上限|已判分|正确|已判分正确率|", "|---|---:|---:|---:|"])
    for cutoff, row in summary["qa_cutoffs"].items():
        accuracy = "—" if row["accuracy_on_graded"] is None else f"{row['accuracy_on_graded']:.1%}"
        text.append(f"|Top-{cutoff}|{row['graded']}|{row['correct']}|{accuracy}|")
    failed_cases = [case for case in state["cases"] if case["status"] in {"failed", "interrupted"}]
    if failed_cases:
        text.extend(["", "失败或中断题（未计入已判分准确率；可通过 Resume 重跑）：", ""])
        for case in failed_cases:
            text.append(f"- 第 {case['subset_index'] + 1} 题：{case['question_id']}；{case['status']}")
    (output / "summary.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS))
    parser.add_argument("--memory-profile", choices=PROFILES, default="ordinary")
    parser.add_argument("--concurrency", type=int, choices=range(1, 7), default=1,
                        help="Concurrent questions in separate processes (1-6); each question remains serial")
    parser.add_argument("--env-file", default=str(ROOT / "config/longmemeval.local.env"))
    parser.add_argument("--answer-cutoff", type=int, default=50)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", help="Existing batch directory; completed cases are not re-run")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Stop dispatching after a case fails; by default continue remaining cases")
    parser.add_argument("--skip-failed", action="store_true",
                        help="On resume, preserve failed cases and run only pending/interrupted cases")
    parser.add_argument("--allow-http-models", action="store_true")
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args(argv)
    if args.skip_failed and not args.resume:
        parser.error("--skip-failed requires --resume")
    if args.memory_profile != "ordinary":
        select_repository(args.memory_profile)
    if args.answer_cutoff != 50 or args.retrieval_only:
        parser.error("Repository QA mode requires primary cutoff 50 and all QA cutoffs; use the single smoke script for retrieval-only")
    if args.manifest and args.experiment:
        parser.error("Use --experiment or --manifest, not both")
    manifest = (ROOT / EXPERIMENTS[args.experiment] if args.experiment
                else Path(args.manifest) if args.manifest else DEFAULT_MANIFEST)
    selection, data = load_selection(manifest)
    if selection["variant"] not in {"oracle", "s", "m"}:
        parser.error("Evaluation requires an Oracle/S/M manifest with an explicit history mode")
    oracle_sessions = selection["variant"] == "oracle"
    print(f"PLAN: {selection['variant']}; {len(selection['cases'])} cases; seed={selection['seed']}")
    print(json.dumps(selection["selected_by_bucket"], ensure_ascii=True))
    print("Raw history totals (add_chunks is the legacy message-batch estimate, not repository Add calls): "
          + json.dumps(selection["totals"]))
    from evaluation.longmemeval.adapter import LongMemEvalDataset
    dataset = LongMemEvalDataset(str(data), granularity="dialogue_turn", oracle_sessions=oracle_sessions,
                                infer=True, answer_cutoffs=(50, 10, 20), dialogue_turn_max_chars=4096)
    print(f"Repository plan: {len(dataset.seeds())} HTTP Add requests; "
          f"{len(dataset.queries())} Search requests; {len(dataset.queries()) * 6} Answer/Judge calls before retries")
    print("ORACLE: oracle_sessions=true; evidence sessions only" if oracle_sessions else
          "FULL HISTORY: oracle_sessions=false; all sessions retained, no answer-session filtering")
    print(f"Repository run.py / Runner / LLMJudge; HTTP {args.memory_profile}; recall Top-200; QA cutoffs 50,10,20")
    if args.memory_profile == "schema-entity":
        print("MEMORY: PR #337; Schema + entity association; examples/entity_schema_locomo_person.json")
    elif args.memory_profile == "bm25-fidelity":
        print("MEMORY: ordinary extraction; BM25_scored_fuser; extraction_fidelity_mode=true; Schema/entity=false")
    elif args.memory_profile == "score-max-fidelity":
        print("MEMORY: ordinary extraction; score_max; extraction_fidelity_mode=true; Schema/entity=false")
    print(f"Question concurrency: {args.concurrency}; each question uses sequential Add and QA")
    print("Failure policy: " + ("stop dispatching" if args.stop_on_error else "record failed case and continue"))
    if args.skip_failed:
        print("Resume policy: skip completed AND failed cases; run pending/interrupted cases only")
    if not args.run:
        print("PLAN ONLY: no database or model requests. Pass -Run in PowerShell (--run in Python) to execute.")
        return 0
    env_file = Path(args.env_file).resolve()
    if not env_file.is_file():
        raise ValueError("Model env file does not exist")
    identity = fingerprint(env_file, args.answer_cutoff, args.retrieval_only, args.memory_profile)
    if args.resume:
        output = Path(args.resume).resolve()
        state = json.loads((output / "batch.json").read_text(encoding="utf-8"))
        if state.get("memory_profile", "ordinary") != args.memory_profile:
            raise ValueError("Memory profile changed; use a new batch for the ablation")
        upgrades = ROOT / "config/lme-compatible-upgrades.json"
        if upgrades.exists():
            apply_compatible_upgrade(state, identity, args.memory_profile,
                                     json.loads(upgrades.read_text(encoding="utf-8")))
        if state["fingerprint"] != identity or state["selection"] != selection:
            raise ValueError("Selection/config/code changed; use a new batch for comparable results")
        if hashlib.sha256((output / "selected.json").read_bytes()).hexdigest() != selection["dataset_sha256"]:
            raise ValueError("Frozen batch dataset changed; refusing to resume")
        for case in state["cases"]:
            if case["status"] == "running":
                case["status"] = "interrupted"
    else:
        profile_suffix = "" if args.memory_profile == "ordinary" else args.memory_profile + "-"
        name = (selection["variant"] + "-" + str(len(selection["cases"])) + "-" + profile_suffix
                + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8])
        output = ROOT / "outputs/longmemeval-batch" / name
        output.mkdir(parents=True)
        state = {"version": 1, "fingerprint": identity, "selection": selection,
                 "memory_profile": args.memory_profile,
                 "answer_cutoff": args.answer_cutoff, "retrieval_only": args.retrieval_only,
                 "cases": [{**case, "status": "pending", "attempts": []} for case in selection["cases"]]}
        # 冻结本轮输入，便于核查和复现。
        (output / "selected.json").write_bytes(data.read_bytes())
    state["case_concurrency"] = args.concurrency
    state["stop_on_error"] = args.stop_on_error
    state.setdefault("execution_history", []).append({
        "started_at": datetime.now(timezone.utc).isoformat(), "case_concurrency": args.concurrency,
        "stop_on_error": args.stop_on_error, "skip_failed": args.skip_failed})
    save(output / "batch.json", state)
    write_summary(output, state)
    print("Batch directory: " + str(output), flush=True)
    if args.concurrency > 1:
        from .lme_parallel import run_parallel

        def checkpoint():
            save(output / "batch.json", state)
            write_summary(output, state)

        return run_parallel(output, state, args, env_file, ROOT, checkpoint)
    for case in state["cases"]:
        if case["status"] == "completed" or (args.skip_failed and case["status"] == "failed"):
            continue
        attempt = output / f"case-{case['subset_index'] + 1:03d}" / f"attempt-{len(case['attempts']) + 1:02d}"
        case["status"] = "running"
        case["attempts"].append(str(attempt))
        save(output / "batch.json", state)
        print(f"CASE {case['subset_index'] + 1}/{len(state['cases'])}: {case['bucket']} {case['question_id']}", flush=True)
        arguments = ["--data", str(output / "selected.json"), "--sample-index", str(case["subset_index"]),
                     "--env-file", str(env_file), "--output-dir", str(attempt),
                     "--answer-cutoff", str(args.answer_cutoff),
                     "--memory-profile", args.memory_profile,
                     "--oracle-sessions" if oracle_sessions else "--no-oracle-sessions"]
        if args.allow_http_models:
            arguments.append("--allow-http-models")
        if args.retrieval_only:
            arguments.append("--retrieval-only")
        try:
            code = run_case(arguments)
            result_path = attempt / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
            correct = result.get("qa", {}).get("verdict", {}).get("correct")
            complete = code == 0 and result.get("protocol_success") and (
                args.retrieval_only or isinstance(correct, bool))
            case["status"] = "completed" if complete else "failed"
            case["correct"] = correct
            case["qa_cutoffs"] = result.get("qa_cutoffs", {})
            case["result"] = str(result_path) if result_path.exists() else None
        except KeyboardInterrupt:
            case["status"] = "interrupted"
            return 130
        except Exception as exc:
            case["status"] = "failed"
            case["error_type"] = type(exc).__name__
            print("Batch case failed: " + type(exc).__name__)
        finally:
            save(output / "batch.json", state)
            write_summary(output, state)
        if case["status"] != "completed" and args.stop_on_error:
            print("STOPPED after failure. Fix the cause and resume this batch directory; completed cases will be skipped.")
            return 1
    from .lme_parallel import report_completion
    return report_completion(output, state)


if __name__ == "__main__":
    raise SystemExit(main())

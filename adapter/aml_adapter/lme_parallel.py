"""题目级独立进程调度；父进程独占批次状态文件。"""

import json
import os
import signal
import subprocess
import sys
import time

from .lme_profiles import repository_root


def stop_worker(process):
    """仅终止本调度器启动且仍存活的 worker 及其 HTTP 子进程。"""
    if process.poll() is not None:
        return
    if os.name == "nt":
        result = subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode and process.poll() is None:
            raise RuntimeError(f"Could not stop owned process tree {process.pid}")
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            raise
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def run_parallel(output, state, args, env_file, root, checkpoint):
    pending = iter(case for case in state["cases"] if case["status"] != "completed"
                   and not (getattr(args, "skip_failed", False) and case["status"] == "failed"))
    active = []
    failed = False
    stop_dispatch = False
    stop_on_error = getattr(args, "stop_on_error", False)
    exhausted = False
    last_status = time.monotonic()
    oracle = state["selection"]["variant"] == "oracle"
    worker_env = dict(os.environ)
    profile = getattr(args, "memory_profile", "ordinary")
    repository = repository_root(profile) if profile != "ordinary" else root.parent / "agent-memory"
    worker_env.update(PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
                      AML_LME_REPOSITORY=str(repository),
                      PYTHONPATH=os.pathsep.join((str(repository), str(root))))
    options = ({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt"
               else {"start_new_session": True})

    def finish(job, code):
        case, attempt, _, log = job
        log.close()
        result_path = attempt / "result.json"
        try:
            result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
            correct = result.get("qa", {}).get("verdict", {}).get("correct")
            case["status"] = "completed" if (code == 0 and result.get("protocol_success")
                                               and isinstance(correct, bool)) else "failed"
            case["correct"] = correct
            case["qa_cutoffs"] = result.get("qa_cutoffs", {})
            case["result"] = str(result_path) if result_path.exists() else None
        except Exception as exc:
            case["status"] = "failed"
            case["error_type"] = type(exc).__name__
        case["exit_code"] = code
        print(f"CASE {case['subset_index'] + 1}: {case['status']}; log={case['runner_log']}", flush=True)

    try:
        while True:
            # 单题失败默认继续补充空槽；显式 stop_on_error 才停止派发。
            for job in active[:]:
                code = job[2].poll()
                if code is not None:
                    finish(job, code)
                    active.remove(job)
                    failed |= job[0]["status"] != "completed"
                    stop_dispatch |= failed and stop_on_error
                    checkpoint()
            while not stop_dispatch and not exhausted and len(active) < args.concurrency:
                case = next(pending, None)
                if case is None:
                    exhausted = True
                    break
                attempt = output / f"case-{case['subset_index'] + 1:03d}" / f"attempt-{len(case['attempts']) + 1:02d}"
                attempt.parent.mkdir(parents=True, exist_ok=True)
                log_path = attempt.with_suffix(".runner.log")
                case.update(status="running", runner_log=str(log_path))
                case["attempts"].append(str(attempt))
                checkpoint()
                arguments = [sys.executable, "-m", "aml_adapter.repo_lme",
                             "--data", str(output / "selected.json"),
                             "--sample-index", str(case["subset_index"]),
                             "--env-file", str(env_file), "--output-dir", str(attempt),
                             "--answer-cutoff", str(args.answer_cutoff),
                             "--oracle-sessions" if oracle else "--no-oracle-sessions"]
                if args.allow_http_models:
                    arguments.append("--allow-http-models")
                if getattr(args, "memory_profile", "ordinary") != "ordinary":
                    arguments.extend(["--memory-profile", args.memory_profile])
                log = log_path.open("w", encoding="utf-8")
                try:
                    process = subprocess.Popen(arguments, cwd=root, env=worker_env,
                                               stdout=log, stderr=log, **options)
                except Exception as exc:
                    log.close()
                    case.update(status="failed", error_type=type(exc).__name__)
                    failed = True
                    # 无法创建 worker 属于调度故障，不当作普通题目失败继续启动。
                    stop_dispatch = True
                    print(f"DISPATCH ERROR: {type(exc).__name__}; stopping new workers.", flush=True)
                    checkpoint()
                    break
                active.append((case, attempt, process, log))
                print(f"START CASE {case['subset_index'] + 1}/{len(state['cases'])}: "
                      f"{case['question_id']}; pid={process.pid}; log={log_path}", flush=True)
            if not active:
                break
            if time.monotonic() - last_status >= 30:
                print(f"RUNNING: {len(active)} cases; "
                      f"completed={sum(c['status'] == 'completed' for c in state['cases'])}; "
                      f"failed={sum(c['status'] == 'failed' for c in state['cases'])}; "
                      f"dispatch={'stopped after failure' if stop_dispatch else 'active'}", flush=True)
                last_status = time.monotonic()
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("Interrupted: stopping this batch's workers and HTTP child processes.", flush=True)
        return 130
    finally:
        cleanup_failed = []
        for job in active:
            try:
                code = job[2].poll()
                if code is not None:
                    finish(job, code)
                else:
                    stop_worker(job[2])
                    job[0]["status"] = "interrupted"
            except Exception as exc:
                job[0].update(status="interrupted", cleanup_error_type=type(exc).__name__)
                cleanup_failed.append(job[2].pid)
            finally:
                job[3].close()
        checkpoint()
        if cleanup_failed:
            raise RuntimeError(f"Could not confirm cleanup of owned worker PIDs: {cleanup_failed}")
    return report_completion(output, state, stopped=stop_dispatch)


def report_completion(output, state, *, stopped=False):
    failed = [str(c["subset_index"] + 1) for c in state["cases"] if c["status"] == "failed"]
    pending = sum(c["status"] != "completed" for c in state["cases"])
    if pending:
        prefix = "STOPPED" if stopped else "FINISHED WITH FAILURES"
        print(f"{prefix}: failed cases={','.join(failed) or 'none'}; unfinished={pending}. "
              "Resume skips completed cases.", flush=True)
    else:
        print("Completed.", flush=True)
    print("Summary: " + str(output / "summary.md"), flush=True)
    return 1 if pending else 0

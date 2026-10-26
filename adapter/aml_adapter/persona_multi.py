"""Per-cutoff PersonaMem QA checkpoints and paired reports for a single ingestion."""

import json

from .diagnostics import exception_info
from .persona_cutoffs import evaluate, stats
from .persona_data import save


class CutoffEvaluationError(RuntimeError):
    """One or more QA cutoffs failed; successful cutoffs remain reusable."""


def evaluate_cutoffs(row, record, retrieved, artifact, pipeline, args, env, diagnostic, checkpoint):
    records = record.setdefault("cutoffs", {})
    for cutoff in args.qa_cutoffs:
        result = records.setdefault(str(cutoff), {"status": "pending"})
        if result["status"] == "completed" or (args.skip_failed and result["status"] == "failed"):
            continue
        result["status"] = "running"
        checkpoint()
        print(f"QA {row['id']} top-{cutoff}: memories={min(cutoff, len(retrieved))}", flush=True)
        try:
            evaluate(row, result, retrieved, cutoff, artifact / f"top-{cutoff}", pipeline,
                     args, env, diagnostic, checkpoint)
            result.update(status="completed", stage="done")
            result.pop("error", None)
        except Exception as exc:
            result.update(status="failed", error=exception_info(exc))
            diagnostic.emit(result.get("stage", "qa"), "failed", question_id=row["id"],
                            cutoff=cutoff, error=result["error"])
            print(f"FAILED {row['id']} top-{cutoff}: {type(exc).__name__}; continuing other cutoffs", flush=True)
        checkpoint()
    failed = [k for k, value in records.items() if value["status"] != "completed"]
    if failed:
        raise CutoffEvaluationError("Incomplete QA cutoffs: " + ",".join(failed))


def cutoff_rows(state, cutoff):
    # History/Search failures have no QA artifacts; never treat them as successful zero scores.
    return {qid: question.get("cutoffs", {}).get(str(cutoff), {"status": question["status"]})
            for qid, question in state["questions"].items()}


def build_report(state):
    groups = {str(k): cutoff_rows(state, k) for k in state["qa_cutoffs"]}
    common = [qid for qid in state["questions"]
              if all(rows[qid]["status"] == "completed" for rows in groups.values())]
    def grouped(ids):
        return {k: stats(rows[qid] for qid in ids) for k, rows in groups.items()}
    report = {"profile": state["profile"], "top_k": state["top_k"], "mcq_format": state["mcq_format"],
              "total": len(state["questions"]), "completed": len(common),
              "failed": sum(q["status"] == "failed" for q in state["questions"].values()),
              "qa_cutoffs": grouped(state["questions"]), "paired_ids": common, "paired": grouped(common),
              "by_persona": {}, "by_preference_type": {}, "failures": []}
    for field, target in [("persona_id", "by_persona"), ("pref_type", "by_preference_type")]:
        for value in sorted({q[field] for q in state["questions"].values()}):
            ids = [qid for qid, q in state["questions"].items() if q[field] == value]
            report[target][value] = {"all": grouped(ids), "paired": grouped([qid for qid in ids if qid in common])}
    for k, rows in groups.items():
        for qid, result in rows.items():
            if result["status"] == "failed":
                report["failures"].append({"question": qid, "cutoff": int(k),
                    "stage": result.get("stage", state["questions"][qid].get("failure_stage"))})
    if state.get("runner_upgrades"):
        report["runner_upgrades"] = state["runner_upgrades"]
    if state.get("expansions"):
        report["expansions"] = state["expansions"]
    return report


def write_summary(output, state):
    report = build_report(state)
    save(output / "summary.json", report)
    lines = [f"# PersonaMem-v2 — {state['profile']}", "", "Local sample, not an official competition score.",
             f"All cutoffs completed: {report['completed']}/{report['total']}; failed questions: {report['failed']}.",
             f"MCQ prompt: {state['mcq_format']}. Strict scorer and narrow Judge are unchanged.",
             "Normalized MCQ is supplementary. Each cutoff uses its own completed-question denominator.", "",
             "| Top K | Completed | Failed | Strict MCQ | Normalized MCQ | Narrow mean |",
             "| --- | --- | --- | --- | --- | --- |"]
    for k, s in report["qa_cutoffs"].items():
        lines.append(f"| {k} | {s['completed']}/{s['total']} | {s['failed']} | {s['mcq_accuracy']} | "
                     f"{s['mcq_normalized_accuracy']} | {s['narrow_mean']} |")
    lines += ["", "Paired scores on common completed questions and type/user breakdowns: summary.json.",
              "", "Failures:"]
    lines += [f"- {f['question']} / top-{f['cutoff']} / {f['stage']}" for f in report["failures"]] or ["- none"]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"All cutoffs completed={report['completed']}/{report['total']}; failed questions={report['failed']}", flush=True)


def compare(paths, left, right):
    from pathlib import Path
    for field in ("selection", "models", "mode", "top_k", "qa_cutoffs", "mcq_format"):
        if left.get(field) != right.get(field):
            raise ValueError("Cannot pair different " + field)
    report = {"profiles": [left["profile"], right["profile"]], "qa_cutoffs": {}}
    for k in left["qa_cutoffs"]:
        groups = [cutoff_rows(s, k) for s in (left, right)]
        common = [qid for qid in groups[0] if all(g[qid]["status"] == "completed" for g in groups)]
        report["qa_cutoffs"][str(k)] = {
            "paired_ids": common, "paired": [stats(g[qid] for qid in common) for g in groups],
            "all": [stats(g.values()) for g in groups], "by_preference_type_paired": {
                kind: [stats(g[qid] for qid in common if left["questions"][qid]["pref_type"] == kind)
                       for g in groups] for kind in sorted({q["pref_type"] for q in left["questions"].values()})}}
    save(Path(paths[1]) / "comparison.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))

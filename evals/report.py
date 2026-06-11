from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evals.critical_failures import count_failures
from evals.metrics import task_metrics
from evals.regional_breakdown import grouped_metrics


def write_reports(output_dir: Path, metrics: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    metrics_json = json.dumps(metrics, indent=2)
    markdown = "\n".join(["# local-lm eval report", "", f"```json\n{metrics_json}\n```"])
    (output_dir / "report.md").write_text(markdown, encoding="utf-8")


def summarize_target(target_name: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    correct = sum(1 for result in results if result["score"] >= 1.0)
    failures = [failure for result in results for failure in result["critical_failures"]]
    return {
        "target": target_name,
        "total_examples": total,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "metrics": task_metrics(results),
        "critical_failure_total": len(failures),
        "critical_failures": count_failures(failures),
        "groups": grouped_metrics(results),
    }


def write_eval_summary(
    reports_dir: Path,
    target_results: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    failures_dir = reports_dir / "failures"
    examples_dir = reports_dir / "examples"
    failures_dir.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)

    targets = {
        target: summarize_target(target, results) for target, results in target_results.items()
    }
    summary = {
        "local_only": True,
        "compares": list(target_results),
        "targets": targets,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "eval_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (reports_dir / "eval_summary.md").write_text(
        _render_markdown_summary(summary),
        encoding="utf-8",
    )

    for target, results in target_results.items():
        _write_failures(failures_dir / f"{target}.jsonl", results)
        _write_examples(examples_dir / f"{target}.md", target, results)
    return summary


def _write_failures(path: Path, results: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as failure_file:
        for result in results:
            for failure in result["critical_failures"]:
                payload = {
                    "target": result["target"],
                    "prediction": result["prediction"],
                    **failure,
                }
                failure_file.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_examples(path: Path, target: str, results: list[dict[str, Any]]) -> None:
    lines = [f"# Eval Examples: {target}", ""]
    for result in results:
        example = result["example"]
        lines.extend(
            [
                f"## {example['id']} - {example['task']}",
                "",
                f"- region: {example['region']}",
                f"- country: {example['country']}",
                f"- language: {example['language']}",
                f"- document_type: {example['document_type']}",
                f"- score: {result['score']}",
                f"- critical_failures: {len(result['critical_failures'])}",
                "",
                "```json",
                json.dumps(result["prediction"], indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _render_markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Unified Local Eval Summary",
        "",
        f"- local_only: {summary['local_only']}",
        f"- compared_targets: {', '.join(summary['compares'])}",
        "",
        "| Target | Accuracy | JSON validity | Route accuracy | Critical failures |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for target, metrics in summary["targets"].items():
        task_metrics_payload = metrics["metrics"]
        lines.append(
            f"| {target} | {metrics['accuracy']} | "
            f"{task_metrics_payload['json_validity']} | "
            f"{task_metrics_payload['route_accuracy']} | "
            f"{metrics['critical_failure_total']} |"
        )
    lines.append("")
    lines.append("## Target Details")
    for target, metrics in summary["targets"].items():
        lines.extend(
            [
                "",
                f"### {target}",
                "",
                f"- total_examples: {metrics['total_examples']}",
                f"- accuracy: {metrics['accuracy']}",
                "- metrics:",
            ]
        )
        for name, value in metrics["metrics"].items():
            lines.append(f"  - {name}: {value}")
        lines.extend(
            [
                f"- critical_failure_total: {metrics['critical_failure_total']}",
                "- critical_failures:",
            ]
        )
        for name, count in metrics["critical_failures"].items():
            lines.append(f"  - {name}: {count}")
    return "\n".join(lines) + "\n"

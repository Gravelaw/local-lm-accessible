from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.schemas.source_registry import read_jsonl, validate_candidate_records, write_jsonl
from data.schemas.task_mapping import map_candidate_to_tasks
from scripts.registry_common import APPROVED_PATH, REPORTS_DIR, TASK_MAPPED_PATH


def map_datasets(
    approved_path: Path,
    output_path: Path,
    reports_dir: Path,
) -> dict[str, object]:
    raw_records = read_jsonl(approved_path)
    candidates = validate_candidate_records(raw_records)
    acceptance_by_id = {
        str(record["dataset_id"]): record.get("acceptance", {}).get("status", "")
        for record in raw_records
    }
    mappings = [
        map_candidate_to_tasks(candidate, acceptance_by_id.get(candidate.dataset_id))
        for candidate in candidates
    ]
    write_jsonl(output_path, mappings)
    coverage: dict[str, int] = {}
    for mapping in mappings:
        for task in mapping.mapped_tasks:
            coverage[str(task)] = coverage.get(str(task), 0) + 1
    report = {
        "mapped_datasets": len(mappings),
        "task_coverage": dict(sorted(coverage.items())),
        "eval_only": sum(1 for mapping in mappings if mapping.eval_only),
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "dataset_task_coverage.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (reports_dir / "dataset_task_coverage.md").write_text(
        render_task_coverage(report),
        encoding="utf-8",
    )
    return report


def render_task_coverage(report: dict[str, object]) -> str:
    lines = [
        "# Dataset Task Coverage",
        "",
        f"- mapped_datasets: {report['mapped_datasets']}",
        f"- eval_only: {report['eval_only']}",
        "",
        "## Tasks",
        "",
    ]
    coverage = report["task_coverage"]
    if not isinstance(coverage, dict):
        raise TypeError("task_coverage must be a mapping")
    for task, count in coverage.items():
        lines.append(f"- {task}: {count}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approved", type=Path, default=APPROVED_PATH)
    parser.add_argument("--output", type=Path, default=TASK_MAPPED_PATH)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()
    report = map_datasets(args.approved, args.output, args.reports_dir)
    print(f"mapped {report['mapped_datasets']} datasets to tasks")


if __name__ == "__main__":
    main()

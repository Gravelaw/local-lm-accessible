from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

from registry_common import (
    APPROVED_PATH,
    CANDIDATES_PATH,
    REJECTED_PATH,
    REPORTS_DIR,
    RESEARCH_EVAL_PATH,
)

from data.schemas.dataset_acceptance import AcceptanceStatus, evaluate_candidate
from data.schemas.source_registry import (
    DatasetCandidate,
    count_by,
    read_jsonl,
    validate_candidate_records,
    write_jsonl,
)


def _decision_payload(candidate: DatasetCandidate) -> dict[str, object]:
    decision = evaluate_candidate(candidate)
    payload = candidate.model_dump(mode="json")
    payload["acceptance"] = decision.model_dump(mode="json")
    return payload


def _counts(records: Iterable[DatasetCandidate]) -> dict[str, dict[str, int]]:
    return {
        "source": count_by(records, "source_catalog"),
        "task": count_by(records, "candidate_tasks"),
        "region": count_by(records, "regions"),
        "language": count_by(records, "languages"),
        "modality": count_by(records, "modality"),
        "license": count_by(records, "license_name"),
        "pii_risk": count_by(records, "pii_risk"),
    }


def audit_registry(
    candidates_path: Path,
    approved_path: Path,
    research_eval_path: Path,
    rejected_path: Path,
    reports_dir: Path,
) -> dict[str, object]:
    candidates = validate_candidate_records(read_jsonl(candidates_path))
    approved: list[dict[str, object]] = []
    research_eval_only: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []

    for candidate in candidates:
        payload = _decision_payload(candidate)
        status = payload["acceptance"]["status"]  # type: ignore[index]
        if status == AcceptanceStatus.REJECTED:
            rejected.append(payload)
        elif status == AcceptanceStatus.RESEARCH_EVAL_ONLY:
            research_eval_only.append(payload)
        else:
            approved.append(payload)

    write_jsonl(approved_path, approved)
    write_jsonl(research_eval_path, research_eval_only)
    write_jsonl(rejected_path, rejected)

    report = {
        "total_candidates": len(candidates),
        "approved": len(approved),
        "research_eval_only": len(research_eval_only),
        "rejected": len(rejected),
        "counts": _counts(candidates),
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "dataset_registry_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (reports_dir / "dataset_registry_audit.md").write_text(
        render_markdown_report(report),
        encoding="utf-8",
    )
    return report


def render_markdown_report(report: dict[str, object]) -> str:
    lines = [
        "# Dataset Registry Audit",
        "",
        f"- total_candidates: {report['total_candidates']}",
        f"- approved: {report['approved']}",
        f"- research_eval_only: {report['research_eval_only']}",
        f"- rejected: {report['rejected']}",
        "",
    ]
    counts = report["counts"]
    if not isinstance(counts, dict):
        raise TypeError("report counts must be a mapping")
    for section_name, section_counts in counts.items():
        lines.extend([f"## Counts by {section_name}", ""])
        if not isinstance(section_counts, dict):
            raise TypeError("count sections must be mappings")
        for key, value in section_counts.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    return "\n".join(lines)


def print_counts(report: dict[str, object]) -> None:
    counts = report["counts"]
    if not isinstance(counts, dict):
        raise TypeError("report counts must be a mapping")
    for section_name, section_counts in counts.items():
        print(f"counts by {section_name}:")
        if not isinstance(section_counts, dict):
            raise TypeError("count sections must be mappings")
        for key, value in section_counts.items():
            print(f"  {key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=CANDIDATES_PATH)
    parser.add_argument("--approved", type=Path, default=APPROVED_PATH)
    parser.add_argument("--research-eval", type=Path, default=RESEARCH_EVAL_PATH)
    parser.add_argument("--rejected", type=Path, default=REJECTED_PATH)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()

    report = audit_registry(
        args.candidates,
        args.approved,
        args.research_eval,
        args.rejected,
        args.reports_dir,
    )
    print_counts(report)


if __name__ == "__main__":
    main()

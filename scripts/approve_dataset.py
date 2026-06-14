from __future__ import annotations

import argparse
from pathlib import Path

from data.schemas.dataset_acceptance import AcceptanceStatus, evaluate_candidate
from data.schemas.source_registry import append_jsonl
from scripts.registry_common import APPROVED_PATH, CANDIDATES_PATH, load_candidates


def approve_dataset(
    dataset_id_or_name: str,
    candidates_path: Path,
    approved_path: Path,
    force: bool = False,
) -> dict[str, object]:
    matches = [
        candidate
        for candidate in load_candidates(candidates_path)
        if (
            candidate.dataset_id == dataset_id_or_name
            or candidate.dataset_name == dataset_id_or_name
        )
    ]
    if not matches:
        raise ValueError(f"dataset candidate not found: {dataset_id_or_name}")
    candidate = matches[0]
    decision = evaluate_candidate(candidate)
    if decision.status != AcceptanceStatus.APPROVED and not force:
        raise ValueError(
            f"dataset is not training-approvable without documented review: {dataset_id_or_name}"
        )
    payload = candidate.model_dump(mode="json")
    payload["acceptance"] = decision.model_dump(mode="json")
    payload["manual_status"] = "approved"
    payload["reviewer_notes"] = "manual approval recorded"
    append_jsonl(approved_path, [payload])
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--candidates", type=Path, default=CANDIDATES_PATH)
    parser.add_argument("--approved", type=Path, default=APPROVED_PATH)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    payload = approve_dataset(args.dataset, args.candidates, args.approved, args.force)
    print(f"approved {payload['dataset_name']} from {payload['source_catalog']}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

from data.schemas.dataset_acceptance import evaluate_candidate
from data.schemas.source_registry import append_jsonl
from scripts.registry_common import CANDIDATES_PATH, REJECTED_PATH, load_candidates


def reject_dataset(
    dataset_id_or_name: str,
    reason: str,
    candidates_path: Path,
    rejected_path: Path,
) -> dict[str, object]:
    if not reason.strip():
        raise ValueError("manual rejection reason is required")
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
    payload = candidate.model_dump(mode="json")
    payload["acceptance"] = decision.model_dump(mode="json")
    payload["manual_status"] = "rejected"
    payload["manual_reason"] = reason
    append_jsonl(rejected_path, [payload])
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--reason", required=True)
    parser.add_argument("--candidates", type=Path, default=CANDIDATES_PATH)
    parser.add_argument("--rejected", type=Path, default=REJECTED_PATH)
    args = parser.parse_args()
    payload = reject_dataset(args.dataset, args.reason, args.candidates, args.rejected)
    print(f"rejected {payload['dataset_name']} from {payload['source_catalog']}")


if __name__ == "__main__":
    main()

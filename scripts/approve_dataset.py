from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.schemas.dataset_acceptance import AcceptanceStatus, evaluate_candidate  # noqa: E402
from data.schemas.source_registry import (  # noqa: E402
    append_jsonl,
    read_jsonl,
    validate_candidate_records,
)
from scripts.registry_common import APPROVED_PATH, CANDIDATES_PATH  # noqa: E402


def approve_dataset(
    dataset_name: str,
    candidates_path: Path,
    approved_path: Path,
    force: bool = False,
) -> dict[str, object]:
    candidates = validate_candidate_records(read_jsonl(candidates_path))
    matches = [candidate for candidate in candidates if candidate.dataset_name == dataset_name]
    if not matches:
        raise ValueError(f"dataset candidate not found: {dataset_name}")
    candidate = matches[0]
    decision = evaluate_candidate(candidate)
    if decision.status == AcceptanceStatus.REJECTED and not force:
        message = "policy rejected dataset; use --force only after documented review"
        raise ValueError(f"{message}: {dataset_name}")
    if decision.status == AcceptanceStatus.RESEARCH_EVAL_ONLY:
        message = "dataset is research/eval-only and cannot be added to training approvals"
        raise ValueError(f"{message}: {dataset_name}")

    payload = candidate.model_dump(mode="json")
    payload["acceptance"] = decision.model_dump(mode="json")
    payload["manual_status"] = "approved"
    append_jsonl(approved_path, [payload])
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name")
    parser.add_argument("--candidates", type=Path, default=CANDIDATES_PATH)
    parser.add_argument("--approved", type=Path, default=APPROVED_PATH)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    payload = approve_dataset(args.dataset_name, args.candidates, args.approved, args.force)
    print(f"approved {payload['dataset_name']} from {payload['source_catalog']}")


if __name__ == "__main__":
    main()

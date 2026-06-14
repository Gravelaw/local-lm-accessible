from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.schemas.source_registry import read_jsonl, write_jsonl
from scripts.registry_common import APPROVED_PATH, SPLITS_DIR

SPLITS = ("train", "validation", "test", "regional_stress_test")


def build_training_mix(approved_path: Path, output_dir: Path) -> dict[str, int]:
    approved = [record for record in read_jsonl(approved_path) if _is_training_candidate(record)]
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for index, split in enumerate(SPLITS):
        records = [
            record
            for offset, record in enumerate(approved)
            if offset % len(SPLITS) == index
        ]
        if split == "regional_stress_test":
            records = [
                record for record in approved if len(record.get("regions", [])) > 1
            ] or records
        write_jsonl(output_dir / f"{split}.jsonl", records)
        counts[split] = len(records)
    (output_dir / "training_mix_summary.json").write_text(
        json.dumps(counts, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return counts


def _is_training_candidate(record: dict[str, object]) -> bool:
    acceptance = record.get("acceptance", {})
    if not isinstance(acceptance, dict):
        return False
    if acceptance.get("status") != "approved":
        return False
    return acceptance.get("effective_use") not in {"eval", "eval_only", "research_eval", None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approved", type=Path, default=APPROVED_PATH)
    parser.add_argument("--output-dir", type=Path, default=SPLITS_DIR)
    args = parser.parse_args()
    counts = build_training_mix(args.approved, args.output_dir)
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()

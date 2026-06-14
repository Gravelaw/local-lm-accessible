from __future__ import annotations

import argparse
from pathlib import Path

from data.schemas.dataset_card import card_from_payload, slugify, write_dataset_card
from data.schemas.source_registry import read_jsonl
from scripts.registry_common import APPROVED_PATH, DATASET_CARDS_DIR, TASK_MAPPED_PATH


def create_cards(
    approved_path: Path,
    mapped_path: Path,
    output_dir: Path,
) -> list[Path]:
    mappings = {record["dataset_id"]: record for record in read_jsonl(mapped_path)}
    paths: list[Path] = []
    for payload in read_jsonl(approved_path):
        card = card_from_payload(payload, mappings.get(payload["dataset_id"]))
        path = output_dir / f"{slugify(card.dataset_name)}.md"
        write_dataset_card(path, card)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approved", type=Path, default=APPROVED_PATH)
    parser.add_argument("--mapped", type=Path, default=TASK_MAPPED_PATH)
    parser.add_argument("--output-dir", type=Path, default=DATASET_CARDS_DIR)
    args = parser.parse_args()
    paths = create_cards(args.approved, args.mapped, args.output_dir)
    print(f"wrote {len(paths)} dataset cards to {args.output_dir}")


if __name__ == "__main__":
    main()

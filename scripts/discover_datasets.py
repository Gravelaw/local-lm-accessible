from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.registry_common import CANDIDATES_PATH, filter_records, seed_records, write_candidates

SOURCE_ALIASES = {
    "huggingface": "huggingface",
    "hf": "huggingface",
    "kaggle": "kaggle",
    "uci": "uci_ml_repository",
    "aws": "aws_open_data_registry",
    "google": "google_public_datasets",
    "gcp": "google_public_datasets",
    "azure": "azure_open_datasets",
    "europe": "european_data_portal",
    "epo": "epo",
    "awesome": "awesome_public_datasets",
    "google_dataset_search": "google_dataset_search",
    "wikimedia": "wikimedia",
    "manual": "manual",
    "synthetic": "synthetic",
    "user_opt_in_redacted": "user_opt_in_redacted",
}


def discover_records(
    *,
    sources: str = "",
    query: str = "",
    max_results: int | None = None,
) -> list[object]:
    records = seed_records()
    selected_sources = _parse_sources(sources)
    if selected_sources:
        records = [record for record in records if str(record.source_catalog) in selected_sources]
    return filter_records(records, query=query, max_results=max_results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default="", help="Comma-separated source aliases.")
    parser.add_argument("--query", default="")
    parser.add_argument("--max-results", type=int, default=50)
    parser.add_argument("--output", type=Path, default=CANDIDATES_PATH)
    parser.add_argument("--no-download", action="store_true", default=True)
    parser.add_argument("--download-approved-only", action="store_true")
    args = parser.parse_args()

    if args.download_approved_only:
        print(
            "download-approved-only requested, but discovery only writes metadata; "
            "use approval and preparation scripts after audit.",
            file=sys.stderr,
        )

    records = discover_records(
        sources=args.sources,
        query=args.query,
        max_results=args.max_results,
    )
    write_candidates(records, args.output)
    large = [
        {
            "dataset_id": record.dataset_id,
            "dataset_name": record.dataset_name,
            "size_bytes": record.size_bytes,
            "source_url": record.source_url,
        }
        for record in records
        if record.is_large
    ]
    large_path = args.output.with_suffix(".large_downloads.json")
    large_path.write_text(json.dumps(large, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {len(records)} candidate records to {args.output}")
    print(f"wrote {len(large)} large download approval records to {large_path}")


def _parse_sources(value: str) -> set[str]:
    if not value.strip():
        return set()
    selected = set()
    for item in value.split(","):
        alias = item.strip().casefold()
        if not alias:
            continue
        if alias not in SOURCE_ALIASES:
            raise ValueError(f"unknown source alias: {item}")
        selected.add(SOURCE_ALIASES[alias])
    return selected


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

from import_awesome_public_datasets import records as awesome_records
from import_aws_open_data_registry import records as aws_records
from import_azure_open_datasets import records as azure_records
from import_google_cloud_marketplace_manifest import records as google_marketplace_records
from import_google_dataset_search_manifest import records as google_search_records
from import_uci_repository import records as uci_records
from import_wikimedia_manifest import records as wikimedia_records
from registry_common import CANDIDATES_PATH, write_candidates

from data.schemas.source_registry import dedupe_candidates


def discover_records() -> list[object]:
    records = [
        *awesome_records(),
        *aws_records(),
        *azure_records(),
        *uci_records(),
        *google_search_records(),
        *google_marketplace_records(),
        *wikimedia_records(),
    ]
    return dedupe_candidates(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=CANDIDATES_PATH)
    args = parser.parse_args()

    records = discover_records()
    write_candidates(records, args.output)
    print(f"wrote {len(records)} candidate records to {args.output}")


if __name__ == "__main__":
    main()

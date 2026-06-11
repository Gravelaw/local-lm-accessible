from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.schemas.source_registry import DatasetCandidate, write_jsonl  # noqa: E402

CANDIDATES_PATH = ROOT / "data" / "registry" / "dataset_candidates.jsonl"
APPROVED_PATH = ROOT / "data" / "registry" / "approved_datasets.jsonl"
RESEARCH_EVAL_PATH = ROOT / "data" / "registry" / "research_eval_datasets.jsonl"
REJECTED_PATH = ROOT / "data" / "registry" / "rejected_datasets.jsonl"
REPORTS_DIR = ROOT / "reports"


def candidate(**values: object) -> DatasetCandidate:
    defaults = {
        "metadata_only": True,
        "redacted": False,
        "explicit_user_opt_in": False,
        "cloud_hosted": False,
    }
    defaults.update(values)
    return DatasetCandidate.model_validate(defaults)


def write_candidates(records: Iterable[DatasetCandidate], output: Path) -> None:
    write_jsonl(output, records)


def output_arg(default: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=default)
    return parser

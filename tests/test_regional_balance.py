from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from data.schemas.source_registry import write_jsonl
from scripts.check_regional_balance import check_regional_balance


def _record(dataset_id: str, regions: list[str]) -> dict[str, object]:
    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset_id,
        "regions": regions,
        "acceptance": {"status": "approved"},
    }


def test_regional_balance_passes_when_thresholds_are_met(tmp_path: Path) -> None:
    approved = tmp_path / "approved.jsonl"
    config = tmp_path / "training_mix.yaml"
    reports = tmp_path / "reports"
    write_jsonl(
        approved,
        [
            _record("india-1", ["India"]),
            _record("sea-1", ["Southeast Asia"]),
            _record("na-1", ["North America"]),
            _record("eu-1", ["Europe"]),
        ],
    )
    config.write_text(
        yaml.safe_dump(
            {
                "general": {
                    "india": 0.25,
                    "southeast_asia": 0.25,
                    "north_america": 0.25,
                    "europe": 0.25,
                }
            }
        ),
        encoding="utf-8",
    )

    report = check_regional_balance(approved, config, reports)

    assert report["failures"] == []
    assert (reports / "regional_balance.json").exists()
    assert (reports / "regional_balance.md").exists()


def test_regional_balance_fails_when_required_regions_are_missing(tmp_path: Path) -> None:
    approved = tmp_path / "approved.jsonl"
    config = tmp_path / "training_mix.yaml"
    reports = tmp_path / "reports"
    write_jsonl(approved, [_record("india-1", ["India"]), _record("india-2", ["India"])])
    config.write_text(
        yaml.safe_dump(
            {
                "general": {
                    "india": 0.35,
                    "southeast_asia": 0.20,
                    "north_america": 0.25,
                    "europe": 0.20,
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="southeast_asia"):
        check_regional_balance(approved, config, reports)

    assert (reports / "regional_balance.json").exists()

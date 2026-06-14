from __future__ import annotations

from scripts.verify_dataset_locality import verify_dataset_locality


def test_dataset_discovery_sources_are_not_runtime_dependencies() -> None:
    report = verify_dataset_locality()

    assert report["status"] == "ok"
    assert report["findings"] == []

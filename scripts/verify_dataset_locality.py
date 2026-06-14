from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from scripts.registry_common import ROOT

CLOUD_RUNTIME_MARKERS = ("aws", "boto3", "google.cloud", "azure.", "kaggle", "load_dataset(")


def verify_dataset_locality(root: Path = ROOT) -> dict[str, object]:
    runtime_files = [
        root / "app.py",
        root / "services" / "gateway" / "app.py",
        root / "configs" / "routes.yaml",
        root / "configs" / "safety.yaml",
    ]
    findings: list[str] = []
    for path in runtime_files:
        text = path.read_text(encoding="utf-8").casefold()
        for marker in CLOUD_RUNTIME_MARKERS:
            if marker in text:
                findings.append(f"{path.relative_to(root)} contains runtime marker {marker}")
    safety = yaml.safe_load((root / "configs" / "safety.yaml").read_text(encoding="utf-8"))
    if safety.get("runtime", {}).get("allow_web", False):
        findings.append("configs/safety.yaml enables allow_web by default")
    return {"status": "ok" if not findings else "failed", "findings": findings}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    report = verify_dataset_locality(args.root)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

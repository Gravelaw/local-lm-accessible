from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from data.schemas.source_registry import read_jsonl
from scripts.registry_common import APPROVED_PATH, REPORTS_DIR, ROOT

DEFAULT_CONFIG = ROOT / "configs" / "training_mix.yaml"


def check_regional_balance(
    approved_path: Path,
    config_path: Path,
    reports_dir: Path,
    profile: str = "general",
) -> dict[str, object]:
    records = read_jsonl(approved_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    thresholds = config[profile]
    counts = {"india": 0, "southeast_asia": 0, "north_america": 0, "europe": 0}
    for record in records:
        for region in record.get("regions", []):
            key = str(region).lower().replace(" ", "_")
            if key in counts:
                counts[key] += 1
    total = max(1, sum(counts.values()))
    ratios = {key: value / total for key, value in counts.items()}
    failures = [
        f"{key}={ratios[key]:.2f} below target {float(threshold):.2f}"
        for key, threshold in thresholds.items()
        if isinstance(threshold, (int, float)) and ratios.get(key, 0.0) < float(threshold)
    ]
    report = {"profile": profile, "counts": counts, "ratios": ratios, "failures": failures}
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "regional_balance.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (reports_dir / "regional_balance.md").write_text(render_balance(report), encoding="utf-8")
    if failures:
        raise ValueError("; ".join(failures))
    return report


def render_balance(report: dict[str, object]) -> str:
    lines = ["# Regional Balance", "", f"- profile: {report['profile']}", ""]
    ratios = report["ratios"]
    if not isinstance(ratios, dict):
        raise TypeError("ratios must be a mapping")
    for key, value in ratios.items():
        lines.append(f"- {key}: {float(value):.3f}")
    lines.append("")
    failures = report["failures"]
    if failures:
        lines.append("## Failures")
        lines.extend(f"- {failure}" for failure in failures)  # type: ignore[union-attr]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approved", type=Path, default=APPROVED_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--profile", default="general")
    args = parser.parse_args()
    report = check_regional_balance(args.approved, args.config, args.reports_dir, args.profile)
    print(json.dumps(report["ratios"], sort_keys=True))


if __name__ == "__main__":
    main()

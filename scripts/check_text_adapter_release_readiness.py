from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def check_readiness(
    *,
    adapter_dir: Path,
    eval_report: Path,
    output_json: Path,
    output_md: Path,
) -> dict[str, Any]:
    failures: list[str] = []
    if not adapter_dir.exists():
        failures.append(f"missing adapter directory: {adapter_dir}")
    for filename in REQUIRED_ADAPTER_FILES:
        if not (adapter_dir / filename).exists():
            failures.append(f"missing adapter file: {filename}")

    manifest_path = adapter_dir / "adapter_manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        failures.append(f"missing adapter manifest: {manifest_path}")

    metrics: dict[str, Any] = {}
    if eval_report.exists():
        metrics = json.loads(eval_report.read_text(encoding="utf-8"))
    else:
        failures.append(f"missing eval report: {eval_report}")

    prediction_sources = set(metrics.get("prediction_sources", []))
    if "lora_adapter_generation" not in prediction_sources:
        failures.append("eval report must use lora_adapter_generation predictions")
    if float(metrics.get("invalid_refusal_rate", 1.0)) > 0.2:
        failures.append("invalid_refusal_rate exceeds 0.20")
    if float(metrics.get("unsafe_certainty_rate", 1.0)) > 0.1:
        failures.append("unsafe_certainty_rate exceeds 0.10")
    if float(metrics.get("json_validity", 0.0)) < 0.6:
        failures.append("json_validity is below 0.60")

    report = {
        "ready_for_merge_and_gguf": not failures,
        "failures": failures,
        "adapter_dir": str(adapter_dir),
        "base_model": manifest.get("base_model", ""),
        "eval_report": str(eval_report),
        "metrics": metrics,
        "required_adapter_files": list(REQUIRED_ADAPTER_FILES),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(report), encoding="utf-8")
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Text Adapter Release Readiness",
        "",
        f"- ready_for_merge_and_gguf: {report['ready_for_merge_and_gguf']}",
        f"- adapter_dir: {report['adapter_dir']}",
        f"- base_model: {report['base_model']}",
        f"- eval_report: {report['eval_report']}",
        "",
        "## Failures",
        "",
    ]
    failures = report["failures"]
    if failures:
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("- none")
    lines.extend(["", "## Metrics", ""])
    for key, value in sorted(report["metrics"].items()):
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--eval-report", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    report = check_readiness(
        adapter_dir=args.adapter_dir,
        eval_report=args.eval_report,
        output_json=args.output_json,
        output_md=args.output_md,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

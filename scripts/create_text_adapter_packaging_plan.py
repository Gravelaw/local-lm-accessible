from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def create_packaging_plan(
    *,
    readiness_report: Path,
    adapter_dir: Path,
    merged_dir: Path,
    f16_gguf: Path,
    quantized_dir: Path,
    output_json: Path,
    output_md: Path,
) -> dict[str, Any]:
    if not readiness_report.exists():
        raise FileNotFoundError(f"missing readiness report: {readiness_report}")
    readiness = json.loads(readiness_report.read_text(encoding="utf-8"))
    base_model = str(readiness.get("base_model", ""))
    if not base_model:
        raise ValueError("readiness report is missing base_model")

    ready = bool(readiness.get("ready_for_merge_and_gguf", False))
    commands = {
        "merge_adapter": [
            "python",
            "training/text/merge_adapter.py",
            "--base-model",
            base_model,
            "--adapter-dir",
            str(adapter_dir),
            "--output-dir",
            str(merged_dir),
            "--allow-remote-files",
        ],
        "export_f16_gguf": [
            "bash",
            "training/text/export_gguf.sh",
            str(merged_dir),
            str(f16_gguf),
        ],
        "quantize_gguf": [
            "bash",
            "training/text/quantize_gguf.sh",
            str(f16_gguf),
            str(quantized_dir),
        ],
        "verify_release_gate": ["python", "scripts/release_gate.py"],
    }
    plan = {
        "ready_to_package": ready,
        "readiness_report": str(readiness_report),
        "readiness_failures": readiness.get("failures", []),
        "base_model": base_model,
        "adapter_dir": str(adapter_dir),
        "merged_dir": str(merged_dir),
        "f16_gguf": str(f16_gguf),
        "quantized_dir": str(quantized_dir),
        "commands": commands,
        "notes": [
            "Run merge and GGUF conversion only after reviewing Modal storage and GPU budget.",
            (
                "Update models/manifest.json and configs/models.yaml after "
                "quantized GGUF checksums exist."
            ),
            "Run local llama.cpp smoke tests before marking the release bundle ready.",
        ],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(plan), encoding="utf-8")
    return plan


def render_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# Text Adapter Packaging Plan",
        "",
        f"- ready_to_package: {plan['ready_to_package']}",
        f"- base_model: {plan['base_model']}",
        f"- adapter_dir: {plan['adapter_dir']}",
        f"- merged_dir: {plan['merged_dir']}",
        f"- f16_gguf: {plan['f16_gguf']}",
        f"- quantized_dir: {plan['quantized_dir']}",
        "",
        "## Commands",
        "",
    ]
    for name, command in plan["commands"].items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append("```bash")
        lines.append(" ".join(command))
        lines.append("```")
        lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.extend(f"- {note}" for note in plan["notes"])
    failures = plan["readiness_failures"]
    if failures:
        lines.extend(["", "## Readiness Failures", ""])
        lines.extend(f"- {failure}" for failure in failures)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readiness-report", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--merged-dir", type=Path, required=True)
    parser.add_argument("--f16-gguf", type=Path, required=True)
    parser.add_argument("--quantized-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    plan = create_packaging_plan(
        readiness_report=args.readiness_report,
        adapter_dir=args.adapter_dir,
        merged_dir=args.merged_dir,
        f16_gguf=args.f16_gguf,
        quantized_dir=args.quantized_dir,
        output_json=args.output_json,
        output_md=args.output_md,
    )
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

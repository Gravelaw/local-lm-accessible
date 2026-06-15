from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_ADAPTER_DIR = Path("outputs/text/llama_nemotron_nano_lora")
DEFAULT_FINALIZATION = Path("reports/final_finetuning_summary.json")
DEFAULT_EVAL = Path("reports/final_text_adapter_eval.json")
DEFAULT_READINESS = Path("reports/final_text_adapter_readiness.json")
DEFAULT_PACKAGING = Path("reports/final_text_adapter_packaging_result.json")
DEFAULT_SMOKE = Path("reports/final_text_gguf_smoke.json")
DEFAULT_VISION = Path("reports/vision_readiness.json")
DEFAULT_ASR = Path("reports/asr_contingency.json")
DEFAULT_REPORT_JSON = Path("reports/finetuning_completion.json")
DEFAULT_REPORT_MD = Path("reports/finetuning_completion.md")


def check_finetuning_completion(
    *,
    adapter_dir: Path = DEFAULT_ADAPTER_DIR,
    finalization_report: Path = DEFAULT_FINALIZATION,
    eval_report: Path = DEFAULT_EVAL,
    readiness_report: Path = DEFAULT_READINESS,
    packaging_report: Path = DEFAULT_PACKAGING,
    smoke_report: Path = DEFAULT_SMOKE,
    vision_report: Path = DEFAULT_VISION,
    asr_report: Path = DEFAULT_ASR,
    report_json: Path = DEFAULT_REPORT_JSON,
    report_md: Path = DEFAULT_REPORT_MD,
) -> dict[str, Any]:
    text = _text_status(
        adapter_dir=adapter_dir,
        finalization_report=finalization_report,
        eval_report=eval_report,
        readiness_report=readiness_report,
        packaging_report=packaging_report,
        smoke_report=smoke_report,
    )
    report = {
        "complete": text["complete"] and _json_exists(vision_report) and _json_exists(asr_report),
        "text_finetuning": text,
        "vision_readiness": {
            "complete": _json_exists(vision_report),
            "report": str(vision_report),
        },
        "asr_contingency": {
            "complete": _json_exists(asr_report),
            "report": str(asr_report),
        },
        "next_modal_actions": _next_modal_actions(text, vision_report, asr_report),
    }
    _write_reports(report, report_json, report_md)
    return report


def _text_status(
    *,
    adapter_dir: Path,
    finalization_report: Path,
    eval_report: Path,
    readiness_report: Path,
    packaging_report: Path,
    smoke_report: Path,
) -> dict[str, Any]:
    adapter_files = {
        "adapter_config": adapter_dir / "adapter_config.json",
        "adapter_weights": adapter_dir / "adapter_model.safetensors",
        "adapter_manifest": adapter_dir / "adapter_manifest.json",
    }
    trained = all(path.exists() for path in adapter_files.values()) and _json_exists(
        finalization_report
    )
    readiness = _read_json_if_exists(readiness_report)
    eval_metrics = _read_json_if_exists(eval_report)
    packaging = _read_json_if_exists(packaging_report)
    smoke = _read_json_if_exists(smoke_report)
    eval_passed = bool(readiness.get("ready_for_merge_and_gguf")) if readiness else False
    packaging_complete = _packaging_complete(packaging)
    smoke_passed = bool(smoke.get("passed")) if smoke else False
    missing = [
        label
        for label, present in {
            "adapter_config.json": adapter_files["adapter_config"].exists(),
            "adapter_model.safetensors": adapter_files["adapter_weights"].exists(),
            "adapter_manifest.json": adapter_files["adapter_manifest"].exists(),
            "final_finetuning_summary.json": finalization_report.exists(),
            "final_text_adapter_eval.json": eval_report.exists(),
            "final_text_adapter_readiness.json": readiness_report.exists(),
            "final_text_adapter_packaging_result.json": packaging_report.exists(),
            "final_text_gguf_smoke.json": smoke_report.exists(),
        }.items()
        if not present
    ]
    complete = trained and eval_passed and packaging_complete and smoke_passed
    return {
        "complete": complete,
        "adapter_dir": str(adapter_dir),
        "trained": trained,
        "eval_passed": eval_passed,
        "packaging_complete": packaging_complete,
        "smoke_passed": smoke_passed,
        "missing": missing,
        "eval_metrics": eval_metrics,
    }


def _packaging_complete(packaging: dict[str, Any]) -> bool:
    if not packaging:
        return False
    for key in ("f16_gguf", "q4_gguf", "q5_gguf"):
        artifact = packaging.get(key)
        if not isinstance(artifact, dict) or not artifact.get("sha256"):
            return False
    return True


def _json_exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not _json_exists(path):
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _next_modal_actions(
    text: dict[str, Any],
    vision_report: Path,
    asr_report: Path,
) -> list[str]:
    actions: list[str] = []
    if not text["trained"]:
        actions.append("modal run modal_workflows/local_lm_pipeline.py --action finetune_text")
    if text["trained"] and not text["eval_passed"]:
        actions.append(
            "modal run modal_workflows/local_lm_pipeline.py --action evaluate_text_adapter"
        )
    if text["eval_passed"] and not text["packaging_complete"]:
        actions.append(
            "modal run modal_workflows/local_lm_pipeline.py --action run_text_adapter_packaging"
        )
    if text["packaging_complete"] and not text["smoke_passed"]:
        actions.append(
            "modal run modal_workflows/local_lm_pipeline.py --action smoke_test_packaged_gguf"
        )
    if not _json_exists(vision_report):
        actions.append(
            "modal run modal_workflows/local_lm_pipeline.py --action create_vision_readiness"
        )
    if not _json_exists(asr_report):
        actions.append(
            "modal run modal_workflows/local_lm_pipeline.py --action check_asr_contingency"
        )
    return actions


def _write_reports(report: dict[str, Any], report_json: Path, report_md: Path) -> None:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Fine-Tuning Completion",
        "",
        f"- complete: {report['complete']}",
        f"- text_finetuning_complete: {report['text_finetuning']['complete']}",
        f"- text_trained: {report['text_finetuning']['trained']}",
        f"- text_eval_passed: {report['text_finetuning']['eval_passed']}",
        f"- text_packaging_complete: {report['text_finetuning']['packaging_complete']}",
        f"- text_smoke_passed: {report['text_finetuning']['smoke_passed']}",
        f"- vision_readiness_complete: {report['vision_readiness']['complete']}",
        f"- asr_contingency_complete: {report['asr_contingency']['complete']}",
        "",
        "## Missing Text Artifacts",
        "",
    ]
    missing = report["text_finetuning"]["missing"]
    lines.extend(f"- {item}" for item in missing) if missing else lines.append("- none")
    lines.extend(["", "## Next Modal Actions", ""])
    actions = report["next_modal_actions"]
    lines.extend(f"- `{action}`" for action in actions) if actions else lines.append("- none")
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check fine-tuning completion gates")
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--finalization-report", type=Path, default=DEFAULT_FINALIZATION)
    parser.add_argument("--eval-report", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--readiness-report", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--packaging-report", type=Path, default=DEFAULT_PACKAGING)
    parser.add_argument("--smoke-report", type=Path, default=DEFAULT_SMOKE)
    parser.add_argument("--vision-report", type=Path, default=DEFAULT_VISION)
    parser.add_argument("--asr-report", type=Path, default=DEFAULT_ASR)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    args = parser.parse_args()
    report = check_finetuning_completion(
        adapter_dir=args.adapter_dir,
        finalization_report=args.finalization_report,
        eval_report=args.eval_report,
        readiness_report=args.readiness_report,
        packaging_report=args.packaging_report,
        smoke_report=args.smoke_report,
        vision_report=args.vision_report,
        asr_report=args.asr_report,
        report_json=args.report_json,
        report_md=args.report_md,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

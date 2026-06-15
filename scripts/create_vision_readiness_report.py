from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.preflight_finetuning_manifests import preflight_vision  # noqa: E402
from training.vision.prepare_dataset import validate_records  # noqa: E402
from training.vision.train_minicpm_v_lora import dry_run, load_config  # noqa: E402

DEFAULT_CONFIG = Path("training/vision/configs/minicpm_v_modal_document_lora.yaml")
DEFAULT_REPORT_JSON = Path("reports/vision_readiness.json")
DEFAULT_REPORT_MD = Path("reports/vision_readiness.md")


def create_vision_readiness_report(
    *,
    config_path: Path = DEFAULT_CONFIG,
    report_json: Path = DEFAULT_REPORT_JSON,
    report_md: Path = DEFAULT_REPORT_MD,
    limit: int = 6,
    require_images: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    preflight = preflight_vision(config_path, require_images=require_images)
    dry_run_summary = dry_run(config, limit=limit, require_images=require_images)
    train_records = validate_records(
        Path(config["data"]["train_file"]),
        limit=limit,
        require_images=require_images,
    )
    eval_records = validate_records(
        Path(config["data"]["eval_file"]),
        limit=limit,
        require_images=require_images,
    )
    all_records = [*train_records, *eval_records]
    human_review_tasks = sorted(
        {record.task for record in all_records if record.human_review_required}
    )
    pii_counts = Counter(record.pii_status for record in all_records)

    status = "ready_for_training" if require_images else "ready_for_modal_dry_run"
    report = {
        "status": status,
        "config": str(config_path),
        "base_model": config["model"]["base_model"],
        "allowed_vendor": config["model"]["allowed_vendor"],
        "training_backend": config["training"]["backend"],
        "fallback_backend": config["training"].get("fallback_backend", "swift"),
        "train_rows": preflight["train_rows"],
        "eval_rows": preflight["eval_rows"],
        "sampled_train_rows": len(train_records),
        "sampled_eval_rows": len(eval_records),
        "tasks": sorted({record.task for record in all_records}),
        "regions": dict(sorted(Counter(record.region for record in all_records).items())),
        "languages": dict(sorted(Counter(record.language for record in all_records).items())),
        "pii_status": dict(sorted(pii_counts.items())),
        "human_review_required_tasks": human_review_tasks,
        "require_images": require_images,
        "local_only": dry_run_summary["local_only"],
        "wandb_disabled": dry_run_summary["wandb_disabled"],
        "lora_enabled": dry_run_summary["lora_enabled"],
        "qlora_enabled": dry_run_summary["qlora_enabled"],
        "gradient_checkpointing": dry_run_summary["gradient_checkpointing"],
        "train_eval_split_separate": dry_run_summary["train_eval_split_separate"],
        "command_plan": dry_run_summary["command_plan"],
        "next_action": _next_action(require_images=require_images),
        "competition_evidence": (
            "MiniCPM-V LoRA/QLoRA scaffold validated on prepared local manifests; "
            "full training remains gated on approved images being present in Modal volume."
        ),
    }
    _write_reports(report, report_json, report_md)
    return report


def _next_action(*, require_images: bool) -> str:
    if require_images:
        return "Run finetune_vision with dry_run=false after reviewer approval."
    return "Run Modal finetune_vision dry-run, then repeat with --require-images."


def _write_reports(report: dict[str, Any], report_json: Path, report_md: Path) -> None:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Vision Fine-Tuning Readiness",
        "",
        f"- status: {report['status']}",
        f"- config: {report['config']}",
        f"- base_model: {report['base_model']}",
        f"- allowed_vendor: {report['allowed_vendor']}",
        f"- backend: {report['training_backend']}",
        f"- fallback_backend: {report['fallback_backend']}",
        f"- train_rows: {report['train_rows']}",
        f"- eval_rows: {report['eval_rows']}",
        f"- sampled_train_rows: {report['sampled_train_rows']}",
        f"- sampled_eval_rows: {report['sampled_eval_rows']}",
        f"- tasks: {', '.join(report['tasks'])}",
        f"- regions: {json.dumps(report['regions'], sort_keys=True)}",
        f"- languages: {json.dumps(report['languages'], sort_keys=True)}",
        f"- pii_status: {json.dumps(report['pii_status'], sort_keys=True)}",
        f"- human_review_required_tasks: {', '.join(report['human_review_required_tasks'])}",
        f"- require_images: {report['require_images']}",
        f"- local_only: {report['local_only']}",
        f"- wandb_disabled: {report['wandb_disabled']}",
        f"- lora_enabled: {report['lora_enabled']}",
        f"- qlora_enabled: {report['qlora_enabled']}",
        f"- train_eval_split_separate: {report['train_eval_split_separate']}",
        f"- next_action: {report['next_action']}",
        "",
        "## Command Plan",
        "",
        "```json",
        json.dumps(report["command_plan"], indent=2, sort_keys=True),
        "```",
    ]
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create MiniCPM-V readiness report")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--require-images", action="store_true")
    args = parser.parse_args()
    report = create_vision_readiness_report(
        config_path=args.config,
        report_json=args.report_json,
        report_md=args.report_md,
        limit=args.limit,
        require_images=args.require_images,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

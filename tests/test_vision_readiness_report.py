from __future__ import annotations

import json
from pathlib import Path

from scripts.create_vision_readiness_report import create_vision_readiness_report

CONFIG_PATH = Path("training/vision/configs/minicpm_v_document_lora.yaml")


def test_vision_readiness_report_records_minicpm_modal_dry_run(tmp_path: Path) -> None:
    report_json = tmp_path / "vision_readiness.json"
    report_md = tmp_path / "vision_readiness.md"

    report = create_vision_readiness_report(
        config_path=CONFIG_PATH,
        report_json=report_json,
        report_md=report_md,
        limit=6,
        require_images=False,
    )

    assert report["status"] == "ready_for_modal_dry_run"
    assert report["base_model"] == "openbmb/MiniCPM-V-4.6"
    assert report["allowed_vendor"] == "OpenBMB"
    assert report["local_only"] is True
    assert report["wandb_disabled"] is True
    assert report["lora_enabled"] is True
    assert report["qlora_enabled"] is True
    assert report["command_plan"]["executes_training"] is False
    assert "receipt_extraction" in report["tasks"]

    saved_report = json.loads(report_json.read_text(encoding="utf-8"))
    assert saved_report["train_eval_split_separate"] is True
    markdown = report_md.read_text(encoding="utf-8")
    assert "# Vision Fine-Tuning Readiness" in markdown
    assert "MiniCPM-V" in markdown

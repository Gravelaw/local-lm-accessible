from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MANIFEST = Path("models/manifest.json")
DEFAULT_EVAL_REPORT = Path("reports/asr_eval.json")
DEFAULT_REPORT_JSON = Path("reports/asr_contingency.json")
DEFAULT_REPORT_MD = Path("reports/asr_contingency.md")
ALLOWED_VENDOR = "NVIDIA"
DEFAULT_WER_THRESHOLD = 0.15
DEFAULT_UNSUPPORTED_LANGUAGE_THRESHOLD = 1.0


def build_asr_contingency_report(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    eval_report_path: Path | None = DEFAULT_EVAL_REPORT,
    report_json: Path = DEFAULT_REPORT_JSON,
    report_md: Path = DEFAULT_REPORT_MD,
    wer_threshold: float = DEFAULT_WER_THRESHOLD,
    unsupported_language_threshold: float = DEFAULT_UNSUPPORTED_LANGUAGE_THRESHOLD,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    asr_model = _find_model(manifest, "asr")
    eval_metrics = _read_optional_eval(eval_report_path)
    primary_ready = _primary_ready(asr_model)
    eval_ready = _eval_ready(
        eval_metrics,
        wer_threshold=wer_threshold,
        unsupported_language_threshold=unsupported_language_threshold,
    )
    alternate_required = not primary_ready or not eval_ready
    status = (
        "evaluate_alternate"
        if alternate_required
        else "keep_primary_with_runtime_validation"
    )
    report = {
        "status": status,
        "alternate_required": alternate_required,
        "manifest": str(manifest_path),
        "eval_report": str(eval_report_path) if eval_report_path else None,
        "thresholds": {
            "wer": wer_threshold,
            "unsupported_language_detection": unsupported_language_threshold,
        },
        "primary": _primary_summary(asr_model),
        "eval_metrics": eval_metrics,
        "candidate_rules": [
            "Allowed ASR vendor is NVIDIA only.",
            "Candidate must run locally with no remote audio upload.",
            "Candidate must have a recorded license, local path, and checksum before approval.",
            "Unsupported Indian and Southeast Asian non-English ASR must stay experimental.",
            "Use eval-only behavior until WER and unsupported-language checks pass.",
        ],
        "candidate_backlog": [
            {
                "candidate": "nvidia_parakeet_ctc_family",
                "hub_repo_id": None,
                "status": "candidate_pending_hub_verification",
                "reason": "Potential NVIDIA ASR fallback if TDT runtime is unavailable.",
            },
            {
                "candidate": "nvidia_canary_family",
                "hub_repo_id": None,
                "status": "candidate_pending_hub_verification",
                "reason": "Potential multilingual NVIDIA ASR fallback subject to license review.",
            },
        ],
        "next_action": _next_action(alternate_required),
        "remote_audio_uploads": False,
    }
    _write_reports(report, report_json, report_md)
    return report


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _find_model(manifest: dict[str, Any], key: str) -> dict[str, Any] | None:
    models = manifest.get("models")
    if not isinstance(models, list):
        raise ValueError("models manifest must contain a models list")
    for model in models:
        if isinstance(model, dict) and model.get("key") == key:
            return model
    return None


def _read_optional_eval(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected ASR eval JSON object: {path}")
    return payload


def _primary_ready(model: dict[str, Any] | None) -> bool:
    if model is None:
        return False
    return (
        model.get("vendor") == ALLOWED_VENDOR
        and bool(str(model.get("license", "")).strip())
        and bool(str(model.get("local_path", "")).strip())
        and bool(str(model.get("sha256", "")).strip())
    )


def _eval_ready(
    metrics: dict[str, Any] | None,
    *,
    wer_threshold: float,
    unsupported_language_threshold: float,
) -> bool:
    if metrics is None:
        return False
    return (
        float(metrics.get("wer", 1.0)) <= wer_threshold
        and float(metrics.get("unsupported_language_detection", 0.0))
        >= unsupported_language_threshold
        and not metrics.get("missing_predictions")
        and metrics.get("remote_uploads") is False
    )


def _primary_summary(model: dict[str, Any] | None) -> dict[str, Any]:
    if model is None:
        return {"configured": False}
    return {
        "configured": True,
        "model_id": model.get("model_id"),
        "vendor": model.get("vendor"),
        "license": model.get("license"),
        "local_path": model.get("local_path"),
        "checksum_configured": bool(str(model.get("sha256", "")).strip()),
        "supported_languages": model.get("supported_languages", []),
        "unsupported_languages": model.get("unsupported_languages", []),
        "commercial_status": model.get("commercial_status"),
        "runtime": model.get("runtime"),
    }


def _next_action(alternate_required: bool) -> str:
    if alternate_required:
        return "Run Parakeet local eval and verify NVIDIA fallback candidates before training."
    return "Keep Parakeet as primary and run runtime smoke tests on staged local audio."


def _write_reports(report: dict[str, Any], report_json: Path, report_md: Path) -> None:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# ASR Contingency",
        "",
        f"- status: {report['status']}",
        f"- alternate_required: {report['alternate_required']}",
        f"- primary_model_id: {report['primary'].get('model_id')}",
        f"- primary_vendor: {report['primary'].get('vendor')}",
        f"- checksum_configured: {report['primary'].get('checksum_configured')}",
        f"- eval_report: {report['eval_report']}",
        f"- remote_audio_uploads: {report['remote_audio_uploads']}",
        f"- next_action: {report['next_action']}",
        "",
        "## Candidate Rules",
        "",
        *[f"- {rule}" for rule in report["candidate_rules"]],
        "",
        "## Candidate Backlog",
        "",
        *[
            f"- {candidate['candidate']}: {candidate['status']}"
            for candidate in report["candidate_backlog"]
        ],
    ]
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check ASR fallback readiness")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--eval-report", type=Path, default=DEFAULT_EVAL_REPORT)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--wer-threshold", type=float, default=DEFAULT_WER_THRESHOLD)
    parser.add_argument(
        "--unsupported-language-threshold",
        type=float,
        default=DEFAULT_UNSUPPORTED_LANGUAGE_THRESHOLD,
    )
    args = parser.parse_args()
    report = build_asr_contingency_report(
        manifest_path=args.manifest,
        eval_report_path=args.eval_report,
        report_json=args.report_json,
        report_md=args.report_md,
        wer_threshold=args.wer_threshold,
        unsupported_language_threshold=args.unsupported_language_threshold,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

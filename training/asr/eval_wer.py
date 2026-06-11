from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.asr.prepare_manifest import validate_manifest  # noqa: E402


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().strip().split())


def _edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    previous = list(range(cols))
    for row in range(1, rows):
        current = [row] + [0] * (cols - 1)
        for col in range(1, cols):
            substitution = previous[col - 1] + (reference[row - 1] != hypothesis[col - 1])
            insertion = current[col - 1] + 1
            deletion = previous[col] + 1
            current[col] = min(substitution, insertion, deletion)
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = _normalize_text(reference).split()
    hyp_words = _normalize_text(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _edit_distance(ref_words, hyp_words) / len(ref_words)


def char_error_rate(reference: str, hypothesis: str) -> float:
    ref_chars = list(_normalize_text(reference).replace(" ", ""))
    hyp_chars = list(_normalize_text(hypothesis).replace(" ", ""))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return _edit_distance(ref_chars, hyp_chars) / len(ref_chars)


def read_predictions(path: Path) -> dict[str, dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("predictions must be a JSON list")
    predictions: dict[str, dict[str, str]] = {}
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("prediction rows must be objects")
        audio_filepath = str(item["audio_filepath"])
        predictions[audio_filepath] = {
            "text": str(item.get("prediction", "")),
            "language": str(item.get("language", "")),
            "unsupported_language": str(item.get("unsupported_language", "false")).casefold(),
        }
    return predictions


def evaluate_manifest(manifest_path: Path, predictions_path: Path) -> dict[str, Any]:
    records = validate_manifest(manifest_path)
    predictions = read_predictions(predictions_path)
    total_wer = total_cer = 0.0
    language_correct = unsupported_total = unsupported_correct = 0
    noisy_room_wers: list[float] = []
    elderly_wers: list[float] = []
    missing_predictions: list[str] = []
    unsupported_language_failures: list[str] = []
    language_counts: Counter[str] = Counter()

    for record in records:
        prediction = predictions.get(record.audio_filepath)
        if prediction is None:
            missing_predictions.append(record.audio_filepath)
            prediction = {"text": "", "language": "", "unsupported_language": "false"}
        wer = word_error_rate(record.text, prediction["text"])
        cer = char_error_rate(record.text, prediction["text"])
        total_wer += wer
        total_cer += cer
        language_counts[record.language] += 1
        if prediction["language"] == record.language:
            language_correct += 1
        if record.experimental or not record.supported_by_parakeet_v3:
            unsupported_total += 1
            if prediction["unsupported_language"] == "true":
                unsupported_correct += 1
            else:
                unsupported_language_failures.append(record.audio_filepath)
        if "noisy_room" in record.accent:
            noisy_room_wers.append(wer)
        if record.speaker_age_bucket == "elderly":
            elderly_wers.append(wer)

    count = len(records)
    return {
        "count": count,
        "wer": _average(total_wer, count),
        "cer": _average(total_cer, count),
        "language_detection_accuracy": _average(language_correct, count),
        "unsupported_language_detection": _average(unsupported_correct, unsupported_total),
        "noisy_room_wer": _list_average(noisy_room_wers),
        "elderly_speaker_wer": _list_average(elderly_wers),
        "missing_predictions": missing_predictions,
        "unsupported_language_failures": unsupported_language_failures,
        "language_counts": dict(sorted(language_counts.items())),
        "remote_uploads": False,
    }


def _average(numerator: float, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)


def _list_average(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def write_reports(metrics: dict[str, Any], report_json: Path, report_md: Path) -> None:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["# ASR Eval", ""]
    for key, value in metrics.items():
        lines.append(f"- {key}: {value}")
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, default=Path("reports/asr_eval.json"))
    parser.add_argument("--report-md", type=Path, default=Path("reports/asr_eval.md"))
    args = parser.parse_args()
    metrics = evaluate_manifest(args.manifest, args.predictions)
    write_reports(metrics, args.report_json, args.report_md)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

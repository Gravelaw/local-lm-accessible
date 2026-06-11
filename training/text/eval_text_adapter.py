from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.text.train_nemotron_lora import (  # noqa: E402
    TextSFTExample,
    load_jsonl,
    render_prompt_text,
    validate_examples,
)

UNCERTAINTY_TERMS = {"may", "might", "uncertain", "review", "check", "verify", "consult"}


def _assistant_text(record: dict[str, Any]) -> str:
    return str(record["messages"][-1]["content"])


def baseline_predictions_from_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validate_examples(records)
    predictions = []
    for index, record in enumerate(records, start=1):
        predictions.append(
            {
                "example_id": str(record.get("id", f"sample-{index:04d}")),
                "task": str(record["metadata"]["task"]),
                "prediction": _assistant_text(record),
                "reference": _assistant_text(record),
                "metadata": record["metadata"],
                "prediction_source": "assistant_label_baseline",
            }
        )
    return predictions


def prediction_from_example(
    example: TextSFTExample,
    prediction: str,
    *,
    example_id: str,
    prediction_source: str,
) -> dict[str, Any]:
    return {
        "example_id": example_id,
        "task": example.metadata.task,
        "prompt": render_prompt_text(example),
        "prediction": prediction,
        "reference": example.messages[-1].content,
        "metadata": example.metadata.model_dump(mode="json"),
        "prediction_source": prediction_source,
    }


def _json_or_none(text: str) -> Any | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _readability_score(text: str) -> float:
    words = [word for word in text.replace("\n", " ").split(" ") if word]
    if not words:
        return 0.0
    avg_word_length = mean(len(word.strip(".,;:!?")) for word in words)
    sentence_count = max(1, sum(text.count(mark) for mark in ".!?"))
    words_per_sentence = len(words) / sentence_count
    score = 100.0 - (avg_word_length * 8.0) - (words_per_sentence * 1.2)
    return max(0.0, min(100.0, round(score, 2)))


def evaluate_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(predictions)
    route_total = route_correct = 0
    json_total = json_valid = 0
    tool_total = tool_args_correct = 0
    source_total = source_covered = 0
    certainty_total = unsafe_certainty = 0
    refusals = 0
    readability_scores: list[float] = []

    for record in predictions:
        task = str(record["task"])
        text = str(record["prediction"])
        readability_scores.append(_readability_score(text))
        if "I cannot help" in text or "I can't help" in text:
            refusals += 1
        if task == "route_task":
            route_total += 1
            parsed = _json_or_none(text)
            if isinstance(parsed, dict) and isinstance(parsed.get("route"), str):
                route_correct += 1
        if task in {"tool_call_json", "repair_json"}:
            json_total += 1
            parsed = _json_or_none(text)
            if isinstance(parsed, dict):
                json_valid += 1
        if task == "tool_call_json":
            tool_total += 1
            parsed = _json_or_none(text)
            if isinstance(parsed, dict) and isinstance(parsed.get("arguments"), dict):
                tool_args_correct += 1
        if task == "summarize_with_sources":
            source_total += 1
            if "[source:" in text.casefold():
                source_covered += 1
        if task == "uncertainty_warning":
            certainty_total += 1
            lowered = text.casefold()
            if not any(term in lowered for term in UNCERTAINTY_TERMS):
                unsafe_certainty += 1

    return {
        "total_examples": total,
        "prediction_sources": sorted(
            {str(record.get("prediction_source", "unknown")) for record in predictions}
        ),
        "route_accuracy": _ratio(route_correct, route_total),
        "json_validity": _ratio(json_valid, json_total),
        "tool_call_argument_accuracy": _ratio(tool_args_correct, tool_total),
        "summary_source_coverage": _ratio(source_covered, source_total),
        "unsafe_certainty_rate": _ratio(unsafe_certainty, certainty_total),
        "readability_score": round(mean(readability_scores), 2) if readability_scores else 0.0,
        "invalid_refusal_rate": _ratio(refusals, total),
    }


def evaluate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return evaluate_predictions(baseline_predictions_from_records(records))


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def write_reports(metrics: dict[str, Any], json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["# Text Adapter Eval", ""]
    for key, value in metrics.items():
        lines.append(f"- {key}: {value}")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("training/text/sample_data/router_summary_32.jsonl"),
    )
    parser.add_argument("--report-json", type=Path, default=Path("reports/text_eval.json"))
    parser.add_argument("--report-md", type=Path, default=Path("reports/text_eval.md"))
    args = parser.parse_args()

    metrics = evaluate_records(load_jsonl(args.input))
    write_reports(metrics, args.report_json, args.report_md)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

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


def adapter_predictions_from_records(
    records: list[dict[str, Any]],
    *,
    base_model: str,
    adapter_dir: Path,
    max_new_tokens: int = 192,
    limit: int | None = None,
    local_files_only: bool = False,
) -> list[dict[str, Any]]:
    examples = validate_examples(records[:limit] if limit is not None else records)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"missing adapter directory: {adapter_dir}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        local_files_only=local_files_only,
        trust_remote_code=False,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        local_files_only=local_files_only,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(
        model,
        adapter_dir,
        is_trainable=False,
        local_files_only=True,
    )
    model.eval()

    predictions: list[dict[str, Any]] = []
    for index, example in enumerate(examples, start=1):
        prompt = render_prompt_text(example, tokenizer)
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.inference_mode():
            output_ids = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_token_count = int(encoded["input_ids"].shape[-1])
        generated_ids = output_ids[0][prompt_token_count:]
        prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        predictions.append(
            prediction_from_example(
                example,
                prediction,
                example_id=f"adapter-{index:04d}",
                prediction_source="lora_adapter_generation",
            )
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
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--base-model")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only locally cached base-model files.",
    )
    args = parser.parse_args()

    records = load_jsonl(args.input)
    if args.adapter_dir is None:
        predictions = baseline_predictions_from_records(records)
    else:
        if not args.base_model:
            raise ValueError("--base-model is required when --adapter-dir is set")
        predictions = adapter_predictions_from_records(
            records,
            base_model=args.base_model,
            adapter_dir=args.adapter_dir,
            max_new_tokens=args.max_new_tokens,
            limit=args.limit,
            local_files_only=args.local_files_only,
        )
    metrics = evaluate_predictions(predictions)
    write_reports(metrics, args.report_json, args.report_md)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

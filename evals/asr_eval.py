from __future__ import annotations

from typing import Any

from evals.critical_failures import detect_critical_failures
from evals.target_adapters import EvalTargetAdapter


def sample_examples() -> list[dict[str, Any]]:
    return [
        {
            "id": "asr-india-en-001",
            "task": "speech_to_text",
            "region": "India",
            "country": "India",
            "language": "en",
            "document_type": "audio",
            "input": "synthetic elderly Indian English audio",
            "expected": {"transcript": "please read the invoice total"},
            "expects_json": False,
            "unsupported_language": False,
            "safety_domain": "general",
        },
        {
            "id": "asr-india-hi-001",
            "task": "speech_to_text",
            "region": "India",
            "country": "India",
            "language": "hi",
            "document_type": "audio",
            "input": "IndicVoices evaluation placeholder",
            "expected": {"transcript": ""},
            "expects_json": False,
            "unsupported_language": True,
            "safety_domain": "general",
        },
    ]


def evaluate(target: str | EvalTargetAdapter) -> list[dict[str, Any]]:
    adapter = (
        target
        if isinstance(target, EvalTargetAdapter)
        else EvalTargetAdapter(str(target), str(target))
    )
    return [_evaluate_one(adapter, example) for example in sample_examples()]


def run() -> dict[str, object]:
    results = evaluate("base")
    return {"name": "asr", "implemented": True, "local_only": True, "count": len(results)}


def _evaluate_one(adapter: EvalTargetAdapter, example: dict[str, Any]) -> dict[str, Any]:
    prediction = adapter.predict("asr", example)
    score = _score(example, prediction)
    failures = detect_critical_failures(example, prediction)
    return {
        "target": adapter.target_name,
        "modality": "asr",
        "example": example,
        "prediction": prediction,
        "score": score,
        "critical_failures": failures,
    }


def sample_predict(target: str, example: dict[str, Any]) -> dict[str, Any]:
    if bool(example.get("unsupported_language")):
        if target in {"fine_tuned_adapter", "merged_hf_model", "quantized_gguf_model"}:
            return {
                "output": "",
                "transcript": "",
                "unsupported_language_flag": True,
                "confidence": 0.4,
                "human_review_required": True,
            }
        return {
            "output": "please read total",
            "transcript": "please read total",
            "unsupported_language_flag": False,
            "confidence": 0.5,
            "human_review_required": False,
        }
    return {
        "output": "please read the invoice total",
        "transcript": "please read the invoice total",
        "unsupported_language_flag": False,
        "confidence": 0.92,
        "human_review_required": False,
    }


def _score(example: dict[str, Any], prediction: dict[str, Any]) -> float:
    if bool(example.get("unsupported_language")):
        return 1.0 if prediction.get("unsupported_language_flag") else 0.0
    return 1.0 if prediction.get("transcript") == example["expected"]["transcript"] else 0.0

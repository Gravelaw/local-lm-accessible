from __future__ import annotations

import json
from typing import Any

from evals.critical_failures import detect_critical_failures
from evals.target_adapters import EvalTargetAdapter

TEXT_TASKS = {"route_task", "summarize_wikipedia", "summarize_web_page"}


def sample_examples() -> list[dict[str, Any]]:
    return [
        {
            "id": "text-route-india-001",
            "task": "route_task",
            "region": "India",
            "country": "India",
            "language": "en",
            "document_type": "request",
            "input": "Read this invoice image and export the rows to Excel.",
            "expected": {"route": "invoice_to_excel"},
            "expects_json": True,
            "safety_domain": "general",
        },
        {
            "id": "text-wiki-europe-001",
            "task": "summarize_wikipedia",
            "region": "Europe",
            "country": "Germany",
            "language": "en",
            "document_type": "wiki_article",
            "input": "Offline Wikipedia article about VAT invoices.",
            "expected": {"must_include": ["VAT", "invoice", "source"]},
            "expects_json": False,
            "safety_domain": "financial",
        },
        {
            "id": "text-web-na-001",
            "task": "summarize_web_page",
            "region": "North America",
            "country": "United States",
            "language": "en",
            "document_type": "web_page",
            "input": "Local cached web page about clinic opening hours.",
            "expected": {"must_include": ["clinic", "hours"]},
            "expects_json": False,
            "safety_domain": "medical",
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
    return {"name": "text", "implemented": True, "local_only": True, "count": len(results)}


def _evaluate_one(adapter: EvalTargetAdapter, example: dict[str, Any]) -> dict[str, Any]:
    prediction = adapter.predict("text", example)
    score = _score(example, prediction)
    failures = detect_critical_failures(example, prediction)
    return {
        "target": adapter.target_name,
        "modality": "text",
        "example": example,
        "prediction": prediction,
        "score": score,
        "critical_failures": failures,
    }


def sample_predict(target: str, example: dict[str, Any]) -> dict[str, Any]:
    task = example["task"]
    if task == "route_task":
        route = "invoice_to_excel"
        if target == "base":
            route = "document_to_excel"
        output = json.dumps({"route": route})
        return {"output": output, "route": route, "confidence": 0.92}
    if task == "summarize_wikipedia":
        return {
            "output": "VAT invoices list VAT, invoice identifiers, and source references.",
            "confidence": 0.86,
            "human_review_required": True,
        }
    if task == "summarize_web_page":
        if target == "llama_cpp_endpoint":
            return {
                "output": "The clinic hours are summarized from the local cached page.",
                "confidence": 0.83,
                "human_review_required": True,
            }
        return {
            "output": "The clinic hours are summarized. Verify medical scheduling details.",
            "confidence": 0.88,
            "human_review_required": True,
        }
    raise ValueError(f"unsupported text task: {task}")


def _score(example: dict[str, Any], prediction: dict[str, Any]) -> float:
    task = example["task"]
    if task == "route_task":
        return 1.0 if prediction.get("route") == example["expected"]["route"] else 0.0
    expected_terms = example["expected"]["must_include"]
    output = str(prediction.get("output", "")).casefold()
    matched = sum(1 for term in expected_terms if term.casefold() in output)
    return round(matched / len(expected_terms), 4)

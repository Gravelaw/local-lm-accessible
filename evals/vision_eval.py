from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from evals.critical_failures import detect_critical_failures
from evals.target_adapters import EvalTargetAdapter
from scripts.synthetic_documents import generate_documents

VISION_TASKS = {
    "invoice_to_json",
    "invoice_to_excel",
    "bank_statement_to_transactions",
    "handwritten_note_to_text",
    "image_accessibility_description",
    "image_translation",
}
FIXTURE_ROOT = Path(tempfile.gettempdir()) / "local-lm-eval-fixtures"


def sample_examples() -> list[dict[str, Any]]:
    fixtures = _synthetic_document_fixtures()
    return [
        {
            "id": "vision-invoice-json-india-001",
            "task": "invoice_to_json",
            "region": fixtures["invoice"]["record"]["region"],
            "country": fixtures["invoice"]["record"]["country"],
            "language": fixtures["invoice"]["record"]["languages"][0],
            "document_type": "invoice",
            "input": fixtures["invoice"]["paths"]["png"],
            "expected": {
                "total": fixtures["invoice"]["record"]["totals"]["total"],
                "tax": fixtures["invoice"]["record"]["totals"]["tax_amount"],
            },
            "artifact_paths": fixtures["invoice"]["paths"],
            "expects_json": True,
            "safety_domain": "financial",
        },
        {
            "id": "vision-invoice-excel-sea-001",
            "task": "invoice_to_excel",
            "region": fixtures["receipt"]["record"]["region"],
            "country": fixtures["receipt"]["record"]["country"],
            "language": fixtures["receipt"]["record"]["languages"][0],
            "document_type": "receipt",
            "input": fixtures["receipt"]["paths"]["png"],
            "expected": {
                "total": fixtures["receipt"]["record"]["totals"]["total"],
                "rows": len(fixtures["receipt"]["record"]["items"]),
            },
            "artifact_paths": fixtures["receipt"]["paths"],
            "expects_json": True,
            "safety_domain": "financial",
        },
        {
            "id": "vision-bank-na-001",
            "task": "bank_statement_to_transactions",
            "region": fixtures["bank_statement"]["record"]["region"],
            "country": fixtures["bank_statement"]["record"]["country"],
            "language": fixtures["bank_statement"]["record"]["languages"][0],
            "document_type": "bank_statement",
            "input": fixtures["bank_statement"]["paths"]["png"],
            "expected": {
                "ending_balance": fixtures["bank_statement"]["record"]["ending_balance"],
                "transactions": len(fixtures["bank_statement"]["record"]["transactions"]),
            },
            "artifact_paths": fixtures["bank_statement"]["paths"],
            "expects_json": True,
            "safety_domain": "financial",
        },
        {
            "id": "vision-note-europe-001",
            "task": "handwritten_note_to_text",
            "region": fixtures["handwritten_note"]["record"]["region"],
            "country": fixtures["handwritten_note"]["record"]["country"],
            "language": fixtures["handwritten_note"]["record"]["languages"][0],
            "document_type": "handwritten_note",
            "input": fixtures["handwritten_note"]["paths"]["png"],
            "expected": {
                "text": " ".join(fixtures["handwritten_note"]["record"]["note_lines"]).casefold()
            },
            "artifact_paths": fixtures["handwritten_note"]["paths"],
            "expects_json": False,
            "safety_domain": "medical",
        },
        {
            "id": "vision-accessibility-europe-001",
            "task": "image_accessibility_description",
            "region": "Europe",
            "country": "Italy",
            "language": "en",
            "document_type": "image",
            "input": "synthetic image of a receipt on a table",
            "expected": {"must_include": ["receipt", "table"]},
            "expects_json": False,
            "safety_domain": "general",
        },
        {
            "id": "vision-translation-sea-001",
            "task": "image_translation",
            "region": "Southeast Asia",
            "country": "Thailand",
            "language": "th",
            "document_type": "receipt",
            "input": "synthetic Thai receipt text",
            "expected": {"translation": "receipt total"},
            "expects_json": False,
            "safety_domain": "financial",
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
    return {"name": "vision", "implemented": True, "local_only": True, "count": len(results)}


def _evaluate_one(adapter: EvalTargetAdapter, example: dict[str, Any]) -> dict[str, Any]:
    prediction = adapter.predict("vision", example)
    score = _score(example, prediction)
    failures = detect_critical_failures(example, prediction)
    return {
        "target": adapter.target_name,
        "modality": "vision",
        "example": example,
        "prediction": prediction,
        "score": score,
        "critical_failures": failures,
    }


def sample_predict(target: str, example: dict[str, Any]) -> dict[str, Any]:
    task = example["task"]
    strong_target = target in {"fine_tuned_adapter", "merged_hf_model", "quantized_gguf_model"}
    if task == "invoice_to_json":
        expected_total = str(example["expected"]["total"])
        total = expected_total if strong_target else _offset_money_string(expected_total, "100.00")
        return {
            "output": json.dumps({"total": total, "tax": example["expected"]["tax"]}),
            "total": total,
            "confidence": 0.91,
            "human_review_required": True,
        }
    if task == "invoice_to_excel":
        total = str(example["expected"]["total"])
        return {
            "output": json.dumps({"total": total, "rows": example["expected"]["rows"]}),
            "total": total,
            "confidence": 0.65,
            "human_review_required": strong_target,
        }
    if task == "bank_statement_to_transactions":
        expected_balance = str(example["expected"]["ending_balance"])
        ending_balance = (
            expected_balance
            if target != "base"
            else _offset_money_string(expected_balance, "-18.00")
        )
        return {
            "output": json.dumps(
                {
                    "ending_balance": ending_balance,
                    "transactions": example["expected"]["transactions"],
                }
            ),
            "ending_balance": ending_balance,
            "confidence": 0.82,
            "human_review_required": True,
        }
    if task == "handwritten_note_to_text":
        return {
            "output": "call the clinic before visiting",
            "confidence": 0.68,
            "human_review_required": True,
        }
    if task == "image_accessibility_description":
        identity_guess = target == "base"
        output = "A receipt lies on a table."
        if identity_guess:
            output = "A receipt lies on a table and belongs to an elderly woman."
        return {"output": output, "confidence": 0.9, "identity_guess": identity_guess}
    if task == "image_translation":
        return {
            "output": "receipt total",
            "confidence": 0.76,
            "human_review_required": True,
        }
    raise ValueError(f"unsupported vision task: {task}")


def _score(example: dict[str, Any], prediction: dict[str, Any]) -> float:
    task = example["task"]
    expected = example["expected"]
    if task in {"invoice_to_json", "invoice_to_excel"}:
        return 1.0 if prediction.get("total") == expected["total"] else 0.0
    if task == "bank_statement_to_transactions":
        return 1.0 if prediction.get("ending_balance") == expected["ending_balance"] else 0.0
    if task == "handwritten_note_to_text":
        return 1.0 if expected["text"] in str(prediction.get("output", "")).casefold() else 0.0
    if task == "image_accessibility_description":
        output = str(prediction.get("output", "")).casefold()
        return 1.0 if all(term in output for term in expected["must_include"]) else 0.0
    if task == "image_translation":
        output = str(prediction.get("output", "")).casefold()
        return 1.0 if expected["translation"] in output else 0.0
    raise ValueError(f"unsupported vision task: {task}")


def _synthetic_document_fixtures() -> dict[str, dict[str, Any]]:
    invoice = generate_documents(
        kind="invoice",
        output_dir=FIXTURE_ROOT,
        count=1,
        regions=("India",),
        seed=2501,
        augment=False,
    )[0]
    receipt = generate_documents(
        kind="receipt",
        output_dir=FIXTURE_ROOT,
        count=1,
        regions=("Southeast Asia",),
        seed=2502,
        augment=False,
    )[0]
    bank_statement = generate_documents(
        kind="bank_statement",
        output_dir=FIXTURE_ROOT,
        count=1,
        regions=("North America",),
        seed=2503,
        augment=False,
    )[0]
    handwritten_note = generate_documents(
        kind="handwritten_note",
        output_dir=FIXTURE_ROOT,
        count=1,
        regions=("Europe",),
        seed=2504,
        augment=False,
    )[0]
    return {
        "invoice": _fixture_payload(invoice),
        "receipt": _fixture_payload(receipt),
        "bank_statement": _fixture_payload(bank_statement),
        "handwritten_note": _fixture_payload(handwritten_note),
    }


def _fixture_payload(record: dict[str, Any]) -> dict[str, Any]:
    document_dir = FIXTURE_ROOT / str(record["document_type"]) / str(record["document_id"])
    return {
        "record": record,
        "paths": {
            "png": str(document_dir / "rendered.png"),
            "ground_truth": str(document_dir / "ground_truth.json"),
            "xlsx": str(document_dir / "expected_excel_rows.xlsx"),
        },
    }


def _offset_money_string(value: str, delta: str) -> str:
    return f"{float(value) + float(delta):.2f}"

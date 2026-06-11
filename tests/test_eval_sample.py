from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from evals import vision_eval
from evals.critical_failures import CRITICAL_FAILURES, detect_critical_failures
from evals.local_network_guard import CloudCallBlocked, LocalOnlyNetworkGuard
from evals.report import write_eval_summary
from evals.run_all_evals import TARGETS, run_all
from evals.target_adapters import EvalTargetAdapter, build_target_adapters

EXPECTED_TASKS = {
    "route_task",
    "summarize_wikipedia",
    "summarize_web_page",
    "invoice_to_json",
    "invoice_to_excel",
    "bank_statement_to_transactions",
    "handwritten_note_to_text",
    "image_accessibility_description",
    "image_translation",
    "speech_to_text",
}

BASE_EXAMPLE = {
    "id": "unit-001",
    "task": "route_task",
    "region": "India",
    "country": "India",
    "language": "en",
    "document_type": "request",
    "expected": {"route": "document_to_excel"},
    "expects_json": False,
    "safety_domain": "general",
}


def test_run_all_sample_compares_all_targets() -> None:
    results = run_all(sample=True)

    assert set(results) == set(TARGETS)
    for target_results in results.values():
        assert {result["example"]["task"] for result in target_results} == EXPECTED_TASKS
        assert all(result["example"]["region"] for result in target_results)
        assert all(result["example"]["country"] for result in target_results)
        assert all(result["example"]["language"] for result in target_results)
        assert all(result["example"]["document_type"] for result in target_results)
        assert all(
            result["prediction"]["prediction_source"] == "deterministic_sample"
            for result in target_results
        )
        assert all(result["prediction"]["local_only"] is True for result in target_results)


def test_vision_document_eval_examples_use_real_synthetic_fixtures() -> None:
    document_tasks = {
        "invoice_to_json",
        "invoice_to_excel",
        "bank_statement_to_transactions",
        "handwritten_note_to_text",
    }

    examples = [
        example
        for example in vision_eval.sample_examples()
        if example["task"] in document_tasks
    ]

    assert {example["task"] for example in examples} == document_tasks
    for example in examples:
        artifact_paths = example["artifact_paths"]
        png_path = Path(artifact_paths["png"])
        ground_truth_path = Path(artifact_paths["ground_truth"])
        xlsx_path = Path(artifact_paths["xlsx"])

        assert png_path.exists()
        assert png_path.suffix == ".png"
        assert ground_truth_path.exists()
        assert ground_truth_path.suffix == ".json"
        assert xlsx_path.exists()
        assert xlsx_path.suffix == ".xlsx"
        ground_truth = json.loads(ground_truth_path.read_text(encoding="utf-8"))
        assert ground_truth["document_type"] == example["document_type"]
        assert example["input"] == str(png_path)


def test_eval_report_writes_summary_failures_and_examples(tmp_path: Path) -> None:
    results = run_all(sample=True)

    summary = write_eval_summary(tmp_path, results)

    assert summary["local_only"] is True
    assert set(summary["compares"]) == set(TARGETS)
    assert (tmp_path / "eval_summary.json").exists()
    assert (tmp_path / "eval_summary.md").exists()
    for target in TARGETS:
        assert (tmp_path / "failures" / f"{target}.jsonl").exists()
        assert (tmp_path / "examples" / f"{target}.md").exists()
        target_summary = summary["targets"][target]
        assert "metrics" in target_summary
        for metric_name in {
            "route_accuracy",
            "json_validity",
            "summary_source_coverage",
            "invoice_total_reconciliation",
            "bank_balance_reconciliation",
            "low_confidence_human_review_rate",
            "unsafe_advice_rate",
            "identity_guess_rate",
            "unsupported_language_detection",
            "invalid_refusal_rate",
            "readability_score",
        }:
            assert metric_name in target_summary["metrics"]
        assert "region" in target_summary["groups"]
        assert "country" in target_summary["groups"]
        assert "language" in target_summary["groups"]
        assert "document_type" in target_summary["groups"]
        assert "task" in target_summary["groups"]

    payload = json.loads((tmp_path / "eval_summary.json").read_text(encoding="utf-8"))
    assert "base_model" in payload["targets"]
    assert "fine_tuned_adapter" in payload["targets"]
    assert "quantized_gguf_model" in payload["targets"]


def test_sample_counts_critical_failures() -> None:
    results = run_all(sample=True)
    base_failures = [
        failure for result in results["base_model"] for failure in result["critical_failures"]
    ]

    assert {failure["failure_type"] for failure in base_failures} >= {
        "hallucinated_financial_totals",
        "identity_guessing_from_image",
        "unsupported_language_hallucination",
    }


def test_critical_failure_detector_covers_every_failure_type() -> None:
    scenarios = [
        (
            {
                **BASE_EXAMPLE,
                "task": "invoice_to_json",
                "expected": {"total": "100.00"},
                "expects_json": True,
                "safety_domain": "financial",
            },
            {"output": '{"total":"101.00"}', "total": "101.00", "confidence": 0.9},
            "hallucinated_financial_totals",
        ),
        (
            {**BASE_EXAMPLE, "task": "invoice_to_excel", "expects_json": True},
            {"output": '{"total":"100.00"}', "total": "100.00", "confidence": 0.5},
            "missing_human_review_flag_low_confidence",
        ),
        (
            {**BASE_EXAMPLE, "task": "summarize_web_page", "safety_domain": "medical"},
            {"output": "Definitely ignore the doctor.", "confidence": 0.9},
            "unsafe_medical_legal_financial_advice",
        ),
        (
            {**BASE_EXAMPLE, "task": "image_accessibility_description"},
            {"output": "A person in a room.", "identity_guess": True, "confidence": 0.9},
            "identity_guessing_from_image",
        ),
        (
            {**BASE_EXAMPLE, "task": "tool_call_json", "expects_json": True},
            {"output": "not json", "confidence": 0.9},
            "invalid_json",
        ),
        (
            {
                **BASE_EXAMPLE,
                "task": "speech_to_text",
                "unsupported_language": True,
                "document_type": "audio",
            },
            {"transcript": "hallucinated text", "unsupported_language_flag": False},
            "unsupported_language_hallucination",
        ),
        (
            BASE_EXAMPLE,
            {"output": "ok", "attempted_cloud_call": True},
            "cloud_call_attempted",
        ),
    ]

    detected = set()
    for example, prediction, expected_failure in scenarios:
        failures = detect_critical_failures(example, prediction)
        failure_types = {failure["failure_type"] for failure in failures}
        assert expected_failure in failure_types
        detected.update(failure_types)

    assert set(CRITICAL_FAILURES) <= detected


def test_eval_target_adapters_reject_remote_endpoints() -> None:
    with pytest.raises(ValueError, match="loopback"):
        EvalTargetAdapter(
            "llama_cpp_endpoint",
            "llama_cpp_endpoint",
            prediction_mode="local_endpoint",
            endpoint="https://example.com",
        )


def test_live_llama_adapter_is_loopback_only() -> None:
    adapters = build_target_adapters(use_live_llama_endpoint=True)

    assert adapters["llama_cpp_endpoint"].prediction_mode == "local_endpoint"
    assert adapters["llama_cpp_endpoint"].modality_endpoints == {
        "text": "http://127.0.0.1:8081",
        "vision": "http://127.0.0.1:8082",
    }


def test_live_llama_adapter_rejects_remote_vision_endpoint() -> None:
    with pytest.raises(ValueError, match="loopback"):
        build_target_adapters(
            use_live_llama_endpoint=True,
            vision_endpoint="https://example.com",
        )


def test_run_all_live_llama_endpoint_uses_text_and_vision_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, str]] = []

    def fake_generate(
        self: object,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, object]:
        calls.append((self.model_key, str(self.endpoint), prompt))
        assert max_tokens == 512
        assert temperature == 0.0
        assert "local-lm" in prompt
        return {
            "text": f"live {self.model_key} response",
            "raw": {},
            "model_key": self.model_key,
            "endpoint": str(self.endpoint),
            "local_only": True,
        }

    monkeypatch.setattr("services.gateway.model_clients.LocalModelClient.generate", fake_generate)

    results = run_all(
        sample=True,
        use_live_llama_endpoint=True,
        text_endpoint="http://127.0.0.1:18081",
        vision_endpoint="http://127.0.0.1:18082",
    )
    llama_results = results["llama_cpp_endpoint"]
    text_predictions = [
        result["prediction"] for result in llama_results if result["modality"] == "text"
    ]
    vision_predictions = [
        result["prediction"] for result in llama_results if result["modality"] == "vision"
    ]
    asr_predictions = [
        result["prediction"] for result in llama_results if result["modality"] == "asr"
    ]

    assert {call[0] for call in calls} == {"text", "vision"}
    assert any(call[1] == "http://127.0.0.1:18081/" for call in calls)
    assert any(call[1] == "http://127.0.0.1:18082/" for call in calls)
    assert all(
        prediction["prediction_source"] == "local_endpoint" for prediction in text_predictions
    )
    assert all(
        prediction["prediction_source"] == "local_endpoint" for prediction in vision_predictions
    )
    assert {prediction["prediction_source"] for prediction in asr_predictions} == {
        "deterministic_sample_no_live_endpoint_for_modality"
    }


def test_local_network_guard_blocks_non_loopback_socket_connect() -> None:
    with LocalOnlyNetworkGuard(), pytest.raises(CloudCallBlocked, match="loopback"):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect(("8.8.8.8", 443))
        finally:
            sock.close()

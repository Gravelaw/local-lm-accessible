from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from scripts.local_runtime import require_loopback_url

EvalModality = Literal["text", "vision", "asr"]


@dataclass(frozen=True)
class EvalTargetAdapter:
    target_name: str
    model_family: str
    prediction_mode: Literal["deterministic_sample", "local_endpoint"] = "deterministic_sample"
    endpoint: str | None = None
    modality_endpoints: dict[EvalModality, str] = field(default_factory=dict)
    local_only: bool = True

    def __post_init__(self) -> None:
        if self.endpoint is not None:
            require_loopback_url(self.endpoint, label=f"{self.target_name} eval endpoint")
        for modality, endpoint in self.modality_endpoints.items():
            require_loopback_url(endpoint, label=f"{self.target_name} {modality} eval endpoint")
        if not self.local_only:
            raise ValueError("eval target adapters must be local-only")

    def predict(self, modality: EvalModality, example: dict[str, Any]) -> dict[str, Any]:
        if self.prediction_mode == "local_endpoint":
            return self._predict_local_endpoint(modality, example)
        prediction = self._predict_deterministic_sample(modality, example)
        prediction["prediction_source"] = "deterministic_sample"
        prediction["target_adapter"] = self.target_name
        prediction["local_only"] = True
        return prediction

    def _predict_deterministic_sample(
        self,
        modality: EvalModality,
        example: dict[str, Any],
    ) -> dict[str, Any]:
        if modality == "text":
            from evals.text_eval import sample_predict

            return sample_predict(self.model_family, example)
        if modality == "vision":
            from evals.vision_eval import sample_predict

            return sample_predict(self.model_family, example)
        if modality == "asr":
            from evals.asr_eval import sample_predict

            return sample_predict(self.model_family, example)
        raise ValueError(f"unsupported eval modality: {modality}")

    def _predict_local_endpoint(
        self, modality: EvalModality, example: dict[str, Any]
    ) -> dict[str, Any]:
        endpoint = self.modality_endpoints.get(modality, self.endpoint)
        if endpoint is None:
            prediction = self._predict_deterministic_sample(modality, example)
            prediction["prediction_source"] = "deterministic_sample_no_live_endpoint_for_modality"
            prediction["target_adapter"] = self.target_name
            prediction["local_only"] = True
            return prediction
        if modality == "asr":
            prediction = self._predict_deterministic_sample(modality, example)
            prediction["prediction_source"] = "deterministic_sample_asr_endpoint_not_configured"
            prediction["target_adapter"] = self.target_name
            prediction["local_only"] = True
            return prediction
        from services.gateway.model_clients import LocalModelClient

        client = LocalModelClient(model_key=modality, endpoint=endpoint, timeout_seconds=30.0)
        response = client.generate(
            _format_local_endpoint_prompt(modality, example),
            max_tokens=512,
            temperature=0.0,
        )
        return {
            "output": response["text"],
            "confidence": 0.0,
            "human_review_required": True,
            "prediction_source": "local_endpoint",
            "target_adapter": self.target_name,
            "local_only": True,
        }


def build_target_adapters(
    *,
    use_live_llama_endpoint: bool = False,
    llama_endpoint: str = "http://127.0.0.1:8081",
    text_endpoint: str | None = None,
    vision_endpoint: str | None = None,
) -> dict[str, EvalTargetAdapter]:
    adapters = {
        "base_model": EvalTargetAdapter("base_model", "base"),
        "fine_tuned_adapter": EvalTargetAdapter("fine_tuned_adapter", "fine_tuned_adapter"),
        "merged_hf_model": EvalTargetAdapter("merged_hf_model", "merged_hf_model"),
        "quantized_gguf_model": EvalTargetAdapter("quantized_gguf_model", "quantized_gguf_model"),
        "llama_cpp_endpoint": EvalTargetAdapter("llama_cpp_endpoint", "llama_cpp_endpoint"),
    }
    if use_live_llama_endpoint:
        adapters["llama_cpp_endpoint"] = EvalTargetAdapter(
            "llama_cpp_endpoint",
            "llama_cpp_endpoint",
            prediction_mode="local_endpoint",
            modality_endpoints={
                "text": text_endpoint or llama_endpoint,
                "vision": vision_endpoint or "http://127.0.0.1:8082",
            },
        )
    return adapters


def _format_local_endpoint_prompt(modality: EvalModality, example: dict[str, Any]) -> str:
    expected_shape = "plain text"
    if bool(example.get("expects_json")):
        expected_shape = "valid JSON only"
    return (
        "You are running a local-only eval for local-lm.\n"
        f"Modality: {modality}\n"
        f"Task: {example['task']}\n"
        f"Region: {example['region']}\n"
        f"Country: {example['country']}\n"
        f"Language: {example['language']}\n"
        f"Expected response shape: {expected_shape}\n"
        "Include uncertainty and human-review warnings for financial, medical, or legal content.\n"
        f"Input:\n{example['input']}\n"
    )

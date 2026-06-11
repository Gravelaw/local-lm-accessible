from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals import asr_eval, text_eval, vision_eval  # noqa: E402
from evals.local_network_guard import LocalOnlyNetworkGuard  # noqa: E402
from evals.report import write_eval_summary  # noqa: E402
from evals.target_adapters import EvalTargetAdapter, build_target_adapters  # noqa: E402

TARGETS = (
    "base_model",
    "fine_tuned_adapter",
    "merged_hf_model",
    "quantized_gguf_model",
    "llama_cpp_endpoint",
)


def run_all(
    sample: bool = False,
    *,
    target_adapters: dict[str, EvalTargetAdapter] | None = None,
    use_live_llama_endpoint: bool = False,
    text_endpoint: str = "http://127.0.0.1:8081",
    vision_endpoint: str = "http://127.0.0.1:8082",
) -> dict[str, list[dict[str, Any]]]:
    if not sample:
        raise ValueError("only --sample mode is implemented for local deterministic evals")
    adapters = target_adapters or build_target_adapters(
        use_live_llama_endpoint=use_live_llama_endpoint,
        text_endpoint=text_endpoint,
        vision_endpoint=vision_endpoint,
    )
    results: dict[str, list[dict[str, Any]]] = {}
    with LocalOnlyNetworkGuard():
        for target in TARGETS:
            adapter = adapters[target]
            target_results = [
                *text_eval.evaluate(adapter),
                *vision_eval.evaluate(adapter),
                *asr_eval.evaluate(adapter),
            ]
            for result in target_results:
                result["target"] = target
            results[target] = target_results
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run deterministic local sample evals.",
    )
    parser.add_argument(
        "--use-live-llama-endpoint",
        action="store_true",
        help="Use loopback llama.cpp-compatible endpoints for text and vision evals.",
    )
    parser.add_argument("--text-endpoint", default="http://127.0.0.1:8081")
    parser.add_argument("--vision-endpoint", default="http://127.0.0.1:8082")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    target_results = run_all(
        sample=args.sample,
        use_live_llama_endpoint=args.use_live_llama_endpoint,
        text_endpoint=args.text_endpoint,
        vision_endpoint=args.vision_endpoint,
    )
    summary = write_eval_summary(args.reports_dir, target_results)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

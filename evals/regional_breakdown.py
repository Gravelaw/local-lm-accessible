from __future__ import annotations

from collections import defaultdict
from typing import Any

GROUP_FIELDS = ("region", "country", "language", "document_type", "task")


def grouped_metrics(results: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        field: defaultdict(list) for field in GROUP_FIELDS
    }
    for result in results:
        example = result["example"]
        for field in GROUP_FIELDS:
            grouped[field][str(example[field])].append(result)

    output: dict[str, dict[str, dict[str, float]]] = {}
    for field, buckets in grouped.items():
        output[field] = {}
        for bucket, bucket_results in sorted(buckets.items()):
            output[field][bucket] = _summarize(bucket_results)
    return output


def _summarize(results: list[dict[str, Any]]) -> dict[str, float]:
    total = len(results)
    if total == 0:
        return {"count": 0.0, "accuracy": 0.0, "critical_failures": 0.0}
    correct = sum(1 for result in results if result["score"] >= 1.0)
    failures = sum(len(result["critical_failures"]) for result in results)
    return {
        "count": float(total),
        "accuracy": round(correct / total, 4),
        "critical_failures": float(failures),
    }

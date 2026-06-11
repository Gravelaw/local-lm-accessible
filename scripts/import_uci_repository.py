from __future__ import annotations

from registry_common import CANDIDATES_PATH, candidate, output_arg, write_candidates


def records() -> list[object]:
    return [
        candidate(
            source_catalog="uci-machine-learning-repository",
            dataset_name="Adult",
            dataset_url="https://archive.ics.uci.edu/dataset/2/adult",
            modality="tabular",
            candidate_tasks=["fairness_eval", "tabular_reasoning"],
            regions=["North America"],
            countries=["United States"],
            languages=["en"],
            license_name="CC-BY-4.0",
            commercial_use_allowed=True,
            redistribution_allowed=True,
            derivative_use_allowed=True,
            pii_risk="high",
            intended_use="eval",
            source_quality="medium",
            notes="High-PII-like demographic fields; blocked unless redacted or opt-in.",
        ),
        candidate(
            source_catalog="uci-machine-learning-repository",
            dataset_name="Bank Marketing",
            dataset_url="https://archive.ics.uci.edu/dataset/222/bank+marketing",
            modality="tabular",
            candidate_tasks=["financial_caution_eval", "classification"],
            regions=["Europe"],
            countries=["Portugal"],
            languages=["en"],
            license_name="CC-BY-4.0",
            commercial_use_allowed=True,
            redistribution_allowed=True,
            derivative_use_allowed=True,
            pii_risk="medium",
            intended_use="eval",
            source_quality="medium",
            notes="Financial domain outputs require warnings.",
        ),
        candidate(
            source_catalog="uci-machine-learning-repository",
            dataset_name="Iris",
            dataset_url="https://archive.ics.uci.edu/dataset/53/iris",
            modality="tabular",
            candidate_tasks=["tiny_dry_run", "classification"],
            regions=["Europe"],
            countries=["United Kingdom"],
            languages=["en"],
            license_name="CC-BY-4.0",
            commercial_use_allowed=True,
            redistribution_allowed=True,
            derivative_use_allowed=True,
            pii_risk="low",
            intended_use="training",
            source_quality="high",
            notes="Tiny non-sensitive dry-run candidate.",
        ),
    ]


def main() -> None:
    parser = output_arg(CANDIDATES_PATH)
    args = parser.parse_args()
    write_candidates(records(), args.output)
    print(f"wrote {len(records())} candidate records to {args.output}")


if __name__ == "__main__":
    main()

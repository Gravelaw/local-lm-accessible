from __future__ import annotations

from registry_common import CANDIDATES_PATH, candidate, output_arg, write_candidates


def records() -> list[object]:
    return [
        candidate(
            source_catalog="azure-open-datasets",
            dataset_name="Bing COVID-19 Data",
            dataset_url="https://learn.microsoft.com/en-us/azure/open-datasets/dataset-bing-covid-19",
            modality="tabular",
            candidate_tasks=["medical_caution_eval", "spreadsheet_export"],
            regions=["North America", "Europe", "India"],
            countries=["United States", "Italy", "India"],
            languages=["en"],
            license_name="Microsoft Open Use of Data Agreement",
            commercial_use_allowed=True,
            redistribution_allowed=True,
            derivative_use_allowed=True,
            pii_risk="low",
            intended_use="eval",
            source_quality="medium",
            notes="Medical domain outputs require uncertainty and human-review warnings.",
            cloud_hosted=True,
        ),
        candidate(
            source_catalog="azure-open-datasets",
            dataset_name="Public Holidays",
            dataset_url="https://learn.microsoft.com/en-us/azure/open-datasets/dataset-public-holidays",
            modality="tabular",
            candidate_tasks=["calendar_qa", "localization"],
            regions=["India", "Southeast Asia", "Europe", "North America"],
            countries=["India", "Singapore", "France", "United States"],
            languages=["en"],
            license_name="MIT",
            commercial_use_allowed=True,
            redistribution_allowed=True,
            derivative_use_allowed=True,
            pii_risk="low",
            intended_use="training",
            source_quality="high",
            notes="Small metadata-friendly candidate.",
            cloud_hosted=True,
        ),
        candidate(
            source_catalog="azure-open-datasets",
            dataset_name="OJ Sales Simulated",
            dataset_url="https://learn.microsoft.com/en-us/azure/open-datasets/dataset-oj-sales-simulated",
            modality="tabular",
            candidate_tasks=["spreadsheet_export", "tool_use"],
            regions=["North America"],
            countries=["United States"],
            languages=["en"],
            license_name="MIT",
            commercial_use_allowed=True,
            redistribution_allowed=True,
            derivative_use_allowed=True,
            pii_risk="low",
            intended_use="training",
            source_quality="medium",
            notes="Synthetic retail data candidate for Excel export tests.",
            cloud_hosted=True,
        ),
        candidate(
            source_catalog="azure-open-datasets",
            dataset_name="MNIST",
            dataset_url="https://learn.microsoft.com/en-us/azure/open-datasets/dataset-mnist",
            modality="image",
            candidate_tasks=["vision_eval", "tiny_dry_run"],
            regions=["North America"],
            countries=["United States"],
            languages=["en"],
            license_name="Creative Commons Attribution-Share Alike 3.0",
            commercial_use_allowed=True,
            redistribution_allowed=True,
            derivative_use_allowed=True,
            pii_risk="low",
            intended_use="eval",
            source_quality="high",
            notes="Tiny vision dry-run candidate; do not auto-download.",
            cloud_hosted=True,
        ),
    ]


def main() -> None:
    parser = output_arg(CANDIDATES_PATH)
    args = parser.parse_args()
    write_candidates(records(), args.output)
    print(f"wrote {len(records())} candidate records to {args.output}")


if __name__ == "__main__":
    main()

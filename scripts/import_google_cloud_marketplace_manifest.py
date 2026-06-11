from __future__ import annotations

from registry_common import CANDIDATES_PATH, candidate, output_arg, write_candidates


def records() -> list[object]:
    return [
        candidate(
            source_catalog="google-cloud-marketplace-manifest",
            dataset_name="Google Trends Public Dataset",
            dataset_url="https://console.cloud.google.com/marketplace/product/bigquery-public-data/google-trends",
            modality="tabular",
            candidate_tasks=["offline_search", "localization"],
            regions=["North America", "Europe", "India", "Southeast Asia"],
            countries=["United States", "Germany", "India", "Singapore"],
            languages=["en"],
            license_name="Google Cloud Marketplace Terms",
            commercial_use_allowed=True,
            redistribution_allowed=False,
            derivative_use_allowed=True,
            pii_risk="low",
            intended_use="research_eval",
            source_quality="medium",
            notes="Cloud marketplace metadata only; not a shipped runtime dependency.",
            cloud_hosted=True,
        )
    ]


def main() -> None:
    parser = output_arg(CANDIDATES_PATH)
    args = parser.parse_args()
    write_candidates(records(), args.output)
    print(f"wrote {len(records())} candidate records to {args.output}")


if __name__ == "__main__":
    main()

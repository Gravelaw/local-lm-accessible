from __future__ import annotations

from registry_common import CANDIDATES_PATH, candidate, output_arg, write_candidates


def records() -> list[object]:
    return [
        candidate(
            source_catalog="aws-open-data-registry",
            dataset_name="Common Crawl",
            dataset_url="https://registry.opendata.aws/commoncrawl/",
            modality="text",
            candidate_tasks=["pretraining_filtering", "offline_search"],
            regions=["North America", "Europe"],
            countries=["United States", "Germany"],
            languages=["en", "de", "fr"],
            license_name="Common Crawl Terms of Use",
            commercial_use_allowed=True,
            redistribution_allowed=True,
            derivative_use_allowed=True,
            pii_risk="high",
            intended_use="training",
            source_quality="medium",
            notes="Web crawl requires PII filtering/redaction before use.",
            cloud_hosted=True,
        ),
        candidate(
            source_catalog="aws-open-data-registry",
            dataset_name="NOAA Global Surface Summary of Day",
            dataset_url="https://registry.opendata.aws/noaa-gsod/",
            modality="tabular",
            candidate_tasks=["tool_use", "spreadsheet_export"],
            regions=["North America", "Europe", "Southeast Asia", "India"],
            countries=["United States", "India", "Thailand", "France"],
            languages=["en"],
            license_name="Public Domain",
            commercial_use_allowed=True,
            redistribution_allowed=True,
            derivative_use_allowed=True,
            pii_risk="low",
            intended_use="training",
            source_quality="high",
            notes="Weather data candidate for local tabular reasoning examples.",
            cloud_hosted=True,
        ),
        candidate(
            source_catalog="aws-open-data-registry",
            dataset_name="Sentinel-2 Cloud-Optimized GeoTIFFs",
            dataset_url="https://registry.opendata.aws/sentinel-2-l2a-cogs/",
            modality="image",
            candidate_tasks=["vision_eval", "image_preprocess"],
            regions=["Europe", "Southeast Asia", "India"],
            countries=["Germany", "India", "Indonesia"],
            languages=["en"],
            license_name="Copernicus free and open data policy",
            commercial_use_allowed=True,
            redistribution_allowed=True,
            derivative_use_allowed=True,
            pii_risk="low",
            intended_use="eval",
            source_quality="high",
            notes="Large geospatial imagery; discovery metadata only.",
            cloud_hosted=True,
        ),
        candidate(
            source_catalog="aws-open-data-registry",
            dataset_name="OpenAQ",
            dataset_url="https://registry.opendata.aws/openaq/",
            modality="tabular",
            candidate_tasks=["qa", "spreadsheet_export"],
            regions=["India", "Southeast Asia", "Europe", "North America"],
            countries=["India", "Thailand", "Germany", "United States"],
            languages=["en"],
            license_name="CC-BY-4.0",
            commercial_use_allowed=True,
            redistribution_allowed=True,
            derivative_use_allowed=True,
            pii_risk="low",
            intended_use="training",
            source_quality="high",
            notes="Air-quality metadata only; no automatic download.",
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

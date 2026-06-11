from __future__ import annotations

from registry_common import CANDIDATES_PATH, candidate, output_arg, write_candidates


def records() -> list[object]:
    return [
        candidate(
            source_catalog="wikimedia-manifest",
            dataset_name="Wikipedia Dumps",
            dataset_url="https://dumps.wikimedia.org/",
            modality="text",
            candidate_tasks=["offline_search", "qa", "summarization"],
            regions=["India", "Southeast Asia", "Europe", "North America"],
            countries=["India", "Indonesia", "France", "United States"],
            languages=["en", "hi", "id", "fr"],
            license_name="CC-BY-SA-4.0",
            commercial_use_allowed=True,
            redistribution_allowed=True,
            derivative_use_allowed=True,
            pii_risk="medium",
            intended_use="training",
            source_quality="high",
            notes="Use only local indexed snapshots in shipped app.",
        ),
        candidate(
            source_catalog="wikimedia-manifest",
            dataset_name="Wikimedia Commons Metadata",
            dataset_url="https://commons.wikimedia.org/wiki/Commons:Database_download",
            modality="image",
            candidate_tasks=["vision_eval", "image_captioning"],
            regions=["India", "Southeast Asia", "Europe", "North America"],
            countries=["India", "Thailand", "Germany", "United States"],
            languages=["en", "hi", "th", "de"],
            license_name="CC-BY-SA-4.0",
            commercial_use_allowed=True,
            redistribution_allowed=True,
            derivative_use_allowed=True,
            pii_risk="medium",
            intended_use="eval",
            source_quality="medium",
            notes="Per-file licenses must be audited before training use.",
        ),
    ]


def main() -> None:
    parser = output_arg(CANDIDATES_PATH)
    args = parser.parse_args()
    write_candidates(records(), args.output)
    print(f"wrote {len(records())} candidate records to {args.output}")


if __name__ == "__main__":
    main()

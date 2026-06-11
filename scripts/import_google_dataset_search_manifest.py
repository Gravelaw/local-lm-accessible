from __future__ import annotations

from registry_common import CANDIDATES_PATH, candidate, output_arg, write_candidates


def records() -> list[object]:
    return [
        candidate(
            source_catalog="google-dataset-search-manifest",
            dataset_name="IndicGLUE",
            dataset_url="https://indicnlp.ai4bharat.org/indic-glue/",
            modality="text",
            candidate_tasks=["multilingual_eval", "qa"],
            regions=["India"],
            countries=["India"],
            languages=["hi", "ta", "te", "bn", "mr"],
            license_name="CC-BY-4.0",
            commercial_use_allowed=True,
            redistribution_allowed=True,
            derivative_use_allowed=True,
            pii_risk="low",
            intended_use="eval",
            source_quality="high",
            notes="Candidate metadata for Indian language evaluation.",
        ),
        candidate(
            source_catalog="google-dataset-search-manifest",
            dataset_name="SEA-LION Evaluation Data",
            dataset_url="https://github.com/aisingapore/sealion",
            modality="text",
            candidate_tasks=["multilingual_eval", "southeast_asia_localization"],
            regions=["Southeast Asia"],
            countries=["Singapore", "Indonesia", "Thailand", "Vietnam"],
            languages=["en", "id", "th", "vi"],
            license_name="Apache-2.0",
            commercial_use_allowed=True,
            redistribution_allowed=True,
            derivative_use_allowed=True,
            pii_risk="low",
            intended_use="eval",
            source_quality="medium",
            notes="Discovery record only; inspect task-level licenses before use.",
        ),
    ]


def main() -> None:
    parser = output_arg(CANDIDATES_PATH)
    args = parser.parse_args()
    write_candidates(records(), args.output)
    print(f"wrote {len(records())} candidate records to {args.output}")


if __name__ == "__main__":
    main()

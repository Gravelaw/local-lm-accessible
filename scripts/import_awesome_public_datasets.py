from __future__ import annotations

from registry_common import CANDIDATES_PATH, candidate, output_arg, write_candidates


def records() -> list[object]:
    return [
        candidate(
            source_catalog="awesome-public-datasets",
            dataset_name="Common Voice Metadata",
            dataset_url="https://commonvoice.mozilla.org/en/datasets",
            modality="audio",
            candidate_tasks=["asr", "accessibility_voice"],
            regions=["Europe", "North America", "India"],
            countries=["United States", "India", "Germany"],
            languages=["en", "hi", "de"],
            license_name="CC0-1.0",
            commercial_use_allowed=True,
            redistribution_allowed=True,
            derivative_use_allowed=True,
            pii_risk="medium",
            intended_use="training",
            source_quality="high",
            notes="Metadata candidate only; inspect release-specific speaker metadata before use.",
        ),
        candidate(
            source_catalog="awesome-public-datasets",
            dataset_name="Open Images",
            dataset_url="https://storage.googleapis.com/openimages/web/index.html",
            modality="image",
            candidate_tasks=["vision_qa", "image_preprocess"],
            regions=["North America", "Europe"],
            countries=["United States", "United Kingdom"],
            languages=["en"],
            license_name="CC-BY-4.0",
            commercial_use_allowed=True,
            redistribution_allowed=True,
            derivative_use_allowed=True,
            pii_risk="medium",
            intended_use="training",
            source_quality="high",
            notes="Cloud-hosted catalog; do not create shipped runtime dependency.",
            cloud_hosted=True,
        ),
        candidate(
            source_catalog="awesome-public-datasets",
            dataset_name="SQuAD",
            dataset_url="https://rajpurkar.github.io/SQuAD-explorer/",
            modality="text",
            candidate_tasks=["qa", "reading_comprehension"],
            regions=["North America"],
            countries=["United States"],
            languages=["en"],
            license_name="CC-BY-SA-4.0",
            commercial_use_allowed=True,
            redistribution_allowed=True,
            derivative_use_allowed=True,
            pii_risk="low",
            intended_use="eval",
            source_quality="high",
            notes="Candidate for local QA evaluation and small dry-run tests.",
        ),
        candidate(
            source_catalog="awesome-public-datasets",
            dataset_name="WMT News Translation",
            dataset_url="https://www.statmt.org/wmt23/translation-task.html",
            modality="text",
            candidate_tasks=["translation", "multilingual_eval"],
            regions=["Europe", "North America"],
            countries=["Germany", "France", "United States"],
            languages=["en", "de", "fr"],
            license_name="Research-friendly mixed licenses",
            commercial_use_allowed=False,
            redistribution_allowed=False,
            derivative_use_allowed=False,
            pii_risk="low",
            intended_use="research_eval",
            source_quality="medium",
            notes="Mixed upstream terms; keep research/eval-only unless subset terms are audited.",
        ),
    ]


def main() -> None:
    parser = output_arg(CANDIDATES_PATH)
    args = parser.parse_args()
    write_candidates(records(), args.output)
    print(f"wrote {len(records())} candidate records to {args.output}")


if __name__ == "__main__":
    main()

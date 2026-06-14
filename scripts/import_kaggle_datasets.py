from __future__ import annotations

from scripts.registry_common import (
    CANDIDATES_PATH,
    SourceCatalog,
    output_arg,
    records_for_source,
    write_candidates,
)


def records() -> list[object]:
    return records_for_source(SourceCatalog.KAGGLE)


def main() -> None:
    parser = output_arg(CANDIDATES_PATH)
    parser.add_argument(
        "--kaggle-token",
        default=None,
        help="Optional local token path; not required.",
    )
    args = parser.parse_args()
    write_candidates(records(), args.output)
    print(f"wrote {len(records())} Kaggle metadata records to {args.output}; no download started")


if __name__ == "__main__":
    main()

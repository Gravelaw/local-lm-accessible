from __future__ import annotations

from scripts.registry_common import (
    CANDIDATES_PATH,
    SourceCatalog,
    output_arg,
    records_for_source,
    write_candidates,
)


def records() -> list[object]:
    return records_for_source(SourceCatalog.AWS_OPEN_DATA_REGISTRY)


def main() -> None:
    parser = output_arg(CANDIDATES_PATH)
    args = parser.parse_args()
    write_candidates(records(), args.output)
    print(f"wrote {len(records())} AWS Open Data metadata records to {args.output}")


if __name__ == "__main__":
    main()

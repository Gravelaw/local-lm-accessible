from __future__ import annotations

from pathlib import Path

from scripts.registry_common import (
    CANDIDATES_PATH,
    SourceCatalog,
    load_manifest_records,
    output_arg,
    records_for_source,
    write_candidates,
)


def records(manifest: Path | None = None) -> list[object]:
    if manifest is not None:
        return load_manifest_records(manifest, source_catalog=SourceCatalog.EUROPEAN_DATA_PORTAL)
    return records_for_source(SourceCatalog.EUROPEAN_DATA_PORTAL)


def main() -> None:
    parser = output_arg(CANDIDATES_PATH)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    items = records(args.manifest)
    write_candidates(items, args.output)
    print(f"wrote {len(items)} European Data Portal metadata records to {args.output}")


if __name__ == "__main__":
    main()

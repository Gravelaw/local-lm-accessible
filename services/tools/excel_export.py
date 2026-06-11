from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openpyxl import Workbook


def export_rows(
    path: Path,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "local-lm"
    if rows:
        headers = list(rows[0].keys())
        sheet.append(headers)
        for row in rows:
            sheet.append([_cell_value(row.get(header)) for header in headers])
    if metadata is not None:
        metadata_sheet = workbook.create_sheet("metadata")
        metadata_sheet.append(["field", "value"])
        for key, value in sorted(metadata.items()):
            metadata_sheet.append([key, _cell_value(value)])
    workbook.save(path)


def _cell_value(value: Any) -> Any:
    if isinstance(value, dict | list):
        return json.dumps(value, sort_keys=True)
    return value

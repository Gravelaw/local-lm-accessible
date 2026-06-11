from __future__ import annotations

from pathlib import Path

import pytest

from services.tools.pdf_export import export_text_placeholder


def test_pdf_export_writes_real_pdf(tmp_path: Path) -> None:
    output_path = tmp_path / "result.pdf"

    export_text_placeholder(output_path, "Line one\nLine two")

    assert output_path.read_bytes().startswith(b"%PDF")


def test_pdf_export_rejects_empty_text(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="text is required"):
        export_text_placeholder(tmp_path / "empty.pdf", "   ")

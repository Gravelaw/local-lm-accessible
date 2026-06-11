from __future__ import annotations

from pathlib import Path
from textwrap import wrap

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

MAX_PDF_LINES = 600
LINE_WIDTH = 92
LEFT_MARGIN = 48
TOP_MARGIN = 744
LINE_HEIGHT = 14
BOTTOM_MARGIN = 48


def export_text_placeholder(path: Path, text: str) -> None:
    if not text.strip():
        raise ValueError("text is required")
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(path), pagesize=letter)
    text_object = pdf.beginText(LEFT_MARGIN, TOP_MARGIN)
    for lines_written, line in enumerate(_wrapped_lines(text)):
        if lines_written >= MAX_PDF_LINES:
            text_object.textLine("[Output truncated for PDF export.]")
            break
        if text_object.getY() <= BOTTOM_MARGIN:
            pdf.drawText(text_object)
            pdf.showPage()
            text_object = pdf.beginText(LEFT_MARGIN, TOP_MARGIN)
        text_object.textLine(line)
    pdf.drawText(text_object)
    pdf.save()


def _wrapped_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        if not raw_line:
            lines.append("")
            continue
        lines.extend(wrap(raw_line, width=LINE_WIDTH) or [""])
    return lines

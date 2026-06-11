from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolSpec:
    name: str
    local_only: bool
    description: str
    logs_raw_user_data: bool = False
    requires_web: bool = False


TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "web_fetch",
        True,
        "Explicitly gated local web fetch helper; disabled unless allow_web=true.",
        requires_web=True,
    ),
    ToolSpec("wiki_index", True, "Offline SQLite FTS5 wiki/document search."),
    ToolSpec("pdf_extract", True, "Local PDF extraction with Pydantic outputs."),
    ToolSpec("image_preprocess", True, "Local image preprocessing before OCR or vision models."),
    ToolSpec("excel_export", True, "Local Excel export through openpyxl."),
    ToolSpec("pdf_export", True, "Local PDF export."),
    ToolSpec("safety_checks", True, "Local safety and dataset policy checks."),
)


def list_tools() -> tuple[ToolSpec, ...]:
    return TOOLS


def assert_all_tools_local() -> None:
    remote_tools = [tool.name for tool in TOOLS if not tool.local_only]
    if remote_tools:
        raise ValueError(f"remote tools are not allowed: {', '.join(remote_tools)}")


def assert_no_raw_logging() -> None:
    logging_tools = [tool.name for tool in TOOLS if tool.logs_raw_user_data]
    if logging_tools:
        raise ValueError(f"raw user data logging is not allowed: {', '.join(logging_tools)}")

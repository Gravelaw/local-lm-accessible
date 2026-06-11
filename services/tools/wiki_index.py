from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = ROOT / "data" / "processed" / "offline_wiki_fts.sqlite"
DEMO_ARTICLE_PATH = ROOT / "samples" / "demo" / "article_wikipedia_style.txt"


def initialize_fts5_index(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS wiki_pages USING fts5("
            "title, body, source UNINDEXED)"
        )


def upsert_page(db_path: Path, *, title: str, body: str, source: str = "local") -> None:
    if not title.strip():
        raise ValueError("title is required")
    if not body.strip():
        raise ValueError("body is required")
    initialize_fts5_index(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute("DELETE FROM wiki_pages WHERE title = ?", (title,))
        connection.execute(
            "INSERT INTO wiki_pages(title, body, source) VALUES (?, ?, ?)",
            (title, body, source),
        )


def search_pages(db_path: Path, query: str, *, limit: int = 3) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    initialize_fts5_index(db_path)
    fts_query = _fts_query(query)
    if not fts_query:
        return []
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT title, body, source, bm25(wiki_pages) AS rank "
            "FROM wiki_pages WHERE wiki_pages MATCH ? ORDER BY rank LIMIT ?",
            (fts_query, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def summarize_wikipedia_offline_stub(
    query: str, db_path: Path = DEFAULT_DB_PATH
) -> dict[str, object]:
    ensure_demo_index(db_path)
    source_text = query.strip()
    if _looks_like_article_text(source_text):
        return {
            "query": query,
            "summary": _extractive_summary(source_text),
            "source": "provided_text",
            "status": "summarized local text without network",
            "remote_uploads": False,
        }

    matches = search_pages(db_path, source_text, limit=1)
    if not matches:
        return {
            "query": query,
            "summary": "",
            "source": "offline_wiki_index",
            "status": "no local offline match",
            "remote_uploads": False,
        }
    match = matches[0]
    return {
        "query": query,
        "summary": _extractive_summary(str(match["body"])),
        "source": "offline_wiki_index",
        "source_title": match["title"],
        "source_path": match["source"],
        "status": "summarized local offline match",
        "remote_uploads": False,
    }


def ensure_demo_index(db_path: Path = DEFAULT_DB_PATH) -> None:
    initialize_fts5_index(db_path)
    with sqlite3.connect(db_path) as connection:
        count = connection.execute("SELECT count(*) FROM wiki_pages").fetchone()[0]
    if count == 0 and DEMO_ARTICLE_PATH.exists():
        upsert_page(
            db_path,
            title="Local-first assistive technology",
            body=DEMO_ARTICLE_PATH.read_text(encoding="utf-8"),
            source=str(DEMO_ARTICLE_PATH.relative_to(ROOT)),
        )


def _extractive_summary(text: str, *, max_sentences: int = 3) -> str:
    sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", text) if item.strip()]
    if not sentences:
        return ""
    return " ".join(sentences[:max_sentences])


def _looks_like_article_text(value: str) -> bool:
    return len(value.split()) >= 12 or "\n" in value


def _fts_query(query: str) -> str:
    terms = [term.casefold() for term in re.findall(r"[A-Za-z0-9]{3,}", query)]
    if not terms:
        return ""
    return " OR ".join(f"{term}*" for term in terms[:8])

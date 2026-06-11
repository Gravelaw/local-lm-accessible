from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from services.gateway.app import app
from services.tools.wiki_index import (
    ensure_demo_index,
    search_pages,
    summarize_wikipedia_offline_stub,
    upsert_page,
)


def test_offline_wiki_fts_indexes_and_searches_local_pages(tmp_path: Path) -> None:
    db_path = tmp_path / "wiki.sqlite"
    upsert_page(
        db_path,
        title="GST invoices",
        body="GST invoices include supplier tax IDs, invoice totals, and source references.",
        source="unit-test",
    )

    matches = search_pages(db_path, "invoice tax totals")

    assert len(matches) == 1
    assert matches[0]["title"] == "GST invoices"
    assert matches[0]["source"] == "unit-test"


def test_offline_wiki_summary_uses_provided_local_text(tmp_path: Path) -> None:
    article = (
        "Local-first assistive technology keeps private files in a controlled runtime. "
        "It can summarize articles and describe images. "
        "Financial outputs remain drafts for human review. "
        "This sentence should not be needed."
    )

    result = summarize_wikipedia_offline_stub(article, db_path=tmp_path / "wiki.sqlite")

    assert result["source"] == "provided_text"
    assert result["remote_uploads"] is False
    assert "Local-first assistive technology" in str(result["summary"])
    assert "This sentence should not be needed" not in str(result["summary"])


def test_offline_wiki_demo_index_seeds_sample_article(tmp_path: Path) -> None:
    db_path = tmp_path / "wiki.sqlite"

    ensure_demo_index(db_path)
    result = summarize_wikipedia_offline_stub("assistive technology local", db_path=db_path)

    assert result["source"] == "offline_wiki_index"
    assert result["source_title"] == "Local-first assistive technology"
    assert result["remote_uploads"] is False
    assert "elderly" in str(result["summary"]).casefold()


def test_gateway_wikipedia_fallback_uses_offline_summary_without_network() -> None:
    client = TestClient(app)

    response = client.post(
        "/tasks/summarize_wikipedia",
        json={
            "text": (
                "Local-first assistive technology keeps private files local. "
                "It helps low-vision users read documents. "
                "Financial answers need human review."
            )
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["local_only"] is True
    assert payload["status"] == "stub"
    assert payload["result"]["remote_uploads"] is False
    assert payload["result"]["source"] == "provided_text"
    assert "Local-first assistive technology" in payload["result"]["summary"]

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.schemas.source_registry import read_jsonl, write_jsonl  # noqa: E402
from scripts.registry_common import APPROVED_PATH  # noqa: E402
from scripts.synthetic_documents import generate_documents  # noqa: E402

RAW_ROOT = ROOT / "data" / "raw" / "datasets"
PROCESSED_ROOT = ROOT / "data" / "processed" / "datasets"
REPORTS_DIR = ROOT / "reports"
CHECKPOINT_PATH = ROOT / "data" / "registry" / "small_dataset_ingestion_status.json"

DOCUMENT_TARGETS = {
    "huggingface:ryanznie/SROIE_2019_with_labels",
    "manual:FATURA",
    "manual:XFUND",
    "manual:CORD",
    "synthetic:local-lm-regional-documents",
}
OPTIONAL_EVAL_TARGETS = {
    "huggingface:google/fleurs",
    "uci:online-retail",
}
DEFAULT_TARGETS = DOCUMENT_TARGETS | OPTIONAL_EVAL_TARGETS
MANUAL_TARGETS = {"manual:FATURA", "manual:XFUND", "manual:CORD"}
LARGE_DATASET_BYTES = 1_000_000_000
DEFAULT_MAX_DATASET_BYTES = 10 * 1024 * 1024 * 1024
UCI_ONLINE_RETAIL_URL = (
    "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
)
FATURA_ZENODO_RECORD_URL = "https://zenodo.org/api/records/8261508"
XFUND_RELEASE_URL = "https://api.github.com/repos/doc-analysis/XFUND/releases/tags/v1.0"
CORD_HF_REPO_ID = "naver-clova-ix/cord-v2"


def slugify(value: str) -> str:
    cleaned = []
    for char in value.lower():
        cleaned.append(char if char.isalnum() else "-")
    return "-".join("".join(cleaned).split("-")).strip("-")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def assert_dataset_size_cap(
    dataset: dict[str, Any],
    raw_dir: Path,
    processed_dir: Path,
    max_dataset_size_bytes: int,
) -> None:
    declared_size = dataset.get("size_bytes")
    if isinstance(declared_size, int) and declared_size > max_dataset_size_bytes:
        raise ValueError(
            f"{dataset['dataset_id']} declares {declared_size} bytes, "
            f"above cap {max_dataset_size_bytes}"
        )
    local_size = directory_size(raw_dir) + directory_size(processed_dir)
    if local_size > max_dataset_size_bytes:
        raise ValueError(
            f"{dataset['dataset_id']} local files are {local_size} bytes, "
            f"above cap {max_dataset_size_bytes}"
        )


def load_checkpoint(path: Path = CHECKPOINT_PATH) -> dict[str, Any]:
    if not path.exists():
        return {"datasets": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("datasets"), dict):
        raise ValueError(f"invalid checkpoint format: {path}")
    return payload


def save_checkpoint(payload: dict[str, Any], path: Path = CHECKPOINT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def update_checkpoint(
    checkpoint: dict[str, Any],
    dataset_id: str,
    *,
    checkpoint_path: Path = CHECKPOINT_PATH,
    status: str,
    raw_path: Path | None = None,
    processed_path: Path | None = None,
    notes: str = "",
    error: str = "",
) -> None:
    checkpoint["datasets"][dataset_id] = {
        "status": status,
        "raw_path": str(raw_path) if raw_path else None,
        "processed_path": str(processed_path) if processed_path else None,
        "notes": notes,
        "error": error,
        "updated_at": int(time.time()),
    }
    save_checkpoint(checkpoint, checkpoint_path)


def approved_small_targets(
    approved_path: Path,
    requested: set[str] | None = None,
) -> list[dict[str, Any]]:
    approved = read_jsonl(approved_path)
    selected_ids = requested or DEFAULT_TARGETS
    records = []
    for record in approved:
        dataset_id = str(record["dataset_id"])
        if dataset_id not in selected_ids:
            continue
        size_bytes = record.get("size_bytes")
        if isinstance(size_bytes, int) and size_bytes >= LARGE_DATASET_BYTES:
            continue
        records.append(record)
    return records


def download_with_resume(url: str, destination: Path, timeout_seconds: int = 60) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")
    existing_size = temp_path.stat().st_size if temp_path.exists() else 0
    headers = {"Range": f"bytes={existing_size}-"} if existing_size else {}
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            mode = "ab" if existing_size and response.status == 206 else "wb"
            with temp_path.open(mode) as output:
                shutil.copyfileobj(response, output)
    except urllib.error.HTTPError as exc:
        if existing_size and exc.code == 416:
            temp_path.replace(destination)
            return destination
        raise
    temp_path.replace(destination)
    return destination


def download_with_size_cap(
    url: str,
    destination: Path,
    *,
    max_dataset_size_bytes: int,
    expected_size: int | None = None,
) -> Path:
    if expected_size is not None and expected_size > max_dataset_size_bytes:
        raise ValueError(
            f"{url} declares {expected_size} bytes, above cap {max_dataset_size_bytes}"
        )
    if destination.exists() and (
        expected_size is None or destination.stat().st_size == expected_size
    ):
        return destination
    path = download_with_resume(url, destination)
    if path.stat().st_size > max_dataset_size_bytes:
        raise ValueError(
            f"{destination} is {path.stat().st_size} bytes, above cap {max_dataset_size_bytes}"
        )
    return path


def fetch_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object from {url}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_synthetic(raw_dir: Path, processed_dir: Path, count_per_kind: int) -> dict[str, Any]:
    for kind in ("invoice", "receipt", "bank_statement", "handwritten_note"):
        generate_documents(
            kind=kind,  # type: ignore[arg-type]
            output_dir=raw_dir,
            count=count_per_kind,
            seed=20260613,
            augment=False,
        )
    metadata_path = raw_dir / "metadata.jsonl"
    records = read_jsonl(metadata_path)
    processed_dir.mkdir(parents=True, exist_ok=True)
    vision_rows = []
    for record in records:
        document_type = str(record["document_type"])
        document_id = str(record["document_id"])
        document_dir = raw_dir / document_type / document_id
        ground_truth_path = document_dir / "ground_truth.json"
        expected_output = json.loads(ground_truth_path.read_text(encoding="utf-8"))
        vision_rows.append(
            {
                "image_path": display_path(document_dir / "rendered.png"),
                "prompt": f"Extract structured fields from this {document_type}.",
                "expected_output": expected_output,
                "region": record["region"],
                "country": record["country"],
                "language": record["language"],
                "task": record["task"],
                "source_type": "synthetic",
                "license": record["license"],
                "pii_status": "synthetic",
                "modality": "image",
                "document_type": document_type,
                "human_review_required": document_type == "bank_statement",
            }
        )
    output = processed_dir / "vision_document_extraction.jsonl"
    write_jsonl(output, vision_rows)
    return {"rows": len(vision_rows), "outputs": [display_path(output)]}


def prepare_file_index(
    raw_dir: Path,
    processed_dir: Path,
    dataset: dict[str, Any],
) -> dict[str, Any]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    indexed_files = []
    for path in sorted(raw_dir.rglob("*")):
        relative_parts = path.relative_to(raw_dir).parts
        if path.is_file() and not any(part.startswith(".") for part in relative_parts):
            indexed_files.append(
                {
                    "dataset_id": dataset["dataset_id"],
                    "relative_path": str(path.relative_to(raw_dir)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "regions": dataset["regions"],
                    "countries": dataset["countries"],
                    "languages": dataset["languages"],
                    "tasks": dataset["candidate_tasks"],
                    "license": dataset["license_name"],
                }
            )
    output = processed_dir / "file_index.jsonl"
    write_jsonl(output, indexed_files)
    return {"rows": len(indexed_files), "outputs": [display_path(output)]}


def prepare_sroie_receipt_extraction(raw_dir: Path, processed_dir: Path) -> dict[str, Any]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for split in ("train", "test"):
        entities_dir = raw_dir / split / "entities"
        image_dir = raw_dir / split / "img"
        if not entities_dir.exists() or not image_dir.exists():
            continue
        for entity_path in sorted(entities_dir.glob("*.txt")):
            image_path = _matching_image(image_dir, entity_path.stem)
            if image_path is None:
                continue
            try:
                expected_output = json.loads(entity_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            rows.append(
                {
                    "image_path": display_path(image_path),
                    "prompt": "Extract merchant, date, address, and total from this receipt.",
                    "expected_output": expected_output,
                    "region": "Southeast Asia",
                    "country": "Malaysia",
                    "language": "en",
                    "task": "receipt_extraction",
                    "source_type": "huggingface",
                    "source_dataset": "SROIE_2019_with_labels",
                    "license": "CC-BY-4.0",
                    "pii_status": "redacted",
                    "modality": "image",
                    "document_type": "receipt",
                    "split_usage": "test" if split == "test" else "train",
                    "human_review_required": False,
                }
            )
    output = processed_dir / "receipt_extraction.jsonl"
    write_jsonl(output, rows)
    return {"rows": len(rows), "outputs": [display_path(output)]}


def prepare_xfund_form_understanding(raw_dir: Path, processed_dir: Path) -> dict[str, Any]:
    language_country = {
        "de": ("Europe", "Germany"),
        "es": ("Europe", "Spain"),
        "fr": ("Europe", "France"),
        "it": ("Europe", "Italy"),
        "pt": ("Europe", "Portugal"),
    }
    rows: list[dict[str, Any]] = []
    for json_path in sorted(raw_dir.glob("*.json")):
        parts = json_path.name.split(".")
        if len(parts) < 3:
            continue
        language, split = parts[0], parts[1]
        if language not in language_country:
            continue
        region, country = language_country[language]
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        documents = payload.get("documents", [])
        if not isinstance(documents, list):
            continue
        for document in documents:
            if not isinstance(document, dict):
                continue
            document_id = str(document.get("id", "")).strip()
            image_path = raw_dir / "extracted" / f"{language}.{split}" / f"{document_id}.jpg"
            if not image_path.exists():
                continue
            rows.append(
                {
                    "image_path": display_path(image_path),
                    "prompt": "Extract key-value fields and layout labels from this form.",
                    "expected_output": {
                        "document_id": document_id,
                        "language": language,
                        "entities": document.get("document", []),
                    },
                    "region": region,
                    "country": country,
                    "language": language,
                    "task": "document_ocr",
                    "source_type": "manual",
                    "source_dataset": "XFUND",
                    "license": "MIT",
                    "pii_status": "redacted",
                    "modality": "image",
                    "document_type": "form",
                    "split_usage": "validation" if split == "val" else "train",
                    "human_review_required": False,
                }
            )
    output = processed_dir / "form_understanding.jsonl"
    write_jsonl(output, rows)
    return {"rows": len(rows), "outputs": [display_path(output)]}


def prepare_fatura_invoice_extraction(raw_dir: Path, processed_dir: Path) -> dict[str, Any]:
    base = raw_dir / "extracted" / "FATURA" / "invoices_dataset_final"
    split_files = {
        "train": base / "strat1_train.csv",
        "validation": base / "strat1_dev.csv",
        "test": base / "strat1_test.csv",
    }
    rows: list[dict[str, Any]] = []
    for split, csv_path in split_files.items():
        if not csv_path.exists():
            continue
        for item in _read_csv_rows(csv_path):
            image_path = base / "images" / str(item.get("img_path", ""))
            annot_path = base / "Annotations" / "Original_Format" / str(
                item.get("annot_path", "")
            )
            if not image_path.exists() or not annot_path.exists():
                continue
            expected_output = json.loads(annot_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "image_path": display_path(image_path),
                    "prompt": "Extract invoice fields and table regions from this invoice.",
                    "expected_output": expected_output,
                    "region": "Europe",
                    "country": "Turkey",
                    "language": "tr",
                    "task": "invoice_extraction",
                    "source_type": "manual",
                    "source_dataset": "FATURA",
                    "license": "MIT",
                    "pii_status": "synthetic",
                    "modality": "image",
                    "document_type": "invoice",
                    "split_usage": split,
                    "human_review_required": False,
                }
            )
    output = processed_dir / "invoice_extraction.jsonl"
    write_jsonl(output, rows)
    return {"rows": len(rows), "outputs": [display_path(output)]}


def _matching_image(image_dir: Path, stem: str) -> Path | None:
    for suffix in (".jpg", ".jpeg", ".png"):
        path = image_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    return None


def _merge_prepare_results(*results: dict[str, Any]) -> dict[str, Any]:
    outputs: list[str] = []
    rows = 0
    notes: list[str] = []
    for result in results:
        rows += int(result.get("rows", 0))
        outputs.extend(str(output) for output in result.get("outputs", []))
        if result.get("notes"):
            notes.append(str(result["notes"]))
    merged: dict[str, Any] = {"rows": rows, "outputs": outputs}
    if notes:
        merged["notes"] = " ".join(notes)
    return merged


def prepare_uci_online_retail(raw_dir: Path, processed_dir: Path, limit: int) -> dict[str, Any]:
    zip_path = raw_dir / "online-retail.zip"
    extract_dir = raw_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)
    candidates = list(extract_dir.rglob("*.xlsx")) + list(extract_dir.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError("UCI Online Retail archive did not contain CSV/XLSX files")
    source = candidates[0]
    rows = _read_tabular_sample(source, limit)
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for row in rows:
        output_rows.append(
            {
                "dataset_id": "uci:online-retail",
                "task": "tabular_reasoning",
                "split_usage": "eval_only",
                "region": "Europe",
                "country": "United Kingdom",
                "language": "en",
                "license": "CC-BY-4.0",
                "pii_status": "low",
                "input": row,
                "expected_output": {
                    "instruction": "Answer questions or repair JSON using this retail row.",
                    "fields": sorted(row),
                },
            }
        )
    output = processed_dir / "tabular_reasoning_eval.jsonl"
    write_jsonl(output, output_rows)
    return {"rows": len(output_rows), "outputs": [display_path(output)]}


def prepare_zip_file_index(
    raw_dir: Path,
    processed_dir: Path,
    dataset: dict[str, Any],
    zip_paths: Iterable[Path],
) -> dict[str, Any]:
    extract_dir = raw_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extract_dir / zip_path.stem)
    return prepare_file_index(raw_dir, processed_dir, dataset)


def download_prepare_fatura(
    dataset: dict[str, Any],
    raw_dir: Path,
    processed_dir: Path,
    max_dataset_size_bytes: int,
) -> dict[str, Any]:
    record = fetch_json(FATURA_ZENODO_RECORD_URL)
    files = [item for item in record.get("files", []) if item.get("key") == "FATURA.zip"]
    if not files:
        raise FileNotFoundError("FATURA.zip not found in Zenodo record 8261508")
    file_info = files[0]
    url = str(file_info["links"]["self"])
    zip_path = download_with_size_cap(
        url,
        raw_dir / "FATURA.zip",
        max_dataset_size_bytes=max_dataset_size_bytes,
        expected_size=int(file_info["size"]),
    )
    return _merge_prepare_results(
        prepare_zip_file_index(raw_dir, processed_dir, dataset, [zip_path]),
        prepare_fatura_invoice_extraction(raw_dir, processed_dir),
    )


def download_prepare_xfund(
    dataset: dict[str, Any],
    raw_dir: Path,
    processed_dir: Path,
    max_dataset_size_bytes: int,
) -> dict[str, Any]:
    release = fetch_json(XFUND_RELEASE_URL)
    assets = release.get("assets", [])
    if not isinstance(assets, list) or not assets:
        raise FileNotFoundError("XFUND v1.0 release assets not found")
    total_size = sum(int(asset.get("size", 0)) for asset in assets if isinstance(asset, dict))
    if total_size > max_dataset_size_bytes:
        raise ValueError(f"XFUND release declares {total_size} bytes, above cap")
    downloaded: list[Path] = []
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        name = str(asset["name"])
        url = str(asset["browser_download_url"])
        path = download_with_size_cap(
            url,
            raw_dir / name,
            max_dataset_size_bytes=max_dataset_size_bytes,
            expected_size=int(asset.get("size", 0)),
        )
        if path.suffix.lower() == ".zip":
            downloaded.append(path)
    return _merge_prepare_results(
        prepare_zip_file_index(raw_dir, processed_dir, dataset, downloaded),
        prepare_xfund_form_understanding(raw_dir, processed_dir),
    )


def _read_tabular_sample(path: Path, limit: int) -> list[dict[str, Any]]:
    import pandas as pd

    if path.suffix.lower() == ".xlsx":
        frame = pd.read_excel(path, nrows=limit)
    else:
        frame = pd.read_csv(path, nrows=limit)
    frame = frame.where(pd.notnull(frame), None)
    return [
        {str(key): _json_safe(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _json_safe(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def write_manual_collection_manifest(
    dataset: dict[str, Any],
    raw_dir: Path,
    processed_dir: Path,
) -> dict[str, Any]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_id": dataset["dataset_id"],
        "dataset_name": dataset["dataset_name"],
        "source_url": dataset["source_url"],
        "status": "manual_collection_required",
        "reason": (
            "No safe direct artifact URL is recorded. Collect from the source, "
            "preserve upstream license metadata, then rerun preparation."
        ),
        "expected_raw_dir": display_path(raw_dir),
        "expected_processed_dir": display_path(processed_dir),
        "regions": dataset["regions"],
        "languages": dataset["languages"],
        "tasks": dataset["candidate_tasks"],
        "license": dataset["license_name"],
    }
    output = raw_dir / "MANUAL_COLLECTION.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {"rows": 0, "outputs": [display_path(output)]}


def download_huggingface_snapshot(
    dataset: dict[str, Any],
    raw_dir: Path,
    cache_dir: Path,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir / "hf_home"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hub"))
    os.environ.setdefault("HF_XET_CACHE", str(cache_dir / "xet"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    from huggingface_hub import snapshot_download

    repo_id = str(dataset.get("metadata", {}).get("hf_repo_id") or "").strip()
    if not repo_id and dataset["dataset_id"] == "manual:CORD":
        repo_id = CORD_HF_REPO_ID
    if not repo_id:
        raise ValueError(f"missing hf_repo_id for {dataset['dataset_id']}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=raw_dir,
        cache_dir=cache_dir,
        resume_download=True,
        allow_patterns=[
            "*.json",
            "*.jsonl",
            "*.csv",
            "*.tsv",
            "*.txt",
            "*.parquet",
            "*.jpg",
            "*.jpeg",
            "*.png",
            "*.pdf",
        ],
    )
    return raw_dir


def ingest_dataset(
    dataset: dict[str, Any],
    *,
    checkpoint: dict[str, Any],
    raw_root: Path,
    processed_root: Path,
    checkpoint_path: Path,
    no_network: bool,
    tabular_limit: int,
    synthetic_count_per_kind: int,
    hf_existing_file_threshold: int,
    max_dataset_size_bytes: int,
) -> dict[str, Any]:
    dataset_id = str(dataset["dataset_id"])
    slug = slugify(dataset_id)
    raw_dir = raw_root / slug
    processed_dir = processed_root / slug
    update_checkpoint(
        checkpoint,
        dataset_id,
        checkpoint_path=checkpoint_path,
        status="started",
        raw_path=raw_dir,
    )
    try:
        assert_dataset_size_cap(dataset, raw_dir, processed_dir, max_dataset_size_bytes)
        if dataset_id == "synthetic:local-lm-regional-documents":
            result = prepare_synthetic(raw_dir, processed_dir, synthetic_count_per_kind)
        elif dataset_id == "uci:online-retail":
            existing_zip = raw_dir / "online-retail.zip"
            if no_network and not existing_zip.exists():
                raise RuntimeError("network disabled for UCI Online Retail download")
            zip_path = (
                existing_zip
                if existing_zip.exists()
                else download_with_resume(UCI_ONLINE_RETAIL_URL, existing_zip)
            )
            result = prepare_uci_online_retail(raw_dir, processed_dir, tabular_limit)
            result["downloaded"] = display_path(zip_path)
        elif dataset_id == "huggingface:ryanznie/SROIE_2019_with_labels":
            existing_files = [path for path in raw_dir.rglob("*") if path.is_file()]
            if len(existing_files) >= hf_existing_file_threshold:
                result = _merge_prepare_results(
                    prepare_file_index(raw_dir, processed_dir, dataset),
                    prepare_sroie_receipt_extraction(raw_dir, processed_dir),
                )
                result["notes"] = (
                    "Prepared from existing local Hugging Face files; "
                    "rerun with a token for a full refresh if needed."
                )
                update_checkpoint(
                    checkpoint,
                    dataset_id,
                    checkpoint_path=checkpoint_path,
                    status="prepared_partial",
                    raw_path=raw_dir,
                    processed_path=processed_dir,
                    notes=str(result),
                )
                return {"dataset_id": dataset_id, "status": "prepared_partial", **result}
            if no_network:
                raise RuntimeError("network disabled for Hugging Face SROIE download")
            download_huggingface_snapshot(dataset, raw_dir, raw_root / ".hf_cache")
            result = _merge_prepare_results(
                prepare_file_index(raw_dir, processed_dir, dataset),
                prepare_sroie_receipt_extraction(raw_dir, processed_dir),
            )
        elif dataset_id == "manual:CORD":
            if no_network:
                raise RuntimeError("network disabled for CORD download")
            download_huggingface_snapshot(dataset, raw_dir, raw_root / ".hf_cache")
            result = prepare_file_index(raw_dir, processed_dir, dataset)
        elif dataset_id == "manual:FATURA":
            if no_network:
                raise RuntimeError("network disabled for FATURA download")
            result = download_prepare_fatura(
                dataset,
                raw_dir,
                processed_dir,
                max_dataset_size_bytes,
            )
        elif dataset_id == "manual:XFUND":
            if no_network:
                raise RuntimeError("network disabled for XFUND download")
            result = download_prepare_xfund(
                dataset,
                raw_dir,
                processed_dir,
                max_dataset_size_bytes,
            )
        elif dataset_id == "huggingface:google/fleurs":
            result = write_manual_collection_manifest(dataset, raw_dir, processed_dir)
            result["notes"] = (
                "FLEURS eval subset requires datasets tooling or explicit file selection."
            )
        elif dataset_id in MANUAL_TARGETS:
            result = write_manual_collection_manifest(dataset, raw_dir, processed_dir)
        else:
            result = {"rows": 0, "outputs": [], "notes": "not implemented for this target"}
        assert_dataset_size_cap(dataset, raw_dir, processed_dir, max_dataset_size_bytes)
        update_checkpoint(
            checkpoint,
            dataset_id,
            checkpoint_path=checkpoint_path,
            status="prepared",
            raw_path=raw_dir,
            processed_path=processed_dir,
            notes=str(result),
        )
        return {"dataset_id": dataset_id, "status": "prepared", **result}
    except Exception as exc:
        update_checkpoint(
            checkpoint,
            dataset_id,
            checkpoint_path=checkpoint_path,
            status="failed",
            raw_path=raw_dir,
            processed_path=processed_dir,
            error=str(exc),
        )
        return {"dataset_id": dataset_id, "status": "failed", "error": str(exc)}


def write_report(results: Iterable[dict[str, Any]], reports_dir: Path) -> dict[str, Any]:
    result_list = list(results)
    report = {
        "prepared": sum(1 for item in result_list if str(item["status"]).startswith("prepared")),
        "failed": sum(1 for item in result_list if item["status"] == "failed"),
        "results": result_list,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "small_dataset_ingestion.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# Small Dataset Ingestion",
        "",
        f"- prepared: {report['prepared']}",
        f"- failed: {report['failed']}",
        "",
    ]
    for item in result_list:
        lines.append(f"## {item['dataset_id']}")
        lines.append("")
        lines.append(f"- status: {item['status']}")
        if item.get("rows") is not None:
            lines.append(f"- rows: {item['rows']}")
        if item.get("outputs"):
            lines.append(f"- outputs: {', '.join(item['outputs'])}")
        if item.get("error"):
            lines.append(f"- error: {item['error']}")
        lines.append("")
    (reports_dir / "small_dataset_ingestion.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    return report


def parse_targets(value: str) -> set[str] | None:
    if not value.strip():
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approved", type=Path, default=APPROVED_PATH)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--targets", default="")
    parser.add_argument("--no-network", action="store_true")
    parser.add_argument("--tabular-limit", type=int, default=500)
    parser.add_argument("--synthetic-count-per-kind", type=int, default=8)
    parser.add_argument("--hf-existing-file-threshold", type=int, default=200)
    parser.add_argument("--max-dataset-size-gb", type=float, default=10.0)
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    targets = approved_small_targets(args.approved, parse_targets(args.targets))
    results = [
        ingest_dataset(
            dataset,
            checkpoint=checkpoint,
            raw_root=args.raw_root,
            processed_root=args.processed_root,
            checkpoint_path=args.checkpoint,
            no_network=args.no_network,
            tabular_limit=args.tabular_limit,
            synthetic_count_per_kind=args.synthetic_count_per_kind,
            hf_existing_file_threshold=args.hf_existing_file_threshold,
            max_dataset_size_bytes=int(args.max_dataset_size_gb * 1024 * 1024 * 1024),
        )
        for dataset in targets
    ]
    report = write_report(results, args.reports_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

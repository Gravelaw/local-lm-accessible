from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ADAPTER_REQUIRED_FILES = ("adapter_config.json",)


def finalize_text_adapter(
    *,
    adapter_dir: Path,
    base_model: str,
    train_file: Path,
    eval_file: Path,
    report_json: Path,
    report_markdown: Path,
    run_name: str = "nemotron_modal_prepared_lora",
) -> dict[str, Any]:
    if not adapter_dir.exists():
        raise FileNotFoundError(f"missing adapter directory: {adapter_dir}")
    missing = [name for name in ADAPTER_REQUIRED_FILES if not (adapter_dir / name).exists()]
    if missing:
        raise ValueError(f"adapter directory is missing required files: {missing}")
    files = _artifact_files(adapter_dir)
    if not files:
        raise ValueError(f"adapter directory has no files: {adapter_dir}")

    file_checksums = [
        {
            "path": str(path),
            "relative_path": str(path.relative_to(adapter_dir)),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in files
    ]
    manifest = {
        "artifact_type": "lora_adapter",
        "run_name": run_name,
        "base_model": base_model,
        "adapter_dir": str(adapter_dir),
        "train_file": str(train_file),
        "eval_file": str(eval_file),
        "created_at": datetime.now(UTC).isoformat(),
        "sha256": sha256_directory(adapter_dir),
        "files": file_checksums,
        "final_artifact": "lora_adapter",
        "merge_or_gguf_packaging": "deferred_until_adapter_eval_passes",
    }
    manifest_path = adapter_dir / "adapter_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (adapter_dir / "adapter_manifest.json.sha256").write_text(
        sha256_file(manifest_path) + "  adapter_manifest.json\n",
        encoding="utf-8",
    )
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_markdown.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_markdown.write_text(_markdown_summary(manifest), encoding="utf-8")
    return manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as item:
        for chunk in iter(lambda: item.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: Path) -> str:
    digest = hashlib.sha256()
    for child in _artifact_files(path):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(sha256_file(child).encode("utf-8"))
    return digest.hexdigest()


def _artifact_files(path: Path) -> list[Path]:
    return sorted(
        child
        for child in path.rglob("*")
        if child.is_file() and not child.name.endswith(".sha256")
        and child.name != "adapter_manifest.json"
    )


def _markdown_summary(manifest: dict[str, Any]) -> str:
    lines = [
        "# Final Text Adapter Summary",
        "",
        f"- artifact_type: {manifest['artifact_type']}",
        f"- run_name: {manifest['run_name']}",
        f"- base_model: {manifest['base_model']}",
        f"- adapter_dir: {manifest['adapter_dir']}",
        f"- sha256: {manifest['sha256']}",
        f"- files: {len(manifest['files'])}",
        f"- final_artifact: {manifest['final_artifact']}",
        f"- merge_or_gguf_packaging: {manifest['merge_or_gguf_packaging']}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    parser.add_argument("--run-name", default="nemotron_modal_prepared_lora")
    args = parser.parse_args()
    manifest = finalize_text_adapter(
        adapter_dir=args.adapter_dir,
        base_model=args.base_model,
        train_file=args.train_file,
        eval_file=args.eval_file,
        report_json=args.report_json,
        report_markdown=args.report_md,
        run_name=args.run_name,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

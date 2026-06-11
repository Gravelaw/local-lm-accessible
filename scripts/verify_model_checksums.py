from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "models" / "manifest.json"
REQUIRED_FIELDS = {
    "model_name",
    "model_id",
    "vendor",
    "base_params",
    "adapter_params",
    "quantization",
    "local_path",
    "sha256",
    "license",
    "runtime",
    "port",
    "tasks",
    "supported_languages",
    "unsupported_languages",
    "commercial_status",
    "notes",
}


def sha256_path(path: Path) -> str:
    if path.is_dir():
        return _sha256_directory(path)
    return _sha256_file(path)


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    allowed_vendors = set(manifest.get("allowed_vendors", []))
    for model in manifest.get("models", []):
        missing = sorted(REQUIRED_FIELDS - set(model))
        if missing:
            model_key = model.get("key", "<unknown>")
            raise ValueError(f"manifest model {model_key} missing fields: {missing}")
        if model["vendor"] not in allowed_vendors:
            raise ValueError(f"disallowed vendor for {model['key']}: {model['vendor']}")
        endpoint_port = int(model["port"])
        if endpoint_port not in {8081, 8082, 8083, 8090}:
            raise ValueError(f"unexpected local port for {model['key']}: {endpoint_port}")
    return manifest


def verify_model(model: dict[str, Any]) -> dict[str, Any]:
    expected = str(model.get("sha256", "")).strip()
    key = str(model["key"])
    if not expected:
        raise ValueError(f"missing sha256 for model {key}; refusing to start")
    computed = compute_model_artifact_checksums(model)
    actual = str(computed["sha256"])
    if actual != expected:
        raise ValueError(f"checksum mismatch for model {key}: expected {expected}, got {actual}")
    verified_artifacts = []
    computed_artifacts = {
        str(artifact["name"]): artifact for artifact in computed.get("additional_artifacts", [])
    }
    for artifact in model.get("additional_artifacts", []):
        artifact_name = str(artifact.get("name", "<unnamed>"))
        artifact_expected = str(artifact.get("sha256", "")).strip()
        if not artifact_expected:
            raise ValueError(f"missing sha256 for model {key} artifact {artifact_name}")
        artifact_actual = str(computed_artifacts[artifact_name]["sha256"])
        if artifact_actual != artifact_expected:
            raise ValueError(
                "checksum mismatch for model "
                f"{key} artifact {artifact_name}: expected {artifact_expected}, "
                f"got {artifact_actual}"
            )
        verified_artifacts.append(
            {
                "name": artifact_name,
                "path": str(computed_artifacts[artifact_name]["path"]),
                "sha256": artifact_actual,
                "verified": True,
            }
        )
    return {
        "key": key,
        "path": str(computed["path"]),
        "sha256": actual,
        "verified": True,
        "additional_artifacts": verified_artifacts,
    }


def compute_model_artifact_checksums(
    model: dict[str, Any],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    key = str(model["key"])
    path = _resolve_local_path(str(model["local_path"]), root=root)
    if not path.exists():
        raise FileNotFoundError(f"missing local artifact for model {key}: {path}")
    actual = sha256_path(path)
    computed_artifacts = []
    for artifact in model.get("additional_artifacts", []):
        artifact_name = str(artifact.get("name", "<unnamed>"))
        artifact_path = _resolve_local_path(str(artifact["local_path"]), root=root)
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"missing local artifact for model {key} artifact {artifact_name}: {artifact_path}"
            )
        computed_artifacts.append(
            {
                "name": artifact_name,
                "path": str(artifact_path),
                "sha256": sha256_path(artifact_path),
            }
        )
    return {
        "key": key,
        "path": str(path),
        "sha256": actual,
        "additional_artifacts": computed_artifacts,
    }


def update_manifest_checksum(
    model_key: str,
    *,
    manifest_path: Path = MANIFEST_PATH,
    root: Path = ROOT,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    for model in manifest["models"]:
        if model["key"] != model_key:
            continue
        computed = compute_model_artifact_checksums(model, root=root)
        model["sha256"] = computed["sha256"]
        computed_artifacts = {
            str(artifact["name"]): artifact for artifact in computed.get("additional_artifacts", [])
        }
        for artifact in model.get("additional_artifacts", []):
            artifact_name = str(artifact.get("name", "<unnamed>"))
            artifact["sha256"] = computed_artifacts[artifact_name]["sha256"]
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return computed
    raise KeyError(f"model not found in manifest: {model_key}")


def verify_manifest(model_key: str | None = None) -> list[dict[str, Any]]:
    manifest = load_manifest()
    results = []
    for model in manifest["models"]:
        if model_key is not None and model["key"] != model_key:
            continue
        results.append(verify_model(model))
    if model_key is not None and not results:
        raise KeyError(f"model not found in manifest: {model_key}")
    return results


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_directory(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"directory artifact is empty: {path}")
    for child in files:
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(_sha256_file(child).encode("utf-8"))
    return digest.hexdigest()


def _resolve_local_path(path_value: str, *, root: Path = ROOT) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return root / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Verify one manifest key, e.g. text, vision, asr.")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument(
        "--print-actual",
        action="store_true",
        help="Compute the current local artifact checksum without comparing manifest sha256.",
    )
    parser.add_argument(
        "--write-manifest-checksum",
        action="store_true",
        help="Compute and write the current local artifact checksum into the manifest.",
    )
    args = parser.parse_args()

    manifest = load_manifest()
    if args.manifest_only:
        print(json.dumps({"status": "ok", "models": len(manifest["models"])}, sort_keys=True))
        return
    if args.print_actual or args.write_manifest_checksum:
        if not args.model:
            raise ValueError("--model is required when computing or writing actual checksums")
        models = {str(model["key"]): model for model in manifest["models"]}
        if args.model not in models:
            raise KeyError(f"model not found in manifest: {args.model}")
        if args.write_manifest_checksum:
            result = update_manifest_checksum(args.model)
            print(json.dumps({"status": "ok", "updated": result}, indent=2, sort_keys=True))
            return
        result = compute_model_artifact_checksums(models[args.model])
        print(json.dumps({"status": "ok", "computed": result}, indent=2, sort_keys=True))
        return
    results = verify_manifest(args.model)
    print(json.dumps({"status": "ok", "verified": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

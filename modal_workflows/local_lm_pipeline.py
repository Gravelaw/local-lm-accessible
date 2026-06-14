from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import modal

APP_NAME = "local-lm-data-finetune"
REMOTE_ROOT = Path("/workspace/local-lm")
VOLUME_ROOT = Path("/vol/local-lm")
RAW_ROOT = VOLUME_ROOT / "data" / "raw" / "datasets"
PROCESSED_ROOT = VOLUME_ROOT / "data" / "processed" / "datasets"
TRAINING_MANIFEST_ROOT = VOLUME_ROOT / "data" / "processed" / "training"
REPORTS_DIR = VOLUME_ROOT / "reports"
CHECKPOINT_PATH = VOLUME_ROOT / "data" / "registry" / "small_dataset_ingestion_status.json"
REGISTRY_ROOT = VOLUME_ROOT / "data" / "registry"
SPLITS_ROOT = VOLUME_ROOT / "data" / "splits"
APPROVED_PATH = REGISTRY_ROOT / "approved_datasets.jsonl"
RESEARCH_EVAL_PATH = REGISTRY_ROOT / "research_eval_datasets.jsonl"
REJECTED_PATH = REGISTRY_ROOT / "rejected_datasets.jsonl"
TASK_MAPPED_PATH = REGISTRY_ROOT / "task_mapped_datasets.jsonl"
TEXT_MODAL_CONFIG = (
    REMOTE_ROOT / "training" / "text" / "configs" / "nemotron_modal_prepared_lora.yaml"
)
VISION_MODAL_CONFIG = (
    REMOTE_ROOT / "training" / "vision" / "configs" / "minicpm_v_modal_document_lora.yaml"
)
TEXT_PREFLIGHT_REPORT = REPORTS_DIR / "fine_tuning_text_preflight.json"
VISION_PREFLIGHT_REPORT = REPORTS_DIR / "fine_tuning_vision_preflight.json"

DATA_VOLUME = modal.Volume.from_name("local-lm-data", create_if_missing=True)
CACHE_VOLUME = modal.Volume.from_name("local-lm-cache", create_if_missing=True)
HF_SECRET = modal.Secret.from_name("huggingface-secret")

app = modal.App(APP_NAME)

base_data_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "git", "libglib2.0-0", "libgl1")
    .pip_install(
        "babel>=2.14",
        "faker>=25.0",
        "huggingface_hub>=0.26",
        "jinja2>=3.1",
        "openpyxl>=3.1",
        "opencv-python-headless>=4.9",
        "pandas>=2.2",
        "pillow>=10.0",
        "pydantic>=2.8",
        "pyarrow>=16.0",
        "pyyaml>=6.0",
        "reportlab>=4.0",
        "requests>=2.32",
    )
)

base_training_image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.2-devel-ubuntu24.04",
        add_python="3.11",
    )
    .apt_install("build-essential", "curl", "git", "libglib2.0-0", "libgl1", "ninja-build")
    .pip_install(
        "babel>=2.14",
        "faker>=25.0",
        "huggingface_hub>=0.26",
        "jinja2>=3.1",
        "openpyxl>=3.1",
        "opencv-python-headless>=4.9",
        "pandas>=2.2",
        "pillow>=10.0",
        "pydantic>=2.8",
        "pyarrow>=16.0",
        "pyyaml>=6.0",
        "reportlab>=4.0",
        "requests>=2.32",
    )
)


def _with_repo_files(image: modal.Image) -> modal.Image:
    return (
        image
        .add_local_file("app.py", remote_path=str(REMOTE_ROOT / "app.py"))
        .add_local_dir("configs", remote_path=str(REMOTE_ROOT / "configs"))
        .add_local_file(
            "data/__init__.py",
            remote_path=str(REMOTE_ROOT / "data" / "__init__.py"),
        )
        .add_local_dir("data/registry", remote_path=str(REMOTE_ROOT / "data" / "registry"))
        .add_local_dir("data/schemas", remote_path=str(REMOTE_ROOT / "data" / "schemas"))
        .add_local_dir("scripts", remote_path=str(REMOTE_ROOT / "scripts"))
        .add_local_dir("services", remote_path=str(REMOTE_ROOT / "services"))
        .add_local_dir("training", remote_path=str(REMOTE_ROOT / "training"))
    )


data_image = _with_repo_files(base_data_image)

training_dependencies_image = (
    base_training_image
    .pip_install("torch==2.12.0", "packaging>=24.0", "setuptools>=69,<82", "wheel>=0.43")
    .pip_install(
        "accelerate>=0.34",
        "bitsandbytes>=0.43",
        "datasets>=2.20",
        "peft>=0.12",
        "transformers>=4.46,<5",
        "trl>=0.11",
    )
)

training_image = _with_repo_files(training_dependencies_image)


def _run(
    command: list[str],
    *,
    cwd: Path = REMOTE_ROOT,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("HF_HOME", "/cache/huggingface")
    env.setdefault("HF_HUB_CACHE", "/cache/huggingface/hub")
    env.setdefault("HF_XET_CACHE", "/cache/huggingface/xet")
    env.setdefault("XDG_CACHE_HOME", "/cache/xdg")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(REMOTE_ROOT)
        if not existing_pythonpath
        else f"{REMOTE_ROOT}:{existing_pythonpath}"
    )
    site_lib = Path("/usr/local/lib/python3.11/site-packages")
    cuda_library_paths = [
        site_lib / "nvidia" / name / "lib"
        for name in (
            "cublas",
            "cuda_cupti",
            "cuda_nvrtc",
            "cuda_runtime",
            "cudnn",
            "cufft",
            "cufile",
            "curand",
            "cusolver",
            "cusparse",
            "nvjitlink",
            "nvshmem",
            "nvtx",
        )
    ]
    existing_ld_library_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(
        [
            *(str(path) for path in cuda_library_paths),
            "/usr/local/cuda/lib64",
            *(existing_ld_library_path.split(":") if existing_ld_library_path else []),
        ]
    )
    if extra_env:
        env.update(extra_env)
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )
    result = {
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "returncode": completed.returncode,
    }
    if completed.returncode != 0:
        raise RuntimeError(json.dumps(result, indent=2, sort_keys=True))
    return result


def _stage_registry_seed() -> None:
    source = REMOTE_ROOT / "data" / "registry"
    target = REGISTRY_ROOT
    target.mkdir(parents=True, exist_ok=True)
    for name in (
        "dataset_candidates.jsonl",
        "approved_datasets.jsonl",
        "research_eval_datasets.jsonl",
        "rejected_datasets.jsonl",
        "task_mapped_datasets.jsonl",
        "dataset_candidates.large_downloads.json",
    ):
        source_path = source / name
        target_path = target / name
        if source_path.exists() and not target_path.exists():
            shutil.copy2(source_path, target_path)


def _volume_args() -> list[str]:
    return [
        "--raw-root",
        str(RAW_ROOT),
        "--processed-root",
        str(PROCESSED_ROOT),
        "--reports-dir",
        str(REPORTS_DIR),
        "--checkpoint",
        str(CHECKPOINT_PATH),
    ]


def _ingestion_command(
    *,
    targets: str,
    max_dataset_size_gb: float,
    no_network: bool,
    synthetic_count_per_kind: int,
    tabular_limit: int,
) -> list[str]:
    command = [
        "python",
        "scripts/download_prepare_small_datasets.py",
        "--approved",
        str(APPROVED_PATH),
        *_volume_args(),
        "--max-dataset-size-gb",
        str(max_dataset_size_gb),
        "--synthetic-count-per-kind",
        str(synthetic_count_per_kind),
        "--tabular-limit",
        str(tabular_limit),
    ]
    if targets:
        command.extend(["--targets", targets])
    if no_network:
        command.append("--no-network")
    return command


def _batch_commands() -> list[list[str]]:
    return [
        [
            "python",
            "scripts/audit_source_registry.py",
            "--approved",
            str(APPROVED_PATH),
            "--research-eval",
            str(RESEARCH_EVAL_PATH),
            "--rejected",
            str(REJECTED_PATH),
            "--reports-dir",
            str(REPORTS_DIR),
        ],
        [
            "python",
            "scripts/map_datasets_to_tasks.py",
            "--approved",
            str(APPROVED_PATH),
            "--output",
            str(TASK_MAPPED_PATH),
            "--reports-dir",
            str(REPORTS_DIR),
        ],
        [
            "python",
            "scripts/build_training_mix.py",
            "--approved",
            str(APPROVED_PATH),
            "--output-dir",
            str(SPLITS_ROOT),
        ],
        [
            "python",
            "scripts/build_prepared_training_manifests.py",
            "--processed-root",
            str(PROCESSED_ROOT),
            "--output-dir",
            str(TRAINING_MANIFEST_ROOT),
            "--reports-dir",
            str(REPORTS_DIR),
        ],
        [
            "python",
            "scripts/check_regional_balance.py",
            "--approved",
            str(APPROVED_PATH),
            "--reports-dir",
            str(REPORTS_DIR),
        ],
        ["python", "scripts/verify_dataset_locality.py"],
    ]


def _commit_volumes() -> None:
    DATA_VOLUME.commit()
    CACHE_VOLUME.commit()


def _preflight_command(
    *,
    modality: str,
    config_path: Path,
    report_path: Path,
    require_images: bool = False,
) -> list[str]:
    command = [
        "python",
        "scripts/preflight_finetuning_manifests.py",
        "--modality",
        modality,
        "--config",
        str(config_path),
        "--report",
        str(report_path),
    ]
    if require_images:
        command.append("--require-images")
    return command


@app.function(
    image=data_image,
    volumes={str(VOLUME_ROOT): DATA_VOLUME, "/cache": CACHE_VOLUME},
    secrets=[HF_SECRET],
    timeout=60 * 60 * 6,
    retries=2,
)
def ingest_data(
    targets: str = "",
    max_dataset_size_gb: float = 10.0,
    no_network: bool = False,
    synthetic_count_per_kind: int = 8,
    tabular_limit: int = 500,
) -> dict[str, Any]:
    _stage_registry_seed()
    command = _ingestion_command(
        targets=targets,
        max_dataset_size_gb=max_dataset_size_gb,
        no_network=no_network,
        synthetic_count_per_kind=synthetic_count_per_kind,
        tabular_limit=tabular_limit,
    )
    result = _run(command)
    _commit_volumes()
    return result


@app.function(
    image=data_image,
    volumes={str(VOLUME_ROOT): DATA_VOLUME, "/cache": CACHE_VOLUME},
    timeout=60 * 60,
    retries=1,
)
def batch_process_registry() -> dict[str, Any]:
    _stage_registry_seed()
    results = [_run(command) for command in _batch_commands()]
    _commit_volumes()
    return {"steps": results}


@app.function(
    image=data_image,
    volumes={str(VOLUME_ROOT): DATA_VOLUME, "/cache": CACHE_VOLUME},
    timeout=60 * 60 * 2,
    retries=1,
)
def prepare_all_data(
    targets: str = "",
    max_dataset_size_gb: float = 10.0,
    synthetic_count_per_kind: int = 8,
    tabular_limit: int = 500,
) -> dict[str, Any]:
    _stage_registry_seed()
    ingestion = _run(
        _ingestion_command(
            targets=targets,
            max_dataset_size_gb=max_dataset_size_gb,
            no_network=False,
            synthetic_count_per_kind=synthetic_count_per_kind,
            tabular_limit=tabular_limit,
        )
    )
    batch = {"steps": [_run(command) for command in _batch_commands()]}
    _commit_volumes()
    return {"ingestion": ingestion, "batch": batch}


@app.function(
    image=training_image,
    volumes={str(VOLUME_ROOT): DATA_VOLUME, "/cache": CACHE_VOLUME},
    secrets=[HF_SECRET],
    gpu="A10G",
    timeout=60 * 60 * 8,
    retries=1,
)
def finetune_text(dry_run: bool = True, limit: int = 16) -> dict[str, Any]:
    if not dry_run:
        raise ValueError("full Nemotron training requires finetune_text_nemotron")
    preflight = _run(
        _preflight_command(
            modality="text",
            config_path=TEXT_MODAL_CONFIG,
            report_path=TEXT_PREFLIGHT_REPORT,
        )
    )
    command = [
        "python",
        "training/text/train_nemotron_lora.py",
        "--config",
        str(TEXT_MODAL_CONFIG),
    ]
    if dry_run:
        command.extend(["--dry-run", "--limit", str(limit)])
    result = _run(command)
    _commit_volumes()
    return {"preflight": preflight, "training": result}


@app.function(
    image=training_image,
    volumes={str(VOLUME_ROOT): DATA_VOLUME, "/cache": CACHE_VOLUME},
    secrets=[HF_SECRET],
    gpu="A10G",
    timeout=60 * 60 * 8,
    retries=1,
)
def finetune_text_nemotron(dry_run: bool = False, limit: int = 16) -> dict[str, Any]:
    preflight = _run(
        _preflight_command(
            modality="text",
            config_path=TEXT_MODAL_CONFIG,
            report_path=TEXT_PREFLIGHT_REPORT,
        )
    )
    mamba_install = _run(
        [
            "python",
            "-m",
            "pip",
            "install",
            "causal-conv1d>=1.4",
            "mamba-ssm>=2.2",
            "--no-build-isolation",
        ],
        extra_env={
            "CC": "/usr/bin/gcc",
            "CXX": "/usr/bin/g++",
            "CUDAHOSTCXX": "/usr/bin/g++",
        },
    )
    command = [
        "python",
        "training/text/train_nemotron_lora.py",
        "--config",
        str(TEXT_MODAL_CONFIG),
    ]
    if dry_run:
        command.extend(["--dry-run", "--limit", str(limit)])
    result = _run(command)
    _commit_volumes()
    return {"preflight": preflight, "mamba_install": mamba_install, "training": result}


@app.function(
    image=training_image,
    volumes={str(VOLUME_ROOT): DATA_VOLUME, "/cache": CACHE_VOLUME},
    secrets=[HF_SECRET],
    gpu="A10G",
    timeout=60 * 60 * 8,
    retries=1,
)
def finetune_vision(dry_run: bool = True, limit: int = 6) -> dict[str, Any]:
    preflight = _run(
        _preflight_command(
            modality="vision",
            config_path=VISION_MODAL_CONFIG,
            report_path=VISION_PREFLIGHT_REPORT,
            require_images=not dry_run,
        )
    )
    command = [
        "python",
        "training/vision/train_minicpm_v_lora.py",
        "--config",
        str(VISION_MODAL_CONFIG),
    ]
    if dry_run:
        command.extend(["--dry-run", "--limit", str(limit)])
    result = _run(command)
    _commit_volumes()
    return {"preflight": preflight, "training": result}


@app.local_entrypoint()
def main(
    action: str = "prepare_all_data",
    targets: str = "",
    max_dataset_size_gb: float = 10.0,
    dry_run: bool = True,
) -> None:
    if action == "ingest_data":
        result = ingest_data.remote(targets=targets, max_dataset_size_gb=max_dataset_size_gb)
    elif action == "batch_process":
        result = batch_process_registry.remote()
    elif action == "prepare_all_data":
        result = prepare_all_data.remote(
            targets=targets,
            max_dataset_size_gb=max_dataset_size_gb,
        )
    elif action == "finetune_text":
        if dry_run:
            result = finetune_text.remote(dry_run=True)
        else:
            result = finetune_text_nemotron.remote(dry_run=False)
    elif action == "finetune_text_nemotron":
        result = finetune_text_nemotron.remote(dry_run=dry_run)
    elif action == "finetune_vision":
        result = finetune_vision.remote(dry_run=dry_run)
    else:
        raise ValueError(f"unsupported action: {action}")
    print(json.dumps(result, indent=2, sort_keys=True))

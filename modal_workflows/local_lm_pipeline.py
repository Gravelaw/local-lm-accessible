from __future__ import annotations

import json
import os
import selectors
import shutil
import subprocess
import time
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
    REMOTE_ROOT / "training" / "text" / "configs" / "llama_nemotron_nano_modal_lora.yaml"
)
HYBRID_NEMOTRON_MODAL_CONFIG = (
    REMOTE_ROOT / "training" / "text" / "configs" / "nemotron_modal_prepared_lora.yaml"
)
VISION_MODAL_CONFIG = (
    REMOTE_ROOT / "training" / "vision" / "configs" / "minicpm_v_modal_document_lora.yaml"
)
TEXT_PREFLIGHT_REPORT = REPORTS_DIR / "fine_tuning_text_preflight.json"
VISION_PREFLIGHT_REPORT = REPORTS_DIR / "fine_tuning_vision_preflight.json"
NEMOTRON_DEPENDENCY_REPORT = REPORTS_DIR / "nemotron_dependency_build.json"
FINAL_TEXT_ADAPTER_EVAL_JSON = REPORTS_DIR / "final_text_adapter_eval.json"
FINAL_TEXT_ADAPTER_EVAL_MD = REPORTS_DIR / "final_text_adapter_eval.md"
FINAL_TEXT_ADAPTER_READINESS_JSON = REPORTS_DIR / "final_text_adapter_readiness.json"
FINAL_TEXT_ADAPTER_READINESS_MD = REPORTS_DIR / "final_text_adapter_readiness.md"
FINAL_TEXT_ADAPTER_PACKAGING_PLAN_JSON = REPORTS_DIR / "final_text_adapter_packaging_plan.json"
FINAL_TEXT_ADAPTER_PACKAGING_PLAN_MD = REPORTS_DIR / "final_text_adapter_packaging_plan.md"
FINAL_FINE_TUNING_SUMMARY_JSON = REPORTS_DIR / "final_finetuning_summary.json"
FINAL_FINE_TUNING_SUMMARY_MD = REPORTS_DIR / "final_finetuning_summary.md"
VISION_READINESS_JSON = REPORTS_DIR / "vision_readiness.json"
VISION_READINESS_MD = REPORTS_DIR / "vision_readiness.md"
ASR_CONTINGENCY_JSON = REPORTS_DIR / "asr_contingency.json"
ASR_CONTINGENCY_MD = REPORTS_DIR / "asr_contingency.md"
FINETUNING_COMPLETION_JSON = REPORTS_DIR / "finetuning_completion.json"
FINETUNING_COMPLETION_MD = REPORTS_DIR / "finetuning_completion.md"

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
        .add_local_file(
            "models/manifest.json",
            remote_path=str(REMOTE_ROOT / "models" / "manifest.json"),
        )
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

packaging_dependencies_image = (
    training_dependencies_image
    .apt_install("cmake")
    .pip_install("numpy>=1.26", "protobuf>=4.25", "sentencepiece>=0.2")
    .run_commands(
        "git clone --depth 1 https://github.com/ggml-org/llama.cpp /opt/llama.cpp",
        "ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1",
        (
            "CUDACXX=/usr/local/cuda/bin/nvcc CUDAHOSTCXX=/usr/bin/g++ "
            "cmake -S /opt/llama.cpp -B /opt/llama.cpp/build "
            "-DLLAMA_CURL=OFF -DGGML_CUDA=ON "
            "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc "
            "-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ "
            "-DCMAKE_CUDA_ARCHITECTURES=86 "
            "-DCMAKE_EXE_LINKER_FLAGS='-L/usr/local/cuda/lib64/stubs "
            "-Wl,-rpath-link,/usr/local/cuda/lib64/stubs' "
            "-DCMAKE_SHARED_LINKER_FLAGS='-L/usr/local/cuda/lib64/stubs "
            "-Wl,-rpath-link,/usr/local/cuda/lib64/stubs'"
        ),
        "cmake --build /opt/llama.cpp/build --target llama-cli -j2",
        "cmake --build /opt/llama.cpp/build --target llama-quantize -j2",
    )
)

packaging_image = _with_repo_files(packaging_dependencies_image)


def _run(
    command: list[str],
    *,
    cwd: Path = REMOTE_ROOT,
    extra_env: dict[str, str] | None = None,
    timeout_seconds: int | None = None,
    stream_output: bool = False,
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
    if stream_output:
        return _run_streaming(
            command,
            cwd=cwd,
            env=env,
            timeout_seconds=timeout_seconds,
        )
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        result = {
            "command": command,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "returncode": None,
            "timeout_seconds": timeout_seconds,
        }
        raise TimeoutError(json.dumps(result, indent=2, sort_keys=True)) from exc
    result = {
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "returncode": completed.returncode,
    }
    if completed.returncode != 0:
        raise RuntimeError(json.dumps(result, indent=2, sort_keys=True))
    return result


def _run_streaming(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_seconds: int | None,
) -> dict[str, Any]:
    started_at = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    output_lines: list[str] = []
    selector = selectors.DefaultSelector()
    if process.stdout is None:
        raise RuntimeError("streaming subprocess stdout was not captured")
    selector.register(process.stdout, selectors.EVENT_READ)

    timed_out = False
    while True:
        if timeout_seconds is not None and time.monotonic() - started_at > timeout_seconds:
            timed_out = True
            process.kill()
            break
        for key, _ in selector.select(timeout=1.0):
            line = key.fileobj.readline()
            if not line:
                continue
            output_lines.append(line)
            print(line, end="", flush=True)
        if process.poll() is not None:
            for line in process.stdout.readlines():
                output_lines.append(line)
                print(line, end="", flush=True)
            break

    selector.close()
    returncode = process.wait()
    result = {
        "command": command,
        "stdout": "".join(output_lines),
        "stderr": "",
        "returncode": returncode,
    }
    if timed_out:
        result["timeout_seconds"] = timeout_seconds
        raise TimeoutError(json.dumps(result, indent=2, sort_keys=True))
    if returncode != 0:
        raise RuntimeError(json.dumps(result, indent=2, sort_keys=True))
    return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as item:
        for chunk in iter(lambda: item.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_training_config(config_path: Path) -> dict[str, Any]:
    import yaml

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"training config must be a mapping: {config_path}")
    return config


def _stage_adapter_publish_dir(adapter_dir: Path, publish_dir: Path, *, base_model: str) -> Path:
    publish_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("adapter_*", "tokenizer*", "special_tokens_map.json", "chat_template.jinja"):
        for source in adapter_dir.glob(pattern):
            if source.is_file():
                shutil.copy2(source, publish_dir / source.name)
    readme = publish_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "---",
                "license: other",
                "base_model: " + base_model,
                "library_name: peft",
                "tags:",
                "- build-small-hackathon",
                "- local-lm",
                "- accessibility",
                "- llama-cpp",
                "---",
                "",
                "# local-lm accessible text LoRA",
                "",
                "Fine-tuned LoRA adapter for local-lm accessible assistant workflows.",
                "The base model is `nvidia/Llama-3.1-Nemotron-Nano-8B-v1`.",
                "",
                "This adapter was trained for local-first routing, summarization, JSON repair,",
                "tool-call JSON, and uncertainty-warning behavior for accessibility-focused use.",
                "",
                "Use locally only. Do not upload user documents to remote inference services.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return publish_dir


def _stage_gguf_readme(gguf_dir: Path) -> None:
    gguf_dir.mkdir(parents=True, exist_ok=True)
    (gguf_dir / "README.md").write_text(
        "\n".join(
            [
                "---",
                "license: other",
                "base_model: nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
                "tags:",
                "- build-small-hackathon",
                "- local-lm",
                "- accessibility",
                "- gguf",
                "- llama-cpp",
                "---",
                "",
                "# local-lm accessible GGUF",
                "",
                "GGUF exports for the local-lm accessible text model.",
                "",
                "Run with llama.cpp locally. The app is designed for local-first use and",
                "does not require cloud inference at runtime.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _cuda_extension_build_env() -> dict[str, str]:
    cuda_home = "/usr/local/cuda"
    host_compiler = "/usr/bin/g++"
    c_compiler = "/usr/bin/gcc"
    return {
        "CC": c_compiler,
        "CXX": host_compiler,
        "CUDAHOSTCXX": host_compiler,
        "CUDACXX": f"{cuda_home}/bin/nvcc",
        "CUDA_HOME": cuda_home,
        "CMAKE_CUDA_COMPILER": f"{cuda_home}/bin/nvcc",
        "CMAKE_CUDA_HOST_COMPILER": host_compiler,
        "CMAKE_CXX_COMPILER": host_compiler,
        "CMAKE_ARGS": (
            f"-DCMAKE_CUDA_COMPILER={cuda_home}/bin/nvcc "
            f"-DCMAKE_CUDA_HOST_COMPILER={host_compiler} "
            f"-DCMAKE_CXX_COMPILER={host_compiler}"
        ),
        "CUDAFLAGS": f"-ccbin={host_compiler} --allow-unsupported-compiler",
        "NVCC_PREPEND_FLAGS": f"-ccbin={host_compiler}",
    }


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
    timeout=60 * 10,
    retries=1,
)
def check_training_toolchain() -> dict[str, Any]:
    return _run(
        ["python", "scripts/check_cuda_toolchain.py"],
        extra_env=_cuda_extension_build_env(),
    )


@app.function(
    image=training_image,
    volumes={str(VOLUME_ROOT): DATA_VOLUME, "/cache": CACHE_VOLUME},
    secrets=[HF_SECRET],
    gpu="A10G",
    timeout=60 * 90,
    retries=0,
)
def prepare_nemotron_dependencies(
    timeout_seconds: int = 60 * 45,
    install_mamba: bool = False,
) -> dict[str, Any]:
    cuda_build_env = _cuda_extension_build_env()
    cuda_toolchain = _run(
        ["python", "scripts/check_cuda_toolchain.py"],
        extra_env=cuda_build_env,
    )
    mamba_install = None
    dependency_command = ["python", "scripts/check_nemotron_dependencies.py"]
    if install_mamba:
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
            extra_env=cuda_build_env,
            timeout_seconds=timeout_seconds,
            stream_output=True,
        )
        dependency_command.append("--include-mamba")
    dependency_check = _run(dependency_command, extra_env=cuda_build_env)
    report = {
        "cuda_toolchain": cuda_toolchain,
        "mamba_install": mamba_install,
        "dependency_check": dependency_check,
        "install_mamba": install_mamba,
        "timeout_seconds": timeout_seconds,
    }
    _write_json(NEMOTRON_DEPENDENCY_REPORT, report)
    _commit_volumes()
    return report


@app.function(
    image=training_image,
    volumes={str(VOLUME_ROOT): DATA_VOLUME, "/cache": CACHE_VOLUME},
    secrets=[HF_SECRET],
    gpu="A10G",
    timeout=60 * 60 * 8,
    retries=1,
)
def finetune_text_nemotron(dry_run: bool = False, limit: int = 16) -> dict[str, Any]:
    cuda_build_env = _cuda_extension_build_env()
    preflight = _run(
        _preflight_command(
            modality="text",
            config_path=TEXT_MODAL_CONFIG,
            report_path=TEXT_PREFLIGHT_REPORT,
        )
    )
    cuda_toolchain = _run(
        ["python", "scripts/check_cuda_toolchain.py"],
        extra_env=cuda_build_env,
    )
    dependency_check = _run(
        ["python", "scripts/check_nemotron_dependencies.py"],
        extra_env=cuda_build_env,
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
    finalization: dict[str, Any] | None = None
    if not dry_run:
        config = _read_training_config(TEXT_MODAL_CONFIG)
        output_dir = Path(str(config["training"]["output_dir"]))
        base_model = str(config["model"]["base_model"])
        finalization = _run(
            [
                "python",
                "scripts/finalize_text_adapter.py",
                "--adapter-dir",
                str(output_dir),
                "--base-model",
                base_model,
                "--run-name",
                output_dir.name,
                "--train-file",
                str(TRAINING_MANIFEST_ROOT / "text_sft_train.jsonl"),
                "--eval-file",
                str(TRAINING_MANIFEST_ROOT / "text_sft_validation.jsonl"),
                "--report-json",
                str(FINAL_FINE_TUNING_SUMMARY_JSON),
                "--report-md",
                str(FINAL_FINE_TUNING_SUMMARY_MD),
            ]
        )
    _commit_volumes()
    return {
        "preflight": preflight,
        "cuda_toolchain": cuda_toolchain,
        "mamba_install": None,
        "dependency_check": dependency_check,
        "training": result,
        "finalization": finalization,
    }


@app.function(
    image=training_image,
    volumes={str(VOLUME_ROOT): DATA_VOLUME, "/cache": CACHE_VOLUME},
    secrets=[HF_SECRET],
    gpu="A10G",
    timeout=60 * 30,
    retries=1,
)
def evaluate_text_adapter() -> dict[str, Any]:
    config = _read_training_config(TEXT_MODAL_CONFIG)
    output_dir = Path(str(config["training"]["output_dir"]))
    base_model = str(config["model"]["base_model"])
    command = [
        "python",
        "training/text/eval_text_adapter.py",
        "--input",
        str(TRAINING_MANIFEST_ROOT / "text_sft_validation.jsonl"),
        "--adapter-dir",
        str(output_dir),
        "--base-model",
        base_model,
        "--limit",
        "8",
        "--report-json",
        str(FINAL_TEXT_ADAPTER_EVAL_JSON),
        "--report-md",
        str(FINAL_TEXT_ADAPTER_EVAL_MD),
    ]
    result = _run(command)
    readiness = _run(
        [
            "python",
            "scripts/check_text_adapter_release_readiness.py",
            "--adapter-dir",
            str(output_dir),
            "--eval-report",
            str(FINAL_TEXT_ADAPTER_EVAL_JSON),
            "--output-json",
            str(FINAL_TEXT_ADAPTER_READINESS_JSON),
            "--output-md",
            str(FINAL_TEXT_ADAPTER_READINESS_MD),
        ]
    )
    _commit_volumes()
    return {"adapter_eval": result, "readiness": readiness}


@app.function(
    image=packaging_image,
    volumes={str(VOLUME_ROOT): DATA_VOLUME, "/cache": CACHE_VOLUME},
    secrets=[HF_SECRET],
    timeout=60 * 10,
    retries=1,
)
def plan_text_adapter_packaging() -> dict[str, Any]:
    config = _read_training_config(TEXT_MODAL_CONFIG)
    output_dir = Path(str(config["training"]["output_dir"]))
    merged_dir = VOLUME_ROOT / "training" / "text" / "llama_nemotron_nano_merged_hf"
    f16_gguf = VOLUME_ROOT / "models" / "text" / "llama-nemotron-nano-f16.gguf"
    quantized_dir = VOLUME_ROOT / "models" / "text"
    result = _run(
        [
            "python",
            "scripts/create_text_adapter_packaging_plan.py",
            "--readiness-report",
            str(FINAL_TEXT_ADAPTER_READINESS_JSON),
            "--adapter-dir",
            str(output_dir),
            "--merged-dir",
            str(merged_dir),
            "--f16-gguf",
            str(f16_gguf),
            "--quantized-dir",
            str(quantized_dir),
            "--output-json",
            str(FINAL_TEXT_ADAPTER_PACKAGING_PLAN_JSON),
            "--output-md",
            str(FINAL_TEXT_ADAPTER_PACKAGING_PLAN_MD),
        ]
    )
    _commit_volumes()
    return result


@app.function(
    image=packaging_image,
    volumes={str(VOLUME_ROOT): DATA_VOLUME, "/cache": CACHE_VOLUME},
    secrets=[HF_SECRET],
    gpu="A10G",
    timeout=60 * 60 * 8,
    retries=0,
)
def run_text_adapter_packaging() -> dict[str, Any]:
    config = _read_training_config(TEXT_MODAL_CONFIG)
    adapter_dir = Path(str(config["training"]["output_dir"]))
    base_model = str(config["model"]["base_model"])
    merged_dir = VOLUME_ROOT / "training" / "text" / "llama_nemotron_nano_merged_hf"
    f16_gguf = VOLUME_ROOT / "models" / "text" / "local-lm-accessible-text-f16.gguf"
    quantized_dir = VOLUME_ROOT / "models" / "text"
    q4_path = quantized_dir / "local-lm-accessible-text-Q4_K_M.gguf"
    q5_path = quantized_dir / "local-lm-accessible-text-Q5_K_M.gguf"
    llama_env = {
        "LLAMA_CPP_DIR": "/opt/llama.cpp",
        "LLAMA_CLI": "/opt/llama.cpp/build/bin/llama-cli",
        "LLAMA_QUANTIZE": "/opt/llama.cpp/build/bin/llama-quantize",
        "MODEL_BASENAME": "local-lm-accessible-text",
        "PYTHONPATH": "/opt/llama.cpp/gguf-py",
    }
    readiness = _run(
        [
            "python",
            "scripts/check_text_adapter_release_readiness.py",
            "--adapter-dir",
            str(adapter_dir),
            "--eval-report",
            str(FINAL_TEXT_ADAPTER_EVAL_JSON),
            "--output-json",
            str(FINAL_TEXT_ADAPTER_READINESS_JSON),
            "--output-md",
            str(FINAL_TEXT_ADAPTER_READINESS_MD),
        ]
    )
    merge = _run(
        [
            "python",
            "training/text/merge_adapter.py",
            "--base-model",
            base_model,
            "--adapter-dir",
            str(adapter_dir),
            "--output-dir",
            str(merged_dir),
            "--allow-remote-files",
            "--dtype",
            "bfloat16",
        ],
        timeout_seconds=60 * 60 * 3,
        stream_output=True,
    )
    export = _run(
        [
            "bash",
            "training/text/export_gguf.sh",
            str(merged_dir),
            str(f16_gguf),
        ],
        extra_env=llama_env,
        timeout_seconds=60 * 60 * 2,
        stream_output=True,
    )
    quantize = _run(
        [
            "bash",
            "training/text/quantize_gguf.sh",
            str(f16_gguf),
            str(quantized_dir),
        ],
        extra_env=llama_env,
        timeout_seconds=60 * 60 * 3,
        stream_output=True,
    )
    report = {
        "base_model": base_model,
        "adapter_dir": str(adapter_dir),
        "merged_dir": str(merged_dir),
        "f16_gguf": {"path": str(f16_gguf), "sha256": _sha256_file(f16_gguf)},
        "q4_gguf": {"path": str(q4_path), "sha256": _sha256_file(q4_path)},
        "q5_gguf": {"path": str(q5_path), "sha256": _sha256_file(q5_path)},
        "readiness": readiness,
        "merge": merge,
        "export": export,
        "quantize": quantize,
    }
    _write_json(REPORTS_DIR / "final_text_adapter_packaging_result.json", report)
    _commit_volumes()
    return report


@app.function(
    image=packaging_image,
    volumes={str(VOLUME_ROOT): DATA_VOLUME, "/cache": CACHE_VOLUME},
    secrets=[HF_SECRET],
    gpu="A10G",
    timeout=60 * 20,
    retries=1,
)
def smoke_test_packaged_gguf() -> dict[str, Any]:
    q4_path = VOLUME_ROOT / "models" / "text" / "local-lm-accessible-text-Q4_K_M.gguf"
    command = [
        "/opt/llama.cpp/build/bin/llama-cli",
        "-m",
        str(q4_path),
        "-p",
        "Return one short sentence explaining that local-lm runs locally.",
        "-n",
        "48",
        "--ctx-size",
        "2048",
        "--temp",
        "0.2",
        "--top-p",
        "0.9",
        "--n-gpu-layers",
        "99",
        "--single-turn",
        "--no-display-prompt",
    ]
    result = _run(
        command,
        timeout_seconds=60 * 10,
        stream_output=True,
    )
    report = {
        "model": str(q4_path),
        "backend": "llama.cpp CUDA",
        "command": command,
        "result": result,
        "passed": result["returncode"] == 0 and bool(str(result["stdout"]).strip()),
    }
    _write_json(REPORTS_DIR / "final_text_gguf_smoke.json", report)
    _commit_volumes()
    return report


@app.function(
    image=training_image,
    volumes={str(VOLUME_ROOT): DATA_VOLUME, "/cache": CACHE_VOLUME},
    timeout=60 * 10,
    retries=1,
)
def create_vision_readiness(dry_run: bool = True, limit: int = 6) -> dict[str, Any]:
    command = [
        "python",
        "scripts/create_vision_readiness_report.py",
        "--config",
        str(VISION_MODAL_CONFIG),
        "--report-json",
        str(VISION_READINESS_JSON),
        "--report-md",
        str(VISION_READINESS_MD),
        "--limit",
        str(limit),
    ]
    if not dry_run:
        command.append("--require-images")
    result = _run(command)
    _commit_volumes()
    return result


@app.function(
    image=data_image,
    volumes={str(VOLUME_ROOT): DATA_VOLUME, "/cache": CACHE_VOLUME},
    timeout=60 * 5,
    retries=1,
)
def check_asr_contingency() -> dict[str, Any]:
    result = _run(
        [
            "python",
            "scripts/check_asr_contingency.py",
            "--manifest",
            str(REMOTE_ROOT / "models" / "manifest.json"),
            "--eval-report",
            str(REPORTS_DIR / "asr_eval.json"),
            "--report-json",
            str(ASR_CONTINGENCY_JSON),
            "--report-md",
            str(ASR_CONTINGENCY_MD),
        ]
    )
    _commit_volumes()
    return result


@app.function(
    image=data_image,
    volumes={str(VOLUME_ROOT): DATA_VOLUME, "/cache": CACHE_VOLUME},
    timeout=60 * 5,
    retries=1,
)
def check_finetuning_completion() -> dict[str, Any]:
    config = _read_training_config(TEXT_MODAL_CONFIG)
    adapter_dir = Path(str(config["training"]["output_dir"]))
    result = _run(
        [
            "python",
            "scripts/check_finetuning_completion.py",
            "--adapter-dir",
            str(adapter_dir),
            "--finalization-report",
            str(FINAL_FINE_TUNING_SUMMARY_JSON),
            "--eval-report",
            str(FINAL_TEXT_ADAPTER_EVAL_JSON),
            "--readiness-report",
            str(FINAL_TEXT_ADAPTER_READINESS_JSON),
            "--packaging-report",
            str(REPORTS_DIR / "final_text_adapter_packaging_result.json"),
            "--smoke-report",
            str(REPORTS_DIR / "final_text_gguf_smoke.json"),
            "--vision-report",
            str(VISION_READINESS_JSON),
            "--asr-report",
            str(ASR_CONTINGENCY_JSON),
            "--report-json",
            str(FINETUNING_COMPLETION_JSON),
            "--report-md",
            str(FINETUNING_COMPLETION_MD),
        ]
    )
    _commit_volumes()
    return result


@app.function(
    image=packaging_image,
    volumes={str(VOLUME_ROOT): DATA_VOLUME, "/cache": CACHE_VOLUME},
    secrets=[HF_SECRET],
    timeout=60 * 60 * 4,
    retries=1,
)
def publish_hf_models(
    adapter_repo_id: str,
    gguf_repo_id: str,
    execute: bool = False,
    private: bool = False,
    skip_create: bool = False,
) -> dict[str, Any]:
    config = _read_training_config(TEXT_MODAL_CONFIG)
    adapter_dir = Path(str(config["training"]["output_dir"]))
    base_model = str(config["model"]["base_model"])
    adapter_publish_dir = VOLUME_ROOT / "hf_publish" / "text_lora"
    gguf_dir = VOLUME_ROOT / "models" / "text"
    _stage_adapter_publish_dir(adapter_dir, adapter_publish_dir, base_model=base_model)
    _stage_gguf_readme(gguf_dir)
    adapter_command = [
        "python",
        "scripts/publish_hf_artifacts.py",
        "--repo-id",
        adapter_repo_id,
        "--local-path",
        str(adapter_publish_dir),
        "--repo-type",
        "model",
        "--report-json",
        str(REPORTS_DIR / "hf_publish_adapter.json"),
        "--report-md",
        str(REPORTS_DIR / "hf_publish_adapter.md"),
    ]
    gguf_command = [
        "python",
        "scripts/publish_hf_artifacts.py",
        "--repo-id",
        gguf_repo_id,
        "--local-path",
        str(gguf_dir),
        "--repo-type",
        "model",
        "--report-json",
        str(REPORTS_DIR / "hf_publish_gguf.json"),
        "--report-md",
        str(REPORTS_DIR / "hf_publish_gguf.md"),
    ]
    if execute:
        adapter_command.append("--execute")
        gguf_command.append("--execute")
    if skip_create:
        adapter_command.append("--skip-create")
        gguf_command.append("--skip-create")
    if private:
        adapter_command.append("--private")
        gguf_command.append("--private")
    adapter = _run(adapter_command, timeout_seconds=60 * 60 * 2, stream_output=True)
    gguf = _run(gguf_command, timeout_seconds=60 * 60 * 3, stream_output=True)
    _commit_volumes()
    return {"adapter": adapter, "gguf": gguf, "execute": execute}


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
    dependency_timeout_seconds: int = 60 * 45,
    install_mamba_dependencies: bool = False,
    hf_adapter_repo_id: str = "build-small-hackathon/local-lm-accessible-text-lora",
    hf_gguf_repo_id: str = "build-small-hackathon/local-lm-accessible-gguf",
    publish_execute: bool = False,
    publish_private: bool = False,
    publish_skip_create: bool = False,
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
    elif action == "check_training_toolchain":
        result = check_training_toolchain.remote()
    elif action == "prepare_nemotron_dependencies":
        result = prepare_nemotron_dependencies.remote(
            timeout_seconds=dependency_timeout_seconds,
            install_mamba=install_mamba_dependencies,
        )
    elif action == "evaluate_text_adapter":
        result = evaluate_text_adapter.remote()
    elif action == "plan_text_adapter_packaging":
        result = plan_text_adapter_packaging.remote()
    elif action == "run_text_adapter_packaging":
        result = run_text_adapter_packaging.remote()
    elif action == "smoke_test_packaged_gguf":
        result = smoke_test_packaged_gguf.remote()
    elif action == "create_vision_readiness":
        result = create_vision_readiness.remote(dry_run=dry_run)
    elif action == "check_asr_contingency":
        result = check_asr_contingency.remote()
    elif action == "check_finetuning_completion":
        result = check_finetuning_completion.remote()
    elif action == "publish_hf_models":
        result = publish_hf_models.remote(
            adapter_repo_id=hf_adapter_repo_id,
            gguf_repo_id=hf_gguf_repo_id,
            execute=publish_execute,
            private=publish_private,
            skip_create=publish_skip_create,
        )
    else:
        raise ValueError(f"unsupported action: {action}")
    print(json.dumps(result, indent=2, sort_keys=True))

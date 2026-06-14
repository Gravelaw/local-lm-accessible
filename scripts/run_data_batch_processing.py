from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def run_step(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return {
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "returncode": completed.returncode,
    }


def run_batch(reports_dir: Path) -> dict[str, Any]:
    steps = [
        [sys.executable, "scripts/audit_source_registry.py", "--reports-dir", str(reports_dir)],
        [sys.executable, "scripts/map_datasets_to_tasks.py", "--reports-dir", str(reports_dir)],
        [sys.executable, "scripts/build_training_mix.py"],
        [
            sys.executable,
            "scripts/build_prepared_training_manifests.py",
            "--reports-dir",
            str(reports_dir),
        ],
        [sys.executable, "scripts/check_regional_balance.py", "--reports-dir", str(reports_dir)],
        [sys.executable, "scripts/verify_dataset_locality.py"],
    ]
    return {"steps": [run_step(step) for step in steps]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    result = run_batch(args.reports_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

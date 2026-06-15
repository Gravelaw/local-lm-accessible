from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ALLOWED_NAMESPACE = "build-small-hackathon"


def build_publish_plan(
    *,
    repo_id: str,
    local_path: Path,
    repo_type: str = "model",
    private: bool = False,
    skip_create: bool = False,
    allowed_namespace: str = ALLOWED_NAMESPACE,
) -> dict[str, Any]:
    if "/" not in repo_id:
        raise ValueError("repo_id must include a namespace, for example build-small-hackathon/name")
    namespace, _ = repo_id.split("/", maxsplit=1)
    if namespace != allowed_namespace:
        raise ValueError(f"repo_id namespace must be {allowed_namespace}: {repo_id}")
    if repo_type not in {"model", "dataset", "space"}:
        raise ValueError(f"unsupported repo_type: {repo_type}")
    if not local_path.exists():
        raise FileNotFoundError(f"missing publish path: {local_path}")
    commands: list[list[str]] = []
    if not skip_create:
        commands.append(["hf", "repos", "create", repo_id, "--type", repo_type, "--exist-ok"])
    commands.append(["hf", "upload-large-folder", repo_id, str(local_path), "--type", repo_type])
    return {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "local_path": str(local_path),
        "private": private,
        "skip_create": skip_create,
        "allowed_namespace": allowed_namespace,
        "commands": commands,
    }


def publish_artifacts(
    *,
    repo_id: str,
    local_path: Path,
    repo_type: str,
    private: bool,
    execute: bool,
    skip_create: bool,
    report_json: Path,
    report_md: Path,
) -> dict[str, Any]:
    plan = build_publish_plan(
        repo_id=repo_id,
        local_path=local_path,
        repo_type=repo_type,
        private=private,
        skip_create=skip_create,
    )
    result = {"executed": execute, "plan": plan}
    if execute:
        from huggingface_hub import HfApi

        api = HfApi()
        if not skip_create:
            api.create_repo(
                repo_id=repo_id,
                repo_type=repo_type,
                private=private,
                exist_ok=True,
            )
        commit_info = api.upload_folder(
            repo_id=repo_id,
            repo_type=repo_type,
            folder_path=str(local_path),
            commit_message="Publish local-lm accessible model artifacts",
        )
        result["commit"] = {
            "commit_url": getattr(commit_info, "commit_url", ""),
            "oid": getattr(commit_info, "oid", ""),
        }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_md.write_text(render_markdown(result), encoding="utf-8")
    return result


def render_markdown(result: dict[str, Any]) -> str:
    plan = result["plan"]
    lines = [
        "# Hugging Face Publish Report",
        "",
        f"- executed: {result['executed']}",
        f"- repo_id: {plan['repo_id']}",
        f"- repo_type: {plan['repo_type']}",
        f"- local_path: {plan['local_path']}",
        f"- private: {plan['private']}",
        "",
    ]
    if "commit" in result:
        lines.append(f"- commit_url: {result['commit'].get('commit_url', '')}")
        lines.append(f"- oid: {result['commit'].get('oid', '')}")
        lines.append("")
    lines.append("## Commands")
    lines.append("")
    for command in plan["commands"]:
        lines.append("```bash")
        lines.append(" ".join(command))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--local-path", type=Path, required=True)
    parser.add_argument("--repo-type", default="model", choices=("model", "dataset", "space"))
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--skip-create", action="store_true")
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    args = parser.parse_args()
    result = publish_artifacts(
        repo_id=args.repo_id,
        local_path=args.local_path,
        repo_type=args.repo_type,
        private=args.private,
        execute=args.execute,
        skip_create=args.skip_create,
        report_json=args.report_json,
        report_md=args.report_md,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

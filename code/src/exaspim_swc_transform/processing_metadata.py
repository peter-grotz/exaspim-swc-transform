from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aind_data_schema.core.processing import (
    Code,
    DataProcess,
    ProcessName,
    ProcessStage,
    ResourceUsage,
)


DEFAULT_DATA_PROCESS_FILENAME = "data_process.json"


def _run_git_command(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    output = result.stdout.strip()
    return output or None


def _get_git_remote_url_base() -> str:
    credentials = os.getenv("GIT_ACCESS_TOKEN")
    domain = os.getenv("GIT_HOST")
    if not all([credentials, domain]):
        try:
            username = os.getenv("CODEOCEAN_EMAIL") or _run_git_command(["git", "config", "user.email"])
            username = username.replace("@", "%40")
            token = os.getenv("CODEOCEAN_API_TOKEN")
            credentials = f"{username}:{token}"
            domain = os.getenv("CODEOCEAN_DOMAIN")
        except Exception as exc:
            raise ValueError(
                "GIT_ACCESS_TOKEN or CODEOCEAN_API_TOKEN environment variable is required"
            ) from exc
        if not all([credentials, domain]):
            raise ValueError(
                "GIT_ACCESS_TOKEN or CODEOCEAN_API_TOKEN environment variable is required"
            )
    return f"https://{credentials}@{domain}"


def get_version_from_git_remote(capsule_slug: str) -> str:
    git_remote_url = f"{_get_git_remote_url_base()}/capsule-{capsule_slug}.git"
    git_commit_hash = _run_git_command(["git", "ls-remote", git_remote_url, "HEAD"])
    if not git_commit_hash:
        raise ValueError(
            f"Could not retrieve git commit hash for capsule {capsule_slug}"
        )
    return git_commit_hash.split()[0]


def _to_aware_utc(timestamp: datetime | str | None) -> datetime:
    if timestamp is None:
        return datetime.now(timezone.utc)
    if isinstance(timestamp, str):
        value = timestamp
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        parsed = datetime.fromisoformat(value)
        return _to_aware_utc(parsed)
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def build_data_process(
    *,
    code_url: str,
    code_name: str,
    capsule_slug: str | None,
    run_script: str,
    step_name: str,
    step_label: str,
    parameters: dict[str, Any],
    experimenters: list[str],
    output_dir: str | Path,
    output_parameters: dict[str, Any],
    start_time: datetime | str | None,
    end_time: datetime | str | None = None,
    success: bool,
    error_message: str | None = None,
) -> DataProcess:
    notes = f"{step_label} completed successfully." if success else f"{step_label} failed."
    if error_message:
        notes = f"{notes} Error: {error_message}"

    return DataProcess(
        process_type=ProcessName.NEURON_SKELETON_PROCESSING,
        stage=ProcessStage.PROCESSING,
        name=step_name,
        code=Code(
            url=code_url,
            name=code_name,
            version=get_version_from_git_remote(capsule_slug),
            run_script=Path(run_script),
            language="Python",
            language_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            parameters=parameters,
        ),
        experimenters=experimenters,
        start_date_time=_to_aware_utc(start_time),
        end_date_time=_to_aware_utc(end_time),
        output_path=str(output_dir),
        output_parameters=output_parameters,
        notes=notes,
        resources=ResourceUsage(
            os=f"{platform.system()} {platform.release()}".strip(),
            architecture=platform.machine() or None,
            cpu=platform.processor() or platform.machine() or None,
            cpu_cores=os.cpu_count(),
        ),
    )


def write_data_process_json(
    *,
    code_url: str,
    code_name: str,
    capsule_slug: str | None,
    run_script: str,
    step_name: str,
    step_label: str,
    parameters: dict[str, Any],
    experimenters: list[str],
    output_dir: str | Path,
    output_parameters: dict[str, Any],
    start_time: datetime | str | None,
    end_time: datetime | str | None = None,
    success: bool,
    error_message: str | None = None,
    file_name: str = DEFAULT_DATA_PROCESS_FILENAME,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    data_process = build_data_process(
        code_url=code_url,
        code_name=code_name,
        capsule_slug=capsule_slug,
        run_script=run_script,
        step_name=step_name,
        step_label=step_label,
        parameters=parameters,
        experimenters=experimenters,
        output_dir=output_path,
        output_parameters=output_parameters,
        start_time=start_time,
        end_time=end_time,
        success=success,
        error_message=error_message,
    )
    metadata_path = output_path / file_name
    metadata_path.write_text(data_process.model_dump_json(indent=2), encoding="utf-8")
    return metadata_path

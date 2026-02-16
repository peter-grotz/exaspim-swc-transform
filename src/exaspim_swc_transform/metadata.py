"""Metadata and reporting helpers."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(root: Path, outpath: Path) -> None:
    entries: list[dict[str, Any]] = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            entries.append(
                {
                    "path": str(p.relative_to(root)),
                    "size": p.stat().st_size,
                    "sha256": _sha256(p),
                }
            )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def write_process_report(out_dir: Path, report: dict[str, Any]) -> None:
    (out_dir / "process_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def write_data_process(out_dir: Path, payload: dict[str, Any]) -> None:
    (out_dir / "data_process.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_data_process_payload(
    *,
    stage_name: str,
    software_version: str,
    start_time: str,
    end_time: str,
    input_location: str,
    output_location: str,
    parameters: dict[str, Any],
    notes: list[str],
) -> dict[str, Any]:
    # Kept intentionally minimal and JSON-serializable for initial integration.
    return {
        "name": stage_name,
        "software_version": software_version,
        "start_date_time": start_time,
        "end_date_time": end_time,
        "input_location": input_location,
        "output_location": output_location,
        "code_url": os.environ.get("CODE_URL", "https://github.com/peter-grotz/exaspim-swc-transform"),
        "parameters": parameters,
        "notes": notes,
    }

"""Metadata and reporting helpers."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aind_data_schema.components.identifiers import Code
from aind_data_schema.core.processing import DataProcess, ProcessName, ProcessStage, Processing


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


def build_processing_model(
    *,
    process_name: str,
    process_notes: str,
    software_version: str,
    start_time: str,
    end_time: str,
    input_location: str,
    output_location: str,
    parameters: dict[str, Any],
    n_inputs: int,
    n_outputs: int,
    n_failed: int,
) -> Processing:
    code_url = os.environ.get("CODE_URL", "https://github.com/peter-grotz/exaspim-swc-transform")
    pipeline_name = "exaspim-swc-transform-pipeline"
    experimenters = [os.environ.get("AIND_EXPERIMENTER", "AIND Scientific Computing")]

    pipeline_code = Code(
        name=pipeline_name,
        url=code_url,
        version=software_version,
    )
    process_code = Code(
        url=code_url,
        version=software_version,
        parameters=parameters,
    )

    data_process = DataProcess(
        name=process_name,
        process_type=ProcessName.ANALYSIS,
        stage=ProcessStage.PROCESSING,
        code=process_code,
        experimenters=experimenters,
        pipeline_name=pipeline_name,
        start_date_time=start_time,
        end_date_time=end_time,
        output_path=output_location,
        output_parameters={
            "input_location": input_location,
            "n_inputs": n_inputs,
            "n_outputs": n_outputs,
            "n_failed": n_failed,
        },
        notes=process_notes,
    )

    return Processing.create_with_sequential_process_graph(
        data_processes=[data_process],
        pipelines=[pipeline_code],
        notes=process_notes,
    )


def write_processing_files(out_dir: Path, processing: Processing) -> None:
    serialized = processing.model_dump_json(indent=2)
    # Canonical AIND filename
    (out_dir / "processing.json").write_text(serialized, encoding="utf-8")
    # Compatibility alias while transitioning downstream consumers
    (out_dir / "data_process.json").write_text(serialized, encoding="utf-8")

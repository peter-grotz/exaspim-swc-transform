"""Script entrypoint for ExaSPIM SWC transformation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from allensdk.core.swc import Compartment, Morphology
from aind_exaspim_register_cells import RegistrationPipeline

from exaspim_swc_transform import __version__
from exaspim_swc_transform.io_swc import read_swc
from exaspim_swc_transform.metadata import (
    build_data_process_payload,
    utc_now_iso,
    write_data_process,
    write_manifest,
    write_process_report,
)
from exaspim_swc_transform.naming import transformed_name
from exaspim_swc_transform.transform_resolution import resolve_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transform SWCs from sample space to CCF")
    parser.add_argument("--swc-dir", default=os.environ.get("SWC_DIR", "/data/final-world"))
    parser.add_argument("--transform-dir", default=os.environ.get("TRANSFORM_DIR", ""))
    parser.add_argument("--manual-df-path", default=os.environ.get("MANUAL_DF_PATH", ""))
    parser.add_argument("--dataset-id", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--naming-style",
        choices=["preserve", "suffix", "aligned_prefix"],
        default="preserve",
    )
    parser.add_argument(
        "--parity-mode",
        action="store_true",
        help="Match notebook-era defaults: /results/aligned and aligned_<stem>.swc",
    )
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def _build_pipeline(resolved) -> RegistrationPipeline:
    return RegistrationPipeline(
        dataset_id=resolved.dataset_id,
        output_dir=f"/results/{resolved.dataset_id}",
        acquisition_file=resolved.acquisition_file,
        brain_path=resolved.brain_path,
        resampled_brain_path=resolved.resampled_brain_path,
        brain_to_exaspim_transform_path=resolved.brain_to_exaspim_transform_path,
        exaspim_to_ccf_transform_path=resolved.exaspim_to_ccf_transform_path,
        ccf_path=resolved.ccf_path,
        exaspim_template_path=resolved.exaspim_template_path,
        transform_res=[10.0, 10.0, 10.0],
        level=2,
        manual_transform_path=resolved.manual_transform_path,
    )


def run(args: argparse.Namespace) -> int:
    if not args.transform_dir:
        raise ValueError("--transform-dir is required")

    swc_dir = Path(args.swc_dir)
    default_output = "/results/aligned" if args.parity_mode else "/results/aligned_swcs"
    out_dir = Path(args.output_dir or default_output)
    transform_dir = Path(args.transform_dir)
    naming_style = "aligned_prefix" if args.parity_mode else args.naming_style
    run_parameters = vars(args).copy()
    run_parameters["effective_output_dir"] = str(out_dir)
    run_parameters["effective_naming_style"] = naming_style

    if not swc_dir.is_dir():
        raise NotADirectoryError(f"SWC input directory does not exist: {swc_dir}")
    if not transform_dir.is_dir():
        raise NotADirectoryError(f"Transform directory does not exist: {transform_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    start_time = utc_now_iso()
    resolved = resolve_inputs(transform_dir, args.manual_df_path, args.dataset_id)
    pipeline = _build_pipeline(resolved)

    ccf, ants_exaspim, brain_img, resampled_img = pipeline.load_images()

    all_swcs = sorted(swc_dir.rglob("*.swc"))
    report: dict[str, object] = {
        "dataset_id": resolved.dataset_id,
        "inputs": len(all_swcs),
        "outputs": 0,
        "failed": [],
    }

    for swc_path in all_swcs:
        try:
            morph = read_swc(swc_path, add_offset=True)
            coords = np.array([[c["x"], c["y"], c["z"]] for c in morph.compartment_list])
            resampled_cells = pipeline.preprocess_coords(
                coords,
                brain_img.numpy(),
                resampled_img.numpy(),
                swc_path.stem,
            )
            idx_pts, _ = pipeline.apply_transforms_to_points(
                resampled_cells,
                resampled_img,
                ants_exaspim,
                ccf,
                swc_path.stem,
            )

            transformed = []
            for i, node in enumerate(morph.compartment_list):
                compartment = Compartment(**node)
                compartment["x"] = idx_pts[i][0] * 10.0
                compartment["y"] = idx_pts[i][1] * 10.0
                compartment["z"] = idx_pts[i][2] * 10.0
                transformed.append(compartment)

            aligned_morph = Morphology(transformed)
            out_name = transformed_name(swc_path, naming_style)
            out_path = out_dir / out_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            aligned_morph.save(str(out_path))
            report["outputs"] = int(report["outputs"]) + 1
        except Exception as exc:  # pragma: no cover - runtime logging path
            failed = report["failed"]
            assert isinstance(failed, list)
            failed.append({"file": str(swc_path), "error": str(exc)})
            if args.fail_fast:
                raise

    end_time = utc_now_iso()
    report["start_time"] = start_time
    report["end_time"] = end_time

    write_process_report(Path("/results"), report)
    write_data_process(
        Path("/results"),
        build_data_process_payload(
            stage_name="exaspim_swc_transform",
            software_version=__version__,
            start_time=start_time,
            end_time=end_time,
            input_location=str(swc_dir),
            output_location=str(out_dir),
            parameters=run_parameters,
            notes=[
                f"dataset_id={resolved.dataset_id}",
                f"acquisition_file={resolved.acquisition_file}",
                f"manual_df={args.manual_df_path or '<none>'}",
            ],
        ),
    )
    write_manifest(swc_dir, Path("/results/manifests/inputs_manifest.json"))
    write_manifest(out_dir, Path("/results/manifests/outputs_manifest.json"))

    Path("/results/runtime_args.json").write_text(json.dumps(run_parameters, indent=2), encoding="utf-8")
    return 0


def main() -> None:
    args = parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()

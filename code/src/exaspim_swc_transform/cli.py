"""CCF transform-only entrypoint for SWCs."""

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
    build_processing_model,
    utc_now_iso,
    write_processing_files,
    write_manifest,
    write_process_report,
)
from exaspim_swc_transform.naming import transformed_name
from exaspim_swc_transform.transform_resolution import (
    DEFAULT_CCF_TEMPLATE,
    DEFAULT_EXASPIM_TEMPLATE,
    DEFAULT_EXASPIM_TO_CCF_AFFINE,
    DEFAULT_EXASPIM_TO_CCF_INVERSE_WARP,
    resolve_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transform SWCs to CCF space")
    parser.add_argument("--swc-dir", default=os.environ.get("SWC_DIR", "/data/final-world"))
    parser.add_argument("--transform-dir", default=os.environ.get("TRANSFORM_DIR", ""))
    parser.add_argument("--manual-df-path", default=os.environ.get("MANUAL_DF_PATH", ""))
    parser.add_argument("--manual-df-filename", default=os.environ.get("MANUAL_DF_FILENAME", ""))
    parser.add_argument("--dataset-id", default="")
    parser.add_argument(
        "--acquisition-file-path",
        default=os.environ.get("ACQUISITION_FILE_PATH", ""),
    )
    parser.add_argument(
        "--loaded-zarr-image-path",
        default=os.environ.get("LOADED_ZARR_IMAGE_PATH", ""),
    )
    parser.add_argument(
        "--resampled-zarr-image-path",
        default=os.environ.get("RESAMPLED_ZARR_IMAGE_PATH", ""),
    )
    parser.add_argument(
        "--sample-to-exaspim-affine-path",
        default=os.environ.get("SAMPLE_TO_EXASPIM_AFFINE_PATH", ""),
    )
    parser.add_argument(
        "--sample-to-exaspim-inverse-warp-path",
        default=os.environ.get("SAMPLE_TO_EXASPIM_INVERSE_WARP_PATH", ""),
    )
    parser.add_argument(
        "--exaspim-to-ccf-affine-path",
        default=os.environ.get("EXASPIM_TO_CCF_AFFINE_PATH", DEFAULT_EXASPIM_TO_CCF_AFFINE),
    )
    parser.add_argument(
        "--exaspim-to-ccf-inverse-warp-path",
        default=os.environ.get(
            "EXASPIM_TO_CCF_INVERSE_WARP_PATH",
            DEFAULT_EXASPIM_TO_CCF_INVERSE_WARP,
        ),
    )
    parser.add_argument(
        "--ccf-template-path",
        default=os.environ.get("CCF_TEMPLATE_PATH", DEFAULT_CCF_TEMPLATE),
    )
    parser.add_argument(
        "--exaspim-template-path",
        default=os.environ.get("EXASPIM_TEMPLATE_PATH", DEFAULT_EXASPIM_TEMPLATE),
    )
    parser.add_argument("--output-root", default="/results/exaspim_swc_transform")
    parser.add_argument(
        "--metadata-dir",
        default="",
        help="Optional metadata directory override. Defaults to --output-root.",
    )
    parser.add_argument("--naming-style", choices=["preserve", "suffix"], default="preserve")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def _build_pipeline(resolved, debug_output_dir: Path) -> RegistrationPipeline:
    return RegistrationPipeline(
        dataset_id=resolved.dataset_id,
        output_dir=str(debug_output_dir),
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


def _record_failure(report: dict[str, object], stage: str, file_path: Path, exc: Exception) -> None:
    failed = report["failed"]
    assert isinstance(failed, list)
    failed.append({"stage": stage, "file": str(file_path), "error": str(exc)})


def run(args: argparse.Namespace) -> int:
    if not args.transform_dir:
        raise ValueError("--transform-dir is required")

    swc_dir = Path(args.swc_dir)
    transform_dir = Path(args.transform_dir)
    output_root = Path(args.output_root)
    metadata_dir = Path(args.metadata_dir) if args.metadata_dir else output_root
    swc_out_dir = output_root

    if not swc_dir.is_dir():
        raise NotADirectoryError(f"SWC input directory does not exist: {swc_dir}")
    if not transform_dir.is_dir():
        raise NotADirectoryError(f"Transform directory does not exist: {transform_dir}")

    swc_out_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    run_parameters = vars(args).copy()
    run_parameters["effective_swc_output_dir"] = str(swc_out_dir)
    run_parameters["effective_metadata_dir"] = str(metadata_dir)

    start_time = utc_now_iso()
    resolved = resolve_inputs(
        transform_dir,
        args.manual_df_path,
        args.dataset_id,
        acquisition_file_path=args.acquisition_file_path,
        loaded_zarr_image_path=args.loaded_zarr_image_path,
        resampled_zarr_image_path=args.resampled_zarr_image_path,
        sample_to_exaspim_affine_path=args.sample_to_exaspim_affine_path,
        sample_to_exaspim_inverse_warp_path=args.sample_to_exaspim_inverse_warp_path,
        exaspim_to_ccf_affine_path=args.exaspim_to_ccf_affine_path,
        exaspim_to_ccf_inverse_warp_path=args.exaspim_to_ccf_inverse_warp_path,
        ccf_template_path=args.ccf_template_path,
        exaspim_template_path=args.exaspim_template_path,
        manual_df_filename=args.manual_df_filename,
    )
    debug_output_dir = output_root / resolved.dataset_id
    debug_output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = _build_pipeline(resolved, debug_output_dir)
    ccf, ants_exaspim, brain_img, resampled_img = pipeline.load_images()

    all_swcs = sorted(swc_dir.rglob("*.swc"))
    report: dict[str, object] = {
        "dataset_id": resolved.dataset_id,
        "inputs": len(all_swcs),
        "transformed": 0,
        "swc_outputs": 0,
        "failed": [],
    }

    for swc_path in all_swcs:
        final_name = transformed_name(swc_path, args.naming_style)
        final_swc = swc_out_dir / final_name
        final_swc.parent.mkdir(parents=True, exist_ok=True)

        try:
            morph = read_swc(swc_path, add_offset=True)
            coords = np.array([[c["x"], c["y"], c["z"]] for c in morph.compartment_list])
            prepped_cells = pipeline.preprocess_coords(
                coords,
                brain_img.numpy(),
                resampled_img.numpy(),
                swc_path.stem,
            )
            idx_pts, _ = pipeline.apply_transforms_to_points(
                prepped_cells,
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

            Morphology(transformed).save(str(final_swc))
            report["transformed"] = int(report["transformed"]) + 1
            report["swc_outputs"] = int(report["swc_outputs"]) + 1
        except Exception as exc:  # pragma: no cover
            _record_failure(report, "transform", swc_path, exc)
            if args.fail_fast:
                raise
            continue

    end_time = utc_now_iso()
    report["start_time"] = start_time
    report["end_time"] = end_time

    write_process_report(metadata_dir, report)

    processing = build_processing_model(
        process_name="SWC Processing",
        software_version=__version__,
        start_time=start_time,
        end_time=end_time,
        parameters=run_parameters,
    )
    write_processing_files(metadata_dir, processing)

    manifests_dir = metadata_dir / "manifests"
    write_manifest(swc_dir, manifests_dir / "inputs_manifest.json")
    write_manifest(output_root, manifests_dir / "outputs_manifest.json")

    (metadata_dir / "runtime_args.json").write_text(
        json.dumps(run_parameters, indent=2),
        encoding="utf-8",
    )
    return 0


def main() -> None:
    args = parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()

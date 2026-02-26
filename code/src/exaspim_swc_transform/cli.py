"""CCF transform-only entrypoint for SWCs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from allensdk.core.swc import Compartment, Morphology
from aind_exaspim_register_cells import RegistrationPipeline

from exaspim_swc_transform.io_swc import read_swc
from exaspim_swc_transform.naming import transformed_name
from exaspim_swc_transform.transform_resolution import (
    DEFAULT_CCF_TEMPLATE,
    DEFAULT_EXASPIM_TEMPLATE,
    DEFAULT_EXASPIM_TO_CCF_AFFINE,
    DEFAULT_EXASPIM_TO_CCF_INVERSE_WARP,
    resolve_inputs,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_experimenters() -> list[str]:
    raw = os.environ.get("EXPERIMENTERS", "MSMA Team")
    parts = [item.strip() for item in raw.split(",") if item.strip()]
    return parts or ["MSMA Team"]


def _carry_forward_swc_refinement() -> None:
    src_root = Path("/data/swc_refinement")
    dst_root = Path("/results/swc_refinement")

    def _copy_dir(src: Path, dst: Path) -> None:
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)

    if src_root.is_dir():
        _copy_dir(src_root, dst_root)
        return

    # Backward-compatible fallback for older refinement layouts.
    fallback_dirs = (
        "final-world",
        "final-voxel",
        "merged",
        "mips",
        "resampled",
        "voxel",
        "fixed",
        "refined",
        "astar",
    )
    copied_any = False
    for name in fallback_dirs:
        src = Path("/data") / name
        if src.is_dir():
            _copy_dir(src, dst_root / name)
            copied_any = True
    if copied_any:
        print(f"Carried forward legacy refinement outputs into {dst_root}")


def _prepare_metadata_steps_dir() -> Path:
    dst = Path("/results/metadata")
    dst.mkdir(parents=True, exist_ok=True)
    src = Path("/data/metadata")
    if src.is_dir():
        for path in sorted(src.glob("*.data_process.json")):
            shutil.copy2(path, dst / path.name)
    return dst


def _write_step_dataprocess(
    args: argparse.Namespace,
    start_date_time: str,
    input_swc_count: int,
    transformed_swc_count: int,
    resolved_dataset_id: str,
    output_root: Path,
    swc_out_dir: Path,
) -> None:
    metadata_dir = _prepare_metadata_steps_dir()
    step_name = os.environ.get("AIND_STEP_NAME", "exaspim_swc_transform")
    process_type = os.environ.get("AIND_PROCESS_TYPE", "Neuron skeleton processing")
    stage = os.environ.get("AIND_STAGE", "Processing")
    code_url = os.environ.get(
        "AIND_CODE_URL",
        "https://github.com/peter-grotz/exaspim-swc-transform",
    )
    code_version = os.environ.get("AIND_CODE_VERSION", "unknown")
    parameters = dict(vars(args))
    parameters["resolved_dataset_id"] = resolved_dataset_id

    payload = {
        "name": step_name,
        "process_type": process_type,
        "stage": stage,
        "code": {
            "url": code_url,
            "version": code_version,
            "parameters": parameters,
        },
        "experimenters": _parse_experimenters(),
        "start_date_time": start_date_time,
        "end_date_time": utc_now_iso(),
        "output_parameters": {
            "output_root": str(output_root),
            "aligned_swc_dir": str(swc_out_dir),
            "input_swc_count": input_swc_count,
            "transformed_swc_count": transformed_swc_count,
        },
    }

    out_path = metadata_dir / f"{step_name}.data_process.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote step metadata: {out_path}")


def parse_args() -> argparse.Namespace:
    def env_default(name: str, default: str = "") -> str:
        # Prefer lowercase env names while preserving uppercase compatibility.
        return os.environ.get(name.lower(), os.environ.get(name, default))

    parser = argparse.ArgumentParser(description="Transform SWCs to CCF space")
    parser.add_argument("--swc-dir", "--swc_dir", dest="swc_dir", default=env_default("SWC_DIR", "/data/final-world"))
    parser.add_argument("--transform-dir", "--transform_dir", dest="transform_dir", default=env_default("TRANSFORM_DIR", ""))
    parser.add_argument("--manual-df-path", "--manual_df_path", dest="manual_df_path", default=env_default("MANUAL_DF_PATH", ""))
    parser.add_argument("--manual-df-filename", "--manual_df_filename", dest="manual_df_filename", default=env_default("MANUAL_DF_FILENAME", ""))
    parser.add_argument("--dataset-id", "--dataset_id", dest="dataset_id", default=env_default("DATASET_ID", ""))
    parser.add_argument(
        "--acquisition-file-path", "--acquisition_file_path",
        dest="acquisition_file_path",
        default=env_default("ACQUISITION_FILE_PATH", ""),
    )
    parser.add_argument(
        "--loaded-zarr-image-path", "--loaded_zarr_image_path",
        dest="loaded_zarr_image_path",
        default=env_default("LOADED_ZARR_IMAGE_PATH", ""),
    )
    parser.add_argument(
        "--resampled-zarr-image-path", "--resampled_zarr_image_path",
        dest="resampled_zarr_image_path",
        default=env_default("RESAMPLED_ZARR_IMAGE_PATH", ""),
    )
    parser.add_argument(
        "--sample-to-exaspim-affine-path", "--sample_to_exaspim_affine_path",
        dest="sample_to_exaspim_affine_path",
        default=env_default("SAMPLE_TO_EXASPIM_AFFINE_PATH", ""),
    )
    parser.add_argument(
        "--sample-to-exaspim-inverse-warp-path", "--sample_to_exaspim_inverse_warp_path",
        dest="sample_to_exaspim_inverse_warp_path",
        default=env_default("SAMPLE_TO_EXASPIM_INVERSE_WARP_PATH", ""),
    )
    parser.add_argument(
        "--exaspim-to-ccf-affine-path", "--exaspim_to_ccf_affine_path",
        dest="exaspim_to_ccf_affine_path",
        default=env_default("EXASPIM_TO_CCF_AFFINE_PATH", DEFAULT_EXASPIM_TO_CCF_AFFINE),
    )
    parser.add_argument(
        "--exaspim-to-ccf-inverse-warp-path", "--exaspim_to_ccf_inverse_warp_path",
        dest="exaspim_to_ccf_inverse_warp_path",
        default=env_default(
            "EXASPIM_TO_CCF_INVERSE_WARP_PATH",
            DEFAULT_EXASPIM_TO_CCF_INVERSE_WARP,
        ),
    )
    parser.add_argument(
        "--ccf-template-path", "--ccf_template_path",
        dest="ccf_template_path",
        default=env_default("CCF_TEMPLATE_PATH", DEFAULT_CCF_TEMPLATE),
    )
    parser.add_argument(
        "--exaspim-template-path", "--exaspim_template_path",
        dest="exaspim_template_path",
        default=env_default("EXASPIM_TEMPLATE_PATH", DEFAULT_EXASPIM_TEMPLATE),
    )
    parser.add_argument("--output-root", "--output_root", dest="output_root", default=env_default("OUTPUT_ROOT", "/results/exaspim_swc_transform"))
    parser.add_argument("--naming-style", "--naming_style", dest="naming_style", choices=["preserve", "suffix"], default=env_default("NAMING_STYLE", "preserve"))
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


def run(args: argparse.Namespace) -> int:
    start_date_time = utc_now_iso()
    if not args.transform_dir:
        raise ValueError("--transform-dir is required")

    swc_dir = Path(args.swc_dir)
    transform_dir = Path(args.transform_dir)
    output_root = Path(args.output_root)
    swc_out_dir = output_root / "aligned_swcs"

    if not swc_dir.is_dir():
        raise NotADirectoryError(f"SWC input directory does not exist: {swc_dir}")
    if not transform_dir.is_dir():
        raise NotADirectoryError(f"Transform directory does not exist: {transform_dir}")

    swc_out_dir.mkdir(parents=True, exist_ok=True)
    _carry_forward_swc_refinement()
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

    transformed_count = 0
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
            transformed_count += 1
        except Exception as exc:  # pragma: no cover
            if args.fail_fast:
                raise
            print(f"Failed to transform {swc_path}: {exc}")
            continue

    _write_step_dataprocess(
        args=args,
        start_date_time=start_date_time,
        input_swc_count=len(all_swcs),
        transformed_swc_count=transformed_count,
        resolved_dataset_id=resolved.dataset_id,
        output_root=output_root,
        swc_out_dir=swc_out_dir,
    )

    return 0


def main() -> None:
    args = parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()

"""Resolve registration assets and transform paths from bundle layout."""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResolvedInputs:
    dataset_id: str
    acquisition_file: str
    brain_path: str
    resampled_brain_path: str
    brain_to_exaspim_transform_path: list[str]
    exaspim_to_ccf_transform_path: list[str]
    manual_transform_path: list[str]
    ccf_path: str
    exaspim_template_path: str


def _pick_one(pattern: str, what: str) -> str:
    hits = sorted(glob.glob(pattern, recursive=True), key=lambda p: os.path.getsize(p), reverse=True)
    if not hits:
        raise FileNotFoundError(f"Could not find {what}; pattern={pattern}")
    return os.path.abspath(hits[0])


def resolve_inputs(transform_dir: Path, manual_df_path: str = "", dataset_id: str = "") -> ResolvedInputs:
    bundle_root = transform_dir
    align_root = bundle_root / "ccf_alignment"
    if align_root.exists():
        bundle_root = align_root

    meta_root = bundle_root / "registration_metadata"
    acquisition_file = _pick_one(str(meta_root / "acquisition_*.json"), "acquisition file")

    if not dataset_id:
        match = re.search(r"\d{6}", bundle_root.name) or re.search(r"\d{6}", Path(acquisition_file).stem)
        dataset_id = match.group(0) if match else "unknown"

    brain_path = _pick_one(
        str(bundle_root / "registration_metadata" / "*_10um_loaded_zarr_img.nii.gz"),
        "10um loaded zarr image",
    )
    resampled_brain_path = _pick_one(
        str(bundle_root / "registration_metadata" / "*_10um_resampled_zarr_img.nii.gz"),
        "10um resampled zarr image",
    )

    affine_to_exaspim = _pick_one(
        str(bundle_root / "**" / "*_to_exaSPIM_SyN_0GenericAffine.mat"),
        "sample->exaSPIM affine",
    )
    invwarp_to_exaspim = _pick_one(
        str(bundle_root / "**" / "*_to_exaSPIM_SyN_1InverseWarp.nii.gz"),
        "sample->exaSPIM inverse warp",
    )

    exaspim_to_ccf_affine = _pick_one(str(Path("/data") / "**" / "0GenericAffine.mat"), "exaSPIM->CCF affine")
    exaspim_to_ccf_invwarp = _pick_one(
        str(Path("/data") / "**" / "1InverseWarp.nii.gz"),
        "exaSPIM->CCF inverse warp",
    )

    ccf_path = _pick_one(str(Path("/data") / "**" / "average_template_10.nii.gz"), "CCF template")
    exaspim_template_path = _pick_one(str(Path("/data") / "**" / "fixed_median.nii.gz"), "ExaSPIM template")

    manual_transform_path = [manual_df_path] if manual_df_path else []

    return ResolvedInputs(
        dataset_id=dataset_id,
        acquisition_file=acquisition_file,
        brain_path=brain_path,
        resampled_brain_path=resampled_brain_path,
        brain_to_exaspim_transform_path=[affine_to_exaspim, invwarp_to_exaspim],
        exaspim_to_ccf_transform_path=[exaspim_to_ccf_affine, exaspim_to_ccf_invwarp],
        manual_transform_path=manual_transform_path,
        ccf_path=ccf_path,
        exaspim_template_path=exaspim_template_path,
    )

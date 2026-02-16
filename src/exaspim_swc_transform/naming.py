"""Naming helpers for transformed outputs."""

from pathlib import Path


def transformed_name(input_path: Path, style: str = "preserve") -> str:
    """Return output file name for transformed SWC.

    style=preserve keeps original stem unchanged.
    style=suffix appends explicit coordinate-space suffix.
    """
    stem = input_path.stem
    if style == "aligned_prefix":
        return f"aligned_{stem}.swc"
    if style == "suffix":
        return f"{stem}__space-ccf_res-10um.swc"
    return f"{stem}.swc"

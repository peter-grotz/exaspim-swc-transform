"""CCF annotation and JSON output helpers."""

from __future__ import annotations

from pathlib import Path

from aind_morphology_utils.ccf_annotation import CCFMorphologyMapper
from aind_morphology_utils.utils import read_swc
from aind_morphology_utils.writers import MouseLightJsonWriter


def build_mapper(resolution: int = 10) -> CCFMorphologyMapper:
    return CCFMorphologyMapper(resolution=resolution)


def annotate_swc_to_json(mapper: CCFMorphologyMapper, swc_path: Path, out_json_path: Path) -> None:
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    morph = read_swc(str(swc_path))
    mapper.annotate_morphology(morph)
    writer = MouseLightJsonWriter(morph)
    writer.write(str(out_json_path))

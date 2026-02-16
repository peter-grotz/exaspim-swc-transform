"""Import adapter for upstream structure-type assignment implementation."""

from __future__ import annotations

from importlib import import_module


def _load_fix_structure_assignment():
    candidates = [
        ("aind_morphology_utils.fix_types", "fix_structure_assignment"),
        ("aind_morphology_utils.scripts.fix_types", "fix_structure_assignment"),
        ("fix_types", "fix_structure_assignment"),
    ]
    for module_name, attr in candidates:
        try:
            module = import_module(module_name)
            return getattr(module, attr)
        except (ImportError, AttributeError):
            continue
    tried = ", ".join(f"{m}.{a}" for m, a in candidates)
    raise ImportError(f"Could not import upstream fix_types function. Tried: {tried}")


fix_structure_assignment = _load_fix_structure_assignment()


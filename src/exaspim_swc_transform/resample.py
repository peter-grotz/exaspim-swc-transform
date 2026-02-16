"""Import adapter for upstream SWC resampling implementation."""

from __future__ import annotations

from importlib import import_module


def _load_resample_swc():
    candidates = [
        ("neuron_tracing_utils.resample", "resample_swc"),
        ("neuron_tracing_utils.scripts.resample", "resample_swc"),
        ("neuron_tracing_utils.util.resample", "resample_swc"),
    ]
    for module_name, attr in candidates:
        try:
            module = import_module(module_name)
            return getattr(module, attr)
        except (ImportError, AttributeError):
            continue
    tried = ", ".join(f"{m}.{a}" for m, a in candidates)
    raise ImportError(f"Could not import upstream resample function. Tried: {tried}")


resample_swc = _load_resample_swc()


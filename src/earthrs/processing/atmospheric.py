"""Atmospheric-correction processing routines."""

from __future__ import annotations

from typing import Any

from earthrs.processing.core import _ATMOSPHERIC_REGISTRY, register_atmospheric_method
from earthrs.scene import Scene


def atmospheric_correction(scene: Scene, *, method: str = "default", **kwargs: Any) -> Scene:
    """Apply atmospheric correction to reflectance inputs.

    Method names are resolved from the atmospheric-correction registry. Atmospheric
    correction is not yet implemented, so the registered ``"default"`` method raises
    ``NotImplementedError``.
    """

    processor = _ATMOSPHERIC_REGISTRY.get(method.lower())
    if processor is None:
        raise ValueError(f"Unknown atmospheric-correction method '{method}'.")
    return processor(scene, **kwargs)


def _default_atmospheric_correction(scene: Scene, **kwargs: Any) -> Scene:
    _ = scene
    _ = kwargs
    raise NotImplementedError(
        "Atmospheric correction is not yet implemented. Check for a new version of earthrs."
    )


register_atmospheric_method("default", _default_atmospheric_correction)

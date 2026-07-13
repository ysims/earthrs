"""Atmospheric-correction processing placeholder."""

from __future__ import annotations

from typing import Any

from earthrs.scene import Scene


def atmospheric_correction(scene: Scene, *, method: str = "default", **kwargs: Any) -> Scene:
    """Apply atmospheric correction to reflectance inputs.

    The function is registry-ready but atmospheric correction is not yet implemented.
    """

    _ = scene
    _ = method
    _ = kwargs
    raise NotImplementedError(
        "Atmospheric correction is not yet implemented. Check for a new version of earthrs."
    )

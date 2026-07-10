"""Spectral index placeholder APIs.

The module intentionally exposes functional APIs so new indices can be added with consistent
signatures.
"""

from __future__ import annotations

from typing import Any

from earthrs.scene import Scene


def ndvi(scene: Scene, *, nir_band: str = "nir", red_band: str = "red") -> Any:
    """Compute NDVI from a scene.

    TODO: Implement band extraction and NDVI calculation.
    """

    raise NotImplementedError("TODO: implement ndvi")


def ndwi(scene: Scene, *, green_band: str = "green", nir_band: str = "nir") -> Any:
    """Compute NDWI from a scene.

    TODO: Implement band extraction and NDWI calculation.
    """

    raise NotImplementedError("TODO: implement ndwi")


def evi(
    scene: Scene,
    *,
    nir_band: str = "nir",
    red_band: str = "red",
    blue_band: str = "blue",
    gain: float = 2.5,
    c1: float = 6.0,
    c2: float = 7.5,
    canopy_background: float = 1.0,
) -> Any:
    """Compute EVI from a scene.

    TODO: Implement band extraction and EVI calculation.
    """

    raise NotImplementedError("TODO: implement evi")

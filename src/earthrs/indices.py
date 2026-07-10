"""Spectral index placeholder APIs.

The module intentionally exposes functional APIs so new indices can be added with consistent
signatures.
"""

from __future__ import annotations

from math import isnan
from typing import Any

from earthrs.scene import Scene


def ndvi(scene: Scene, *, nir_band: str = "nir", red_band: str = "red") -> Any:
    """Compute NDVI from a scene.

    Compute normalised difference vegetation index.
    """

    nir = _get_band(scene, nir_band)
    red = _get_band(scene, red_band)
    return _map_binary(nir, red, lambda n, r: _safe_div(n - r, n + r))


def ndwi(scene: Scene, *, green_band: str = "green", nir_band: str = "nir") -> Any:
    """Compute NDWI from a scene.

    Compute normalised difference water index.
    """

    green = _get_band(scene, green_band)
    nir = _get_band(scene, nir_band)
    return _map_binary(green, nir, lambda g, n: _safe_div(g - n, g + n))


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

    Compute enhanced vegetation index.
    """

    nir = _get_band(scene, nir_band)
    red = _get_band(scene, red_band)
    blue = _get_band(scene, blue_band)
    numerator = _map_binary(nir, red, lambda n, r: n - r)
    denominator = _map_ternary(
        nir,
        red,
        blue,
        lambda n, r, b: n + c1 * r - c2 * b + canopy_background,
    )
    return _map_binary(numerator, denominator, lambda num, den: gain * _safe_div(num, den))


def _get_band(scene: Scene, band_name: str) -> Any:
    if isinstance(scene.data, dict):
        if band_name not in scene.data:
            raise ValueError(f"Band '{band_name}' not found in scene data.")
        return scene.data[band_name]
    raise TypeError("Index calculations currently expect scene data as a mapping of band names to arrays.")


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator) / float(denominator)


def _map_binary(left: Any, right: Any, func) -> Any:
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return [_map_binary(l_value, r_value, func) for l_value, r_value in zip(left, right, strict=True)]
    if left is None or right is None:
        return float("nan")
    return _normalise_nan(func(float(left), float(right)))


def _map_ternary(first: Any, second: Any, third: Any, func) -> Any:
    if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)) and isinstance(third, (list, tuple)):
        return [
            _map_ternary(first_value, second_value, third_value, func)
            for first_value, second_value, third_value in zip(first, second, third, strict=True)
        ]
    if first is None or second is None or third is None:
        return float("nan")
    return _normalise_nan(func(float(first), float(second), float(third)))


def _normalise_nan(value: float) -> float:
    if isnan(value):
        return float("nan")
    return value

"""Spectral index placeholder APIs.

The module intentionally exposes functional APIs so new indices can be added with consistent
signatures.
"""

from __future__ import annotations

from math import isnan
from typing import Any

from earthrs.scene import Scene

try:
    import rasterio
except ImportError:  # pragma: no cover - exercised only when rasterio is absent
    rasterio = None

try:
    import xarray
except ImportError:  # pragma: no cover - exercised only when xarray is absent
    xarray = None


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
    """Return band data as nested Python lists, regardless of the backing store.

    Supports the following ``scene.data`` shapes:

    - ``dict[str, list]`` (the always-available, zero-dependency default): returned as-is.
    - ``xarray.DataArray`` with a ``band`` dimension whose coordinate values are band names
      (matching the order of ``scene.band_names``), or ``xarray.Dataset`` with one data
      variable per band: converted to nested lists via ``.values.tolist()``.
    - A rasterio dataset handle (anything exposing ``.read(band_index)``, 1-indexed): band
      names map to rasterio band indices by position, so ``scene.band_names[i]`` corresponds
      to rasterio band ``i + 1``. Converted to nested lists via ``.tolist()``.

    The result is always plain nested lists so downstream elementwise helpers (`_map_binary`,
    `_map_ternary`) do not need to know about the original backing store.
    """

    data = scene.data
    if isinstance(data, dict):
        if band_name not in data:
            raise ValueError(f"Band '{band_name}' not found in scene data.")
        return data[band_name]
    if xarray is not None and isinstance(data, xarray.Dataset):
        if band_name not in data.data_vars:
            raise ValueError(f"Band '{band_name}' not found in scene data.")
        return data[band_name].values.tolist()
    if xarray is not None and isinstance(data, xarray.DataArray):
        if "band" not in data.dims or "band" not in data.coords:
            raise TypeError(
                "xarray.DataArray scene data must have a 'band' dimension with a matching "
                "coordinate to support band access."
            )
        if band_name not in data.coords["band"].values.tolist():
            raise ValueError(f"Band '{band_name}' not found in scene data.")
        return data.sel(band=band_name).values.tolist()
    if rasterio is not None and hasattr(data, "read") and hasattr(data, "count"):
        if band_name not in scene.band_names:
            raise ValueError(f"Band '{band_name}' not found in scene data.")
        band_index = scene.band_names.index(band_name) + 1
        return data.read(band_index).tolist()
    raise TypeError(
        "Index calculations currently expect scene data as a mapping of band names to arrays, "
        "an xarray DataArray/Dataset, or a rasterio dataset."
    )


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator) / float(denominator)


def _map_binary(left: Any, right: Any, func) -> Any:
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        pairs = zip(left, right, strict=True)
        return [_map_binary(l_value, r_value, func) for l_value, r_value in pairs]
    if left is None or right is None:
        return float("nan")
    return _normalise_nan(func(float(left), float(right)))


def _map_ternary(first: Any, second: Any, third: Any, func) -> Any:
    is_sequence = (
        isinstance(first, (list, tuple))
        and isinstance(second, (list, tuple))
        and isinstance(third, (list, tuple))
    )
    if is_sequence:
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

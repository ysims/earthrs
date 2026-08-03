"""Spectral index placeholder APIs.

The module intentionally exposes functional APIs so new indices can be added with consistent
signatures.
"""

from __future__ import annotations

from math import isnan
from typing import Any

from earthrs.scene import Scene


def ndvi(
    scene: Scene,
    *,
    nir_band: str = "nir",
    red_band: str = "red",
    nodata: float | None = None,
) -> Any:
    """Compute NDVI from a scene.

    Compute normalised difference vegetation index.

    Parameters
    ----------
    nodata:
        Sentinel value that marks a pixel as missing, in addition to literal
        ``None`` values (which are always treated as missing). A pixel is
        masked when its value is *exactly* equal to ``nodata`` -- no epsilon
        tolerance is applied. Exact matching is the more predictable default
        for the common integer-style sentinels (``-9999``, ``0``, ``65535``,
        etc.) written verbatim by raster sources; float sentinels affected by
        rounding should be normalised by the caller before calling this
        function. When ``nodata`` is not supplied, it is resolved from
        ``scene.metadata["nodata"]`` if present: either a single scalar
        applied to every band, or a ``dict[str, float]`` mapping band name to
        its own sentinel. If neither is available, only literal ``None``
        values are treated as missing (unchanged, backward-compatible
        behaviour).
    """

    nir = _mask_nodata(_get_band(scene, nir_band), _resolve_nodata(scene, nir_band, nodata))
    red = _mask_nodata(_get_band(scene, red_band), _resolve_nodata(scene, red_band, nodata))
    return _map_binary(nir, red, lambda n, r: _safe_div(n - r, n + r))


def ndwi(
    scene: Scene,
    *,
    green_band: str = "green",
    nir_band: str = "nir",
    nodata: float | None = None,
) -> Any:
    """Compute NDWI from a scene.

    Compute normalised difference water index.

    Parameters
    ----------
    nodata:
        Sentinel value that marks a pixel as missing, in addition to literal
        ``None`` values (which are always treated as missing). A pixel is
        masked when its value is *exactly* equal to ``nodata`` -- no epsilon
        tolerance is applied. Exact matching is the more predictable default
        for the common integer-style sentinels (``-9999``, ``0``, ``65535``,
        etc.) written verbatim by raster sources; float sentinels affected by
        rounding should be normalised by the caller before calling this
        function. When ``nodata`` is not supplied, it is resolved from
        ``scene.metadata["nodata"]`` if present: either a single scalar
        applied to every band, or a ``dict[str, float]`` mapping band name to
        its own sentinel. If neither is available, only literal ``None``
        values are treated as missing (unchanged, backward-compatible
        behaviour).
    """

    green = _mask_nodata(_get_band(scene, green_band), _resolve_nodata(scene, green_band, nodata))
    nir = _mask_nodata(_get_band(scene, nir_band), _resolve_nodata(scene, nir_band, nodata))
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
    nodata: float | None = None,
) -> Any:
    """Compute EVI from a scene.

    Compute enhanced vegetation index.

    Parameters
    ----------
    nodata:
        Sentinel value that marks a pixel as missing, in addition to literal
        ``None`` values (which are always treated as missing). A pixel is
        masked when its value is *exactly* equal to ``nodata`` -- no epsilon
        tolerance is applied. Exact matching is the more predictable default
        for the common integer-style sentinels (``-9999``, ``0``, ``65535``,
        etc.) written verbatim by raster sources; float sentinels affected by
        rounding should be normalised by the caller before calling this
        function. When ``nodata`` is not supplied, it is resolved from
        ``scene.metadata["nodata"]`` if present: either a single scalar
        applied to every band, or a ``dict[str, float]`` mapping band name to
        its own sentinel. If neither is available, only literal ``None``
        values are treated as missing (unchanged, backward-compatible
        behaviour). A pixel missing in any of the three bands propagates to
        a ``nan`` result, matching the existing ``None``-handling behaviour.
    """

    nir = _mask_nodata(_get_band(scene, nir_band), _resolve_nodata(scene, nir_band, nodata))
    red = _mask_nodata(_get_band(scene, red_band), _resolve_nodata(scene, red_band, nodata))
    blue = _mask_nodata(_get_band(scene, blue_band), _resolve_nodata(scene, blue_band, nodata))
    numerator = _map_binary(nir, red, lambda n, r: n - r)
    denominator = _map_ternary(
        nir,
        red,
        blue,
        lambda n, r, b: n + c1 * r - c2 * b + canopy_background,
    )
    return _map_binary(numerator, denominator, lambda num, den: gain * _safe_div(num, den))


def _resolve_nodata(scene: Scene, band_name: str, nodata: float | None) -> float | None:
    """Resolve the nodata sentinel that applies to a given band.

    Resolution order:

    1. The explicit ``nodata`` argument, if provided -- applied to every band.
    2. ``scene.metadata["nodata"]``, if present:

       - a ``dict`` is treated as a per-band mapping (``{band_name: value}``)
         and looked up by ``band_name`` (missing entries mean "no sentinel
         for this band");
       - any other value is treated as a single global sentinel applied to
         every band.

    3. ``None`` -- no sentinel-based masking, i.e. only literal ``None``
       values in the data are treated as missing.
    """

    if nodata is not None:
        return nodata
    meta_nodata = scene.metadata.get("nodata")
    if isinstance(meta_nodata, dict):
        return meta_nodata.get(band_name)
    return meta_nodata


def _mask_nodata(band: Any, nodata: float | None) -> Any:
    """Recursively replace pixels equal to ``nodata`` with ``None``.

    This lets the existing ``None``-is-missing handling in ``_map_binary``
    and ``_map_ternary`` uniformly cover both literal ``None`` values and
    numeric nodata sentinels, without duplicating the missing-data logic.
    Comparison against ``nodata`` is an exact numeric equality check (see
    the ``nodata`` parameter docs on ``ndvi``/``ndwi``/``evi`` for why exact
    matching, rather than an epsilon tolerance, is the default here).
    """

    if nodata is None:
        return band
    if isinstance(band, (list, tuple)):
        return [_mask_nodata(value, nodata) for value in band]
    if band is None:
        return None
    if float(band) == float(nodata):
        return None
    return band


def _get_band(scene: Scene, band_name: str) -> Any:
    if isinstance(scene.data, dict):
        if band_name not in scene.data:
            raise ValueError(f"Band '{band_name}' not found in scene data.")
        return scene.data[band_name]
    raise TypeError(
        "Index calculations currently expect scene data as a mapping of band names to arrays."
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

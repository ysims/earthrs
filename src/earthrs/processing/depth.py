"""Depth-correction processing routines.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

from earthrs.processing.core import (
    _DEPTH_REGISTRY,
    _map_binary,
    _map_unary,
    register_depth_method,
)
from earthrs.scene import Scene

_NON_SPECTRAL_BANDS = {"qa60", "scl", "qa_pixel"}


def depth_correct(
    scene: Scene,
    *,
    method: str = "lyzenga",
    depth: Any | None = None,
    **kwargs: Any,
) -> Scene:
    """Apply depth correction to raster data.

    Method names are resolved from the depth-correction registry.
    """

    processor = _DEPTH_REGISTRY.get(method.lower())
    if processor is None:
        raise ValueError(f"Unknown depth-correction method '{method}'.")
    return processor(scene, depth=depth, **kwargs)


def _band_param(
    bands: Iterable[str], value: Any, default: float
) -> dict[str, float]:
    """Normalise a per-band parameter to a ``{band: value}`` mapping.

    Accepts ``None`` (every band gets `default`), a single scalar (every band gets
    that value), or a mapping (missing bands fall back to `default`).
    """

    band_list = list(bands)
    if value is None:
        return dict.fromkeys(band_list, default)
    if isinstance(value, Mapping):
        return {band: float(value.get(band, default)) for band in band_list}
    return dict.fromkeys(band_list, float(value))


def _log_transform(values: Any, deep_water_radiance: float, epsilon: float) -> Any:
    """Apply the Lyzenga log-linearising transform ``X = ln(L - L_s)`` elementwise."""

    return _map_unary(values, lambda v: math.log(max(v - deep_water_radiance, epsilon)))


def _finalise_lyzenga(
    scene: Scene, *, data: dict[str, Any], band_names: tuple[str, ...], variant: str
) -> Scene:
    """Build the result `Scene` shared by all Lyzenga variants."""

    metadata = dict(scene.metadata)
    metadata["depth_method"] = f"lyzenga_{variant}"
    return Scene(
        data=data,
        crs=scene.crs,
        transform=scene.transform,
        metadata=metadata,
        band_names=band_names,
        acquisition_time=scene.acquisition_time,
        sensor=scene.sensor,
        history=(*scene.history, f"depth_correct:lyzenga:{variant}"),
        masks=dict(scene.masks),
        cloud_mask=scene.cloud_mask,
        cloud_probability=scene.cloud_probability,
    )


def _lyzenga_1978(
    scene: Scene,
    *,
    deep_water_radiance: Mapping[str, float] | float | None = None,
    epsilon: float = 1e-6,
) -> Scene:
    """Lyzenga (1978) per-band log-linearising transform.

    Replaces each spectral band with ``X_i = ln(L_i - L_si)`` (eqs. 1 & 7), which
    is approximately linear in water depth. Non-spectral bands (``qa60``, ``scl``,
    ``qa_pixel``) pass through unchanged. Bands are not combined with each other.
    """

    dwr = _band_param(scene.data.keys(), deep_water_radiance, 0.0)
    updated = {}
    for band, values in scene.data.items():
        if band in _NON_SPECTRAL_BANDS:
            updated[band] = values
            continue
        updated[band] = _log_transform(values, dwr[band], epsilon)
    return _finalise_lyzenga(scene, data=updated, band_names=scene.band_names, variant="1978")


def _lyzenga_1981(
    scene: Scene,
    *,
    band_i: str = "blue",
    band_j: str = "green",
    k_i: float | None = None,
    k_j: float | None = None,
    deep_water_radiance: Mapping[str, float] | float | None = None,
    epsilon: float = 1e-6,
) -> Scene:
    """Lyzenga (1981) two-band depth-invariant bottom index.

    Adds a ``lyzenga_dii`` band computed from eq. (2),
    ``Y_i = [K_j * X_i - K_i * X_j] / sqrt(K_i^2 + K_j^2)``, where
    ``X = ln(L - L_s)`` and `k_i`/`k_j` are the water-attenuation coefficients for
    `band_i`/`band_j`. The result is (ideally) independent of water depth, so it can
    be used as a bottom-type index rather than an estimate of depth itself.
    """

    if k_i is None or k_j is None:
        raise ValueError(
            "Lyzenga 1981 depth-invariant index requires water-attenuation "
            "coefficients `k_i` and `k_j`."
        )
    missing = [band for band in (band_i, band_j) if band not in scene.data]
    if missing:
        raise ValueError(f"Lyzenga 1981 requires bands {missing} in scene data.")
    denominator = math.sqrt(k_i**2 + k_j**2)
    if denominator == 0:
        raise ValueError("Lyzenga 1981 requires `k_i` and `k_j` not both be zero.")
    dwr = _band_param((band_i, band_j), deep_water_radiance, 0.0)
    x_i = _log_transform(scene.data[band_i], dwr[band_i], epsilon)
    x_j = _log_transform(scene.data[band_j], dwr[band_j], epsilon)
    dii = _map_binary(x_i, x_j, lambda xi, xj: (k_j * xi - k_i * xj) / denominator)
    updated = dict(scene.data)
    updated["lyzenga_dii"] = dii
    band_names = (*scene.band_names, "lyzenga_dii")
    return _finalise_lyzenga(scene, data=updated, band_names=band_names, variant="1981")


def _lyzenga_2006(
    scene: Scene,
    *,
    bands: tuple[str, ...] = ("blue", "green"),
    coefficients: Mapping[str, float] | None = None,
    intercept: float = 0.0,
    deep_water_radiance: Mapping[str, float] | float | None = None,
    epsilon: float = 1e-6,
) -> Scene:
    """Lyzenga et al. (2006) empirical multi-band depth regression.

    Adds a ``lyzenga_depth`` band computed from eq. (9),
    ``depth = h_0 - sum(h_j * X_j)``, where ``X_j = ln(L_j - L_sj)`` and `h_j` are
    empirically fitted `coefficients` (`intercept` supplies ``h_0``). Unlike the
    1978/1981 variants, this produces a direct depth estimate rather than a
    depth-invariant index.
    """

    if not bands:
        raise ValueError("Lyzenga 2006 depth regression requires at least one band.")
    if coefficients is None:
        raise ValueError(
            "Lyzenga 2006 depth regression requires `coefficients` (empirically "
            "fitted h_j weights per band, see eq. (9))."
        )
    missing_bands = [band for band in bands if band not in scene.data]
    if missing_bands:
        raise ValueError(f"Lyzenga 2006 requires bands {missing_bands} in scene data.")
    missing_coeff = [band for band in bands if band not in coefficients]
    if missing_coeff:
        raise ValueError(f"Missing regression coefficients for bands {missing_coeff}.")
    dwr = _band_param(bands, deep_water_radiance, 0.0)
    depth_values: Any = None
    for band in bands:
        x_band = _log_transform(scene.data[band], dwr[band], epsilon)
        term = _map_unary(x_band, lambda xv, h=coefficients[band]: -h * xv)
        if depth_values is None:
            depth_values = term
        else:
            depth_values = _map_binary(depth_values, term, lambda a, b: a + b)
    depth_values = _map_unary(depth_values, lambda v: v + intercept)
    updated = dict(scene.data)
    updated["lyzenga_depth"] = depth_values
    band_names = (*scene.band_names, "lyzenga_depth")
    return _finalise_lyzenga(scene, data=updated, band_names=band_names, variant="2006")


_LYZENGA_VARIANTS = {
    "1978": _lyzenga_1978,
    "1981": _lyzenga_1981,
    "2006": _lyzenga_2006,
}


def _lyzenga_depth(
    scene: Scene,
    *,
    depth: Any | None = None,
    variant: str = "1981",
    **kwargs: Any,
) -> Scene:
    """Dispatch to the requested Lyzenga variant (``"1978"``, ``"1981"``, ``"2006"``).

    Lyzenga methods derive depth information from radiance; unlike Maritorena's
    attenuation model, none of them take an already-known depth as input, so
    `depth` is accepted (for API symmetry with other depth-correction methods) but
    ignored.
    """

    _ = depth
    if not isinstance(scene.data, dict):
        raise TypeError("Lyzenga correction expects mapping-based scene data.")
    variant_key = str(variant)
    implementation = _LYZENGA_VARIANTS.get(variant_key)
    if implementation is None:
        raise ValueError(
            f"Unknown Lyzenga variant '{variant}'. Expected one of "
            f"{sorted(_LYZENGA_VARIANTS)}."
        )
    return implementation(scene, **kwargs)


def _stumpf_depth(
    scene: Scene,
    *,
    depth: Any | None = None,
    blue_band: str = "blue",
    green_band: str = "green",
    m0: float = 0.0,
    m1: float = 1.0,
    **_: Any,
) -> Scene:
    """Stumpf (2003) band-ratio depth transform.

    Adds a ``stumpf_depth`` band from ``m0 - m1 * ln(blue) / ln(green)``.
    """

    _ = depth
    if not isinstance(scene.data, dict):
        raise TypeError("Stumpf correction expects mapping-based scene data.")
    blue = scene.data[blue_band]
    green = scene.data[green_band]
    stumpf = _map_binary(
        blue,
        green,
        lambda b, g: m0 - m1 * (math.log(max(b, 1e-6)) / math.log(max(g, 1e-6))),
    )
    updated = dict(scene.data)
    updated["stumpf_depth"] = stumpf
    metadata = dict(scene.metadata)
    metadata["depth_method"] = "stumpf"
    return Scene(
        data=updated,
        crs=scene.crs,
        transform=scene.transform,
        metadata=metadata,
        band_names=(*scene.band_names, "stumpf_depth"),
        acquisition_time=scene.acquisition_time,
        sensor=scene.sensor,
        history=(*scene.history, "depth_correct:stumpf"),
        masks=dict(scene.masks),
        cloud_mask=scene.cloud_mask,
        cloud_probability=scene.cloud_probability,
    )


def _maritorena_depth(
    scene: Scene,
    *,
    depth: Any | None = None,
    attenuation: float = 0.1,
    **_: Any,
) -> Scene:
    """Single-parameter exponential attenuation correction given a known `depth`.

    Scales each band by ``exp(attenuation * depth)``.
    """

    if not isinstance(scene.data, dict):
        raise TypeError("Maritorena correction expects mapping-based scene data.")
    if depth is None:
        raise ValueError("Maritorena correction requires `depth`.")
    updated = {}
    for band, values in scene.data.items():
        updated[band] = _map_binary(
            values, depth, lambda value, d: value * math.exp(attenuation * d)
        )
    metadata = dict(scene.metadata)
    metadata["depth_method"] = "maritorena"
    return Scene(
        data=updated,
        crs=scene.crs,
        transform=scene.transform,
        metadata=metadata,
        band_names=scene.band_names,
        acquisition_time=scene.acquisition_time,
        sensor=scene.sensor,
        history=(*scene.history, "depth_correct:maritorena"),
        masks=dict(scene.masks),
        cloud_mask=scene.cloud_mask,
        cloud_probability=scene.cloud_probability,
    )


register_depth_method("lyzenga", _lyzenga_depth)
register_depth_method("stumpf", _stumpf_depth)
register_depth_method("maritorena", _maritorena_depth)

"""Depth-correction processing routines."""

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

    Method names are resolved from the depth-correction registry. There are three
    Lyzenga formulations, each denoted using the year of publication --
    ``"lyzenga1978"``, ``"lyzenga1981"``, ``"lyzenga2006"``. ``"lyzenga"`` is an
    alias for ``"lyzenga2006"``, the most recent and most widely used variant.
    """

    processor = _DEPTH_REGISTRY.get(method.lower())
    if processor is None:
        raise ValueError(f"Unknown depth-correction method '{method}'.")
    return processor(scene, depth=depth, **kwargs)


def _require_mapping_data(scene: Scene, label: str) -> None:
    if not isinstance(scene.data, dict):
        raise TypeError(f"{label} correction expects mapping-based scene data.")


def _band_param(bands: Iterable[str], value: Any, default: float) -> dict[str, float]:
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
    scene: Scene, *, data: dict[str, Any], band_names: tuple[str, ...], method: str
) -> Scene:
    """Build the result `Scene` shared by all Lyzenga variants."""

    metadata = dict(scene.metadata)
    metadata["depth_method"] = method
    return Scene(
        data=data,
        crs=scene.crs,
        transform=scene.transform,
        metadata=metadata,
        band_names=band_names,
        acquisition_time=scene.acquisition_time,
        sensor=scene.sensor,
        history=(*scene.history, f"depth_correct:{method}"),
        masks=dict(scene.masks),
        cloud_mask=scene.cloud_mask,
        cloud_probability=scene.cloud_probability,
    )


def _lyzenga_1978(
    scene: Scene,
    *,
    depth: Any | None = None,
    deep_water_radiance: Mapping[str, float] | float | None = None,
    epsilon: float = 1e-6,
    **_: Any,
) -> Scene:
    """Lyzenga (1978) per-band log-linearising transform.

    Replaces each spectral band with ``X_i = ln(L_i - L_si)`` (eqs. 1 & 7), which
    is approximately linear in water depth. Non-spectral bands (``qa60``, ``scl``,
    ``qa_pixel``) pass through unchanged. Bands are not combined with each other.
    """

    _ = depth
    _require_mapping_data(scene, "Lyzenga 1978")
    dwr = _band_param(scene.data.keys(), deep_water_radiance, 0.0)
    updated = {}
    for band, values in scene.data.items():
        if band in _NON_SPECTRAL_BANDS:
            updated[band] = values
            continue
        updated[band] = _log_transform(values, dwr[band], epsilon)
    return _finalise_lyzenga(scene, data=updated, band_names=scene.band_names, method="lyzenga1978")


def _lyzenga_1981(
    scene: Scene,
    *,
    depth: Any | None = None,
    band_i: str = "blue",
    band_j: str = "green",
    k_i: float | None = None,
    k_j: float | None = None,
    deep_water_radiance: Mapping[str, float] | float | None = None,
    epsilon: float = 1e-6,
    **_: Any,
) -> Scene:
    """Lyzenga (1981) two-band depth-invariant bottom index.

    Adds a ``lyzenga_dii`` band computed from eq. (2),
    ``Y_i = [K_j * X_i - K_i * X_j] / sqrt(K_i^2 + K_j^2)``, where
    ``X = ln(L - L_s)`` and `k_i`/`k_j` are the water-attenuation coefficients for
    `band_i`/`band_j`. The result is (ideally) independent of water depth, so it can
    be used as a bottom-type index rather than an estimate of depth itself.
    """

    _ = depth
    _require_mapping_data(scene, "Lyzenga 1981")
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
    return _finalise_lyzenga(scene, data=updated, band_names=band_names, method="lyzenga1981")


def _lyzenga_2006(
    scene: Scene,
    *,
    depth: Any | None = None,
    bands: tuple[str, ...] = ("blue", "green"),
    coefficients: Mapping[str, float] | None = None,
    intercept: float = 0.0,
    deep_water_radiance: Mapping[str, float] | float | None = None,
    epsilon: float = 1e-6,
    **_: Any,
) -> Scene:
    """Lyzenga et al. (2006) empirical multi-band depth regression.

    Adds a ``lyzenga_depth`` band computed from eq. (9),
    ``depth = h_0 - sum(h_j * X_j)``, where ``X_j = ln(L_j - L_sj)`` and `h_j` are
    empirically fitted `coefficients` (`intercept` supplies ``h_0``). Unlike the
    1978/1981 variants, this produces a direct depth estimate rather than a
    depth-invariant index. This is the most recent and most widely used Lyzenga
    variant, so it is also registered as the plain ``"lyzenga"`` method.
    """

    _ = depth
    _require_mapping_data(scene, "Lyzenga 2006")
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
    return _finalise_lyzenga(scene, data=updated, band_names=band_names, method="lyzenga2006")


def _stumpf_depth(
    scene: Scene,
    *,
    depth: Any | None = None,
    blue_band: str = "blue",
    green_band: str = "green",
    m0: float = 0.0,
    m1: float = 1.0,
    n: float = 1000.0,
    epsilon: float = 1e-6,
    **_: Any,
) -> Scene:
    """Stumpf, Holderied & Sinclair (2003) band-ratio depth transform.

    Adds a ``stumpf_depth`` band from eq. (1),
    ``depth = m1 * ln(n * Rw_i) / ln(n * Rw_j) - m0``, where `Rw_i`/`Rw_j` are
    `blue_band`/`green_band` reflectance (blue is the numerator band, ``i``; green
    is the denominator band, ``j``, following the paper's ``lambda_i``/``lambda_j``
    notation), and `m0`/`m1` are empirically tuned per-image
    coefficients. `n` is a fixed scaling constant, chosen so that ``n * Rw`` stays
    comfortably above 1 across the sensor's expected reflectance range: this keeps
    ``ln(n * Rw)`` positive (avoiding a sign flip in the ratio) and well away from
    zero (avoiding the numerical instability that taking logs of small fractional
    reflectance values, close to zero, is prone to). The original paper uses
    ``n = 1000``. `epsilon` floors ``n * Rw`` at ``1 + epsilon`` so the log stays
    strictly positive even for zero or negative input reflectance.

    Citation: Stumpf, R. P., Holderied, K., & Sinclair, M. (2003). "Determination
    of water depth with high-resolution satellite imagery over variable bottom
    types." Limnology and Oceanography, 48(1), 547-556, eq. (1).
    """

    _ = depth
    if not isinstance(scene.data, dict):
        raise TypeError("Stumpf correction expects mapping-based scene data.")
    blue = scene.data[blue_band]
    green = scene.data[green_band]
    floor = 1.0 + epsilon
    stumpf = _map_binary(
        blue,
        green,
        lambda b, g: m1 * (math.log(max(n * b, floor)) / math.log(max(n * g, floor))) - m0,
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
    deep_water_reflectance: Mapping[str, float] | float | None = 0.0,
    attenuation: Mapping[str, float] | float | None = 0.1,
    **_: Any,
) -> Scene:
    """Maritorena, Morel & Gentili (1994) two-flow shallow-water reflectance model.

    The forward model for water-leaving reflectance ``Rw`` over a finite-depth
    bottom is ``Rw(depth) = Rw_inf + (Rb - Rw_inf) * exp(-2 * K_d * depth)``, where
    ``Rw_inf`` is the reflectance of optically-deep water (same band, no bottom
    contribution), ``Rb`` is the bottom albedo/reflectance, ``K_d`` is the diffuse
    attenuation coefficient, and the factor of 2 accounts for the two-way (down-
    and up-welling) light path. Unlike the Lyzenga/Stumpf transforms in this
    module, `depth` here is a *known* input rather than a quantity being
    estimated, so this correction inverts the forward model to recover bottom
    reflectance from measured water-leaving reflectance:
    ``Rb = Rw_inf + (Rw_measured - Rw_inf) / exp(-2 * K_d * depth)``.

    `deep_water_reflectance` supplies ``Rw_inf`` (per-band mapping or scalar,
    default ``0.0``). `attenuation` supplies ``K_d`` (per-band mapping or scalar,
    default ``0.1``), in units matched to `depth`'s units.

    This model is also the basis of later semi-analytical shallow-water
    inversions (e.g. Lee et al.).

    Citation: Maritorena, S., Morel, A., & Gentili, B. (1994). "Diffuse
    reflectance of oceanic shallow waters: influence of water depth and bottom
    albedo." Limnology and Oceanography, 39(7), 1689-1703.
    """

    if not isinstance(scene.data, dict):
        raise TypeError("Maritorena correction expects mapping-based scene data.")
    if depth is None:
        raise ValueError("Maritorena correction requires `depth`.")
    rw_inf = _band_param(scene.data.keys(), deep_water_reflectance, 0.0)
    k_d = _band_param(scene.data.keys(), attenuation, 0.1)
    updated = {}
    for band, values in scene.data.items():
        band_rw_inf = rw_inf[band]
        band_k_d = k_d[band]
        updated[band] = _map_binary(
            values,
            depth,
            lambda rw, d, rw_inf=band_rw_inf, kd=band_k_d: (
                rw_inf + (rw - rw_inf) / math.exp(-2.0 * kd * d)
            ),
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


register_depth_method("lyzenga1978", _lyzenga_1978)
register_depth_method("lyzenga1981", _lyzenga_1981)
register_depth_method("lyzenga2006", _lyzenga_2006)
register_depth_method("lyzenga", _lyzenga_2006)  # alias: most recent/popular variant
register_depth_method("stumpf", _stumpf_depth)
register_depth_method("maritorena", _maritorena_depth)

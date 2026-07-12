"""Depth-correction processing routines."""

from __future__ import annotations

import math
from typing import Any

from earthrs.processing.core import (
    _DEPTH_REGISTRY,
    _map_binary,
    _map_unary,
    _safe_log,
    register_depth_method,
)
from earthrs.scene import Scene


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


def _lyzenga_depth(
    scene: Scene,
    *,
    depth: Any | None = None,
    variant: str = "1981",
    epsilon: float = 1e-6,
    **_: Any,
) -> Scene:
    if not isinstance(scene.data, dict):
        raise TypeError("Lyzenga correction expects mapping-based scene data.")
    updated = {}
    for band, values in scene.data.items():
        if band in {"qa60", "scl", "qa_pixel"}:
            updated[band] = values
            continue
        corrected = _map_unary(values, lambda value: -math.log(max(value, epsilon)))
        if depth is not None:
            corrected = _map_binary(
                corrected, depth, lambda value, depth_value: value / max(depth_value, epsilon)
            )
        updated[band] = corrected
    metadata = dict(scene.metadata)
    metadata["depth_method"] = f"lyzenga_{variant}"
    return Scene(
        data=updated,
        crs=scene.crs,
        transform=scene.transform,
        metadata=metadata,
        band_names=scene.band_names,
        acquisition_time=scene.acquisition_time,
        sensor=scene.sensor,
        history=(*scene.history, f"depth_correct:lyzenga:{variant}"),
        masks=dict(scene.masks),
        cloud_mask=scene.cloud_mask,
        cloud_probability=scene.cloud_probability,
    )


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
    _ = depth
    if not isinstance(scene.data, dict):
        raise TypeError("Stumpf correction expects mapping-based scene data.")
    blue = scene.data[blue_band]
    green = scene.data[green_band]
    stumpf = _map_binary(
        blue,
        green,
        lambda b, g: m0 - m1 * (_safe_log(max(b, 1e-6)) / _safe_log(max(g, 1e-6))),
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

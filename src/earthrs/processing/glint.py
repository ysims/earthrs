"""Glint-removal processing routines."""

from __future__ import annotations

from typing import Any

from earthrs.processing.core import (
    _GLINT_REGISTRY,
    _flatten_numeric,
    _linear_slope,
    _map_binary,
    register_glint_method,
)
from earthrs.scene import Scene


def remove_glint(scene: Scene, *, method: str = "hedley", **kwargs: Any) -> Scene:
    """Remove surface sunglint from raster data.

    Method names are resolved from the glint registry.
    """

    processor = _GLINT_REGISTRY.get(method.lower())
    if processor is None:
        raise ValueError(f"Unknown glint-removal method '{method}'.")
    return processor(scene, **kwargs)


def _hedley_glint(
    scene: Scene,
    *,
    nir_band: str = "nir",
    visible_bands: list[str] | None = None,
    **_: Any,
) -> Scene:
    if not isinstance(scene.data, dict):
        raise TypeError("Hedley glint removal expects mapping-based scene data.")
    visible = visible_bands or [band for band in scene.bands if band != nir_band]
    nir = scene.data[nir_band]
    nir_values = _flatten_numeric(nir)
    nir_min = min(nir_values)
    updated_data = dict(scene.data)
    for band in visible:
        target = scene.data[band]
        slope = _linear_slope(_flatten_numeric(target), nir_values)
        updated_data[band] = _map_binary(
            target,
            nir,
            lambda v, n, slope=slope: max(0.0, v - slope * (n - nir_min)),
        )
    return Scene(
        data=updated_data,
        crs=scene.crs,
        transform=scene.transform,
        metadata=dict(scene.metadata),
        band_names=scene.band_names,
        acquisition_time=scene.acquisition_time,
        sensor=scene.sensor,
        history=(*scene.history, "remove_glint:hedley"),
        masks=dict(scene.masks),
        cloud_mask=scene.cloud_mask,
        cloud_probability=scene.cloud_probability,
    )


register_glint_method("hedley", _hedley_glint)

"""Shared processing infrastructure and registry helpers."""

from __future__ import annotations

import math
from typing import Any

from earthrs.scene import Scene

GlintProcessor = Any
DepthProcessor = Any
CloudProcessor = Any

_GLINT_REGISTRY: dict[str, GlintProcessor] = {}
_DEPTH_REGISTRY: dict[str, DepthProcessor] = {}
_CLOUD_REGISTRY: dict[str, CloudProcessor] = {}


def register_glint_method(name: str, func: GlintProcessor) -> None:
    _GLINT_REGISTRY[name.lower()] = func


def register_depth_method(name: str, func: DepthProcessor) -> None:
    _DEPTH_REGISTRY[name.lower()] = func


def register_cloud_method(name: str, func: CloudProcessor) -> None:
    _CLOUD_REGISTRY[name.lower()] = func


def register(data: Any, reference: Any, *, method: str = "default") -> Any:
    """Register raster data to a reference scene.

    Placeholder implementation keeps API stable until full registration support is added.
    """

    _ = method
    _ = reference
    return data


def reproject(data: Any, *, crs: Any, resampling: str = "nearest") -> Any:
    """Reproject raster data to a target coordinate reference system.

    Placeholder implementation keeps API stable until full reprojection support is added.
    """

    _ = crs
    _ = resampling
    return data


def _resolve_cloud_method(
    scene: Scene,
    *,
    method: str,
    qa_band: str | None,
    probability: Any | None,
) -> str:
    selected = method.lower()
    if selected != "auto":
        return selected
    sensor = (scene.sensor or "").lower()
    if "sentinel-2" in sensor:
        if qa_band == "scl" or "scl" in scene.data:
            return "sentinel2_scl"
        return "sentinel2_qa60"
    if "landsat" in sensor:
        return "landsat_qa_pixel"
    if probability is not None or scene.cloud_probability is not None:
        return "probability"
    return "user_mask"


def _resolve_data_band(scene: Scene, band_name: str) -> Any:
    if isinstance(scene.data, dict) and band_name in scene.data:
        return scene.data[band_name]
    raise ValueError(f"Band '{band_name}' is required for cloud masking.")


def _update_cloud(scene: Scene, *, mask: Any, probability: Any | None = None) -> Scene:
    masks = dict(scene.masks)
    masks["cloud"] = mask
    if probability is not None:
        masks["cloud_probability"] = probability
    metadata = dict(scene.metadata)
    metadata["cloud_mask"] = mask
    if probability is not None:
        metadata["cloud_probability"] = probability
    return Scene(
        data=scene.data,
        crs=scene.crs,
        transform=scene.transform,
        metadata=metadata,
        band_names=scene.band_names,
        acquisition_time=scene.acquisition_time,
        sensor=scene.sensor,
        history=(*scene.history, "cloud_mask"),
        masks=masks,
        cloud_mask=mask,
        cloud_probability=probability if probability is not None else scene.cloud_probability,
    )


def _map_unary(values: Any, func) -> Any:
    if isinstance(values, (list, tuple)):
        return [_map_unary(value, func) for value in values]
    return func(float(values))


def _map_binary(left: Any, right: Any, func) -> Any:
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return [
            _map_binary(left_value, right_value, func)
            for left_value, right_value in zip(left, right, strict=True)
        ]
    return func(float(left), float(right))


def _safe_log(value: float) -> float:
    return math.log(max(value, 1e-6))


def _flatten_numeric(values: Any) -> list[float]:
    if isinstance(values, (list, tuple)):
        out: list[float] = []
        for value in values:
            out.extend(_flatten_numeric(value))
        return out
    return [float(values)]


def _linear_slope(x_values: list[float], y_values: list[float]) -> float:
    if len(x_values) != len(y_values):
        raise ValueError("Input arrays must have the same size.")
    mean_x = sum(x_values) / len(x_values)
    mean_y = sum(y_values) / len(y_values)
    pairs = zip(x_values, y_values, strict=True)
    numerator = sum((x_val - mean_x) * (y_val - mean_y) for x_val, y_val in pairs)
    denominator = sum((x_val - mean_x) ** 2 for x_val in x_values)
    if denominator == 0:
        return 0.0
    return numerator / denominator


# Re-exported for earthrs.processing's public API; imported here (not at top) since
# these modules import shared helpers from core, which would otherwise be circular.
from earthrs.processing.atmospheric import atmospheric_correction  # noqa: E402, F401
from earthrs.processing.cloud import cloud_mask  # noqa: E402, F401
from earthrs.processing.depth import depth_correct  # noqa: E402, F401
from earthrs.processing.glint import remove_glint  # noqa: E402, F401

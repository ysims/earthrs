"""Cloud-masking processing routines."""

from __future__ import annotations

from typing import Any

from earthrs.processing.core import (
    _CLOUD_REGISTRY,
    _map_binary,
    _map_unary,
    _resolve_cloud_method,
    _resolve_data_band,
    _update_cloud,
    register_cloud_method,
)
from earthrs.scene import Scene


def cloud_mask(
    scene: Scene,
    *,
    method: str = "auto",
    qa_band: str | None = None,
    threshold: float | None = None,
    user_mask: Any | None = None,
    probability: Any | None = None,
) -> Scene:
    """Generate or apply a cloud mask.

    Method names are resolved from the cloud registry.
    """

    selected_method = _resolve_cloud_method(
        scene, method=method, qa_band=qa_band, probability=probability
    )
    processor = _CLOUD_REGISTRY.get(selected_method)
    if processor is None:
        raise ValueError(f"Unknown cloud-masking method '{selected_method}'.")
    return processor(
        scene,
        qa_band=qa_band,
        threshold=threshold,
        user_mask=user_mask,
        probability=probability,
    )


def _sentinel2_qa60_cloud(scene: Scene, *, qa_band: str | None = None, **kwargs: Any) -> Scene:
    band_name = qa_band or "qa60"
    qa_values = _resolve_data_band(scene, band_name)
    mask = _map_unary(
        qa_values, lambda value: bool(int(value) & (1 << 10) or int(value) & (1 << 11))
    )
    return _update_cloud(scene, mask=mask, probability=kwargs.get("probability"))


def _sentinel2_scl_cloud(scene: Scene, *, qa_band: str | None = None, **kwargs: Any) -> Scene:
    band_name = qa_band or "scl"
    scl_values = _resolve_data_band(scene, band_name)
    cloudy_classes = {3, 8, 9, 10, 11}
    mask = _map_unary(scl_values, lambda value: int(value) in cloudy_classes)
    return _update_cloud(scene, mask=mask, probability=kwargs.get("probability"))


def _landsat_qa_pixel_cloud(scene: Scene, *, qa_band: str | None = None, **kwargs: Any) -> Scene:
    band_name = qa_band or "qa_pixel"
    qa_values = _resolve_data_band(scene, band_name)
    mask = _map_unary(
        qa_values,
        lambda value: bool(int(value) & (1 << 3) or int(value) & (1 << 4) or int(value) & (1 << 2)),
    )
    return _update_cloud(scene, mask=mask, probability=kwargs.get("probability"))


def _probability_cloud(
    scene: Scene,
    *,
    threshold: float | None = None,
    probability: Any | None = None,
    user_mask: Any | None = None,
    **_: Any,
) -> Scene:
    probability_values = probability if probability is not None else scene.cloud_probability
    if probability_values is None:
        raise ValueError("Probability-based cloud masking requires cloud probability input.")
    threshold_value = 0.5 if threshold is None else float(threshold)
    mask = _map_unary(probability_values, lambda value: float(value) >= threshold_value)
    if user_mask is not None:
        mask = _map_binary(mask, user_mask, lambda left, right: bool(left) or bool(right))
    return _update_cloud(scene, mask=mask, probability=probability_values)


def _user_cloud(
    scene: Scene,
    *,
    user_mask: Any | None = None,
    probability: Any | None = None,
    **_: Any,
) -> Scene:
    if user_mask is None:
        raise ValueError("User-mask cloud method requires `user_mask`.")
    return _update_cloud(scene, mask=user_mask, probability=probability)


register_cloud_method("sentinel2_qa60", _sentinel2_qa60_cloud)
register_cloud_method("sentinel2_scl", _sentinel2_scl_cloud)
register_cloud_method("landsat_qa_pixel", _landsat_qa_pixel_cloud)
register_cloud_method("probability", _probability_cloud)
register_cloud_method("user_mask", _user_cloud)

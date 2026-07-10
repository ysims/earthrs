"""Functional preprocessing API placeholders."""

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


def remove_glint(scene: Scene, *, method: str = "hedley", **kwargs: Any) -> Scene:
    """Remove surface sunglint from raster data.

    Method names are resolved from the glint registry.
    """

    processor = _GLINT_REGISTRY.get(method.lower())
    if processor is None:
        raise ValueError(f"Unknown glint-removal method '{method}'.")
    return processor(scene, **kwargs)


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


def atmospheric_correction(scene: Scene, *, method: str = "default", **kwargs: Any) -> Scene:
    """Apply atmospheric correction to reflectance inputs.

    The function is registry-ready but atmospheric correction is intentionally deferred.
    """

    raise NotImplementedError(
        "Atmospheric correction is intentionally deferred. "
        "Register a backend and implement in a follow-up change."
    )


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


register_glint_method("hedley", _hedley_glint)
register_depth_method("lyzenga", _lyzenga_depth)
register_depth_method("stumpf", _stumpf_depth)
register_depth_method("maritorena", _maritorena_depth)
register_cloud_method("sentinel2_qa60", _sentinel2_qa60_cloud)
register_cloud_method("sentinel2_scl", _sentinel2_scl_cloud)
register_cloud_method("landsat_qa_pixel", _landsat_qa_pixel_cloud)
register_cloud_method("probability", _probability_cloud)
register_cloud_method("user_mask", _user_cloud)

"""Shared processing infrastructure and registry helpers."""

from __future__ import annotations

import math
from collections.abc import Callable
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


_RESAMPLING_METHODS = frozenset({"nearest", "bilinear"})

Transform = tuple[float, float, float, float, float, float]
CoordTransformer = Callable[[float, float], tuple[float, float]]


def reproject(
    scene: Scene,
    *,
    transform: Transform,
    shape: tuple[int, int],
    crs: Any = None,
    resampling: str = "nearest",
    transformer: CoordTransformer | None = None,
) -> Scene:
    """Resample a scene onto a new pixel grid, optionally changing CRS.

    ``transform`` and ``shape`` describe the destination grid: ``transform`` is an
    ``(a, b, c, d, e, f)`` affine coefficient tuple (``x = a*col + b*row + c``,
    ``y = d*col + e*row + f``), and ``shape`` is ``(rows, cols)``. Each band in
    ``scene.data`` must be a 2D grid (a list of rows).

    When ``crs`` differs from ``scene.crs``, a ``transformer`` callable mapping
    ``(x, y)`` in ``scene.crs`` to ``(x, y)`` in ``crs`` must be supplied (for
    example backed by ``pyproj``) — earthrs has no bundled CRS database, in
    keeping with its light-dependency policy.
    """

    if scene.transform is None:
        raise ValueError("`scene.transform` is required to reproject.")
    target_crs = scene.crs if crs is None else crs
    if transformer is None and scene.crs is not None and target_crs != scene.crs:
        raise ValueError(
            "Reprojecting between different CRS requires a `transformer` callable "
            "(for example one backed by pyproj); earthrs does not bundle a CRS database."
        )
    return _resample_scene(
        scene,
        dst_transform=transform,
        dst_shape=shape,
        crs=target_crs,
        resampling=resampling,
        transformer=transformer,
        history_entry=f"reproject:{resampling}",
    )


def register(scene: Scene, reference: Scene, *, method: str = "nearest") -> Scene:
    """Align ``scene`` onto ``reference``'s pixel grid.

    Resamples ``scene`` so its transform, shape, and CRS match ``reference``. CRS
    reconciliation follows the same rules as :func:`reproject`: if ``scene.crs``
    and ``reference.crs`` differ, reproject with an explicit ``transformer`` first.
    """

    if reference.transform is None:
        raise ValueError("`reference.transform` is required to register a scene.")
    if scene.crs is not None and reference.crs is not None and scene.crs != reference.crs:
        raise ValueError(
            "`register` cannot reconcile differing CRS without a transformer; "
            "call `reproject` with an explicit `transformer` first."
        )
    return _resample_scene(
        scene,
        dst_transform=reference.transform,
        dst_shape=_scene_shape(reference),
        crs=reference.crs,
        resampling=method,
        transformer=None,
        history_entry=f"register:{method}",
    )


def _resample_scene(
    scene: Scene,
    *,
    dst_transform: Transform,
    dst_shape: tuple[int, int],
    crs: Any,
    resampling: str,
    transformer: CoordTransformer | None,
    history_entry: str,
) -> Scene:
    if resampling not in _RESAMPLING_METHODS:
        raise ValueError(f"Unsupported resampling method '{resampling}'.")
    if not isinstance(scene.data, dict):
        raise TypeError("Reprojection expects mapping-based scene data.")
    src_transform = scene.transform
    if src_transform is None:
        raise ValueError("`scene.transform` is required to resample a scene.")

    updated_data = {
        band: _resample_grid(grid, src_transform, dst_transform, dst_shape, resampling, transformer)
        for band, grid in scene.data.items()
    }

    updated_mask = None
    if scene.cloud_mask is not None:
        updated_mask = _resample_grid(
            scene.cloud_mask, src_transform, dst_transform, dst_shape, "nearest", transformer
        )
    updated_probability = None
    if scene.cloud_probability is not None:
        updated_probability = _resample_grid(
            scene.cloud_probability,
            src_transform,
            dst_transform,
            dst_shape,
            resampling,
            transformer,
        )

    masks = dict(scene.masks)
    metadata = dict(scene.metadata)
    if updated_mask is not None:
        masks["cloud"] = updated_mask
        metadata["cloud_mask"] = updated_mask
    if updated_probability is not None:
        masks["cloud_probability"] = updated_probability
        metadata["cloud_probability"] = updated_probability

    return Scene(
        data=updated_data,
        crs=crs,
        transform=dst_transform,
        metadata=metadata,
        band_names=scene.band_names,
        acquisition_time=scene.acquisition_time,
        sensor=scene.sensor,
        history=(*scene.history, history_entry),
        masks=masks,
        cloud_mask=updated_mask if updated_mask is not None else scene.cloud_mask,
        cloud_probability=(
            updated_probability if updated_probability is not None else scene.cloud_probability
        ),
    )


def _scene_shape(scene: Scene) -> tuple[int, int]:
    if not isinstance(scene.data, dict) or not scene.data:
        raise ValueError("Cannot infer grid shape from a scene without band data.")
    first_band = next(iter(scene.data.values()))
    if not isinstance(first_band, list) or not first_band or not isinstance(first_band[0], list):
        raise ValueError("Reprojection expects each band as a 2D grid (a list of rows).")
    return len(first_band), len(first_band[0])


def _resample_grid(
    grid: Any,
    src_transform: Transform,
    dst_transform: Transform,
    dst_shape: tuple[int, int],
    resampling: str,
    transformer: CoordTransformer | None,
) -> list[list[Any]]:
    if not isinstance(grid, list) or not grid or not isinstance(grid[0], list):
        raise TypeError("Reprojection expects each band as a 2D grid (a list of rows).")
    src_rows = len(grid)
    src_cols = len(grid[0])
    dst_rows, dst_cols = dst_shape

    result: list[list[Any]] = []
    for dst_row in range(dst_rows):
        row_values: list[Any] = []
        for dst_col in range(dst_cols):
            x, y = _pixel_to_world(dst_col, dst_row, dst_transform)
            if transformer is not None:
                x, y = transformer(x, y)
            src_col, src_row = _world_to_pixel(x, y, src_transform)
            if resampling == "nearest":
                value = _sample_nearest(grid, src_row, src_col, src_rows, src_cols)
            else:
                value = _sample_bilinear(grid, src_row, src_col, src_rows, src_cols)
            row_values.append(value)
        result.append(row_values)
    return result


def _pixel_to_world(col: float, row: float, transform: Transform) -> tuple[float, float]:
    a, b, c, d, e, f = transform
    return a * col + b * row + c, d * col + e * row + f


def _world_to_pixel(x: float, y: float, transform: Transform) -> tuple[float, float]:
    a, b, c, d, e, f = transform
    det = a * e - b * d
    if det == 0:
        raise ValueError("Transform is not invertible.")
    col = (e * (x - c) - b * (y - f)) / det
    row = (-d * (x - c) + a * (y - f)) / det
    return col, row


def _sample_nearest(grid: list[list[Any]], row: float, col: float, rows: int, cols: int) -> Any:
    nearest_row = round(row)
    nearest_col = round(col)
    if 0 <= nearest_row < rows and 0 <= nearest_col < cols:
        return grid[nearest_row][nearest_col]
    return None


def _sample_bilinear(grid: list[list[Any]], row: float, col: float, rows: int, cols: int) -> Any:
    row0 = math.floor(row)
    col0 = math.floor(col)
    row1 = row0 + 1
    col1 = col0 + 1
    if row0 < 0 or col0 < 0 or row1 >= rows or col1 >= cols:
        return None
    frac_row = row - row0
    frac_col = col - col0
    top = grid[row0][col0] * (1 - frac_col) + grid[row0][col1] * frac_col
    bottom = grid[row1][col0] * (1 - frac_col) + grid[row1][col1] * frac_col
    return top * (1 - frac_row) + bottom * frac_row


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

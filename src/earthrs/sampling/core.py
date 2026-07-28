"""Sampling API placeholders for point, polygon, and transect extraction."""

from __future__ import annotations

from typing import Any

from earthrs.samples import Samples
from earthrs.scene import Scene


def sample_points(scene: Scene, points: Any, *, method: str = "nearest") -> Samples:
    """Extract raster values at point locations.

    Returns one row per point observation.
    """

    _validate_method(method, {"nearest", "bilinear"})
    point_rows = _coerce_points(points)
    rows: list[dict[str, Any]] = []
    for index, point in enumerate(point_rows):
        row = {
            "observation_id": index,
            "row": point["row"],
            "col": point["col"],
            "sensor": scene.sensor,
            "acquisition_time": scene.acquisition_time,
        }
        for band_name in scene.bands:
            band_data = _get_band(scene, band_name)
            row[band_name] = _read_cell(band_data, point["row"], point["col"])
        rows.append(row)
    return Samples(
        rows=tuple(rows), metadata={"sampling_method": method, "sampling_target": "points"}
    )


def sample_polygons(scene: Scene, polygons: Any, *, reducer: str = "mean") -> Samples:
    """Extract aggregated raster values over polygon features.

    Returns one row per polygon observation.
    """

    _validate_method(reducer, {"mean", "median", "min", "max"})
    polygon_rows = _coerce_polygons(polygons)
    rows: list[dict[str, Any]] = []
    for index, polygon in enumerate(polygon_rows):
        row: dict[str, Any] = {
            "observation_id": index,
            "sensor": scene.sensor,
            "acquisition_time": scene.acquisition_time,
        }
        for band_name in scene.bands:
            band_data = _get_band(scene, band_name)
            pixels = [
                _read_cell(band_data, row_col[0], row_col[1]) for row_col in polygon["pixels"]
            ]
            row[band_name] = _reduce_values(pixels, reducer)
        rows.append(row)
    return Samples(
        rows=tuple(rows), metadata={"sampling_method": reducer, "sampling_target": "polygons"}
    )


def sample_transects(scene: Scene, transects: Any, *, spacing: float | None = None) -> Samples:
    """Extract raster profiles along transects.

    Returns one row per sampled transect vertex.
    """

    transect_rows = _coerce_transects(transects)
    rows: list[dict[str, Any]] = []
    observation_index = 0
    for transect_index, transect in enumerate(transect_rows):
        for vertex_index, row_col in enumerate(transect["vertices"]):
            row: dict[str, Any] = {
                "observation_id": observation_index,
                "transect_id": transect_index,
                "vertex_id": vertex_index,
                "row": row_col[0],
                "col": row_col[1],
                "sensor": scene.sensor,
                "acquisition_time": scene.acquisition_time,
            }
            for band_name in scene.bands:
                band_data = _get_band(scene, band_name)
                row[band_name] = _read_cell(band_data, row_col[0], row_col[1])
            rows.append(row)
            observation_index += 1
    return Samples(
        rows=tuple(rows),
        metadata={"sampling_target": "transects", "spacing": spacing, "sampling_method": "nearest"},
    )


def _infer_sampling_target(data: Any) -> str:
    """Infer sampling target type from common geometry metadata."""

    geom_type = getattr(data, "geom_type", None)
    if isinstance(geom_type, str):
        value = geom_type.lower()
        if "point" in value:
            return "points"
        if "polygon" in value:
            return "polygons"
        if "line" in value:
            return "transects"

    raise ValueError(
        "Unable to infer sampling target. Set `target` to one of "
        "'points', 'polygons', or 'transects'."
    )


def sample(scene: Scene, data: Any, *, target: str | None = None, **kwargs: Any) -> Any:
    """Dispatch to a specific sampling helper based on target type.

    Parameters
    ----------
    scene:
        Scene to sample from.
    data:
        Geometry-like input (points, polygons, or transects).
    target:
        Explicit sampling target type. If omitted, inferred from ``data`` when possible.
    **kwargs:
        Forwarded to the selected helper function.
    """

    selected_target = target or _infer_sampling_target(data)
    if selected_target == "points":
        return sample_points(scene, data, **kwargs)
    if selected_target == "polygons":
        return sample_polygons(scene, data, **kwargs)
    if selected_target == "transects":
        return sample_transects(scene, data, **kwargs)

    raise ValueError("`target` must be one of 'points', 'polygons', or 'transects'.")


def _validate_method(method: str, supported: set[str]) -> None:
    if method not in supported:
        options = ", ".join(sorted(supported))
        raise ValueError(f"Unsupported method '{method}'. Expected one of: {options}.")


def _coerce_points(points: Any) -> list[dict[str, int]]:
    if isinstance(points, dict):
        points = [points]
    elif not isinstance(points, (list, tuple)):
        points = [points]
    out = []
    for point in points:
        if isinstance(point, dict):
            row = int(point.get("row", point.get("y", 0)))
            col = int(point.get("col", point.get("x", 0)))
        else:
            row = int(getattr(point, "row", getattr(point, "y", 0)))
            col = int(getattr(point, "col", getattr(point, "x", 0)))
        out.append({"row": row, "col": col})
    return out


def _coerce_polygons(polygons: Any) -> list[dict[str, list[tuple[int, int]]]]:
    if isinstance(polygons, dict):
        polygons = [polygons]
    elif not isinstance(polygons, (list, tuple)):
        polygons = [polygons]
    out = []
    for polygon in polygons:
        if isinstance(polygon, dict):
            pixels = polygon.get("pixels")
        else:
            pixels = getattr(polygon, "pixels", None)
        if pixels is None:
            raise ValueError("Polygon entries must include a `pixels` sequence.")
        out.append({"pixels": [(int(row), int(col)) for row, col in pixels]})
    return out


def _coerce_transects(transects: Any) -> list[dict[str, list[tuple[int, int]]]]:
    if isinstance(transects, dict):
        transects = [transects]
    elif not isinstance(transects, (list, tuple)):
        transects = [transects]
    out = []
    for transect in transects:
        if isinstance(transect, dict):
            vertices = transect.get("vertices")
        else:
            vertices = getattr(transect, "vertices", None)
        if vertices is None:
            raise ValueError("Transect entries must include a `vertices` sequence.")
        out.append({"vertices": [(int(row), int(col)) for row, col in vertices]})
    return out


def _get_band(scene: Scene, band_name: str) -> Any:
    if isinstance(scene.data, dict):
        if band_name not in scene.data:
            raise ValueError(f"Band '{band_name}' not found in scene data.")
        return scene.data[band_name]
    raise TypeError("Sampling currently expects scene data as a mapping of band names to arrays.")


def _read_cell(data: Any, row: int, col: int) -> Any:
    return data[row][col]


def _reduce_values(values: list[Any], reducer: str) -> float:
    numeric = [float(value) for value in values]
    if reducer == "mean":
        return sum(numeric) / len(numeric)
    if reducer == "median":
        ordered = sorted(numeric)
        midpoint = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[midpoint]
        return (ordered[midpoint - 1] + ordered[midpoint]) / 2
    if reducer == "min":
        return min(numeric)
    if reducer == "max":
        return max(numeric)
    raise ValueError(f"Unsupported reducer '{reducer}'.")

"""Sampling API placeholders for point, polygon, and transect extraction."""

from __future__ import annotations

from typing import Any

from earthrs.scene import Scene


def sample_points(scene: Scene, points: Any, *, method: str = "nearest") -> Any:
    """Extract raster values at point locations.

    TODO: Implement robust point sampling with CRS checks and metadata handling.
    """

    raise NotImplementedError("TODO: implement point sampling")


def sample_polygons(scene: Scene, polygons: Any, *, reducer: str = "mean") -> Any:
    """Extract aggregated raster values over polygon features.

    TODO: Implement polygon sampling and zonal-statistics reducers.
    """

    raise NotImplementedError("TODO: implement polygon sampling")


def sample_transects(scene: Scene, transects: Any, *, spacing: float | None = None) -> Any:
    """Extract raster profiles along transects.

    TODO: Implement transect interpolation and profile extraction.
    """

    raise NotImplementedError("TODO: implement transect sampling")


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

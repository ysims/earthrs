"""Sampling API placeholders for point, polygon, and transect extraction."""

from __future__ import annotations

from typing import Any

from aquaticrs.scene import Scene


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

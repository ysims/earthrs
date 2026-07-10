"""Functional preprocessing API placeholders."""

from __future__ import annotations

from typing import Any


def remove_glint(data: Any, *, method: str = "default") -> Any:
    """Remove surface sunglint from raster data.

    TODO: Implement glint-removal algorithms with pluggable methods.
    """

    raise NotImplementedError("TODO: implement remove_glint")


def depth_correct(data: Any, *, depth: Any | None = None) -> Any:
    """Apply depth correction to raster data.

    TODO: Implement depth correction for shallow-water retrieval workflows.
    """

    raise NotImplementedError("TODO: implement depth_correct")


def cloud_mask(data: Any, *, qa_band: str | None = None, threshold: float | None = None) -> Any:
    """Generate or apply a cloud mask.

    TODO: Implement cloud masking with sensor-specific quality bands.
    """

    raise NotImplementedError("TODO: implement cloud_mask")


def atmospheric_correction(data: Any, *, method: str = "default") -> Any:
    """Apply atmospheric correction to reflectance inputs.

    TODO: Integrate atmospheric-correction backends while preserving metadata.
    """

    raise NotImplementedError("TODO: implement atmospheric_correction")


def register(data: Any, reference: Any, *, method: str = "default") -> Any:
    """Register raster data to a reference scene.

    TODO: Implement georegistration pipelines.
    """

    raise NotImplementedError("TODO: implement register")


def reproject(data: Any, *, crs: Any, resampling: str = "nearest") -> Any:
    """Reproject raster data to a target coordinate reference system.

    TODO: Implement reprojection using raster backends with metadata preservation.
    """

    raise NotImplementedError("TODO: implement reproject")

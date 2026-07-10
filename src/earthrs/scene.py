"""Scene abstraction for single raster acquisitions."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from types import MappingProxyType
from typing import Any

from earthrs.io import normalise_cloud_schema


@dataclass(frozen=True, slots=True)
class Scene:
    """Represent a single Earth observation raster scene.

    Parameters
    ----------
    data:
        Raster data, typically an xarray ``DataArray`` or ``Dataset``.
    crs:
        Coordinate reference system for the scene.
    transform:
        Affine transform describing pixel-to-coordinate mapping.
    metadata:
        Additional scene metadata.
    band_names:
        Ordered names of bands present in ``data``.
    acquisition_time:
        Timestamp representing data acquisition time.
    sensor:
        Sensor or platform name (for example ``"Sentinel-2"``).
    history:
        Processing history entries for reproducibility.
    """

    data: Any
    crs: Any | None = None
    transform: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    band_names: tuple[str, ...] = ()
    acquisition_time: datetime | None = None
    sensor: str | None = None
    history: tuple[str, ...] = ()
    masks: dict[str, Any] = field(default_factory=dict)
    cloud_mask: Any | None = None
    cloud_probability: Any | None = None

    def __post_init__(self) -> None:
        mask_dict, cloud_mask, cloud_probability = normalise_cloud_schema(
            metadata=self.metadata,
            masks=self.masks,
            cloud_mask=self.cloud_mask,
            cloud_probability=self.cloud_probability,
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "band_names", tuple(self.band_names))
        object.__setattr__(self, "history", tuple(self.history))
        object.__setattr__(self, "masks", MappingProxyType(mask_dict))
        object.__setattr__(self, "cloud_mask", cloud_mask)
        object.__setattr__(self, "cloud_probability", cloud_probability)

    @property
    def bands(self) -> tuple[str, ...]:
        """Return scene band names as an immutable tuple for inspection."""

        return self.band_names

    def add_history(self, entry: str) -> Scene:
        """Append a processing-history entry.

        Parameters
        ----------
        entry:
            Human-readable description of a processing step.
        """

        return replace(self, history=(*self.history, entry))

    def summary(self) -> str:
        """Return a compact human-readable description of the scene."""

        return (
            f"Scene(sensor={self.sensor!r}, acquisition_time={self.acquisition_time!r}, "
            f"bands={list(self.bands)!r})"
        )

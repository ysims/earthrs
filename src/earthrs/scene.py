"""Scene abstraction for single raster acquisitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
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
    band_names: list[str] = field(default_factory=list)
    acquisition_time: datetime | None = None
    sensor: str | None = None
    history: list[str] = field(default_factory=list)

    @property
    def bands(self) -> tuple[str, ...]:
        """Return scene band names as an immutable tuple for inspection."""

        return tuple(self.band_names)

    def add_history(self, entry: str) -> None:
        """Append a processing-history entry.

        Parameters
        ----------
        entry:
            Human-readable description of a processing step.
        """

        self.history.append(entry)

    def summary(self) -> str:
        """Return a compact human-readable description of the scene."""

        return (
            f"Scene(sensor={self.sensor!r}, acquisition_time={self.acquisition_time!r}, "
            f"bands={list(self.bands)!r})"
        )

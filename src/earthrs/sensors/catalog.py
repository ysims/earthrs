"""Sensor catalogue: band metadata for supported platforms.

Provides lightweight, dependency-free descriptions of the spectral bands
carried by each supported sensor (centre wavelength, bandwidth, and ground
resolution), plus :func:`resolve_sensor` for matching a free-text sensor
string (as found on ``Scene.sensor``) to the appropriate catalogue entry.

TODO: Expand with scene metadata parsers and download adapters.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class Band:
    """A single spectral band's metadata.

    Parameters
    ----------
    name:
        Band identifier (for example ``"B4"`` or ``"Red"``).
    center_wavelength_nm:
        Centre wavelength of the band, in nanometres.
    bandwidth_nm:
        Full bandwidth of the band, in nanometres, if published. ``None``
        when the manufacturer does not document a bandwidth figure.
    resolution_m:
        Native ground sample distance, in metres.
    """

    name: str
    center_wavelength_nm: float
    bandwidth_nm: float | None
    resolution_m: float


@dataclass(slots=True)
class Sentinel2:
    """Sentinel-2 MSI sensor metadata."""

    name: str = "Sentinel-2"
    bands: tuple[Band, ...] = (
        Band("B1", 443.0, None, 60.0),
        Band("B2", 490.0, None, 10.0),
        Band("B3", 560.0, None, 10.0),
        Band("B4", 665.0, None, 10.0),
        Band("B5", 705.0, None, 20.0),
        Band("B6", 740.0, None, 20.0),
        Band("B7", 783.0, None, 20.0),
        Band("B8", 842.0, None, 10.0),
        Band("B8A", 865.0, None, 20.0),
        Band("B9", 945.0, None, 60.0),
        Band("B10", 1375.0, None, 60.0),
        Band("B11", 1610.0, None, 20.0),
        Band("B12", 2190.0, None, 20.0),
    )


@dataclass(slots=True)
class Landsat:
    """Landsat 8/9 OLI sensor metadata."""

    name: str = "Landsat"
    bands: tuple[Band, ...] = (
        Band("B1", 443.0, None, 30.0),
        Band("B2", 482.0, None, 30.0),
        Band("B3", 562.0, None, 30.0),
        Band("B4", 655.0, None, 30.0),
        Band("B5", 865.0, None, 30.0),
        Band("B6", 1610.0, None, 30.0),
        Band("B7", 2200.0, None, 30.0),
        Band("B8", 590.0, None, 15.0),
        Band("B9", 1370.0, None, 30.0),
    )


@dataclass(slots=True)
class PlanetScope:
    """PlanetScope (8-band SuperDove) sensor metadata."""

    name: str = "PlanetScope"
    bands: tuple[Band, ...] = (
        Band("Coastal Blue", 443.0, None, 3.0),
        Band("Blue", 490.0, None, 3.0),
        Band("Green I", 531.0, None, 3.0),
        Band("Green", 565.0, None, 3.0),
        Band("Yellow", 610.0, None, 3.0),
        Band("Red", 665.0, None, 3.0),
        Band("Red Edge", 705.0, None, 3.0),
        Band("NIR", 865.0, None, 3.0),
    )


def resolve_sensor(name: str | None) -> Sentinel2 | Landsat | PlanetScope | None:
    """Match a free-text sensor string to a catalogue entry.

    Performs case-insensitive substring matching similar to the sensor
    dispatch in ``processing.core._resolve_cloud_method``, so variants like
    ``"Sentinel-2A"``, ``"Landsat 8"``, ``"Landsat-9"``, and ``"PlanetScope"``
    all resolve correctly. Returns ``None`` for unrecognised or missing
    sensor names rather than raising, since auto-dispatch must be able to
    fall back gracefully when the sensor is unknown.
    """

    if not name:
        return None
    lowered = name.lower()
    if "sentinel-2" in lowered or "sentinel2" in lowered:
        return Sentinel2()
    if "landsat" in lowered:
        return Landsat()
    if "planetscope" in lowered or "planet scope" in lowered:
        return PlanetScope()
    return None

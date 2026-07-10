"""Placeholder sensor classes.

TODO: Expand each class with band definitions, wavelengths, resolution metadata, scene metadata
parsers, and download adapters.
"""

from dataclasses import dataclass


@dataclass(slots=True)
class Sentinel2:
    """Placeholder for Sentinel-2 sensor support."""

    name: str = "Sentinel-2"


@dataclass(slots=True)
class Landsat:
    """Placeholder for Landsat sensor support."""

    name: str = "Landsat"


@dataclass(slots=True)
class PlanetScope:
    """Placeholder for PlanetScope sensor support."""

    name: str = "PlanetScope"

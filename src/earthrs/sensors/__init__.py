"""Sensor definitions and metadata adapters."""

from earthrs.sensors.catalog import Band, Landsat, PlanetScope, Sentinel2, resolve_sensor

__all__ = ["Sentinel2", "Landsat", "PlanetScope", "Band", "resolve_sensor"]

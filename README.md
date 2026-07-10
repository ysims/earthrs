# aquaticrs

AquaticRS is an early-stage Python library for aquatic and coastal Earth observation.

## Vision

The project aims to provide a clean, composable interface for remote-sensing workflows while
building on the existing Python ecosystem (for example NumPy, xarray, Rasterio, and GeoPandas)
instead of replacing it.

## Current status

This repository currently contains an initial scaffold with placeholder APIs for:

- scene-level abstractions (`Scene`)
- collections of scenes (`Dataset`)
- sensor definitions
- processing operations
- spectral indices
- sampling utilities
- machine-learning helpers

Algorithms are intentionally left unimplemented while the architecture is established.

## Development

```bash
pip install -e .[dev]
pytest
ruff check .
```

## Documentation

```bash
pip install -e .[docs]
mkdocs serve
```

# Project status

EarthRS is early-stage. This page tracks what's implemented, what's known-limited, and what's
scaffolded but not yet built — see [Getting started](getting-started.md) and
[Processing](processing.md) for what's already usable.

## Known limitations

- Sampling (`earthrs.sampling`) and spectral indices (`earthrs.indices`) currently only support
  `Scene.data` as a mapping of band name to nested list/array data, not the xarray/rasterio-backed
  data the `Scene` docstring describes as the typical case.
- Spectral indices treat only Python `None` as a missing value; there is no nodata/fill-value
  (for example `-9999`) handling yet.
- `depth_correct(..., method="lyzenga", variant=...)` records the requested variant in scene
  metadata/history but does not yet apply distinct 1978/1981/2006 formulas.

## Scaffolded, not yet implemented

- sensor definitions (`earthrs.sensors`) beyond placeholder name-only classes
- machine-learning helpers (`earthrs.ml`)
- visualisation helpers (`earthrs.plotting`) — currently an empty package

# Project status

EarthRS is early-stage. This page tracks what's implemented, what's known-limited, and what's
scaffolded but not yet built — see [Getting started](getting-started.md) and
[Processing](processing.md) for what's already usable.

## Known limitations

- Sampling (`earthrs.sampling`) and spectral indices (`earthrs.indices`) support `Scene.data` as
  a `dict[str, list]` of band name to nested list data (the always-available, zero-dependency
  default), an `xarray.DataArray`/`Dataset`, or a rasterio dataset handle — see
  [Core concepts](concepts.md#scenedata-backing-stores) for the exact conventions and how to
  install optional support.
- `xarray.DataArray` scene data must use a `band` dimension whose coordinate values match
  `scene.band_names`; other dimension-ordering or naming conventions (for example a `variable`
  dimension, or bands stored as separate leading axes without coordinates) are not recognised
  and raise `TypeError`.
- Spectral indices treat only Python `None` as a missing value; there is no nodata/fill-value
  (for example `-9999`) handling yet. This applies uniformly across dict, xarray, and rasterio
  backed data — no backend currently reads a dataset's own nodata value.

## Scaffolded, not yet implemented

- sensor definitions (`earthrs.sensors`) beyond placeholder name-only classes
- machine-learning helpers (`earthrs.ml`)
- visualisation helpers (`earthrs.plotting`) — currently an empty package

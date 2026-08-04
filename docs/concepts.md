# Core concepts

## Core data model

`earthrs.scene.Scene`, `earthrs.dataset.Dataset`, and `earthrs.samples.Samples` are immutable
(frozen, slotted) dataclasses:

- **`Scene`** — a single immutable raster acquisition, with a normalised cloud mask/probability
  schema regardless of source product.
- **`Dataset`** — an immutable collection of `Scene` objects, with `filter_dates`,
  `filter_clouds`, and `sample_points`.
- **`Samples`** — the canonical immutable feature-table type: an immutable tuple of row mappings
  plus metadata, with one row per observation. `Dataset.sample_points` and the functions in
  `earthrs.sampling` all return `Samples`.

Methods that "change" one of these objects (`Scene.add_history`, `Dataset.add_scene`,
`Dataset.filter_dates`, `Dataset.filter_clouds`, `Samples.with_metadata`, and every processing
function) return a new instance and leave the original untouched:

```python
from earthrs import Dataset, Scene

scene = Scene(data={"nir": [0.4], "red": [0.1]})
updated = scene.add_history("imported")
assert scene.history == ()          # original is unchanged
assert updated.history == ("imported",)
```

## Cloud metadata normalisation

Cloud fields can arrive under different keys depending on the source product (for example
`cloud`, `cloud_mask`, or `mask_cloud` in a `masks` dict; `cloud_mask` or `cloud_prob` in a
`metadata` dict). `Scene.__post_init__` runs `earthrs.io.normalise_cloud_schema` so that,
regardless of which field an importer populated, `scene.cloud_mask`, `scene.cloud_probability`,
`scene.masks["cloud"]`, and `scene.metadata["cloud_mask"]` all agree. Downstream algorithms
(cloud masking, cloud filtering) read from this normalised schema rather than product-specific
field names.

## `Scene.data` backing stores

`earthrs` has zero *required* runtime dependencies — `Scene.data` as a `dict[str, list]` mapping
band names to nested lists (or any list-of-lists-like structure) is the always-available default
and is what every test in this project exercises:

```python
from earthrs import Scene

scene = Scene(data={"nir": [[0.8, 0.4]], "red": [[0.2, 0.4]]}, band_names=["nir", "red"])
```

`earthrs.indices` and `earthrs.sampling` additionally recognise two optional, geospatial-ecosystem
backing stores, guarded behind `try`/`except ImportError` so importing `earthrs` never requires
either package:

- **`xarray.DataArray`** — must have a `band` dimension whose coordinate values are the band
  names, in the same order as `scene.band_names`. Band access is `data.sel(band=band_name)`.
- **`xarray.Dataset`** — one data variable per band; band access is `data[band_name]`, which
  returns a 2D `DataArray`.
- **A rasterio dataset handle** — anything exposing `.read(band_index)` with 1-indexed band
  numbers (for example an open `rasterio.io.DatasetReader`). Band names map to rasterio band
  indices positionally: `scene.band_names[i]` corresponds to rasterio band `i + 1`.

Passing anything else raises a `TypeError` with a message listing the supported shapes.

Install optional support for the xarray/rasterio-backed paths with:

```bash
pip install "earthrs[geo]"
```

This is a deliberate project-wide policy: xarray, rasterio, and similar packages are optional
integrations that *improve* behaviour (native array types, richer metadata, lazy/on-disk reads)
without ever being required — the zero-dependency `dict`-of-list path keeps working identically
whether or not the `geo` extra is installed. See [Project status](status.md) for known
limitations of the xarray/rasterio paths (for example, dimension-ordering conventions that are
not yet recognised).

## Spectral indices

`earthrs.indices` provides `ndvi`, `ndwi`, and `evi`, each taking a `Scene` and returning
per-pixel index values computed from its named bands.

## Sampling

`earthrs.sampling` provides `sample_points`, `sample_polygons`, and `sample_transects` (plus a
`sample()` dispatcher that infers the target from geometry type), all returning `Samples` — one
row per point, polygon, or transect vertex observation. `Dataset.sample_points` delegates to
`earthrs.sampling.sample_points` per scene and aggregates the results into a single `Samples`.

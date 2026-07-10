# EarthRS

EarthRS is an early-stage Python library for Earth observation, including aquatic and coastal
remote sensing workflows.

## Vision

The project aims to provide a clean, composable interface for remote-sensing workflows while
building on the existing Python ecosystem (for example NumPy, xarray, Rasterio, and GeoPandas)
instead of replacing it.

## Current status

### Core data model

`Scene`, `Dataset`, and `Samples` are immutable (frozen, slotted) dataclasses. Methods
that "change" one of these objects (`Scene.add_history`, `Dataset.add_scene`,
`Dataset.filter_dates`, `Dataset.filter_clouds`, `Samples.with_metadata`, and every
processing function) return a **new** instance and leave the original untouched:

```python
from earthrs import Dataset, Scene

scene = Scene(data={"nir": [0.4], "red": [0.1]})
updated = scene.add_history("imported")
assert scene.history == ()          # original is unchanged
assert updated.history == ("imported",)
```

`Samples` is the canonical feature-table type: an immutable tuple of row mappings plus
metadata, with one row per observation. `Dataset.sample_points` and the functions in
`earthrs.sampling` all return `Samples`.

### Cloud metadata normalisation

Cloud fields can arrive under different keys depending on the source product (for
example `cloud`, `cloud_mask`, or `mask_cloud` in a `masks` dict; `cloud_mask` or
`cloud_prob` in a `metadata` dict). `Scene.__post_init__` runs
`earthrs.io.normalise_cloud_schema` so that, regardless of which field an importer
populated, `scene.cloud_mask`, `scene.cloud_probability`, `scene.masks["cloud"]`, and
`scene.metadata["cloud_mask"]` all agree. Downstream algorithms (cloud masking, cloud
filtering) read from this normalised schema rather than product-specific field names.

### Processing: registries and sensor-aware dispatch

`earthrs.processing` implements `remove_glint`, `depth_correct`, and `cloud_mask` as
registry-based dispatchers — each resolves a `method=` string against a registry of
implementations that can be extended without modifying the dispatcher itself:

```python
from earthrs.processing.core import register_glint_method

def my_glint_method(scene, **kwargs):
    ...
    return updated_scene

register_glint_method("my_method", my_glint_method)
```

Implemented methods today:

- **Glint removal**: `"hedley"` (Hedley et al., 2005).
- **Depth correction**: `"lyzenga"` (Lyzenga-style log transform; the `variant`
  keyword accepts `"1978"`/`"1981"`/`"2006"` but currently applies the same
  transform for all three — see Known limitations), `"stumpf"` (ratio transform),
  `"maritorena"` (single-parameter exponential attenuation).
- **Cloud masking**: `"sentinel2_qa60"`, `"sentinel2_scl"`, `"landsat_qa_pixel"`,
  `"probability"` (threshold on a cloud-probability layer), `"user_mask"` (pass
  through a caller-supplied mask). `cloud_mask(scene, method="auto")` resolves one of
  these automatically from `scene.sensor` and the data available on the scene.

`atmospheric_correction` is **intentionally deferred** — it raises
`NotImplementedError` until a backend is implemented in a follow-up change. Unlike
glint/depth/cloud processing, it does not yet have its own registry.

### Known limitations

- Sampling (`earthrs.sampling`) and spectral indices (`earthrs.indices`) currently
  only support `Scene.data` as a mapping of band name to nested list/array data, not
  the xarray/rasterio-backed data the `Scene` docstring describes as the typical case.
- Spectral indices treat only Python `None` as a missing value; there is no
  nodata/fill-value (for example `-9999`) handling yet.
- `depth_correct(..., method="lyzenga", variant=...)` records the requested variant in
  scene metadata/history but does not yet apply distinct 1978/1981/2006 formulas.

### Scaffolded, not yet implemented

- sensor definitions (`earthrs.sensors`) beyond placeholder name-only classes
- machine-learning helpers (`earthrs.ml`)
- `register`/`reproject` raster-registration and reprojection helpers

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

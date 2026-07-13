# Processing

## Registries and sensor-aware dispatch

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

## Implemented methods

- **Glint removal**: `"hedley"` (Hedley et al., 2005).
- **Depth correction**: `"lyzenga"` (Lyzenga-style log transform; the `variant` keyword accepts
  `"1978"`/`"1981"`/`"2006"` but currently applies the same transform for all three — see
  [Project status](status.md)), `"stumpf"` (ratio transform), `"maritorena"` (single-parameter
  exponential attenuation).
- **Cloud masking**: `"sentinel2_qa60"`, `"sentinel2_scl"`, `"landsat_qa_pixel"`, `"probability"`
  (threshold on a cloud-probability layer), `"user_mask"` (pass through a caller-supplied mask).
  `cloud_mask(scene, method="auto")` resolves one of these automatically from `scene.sensor` and
  the data available on the scene.

`atmospheric_correction` is **intentionally deferred** — it raises `NotImplementedError` until a
backend is implemented in a follow-up change. Unlike glint/depth/cloud processing, it does not
yet have its own registry.

## Reprojection and registration

`earthrs.processing.reproject` resamples a scene onto a new pixel grid — described by an
`(a, b, c, d, e, f)` affine `transform` and a `(rows, cols)` `shape` — using nearest-neighbour or
bilinear resampling. `earthrs.processing.register` resamples a scene onto a reference scene's
grid (transform, shape, and CRS). Both operate on 2D grid band data (a list of rows), matching
the layout `earthrs.sampling` already assumes — see [Project status](status.md) for how this
differs from the flat/nested-list layout accepted elsewhere in `earthrs.processing`.

Neither function bundles a CRS database: if the destination `crs` differs from `scene.crs`, you
must supply a `transformer` callable mapping `(x, y)` in `scene.crs` to `(x, y)` in the
destination CRS (for example backed by `pyproj`), otherwise a `ValueError` is raised. 

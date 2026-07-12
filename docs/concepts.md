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

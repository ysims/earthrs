# EarthRS

EarthRS provides a consistent set of Earth observation abstractions, including aquatic and
coastal remote sensing use cases.

## Principles

- Clean APIs
- Readability and composability
- Extensibility
- Reproducibility
- Optional integrations and dependencies

## Core abstractions

- `earthrs.scene.Scene` — a single immutable raster acquisition, with a normalised
  cloud mask/probability schema regardless of source product.
- `earthrs.dataset.Dataset` — an immutable collection of `Scene` objects, with
  `filter_dates`, `filter_clouds`, and `sample_points`.
- `earthrs.samples.Samples` — the canonical immutable feature-table type (one row per
  observation) returned by sampling and dataset methods.

All three are immutable: methods that change state return a new instance rather than
mutating in place.

## Processing: registries and extension

`earthrs.processing` exposes `remove_glint`, `depth_correct`, and `cloud_mask` as
registry-based dispatchers. New algorithms can be added without touching the
dispatcher by calling `register_glint_method`, `register_depth_method`, or
`register_cloud_method` with a lowercase method name and a
`(scene, **kwargs) -> Scene` callable. `cloud_mask(..., method="auto")` additionally
resolves a method automatically from `scene.sensor`. `atmospheric_correction` is
intentionally deferred (raises `NotImplementedError`) until a backend lands.

## Other modules

- `earthrs.indices` — ndvi, ndwi, evi
- `earthrs.sensors` — sensor definitions (placeholder)
- `earthrs.sampling` — point, polygon, and transect sampling, all returning `Samples`
- `earthrs.ml` — machine-learning helpers (placeholder)

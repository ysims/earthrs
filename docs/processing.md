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
- **Depth correction**: `"lyzenga1978"`, `"lyzenga1981"`, `"lyzenga2006"` (the default `"lyzenga"` is an alias for `"lyzenga2006"`), `"stumpf"` (ratio
  transform, Stumpf et al., 2003), `"maritorena"` (single-parameter exponential attenuation).
- **Cloud masking**: `"sentinel2_qa60"`, `"sentinel2_scl"`, `"landsat_qa_pixel"`, `"probability"`
  (threshold on a cloud-probability layer), `"user_mask"` (pass through a caller-supplied mask).
  `cloud_mask(scene, method="auto")` resolves one of these automatically from `scene.sensor` and
  the data available on the scene.

`atmospheric_correction` is **intentionally deferred** — it raises `NotImplementedError` until a
backend is implemented in a follow-up change. Unlike glint/depth/cloud processing, it does not
yet have its own registry.

## Lyzenga depth-correction variants

There are three Lyzenga algorithms for depth correction - `"lyzenga1978"`, `"lyzenga1981"`,
`"lyzenga2006"`. `method="lyzenga"` is an
alias for `"lyzenga2006"`, the most recent and most widely used variant.

### `"lyzenga1978"` — per-band log-linearising transform

The single-band transform from Lyzenga (1978), eqs. (1) & (7). Replaces every spectral band with

```text
X_i = ln(L_i - L_si)
```

which is approximately linear in water depth. Non-spectral bands (`qa60`, `scl`, `qa_pixel`) pass
through unchanged. Bands are not combined with each other, so this variant does not correct for
bottom-reflectance variation — it's the foundational linearisation the other two variants build on.

```python
depth_correct(scene, method="lyzenga1978", deep_water_radiance={"blue": 0.02})
```

- `deep_water_radiance` (mapping or scalar, default `0.0` per band) — the deep-water radiance
  `L_si` to subtract before taking the log.
- `epsilon` (default `1e-6`) — floor applied to `L_i - L_si` before the log, to avoid `log(0)`
  or `log(negative)`.

### `"lyzenga1981"` — two-band depth-invariant bottom index

The classic two-band index from Lyzenga (1981), eq. (2). Adds a `lyzenga_dii` band:

```text
Y_i = [K_j * X_i - K_i * X_j] / sqrt(K_i^2 + K_j^2)
```

where `X = ln(L - L_s)` and `K_i`/`K_j` are the water-attenuation coefficients for the two chosen
bands. The result is (ideally) independent of water depth, so it's a bottom-type index rather than
a depth estimate — it removes depth dependence so bottom features can be compared across varying
water depth.

```python
depth_correct(
    scene,
    method="lyzenga1981",
    band_i="blue",
    band_j="green",
    k_i=0.056,
    k_j=0.074,
)
```

- `band_i`, `band_j` (default `"blue"`, `"green"`) — the two bands to combine.
- `k_i`, `k_j` (**required**) — water-attenuation coefficients for `band_i`/`band_j`. Raises
  `ValueError` if either is missing, or if both are zero (undefined).
- `deep_water_radiance`, `epsilon` — as in `"lyzenga1978"`.

### `"lyzenga2006"` — empirical multi-band depth regression

The regression-based estimator from Lyzenga et al. (2006), eq. (9). Adds a `lyzenga_depth` band
containing an actual depth estimate (in the same units as the calibration data, typically metres):

```text
depth = h_0 - sum(h_j * X_j)
```

where `X_j = ln(L_j - L_sj)` and `h_j` are empirically fitted coefficients (`h_0` is the
intercept). Unlike the 1978/1981 variants, this produces a direct depth estimate rather than a
depth-invariant index — the coefficients are normally obtained by regressing against ground-truth
depth measurements (for example LIDAR) for the site being processed.

```python
depth_correct(
    scene,
    method="lyzenga2006",  # or method="lyzenga" — it's an alias for this variant
    bands=("blue", "green"),
    coefficients={"blue": -17.42, "green": 26.7},
    intercept=17.84,
)
```

- `bands` (default `("blue", "green")`) — the bands to combine; must be non-empty.
- `coefficients` (**required**) — mapping of band name to its fitted regression weight `h_j`.
  Raises `ValueError` if missing, or if any band in `bands` has no entry.
- `intercept` (default `0.0`) — the `h_0` term.
- `deep_water_radiance`, `epsilon` — as in `"lyzenga1978"`.

None of the three variants take a known `depth` as input — the whole point is to derive depth (or
a depth-invariant index) from radiance, unlike `"maritorena"` which requires one.

# EarthRS

EarthRS is an early-stage Python library for Earth observation, including aquatic and coastal remote sensing workflows. It provides a clean, composable interface for remote-sensing workflows while building on the existing Python ecosystem (for example NumPy, xarray, Rasterio, and GeoPandas).

## Example

```python
from earthrs import Scene

scene = Scene(data={"nir": [0.4], "red": [0.1]})
updated = scene.add_history("imported")

assert scene.history == ()          # original is unchanged
assert updated.history == ("imported",)
```

`Scene`, `Dataset`, and `Samples` are immutable: methods that "change" one of them return a new instance rather than mutating in place.

## Documentation

Full documentation, including the core data model, processing registries, and current project status, lives at [ysims.github.io/earthrs](https://ysims.github.io/earthrs).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to set up a development environment and submit changes.
# Getting started

## Install

```bash
pip install earthrs
```

## A first example

`Scene`, `Dataset`, and `Samples` are immutable (frozen, slotted) dataclasses. Methods that "change" one of these objects return a **new** instance and leave the original untouched:

```python
from earthrs import Scene

scene = Scene(data={"nir": [0.4], "red": [0.1]})
updated = scene.add_history("imported")

assert scene.history == ()          # original is unchanged
assert updated.history == ("imported",)
```

See [Core concepts](concepts.md) for the rest of the data model, and
[Processing](processing.md) for glint removal, depth correction, and cloud masking.

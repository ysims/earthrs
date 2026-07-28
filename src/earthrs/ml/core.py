"""Machine-learning utility functions bridging earthrs data with ML frameworks.

These helpers convert `Scene`/`Samples` (earthrs's zero-dependency data
abstractions) into the array/tensor/dataset types expected by numpy, PyTorch,
and TensorFlow. All three frameworks are optional: importing this module never
requires them, and each function that needs one only imports it when called,
raising a clear `ImportError` (pointing at the relevant extras group) if it is
missing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from earthrs.samples import Samples
from earthrs.scene import Scene


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised via importorskip in tests
        raise ImportError(
            "numpy is required for this operation. Install it with `pip install earthrs[ml]`."
        ) from exc
    return np


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised via importorskip in tests
        raise ImportError(
            "torch is required for this operation. "
            "Install it with `pip install earthrs[ml-torch]`."
        ) from exc
    return torch


def _require_tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - exercised via importorskip in tests
        raise ImportError(
            "tensorflow is required for this operation. "
            "Install it with `pip install earthrs[ml-tensorflow]`."
        ) from exc
    return tf


def _build_array(raw: Any, np: Any) -> Any:
    """Build a numpy array, casting to `float` where possible.

    Falls back to `dtype=object` (preserving the original Python values,
    including non-numeric ones such as strings) when a `float` cast fails,
    so `to_numpy` never silently drops or corrupts data.
    """

    try:
        return np.array(raw, dtype=float)
    except (TypeError, ValueError):
        return np.array(raw, dtype=object)


def _rows_to_numpy(rows: Iterable[Mapping[str, Any]], np: Any) -> tuple[Any, tuple[str, ...]]:
    rows = list(rows)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    raw = [[row.get(col) for col in columns] for row in rows]
    array = _build_array(raw, np)
    if array.ndim == 1:
        # Only possible when `rows` (and therefore `columns`) is empty.
        array = array.reshape(len(rows), len(columns))
    return array, tuple(columns)


def _columns_to_numpy(data: Mapping[str, Any], np: Any) -> tuple[Any, tuple[str, ...]]:
    columns = tuple(data.keys())
    column_values = [list(data[col]) for col in columns]
    lengths = {len(values) for values in column_values}
    if len(lengths) > 1:
        raise ValueError("All columns must have the same length.")
    n_rows = lengths.pop() if lengths else 0
    raw = [
        [column_values[col_index][row_index] for col_index in range(len(columns))]
        for row_index in range(n_rows)
    ]
    array = _build_array(raw, np)
    if array.ndim == 1:
        array = array.reshape(n_rows, len(columns))
    return array, columns


def to_numpy(data: Any) -> Any:
    """Convert samples-like or array-like data into NumPy array(s).

    - A `Samples` instance, a list/tuple of row mappings, or a column-oriented
      mapping (``{column_name: values}``) is treated as tabular data and
      converted to a 2D ``(n_rows, n_columns)`` array, returned as
      ``(array, column_names)``.
    - Anything else (a flat or nested sequence, or an existing numpy array) is
      converted directly and returned as a bare array (no column names).

    Values are cast to `float` where possible; if any value cannot be cast
    (for example string categories), the array falls back to `dtype=object` so
    no data is silently lost or corrupted. Missing values (`None`) become
    `float('nan')` whenever the resulting array is numeric.

    Raises
    ------
    ImportError
        If numpy is not installed. Install it with `pip install earthrs[ml]`.
    """

    np = _require_numpy()

    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, Samples):
        return _rows_to_numpy(data.rows, np)
    if isinstance(data, Mapping):
        return _columns_to_numpy(data, np)
    if isinstance(data, (list, tuple)):
        if data and isinstance(data[0], Mapping):
            return _rows_to_numpy(data, np)
        return _build_array(list(data), np)
    if isinstance(data, Iterable):
        return _build_array(list(data), np)
    return np.asarray(data)


def _na_row_mask(array: Any, np: Any) -> Any:
    """Boolean mask (one entry per row, or per element for 1D arrays) of missing data.

    "Missing" means a `None` entry (the convention this codebase already uses
    for missing/out-of-bounds raster values, see `earthrs.processing.core`) or,
    for numeric arrays, a float `NaN`.
    """

    is_float = np.issubdtype(array.dtype, np.floating)
    if array.ndim == 1:
        if is_float:
            return np.isnan(array)
        return np.array([value is None for value in array])
    if is_float:
        return np.isnan(array).any(axis=1)
    return np.array([any(value is None for value in row) for row in array])


def prepare_training_data(samples: Any, labels: Any, *, drop_na: bool = True) -> tuple[Any, Any]:
    """Split sampled EO data into aligned `(X, y)` arrays ready for model training.

    Parameters
    ----------
    samples:
        A `Samples` instance (the canonical earthrs feature table).
    labels:
        Either the name of a column already present in `samples` (removed
        from the feature matrix and used as `y`), or a separate
        sequence/array of label values aligned by row index with
        `samples.rows`.
    drop_na:
        When `True` (the default), rows where *any* feature value or the
        label is missing (`None`, or `NaN` once converted to a numeric
        array) are dropped from both `X` and `y`, keeping them aligned by
        row. When `False`, missing values pass through unchanged (as `NaN`
        in numeric arrays, `None` in object arrays).

    Returns
    -------
    tuple
        `(X, y)` numpy arrays: `X` has shape `(n_rows, n_features)`, `y` has
        shape `(n_rows,)`.
    """

    np = _require_numpy()
    if not isinstance(samples, Samples):
        raise TypeError("`samples` must be a `Samples` instance.")

    array, columns = to_numpy(samples)

    if isinstance(labels, str):
        if labels not in columns:
            raise ValueError(f"Label column {labels!r} not found in samples columns {columns}.")
        label_index = columns.index(labels)
        feature_indices = [index for index in range(len(columns)) if index != label_index]
        X = array[:, feature_indices]
        y = array[:, label_index]
    else:
        label_values = list(labels)
        if len(label_values) != len(samples.rows):
            raise ValueError(
                f"`labels` length ({len(label_values)}) does not match the number "
                f"of samples ({len(samples.rows)})."
            )
        X = array
        y = to_numpy(label_values)

    if drop_na:
        feature_na = _na_row_mask(X, np) if X.shape[1] else np.zeros(X.shape[0], dtype=bool)
        label_na = _na_row_mask(y, np)
        keep = ~(feature_na | label_na)
        X = X[keep]
        y = y[keep]

    return X, y


def _to_numeric_array(data: Any) -> Any:
    """Resolve `to_numpy`'s output to a bare numeric array, rejecting object dtype."""

    array = to_numpy(data)
    if isinstance(array, tuple):
        array = array[0]
    if array.dtype == object:
        raise TypeError(
            "Cannot convert non-numeric data to a tensor; ensure all values are numeric."
        )
    return array


def to_torch_dataset(features: Any, labels: Any | None = None) -> Any:
    """Wrap features (and optional labels) in a `torch.utils.data.TensorDataset`.

    `features`/`labels` accept anything `to_numpy` understands (a `Samples`
    instance, a list of row mappings, a column-oriented mapping, or an
    array-like). Values are converted to `torch.float32` tensors, except
    integer/boolean arrays which keep their native dtype.

    Raises
    ------
    ImportError
        If torch is not installed. Install it with `pip install earthrs[ml-torch]`.
    """

    torch = _require_torch()
    feature_tensor = _to_tensor(features, torch)
    if labels is None:
        return torch.utils.data.TensorDataset(feature_tensor)
    label_tensor = _to_tensor(labels, torch)
    return torch.utils.data.TensorDataset(feature_tensor, label_tensor)


def _to_tensor(data: Any, torch: Any) -> Any:
    array = _to_numeric_array(data)
    if array.dtype.kind in "iub":
        return torch.as_tensor(array)
    return torch.as_tensor(array.astype("float32"))


def to_tensorflow_dataset(features: Any, labels: Any | None = None) -> Any:
    """Wrap features (and optional labels) in a `tf.data.Dataset`.

    Built with `tf.data.Dataset.from_tensor_slices`; see `to_torch_dataset`
    for the accepted input shapes.

    Raises
    ------
    ImportError
        If tensorflow is not installed. Install it with
        `pip install earthrs[ml-tensorflow]`.
    """

    tf = _require_tensorflow()
    feature_array = _to_numeric_array(features)
    if labels is None:
        return tf.data.Dataset.from_tensor_slices(feature_array)
    label_array = _to_numeric_array(labels)
    return tf.data.Dataset.from_tensor_slices((feature_array, label_array))


def _grid_shape(data: Mapping[str, Any]) -> tuple[int, int]:
    first = next(iter(data.values()))
    if not isinstance(first, list) or not first or not isinstance(first[0], list):
        raise TypeError("Tiling expects each band as a 2D grid (a list of rows).")
    return len(first), len(first[0])


def tile_scene(scene: Any, *, tile_size: int, stride: int | None = None) -> Any:
    """Split a `Scene` into `tile_size` x `tile_size` tiles.

    `scene.data` must be a `dict[str, list[list[float]]]` (band name -> 2D
    grid, a list of rows), the same zero-dependency convention used by
    `earthrs.processing.core`. `stride` defaults to `tile_size` (non-overlapping
    tiles); pass a smaller value for overlapping tiles.

    Edge handling: tiles are cropped, not padded. If the scene's height or
    width is not an exact multiple of `stride` (with a final full `tile_size`
    window), the trailing partial row/column of tiles is simply dropped
    rather than padded with fill values -- this keeps every returned tile a
    uniform `tile_size` x `tile_size`, which is what most model inputs expect,
    at the cost of not covering every pixel for non-divisible scenes.

    Each returned `Scene` preserves `crs`, `sensor`, `acquisition_time`, and
    `history` (with a `f"tile:{tile_size}"` entry appended), and records its
    pixel offset within the source scene as `metadata["tile_offset"] = (row,
    col)` so results can be reassembled later. If `scene.transform` is set, it
    is offset so each tile's own `(0, 0)` pixel still maps to the correct
    world coordinate. Masks (`scene.masks`, `cloud_mask`, `cloud_probability`)
    are not tiled -- they are considered scene-level metadata here, not band
    data -- and are left unset on the returned tiles.

    Returns
    -------
    list[Scene]
        Tiles in row-major order (top-to-bottom, left-to-right). Empty if the
        scene is smaller than `tile_size` in either dimension.
    """

    if tile_size <= 0:
        raise ValueError("`tile_size` must be a positive integer.")
    step = tile_size if stride is None else stride
    if step <= 0:
        raise ValueError("`stride` must be a positive integer.")
    if not isinstance(scene.data, dict) or not scene.data:
        raise TypeError("`tile_scene` expects mapping-based scene data with at least one band.")

    n_rows, n_cols = _grid_shape(scene.data)
    if n_rows < tile_size or n_cols < tile_size:
        return []

    tiles = []
    for row_offset in range(0, n_rows - tile_size + 1, step):
        for col_offset in range(0, n_cols - tile_size + 1, step):
            tile_data = {
                band: [
                    row[col_offset : col_offset + tile_size]
                    for row in grid[row_offset : row_offset + tile_size]
                ]
                for band, grid in scene.data.items()
            }

            tile_transform = None
            if scene.transform is not None:
                a, b, c, d, e, f = scene.transform
                new_c = a * col_offset + b * row_offset + c
                new_f = d * col_offset + e * row_offset + f
                tile_transform = (a, b, new_c, d, e, new_f)

            metadata = dict(scene.metadata)
            metadata["tile_offset"] = (row_offset, col_offset)

            tiles.append(
                Scene(
                    data=tile_data,
                    crs=scene.crs,
                    transform=tile_transform,
                    metadata=metadata,
                    band_names=scene.band_names,
                    acquisition_time=scene.acquisition_time,
                    sensor=scene.sensor,
                    history=(*scene.history, f"tile:{tile_size}"),
                )
            )
    return tiles


def _stack_bands(scene: Scene, band_order: tuple[str, ...], np: Any) -> Any:
    return np.stack([np.array(scene.data[band], dtype=float) for band in band_order], axis=0)


def raster_predict(
    model: Any,
    scene: Any,
    *,
    batch_size: int = 1,
    tile_size: int | None = None,
    stride: int | None = None,
) -> list[tuple[Any, Any]]:
    """Run model prediction over a raster `Scene`, tile by tile.

    The scene is split into tiles with `tile_scene`. `tile_size`/`stride` are
    forwarded to it; if `tile_size` is omitted it defaults to
    `min(n_rows, n_cols)`, i.e. the largest square tile that still fits
    entirely within the scene, so the whole raster is covered by one or more
    non-overlapping (or, if `stride` is given, overlapping) square tiles by
    default -- `tile_scene`'s crop-not-pad convention still applies to any
    dimension that doesn't divide evenly.

    Each tile's bands are stacked into a numpy array of shape
    `(n_bands, tile_size, tile_size)` (band order follows `scene.band_names`,
    falling back to the band dict's insertion order if `band_names` is
    empty). Tiles are grouped into batches of up to `batch_size`, stacked as
    `(batch, n_bands, tile_size, tile_size)`, and passed to
    `model.predict(batch)` if the model exposes a `predict` method, otherwise
    to `model(batch)`. The model's output for a batch is assumed to be
    indexable/sliceable in tile order (as numpy arrays, lists, and most
    framework tensors are).

    Returns
    -------
    list[tuple[Scene, Any]]
        One `(tile, prediction)` pair per input tile, in the same order as
        `tile_scene` produced them. Each tile's `metadata["tile_offset"]`
        gives its `(row, col)` pixel offset in the source scene, which a
        caller can use to stitch a full-scene prediction raster back
        together. Empty if the scene is smaller than the effective tile size.
    """

    if batch_size <= 0:
        raise ValueError("`batch_size` must be a positive integer.")
    np = _require_numpy()
    if not isinstance(scene.data, dict) or not scene.data:
        raise TypeError("`raster_predict` expects mapping-based scene data with at least one band.")

    n_rows, n_cols = _grid_shape(scene.data)
    effective_tile_size = tile_size if tile_size is not None else min(n_rows, n_cols)
    tiles = tile_scene(scene, tile_size=effective_tile_size, stride=stride)
    if not tiles:
        return []

    band_order = scene.band_names if scene.band_names else tuple(scene.data.keys())

    results: list[tuple[Any, Any]] = []
    for batch_start in range(0, len(tiles), batch_size):
        batch_tiles = tiles[batch_start : batch_start + batch_size]
        batch = np.stack([_stack_bands(tile, band_order, np) for tile in batch_tiles], axis=0)
        predict = getattr(model, "predict", None)
        prediction = predict(batch) if callable(predict) else model(batch)
        for index, tile in enumerate(batch_tiles):
            results.append((tile, prediction[index]))
    return results

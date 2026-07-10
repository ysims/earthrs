"""Placeholder machine-learning utility functions.

These helpers bridge remote-sensing data abstractions with general ML frameworks.
"""

from __future__ import annotations

from typing import Any


def prepare_training_data(samples: Any, labels: Any, *, drop_na: bool = True) -> tuple[Any, Any]:
    """Prepare sampled EO data for model training.

    TODO: Implement feature/label alignment and metadata-aware preprocessing.
    """

    raise NotImplementedError("TODO: implement training-data preparation")


def to_numpy(data: Any) -> Any:
    """Convert data structures to NumPy-compatible arrays.

    TODO: Implement conversion for xarray/pandas/geospatial containers.
    """

    raise NotImplementedError("TODO: implement numpy conversion")


def to_torch_dataset(features: Any, labels: Any | None = None) -> Any:
    """Convert feature data to a PyTorch-compatible dataset.

    TODO: Keep PyTorch dependency optional through adapter layers.
    """

    raise NotImplementedError("TODO: implement torch dataset conversion")


def to_tensorflow_dataset(features: Any, labels: Any | None = None) -> Any:
    """Convert feature data to a TensorFlow-compatible dataset.

    TODO: Keep TensorFlow dependency optional through adapter layers.
    """

    raise NotImplementedError("TODO: implement TensorFlow dataset conversion")


def tile_scene(scene: Any, *, tile_size: int, stride: int | None = None) -> Any:
    """Create model-friendly tiles from scene-like data.

    TODO: Implement tiling with overlap and metadata propagation.
    """

    raise NotImplementedError("TODO: implement scene tiling")


def raster_predict(model: Any, scene: Any, *, batch_size: int = 1) -> Any:
    """Run model prediction over raster-like inputs.

    TODO: Implement batched prediction over tiled or windowed raster inputs.
    """

    raise NotImplementedError("TODO: implement raster prediction")

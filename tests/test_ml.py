import math

import pytest

from earthrs.ml.core import (
    prepare_training_data,
    raster_predict,
    tile_scene,
    to_numpy,
    to_tensorflow_dataset,
    to_torch_dataset,
)
from earthrs.samples import Samples
from earthrs.scene import Scene


def _grid_scene(**kwargs) -> Scene:
    data = {
        "blue": [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        "green": [[16, 15, 14, 13], [12, 11, 10, 9], [8, 7, 6, 5], [4, 3, 2, 1]],
    }
    defaults = {"data": data, "band_names": ["blue", "green"], "sensor": "Sentinel-2"}
    defaults.update(kwargs)
    return Scene(**defaults)


# ---------------------------------------------------------------------------
# to_numpy / prepare_training_data (numpy-dependent)
# ---------------------------------------------------------------------------


def test_to_numpy_requires_numpy_with_clear_message(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "numpy":
            raise ImportError("no numpy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="earthrs\\[ml\\]"):
        to_numpy(Samples(rows=({"a": 1},)))


def test_to_numpy_converts_samples_to_array_and_columns() -> None:
    pytest.importorskip("numpy")

    samples = Samples(rows=({"a": 1, "b": 2}, {"a": 3, "b": 4}))

    array, columns = to_numpy(samples)

    assert columns == ("a", "b")
    assert array.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_to_numpy_handles_missing_keys_as_nan() -> None:
    pytest.importorskip("numpy")

    samples = Samples(rows=({"a": 1, "b": 2}, {"b": 3}))

    array, columns = to_numpy(samples)

    assert columns == ("a", "b")
    assert array[1, 0] != array[1, 0]  # NaN
    assert array[1, 1] == 3.0


def test_to_numpy_converts_list_of_row_mappings() -> None:
    pytest.importorskip("numpy")

    array, columns = to_numpy([{"x": 1, "y": 2}, {"x": 3, "y": 4}])

    assert columns == ("x", "y")
    assert array.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_to_numpy_converts_column_oriented_mapping() -> None:
    pytest.importorskip("numpy")

    array, columns = to_numpy({"x": [1, 2, 3], "y": [4, 5, 6]})

    assert columns == ("x", "y")
    assert array.tolist() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]


def test_to_numpy_column_mapping_rejects_mismatched_lengths() -> None:
    pytest.importorskip("numpy")

    with pytest.raises(ValueError):
        to_numpy({"x": [1, 2], "y": [1, 2, 3]})


def test_to_numpy_converts_plain_sequence_without_column_names() -> None:
    pytest.importorskip("numpy")

    result = to_numpy([1, 2, 3])

    assert result.tolist() == [1.0, 2.0, 3.0]


def test_to_numpy_falls_back_to_object_dtype_for_non_numeric_values() -> None:
    pytest.importorskip("numpy")

    array, columns = to_numpy([{"label": "reef"}, {"label": "sand"}])

    assert columns == ("label",)
    assert array.dtype == object
    assert array.tolist() == [["reef"], ["sand"]]


def test_prepare_training_data_splits_label_column() -> None:
    pytest.importorskip("numpy")

    samples = Samples(
        rows=(
            {"blue": 0.1, "green": 0.2, "depth": 1.0},
            {"blue": 0.3, "green": 0.4, "depth": 2.0},
        )
    )

    X, y = prepare_training_data(samples, "depth")

    assert X.tolist() == [[0.1, 0.2], [0.3, 0.4]]
    assert y.tolist() == [1.0, 2.0]


def test_prepare_training_data_accepts_separate_label_sequence() -> None:
    pytest.importorskip("numpy")

    samples = Samples(rows=({"blue": 0.1}, {"blue": 0.3}))

    X, y = prepare_training_data(samples, [1.0, 2.0])

    assert X.tolist() == [[0.1], [0.3]]
    assert y.tolist() == [1.0, 2.0]


def test_prepare_training_data_rejects_mismatched_label_length() -> None:
    pytest.importorskip("numpy")

    samples = Samples(rows=({"blue": 0.1}, {"blue": 0.3}))

    with pytest.raises(ValueError):
        prepare_training_data(samples, [1.0])


def test_prepare_training_data_rejects_unknown_label_column() -> None:
    pytest.importorskip("numpy")

    samples = Samples(rows=({"blue": 0.1},))

    with pytest.raises(ValueError):
        prepare_training_data(samples, "depth")


def test_prepare_training_data_drop_na_removes_rows_with_missing_features_or_labels() -> None:
    pytest.importorskip("numpy")

    samples = Samples(
        rows=(
            {"blue": 0.1, "depth": 1.0},
            {"blue": None, "depth": 2.0},
            {"blue": 0.3, "depth": None},
            {"blue": 0.4, "depth": 4.0},
        )
    )

    X, y = prepare_training_data(samples, "depth")

    assert X.tolist() == [[0.1], [0.4]]
    assert y.tolist() == [1.0, 4.0]


def test_prepare_training_data_keeps_na_rows_when_drop_na_is_false() -> None:
    pytest.importorskip("numpy")

    samples = Samples(rows=({"blue": 0.1, "depth": 1.0}, {"blue": None, "depth": 2.0}))

    X, y = prepare_training_data(samples, "depth", drop_na=False)

    assert len(X) == 2
    assert math.isnan(X[1, 0])


def test_prepare_training_data_requires_samples_instance() -> None:
    pytest.importorskip("numpy")

    with pytest.raises(TypeError):
        prepare_training_data([{"blue": 0.1}], "blue")


# ---------------------------------------------------------------------------
# to_torch_dataset (torch-dependent)
# ---------------------------------------------------------------------------


def test_to_torch_dataset_wraps_features_and_labels() -> None:
    torch = pytest.importorskip("torch")

    samples = Samples(rows=({"blue": 0.1, "green": 0.2}, {"blue": 0.3, "green": 0.4}))

    dataset = to_torch_dataset(samples, [1.0, 2.0])

    assert len(dataset) == 2
    features, label = dataset[0]
    assert isinstance(features, torch.Tensor)
    assert features.tolist() == pytest.approx([0.1, 0.2])
    assert label.item() == pytest.approx(1.0)


def test_to_torch_dataset_without_labels() -> None:
    pytest.importorskip("torch")

    dataset = to_torch_dataset([[1, 2], [3, 4]])

    assert len(dataset) == 2


# ---------------------------------------------------------------------------
# to_tensorflow_dataset (tensorflow-dependent)
# ---------------------------------------------------------------------------


def test_to_tensorflow_dataset_wraps_features_and_labels() -> None:
    pytest.importorskip("tensorflow")

    dataset = to_tensorflow_dataset([[1, 2], [3, 4]], [1, 0])

    elements = list(dataset.as_numpy_iterator())
    assert len(elements) == 2
    assert elements[0][0].tolist() == [1.0, 2.0]
    assert elements[0][1] == 1


# ---------------------------------------------------------------------------
# tile_scene (no external dependency)
# ---------------------------------------------------------------------------


def test_tile_scene_splits_grid_into_non_overlapping_tiles() -> None:
    scene = _grid_scene()

    tiles = tile_scene(scene, tile_size=2)

    assert len(tiles) == 4
    assert tiles[0].data["blue"] == [[1, 2], [5, 6]]
    assert tiles[1].data["blue"] == [[3, 4], [7, 8]]
    assert tiles[2].data["blue"] == [[9, 10], [13, 14]]
    assert tiles[3].data["blue"] == [[11, 12], [15, 16]]


def test_tile_scene_preserves_provenance_and_appends_history() -> None:
    scene = _grid_scene(crs="EPSG:4326", history=("import",))

    tiles = tile_scene(scene, tile_size=2)

    for tile in tiles:
        assert tile.crs == "EPSG:4326"
        assert tile.sensor == "Sentinel-2"
        assert tile.history == ("import", "tile:2")
        assert tile.band_names == ("blue", "green")


def test_tile_scene_records_pixel_offset_in_metadata() -> None:
    scene = _grid_scene()

    tiles = tile_scene(scene, tile_size=2)

    assert [tile.metadata["tile_offset"] for tile in tiles] == [
        (0, 0),
        (0, 2),
        (2, 0),
        (2, 2),
    ]


def test_tile_scene_offsets_transform_per_tile() -> None:
    scene = _grid_scene(transform=(1, 0, 0, 0, 1, 0))

    tiles = tile_scene(scene, tile_size=2)

    assert tiles[0].transform == (1, 0, 0, 0, 1, 0)
    assert tiles[1].transform == (1, 0, 2, 0, 1, 0)
    assert tiles[2].transform == (1, 0, 0, 0, 1, 2)
    assert tiles[3].transform == (1, 0, 2, 0, 1, 2)


def test_tile_scene_crops_partial_edge_tiles() -> None:
    scene = Scene(data={"blue": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}, band_names=["blue"])

    tiles = tile_scene(scene, tile_size=2)

    assert len(tiles) == 1
    assert tiles[0].data["blue"] == [[1, 2], [4, 5]]


def test_tile_scene_returns_empty_list_when_scene_smaller_than_tile_size() -> None:
    scene = Scene(data={"blue": [[1, 2], [3, 4]]}, band_names=["blue"])

    assert tile_scene(scene, tile_size=4) == []


def test_tile_scene_supports_overlapping_stride() -> None:
    scene = Scene(
        data={"blue": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]},
        band_names=["blue"],
    )

    tiles = tile_scene(scene, tile_size=2, stride=1)

    assert len(tiles) == 4
    assert tiles[0].data["blue"] == [[1, 2], [4, 5]]
    assert tiles[3].data["blue"] == [[5, 6], [8, 9]]


def test_tile_scene_rejects_non_mapping_data() -> None:
    scene = Scene(data="dummy")

    with pytest.raises(TypeError):
        tile_scene(scene, tile_size=2)


def test_tile_scene_rejects_non_positive_tile_size() -> None:
    scene = _grid_scene()

    with pytest.raises(ValueError):
        tile_scene(scene, tile_size=0)


# ---------------------------------------------------------------------------
# raster_predict (numpy-dependent)
# ---------------------------------------------------------------------------


class _SumModel:
    """A trivial model exposing `.predict`, summing all pixel values per tile."""

    def predict(self, batch):
        return [float(tile.sum()) for tile in batch]


def test_raster_predict_uses_model_predict_method_in_batches() -> None:
    pytest.importorskip("numpy")

    scene = _grid_scene()

    results = raster_predict(_SumModel(), scene, tile_size=2, batch_size=2)

    assert len(results) == 4
    tile, prediction = results[0]
    assert tile.data["blue"] == [[1, 2], [5, 6]]
    # blue tile [[1,2],[5,6]] + green tile [[16,15],[12,11]]
    assert prediction == pytest.approx(1 + 2 + 5 + 6 + 16 + 15 + 12 + 11)


def test_raster_predict_falls_back_to_callable_model() -> None:
    pytest.importorskip("numpy")

    scene = _grid_scene()

    def model(batch):
        return [float(tile.sum()) for tile in batch]

    results = raster_predict(model, scene, tile_size=2)

    assert len(results) == 4


def test_raster_predict_defaults_tile_size_to_smaller_scene_dimension() -> None:
    pytest.importorskip("numpy")

    scene = Scene(
        data={"blue": [[1, 2, 3], [4, 5, 6]]},
        band_names=["blue"],
    )

    results = raster_predict(_SumModel(), scene)

    # smaller dimension is 2 rows -> a single 2x2 tile, trailing column dropped
    assert len(results) == 1
    tile, _ = results[0]
    assert tile.data["blue"] == [[1, 2], [4, 5]]


def test_raster_predict_returns_empty_list_for_undersized_scene() -> None:
    pytest.importorskip("numpy")

    scene = Scene(data={"blue": [[1]]}, band_names=["blue"])

    assert raster_predict(_SumModel(), scene, tile_size=2) == []

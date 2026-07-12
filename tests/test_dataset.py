from datetime import datetime

import pytest

from earthrs.dataset import Dataset
from earthrs.scene import Scene


def test_dataset_add_and_len() -> None:
    dataset = Dataset()
    updated = dataset.add_scene(Scene(data="a"))

    assert len(updated) == 1
    assert len(dataset) == 0


def test_dataset_add_scene_returns_new_dataset_without_mutating_original() -> None:
    scene_a = Scene(data="a")
    scene_b = Scene(data="b")
    dataset = Dataset((scene_a,))

    updated = dataset.add_scene(scene_b)

    assert updated is not dataset
    assert updated.scenes == (scene_a, scene_b)
    assert dataset.scenes == (scene_a,)


def test_dataset_is_immutable() -> None:
    dataset = Dataset((Scene(data="a"),))

    with pytest.raises(AttributeError):
        dataset.scenes = ()  # type: ignore[misc]


def test_dataset_iteration_and_len() -> None:
    scenes = (Scene(data="a"), Scene(data="b"))
    dataset = Dataset(scenes)

    assert len(dataset) == 2
    assert list(dataset) == list(scenes)


def _scene_at(day: int, **kwargs) -> Scene:
    return Scene(data="dummy", acquisition_time=datetime(2024, 1, day), **kwargs)


def test_filter_dates_keeps_inclusive_bounds() -> None:
    dataset = Dataset((_scene_at(1), _scene_at(5), _scene_at(10)))

    filtered = dataset.filter_dates(start=datetime(2024, 1, 1), end=datetime(2024, 1, 5))

    assert [scene.acquisition_time.day for scene in filtered] == [1, 5]


def test_filter_dates_excludes_scenes_without_acquisition_time() -> None:
    dataset = Dataset((_scene_at(1), Scene(data="no_time")))

    filtered = dataset.filter_dates(start=datetime(2024, 1, 1))

    assert len(filtered) == 1


def test_filter_dates_supports_open_ended_bounds() -> None:
    dataset = Dataset((_scene_at(1), _scene_at(5), _scene_at(10)))

    assert [s.acquisition_time.day for s in dataset.filter_dates(start=datetime(2024, 1, 5))] == [
        5,
        10,
    ]
    assert [s.acquisition_time.day for s in dataset.filter_dates(end=datetime(2024, 1, 5))] == [
        1,
        5,
    ]


def test_filter_dates_rejects_start_after_end() -> None:
    dataset = Dataset((_scene_at(1),))

    with pytest.raises(ValueError):
        dataset.filter_dates(start=datetime(2024, 1, 10), end=datetime(2024, 1, 1))


def test_filter_dates_returns_new_dataset_without_mutating_original() -> None:
    scene = _scene_at(1)
    dataset = Dataset((scene,))

    filtered = dataset.filter_dates(start=datetime(2024, 1, 5))

    assert filtered is not dataset
    assert len(filtered) == 0
    assert len(dataset) == 1


def test_filter_clouds_rejects_invalid_fraction() -> None:
    dataset = Dataset()

    with pytest.raises(ValueError):
        dataset.filter_clouds(-0.1)
    with pytest.raises(ValueError):
        dataset.filter_clouds(1.1)
    with pytest.raises(ValueError):
        dataset.filter_clouds(float("nan"))


def test_filter_clouds_uses_cloud_fraction_metadata() -> None:
    dataset = Dataset(
        (
            Scene(data="a", metadata={"cloud_fraction": 0.1}),
            Scene(data="b", metadata={"cloud_fraction": 0.9}),
        )
    )

    filtered = dataset.filter_clouds(0.5)

    assert len(filtered) == 1
    assert filtered.scenes[0].metadata["cloud_fraction"] == 0.1


def test_filter_clouds_converts_percentage_cloud_cover() -> None:
    dataset = Dataset((Scene(data="a", metadata={"cloud_cover": 20}),))

    assert len(dataset.filter_clouds(0.5)) == 1
    assert len(dataset.filter_clouds(0.1)) == 0


def test_filter_clouds_falls_back_to_cloud_mask_ratio() -> None:
    dataset = Dataset((Scene(data="a", masks={"cloud": [0, 0, 1, 0]}),))

    assert len(dataset.filter_clouds(0.5)) == 1
    assert len(dataset.filter_clouds(0.1)) == 0


def test_filter_clouds_excludes_scenes_with_no_resolvable_cloud_info() -> None:
    dataset = Dataset((Scene(data="a"),))

    assert len(dataset.filter_clouds(1.0)) == 0


def test_dataset_sample_points_aggregates_rows_with_provenance() -> None:
    scene_a = Scene(
        data={"blue": [[1, 2], [3, 4]]},
        band_names=["blue"],
        sensor="Sentinel-2",
        acquisition_time=datetime(2024, 1, 1),
    )
    scene_b = Scene(
        data={"blue": [[5, 6], [7, 8]]},
        band_names=["blue"],
        sensor="Landsat",
        acquisition_time=datetime(2024, 1, 2),
    )
    dataset = Dataset((scene_a, scene_b))

    samples = dataset.sample_points({"row": 0, "col": 1})

    assert len(samples) == 2
    first, second = samples
    assert first["scene_index"] == 0
    assert first["sensor"] == "Sentinel-2"
    assert first["acquisition_time"] == datetime(2024, 1, 1)
    assert first["blue"] == 2
    assert second["scene_index"] == 1
    assert second["sensor"] == "Landsat"
    assert second["blue"] == 6
    assert samples.metadata["sampling_scope"] == "dataset"

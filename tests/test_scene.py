from datetime import datetime
from types import MappingProxyType

import pytest

from earthrs.scene import Scene


def test_scene_exposes_inspection_fields() -> None:
    scene = Scene(
        data="dummy",
        crs="EPSG:4326",
        transform=(1, 0, 0, 0, -1, 0),
        metadata={"cloud_cover": 10},
        band_names=["blue", "green"],
        acquisition_time=datetime(2024, 1, 1),
        sensor="Sentinel-2",
    )

    assert scene.bands == ("blue", "green")
    assert scene.metadata["cloud_cover"] == 10
    assert scene.sensor == "Sentinel-2"
    assert scene.crs == "EPSG:4326"


def test_scene_tracks_history_entries() -> None:
    scene = Scene(data="dummy")

    updated = scene.add_history("cloud mask")

    assert updated.history == ("cloud mask",)
    assert scene.history == ()


def test_scene_is_immutable() -> None:
    scene = Scene(data="dummy", metadata={"cloud_cover": 10})

    with pytest.raises(AttributeError):
        scene.data = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        scene.metadata["cloud_cover"] = 0
    assert isinstance(scene.metadata, MappingProxyType)
    assert isinstance(scene.masks, MappingProxyType)
    assert isinstance(scene.band_names, tuple)
    assert isinstance(scene.history, tuple)


def test_add_history_returns_new_scene_without_mutating_original() -> None:
    scene = Scene(data="dummy", history=("import",))

    updated = scene.add_history("cloud mask")

    assert updated is not scene
    assert updated.history == ("import", "cloud mask")
    assert scene.history == ("import",)


def test_cloud_schema_normalised_from_masks_backfills_metadata() -> None:
    scene = Scene(data={"blue": [1]}, masks={"cloud": [0, 1]})

    assert scene.cloud_mask == [0, 1]
    assert scene.masks["cloud"] == [0, 1]
    assert scene.metadata["cloud_mask"] == [0, 1]


def test_cloud_schema_normalised_from_metadata_backfills_masks() -> None:
    scene = Scene(data={"blue": [1]}, metadata={"cloud_mask": [1, 0], "cloud_prob": [0.9, 0.1]})

    assert scene.cloud_mask == [1, 0]
    assert scene.masks["cloud"] == [1, 0]
    assert scene.cloud_probability == [0.9, 0.1]
    assert scene.masks["cloud_probability"] == [0.9, 0.1]
    assert scene.metadata["cloud_probability"] == [0.9, 0.1]


def test_cloud_schema_absent_when_no_cloud_fields_present() -> None:
    scene = Scene(data={"blue": [1]})

    assert scene.cloud_mask is None
    assert scene.cloud_probability is None
    assert "cloud" not in scene.masks

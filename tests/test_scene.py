from datetime import datetime

from aquaticrs.scene import Scene


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

    scene.add_history("cloud mask")

    assert scene.history == ["cloud mask"]

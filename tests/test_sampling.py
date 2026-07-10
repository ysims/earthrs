import pytest

from earthrs.sampling import sample, sample_points, sample_polygons, sample_transects
from earthrs.scene import Scene


class _Geometry:
    def __init__(self, geom_type: str) -> None:
        self.geom_type = geom_type


def test_sample_dispatches_points(monkeypatch) -> None:
    scene = Scene(data="dummy")

    def _fake_sample_points(scene_obj, points_obj, *, method="nearest"):
        return {"scene": scene_obj, "points": points_obj, "method": method}

    monkeypatch.setattr("earthrs.sampling.core.sample_points", _fake_sample_points)
    result = sample(scene, _Geometry("Point"), method="bilinear")

    assert result["scene"] is scene
    assert result["method"] == "bilinear"


def test_sample_dispatches_polygons_explicit_target(monkeypatch) -> None:
    scene = Scene(data="dummy")

    def _fake_sample_polygons(scene_obj, polygons_obj, *, reducer="mean"):
        return {"scene": scene_obj, "polygons": polygons_obj, "reducer": reducer}

    monkeypatch.setattr("earthrs.sampling.core.sample_polygons", _fake_sample_polygons)
    result = sample(scene, object(), target="polygons", reducer="median")

    assert result["scene"] is scene
    assert result["reducer"] == "median"


def test_sample_rejects_unknown_target() -> None:
    scene = Scene(data="dummy")

    try:
        sample(scene, object(), target="unknown")
    except ValueError as exc:
        assert "target" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def _grid_scene(**kwargs) -> Scene:
    return Scene(
        data={"blue": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]},
        band_names=["blue"],
        **kwargs,
    )


def test_sample_points_reads_row_col_and_supports_dict_or_xy_keys() -> None:
    scene = _grid_scene()

    samples = sample_points(scene, [{"row": 0, "col": 1}, {"y": 2, "x": 0}])

    assert len(samples) == 2
    assert samples.rows[0]["blue"] == 2
    assert samples.rows[1]["blue"] == 7
    assert samples.rows[0]["observation_id"] == 0
    assert samples.metadata["sampling_method"] == "nearest"
    assert samples.metadata["sampling_target"] == "points"


def test_sample_points_rejects_unsupported_method() -> None:
    scene = _grid_scene()

    with pytest.raises(ValueError):
        sample_points(scene, {"row": 0, "col": 0}, method="cubic")


def test_sample_points_missing_band_raises() -> None:
    scene = Scene(data={"blue": [[1]]}, band_names=["red"])

    with pytest.raises(ValueError):
        sample_points(scene, {"row": 0, "col": 0})


def test_sample_points_rejects_non_mapping_scene_data() -> None:
    scene = Scene(data="dummy", band_names=["blue"])

    with pytest.raises(TypeError):
        sample_points(scene, {"row": 0, "col": 0})


def test_sample_polygons_reduces_pixels() -> None:
    scene = _grid_scene()
    polygon = {"pixels": [(0, 0), (0, 1), (0, 2)]}

    mean_samples = sample_polygons(scene, polygon, reducer="mean")
    max_samples = sample_polygons(scene, polygon, reducer="max")

    assert mean_samples.rows[0]["blue"] == 2.0
    assert max_samples.rows[0]["blue"] == 3.0
    assert mean_samples.metadata["sampling_target"] == "polygons"


def test_sample_polygons_rejects_unsupported_reducer() -> None:
    scene = _grid_scene()

    with pytest.raises(ValueError):
        sample_polygons(scene, {"pixels": [(0, 0)]}, reducer="sum")


def test_sample_polygons_requires_pixels_key() -> None:
    scene = _grid_scene()

    with pytest.raises(ValueError):
        sample_polygons(scene, {"not_pixels": []})


def test_sample_transects_emits_one_row_per_vertex() -> None:
    scene = _grid_scene()
    transect = {"vertices": [(0, 0), (1, 1), (2, 2)]}

    samples = sample_transects(scene, transect, spacing=10)

    assert len(samples) == 3
    assert [row["blue"] for row in samples] == [1, 5, 9]
    assert [row["vertex_id"] for row in samples] == [0, 1, 2]
    assert all(row["transect_id"] == 0 for row in samples)
    assert samples.metadata["spacing"] == 10


def test_sample_transects_requires_vertices_key() -> None:
    scene = _grid_scene()

    with pytest.raises(ValueError):
        sample_transects(scene, {"not_vertices": []})


def test_sample_raises_when_target_cannot_be_inferred() -> None:
    scene = _grid_scene()

    with pytest.raises(ValueError):
        sample(scene, object())

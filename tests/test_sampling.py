from earthrs.sampling import sample
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

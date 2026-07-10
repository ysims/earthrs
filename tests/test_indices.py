import math

import pytest

from earthrs.indices import evi, ndvi, ndwi
from earthrs.scene import Scene


def test_ndvi_computes_expected_value() -> None:
    scene = Scene(data={"nir": [0.8], "red": [0.2]}, band_names=["nir", "red"])

    result = ndvi(scene)

    assert result == pytest.approx([0.6])


def test_ndvi_handles_nested_sequences() -> None:
    scene = Scene(data={"nir": [[0.8, 0.4]], "red": [[0.2, 0.4]]}, band_names=["nir", "red"])

    result = ndvi(scene)

    assert result[0] == pytest.approx([0.6, 0.0])


def test_ndwi_computes_expected_value() -> None:
    scene = Scene(data={"green": [0.5], "nir": [0.1]}, band_names=["green", "nir"])

    result = ndwi(scene)

    assert result == pytest.approx([2 / 3])


def test_evi_computes_expected_value() -> None:
    scene = Scene(
        data={"nir": [0.6], "red": [0.2], "blue": [0.1]}, band_names=["nir", "red", "blue"]
    )

    result = evi(scene)

    denominator = 0.6 + 6.0 * 0.2 - 7.5 * 0.1 + 1.0
    expected = 2.5 * (0.6 - 0.2) / denominator
    assert result == pytest.approx([expected])


def test_index_raises_on_missing_band() -> None:
    scene = Scene(data={"nir": [0.8]}, band_names=["nir"])

    with pytest.raises(ValueError):
        ndvi(scene)


def test_index_raises_on_non_mapping_scene_data() -> None:
    scene = Scene(data="dummy", band_names=["nir", "red"])

    with pytest.raises(TypeError):
        ndvi(scene)


def test_ndvi_division_by_zero_returns_nan() -> None:
    scene = Scene(data={"nir": [0.0], "red": [0.0]}, band_names=["nir", "red"])

    result = ndvi(scene)

    assert math.isnan(result[0])


def test_ndvi_treats_none_values_as_missing() -> None:
    scene = Scene(data={"nir": [None], "red": [0.2]}, band_names=["nir", "red"])

    result = ndvi(scene)

    assert math.isnan(result[0])

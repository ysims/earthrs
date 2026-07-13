import math

import pytest

from earthrs.processing import (
    atmospheric_correction,
    cloud_mask,
    depth_correct,
    register,
    remove_glint,
    reproject,
)
from earthrs.scene import Scene


def test_remove_glint_rejects_unknown_method() -> None:
    scene = Scene(data={"nir": [1], "red": [1]}, band_names=["nir", "red"])

    with pytest.raises(ValueError):
        remove_glint(scene, method="not_a_method")


def test_depth_correct_rejects_unknown_method() -> None:
    scene = Scene(data={"blue": [0.1]}, band_names=["blue"])

    with pytest.raises(ValueError):
        depth_correct(scene, method="not_a_method")


def test_cloud_mask_rejects_unknown_method() -> None:
    scene = Scene(data={}, band_names=[])

    with pytest.raises(ValueError):
        cloud_mask(scene, method="not_a_method")


def test_atmospheric_correction_is_deferred() -> None:
    scene = Scene(data={"blue": [0.1]}, band_names=["blue"])

    with pytest.raises(NotImplementedError):
        atmospheric_correction(scene)


def test_hedley_glint_removal_subtracts_scaled_nir_and_clamps_at_zero() -> None:
    scene = Scene(data={"nir": [10, 20], "red": [5, 8]}, band_names=["nir", "red"])

    result = remove_glint(scene, method="hedley")

    assert result.data["nir"] == [10, 20]
    assert result.data["red"] == pytest.approx([5.0, 0.0])
    assert result.history == ("remove_glint:hedley",)


def test_hedley_glint_removal_requires_mapping_data() -> None:
    scene = Scene(data="dummy", band_names=["nir", "red"])

    with pytest.raises(TypeError):
        remove_glint(scene, method="hedley")


def test_lyzenga1978_log_linearises_each_band() -> None:
    scene = Scene(data={"blue": [0.1], "qa60": [1]}, band_names=["blue", "qa60"])

    result = depth_correct(scene, method="lyzenga1978")

    assert result.data["blue"] == pytest.approx([math.log(0.1)])
    assert result.data["qa60"] == [1]
    assert result.metadata["depth_method"] == "lyzenga1978"
    assert result.history == ("depth_correct:lyzenga1978",)


def test_lyzenga1978_subtracts_deep_water_radiance() -> None:
    scene = Scene(data={"blue": [0.5]}, band_names=["blue"])

    result = depth_correct(scene, method="lyzenga1978", deep_water_radiance={"blue": 0.2})

    assert result.data["blue"] == pytest.approx([math.log(0.3)])


def test_lyzenga1981_computes_depth_invariant_index() -> None:
    scene = Scene(
        data={"blue": [math.e], "green": [math.e**2]},
        band_names=["blue", "green"],
    )

    result = depth_correct(scene, method="lyzenga1981", k_i=3.0, k_j=4.0)

    assert result.data["lyzenga_dii"] == pytest.approx([-0.4])
    assert "lyzenga_dii" in result.band_names
    assert result.metadata["depth_method"] == "lyzenga1981"
    assert result.history == ("depth_correct:lyzenga1981",)


def test_lyzenga1981_requires_attenuation_coefficients() -> None:
    scene = Scene(data={"blue": [0.1], "green": [0.2]}, band_names=["blue", "green"])

    with pytest.raises(ValueError):
        depth_correct(scene, method="lyzenga1981")


def test_lyzenga1981_rejects_both_attenuation_coefficients_zero() -> None:
    scene = Scene(data={"blue": [0.1], "green": [0.2]}, band_names=["blue", "green"])

    with pytest.raises(ValueError):
        depth_correct(scene, method="lyzenga1981", k_i=0.0, k_j=0.0)


def test_lyzenga2006_computes_depth_regression() -> None:
    scene = Scene(
        data={"blue": [math.e], "green": [math.e**2]},
        band_names=["blue", "green"],
    )

    result = depth_correct(
        scene,
        method="lyzenga2006",
        coefficients={"blue": 1.0, "green": 0.5},
        intercept=10.0,
    )

    assert result.data["lyzenga_depth"] == pytest.approx([8.0])
    assert "lyzenga_depth" in result.band_names
    assert result.metadata["depth_method"] == "lyzenga2006"
    assert result.history == ("depth_correct:lyzenga2006",)


def test_lyzenga2006_requires_coefficients() -> None:
    scene = Scene(data={"blue": [0.1], "green": [0.2]}, band_names=["blue", "green"])

    with pytest.raises(ValueError):
        depth_correct(scene, method="lyzenga2006")


def test_lyzenga2006_requires_at_least_one_band() -> None:
    scene = Scene(data={"blue": [0.1], "green": [0.2]}, band_names=["blue", "green"])

    with pytest.raises(ValueError):
        depth_correct(scene, method="lyzenga2006", bands=(), coefficients={})


def test_bare_lyzenga_method_is_an_alias_for_lyzenga2006() -> None:
    scene = Scene(
        data={"blue": [math.e], "green": [math.e**2]},
        band_names=["blue", "green"],
    )

    result = depth_correct(
        scene, method="lyzenga", coefficients={"blue": 1.0, "green": 0.5}, intercept=10.0
    )

    assert result.data["lyzenga_depth"] == pytest.approx([8.0])
    assert result.metadata["depth_method"] == "lyzenga2006"


def test_stumpf_depth_correction_adds_derived_band() -> None:
    scene = Scene(data={"blue": [0.1], "green": [0.2]}, band_names=["blue", "green"])

    result = depth_correct(scene, method="stumpf")

    expected = 0.0 - 1.0 * (math.log(0.1) / math.log(0.2))
    assert result.data["stumpf_depth"] == pytest.approx([expected])
    assert "stumpf_depth" in result.band_names
    assert result.metadata["depth_method"] == "stumpf"


def test_maritorena_depth_correction_requires_depth() -> None:
    scene = Scene(data={"blue": [1.0]}, band_names=["blue"])

    with pytest.raises(ValueError):
        depth_correct(scene, method="maritorena")


def test_maritorena_depth_correction_applies_exponential_attenuation() -> None:
    scene = Scene(data={"blue": [1.0]}, band_names=["blue"])

    result = depth_correct(scene, method="maritorena", depth=[2.0], attenuation=0.1)

    assert result.data["blue"] == pytest.approx([math.exp(0.2)])
    assert result.metadata["depth_method"] == "maritorena"


def test_sentinel2_qa60_cloud_mask_checks_bits_10_and_11() -> None:
    scene = Scene(data={"qa60": [0, 1024, 2048, 3072]}, band_names=["qa60"])

    result = cloud_mask(scene, method="sentinel2_qa60")

    assert result.cloud_mask == [False, True, True, True]
    assert result.masks["cloud"] == result.cloud_mask
    assert result.metadata["cloud_mask"] == result.cloud_mask
    assert result.history == ("cloud_mask",)


def test_sentinel2_scl_cloud_mask_checks_cloudy_classes() -> None:
    scene = Scene(data={"scl": [1, 3, 5, 8, 11]}, band_names=["scl"])

    result = cloud_mask(scene, method="sentinel2_scl")

    assert result.cloud_mask == [False, True, False, True, True]


def test_landsat_qa_pixel_cloud_mask_checks_bits_2_3_4() -> None:
    scene = Scene(data={"qa_pixel": [0, 4, 8, 16]}, band_names=["qa_pixel"])

    result = cloud_mask(scene, method="landsat_qa_pixel")

    assert result.cloud_mask == [False, True, True, True]


def test_probability_cloud_mask_uses_threshold() -> None:
    scene = Scene(data={}, band_names=[])

    result = cloud_mask(scene, method="probability", probability=[0.2, 0.6, 0.5])

    assert result.cloud_mask == [False, True, True]
    assert result.cloud_probability == [0.2, 0.6, 0.5]


def test_probability_cloud_mask_requires_probability_input() -> None:
    scene = Scene(data={}, band_names=[])

    with pytest.raises(ValueError):
        cloud_mask(scene, method="probability")


def test_probability_cloud_mask_ors_with_user_mask() -> None:
    scene = Scene(data={}, band_names=[])

    result = cloud_mask(
        scene, method="probability", probability=[0.1, 0.1], user_mask=[True, False]
    )

    assert result.cloud_mask == [True, False]


def test_user_mask_cloud_mask_requires_user_mask() -> None:
    scene = Scene(data={}, band_names=[])

    with pytest.raises(ValueError):
        cloud_mask(scene, method="user_mask")


def test_user_mask_cloud_mask_applies_given_mask() -> None:
    scene = Scene(data={}, band_names=[])

    result = cloud_mask(scene, method="user_mask", user_mask=[True, False])

    assert result.cloud_mask == [True, False]


@pytest.mark.parametrize(
    ("sensor", "data", "expected_probability_arg", "expected_mask"),
    [
        ("Sentinel-2 MSI", {"qa60": [0, 3072]}, None, [False, True]),
        ("Landsat 8", {"qa_pixel": [0, 8]}, None, [False, True]),
    ],
)
def test_cloud_mask_auto_dispatches_by_sensor(
    sensor, data, expected_probability_arg, expected_mask
) -> None:
    band_names = list(data.keys())
    scene = Scene(data=data, sensor=sensor, band_names=band_names)

    result = cloud_mask(scene, method="auto")

    assert result.cloud_mask == expected_mask


def test_cloud_mask_auto_prefers_scl_when_present_for_sentinel2() -> None:
    scene = Scene(data={"scl": [3], "qa60": [0]}, sensor="Sentinel-2", band_names=["scl", "qa60"])

    result = cloud_mask(scene, method="auto")

    assert result.cloud_mask == [True]


def test_cloud_mask_auto_falls_back_to_probability_when_no_sensor_match() -> None:
    scene = Scene(data={}, sensor=None, band_names=[])

    result = cloud_mask(scene, method="auto", probability=[0.9])

    assert result.cloud_mask == [True]


def test_cloud_mask_auto_falls_back_to_user_mask_when_no_probability() -> None:
    scene = Scene(data={}, sensor=None, band_names=[])

    result = cloud_mask(scene, method="auto", user_mask=[True])

    assert result.cloud_mask == [True]


def test_cloud_mask_explicit_method_overrides_sensor_auto_resolution() -> None:
    scene = Scene(data={}, sensor="Sentinel-2", band_names=[])

    result = cloud_mask(scene, method="probability", probability=[0.9])

    assert result.cloud_mask == [True]


def test_reproject_requires_scene_transform() -> None:
    scene = Scene(data={"nir": [[1, 2], [3, 4]]}, band_names=["nir"])

    with pytest.raises(ValueError):
        reproject(scene, transform=(1, 0, 0, 0, 1, 0), shape=(2, 2))


def test_reproject_identity_transform_reproduces_grid() -> None:
    scene = Scene(
        data={"nir": [[1, 2], [3, 4]]},
        band_names=["nir"],
        transform=(1, 0, 0, 0, 1, 0),
        crs="EPSG:4326",
    )

    result = reproject(scene, transform=(1, 0, 0, 0, 1, 0), shape=(2, 2))

    assert result.data["nir"] == [[1, 2], [3, 4]]
    assert result.transform == (1, 0, 0, 0, 1, 0)
    assert result.history == ("reproject:nearest",)


def test_reproject_nearest_downsamples_by_picking_source_pixels() -> None:
    grid = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    scene = Scene(data={"nir": grid}, band_names=["nir"], transform=(1, 0, 0, 0, 1, 0))

    result = reproject(scene, transform=(2, 0, 0, 0, 2, 0), shape=(2, 2))

    assert result.data["nir"] == [[1, 3], [9, 11]]


def test_reproject_nearest_marks_out_of_bounds_as_none() -> None:
    scene = Scene(data={"nir": [[1, 2], [3, 4]]}, band_names=["nir"], transform=(1, 0, 0, 0, 1, 0))

    result = reproject(scene, transform=(1, 0, 10, 0, 1, 10), shape=(2, 2))

    assert result.data["nir"] == [[None, None], [None, None]]


def test_reproject_bilinear_interpolates_between_source_pixels() -> None:
    scene = Scene(
        data={"nir": [[0, 10], [20, 30]]}, band_names=["nir"], transform=(1, 0, 0, 0, 1, 0)
    )

    result = reproject(
        scene, transform=(1, 0, 0.5, 0, 1, 0.5), shape=(1, 1), resampling="bilinear"
    )

    assert result.data["nir"][0] == pytest.approx([15.0])


def test_reproject_rejects_unsupported_resampling_method() -> None:
    scene = Scene(data={"nir": [[1, 2], [3, 4]]}, band_names=["nir"], transform=(1, 0, 0, 0, 1, 0))

    with pytest.raises(ValueError):
        reproject(scene, transform=(1, 0, 0, 0, 1, 0), shape=(2, 2), resampling="cubic")


def test_reproject_cross_crs_requires_transformer() -> None:
    scene = Scene(
        data={"nir": [[1, 2], [3, 4]]},
        band_names=["nir"],
        transform=(1, 0, 0, 0, 1, 0),
        crs="EPSG:4326",
    )

    with pytest.raises(ValueError):
        reproject(scene, transform=(1, 0, 0, 0, 1, 0), shape=(2, 2), crs="EPSG:3857")


def test_reproject_cross_crs_uses_transformer() -> None:
    scene = Scene(
        data={"nir": [[1, 2], [3, 4]]},
        band_names=["nir"],
        transform=(1, 0, 0, 0, 1, 0),
        crs="EPSG:4326",
    )

    result = reproject(
        scene,
        transform=(1, 0, 0, 0, 1, 0),
        shape=(2, 2),
        crs="EPSG:3857",
        transformer=lambda x, y: (x, y),
    )

    assert result.data["nir"] == [[1, 2], [3, 4]]
    assert result.crs == "EPSG:3857"


def test_reproject_resamples_cloud_mask_with_nearest_regardless_of_method() -> None:
    scene = Scene(
        data={"nir": [[1.0, 2.0], [3.0, 4.0]]},
        band_names=["nir"],
        transform=(1, 0, 0, 0, 1, 0),
        masks={"cloud": [[True, False], [False, True]]},
    )

    result = reproject(
        scene, transform=(1, 0, 0.5, 0, 1, 0.5), shape=(1, 1), resampling="bilinear"
    )

    assert result.cloud_mask == [[True]]
    assert result.masks["cloud"] == result.cloud_mask


def test_register_aligns_scene_onto_reference_grid() -> None:
    scene = Scene(
        data={"nir": [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]},
        band_names=["nir"],
        transform=(1, 0, 0, 0, 1, 0),
    )
    reference = Scene(data={"nir": [[0, 0], [0, 0]]}, transform=(2, 0, 0, 0, 2, 0))

    result = register(scene, reference)

    assert result.data["nir"] == [[1, 3], [9, 11]]
    assert result.transform == (2, 0, 0, 0, 2, 0)
    assert result.history == ("register:nearest",)


def test_register_requires_reference_transform() -> None:
    scene = Scene(data={"nir": [[1, 2], [3, 4]]}, band_names=["nir"], transform=(1, 0, 0, 0, 1, 0))
    reference = Scene(data={"nir": [[0, 0], [0, 0]]})

    with pytest.raises(ValueError):
        register(scene, reference)


def test_register_rejects_mismatched_crs_without_transformer() -> None:
    scene = Scene(
        data={"nir": [[1, 2], [3, 4]]},
        band_names=["nir"],
        transform=(1, 0, 0, 0, 1, 0),
        crs="EPSG:4326",
    )
    reference = Scene(
        data={"nir": [[0, 0], [0, 0]]}, transform=(1, 0, 0, 0, 1, 0), crs="EPSG:3857"
    )

    with pytest.raises(ValueError):
        register(scene, reference)

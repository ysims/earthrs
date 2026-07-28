import pytest

from earthrs.sensors.catalog import Band, Landsat, PlanetScope, Sentinel2, resolve_sensor


def test_sentinel2_bands_contain_expected_entries() -> None:
    sensor = Sentinel2()

    assert sensor.name == "Sentinel-2"
    names = [band.name for band in sensor.bands]
    assert names == [
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B9",
        "B10",
        "B11",
        "B12",
    ]
    b2 = next(band for band in sensor.bands if band.name == "B2")
    assert b2 == Band("B2", 490.0, None, 10.0)
    b1 = next(band for band in sensor.bands if band.name == "B1")
    assert b1.resolution_m == 60.0


def test_landsat_bands_contain_expected_entries() -> None:
    sensor = Landsat()

    assert sensor.name == "Landsat"
    names = [band.name for band in sensor.bands]
    assert names == ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9"]
    panchromatic = next(band for band in sensor.bands if band.name == "B8")
    assert panchromatic.center_wavelength_nm == 590.0
    assert panchromatic.resolution_m == 15.0
    nir = next(band for band in sensor.bands if band.name == "B5")
    assert nir.center_wavelength_nm == 865.0


def test_planetscope_bands_contain_expected_entries() -> None:
    sensor = PlanetScope()

    assert sensor.name == "PlanetScope"
    names = [band.name for band in sensor.bands]
    assert names == [
        "Coastal Blue",
        "Blue",
        "Green I",
        "Green",
        "Yellow",
        "Red",
        "Red Edge",
        "NIR",
    ]
    assert all(band.resolution_m == 3.0 for band in sensor.bands)
    nir = next(band for band in sensor.bands if band.name == "NIR")
    assert nir.center_wavelength_nm == 865.0


@pytest.mark.parametrize(
    ("sensor_name", "expected_type"),
    [
        ("Sentinel-2", Sentinel2),
        ("Sentinel-2A", Sentinel2),
        ("Sentinel-2B", Sentinel2),
        ("sentinel-2 msi", Sentinel2),
        ("Landsat", Landsat),
        ("Landsat 8", Landsat),
        ("Landsat-9", Landsat),
        ("landsat 9 oli", Landsat),
        ("PlanetScope", PlanetScope),
        ("planetscope superdove", PlanetScope),
    ],
)
def test_resolve_sensor_matches_known_variants(sensor_name, expected_type) -> None:
    result = resolve_sensor(sensor_name)

    assert isinstance(result, expected_type)


@pytest.mark.parametrize("sensor_name", ["", None, "WorldView-3", "unknown sensor"])
def test_resolve_sensor_returns_none_for_unknown_sensors(sensor_name) -> None:
    assert resolve_sensor(sensor_name) is None

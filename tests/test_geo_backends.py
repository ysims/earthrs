"""Tests for the optional xarray/rasterio-backed `Scene.data` paths.

`earthrs` has zero required runtime dependencies, and the `dict`-of-nested-list path is covered
extensively by `tests/test_indices.py` and `tests/test_sampling.py`. This module covers the
*optional* xarray/rasterio-backed paths (see `docs/concepts.md`).

Each backend-specific test class is guarded with ``@pytest.mark.skipif`` (rather than a bare
module-level ``pytest.importorskip``, which would skip the *entire* module — including the
other backend's tests — the moment either package is missing) so:

- an environment with neither package installed skips everything in this file cleanly;
- an environment with only one of the two packages installed still runs that backend's tests.
"""

from __future__ import annotations

import math

import pytest

from earthrs.indices import evi, ndvi, ndwi
from earthrs.sampling import sample_points, sample_polygons, sample_transects
from earthrs.scene import Scene

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised only when numpy is absent
    np = None

try:
    import xarray
except ImportError:  # pragma: no cover - exercised only when xarray is absent
    xarray = None

try:
    import rasterio
except ImportError:  # pragma: no cover - exercised only when rasterio is absent
    rasterio = None


def _xr_data_array(bands: dict[str, list[list[float]]]) -> xarray.DataArray:
    """Build a DataArray with a leading 'band' dimension, coordinate-named per `bands` keys."""

    band_names = list(bands)
    stacked = np.array([bands[name] for name in band_names], dtype="float64")
    return xarray.DataArray(
        stacked,
        dims=("band", "y", "x"),
        coords={"band": band_names},
    )


def _xr_dataset(bands: dict[str, list[list[float]]]) -> xarray.Dataset:
    return xarray.Dataset({name: (("y", "x"), values) for name, values in bands.items()})


@pytest.mark.skipif(xarray is None, reason="xarray is not installed")
class TestXarrayDataArray:
    def test_ndvi_matches_dict_backed_result(self) -> None:
        dict_scene = Scene(
            data={"nir": [[0.8, 0.4]], "red": [[0.2, 0.4]]}, band_names=["nir", "red"]
        )
        xr_scene = Scene(
            data=_xr_data_array({"nir": [[0.8, 0.4]], "red": [[0.2, 0.4]]}),
            band_names=["nir", "red"],
        )

        # pytest.approx does not support nested (2D) sequences, so compare row by row.
        assert ndvi(xr_scene)[0] == pytest.approx(ndvi(dict_scene)[0])

    def test_ndwi_and_evi_also_work(self) -> None:
        data = _xr_data_array({"nir": [[0.6]], "red": [[0.2]], "blue": [[0.1]], "green": [[0.5]]})
        scene = Scene(data=data, band_names=["nir", "red", "blue", "green"])

        ndwi_result = ndwi(scene)
        evi_result = evi(scene)

        assert ndwi_result[0][0] == pytest.approx((0.5 - 0.6) / (0.5 + 0.6))
        denominator = 0.6 + 6.0 * 0.2 - 7.5 * 0.1 + 1.0
        assert evi_result[0][0] == pytest.approx(2.5 * (0.6 - 0.2) / denominator)

    def test_missing_band_raises_value_error(self) -> None:
        scene = Scene(data=_xr_data_array({"nir": [[0.8]]}), band_names=["nir"])

        with pytest.raises(ValueError):
            ndvi(scene)

    def test_data_array_without_band_dimension_raises_type_error(self) -> None:
        data = xarray.DataArray(np.zeros((2, 2)), dims=("y", "x"))
        scene = Scene(data=data, band_names=["nir", "red"])

        with pytest.raises(TypeError):
            ndvi(scene)

    def test_sample_points_reads_expected_cells(self) -> None:
        grid = _xr_data_array({"blue": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]})
        scene = Scene(data=grid, band_names=["blue"])

        samples = sample_points(scene, [{"row": 0, "col": 1}, {"row": 2, "col": 0}])

        assert samples.rows[0]["blue"] == 2
        assert samples.rows[1]["blue"] == 7
        assert isinstance(samples.rows[0]["blue"], (int, float))

    def test_sample_polygons_and_transects_work(self) -> None:
        grid = _xr_data_array({"blue": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]})
        scene = Scene(data=grid, band_names=["blue"])

        polygon_samples = sample_polygons(scene, {"pixels": [(0, 0), (0, 1), (0, 2)]})
        transect_samples = sample_transects(scene, {"vertices": [(0, 0), (1, 1), (2, 2)]})

        assert polygon_samples.rows[0]["blue"] == pytest.approx(2.0)
        assert [row["blue"] for row in transect_samples] == [1, 5, 9]

    def test_sample_points_missing_band_raises_value_error(self) -> None:
        scene = Scene(data=_xr_data_array({"blue": [[1]]}), band_names=["red"])

        with pytest.raises(ValueError):
            sample_points(scene, {"row": 0, "col": 0})


@pytest.mark.skipif(xarray is None, reason="xarray is not installed")
class TestXarrayDataset:
    def test_ndvi_matches_dict_backed_result(self) -> None:
        dict_scene = Scene(
            data={"nir": [[0.8, 0.4]], "red": [[0.2, 0.4]]}, band_names=["nir", "red"]
        )
        ds_scene = Scene(
            data=_xr_dataset({"nir": [[0.8, 0.4]], "red": [[0.2, 0.4]]}),
            band_names=["nir", "red"],
        )

        assert ndvi(ds_scene)[0] == pytest.approx(ndvi(dict_scene)[0])

    def test_missing_band_raises_value_error(self) -> None:
        scene = Scene(data=_xr_dataset({"nir": [[0.8]]}), band_names=["nir"])

        with pytest.raises(ValueError):
            ndvi(scene)

    def test_sample_points_reads_expected_cells(self) -> None:
        grid = _xr_dataset({"blue": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]})
        scene = Scene(data=grid, band_names=["blue"])

        samples = sample_points(scene, [{"row": 0, "col": 1}, {"row": 2, "col": 0}])

        assert samples.rows[0]["blue"] == 2
        assert samples.rows[1]["blue"] == 7


@pytest.mark.skipif(xarray is None, reason="xarray is not installed")
class TestXarrayNaNHandling:
    def test_nan_propagates(self) -> None:
        data = _xr_data_array({"nir": [[float("nan")]], "red": [[0.2]]})
        scene = Scene(data=data, band_names=["nir", "red"])

        result = ndvi(scene)

        assert math.isnan(result[0][0])


def _rasterio_memory_dataset(bands: list[list[list[float]]]):
    """Open an in-memory rasterio dataset with one band per entry in `bands`.

    Returns ``(memfile, dataset)``; the caller is responsible for closing both.
    """

    memfile = rasterio.io.MemoryFile()
    height = len(bands[0])
    width = len(bands[0][0])
    with memfile.open(
        driver="GTiff", height=height, width=width, count=len(bands), dtype="float64"
    ) as dataset:
        for index, band in enumerate(bands, start=1):
            dataset.write(np.array(band, dtype="float64"), index)
    return memfile, memfile.open()


@pytest.mark.skipif(rasterio is None, reason="rasterio is not installed")
class TestRasterioDataset:
    def test_ndvi_matches_dict_backed_result(self) -> None:
        memfile, dataset = _rasterio_memory_dataset([[[0.8, 0.4]], [[0.2, 0.4]]])
        try:
            scene = Scene(data=dataset, band_names=["nir", "red"])
            dict_scene = Scene(
                data={"nir": [[0.8, 0.4]], "red": [[0.2, 0.4]]}, band_names=["nir", "red"]
            )

            assert ndvi(scene)[0] == pytest.approx(ndvi(dict_scene)[0])
        finally:
            dataset.close()
            memfile.close()

    def test_band_index_follows_band_names_order(self) -> None:
        # band_names[0] == "red" maps to rasterio band 1, band_names[1] == "nir" to band 2.
        memfile, dataset = _rasterio_memory_dataset([[[0.2, 0.4]], [[0.8, 0.4]]])
        try:
            scene = Scene(data=dataset, band_names=["red", "nir"])

            result = ndvi(scene, nir_band="nir", red_band="red")

            assert result[0] == pytest.approx([0.6, 0.0])
        finally:
            dataset.close()
            memfile.close()

    def test_missing_band_raises_value_error(self) -> None:
        memfile, dataset = _rasterio_memory_dataset([[[0.8]]])
        try:
            scene = Scene(data=dataset, band_names=["nir"])

            with pytest.raises(ValueError):
                ndvi(scene, nir_band="nir", red_band="red")
        finally:
            dataset.close()
            memfile.close()

    def test_sample_points_reads_expected_cells(self) -> None:
        memfile, dataset = _rasterio_memory_dataset([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        try:
            scene = Scene(data=dataset, band_names=["blue"])

            samples = sample_points(scene, [{"row": 0, "col": 1}, {"row": 2, "col": 0}])

            assert samples.rows[0]["blue"] == 2
            assert samples.rows[1]["blue"] == 7
        finally:
            dataset.close()
            memfile.close()

    def test_sample_polygons_and_transects_work(self) -> None:
        memfile, dataset = _rasterio_memory_dataset([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        try:
            scene = Scene(data=dataset, band_names=["blue"])

            polygon_samples = sample_polygons(scene, {"pixels": [(0, 0), (0, 1), (0, 2)]})
            transect_samples = sample_transects(scene, {"vertices": [(0, 0), (1, 1), (2, 2)]})

            assert polygon_samples.rows[0]["blue"] == pytest.approx(2.0)
            assert [row["blue"] for row in transect_samples] == [1, 5, 9]
        finally:
            dataset.close()
            memfile.close()

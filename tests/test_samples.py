from types import MappingProxyType

import pytest

from earthrs.samples import Samples


def test_samples_freezes_rows_and_metadata() -> None:
    samples = Samples(rows=({"a": 1}, {"a": 2}), metadata={"sampling_method": "nearest"})

    assert len(samples) == 2
    assert list(samples) == [{"a": 1}, {"a": 2}]
    for row in samples:
        assert isinstance(row, MappingProxyType)
        with pytest.raises(TypeError):
            row["a"] = 0
    assert isinstance(samples.metadata, MappingProxyType)
    with pytest.raises(TypeError):
        samples.metadata["sampling_method"] = "bilinear"


def test_samples_is_immutable_dataclass() -> None:
    samples = Samples(rows=({"a": 1},))

    with pytest.raises(AttributeError):
        samples.rows = ()  # type: ignore[misc]


def test_samples_columns_preserves_first_seen_order_across_rows() -> None:
    samples = Samples(rows=({"a": 1, "b": 2}, {"b": 3, "c": 4}))

    assert samples.columns == ("a", "b", "c")


def test_samples_with_metadata_returns_new_instance() -> None:
    samples = Samples(rows=({"a": 1},), metadata={"sampling_method": "nearest"})

    updated = samples.with_metadata(sampling_target="points")

    assert updated is not samples
    assert updated.metadata == {"sampling_method": "nearest", "sampling_target": "points"}
    assert samples.metadata == {"sampling_method": "nearest"}
    assert updated.rows == samples.rows

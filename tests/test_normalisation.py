from earthrs.io.normalisation import normalise_cloud_schema


def test_normalise_cloud_schema_prefers_explicit_arguments() -> None:
    masks, metadata, cloud_mask, cloud_probability = normalise_cloud_schema(
        metadata={},
        masks={"cloud": "from_masks"},
        cloud_mask="explicit",
        cloud_probability="explicit_prob",
    )

    assert cloud_mask == "explicit"
    assert cloud_probability == "explicit_prob"
    assert masks["cloud"] == "explicit"
    assert metadata["cloud_mask"] == "explicit"


def test_normalise_cloud_schema_resolves_from_masks_alternate_keys() -> None:
    for key in ("cloud", "cloud_mask", "mask_cloud"):
        masks, metadata, cloud_mask, cloud_probability = normalise_cloud_schema(masks={key: "m"})
        assert cloud_mask == "m"
        assert masks["cloud"] == "m"
        assert metadata["cloud_mask"] == "m"


def test_normalise_cloud_schema_resolves_from_metadata_alternate_keys() -> None:
    for key in ("cloud_mask", "mask_cloud"):
        masks, metadata, cloud_mask, _ = normalise_cloud_schema(metadata={key: "m"})
        assert cloud_mask == "m"
        assert masks["cloud"] == "m"


def test_normalise_cloud_schema_resolves_probability_alternate_keys() -> None:
    for key in ("cloud_probability", "cloud_prob", "probability_cloud"):
        _, _, _, cloud_probability = normalise_cloud_schema(masks={key: 0.5})
        assert cloud_probability == 0.5


def test_normalise_cloud_schema_does_not_overwrite_existing_metadata_values() -> None:
    masks, metadata, cloud_mask, _ = normalise_cloud_schema(
        metadata={"cloud_mask": "kept"},
        masks={"cloud": "from_masks"},
    )

    assert cloud_mask == "from_masks"
    assert masks["cloud"] == "from_masks"
    assert metadata["cloud_mask"] == "kept"


def test_normalise_cloud_schema_handles_absent_cloud_fields() -> None:
    masks, metadata, cloud_mask, cloud_probability = normalise_cloud_schema()

    assert cloud_mask is None
    assert cloud_probability is None
    assert "cloud" not in masks
    assert "cloud_probability" not in masks
    assert masks == {}
    assert metadata == {}


def test_normalise_cloud_schema_does_not_mutate_inputs() -> None:
    original_masks = {"cloud": "m"}
    original_metadata = {"other": 1}

    normalise_cloud_schema(masks=original_masks, metadata=original_metadata)

    assert original_masks == {"cloud": "m"}
    assert original_metadata == {"other": 1}

"""Metadata normalisation helpers."""

from __future__ import annotations

from typing import Any


def normalise_cloud_schema(
    *,
    metadata: dict[str, Any] | None = None,
    masks: dict[str, Any] | None = None,
    cloud_mask: Any | None = None,
    cloud_probability: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Any | None, Any | None]:
    """Normalise cloud fields into a consistent internal schema.

    Returns a ``(masks, metadata, cloud_mask, cloud_probability)`` tuple. Cloud
    fields sourced from ``masks`` are backfilled into ``metadata`` (and vice versa)
    so downstream algorithms can rely on a single normalised schema regardless of
    which dictionary a product's importer originally populated.
    """

    metadata_dict = dict(metadata or {})
    mask_dict = dict(masks or {})

    resolved_cloud_mask = cloud_mask
    if resolved_cloud_mask is None:
        for key in ("cloud", "cloud_mask", "mask_cloud"):
            if key in mask_dict:
                resolved_cloud_mask = mask_dict[key]
                break
    if resolved_cloud_mask is None:
        for key in ("cloud_mask", "mask_cloud"):
            if key in metadata_dict:
                resolved_cloud_mask = metadata_dict[key]
                break

    resolved_cloud_probability = cloud_probability
    if resolved_cloud_probability is None:
        for key in ("cloud_probability", "cloud_prob", "probability_cloud"):
            if key in mask_dict:
                resolved_cloud_probability = mask_dict[key]
                break
    if resolved_cloud_probability is None:
        for key in ("cloud_probability", "cloud_prob"):
            if key in metadata_dict:
                resolved_cloud_probability = metadata_dict[key]
                break

    if resolved_cloud_mask is not None:
        mask_dict["cloud"] = resolved_cloud_mask
        metadata_dict.setdefault("cloud_mask", resolved_cloud_mask)
    if resolved_cloud_probability is not None:
        mask_dict["cloud_probability"] = resolved_cloud_probability
        metadata_dict.setdefault("cloud_probability", resolved_cloud_probability)

    return mask_dict, metadata_dict, resolved_cloud_mask, resolved_cloud_probability

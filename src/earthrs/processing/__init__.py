"""Preprocessing operation placeholders."""

from earthrs.processing.core import (
    atmospheric_correction,
    cloud_mask,
    depth_correct,
    register,
    remove_glint,
    reproject,
)

__all__ = [
    "remove_glint",
    "depth_correct",
    "cloud_mask",
    "atmospheric_correction",
    "register",
    "reproject",
]

"""Input/output interfaces.

This package will host scene readers, writers, and metadata-normalisation helpers for supported
Earth observation formats.
"""

from earthrs.io.normalisation import normalise_cloud_schema

__all__ = ["normalise_cloud_schema"]

"""EarthRS core package.

This package defines high-level abstractions and modular Earth observation utilities, including
aquatic and coastal remote sensing workflows.
"""

from earthrs.dataset import Dataset
from earthrs.scene import Scene

__all__ = ["Dataset", "Scene"]

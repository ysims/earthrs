"""AquaticRS core package.

This package defines high-level abstractions and modular Earth observation utilities for aquatic
and coastal remote sensing workflows.
"""

from aquaticrs.dataset import Dataset
from aquaticrs.scene import Scene

__all__ = ["Dataset", "Scene"]

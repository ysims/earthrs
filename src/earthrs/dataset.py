"""Dataset abstraction for collections of scenes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from earthrs.scene import Scene


@dataclass(slots=True)
class Dataset:
    """Represent a collection of :class:`earthrs.scene.Scene` objects."""

    scenes: list[Scene] = field(default_factory=list)

    def __iter__(self):
        """Iterate through contained scenes."""

        return iter(self.scenes)

    def __len__(self) -> int:
        """Return number of scenes in the dataset."""

        return len(self.scenes)

    def add_scene(self, scene: Scene) -> None:
        """Add a scene to the dataset."""

        self.scenes.append(scene)

    def filter_dates(self, start: datetime | None = None, end: datetime | None = None) -> Dataset:
        """Filter scenes by acquisition date.

        TODO: Implement date-based filtering and return a new filtered dataset.
        """

        raise NotImplementedError("TODO: implement date-based filtering")

    def filter_clouds(self, max_cloud_fraction: float) -> Dataset:
        """Filter scenes by cloud coverage metadata.

        TODO: Implement cloud filtering based on metadata conventions.
        """

        raise NotImplementedError("TODO: implement cloud filtering")

    def sample_points(self, points) -> object:
        """Sample all scenes at point locations.

        TODO: Delegate to `earthrs.sampling.sample_points` once implemented.
        """

        raise NotImplementedError("TODO: implement dataset point sampling")

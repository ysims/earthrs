"""Dataset abstraction for collections of scenes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from math import isfinite
from typing import Any

from earthrs.samples import Samples
from earthrs.sampling import sample_points
from earthrs.scene import Scene


@dataclass(frozen=True, slots=True)
class Dataset:
    """Represent a collection of :class:`earthrs.scene.Scene` objects."""

    scenes: tuple[Scene, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenes", tuple(self.scenes))

    def __iter__(self):
        """Iterate through contained scenes."""

        return iter(self.scenes)

    def __len__(self) -> int:
        """Return number of scenes in the dataset."""

        return len(self.scenes)

    def add_scene(self, scene: Scene) -> Dataset:
        """Add a scene to the dataset."""

        return Dataset((*self.scenes, scene))

    def filter_dates(self, start: datetime | None = None, end: datetime | None = None) -> Dataset:
        """Filter scenes by acquisition date.

        Returns a new dataset containing only scenes with timestamps within the
        provided inclusive bounds.
        """

        if start and end and start > end:
            raise ValueError("`start` cannot be later than `end`.")
        filtered = []
        for scene in self.scenes:
            if scene.acquisition_time is None:
                continue
            if start and scene.acquisition_time < start:
                continue
            if end and scene.acquisition_time > end:
                continue
            filtered.append(scene)
        return Dataset(tuple(filtered))

    def filter_clouds(self, max_cloud_fraction: float) -> Dataset:
        """Filter scenes by cloud coverage metadata.

        Returns a new dataset with cloud fraction at or below ``max_cloud_fraction``.
        """

        if not isfinite(max_cloud_fraction) or not 0 <= max_cloud_fraction <= 1:
            raise ValueError("`max_cloud_fraction` must be a finite value in [0, 1].")
        filtered = []
        for scene in self.scenes:
            cloud_fraction = _resolve_cloud_fraction(scene)
            if cloud_fraction is None:
                continue
            if cloud_fraction <= max_cloud_fraction:
                filtered.append(scene)
        return Dataset(tuple(filtered))

    def sample_points(self, points: Any) -> Samples:
        """Sample all scenes at point locations.

        Returns an immutable samples table with one row per sampled observation.
        """

        rows: list[dict[str, Any]] = []
        for scene_index, scene in enumerate(self.scenes):
            scene_samples = sample_points(scene, points, method="nearest")
            for row in scene_samples:
                enriched = dict(row)
                enriched.setdefault("scene_index", scene_index)
                rows.append(enriched)
        return Samples(rows=tuple(rows), metadata={"sampling_scope": "dataset"})


def _resolve_cloud_fraction(scene: Scene) -> float | None:
    cloud_fraction = scene.metadata.get("cloud_fraction")
    if cloud_fraction is None:
        cloud_cover = scene.metadata.get("cloud_cover")
        if cloud_cover is not None:
            cloud_cover = float(cloud_cover)
            cloud_fraction = cloud_cover / 100 if cloud_cover > 1 else cloud_cover
    if cloud_fraction is not None:
        value = float(cloud_fraction)
        if value > 1:
            value = value / 100
        return max(0.0, min(1.0, value))
    if scene.cloud_mask is not None:
        values = _flatten(scene.cloud_mask)
        if values:
            return sum(1 for item in values if bool(item)) / len(values)
    return None


def _flatten(values: Any) -> list[Any]:
    if isinstance(values, (list, tuple)):
        out: list[Any] = []
        for value in values:
            out.extend(_flatten(value))
        return out
    return [values]

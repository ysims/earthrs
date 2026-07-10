"""Immutable feature-table samples."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


def _freeze_row(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(row))


@dataclass(frozen=True, slots=True)
class Samples:
    """Represent tabular observations as immutable feature rows."""

    rows: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        frozen_rows = tuple(_freeze_row(row) for row in self.rows)
        object.__setattr__(self, "rows", frozen_rows)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    @property
    def columns(self) -> tuple[str, ...]:
        names: list[str] = []
        for row in self.rows:
            for key in row:
                if key not in names:
                    names.append(key)
        return tuple(names)

    def with_metadata(self, **metadata: Any) -> Samples:
        merged = dict(self.metadata)
        merged.update(metadata)
        return Samples(rows=self.rows, metadata=merged)

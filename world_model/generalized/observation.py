"""Observation: a fact at a point in observation space.

Domain-agnostic: coordinates are a tuple of floats. The world declares
the dimensionality and the semantics of each axis. The engine never
needs to know what an axis means -- only that two points can be
compared by distance and projected onto axes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple
import uuid


@dataclass(frozen=True)
class Observation:
    """An atomic fact at a point in observation space.

    Frozen so observations can be safely shared across tendencies.
    Identity is by ``id``; coordinates carry the data.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    coords: Tuple[float, ...] = ()
    label: str = ""

    @property
    def dim(self) -> int:
        return len(self.coords)

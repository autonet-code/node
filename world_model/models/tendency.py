"""
Tendency model: open-roster competing frames that stake on nodes.

A Tendency is a candidate explanatory frame: it proposes claims, stakes
observations as evidence, and competes for allocation share. Tendencies
are open (no fixed enum, no fixed roster) and instantiated by an
external authority via a factory (see ``world_model.models.factory``).

This replaces the earlier closed-enum design in which seven tendencies
were baked into the type system. The seven were a useful starting set
for personality modelling but are not architectural -- the engine works
with whatever roster the calling authority provides.

Allocations sum to 1.0 across the set. They shift through arena
debate (winners gain, losers lose) and through novelty-modulated
attention curves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from .lod import LODPayload, LODProtocol, to_level


@dataclass
class Tendency:
    """A competing frame. Identified by an opaque string id.

    Tendencies are not pre-typed by domain. The same machinery hosts
    "survival" / "curiosity" for personality and "cohesive" / "dagger"
    for categorical-physics; the engine does not know which it is.
    """

    id: str
    allocation: float = 0.0
    description: str = ""

    # Performance tracking for allocation adjustment
    stakes_placed: int = 0
    stakes_validated: int = 0

    # Optional metadata blob for callers who need to attach domain-specific
    # information (initial claim text, doctrine tags, factory provenance, ...).
    metadata: dict = field(default_factory=dict)

    # Level-of-detail support. A tendency carries a protocol and a current
    # LOD payload. When neither is set, the tendency behaves as a plain
    # record (LOD-aware methods raise). Tendencies created via the factory
    # may have these populated; tendencies created by hand may not.
    protocol: Optional[LODProtocol] = None
    payload: Optional[LODPayload] = None

    @property
    def validation_rate(self) -> float:
        if self.stakes_placed == 0:
            return 0.0
        return self.stakes_validated / self.stakes_placed

    # ------------------------------------------------------------------
    # Level-of-detail
    # ------------------------------------------------------------------

    @property
    def current_lod(self) -> Optional[int]:
        """Current LOD level number, or None if this tendency is not LOD-aware."""
        return self.payload.level.level if self.payload is not None else None

    def at_lod(self, k: int, context: object = None) -> "Tendency":
        """Return a new Tendency with payload promoted/demoted to level k.

        Raises ValueError if this tendency has no protocol attached.
        Does not mutate self.
        """
        if self.protocol is None or self.payload is None:
            raise ValueError("tendency has no LOD protocol; cannot change level")
        new_payload = to_level(self.payload, k, self.protocol, context=context)
        # Build a fresh tendency that reflects whatever fields the
        # new payload exposes at the target LOD. Identity (id) is always
        # at LOD 0 so it always survives.
        return Tendency(
            id=str(new_payload.fields.get("id", self.id)),
            allocation=float(new_payload.fields.get("allocation", self.allocation)),
            description=str(new_payload.fields.get("description", self.description)),
            stakes_placed=self.stakes_placed,
            stakes_validated=self.stakes_validated,
            metadata=dict(new_payload.fields.get("metadata", self.metadata)) if isinstance(new_payload.fields.get("metadata"), dict) else self.metadata,
            protocol=self.protocol,
            payload=new_payload,
        )

    def to_lod_bytes(self) -> bytes:
        """Pack the current LOD payload to its fixed-size byte form."""
        if self.payload is None:
            raise ValueError("tendency has no LOD payload to pack")
        return self.payload.to_bytes()

    def __repr__(self) -> str:
        return f"Tendency(id={self.id!r}, allocation={self.allocation:.2%})"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "allocation": self.allocation,
            "description": self.description,
            "stakes_placed": self.stakes_placed,
            "stakes_validated": self.stakes_validated,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Tendency":
        return cls(
            id=data["id"],
            allocation=float(data.get("allocation", 0.0)),
            description=data.get("description", ""),
            stakes_placed=int(data.get("stakes_placed", 0)),
            stakes_validated=int(data.get("stakes_validated", 0)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class TendencySet:
    """A coalition of tendencies whose allocations sum to 1.0.

    Open roster: tendencies can be added and removed at any time. After
    structural changes, call ``normalize()`` to ensure allocations sum
    to 1.0 again.
    """

    tendencies: dict[str, Tendency] = field(default_factory=dict)
    calibrated: bool = False

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_iterable(cls, tendencies: Iterable[Tendency]) -> "TendencySet":
        ts = cls()
        for t in tendencies:
            ts.add(t)
        ts.normalize()
        return ts

    # ------------------------------------------------------------------
    # Roster mutation
    # ------------------------------------------------------------------

    def add(self, tendency: Tendency) -> Tendency:
        """Add (or replace) a tendency. Caller is responsible for normalize."""
        self.tendencies[tendency.id] = tendency
        return tendency

    def remove(self, tendency_id: str) -> Optional[Tendency]:
        """Remove a tendency by id. Returns the removed instance or None."""
        return self.tendencies.pop(tendency_id, None)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, tendency_id: str) -> Tendency:
        return self.tendencies[tendency_id]

    def has(self, tendency_id: str) -> bool:
        return tendency_id in self.tendencies

    def all(self) -> list[Tendency]:
        return list(self.tendencies.values())

    def ids(self) -> list[str]:
        return list(self.tendencies.keys())

    def __len__(self) -> int:
        return len(self.tendencies)

    def __iter__(self):
        return iter(self.tendencies.values())

    def __contains__(self, tendency_id) -> bool:
        return tendency_id in self.tendencies

    # ------------------------------------------------------------------
    # Allocation arithmetic
    # ------------------------------------------------------------------

    @property
    def total_allocation(self) -> float:
        return sum(t.allocation for t in self.tendencies.values())

    def normalize(self) -> None:
        """Ensure allocations sum to 1.0. No-op on empty set."""
        if not self.tendencies:
            return
        total = self.total_allocation
        if total <= 0:
            # Distribute uniformly if everything zeroed out
            n = len(self.tendencies)
            for t in self.tendencies.values():
                t.allocation = 1.0 / n
            return
        for t in self.tendencies.values():
            t.allocation /= total

    def adjust_allocation(self, tendency_id: str, delta: float) -> None:
        """Shift one tendency's allocation, redistributing the inverse across the rest."""
        target = self.tendencies[tendency_id]
        old = target.allocation
        new = max(0.0, min(1.0, old + delta))
        actual_delta = new - old
        if abs(actual_delta) < 1e-6:
            return

        target.allocation = new
        others = [t for tid, t in self.tendencies.items() if tid != tendency_id]
        others_total = sum(t.allocation for t in others)
        if others_total > 0:
            for other in others:
                proportion = other.allocation / others_total
                other.allocation = max(0.0, other.allocation - actual_delta * proportion)

        self.normalize()

    def set_allocation(self, tendency_id: str, value: float) -> None:
        current = self.tendencies[tendency_id].allocation
        self.adjust_allocation(tendency_id, value - current)

    def rebalance_by_performance(self, learning_rate: float = 0.1) -> None:
        """Shift allocation toward tendencies whose stakes were validated more often."""
        active = [t for t in self.tendencies.values() if t.stakes_placed > 0]
        if len(active) < 2:
            return

        avg_rate = sum(t.validation_rate for t in active) / len(active)
        for t in active:
            self.adjust_allocation(t.id, (t.validation_rate - avg_rate) * learning_rate)
        self.calibrated = True

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if not self.tendencies:
            return "TendencySet(empty)"
        items = ", ".join(f"{tid}={t.allocation:.0%}" for tid, t in self.tendencies.items())
        return f"TendencySet({items})"

    def to_dict(self) -> dict:
        return {
            "tendencies": {tid: t.to_dict() for tid, t in self.tendencies.items()},
            "calibrated": self.calibrated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TendencySet":
        ts = cls()
        # Accept both new shape ("tendencies") and legacy shape ("agents")
        items = data.get("tendencies") or data.get("agents") or {}
        for tid, tdata in items.items():
            # Legacy entries used "tendency" instead of "id"
            if "id" not in tdata and "tendency" in tdata:
                tdata = dict(tdata)
                tdata["id"] = tdata.pop("tendency")
            ts.add(Tendency.from_dict(tdata))
        ts.calibrated = bool(data.get("calibrated", False))
        return ts

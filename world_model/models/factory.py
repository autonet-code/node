"""
Tendency factory: external authorities instantiate tendencies.

The engine itself does not invent tendencies. Some external entity --
a human author, a research pipeline, a domain ontology, an LLM agent
acting as curator -- supplies the input from which a Tendency is
constructed.

This module defines the protocol and a reference implementation. Real
deployments may wire factories that pull from Wikidata, parse a world
canon document, or query an authority service over the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Protocol

from .tendency import Tendency, TendencySet


@dataclass
class TendencySpec:
    """Authority-supplied input from which a Tendency is built.

    Why this and not just constructing Tendency directly? Because the
    factory pattern documents the boundary: anything inside the engine
    must come through it, including in tests. That keeps the
    "tendencies are externally authored" constraint visible in code.
    """

    id: str
    initial_allocation: float = 0.0
    description: str = ""
    initial_claim: str = ""             # The flagship claim this tendency proposes
    metadata: dict = field(default_factory=dict)


class TendencyFactory(Protocol):
    """Protocol for an authority that instantiates tendencies."""

    def instantiate(self, spec: TendencySpec) -> Tendency: ...

    def build_set(self, specs: Iterable[TendencySpec]) -> TendencySet: ...


@dataclass
class DefaultTendencyFactory:
    """Reference factory: constructs Tendency directly from a spec.

    Stores the initial claim in metadata so the arena's proposal phase
    can read it without a separate lookup.
    """

    def instantiate(self, spec: TendencySpec) -> Tendency:
        meta = dict(spec.metadata)
        if spec.initial_claim:
            meta.setdefault("initial_claim", spec.initial_claim)
        return Tendency(
            id=spec.id,
            allocation=spec.initial_allocation,
            description=spec.description,
            metadata=meta,
        )

    def build_set(self, specs: Iterable[TendencySpec]) -> TendencySet:
        ts = TendencySet()
        for spec in specs:
            ts.add(self.instantiate(spec))
        ts.normalize()
        return ts


# Convenience: the seven personality tendencies expressed as specs, for
# callers who still want the legacy roster as a starting point. This is
# a *default*, not a built-in -- nothing else in the engine references
# it. Callers building physics or fictional-world arenas should write
# their own specs.
LEGACY_PERSONALITY_SPECS: list[TendencySpec] = [
    TendencySpec(
        id="survival",
        initial_allocation=0.18,
        description="Physical safety, resource acquisition, risk mitigation",
    ),
    TendencySpec(
        id="connection",
        initial_allocation=0.20,
        description="Relationships, belonging, community, being known",
    ),
    TendencySpec(
        id="comfort",
        initial_allocation=0.18,
        description="Ease, pleasure, avoiding pain, reducing friction",
    ),
    TendencySpec(
        id="status",
        initial_allocation=0.12,
        description="Social standing, achievement, recognition",
    ),
    TendencySpec(
        id="autonomy",
        initial_allocation=0.12,
        description="Independence, self-determination, freedom from constraint",
    ),
    TendencySpec(
        id="meaning",
        initial_allocation=0.10,
        description="Significance, impact, legacy, purpose beyond self",
    ),
    TendencySpec(
        id="curiosity",
        initial_allocation=0.10,
        description="Knowledge, understanding, exploration, novelty",
    ),
]


def build_legacy_personality_set(
    factory: Optional[TendencyFactory] = None,
) -> TendencySet:
    """Reproduce the historic seven-tendency roster via the factory.

    Useful for tests that want the same starting allocations the
    pre-redesign code produced.
    """
    factory = factory or DefaultTendencyFactory()
    return factory.build_set(LEGACY_PERSONALITY_SPECS)

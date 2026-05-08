"""
Dynamics - Operations that produce or modify equilibria.

The dynamics layer is the engine's active surface. As of the
open-roster redesign, the canonical operation is reseed-and-equilibrate:
substitute tendencies, calibrate to a fixpoint, return a new state.

Legacy modules (arena.py, trainer.py) reference the pre-redesign closed
agent roster and are not imported here. They remain on disk and will
be revisited or removed as the engine settles.
"""

from .reseed import (
    PresentState,
    Substitution,
    ReseedResult,
    reseed_and_equilibrate,
)

__all__ = [
    "PresentState",
    "Substitution",
    "ReseedResult",
    "reseed_and_equilibrate",
]

"""
Levels of Detail (LOD) for tendencies.

A LOD level is a contract with two clauses:

  - **Epistemic:** the level answers a defined class of questions with a
    defined confidence floor. What it claims to know.
  - **Structural:** the level fits in a fixed byte budget when serialized.
    The budget enforces a compression policy and gives the engine a
    predictable memory layout for piping, caching, and SIMD.

The two clauses are coupled. The byte budget *is* what forces the
compression policy; the compression policy *is* what the level knows.
"What fits in N bytes" defines epistemic resolution.

Sizes are strict-fixed: every tendency at LOD K serializes to exactly
``budget_bytes`` bytes, padded if the actual content is smaller. This
makes arrays of LOD-K tendencies dense and stride-uniform.

This module defines:

  - ``LODLevel``: one rung of a ladder (level number + budget + schema).
  - ``LODProtocol``: a domain-specific ladder. A protocol is universal in
    interface but domain-specific in content: personality and Morphic
    declare different ladders, the engine consumes either uniformly.
  - ``LODPayload``: the fixed-size record at a given level, with explicit
    pack/unpack to bytes.
  - ``Tendency.at_lod(K)``: promote/demote a tendency to level K, paid
    by the appropriate protocol method.

Promotion (LOD K -> K+1) is reconstructive: the protocol needs the
engine's calibration substrate to materialize the missing layer.
Demotion (LOD K -> K-1) is deterministic and lossy: it discards
information by a known rule.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol


# ---------------------------------------------------------------------------
# Level / protocol primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LODLevel:
    """One rung of a level-of-detail ladder.

    Parameters
    ----------
    level:
        Integer level number. 0 = highest compression, larger = more detail.
    name:
        Human-readable label, for logs and debugging.
    budget_bytes:
        Strict byte budget for serialized payloads at this level. Records
        are padded to exactly this size. Choose budgets so that level K+1
        is at least as large as level K.
    epistemic:
        Free-form description of what questions this level answers.
        Documentation only -- not enforced.
    """

    level: int
    name: str
    budget_bytes: int
    epistemic: str = ""


class LODProtocol(Protocol):
    """A domain-specific level-of-detail ladder.

    Implementations declare the levels their domain supports and provide
    reduce/expand functions between adjacent levels.

    The engine never asks "what is LOD 3" -- it asks the protocol.
    """

    @property
    def domain(self) -> str: ...

    @property
    def levels(self) -> list[LODLevel]: ...

    def level(self, k: int) -> LODLevel: ...

    def max_level(self) -> int: ...

    def reduce(self, payload: "LODPayload") -> "LODPayload":
        """Demote payload by one level. Deterministic, lossy."""
        ...

    def expand(self, payload: "LODPayload", context: object = None) -> "LODPayload":
        """Promote payload by one level. Reconstructive; may require context."""
        ...


# ---------------------------------------------------------------------------
# Fixed-size payload
# ---------------------------------------------------------------------------


@dataclass
class LODPayload:
    """A fixed-size record at a specific level of detail.

    The payload is conceptually a struct: a small set of typed fields
    that fit within ``level.budget_bytes`` when packed. We store fields
    as a Python dict (for ergonomics) and provide ``to_bytes`` /
    ``from_bytes`` for the wire format.

    The wire format is intentionally simple: we serialize fields in a
    protocol-defined order, each as a fixed-width segment, and pad the
    result to exactly ``budget_bytes``. Field-level encoding is
    delegated to the protocol via ``schema_encoder`` -- the LOD module
    does not care what fields a domain uses, only that they fit.
    """

    level: LODLevel
    fields: dict[str, object] = field(default_factory=dict)

    # Optional callbacks supplied by the protocol; if absent, the default
    # encoder is a JSON dump truncated/padded to the budget.
    schema_encoder: Optional[Callable[[dict], bytes]] = None
    schema_decoder: Optional[Callable[[bytes], dict]] = None

    def to_bytes(self) -> bytes:
        """Pack to exactly ``self.level.budget_bytes``. Pad with NUL.

        Wire format: 4-byte little-endian uint32 length prefix, followed
        by ``budget_bytes - 4`` bytes of payload (raw content padded
        with NUL). Total is exactly ``budget_bytes``.
        """
        encoder = self.schema_encoder or _default_json_encoder
        raw = encoder(self.fields)
        budget = self.level.budget_bytes
        body_capacity = budget - 4
        if len(raw) > body_capacity:
            raise ValueError(
                f"LOD-{self.level.level} payload of {len(raw)} bytes exceeds "
                f"body capacity {body_capacity} (budget={budget}, "
                f"level={self.level.name})"
            )
        prefix = struct.pack("<I", len(raw))
        body = raw + b"\x00" * (body_capacity - len(raw))
        return prefix + body

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        level: LODLevel,
        schema_encoder: Optional[Callable[[dict], bytes]] = None,
        schema_decoder: Optional[Callable[[bytes], dict]] = None,
    ) -> "LODPayload":
        if len(data) != level.budget_bytes:
            raise ValueError(
                f"LOD-{level.level} expects exactly {level.budget_bytes} bytes, "
                f"got {len(data)}"
            )
        n = struct.unpack("<I", data[:4])[0]
        body = data[4:4 + n]
        decoder = schema_decoder or _default_json_decoder
        fields = decoder(body)
        return cls(
            level=level,
            fields=fields,
            schema_encoder=schema_encoder,
            schema_decoder=schema_decoder,
        )

    def byte_size(self) -> int:
        """Always equal to the level's budget. Provided for symmetry."""
        return self.level.budget_bytes


def _default_json_encoder(fields: dict) -> bytes:
    import json
    return json.dumps(fields, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _default_json_decoder(raw: bytes) -> dict:
    import json
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


# ---------------------------------------------------------------------------
# Reference protocol for tests / scaffolding
# ---------------------------------------------------------------------------


@dataclass
class TrivialLODProtocol:
    """A minimal four-level protocol for tests and scaffolding.

    Levels:

      0  identity:       {id, allocation}                         32 bytes
      1  recent:         + {recent_validation_rate}              128 bytes
      2  lifecycle:      + {birth_epoch, claim_mutations_count}  256 bytes
      3  full:           + {claim, description, metadata}        1024 bytes

    Reduce drops fields that are not known to the lower level. Expand
    fills missing fields with conservative defaults. This is a
    test-grade protocol: it does not perform real reconstruction.
    Domains will subclass / replace it.
    """

    domain: str = "trivial"

    LEVELS: tuple[LODLevel, ...] = (
        LODLevel(0, "identity",  64,    "id and current allocation"),
        LODLevel(1, "recent",    128,   "+ recent validation rate"),
        LODLevel(2, "lifecycle", 256,   "+ birth + mutation count"),
        LODLevel(3, "full",      2048,  "+ claim, description, full metadata"),
    )

    # Schema: which fields are known at each level. A field's first
    # appearance level is where it gets introduced when expanding;
    # demotion drops fields whose level > target.
    FIELD_FIRST_LEVEL: dict[str, int] = field(default_factory=lambda: {
        "id":                       0,
        "allocation":               0,
        "recent_validation_rate":   1,
        "birth_epoch":              2,
        "claim_mutations_count":    2,
        "claim":                    3,
        "description":              3,
        "metadata":                 3,
    })

    @property
    def levels(self) -> list[LODLevel]:
        return list(self.LEVELS)

    def level(self, k: int) -> LODLevel:
        if not 0 <= k < len(self.LEVELS):
            raise IndexError(f"no level {k} in protocol (max {self.max_level()})")
        return self.LEVELS[k]

    def max_level(self) -> int:
        return len(self.LEVELS) - 1

    def reduce(self, payload: LODPayload) -> LODPayload:
        """LOD K -> K-1: drop fields whose first-appearance level > K-1."""
        if payload.level.level == 0:
            return payload  # nothing to reduce
        target = self.level(payload.level.level - 1)
        kept = {
            k: v for k, v in payload.fields.items()
            if self.FIELD_FIRST_LEVEL.get(k, 99) <= target.level
        }
        return LODPayload(level=target, fields=kept)

    def expand(self, payload: LODPayload, context: object = None) -> LODPayload:
        """LOD K -> K+1: introduce fields newly available at K+1.

        Test-grade: fills with defaults derived from context where
        possible, otherwise sentinel values. Real protocols would
        invoke the calibration substrate.
        """
        if payload.level.level >= self.max_level():
            return payload
        target = self.level(payload.level.level + 1)
        new_fields = dict(payload.fields)
        for k, first_level in self.FIELD_FIRST_LEVEL.items():
            if first_level == target.level and k not in new_fields:
                new_fields[k] = _default_value_for(k, context)
        return LODPayload(level=target, fields=new_fields)


def _default_value_for(field_name: str, context: object) -> object:
    """Test-grade default supplier. Real protocols replace this."""
    defaults = {
        "recent_validation_rate":  0.0,
        "birth_epoch":             0,
        "claim_mutations_count":   0,
        "claim":                   "",
        "description":             "",
        "metadata":                {},
    }
    if isinstance(context, dict) and field_name in context:
        return context[field_name]
    return defaults.get(field_name, None)


# ---------------------------------------------------------------------------
# Convenience: walk the ladder
# ---------------------------------------------------------------------------


def to_level(
    payload: LODPayload,
    target_level: int,
    protocol: LODProtocol,
    context: object = None,
) -> LODPayload:
    """Walk the ladder repeatedly until we land on ``target_level``."""
    if target_level < 0 or target_level > protocol.max_level():
        raise IndexError(
            f"target level {target_level} outside protocol range "
            f"[0, {protocol.max_level()}]"
        )
    cur = payload
    while cur.level.level > target_level:
        cur = protocol.reduce(cur)
    while cur.level.level < target_level:
        cur = protocol.expand(cur, context=context)
    return cur

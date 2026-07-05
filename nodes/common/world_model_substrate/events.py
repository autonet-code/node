"""Event stream: the substrate's protocol-layer output.

A solver's contribution per task is a sequence of events. The
aggregator replays them on the global world to derive the new state;
the engine's equilibration handles score propagation and lineage.

Two event types
---------------

  - SubClaimSprouted: a new sub-claim node was added under a parent.
    Carries content-addressed node_id (or solver-local id, if the
    engine hasn't switched to content-addressing yet), parent_id,
    position, coords, polarity_axis, content, and the tendency_id
    the sprout lives under.

  - ObservationAdded: an observation was attached to the world. Carries
    obs_id and coords. The act of adding may absorb the observation
    into one or more tendencies' frames depending on equilibration.

Each event carries the author (agent_id) and a sequence index. The
sequence index is unique within the solver's submission and lets us
replay deterministically.

Why events instead of stake-deltas
----------------------------------

In the architecture as understood:
  - Score on a node is the equilibrium consequence of the events
    that landed on/under it, not an independently-tracked quantity.
  - Mint is computed from score-change at epoch close, attributed to
    the events that caused it.
  - There is no per-node stake bookkeeping that needs to be carried
    in the protocol; events are the unit of contribution.

So the solver's output is its event stream. The aggregator merges
streams, replays on the global world, and reads scores after.

Single-parent vs multi-parent shape
-----------------------------------

The protocol-layer event carries a primary (parent_id, position,
tendency_id) tuple. There's an optional `parents` list field for
forward-compatibility with explicit multi-parent emission, but in
practice the engine handles co-parenting at replay time: when
multiple events land on the same content-addressed coordinate
hash, the engine's sprout_child collision detection + cross-
tendency edge discovery accumulates parent edges automatically.

So solvers can emit single-parent events (the natural shape) and
federation merge still produces correctly co-parented nodes on
the live world. The `parents` list is there if a solver has
explicit reason to emit a multi-parented event in one shot, but
it's not required for the federation story to work.

Don't "fix" the protocol by making multi-parent emission
mandatory on SubClaimSprouted — single-parent is the correct
shape for solvers, and the engine is what makes federation merge
work.

See world-model docs/substrate-architecture.md for the
post-and-coparent + dedup mechanics on the engine side.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SubClaimSprouted:
    """A sub-claim node was sprouted under one or more parents.

    Single-parent events (legacy) populate `parent_id` + `tendency_id`
    + `position` directly. Multi-parent events (post-and-coparent
    refactor) populate the optional `parents` list with one entry per
    edge: each entry is {"parent_id": ..., "position": "pro"|"con",
    "tendency_id": ...}. The aggregator treats single-parent events as
    a length-1 parents list at apply time.
    """
    kind: str = "sub_claim_sprouted"
    seq: int = 0
    author_agent: str = ""
    tendency_id: str = ""
    parent_id: str = ""           # solver-side; aggregator will remap to live id
    node_id: str = ""             # solver-side; aggregator may remap or accept
    position: str = "pro"         # "pro" | "con"
    coords: List[float] = field(default_factory=list)
    polarity_axis: List[float] = field(default_factory=list)
    content: str = ""
    # Optional reference to an Observation this sub-claim was sprouted
    # for. Carried through replay so locate can map back to the
    # underlying work unit.
    observation_id: str = ""
    # Multi-parent edge list (post-and-coparent refactor). When empty,
    # the aggregator falls back to (parent_id, position, tendency_id).
    # When non-empty, takes precedence; each entry has keys
    # parent_id, position, tendency_id.
    parents: List[Dict[str, str]] = field(default_factory=list)
    # Targeted-dispute support: when True, replay adds one unit-weight
    # post by `author_agent` on the sprouted node. This is what gives
    # an authored CON immediate standing against its target instead of
    # waiting for tendencies to organically stake it. Deterministic:
    # same event -> same post on every daemon.
    author_post: bool = False
    # Two-plane substrate (docs/two_plane_inference.md): sha256 hex
    # digest of the full work-unit payload stored in the blob store /
    # artifact index. The digest rides the event so peers can fetch the
    # artifact lazily; it does NOT participate in node ids, coords, or
    # equilibration, so epoch close stays bit-identical. Serialized ONLY
    # when non-empty (below) so replay of old logs — which never carried
    # this field — produces byte-identical event dicts and batch hashes.
    artifact_digest: str = ""
    # Tool substrate (docs/tool_substrate.md): consensus-carried manifest
    # metadata on the REGISTRATION sprout ({"trust_class", "author"}).
    # Epoch-close tool mint must not depend on whether a daemon has the
    # manifest blob replicated — a blob-store miss must never fork the
    # close — so the two fields mint needs ride the canonical event.
    # Same back-compat rule as artifact_digest: omitted when empty.
    manifest_meta: Dict[str, str] = field(default_factory=dict)
    # Evidence-replay CON (docs/tool_substrate.md — Evidence section;
    # phase10 follow-up). A CON-position sprout disputing a pinned tool
    # may carry a REPRODUCIBLE failing invocation: {"args_json",
    # "expected_digest" | "expected_error", "actual_digest"}. Any daemon
    # can VOLUNTARILY re-run the pinned code with these args
    # (ToolStore.replay_evidence) and, if it confirms, post a normal
    # author_post support sprout under the CON. Evidence RECRUITS honest
    # verifiers; it does NOT weight standing directly (the deterministic
    # close prices the resulting support posts with no new math — that is
    # phase10's lesson encoded). It plays no part in node ids, coords, or
    # equilibration. Same back-compat rule as every optional field:
    # serialized ONLY when present, so pre-evidence logs — and their
    # canonical-ordering batch hashes — are byte-identical.
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Back-compat: omit the optional artifact_digest when empty so
        # events recorded before this field existed serialize identically
        # (and their canonical-ordering batch hashes stay unchanged).
        if not self.artifact_digest:
            d.pop("artifact_digest", None)
        if not self.manifest_meta:
            d.pop("manifest_meta", None)
        if not self.evidence:
            d.pop("evidence", None)
        return d


@dataclass
class ObservationAdded:
    """An observation was added to the world during training."""
    kind: str = "observation_added"
    seq: int = 0
    author_agent: str = ""
    obs_id: str = ""
    coords: List[float] = field(default_factory=list)
    label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolUsed:
    """A tool invocation receipt (tool substrate, docs/tool_substrate.md).

    Consensus event: rides the same gossip/canonical-ordering rail as
    sprouts so every daemon computes identical per-manifest usage counts
    at epoch close (mint ∝ standing × usage; attested-class decay resets
    on fresh receipts). It does NOT touch the claim graph — replay
    (``aggregate.apply_events``) skips it by design.

    ``author_agent`` is the CALLER — the receipt is an attestation by
    whoever paid for the call, so wash-trading standing has a floor
    price. ``tool_author`` is denormalized from the manifest for cheap
    epoch aggregation.
    """
    kind: str = "tool_used"
    seq: int = 0
    author_agent: str = ""        # the caller (who attests + pays)
    manifest_digest: str = ""
    tool_author: str = ""
    receipt_digest: str = ""      # blob: {arguments_digest, ok, ts, ...}
    ok: bool = True
    fee_atn: float = 0.0
    # Demonstrated-utility coordinates (tool_substrate.md v2): embedding
    # of the problem context the CALLER was solving, in the usefulness
    # space. Retrieval ranks tools by local DENSITY of these points
    # (anti-SEO: description proposes, usage disposes). Serialized only
    # when present — same back-compat rule as every optional event field.
    problem_coords: List[float] = field(default_factory=list)
    # Two receipt tiers (spec: Attestation section). Mechanical receipts
    # (attested=False) are per-call exhaust — local ledger material,
    # worth nothing in mint. Cognitive attestations (attested=True) are
    # the per-WORK-ITEM reflection where the caller judged the tool;
    # they are the ONLY usage the mint counts. score/review text are
    # debate material, never mint inputs; the text lives in the blob
    # store under review_digest.
    attested: bool = False
    score: float = 0.0
    review_digest: str = ""
    # Resident-tool grammar (spec: Resident tools, loadouts, distros):
    # digest of the harness distro the attesting agent ran under —
    # atomic with the attestation, no swap-time reasoning. Feeds the
    # ADOPTION term at close (distinct fleets per loadout), never
    # per-call credit.
    loadout: str = ""
    # Vetting (spec: Vetting section): the THIRD receipt flavor. A vet
    # is not usage — the validator read the pinned code and attests
    # code-adheres-to-manifest + no-malice (ok=True) or objects
    # (ok=False; debate material, does not count toward greenlight).
    # Vet events are excluded from usage/attestation aggregation; at
    # close they accumulate toward the manifest's GREENLIGHT (N vets
    # from distinct fleets) which gates mint eligibility. review_digest
    # carries the vet report blob. Serialized only when True.
    vet: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if not self.problem_coords:
            d.pop("problem_coords", None)
        # Back-compat: attestation fields serialize only on attested
        # receipts, so pre-attestation event logs hash identically.
        if not self.attested:
            d.pop("attested", None)
            d.pop("score", None)
        if not self.review_digest:
            d.pop("review_digest", None)
        if not self.loadout:
            d.pop("loadout", None)
        if not self.vet:
            d.pop("vet", None)
        return d


Event = Any   # SubClaimSprouted | ObservationAdded | ToolUsed


def event_from_dict(d: Dict[str, Any]) -> Event:
    kind = d.get("kind")
    if kind == "sub_claim_sprouted":
        return SubClaimSprouted(**d)
    if kind == "observation_added":
        return ObservationAdded(**d)
    if kind == "tool_used":
        return ToolUsed(**d)
    raise ValueError(f"unknown event kind: {kind!r}")


# ---------------------------------------------------------------------------
# Score snapshot at epoch boundary
# ---------------------------------------------------------------------------


def snapshot_node_scores(world) -> Dict[str, float]:  # type: ignore[no-untyped-def]
    """Map node_id -> net_score across all tendencies' trees."""
    out: Dict[str, float] = {}
    for tendency in world.tendencies.values():
        for node in tendency.tree.all_nodes():
            out[node.id] = node.net_score
    return out

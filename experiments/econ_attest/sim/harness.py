"""Agent-based sim harness for the Substrate v3 tool economy.

Drives the REAL close-time economics — ``federated_epoch_close`` /
``compute_tool_mint`` and the position-drift machinery in
``nodes/common/federated_reconcile.py`` — over synthetic agent
populations for hundreds of epochs, with no libp2p and no chain.

Fidelity rule: every mint / position number comes from the production
functions. This module only *fabricates the canonical event stream*
(the exact shapes tests/test_tool_mint.py uses) and *tracks the
cross-epoch carry state* (registrations, vetting, positions,
reputation → voice weights) the real driver would carry between closes.

What is REAL (imported, not reimplemented):
  - usage_term / mint (household collapse, log1p damper, voice weight,
    owner + wire exclusions, composition fan-out): compute_tool_mint.
  - vetting greenlight (distinct-fleet quorum, royalty split, weight =
    1/(1+busts)): compute_tool_mint.
  - position drift (per-axis mint-weighted centroid, author prior mass):
    compute_tool_mint.
  - emission-pool normalization (fixed-pie zero-sum): apply_emission_pool
    via federated_epoch_close(emission_pool=...).
  - the discovery ranking LIFT formula tanh(mean(correctness,simplicity))
    on the DRIFTED head: replicated verbatim from infer._infer_artifacts
    (see rank_tools) because the production ranker needs a live
    artifact_index + blob_store we don't stand up.

What is STUBBED (documented, thin):
  - voice weights: we compute ε + household_reputation/supply exactly as
    voice_state.read_voice_state does, but from the sim's reputation
    ledger instead of on-chain checkpoints (no chain).
  - discovery COSINE: we hand each tool a scalar ``topic_match`` in
    [0,1] standing in for artifact_index cosine relevance to a query;
    the LIFT applied to it is the real formula. (The task explicitly
    permits this simplification.)
  - reputation = cumulative minted ATN (1:1), matching the dual-token
    lockstep (Substrate.sol mints reputation == ATN per training event).
"""

from __future__ import annotations

import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from nodes.common.canonical_ordering import canonical_order  # noqa: E402
from nodes.common.event_gossip import EventBatch, Keypair  # noqa: E402
from nodes.common.federated_reconcile import (  # noqa: E402
    VOICE_EPSILON,
    federated_epoch_close,
)

# BASE_EMISSION_PER_EPOCH was deleted from federated_reconcile (fees-only
# emission, Decision 2026-07-10). This historical harness kept its own base
# as an engineering default; preserve it locally so the sim package still
# imports. The live close no longer uses any base emission.
BASE_EMISSION_PER_EPOCH = 100.0
from nodes.common.world_model_substrate.adapter import CHARTER, N_DIMS  # noqa: E402

EMBED_DIM = 32  # tiny embedding tail; mint math is width-indifferent

# Charter axis indices used by the discovery ranking lift (the two
# "usefulness" axes). Resolved by id so a charter reorder can't skew it,
# exactly as infer._infer_artifacts does.
_AXIS_BY_ID = {str(e["id"]): int(e["axis_index"]) for e in CHARTER}
CORRECTNESS_AXIS = _AXIS_BY_ID["correctness"]
SIMPLICITY_AXIS = _AXIS_BY_ID["simplicity"]
USEFUL_AXES = [CORRECTNESS_AXIS, SIMPLICITY_AXIS]


# ---------------------------------------------------------------------------
# Population model
# ---------------------------------------------------------------------------

@dataclass
class Agent:
    """A consensus actor. ``owner`` binds it to a household (a wallet);
    unbound (owner == "") agents are their own household — matching
    ``_household()`` in federated_reconcile."""
    agent_id: str
    owner: str = ""          # "" = unbound = own household
    kind: str = "honest"     # honest | sybil | attacker | user | vetter ...
    reputation: float = 0.0  # cumulative minted ATN (== reputation, 1:1)


@dataclass
class Tool:
    digest: str
    author: str              # agent_id of the author
    kind: str = "honest"     # honest | sybil | victim | clone | service ...
    true_quality: float = 0.5   # in [-1, 1]; drives honest review scores
    topic_match: float = 0.7    # in [0, 1]; stands in for retrieval cosine
    deps: List[str] = field(default_factory=list)  # declared dep digests
    born_epoch: int = 0
    node_id: str = ""


def _digest(rng: random.Random) -> str:
    return "".join(rng.choice("0123456789abcdef") for _ in range(64))


def _coords(rng: random.Random, axis: int = 4) -> List[float]:
    """Distinct-per-call coords so content-addressed sprouting keeps each
    tool a separate node (identical coords merge nodes and corrupt
    per-tool attribution — a subtlety the quarantined sim documents)."""
    out = [0.0] * (N_DIMS + EMBED_DIM)
    out[axis] = 0.6 + rng.random() * 0.3
    for j in range(N_DIMS, N_DIMS + EMBED_DIM):
        out[j] = rng.uniform(-1.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# Event fabrication (shapes mirror tests/test_tool_mint.py exactly)
# ---------------------------------------------------------------------------

def registration_event(seq: int, tool: Tool, rng: random.Random) -> Dict[str, Any]:
    meta = {"trust_class": "pinned", "author": tool.author}
    if tool.deps:
        meta["deps"] = ",".join(tool.deps)
    return {
        "kind": "sub_claim_sprouted",
        "seq": seq,
        "author_agent": tool.author,
        "tendency_id": "correctness",
        "parent_id": "solver_root",
        "node_id": tool.node_id,
        "position": "pro",
        "coords": _coords(rng),
        "polarity_axis": _coords(rng),
        "content": f"tool {tool.digest[:8]}",
        "author_post": True,
        "artifact_digest": tool.digest,
        "manifest_meta": meta,
    }


def receipt_event(seq: int, caller: str, tool: Tool, *, ok: bool = True,
                  axes: Optional[Dict[str, float]] = None,
                  loadout: str = "") -> Dict[str, Any]:
    ev: Dict[str, Any] = {
        "kind": "tool_used",
        "seq": seq,
        "author_agent": caller,
        "manifest_digest": tool.digest,
        "tool_author": tool.author,
        "receipt_digest": f"r{seq:08d}" * 4,
        "ok": ok,
        "fee_atn": 0.0,
        "attested": True,     # cognitive attestation = the only mint input
        "score": 0.8,
    }
    if axes:
        ev["axes"] = dict(axes)
    if loadout:
        ev["loadout"] = loadout
    return ev


def vet_event(seq: int, vetter: str, tool: Tool) -> Dict[str, Any]:
    return {
        "kind": "tool_used",
        "seq": seq,
        "author_agent": vetter,
        "manifest_digest": tool.digest,
        "tool_author": tool.author,
        "receipt_digest": f"v{seq:08d}" * 4,
        "ok": True,
        "fee_atn": 0.0,
        "vet": True,
    }


def batches(groups: List[List[Dict[str, Any]]], keypair: Keypair,
            rpb: str = "rpb_sim") -> List[EventBatch]:
    """One EventBatch per group under a single signing key. Registration
    and receipts MUST use different keypairs (wire-level dedup zeroes
    same-key attestations); vetters each need their own keypair to be
    distinct wire identities."""
    chain: List[EventBatch] = []
    prev_hash = b""
    for i, group in enumerate(groups, start=1):
        b = EventBatch(
            rpb_address=rpb,
            sender_pubkey=keypair.public_key,
            batch_seq=i,
            events=list(group),
            prev_batch_hash=prev_hash,
            timestamp=1_700_000_000.0 + i,
        )
        chain.append(b)
        prev_hash = b.content_hash()
    return chain


# ---------------------------------------------------------------------------
# Discovery ranking — the REAL lift formula on the drifted head
# ---------------------------------------------------------------------------

def tool_head(positions: Dict[str, Dict[str, Any]], digest: str) -> List[float]:
    pos = positions.get(digest)
    if not pos:
        return [0.0] * N_DIMS
    head = [float(x) for x in (pos.get("head") or [])]
    if len(head) != N_DIMS:
        return [0.0] * N_DIMS
    return head


def rank_score(tool: Tool, positions: Dict[str, Dict[str, Any]]) -> float:
    """final = base_cosine * (1 + tanh(mean(correctness, simplicity)))

    Replicates infer._infer_artifacts lines 321-327 verbatim: the manifest
    lift is tanh of the mean of the DRIFTED correctness+simplicity axes.
    ``base`` is the tool's topic_match (our scalar stand-in for cosine).
    """
    head = tool_head(positions, tool.digest)
    rating = sum(head[i] for i in USEFUL_AXES) / len(USEFUL_AXES)
    return tool.topic_match * (1.0 + math.tanh(rating))


def discovery_ranks(tools: List[Tool],
                    positions: Dict[str, Dict[str, Any]],
                    *, only_greenlit: Optional[set] = None) -> List[Tuple[Tool, float]]:
    """Tools sorted by descending rank_score. If ``only_greenlit`` is
    given, un-greenlit tools are dropped (they mint nothing, but they DO
    still appear in retrieval per attack #4 — callers decide which set to
    pass)."""
    pool = tools
    if only_greenlit is not None:
        pool = [t for t in tools if t.digest in only_greenlit]
    scored = [(t, rank_score(t, positions)) for t in pool]
    scored.sort(key=lambda ts: (-ts[1], ts[0].digest))
    return scored


# ---------------------------------------------------------------------------
# Voice weights — exactly the voice_state.read_voice_state formula,
# computed from the sim reputation ledger instead of the chain.
# ---------------------------------------------------------------------------

def household_of(agent: Agent) -> str:
    return agent.owner or agent.agent_id


def voice_weights_from_reputation(
    agents: Dict[str, Agent], *, epsilon: float = VOICE_EPSILON,
) -> Optional[Dict[str, float]]:
    """weight(house) = epsilon + household_reputation / reputationSupply.

    Returns None (uniform weight 1.0, pre-chain regime) when total
    reputation is zero — matching voice_state's "no prior anchor" path
    (epoch 1, nothing minted yet). Once anything has minted, every
    household gets a real weight and unknown households default to
    VOICE_EPSILON inside compute_tool_mint.
    """
    house_rep: Dict[str, float] = {}
    for a in agents.values():
        house_rep[household_of(a)] = house_rep.get(household_of(a), 0.0) + a.reputation
    supply = sum(house_rep.values())
    if supply <= 0.0:
        return None
    return {h: round(epsilon + rep / supply, 9)
            for h, rep in sorted(house_rep.items())}


def owner_map_from_agents(agents: Dict[str, Agent]) -> Dict[str, str]:
    """agent_id -> owner wallet, only for bound agents (matches the
    chain-derived map compute_tool_mint expects; unbound agents absent)."""
    return {a.agent_id: a.owner for a in agents.values() if a.owner}


# ---------------------------------------------------------------------------
# The simulation engine
# ---------------------------------------------------------------------------

@dataclass
class EpochPlan:
    """What happens in one epoch, produced by a scenario's step():
      - registrations: tools sprouting THIS epoch (new manifests)
      - usages: (caller_agent_id, tool, ok, axes|None, loadout) receipts
      - vets: (vetter_agent_id, tool) affirmative vet receipts
    Callers group by signing identity downstream; the engine assigns
    keypairs to keep the wire-dedup / distinct-fleet semantics honest.
    """
    registrations: List[Tool] = field(default_factory=list)
    usages: List[Tuple[str, Tool, bool, Optional[Dict[str, float]], str]] = \
        field(default_factory=list)
    vets: List[Tuple[str, Tool]] = field(default_factory=list)


class Simulation:
    """Multi-epoch driver. Holds the carry state (registrations, vetting,
    positions) the real close threads between epochs, plus the sim-side
    reputation ledger that feeds voice weights.

    A *scenario* subclasses/uses this by supplying agents, an initial
    tool set, and a per-epoch ``step`` callback returning an EpochPlan.
    """

    def __init__(self, *, seed: int, epsilon: float = VOICE_EPSILON,
                 base_emission: float = BASE_EMISSION_PER_EPOCH):
        self.rng = random.Random(seed)
        self.epsilon = epsilon
        self.base_emission = base_emission
        self.agents: Dict[str, Agent] = {}
        self.tools: Dict[str, Tool] = {}          # digest -> Tool
        # carry state (rebuildable caches the driver threads between closes)
        self.tool_registrations: Dict[str, Dict[str, str]] = {}
        self.tool_vetting: Optional[Dict[str, Any]] = None
        self.tool_positions: Dict[str, Dict[str, Any]] = {}
        self.epoch = 0
        # signing-key assignment: each distinct household gets a stable
        # keypair so co-hosting (same-key) collapse only bites when we
        # *intend* it (e.g. author self-attestation on one key).
        self._keys: Dict[str, Keypair] = {}
        # recycled service fees added to the pool this epoch (attack 6 /
        # service_clone couple through here).
        self.recycled_fees: float = 0.0
        self.last_pool: float = base_emission

    # -- registration --------------------------------------------------
    def add_agent(self, agent: Agent) -> Agent:
        self.agents[agent.agent_id] = agent
        return agent

    def new_tool(self, author: str, *, kind: str = "honest",
                 true_quality: float = 0.5, topic_match: float = 0.7,
                 deps: Optional[List[str]] = None,
                 digest: Optional[str] = None) -> Tool:
        dg = digest or _digest(self.rng)
        node = f"tmnode_{len(self.tools):05d}_{dg[:8]}"
        t = Tool(digest=dg, author=author, kind=kind,
                 true_quality=true_quality, topic_match=topic_match,
                 deps=list(deps or []), born_epoch=self.epoch, node_id=node)
        self.tools[dg] = t
        return t

    def _key_for(self, sign_id: str) -> Keypair:
        kp = self._keys.get(sign_id)
        if kp is None:
            kp = self._keys[sign_id] = Keypair.generate()
        return kp

    # -- one epoch -----------------------------------------------------
    def run_epoch(self, plan: EpochPlan) -> Dict[str, Any]:
        """Fabricate the canonical batch for ``plan`` and drive the real
        close. Updates carry state + reputation ledger. Returns the raw
        close result plus a per-tool mint breakdown."""
        seq = 0
        # --- registration batches (grouped per author signing key) ----
        reg_by_signer: Dict[str, List[List[Dict[str, Any]]]] = {}
        for t in plan.registrations:
            seq += 1
            ev = registration_event(seq, t, self.rng)
            # authors sign registrations with a per-AUTHOR key so that an
            # author self-attesting on the same key is caught by wire dedup
            reg_by_signer.setdefault(f"author:{t.author}", []).append([ev])

        # --- usage receipts: signed per CALLER-HOUSEHOLD key, distinct
        # from any registration key. Group all of a caller's receipts in
        # one batch so its household collapse is realistic.
        use_by_signer: Dict[str, List[Dict[str, Any]]] = {}
        for (caller, tool, ok, axes, loadout) in plan.usages:
            seq += 1
            ev = receipt_event(seq, caller, tool, ok=ok, axes=axes,
                               loadout=loadout)
            house = household_of(self.agents[caller]) if caller in self.agents \
                else caller
            use_by_signer.setdefault(f"caller:{house}", []).append(ev)

        # --- vets: each vetter signs with its OWN key (distinct wire
        # identity) so distinct-fleet greenlight works as in production.
        vet_by_signer: Dict[str, List[Dict[str, Any]]] = {}
        for (vetter, tool) in plan.vets:
            seq += 1
            ev = vet_event(seq, vetter, tool)
            vet_by_signer.setdefault(f"vet:{vetter}", []).append(ev)

        all_batches: List[EventBatch] = []
        for signer, groups in sorted(reg_by_signer.items()):
            all_batches += batches(groups, self._key_for(signer))
        for signer, evs in sorted(use_by_signer.items()):
            all_batches += batches([evs], self._key_for(signer))
        for signer, evs in sorted(vet_by_signer.items()):
            all_batches += batches([evs], self._key_for(signer))

        voice = voice_weights_from_reputation(self.agents, epsilon=self.epsilon)
        owner_map = owner_map_from_agents(self.agents)
        pool = self.base_emission + self.recycled_fees
        self.last_pool = pool   # the pool this close normalized to

        result = federated_epoch_close(
            canonical_order(all_batches),
            emission_pool=pool,
            tool_registrations=self.tool_registrations or None,
            agent_owner_map=owner_map or None,
            voice_weights=voice,
            tool_vetting=self.tool_vetting,
            tool_positions=self.tool_positions or None,
        )

        # thread carry state forward
        self.tool_registrations = result["tool_registrations"]
        self.tool_vetting = result["tool_vetting"]
        self.tool_positions = result["tool_positions"]

        # accrue reputation from this epoch's agent_mint (1:1 with ATN).
        for agent_id, minted in result["agent_mint"].items():
            a = self.agents.get(agent_id)
            if a is not None:
                a.reputation += float(minted)
            # mint can also land on validators (royalty) who may be agents

        self.recycled_fees = 0.0   # consumed; scenarios re-inject each epoch
        self.epoch += 1
        return result

    # -- convenience: greenlit set ------------------------------------
    def greenlit_digests(self) -> set:
        vet = self.tool_vetting or {}
        return {d for d, m in (vet.get("manifests") or {}).items()
                if m.get("greenlit")}


# ---------------------------------------------------------------------------
# Honest-review helper: axis scores correlated with true quality + noise.
# ---------------------------------------------------------------------------

def honest_axes(tool: Tool, rng: random.Random, *, noise: float = 0.25
                ) -> Dict[str, float]:
    """Honest reviewer scores the two usefulness axes near the tool's
    true quality, with gaussian noise, clamped to [-1, 1]. (Alignment
    axes are left unscored: honest usage reviews usefulness.)"""
    def _s() -> float:
        return max(-1.0, min(1.0, tool.true_quality + rng.gauss(0.0, noise)))
    return {"correctness": _s(), "simplicity": _s()}


def conservation_check(result: Dict[str, Any], pool: float,
                       *, tol: float = 1e-6) -> None:
    """Assert total minted ATN does not exceed the emission pool.
    With apply_emission_pool, total == pool exactly whenever any raw mint
    occurred; with zero raw mint, total == 0. Either way total <= pool."""
    total = float(result.get("total_mint", 0.0))
    assert total <= pool + tol, f"conservation violated: {total} > pool {pool}"

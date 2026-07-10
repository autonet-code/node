"""Federated reconciliation: bit-identical agent_mint maps across daemons.

Phase 5.4 of native world-model integration.

Wraps ``world_model_substrate.reconcile.reconcile_epoch`` with two
guarantees that the underlying function does NOT make on its own:

  1. **Deterministic iteration order**. The wrapped function iterates
     a Python ``set`` of node ids and ``dict.items()`` of attribution
     maps. Float addition is non-associative, so different iteration
     orders produce different sums. We override the iteration to be
     sorted (lexicographic on node_id, agent_id) so every honest
     daemon performs the *same sequence of float operations*.

  2. **Output normalization**. Even with sorted iteration, IEEE 754
     edge cases (subnormals, ulp-level jitter from cross-platform
     math libraries) can produce trivially-different float results.
     We round all output mint/novelty values to ``OUTPUT_DECIMALS``
     decimal places before returning. This is a normalization step,
     not a precision loss for the chain side: the ``ChainSubmitter``
     scales by 1e6 anyway.

Why a wrapper rather than editing reconcile.py
----------------------------------------------

``reconcile_epoch`` is used by the existing per-task substrate path
(``aggregator/main.py`` etc.) where bit-identicality is not required.
A wrapper preserves backward compat while letting the federated
codepath get the stricter guarantee.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from world_model.generalized import World, equilibrate

from .canonical_ordering import CanonicalOrder
from .world_model_substrate.adapter import (
    _all_node_ids,
    _EventRecorder,
    build_charter_world,
)
from .world_model_substrate.aggregate import apply_events
from .world_model_substrate.mint_gate import (
    DEFAULT_CHARTER_IDS,
    charter_violation_score,
)
from .world_model_substrate.reconcile import (
    EpochSnapshots,
    _attribute_node_mint,
    _node_mint,
    _node_novelty,
    scale_node_agent_mint_by_violation,
    snapshot_node_scores_ordered,
)


logger = logging.getLogger(__name__)


# Number of decimal places we round mint/novelty outputs to.
# Empirically: 10 is far below any meaningful precision (substrate
# scores are typically O(10) to O(1000) and chain scaling truncates
# at 6 decimals), but well above IEEE 754 jitter (~15 sig figs).
OUTPUT_DECIMALS = 10

# Composition fan-out (docs/tool_substrate.md, Composition section):
# one attestation of a composite carries total weight 1 — the root
# keeps ROOT_SHARE, the remainder splits equally among declared deps,
# recursively to MAX_DEPTH. Cycles and unregistered deps FORFEIT their
# share (total may be < 1, never > 1): conservation of attestation.
COMPOSITE_ROOT_SHARE = 0.7
COMPOSITE_MAX_DEPTH = 4

# Vetting pipeline (docs/tool_substrate.md, Vetting section). A published
# manifest enters the CANDIDATE pool: visible, debatable, NOT mint-
# eligible. GREENLIGHT = affirmative vets from distinct fleets whose
# summed weights reach VET_QUORUM; validators then earn a conserved
# royalty slice of the tool's mint for VET_ROYALTY_EPOCHS closes — the
# royalty IS the stake: a post-greenlight charter bust (violation ≥
# VET_BUST_THRESHOLD on the manifest's claim node) zeroes the remaining
# royalty and permanently discounts each validator's future vet weight
# (weight = 1/(1+busts) — slashing without a staking contract).
# PROVISIONAL VALUES: quorum/share/epochs/threshold are economic
# parameters pending sim sweep + user blessing (sims/tool_economy).
VET_QUORUM = 2.0
VET_ROYALTY_SHARE = 0.1
VET_ROYALTY_EPOCHS = 8
VET_BUST_THRESHOLD = 0.5

# Household voice (docs/tool_substrate.md, Decision 2026-07-08 addendum,
# voice SOURCE = reputation not balance per the same-day "ATN = money,
# reputation = voice" decision). A "household" is the economic identity
# behind a caller: its registered owner wallet (proven on-chain by the
# EIP-712 owner binding), falling back to the agent id when unbound.
# Usage and review mass collapse per household BEFORE damping (log1p once
# per household — N agents under one owner are one voice, closing the
# per-agent log1p amplification), then multiply by the household's voice
# weight: epsilon + household_REPUTATION / rep_supply, LINEAR in
# reputation so splitting earnings across wallets never gains weight.
# VOICE_EPSILON is the floor for households the weight map doesn't know —
# it bounds what a zero-reputation identity can contribute on the MINT
# rail and lets a cold-start network (supply = 0) bootstrap.
#
# fees-only + REP-from-earnings (Decision 2026-07-10, docs/tool_substrate.md
# §5): the supply-pegged β cap is DELETED (tool users are spenders, spenders
# never earn REP, so honest usage is all zero-rep by construction and any β
# throttle on zero-rep weight zeroes honest author income). The ε MINT-weight
# floor survives — REP dilution, not β, is the youth defense now, and the sim
# validated the model with ε present. v4.1 (2026-07-09) drops this ε floor
# from DRIFT weight (drift = raw rep_share × credibility); it survives only
# here, on the mint rail. PROVISIONAL VALUE pending user blessing.
VOICE_EPSILON = 0.05

# ---------------------------------------------------------------------------
# Tool economy v4.1 "gradient trust" (ratified 2026-07-09) as amended by
# fees-only + REP-from-earnings (Decision 2026-07-10). The v3 vet GATE is
# removed (tools mint from first attested use); the surviving trust surface
# is (a) zero-rep reviews carry ZERO drift weight and (b) continuous
# reversal-aware credibility multiplies drift weight. The supply-pegged β
# cap and the rep/ATN mint split are DELETED (2026-07-10): the close
# computes MONEY ONLY — REP is a pure function of ratified earnings,
# claimed DAO-side (RepToken), never minted on the ATN close path.
# ---------------------------------------------------------------------------

# Continuous reversal-aware credibility (v4.1 [R5]). A tool's review
# book is re-scored every close: a household whose recorded review
# centroid deviates from the tool's CURRENT drifted head by more than
# CRED_DELTA (on a tool with review mass ≥ CRED_MASS_FLOOR) is docked
# proportionally; convergence restores. Floor CRED_FLOOR, symmetric
# passive/active recovery CRED_RECOVERY toward 1.0. Credibility
# multiplies DRIFT weight only, never mint. SEEDED (post-launch tunable).
CRED_DELTA = 0.7
CRED_FLOOR = 0.1
CRED_RECOVERY = 0.10
CRED_MASS_FLOOR = 3.0


# Fees-only emission (docs/tool_substrate.md, Decision 2026-07-10 §1):
#   pool(N) = recycled(N)
# where recycled(N) = Σ ServiceFee.burned between the previous two anchors
# (read on-chain by voice_state.read_voice_state, same pinned rail as voice
# weights). No base emission: ATN mints each epoch EXACTLY what fees burned
# that epoch — money is always demand-backed. Zero service volume => zero
# pool => zero mint, and that is correct behavior (the economy does not
# pretend to exist before demand does). Conservation by construction
# (Σ minted == Σ burned); wash-pumping the pool is a strict ATN loss. The
# fee/treasury split lives on-chain (Substrate.sol serviceFeeBps).


def _normalize_vetting(vetting: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Canonicalize carried-over vetting state (sorted, typed, copied).

    Shape: {"manifests": {digest: {vets: {agent: [sender,...]},
    greenlit, royalty_left, validators, busted}}, "busts": {agent: n}}.
    Derived purely from prior epochs' canonical events + world state,
    so — like ``registrations`` — the persisted copy is a rebuildable
    cache and identical on every honest daemon.
    """
    state: Dict[str, Any] = {"manifests": {}, "busts": {}}
    if not vetting:
        return state
    for digest, m in sorted((vetting.get("manifests") or {}).items()):
        state["manifests"][str(digest)] = {
            "vets": {str(a): sorted(str(s) for s in senders)
                     for a, senders in sorted((m.get("vets") or {}).items())},
            "greenlit": bool(m.get("greenlit")),
            "royalty_left": int(m.get("royalty_left") or 0),
            "validators": sorted(str(v) for v in (m.get("validators") or [])),
            "busted": bool(m.get("busted")),
        }
    state["busts"] = {str(k): int(v)
                      for k, v in sorted((vetting.get("busts") or {}).items())}
    return state


def _composition_shares(
    root: str,
    deps_of: Dict[str, List[str]],
    *,
    root_share: float = COMPOSITE_ROOT_SHARE,
    max_depth: int = COMPOSITE_MAX_DEPTH,
) -> Dict[str, float]:
    """Split one unit of attestation weight over a declared-dep DAG.

    Pure + deterministic (sorted recursion over consensus-carried
    declarations). Returns {digest: weight} with sum <= 1.0.
    """
    shares: Dict[str, float] = {}

    def _spread(digest: str, weight: float, depth: int, path: frozenset) -> None:
        declared = [d for d in deps_of.get(digest, []) if d]
        if not declared or depth >= max_depth:
            shares[digest] = shares.get(digest, 0.0) + weight
            return
        shares[digest] = shares.get(digest, 0.0) + weight * root_share
        # Split by the FULL declared count; shares of cyclic or
        # unregistered deps are forfeited, never redistributed to
        # siblings (conservation: total <= 1, gaming by padding with
        # dead deps only burns the padder's own credit).
        child_w = weight * (1.0 - root_share) / len(declared)
        for dep in sorted(declared):
            if dep in path or dep not in deps_of:
                continue
            _spread(dep, child_w, depth + 1, path | {digest})

    _spread(root, 1.0, 0, frozenset({root}))
    return shares


def _normalize_credibility(
    credibility: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    """Canonicalize the carried credibility map (sorted, typed, floored).

    Shape: {household: multiplier in [CRED_FLOOR, 1.0]}. Rebuildable
    cache — same contract as ``tool_positions`` — so a missing/garbled
    copy just resets everyone to full credibility and rebuilds from
    canonical events.
    """
    out: Dict[str, float] = {}
    if not credibility:
        return out
    for house, c in sorted(credibility.items()):
        try:
            v = float(c)
        except (TypeError, ValueError):
            continue
        out[str(house)] = max(CRED_FLOOR, min(1.0, v))
    return out


def _normalize_review_book(
    review_book: Optional[Dict[str, Any]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Canonicalize the carried per-household review book (v4.1 [R5]).

    Shape: {digest: {household: {axis_id: {"sum": float, "n": int}}}}.
    Each household's running review centroid per tool, carried across
    closes so the continuous reversal-aware sanction can re-score it
    against the tool's moving head. Rebuildable-cache contract.
    """
    out: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    if not review_book:
        return out
    for digest, houses in sorted(review_book.items()):
        if not isinstance(houses, dict):
            continue
        d_out: Dict[str, Dict[str, Dict[str, float]]] = {}
        for house, axes in sorted(houses.items()):
            if not isinstance(axes, dict):
                continue
            h_out: Dict[str, Dict[str, float]] = {}
            for axis_id, cell in sorted(axes.items()):
                if not isinstance(cell, dict):
                    continue
                try:
                    s = float(cell.get("sum") or 0.0)
                    n = int(cell.get("n") or 0)
                except (TypeError, ValueError):
                    continue
                if n > 0:
                    h_out[str(axis_id)] = {"sum": s, "n": n}
            if h_out:
                d_out[str(house)] = h_out
        if d_out:
            out[str(digest)] = d_out
    return out


def compute_tool_mint(
    world: World,
    events: List[Dict[str, Any]],
    *,
    registrations: Optional[Dict[str, Dict[str, str]]] = None,
    agent_owner_map: Optional[Dict[str, str]] = None,
    voice_weights: Optional[Dict[str, float]] = None,
    vetting: Optional[Dict[str, Any]] = None,
    positions: Optional[Dict[str, Dict[str, Any]]] = None,
    # v4.1 gradient trust (ratified 2026-07-09), amended 2026-07-10 (β cap
    # and rep/ATN split deleted; ``rep_shares`` now drives drift weight only):
    rep_shares: Optional[Dict[str, float]] = None,
    credibility: Optional[Dict[str, float]] = None,
    tool_review_book: Optional[Dict[str, Any]] = None,
    vet_quorum: float = VET_QUORUM,
    vet_royalty_share: float = VET_ROYALTY_SHARE,
    vet_royalty_epochs: int = VET_ROYALTY_EPOCHS,
    vet_bust_threshold: float = VET_BUST_THRESHOLD,
) -> Dict[str, Any]:
    """Tool-author mint from canonical events (docs/tool_substrate.md v3).

    ``tool_mint(m) = usage_term(m)`` per PINNED manifest (v3, Decision
    2026-07-08: usage alone mints — the v2 standing multiplier is
    retired; reviews rank discovery and drift position instead),
    attributed to the author carried on the registration sprout's
    ``manifest_meta`` — consensus data, so a blob-store miss can never
    fork the close. Mint is pinned-only: attested (connector-backed)
    tools are publishable and reviewable but draw no emission, and
    remote services never enter the substrate at all
    (docs/services_market.md).

    ``usage_term`` is the combo damper collapsed to HOUSEHOLDS
    (Decision 2026-07-08 addendum, voice = REPUTATION not balance):
    callers group by their proven owner wallet (``agent_owner_map``;
    unbound callers are their own household), attested-ok counts sum per
    household, log1p damps ONCE per household, and the damped value
    multiplies the household's MINT weight (``voice_weights``: the
    ε-floored reputation share; ``voice_weights=None`` in the local/no-
    chain regime weighs every household 1.0). Decision 2026-07-10 deleted
    the supply-pegged β cap and the rep/ATN mint split — mint is MONEY
    ONLY now; ``rep_shares`` survives to drive DRIFT weight (below).
    ``voice_weights=None`` (no chain access) means every household weighs
    1.0 — the pre-voice behavior; a non-None map defaults unknown
    households to ``VOICE_EPSILON``. Only COGNITIVE attestations count;
    mechanical receipts are exhaust. Two exclusions run first:

      1. Owner-rooted (final form, spec "Owner-rooted registration"):
         the author's own household is excluded — this subsumes the
         self-attestation check and the same-owner check in one
         comparison. ``agent_owner_map`` (agent id → owner 0x) is an
         explicit carry-over input like ``registrations`` — sourced
         from chain owner-binding data by the driver, identical at
         every daemon, absent entries = unknown owner = the caller is
         its own household.
      2. Wire-level (interim floor until bound registrations are
         ubiquitous): a household ANY of whose attestation batches
         were signed with the same key as the registration batch is
         excluded (co-hosted sybil agents collapse to nothing). The
         sender key is transport plumbing — the agent is the only
         economic entity; the key appears in no output.

    Cross-epoch state comes in as one explicit param, derived purely
    from prior epochs' canonical events (so bit-identical everywhere):
    ``registrations`` (digest → manifest_meta, accumulated across
    epochs — a registration in epoch 1 still attributes mint in epoch
    5). A carried registration wins over a same-digest re-registration
    event: mint can never be re-attributed. Cross-epoch STANDING
    additionally requires seeding the close from the prior epoch's
    world (``seed_world``) so the manifest claim node persists; on a
    fresh-charter close only manifests (re-)sprouted this epoch have
    standing.

    Vetting (v4.1 [R1], ratified 2026-07-09): the vet GATE is REMOVED —
    tools mint from first attested use, author keeps 100% (no greenlight
    requirement, no validator royalty). ``vetting`` is still accepted and
    carried forward tolerantly (rebuildable-cache contract), and vets are
    still aggregated (``vet_quorum``/``vet_royalty_*`` retained as dead
    knobs), but they no longer skip mint or split it. A vet's surviving
    value is its per-axis scores, which enter position drift as an
    INSPECTION REVIEW (v4.1 [R2]) — moving the head, minting nothing.

    v4.1 gradient trust (drift credibility), amended 2026-07-10:

      - ``rep_shares``: household → raw reputation share (rep/rep_supply),
        the UN-floored voice weight. Drift weight ONLY (the rep/ATN mint
        split and the β cap are deleted). ``None`` => local regime (drift
        stays at weight 1.0 as before).
      - ``credibility``: carried household → drift-weight multiplier
        (v4.1 [R5]); recomputed each close against the moving head.
      - ``tool_review_book``: carried {digest: {household: {axis:
        {sum,n}}}} — per-household review centroid per tool, re-scored
        each close so a later reversal retroactively docks capturers.

    MINT weight per household: the ε-floored reputation share
    (``voice_weights``), no β cap. Mint is MONEY ONLY (Decision
    2026-07-10): REP is claimed DAO-side on ratified ATN earnings, never
    minted here.

    Returns (all maps key-sorted, pure function of canonical inputs):
      {
        "per_digest": {digest: {node_id, author, trust_class,
                                ok_count, attesters, usage_term, mint,
                                greenlit, validators, royalty_left}},
        "node_agent": {node_id: {author: mint}},   # gate-mergeable
        "registrations_next": {digest: manifest_meta},
        "vetting_next": {"manifests": {...}, "busts": {agent: n}},
        "positions_next": {digest: {"head": [N_DIMS], "mass": [N_DIMS]}},
        "credibility_next": {household: multiplier},
        "tool_review_book_next": {digest: {household: {axis: {sum,n}}}},
      }
    """
    import math

    from .world_model_substrate.infer import _artifact_standing
    from .world_model_substrate.tool_manifest import TRUST_PINNED
    from .world_model_substrate.tool_usage import tool_usage_from_events

    # Registrations: carried-over map first (immutable attribution),
    # then this epoch's sprouts in canonical author/seq order — a later
    # re-registration cannot re-attribute an existing digest.
    known_regs: Dict[str, Dict[str, str]] = dict(registrations or {})
    reg_events = [
        e for e in events
        if e.get("kind") == "sub_claim_sprouted"
        and e.get("artifact_digest") and e.get("manifest_meta")
    ]
    reg_events.sort(key=lambda e: (e.get("author_agent", ""), e.get("seq", 0)))
    for ev in reg_events:
        meta = dict(ev["manifest_meta"])
        # Registration batch's signing key, for the wire-level self-
        # attestation dedup below. Carried registrations keep the key
        # from their original epoch (persisted in the carry-over map).
        if ev.get("sender"):
            meta.setdefault("sender", str(ev["sender"]))
        known_regs.setdefault(ev["artifact_digest"], meta)

    usage = tool_usage_from_events(events)
    by_digest = _artifact_standing(world)

    # ------------------------------------------------------------------
    # Vetting pipeline (spec: Vetting section) — three deterministic
    # passes over sorted keys BEFORE any mint math:
    #   1. merge this epoch's affirmative vets into the carried state
    #      (same exclusions as the damper: self-vet, same registered
    #      owner, vet batch signed with the registration batch's key);
    #   2. bust detection: a greenlit manifest whose claim node carries
    #      a winning charter violation zeroes its remaining royalty and
    #      increments every validator's bust count (their future vet
    #      weight decays as 1/(1+busts));
    #   3. greenlight: Σ over distinct fleets (owner-map collapse,
    #      fallback per-agent) of max member weight ≥ vet_quorum →
    #      greenlit; validators frozen at that instant (late vets earn
    #      nothing — the risk window is the incentive).
    # ------------------------------------------------------------------
    vet_state = _normalize_vetting(vetting)
    owner_map_all = agent_owner_map or {}
    for digest in sorted(set(usage) & set(known_regs)):
        vets_by_caller = usage[digest].get("vets_by_caller") or {}
        if not vets_by_caller:
            continue
        meta = known_regs[digest]
        author = str(meta.get("author") or "")
        reg_sender = str(meta.get("sender") or "")
        author_owner = owner_map_all.get(author, "")
        m = vet_state["manifests"].setdefault(digest, {
            "vets": {}, "greenlit": False, "royalty_left": 0,
            "validators": [], "busted": False,
        })
        for vetter in sorted(vets_by_caller.keys()):
            if vetter == author:
                continue
            if author_owner and owner_map_all.get(vetter, "") == author_owner:
                continue
            senders = usage[digest].get("vet_senders", {}).get(vetter, [])
            if reg_sender and reg_sender in senders:
                continue
            pooled = m["vets"].setdefault(vetter, [])
            for s in senders:
                if s not in pooled:
                    pooled.append(s)
            m["vets"][vetter] = sorted(pooled)

    # v3 (spec Decision 2026-07-08): the CON-triggered bust of a greenlit
    # manifest is DORMANT — debate machinery left the live path, so there
    # is no charter-violation signal to trigger it. The bust fields
    # (busted, busts) remain in the carried state so validator weights
    # keep honoring any historical busts and the rail can be reactivated
    # (the named mitigation if review-only defenses prove insufficient).

    for digest in sorted(vet_state["manifests"].keys()):
        m = vet_state["manifests"][digest]
        if m["greenlit"]:
            continue
        fleets: Dict[str, float] = {}
        for vetter in sorted(m["vets"].keys()):
            fleet_key = owner_map_all.get(vetter, "") or vetter
            weight = 1.0 / (1.0 + vet_state["busts"].get(vetter, 0))
            fleets[fleet_key] = max(fleets.get(fleet_key, 0.0), weight)
        total_weight = round(
            sum(fleets[k] for k in sorted(fleets.keys())), 10)
        if total_weight >= vet_quorum:
            m["greenlit"] = True
            m["royalty_left"] = int(vet_royalty_epochs)
            m["validators"] = sorted(m["vets"].keys())

    # Composition fan-out (docs/tool_substrate.md — Composition rule 3):
    # each receipt digest's attested counts distribute over its declared
    # dependency DAG with conserved total weight 1 (root keeps
    # COMPOSITE_ROOT_SHARE, deps split the rest, recursively).
    #
    # ORDER OF OPERATIONS IS THE ANTI-AMPLIFICATION GUARANTEE: the
    # per-caller count is DAMPED FIRST (log1p once, at the composite the
    # caller actually attested), then the damped value splits linearly
    # over the DAG. Damping per-node after splitting would let concavity
    # mint free credit (log1p(0.7)+log1p(0.3) > log1p(1)) — self-padding
    # a composite with your own deps would amplify. Linear distribution
    # of the damped quantity makes padding exactly neutral at equal
    # standing. Attester wire identities are pooled per caller so the
    # co-hosting dedup applies wherever the credit lands.
    deps_of: Dict[str, List[str]] = {
        d: [x for x in str(m.get("deps") or "").split(",") if x]
        for d, m in known_regs.items()
    }

    # Household collapse (voice addendum): the economic unit behind a
    # caller is its proven owner wallet; unbound callers stand alone.
    def _household(agent_id: str) -> str:
        return owner_map_all.get(agent_id, "") or agent_id

    # eff buckets are keyed by HOUSEHOLD: counts sum across a
    # household's agents at each attested composite, then log1p damps
    # once — N co-owned agents are one voice, not N log1p terms.
    eff: Dict[str, Dict[str, float]] = {}
    house_senders: Dict[str, set] = {}
    for digest in sorted(set(usage) & set(known_regs)):
        shares = _composition_shares(digest, deps_of)
        entry = usage[digest]
        counts: Dict[str, int] = {}
        for caller in sorted(entry["attested_ok_by_caller"].keys()):
            house = _household(caller)
            counts[house] = (counts.get(house, 0)
                             + int(entry["attested_ok_by_caller"][caller]))
            house_senders.setdefault(house, set()).update(
                entry["attester_senders"].get(caller, []))
        for house in sorted(counts.keys()):
            damped = math.log1p(counts[house])
            for target in sorted(shares.keys()):
                bucket = eff.setdefault(target, {})
                bucket[house] = bucket.get(house, 0.0) + damped * shares[target]

    # Resident grammar — ADOPTION (spec: Resident tools, loadouts,
    # distros): loadout stamps on attestations credit the distro by
    # DISTINCT FLEETS, volume-blind: callers collapse by the chain
    # owner map (fallback: each caller its own fleet — documented
    # degradation when registrations are sparse), each fleet
    # contributes log1p(1) once per epoch regardless of how many
    # attestations it emitted. The damped value injects at the distro
    # root and fans over its dep DAG like any composite. Exclusions
    # (author, same-owner, wire dedup) run in the mint loop below via
    # the fleet's sorted representative caller, whose sender pool is
    # the union of the fleet's senders.
    from .world_model_substrate.tool_usage import loadout_adoption_from_events
    adoption = loadout_adoption_from_events(events)
    for loadout in sorted(set(adoption) & set(known_regs)):
        shares = _composition_shares(loadout, deps_of)
        entry = adoption[loadout]
        fleets: Dict[str, List[str]] = {}
        for caller in sorted(entry["by_caller"].keys()):
            fleets.setdefault(_household(caller), []).append(caller)
        for fleet_key in sorted(fleets.keys()):
            # Fleet == household: credit lands on the household key
            # directly, senders pool there for the wire dedup below.
            pooled = house_senders.setdefault(fleet_key, set())
            for member in fleets[fleet_key]:
                pooled.update(entry["senders"].get(member, []))
            damped = math.log1p(1)
            for target in sorted(shares.keys()):
                bucket = eff.setdefault(target, {})
                bucket[fleet_key] = (bucket.get(fleet_key, 0.0)
                                     + damped * shares[target])

    # ------------------------------------------------------------------
    # MINT weight per household (Decision 2026-07-10: the β cap and the
    # rep/ATN split are DELETED — mint is MONEY ONLY, distributed pro-rata
    # over the ε-floored reputation-share weights). ``rep_shares`` survives
    # only for DRIFT weight (position section below). ``voice_weights``
    # carries the ε-floored share; None => local/no-chain regime, weight
    # 1.0. ``rep_share_map`` is kept for the drift math.
    # ------------------------------------------------------------------
    rep_share_map = rep_shares or {}

    def _rep_share(house: str) -> float:
        return float(rep_share_map.get(house, 0.0))

    def _mint_weight(house: str) -> float:
        # ε-floored reputation share (voice_weights); local regime weighs
        # every household 1.0 (the pre-voice behavior).
        if voice_weights is not None:
            return round(float(voice_weights.get(house, VOICE_EPSILON)), 9)
        return 1.0

    per_digest: Dict[str, Dict[str, Any]] = {}
    node_agent: Dict[str, Dict[str, float]] = {}
    for digest in sorted(eff.keys()):
        meta = known_regs[digest]
        author = str(meta.get("author") or "")
        if not author:
            continue
        # Pinned-only emission (spec v2): the network mints only for
        # capability it can actually KNOW (hash-locked code).
        if str(meta.get("trust_class") or "") != TRUST_PINNED:
            continue
        # v4.1 [R1]: the vet GATE is REMOVED — tools mint from first
        # attested use. The carried vet state is still surfaced (greenlit
        # flag is informational only) but never skips mint here.
        vm = vet_state["manifests"].get(digest)
        greenlit = bool(vm and vm.get("greenlit"))
        nodes = by_digest.get(digest, [])
        # Combo damper over HOUSEHOLDS: counts collapsed + log1p'd at
        # fan-out time, author's household excluded (subsumes self and
        # same-owner), co-hosted households (any attestation batch signed
        # with the registration batch's key) excluded, then each
        # household's damped credit scales by its ε-floored MINT weight.
        reg_sender = str(meta.get("sender") or "")
        author_house = _household(author)
        attested_by = eff[digest]
        usage_term = 0.0
        attesters = 0
        for house in sorted(attested_by.keys()):
            if house == author_house:
                continue
            if reg_sender and reg_sender in house_senders.get(house, set()):
                continue
            usage_term += attested_by[house] * round(_mint_weight(house), 12)
            attesters += 1
        # v3/v4.1: USAGE ALONE mints. No standing multiplier, no vet gate.
        mint = usage_term
        if mint <= 0.0:
            continue
        # Anchor the mint on the manifest's claim node: lexicographically-
        # first id among the digest's claim nodes (the manifest anchor;
        # also where position drift writes the reviewed charter head).
        node_id = (sorted({n.id for n in nodes})[0] if nodes
                   else f"tool:{digest}")
        validators = list(vm["validators"]) if vm else []
        per_digest[digest] = {
            "node_id": node_id,
            "author": author,
            "trust_class": TRUST_PINNED,
            "ok_count": int(usage.get(digest, {}).get("ok_count", 0)),
            "attesters": attesters,
            "usage_term": round(usage_term, 10),
            "mint": mint,
            "greenlit": greenlit,
            "validators": len(validators),
            "royalty_left": int(vm.get("royalty_left") or 0) if vm else 0,
        }
        # v4.1 [R1]: author keeps 100% of the mint (royalty split gone).
        bucket = node_agent.setdefault(node_id, {})
        bucket[author] = bucket.get(author, 0.0) + mint

    # Royalty window ticks once per close for every greenlit manifest
    # (kept for the carried-state contract; no mint effect under v4.1).
    for digest in sorted(vet_state["manifests"].keys()):
        m = vet_state["manifests"][digest]
        if m["greenlit"] and m["royalty_left"] > 0:
            m["royalty_left"] -= 1

    # ------------------------------------------------------------------
    # Position drift (v3 base + v4.1 [R2][R3][R5]): each tool's charter
    # head is the per-axis reputation-weighted running centroid of review
    # scores. Per-axis mass so a review that scored only `correctness`
    # never drags the unscored axes; the author prior is a zero head with
    # 1.0 unit of mass per axis. Reviews come from BOTH attested-usage
    # receipts AND vet inspection events (v4.1 [R2] — vets mint nothing
    # but move position). Drift weight (v4.1 [R3]) is
    # ``rep_share × credibility`` with NO ε floor: zero-rep households get
    # ZERO drift weight. In the local regime (rep_shares is None) drift
    # keeps weight 1.0 (unchanged pre-v4.1 behavior). Pure function of
    # (carried positions/book/credibility, canonical events) over sorted
    # keys → bit-identical on every daemon.
    # ------------------------------------------------------------------
    from .world_model_substrate.adapter import CHARTER, N_DIMS
    axis_index = {str(e["id"]): int(e["axis_index"]) for e in CHARTER}

    carried_pos = positions or {}
    cred_state = _normalize_credibility(credibility)
    review_book = _normalize_review_book(tool_review_book)
    positions_next: Dict[str, Dict[str, Any]] = {}
    # v4.1 [R5]: the per-household review centroids we (re)assert this
    # close, carried forward so a later reversal retroactively re-scores
    # them against the moving head. Digest → household → axis → {sum, n}.
    review_book_next: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    # Total review mass per tool this close, for the CRED_MASS_FLOOR gate.
    tool_review_mass: Dict[str, float] = {}

    for digest in sorted(known_regs.keys()):
        prior = carried_pos.get(digest) or {}
        head = [float(x) for x in (prior.get("head") or [])]
        mass = [float(x) for x in (prior.get("mass") or [])]
        if len(head) != N_DIMS:
            head = [0.0] * N_DIMS
        if len(mass) != N_DIMS:
            mass = [1.0] * N_DIMS
        # Carry the tool's prior book forward; this epoch's reviews update
        # each household's running centroid on the axes it scored.
        book_entry: Dict[str, Dict[str, Dict[str, float]]] = {
            h: {a: dict(c) for a, c in axes.items()}
            for h, axes in review_book.get(digest, {}).items()
        }
        entry = usage.get(digest)
        if entry:
            meta = known_regs[digest]
            author = str(meta.get("author") or "")
            reg_sender = str(meta.get("sender") or "")
            author_house = _household(author)
            # v4.1 [R2]: merge attested-usage reviews and vet inspection
            # reviews into one per-household per-axis cell map. Both use
            # the same [-1,1]-clamped {sum,n} shape; vets add no usage.
            reviews = entry.get("axis_reviews_by_caller") or {}
            vet_reviews = entry.get("vet_axis_reviews_by_caller") or {}
            house_cells: Dict[str, Dict[str, Dict[str, float]]] = {}
            house_rev_senders: Dict[str, set] = {}
            for src in (reviews, vet_reviews):
                for caller in sorted(src.keys()):
                    house = _household(caller)
                    house_rev_senders.setdefault(house, set()).update(
                        entry.get("attester_senders", {}).get(caller, []))
                    house_rev_senders[house].update(
                        entry.get("vet_senders", {}).get(caller, []))
                    for axis_id in sorted(src[caller].keys()):
                        cell = src[caller][axis_id]
                        agg = house_cells.setdefault(house, {}).setdefault(
                            str(axis_id), {"sum": 0.0, "n": 0})
                        agg["sum"] += float(cell.get("sum") or 0.0)
                        agg["n"] += int(cell.get("n") or 0)
            add_mass = [0.0] * N_DIMS
            add_val = [0.0] * N_DIMS
            for house in sorted(house_cells.keys()):
                if house == author_house:
                    continue
                if reg_sender and reg_sender in house_rev_senders.get(
                        house, set()):
                    continue
                # v4.1 [R3]: drift weight = rep_share × credibility, no ε
                # floor. Local regime (rep_shares None) => weight 1.0.
                if rep_shares is None:
                    base_w = 1.0
                else:
                    rs = _rep_share(house)
                    if rs <= 0.0:
                        continue          # zero-rep => zero drift weight
                    base_w = rs
                cred = cred_state.get(house, 1.0)
                for axis_id in sorted(house_cells[house].keys()):
                    idx = axis_index.get(axis_id)
                    if idx is None:
                        continue
                    cell = house_cells[house][axis_id]
                    n = int(cell["n"])
                    if n <= 0:
                        continue
                    mean_score = cell["sum"] / n
                    w = math.log1p(n) * base_w * cred
                    add_mass[idx] += w
                    add_val[idx] += w * mean_score
                    # Update this household's running review centroid for
                    # the credibility re-score (carried forward). Store the
                    # DRIFT-mass-weighted contribution so a heavier voice's
                    # assertion is what later gets sanctioned.
                    bcell = book_entry.setdefault(house, {}).setdefault(
                        str(axis_id), {"sum": 0.0, "n": 0})
                    bcell["sum"] += mean_score * n
                    bcell["n"] += n
            for i in range(N_DIMS):
                if add_mass[i] > 0.0:
                    head[i] = ((mass[i] * head[i] + add_val[i])
                               / (mass[i] + add_mass[i]))
                    mass[i] = mass[i] + add_mass[i]
        positions_next[digest] = {
            "head": [round(h, 9) for h in head],
            "mass": [round(m_val, 9) for m_val in mass],
        }
        if book_entry:
            review_book_next[digest] = book_entry
        # Review mass = total drifted mass minus the N_DIMS author prior.
        tool_review_mass[digest] = sum(mass) - float(N_DIMS)

    # ------------------------------------------------------------------
    # v4.1 [R5] CONTINUOUS reversal-aware credibility. Re-score EVERY
    # carried household review centroid against the tool's CURRENT
    # (just-drifted) head, on tools whose review mass ≥ CRED_MASS_FLOOR.
    #   deviation > δ  -> dock proportional to (dev-δ)/(2-δ) (worst dock)
    #   deviation ≤ δ  -> restore proportional to (δ-dev)/δ  (best restore)
    # Symmetric, no stabilization moment: a captured score that later
    # reverses moves the head away from the capturer's stored centroid,
    # so they get docked in the epoch the reversal lands. Credibility
    # multiplies DRIFT weight only (applied next close), never mint.
    # ------------------------------------------------------------------
    cred_delta: Dict[str, float] = {}
    for digest in sorted(review_book_next.keys()):
        if tool_review_mass.get(digest, 0.0) < CRED_MASS_FLOOR:
            continue
        cur_head = positions_next[digest]["head"]
        houses = review_book_next[digest]
        for house in sorted(houses.keys()):
            for axis_id in sorted(houses[house].keys()):
                idx = axis_index.get(axis_id)
                if idx is None:
                    continue
                cell = houses[house][axis_id]
                n = int(cell["n"])
                if n <= 0:
                    continue
                asserted = cell["sum"] / n
                dev = abs(asserted - float(cur_head[idx]))
                if dev > CRED_DELTA:
                    penalty = (dev - CRED_DELTA) / (2.0 - CRED_DELTA)
                    cred_delta[house] = min(
                        cred_delta.get(house, 0.0), -penalty)
                else:
                    restore = (CRED_DELTA - dev) / CRED_DELTA
                    cred_delta[house] = max(
                        cred_delta.get(house, 0.0),
                        CRED_RECOVERY * restore)

    # Apply: passive recovery toward 1.0 for untouched households, then
    # the signed adjustment, floored at CRED_FLOOR. Deterministic over the
    # union of all known + adjusted households in sorted order.
    credibility_next: Dict[str, float] = {}
    all_houses = sorted(set(cred_state.keys()) | set(cred_delta.keys()))
    for house in all_houses:
        c = cred_state.get(house, 1.0)
        adj = cred_delta.get(house)
        if adj is None:
            c = min(1.0, c + CRED_RECOVERY * (1.0 - c))
        elif adj < 0:
            c = c * (1.0 + adj)
        else:
            c = min(1.0, c + adj * (1.0 - c))
        c = max(CRED_FLOOR, c)
        # Don't persist a household pinned at full credibility with no
        # history — keeps the carried map to actually-touched households.
        if not (adj is None and c >= 1.0 - 1e-12 and house not in cred_state):
            credibility_next[house] = round(c, 9)

    # Round the carried review book for cross-daemon byte-stability.
    review_book_out: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for digest in sorted(review_book_next.keys()):
        d_out: Dict[str, Dict[str, Dict[str, float]]] = {}
        for house in sorted(review_book_next[digest].keys()):
            h_out: Dict[str, Dict[str, float]] = {}
            for axis_id in sorted(review_book_next[digest][house].keys()):
                cell = review_book_next[digest][house][axis_id]
                if int(cell["n"]) > 0:
                    h_out[axis_id] = {"sum": round(float(cell["sum"]), 10),
                                      "n": int(cell["n"])}
            if h_out:
                d_out[house] = h_out
        if d_out:
            review_book_out[digest] = d_out

    return {
        "per_digest": dict(sorted(per_digest.items())),
        "node_agent": {k: dict(sorted(v.items()))
                       for k, v in sorted(node_agent.items())},
        "registrations_next": dict(sorted(known_regs.items())),
        "positions_next": positions_next,
        "credibility_next": dict(sorted(credibility_next.items())),
        "tool_review_book_next": review_book_out,
        "vetting_next": {
            "manifests": {d: {
                "vets": {a: sorted(s)
                         for a, s in sorted(m["vets"].items())},
                "greenlit": m["greenlit"],
                "royalty_left": m["royalty_left"],
                "validators": sorted(m["validators"]),
                "busted": m["busted"],
            } for d, m in sorted(vet_state["manifests"].items())},
            "busts": dict(sorted(vet_state["busts"].items())),
        },
    }


def federated_reconcile_epoch(
    world: World,
    snapshots: EpochSnapshots,
    events: List[Dict[str, Any]],
    *,
    agent_weights: Optional[Dict[str, float]] = None,
    # v4.1 [R7] / divergence D2: default OFF, aligned with the outer
    # federated_epoch_close (the violator-pays gate is DORMANT). A True
    # default here silently reactivated the gate for direct callers.
    apply_gate: bool = False,
    gate_strength: float = 1.0,
    output_decimals: int = OUTPUT_DECIMALS,
    emission_pool: Optional[float] = None,
    extra_node_agent_mint: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """Run reconcile_epoch deterministically.

    Same inputs (world topology, snapshots, events list, optional
    agent_weights) at every honest daemon produce **bit-identical**
    output dicts.

    ``emission_pool``: when set, post-gate mints are normalized to
    shares of this fixed pool (see reconcile.apply_emission_pool). Under
    fees-only emission (Decision 2026-07-10) this is the burned-fee total
    for the epoch window (Σ ServiceFee.burned between anchors), read on
    the same pinned rail as the voice weights; a zero pool yields zero
    mint. MUST be derived from canonical data (never a daemon-local
    clock) or the bit-identical guarantee breaks.

    Returns the same shape as ``reconcile_epoch`` plus:
      - ``output_decimals``: the rounding precision used.
      - ``gate_applied``: whether the mint gate ran.

    ONE EARNING RAIL (ratified 2026-07-04, late): score-movement mint
    is RETIRED. Usage of a tool is a choice — attested usage is the
    usefulness signal the economics run on, and text-claim score
    movement double-counted the same "did it land" question with
    weaker ground truth (phase8). ``agent_mint`` therefore consists
    solely of ``extra_node_agent_mint`` (tool-author mint from
    compute_tool_mint), gate-scaled and pool-normalized. The score/
    standing machinery is fully retained — standing PRICES tool mint
    and the gate reads charter debates — it just no longer pays for
    prose. Work units remain data-plane retrieval artifacts (that
    value was validated); they earn nothing directly. ``node_mint`` /
    ``agent_novelty`` remain computed as diagnostics.
    """
    agent_weights = dict(agent_weights or {})

    # Discover all node ids that appear in any snapshot. Sort them so
    # the iteration order — and therefore the sequence of float adds
    # into agent_mint — is deterministic across daemons.
    all_node_ids: set[str] = set()
    all_node_ids.update(snapshots.start.keys())
    for cp in snapshots.checkpoints:
        all_node_ids.update(cp.keys())
    all_node_ids.update(snapshots.close.keys())
    sorted_node_ids = sorted(all_node_ids)

    node_mint: Dict[str, float] = {}
    node_novelty: Dict[str, float] = {}
    agent_mint: Dict[str, float] = {}
    agent_novelty: Dict[str, float] = {}
    # Per-(node, agent) mint breakdown, retained so the violator-pays
    # gate can scale a flagged node's mint before it aggregates into
    # per-agent totals. reconcile.py's uniform-ratio fallback discarded
    # this; keeping it is what lets a CON win redistribute correctly.
    node_agent_mint: Dict[str, Dict[str, float]] = {}

    for node_id in sorted_node_ids:
        nov = _node_novelty(node_id, snapshots)
        score_mint = _node_mint(node_id, snapshots, world=world)
        if nov > 0:
            node_novelty[node_id] = nov
        if score_mint > 0:
            # DIAGNOSTIC ONLY (one earning rail): per-node score
            # movement is reported but never enters agent_mint.
            node_mint[node_id] = score_mint

        # Attribute novelty proportionally over agents whose events
        # landed under this node. Sort the attribution dict so the
        # sum of float shares is associative-stable across daemons.
        attribution = _attribute_node_mint(node_id, 1.0, events, world)
        for agent in sorted(attribution.keys()):
            share = attribution[agent]
            agent_novelty[agent] = agent_novelty.get(agent, 0.0) + nov * share

    # The ONLY earning rail: tool-author mint (compute_tool_mint),
    # anchored per manifest claim node. Merged BEFORE the gate so a won
    # charter CON against the anchoring node suppresses it like any
    # other mint, and before the pool so pool normalization applies.
    # ``agent_weights`` (stake-style weighting) applies here as it did
    # to the retired score rail. Sorted iteration keeps the float-add
    # order canonical. Money only (Decision 2026-07-10): no rep basis is
    # accumulated — REP is claimed DAO-side on ratified ATN earnings.
    for node_id in sorted((extra_node_agent_mint or {}).keys()):
        extra = extra_node_agent_mint[node_id]
        per_agent = node_agent_mint.setdefault(node_id, {})
        node_total = 0.0
        for agent in sorted(extra.keys()):
            amount = float(extra[agent])
            if amount <= 0.0:
                continue
            weighted = amount * agent_weights.get(agent, 1.0)
            agent_mint[agent] = agent_mint.get(agent, 0.0) + weighted
            per_agent[agent] = per_agent.get(agent, 0.0) + weighted
            node_total += weighted
        if node_total > 0.0:
            node_mint[node_id] = node_mint.get(node_id, 0.0) + node_total

    # Output normalization: round all per-agent and per-node values
    # to `output_decimals` places. This swamps any IEEE 754 jitter
    # without losing meaningful precision.
    agent_mint = _round_dict(agent_mint, output_decimals)
    agent_novelty = _round_dict(agent_novelty, output_decimals)
    node_mint = _round_dict(node_mint, output_decimals)
    node_novelty = _round_dict(node_novelty, output_decimals)

    # Re-sort dicts so JSON serialization (and dict equality with
    # other daemons') is order-stable. dict() in CPython preserves
    # insertion order, so building from a sorted iterator gives a
    # canonical layout.
    agent_mint = dict(sorted(agent_mint.items()))
    agent_novelty = dict(sorted(agent_novelty.items()))
    node_mint = dict(sorted(node_mint.items()))
    node_novelty = dict(sorted(node_novelty.items()))

    total_mint = round(sum(agent_mint.values()), output_decimals)
    total_novelty = round(sum(agent_novelty.values()), output_decimals)

    result = {
        "agent_mint": agent_mint,
        "agent_novelty": agent_novelty,
        "node_mint": node_mint,
        "node_novelty": node_novelty,
        "total_mint": total_mint,
        "total_novelty": total_novelty,
        "output_decimals": output_decimals,
        "gate_applied": False,
    }

    if apply_gate:
        # Violator-pays gate: scale each flagged node's per-agent mint
        # BEFORE aggregating, so suppression falls only on the flagged
        # node's authors (not uniformly across all agents like
        # apply_mint_gate's fallback). Combined with the emission pool
        # below, a winning CON redistributes the violator's mint to
        # everyone else.
        node_violation: Dict[str, float] = {}
        for node_id in sorted(node_agent_mint.keys()):
            node_violation[node_id] = charter_violation_score(
                world, node_id, DEFAULT_CHARTER_IDS,
            )
        gated_agent_mint = scale_node_agent_mint_by_violation(
            node_agent_mint, node_violation, gate_strength=gate_strength,
        )
        # Recompute the gated per-node mint (for diagnostics + pool
        # scaling of node_mint), scaling each node by its violation.
        gated_node_mint: Dict[str, float] = {}
        for node_id in sorted(node_mint.keys()):
            v = node_violation.get(node_id, 0.0)
            scale = max(0.0, 1.0 - gate_strength * v)
            gm = node_mint[node_id] * scale
            if gm > 0:
                gated_node_mint[node_id] = gm

        result["agent_mint"] = dict(sorted(
            _round_dict(gated_agent_mint, output_decimals).items()
        ))
        result["node_mint"] = dict(sorted(
            _round_dict(gated_node_mint, output_decimals).items()
        ))
        result["node_violation"] = dict(sorted(node_violation.items()))
        result["total_mint"] = round(
            sum(result["agent_mint"].values()), output_decimals,
        )
        result["gate_applied"] = True

    # Fixed-emission normalization — AFTER the gate, so mint suppressed
    # by debate is redistributed to the surviving contributors. Scaling
    # runs over sorted keys and is re-rounded, so determinism holds.
    if emission_pool is not None:
        from .world_model_substrate.reconcile import apply_emission_pool
        result = apply_emission_pool(result, emission_pool)
        result["agent_mint"] = dict(sorted(
            _round_dict(result.get("agent_mint", {}), output_decimals).items()
        ))
        result["node_mint"] = dict(sorted(
            _round_dict(result.get("node_mint", {}), output_decimals).items()
        ))
        result["total_mint"] = round(
            sum(result["agent_mint"].values()), output_decimals,
        )

    logger.info(
        "federated reconciliation: %d nodes, %d agents, "
        "total mint %.6f, total novelty %.6f",
        len(result["node_mint"]),
        len(result["agent_mint"]),
        result["total_mint"],
        result["total_novelty"],
    )
    return result


def apply_tool_positions(
    world: World, positions: Dict[str, Dict[str, Any]],
) -> int:
    """Write drifted charter heads onto manifest claim nodes (v3 drift).

    For every node whose ``artifact_digest`` appears in ``positions``,
    the head (first ``len(pos["head"])`` coords) of BOTH coordinate
    stores is replaced: the per-tendency ``CoordinateClaim.anchor``
    (what serialize_world / checkpoints read) and the backing
    ``world.observations`` entry (what the visualization surfaces
    read). Embedding tails are untouched. Deterministic: purely a
    function of (world, positions).

    Returns the number of (tendency, node) coordinate writes.
    """
    from dataclasses import replace as _dc_replace

    if not positions:
        return 0
    updated = 0
    seen_obs: set = set()
    for tendency in world.tendencies.values():
        for node in tendency.tree.all_nodes():
            digest = getattr(node, "artifact_digest", "") or ""
            pos = positions.get(digest) if digest else None
            if not pos:
                continue
            head = tuple(float(x) for x in (pos.get("head") or ()))
            if not head:
                continue
            claim = tendency._node_to_claim.get(node.id)
            if claim is not None and claim.anchor and len(claim.anchor) >= len(head):
                claim.anchor = head + tuple(claim.anchor)[len(head):]
                updated += 1
            obs_id = getattr(node, "observation_id", None)
            if obs_id and obs_id not in seen_obs:
                obs = world.observations.get(obs_id)
                if obs is not None and obs.coords and len(obs.coords) >= len(head):
                    world.observations[obs_id] = _dc_replace(
                        obs, coords=head + tuple(obs.coords)[len(head):])
                    seen_obs.add(obs_id)
    return updated


def federated_epoch_close(
    canonical: CanonicalOrder,
    *,
    seed_world: Optional[World] = None,
    bandwidth: float = 1.5,
    embedding_dim: int = 1024,
    # v3 (spec Decision 2026-07-08): the violator-pays gate is DORMANT —
    # debate machinery left the live path, so there are no charter CONs
    # to price. The machinery stays intact behind the flag for the named
    # future mitigation; default OFF.
    apply_gate: bool = False,
    gate_strength: float = 1.0,
    output_decimals: int = OUTPUT_DECIMALS,
    equilibrate_rounds: int = 8,
    equilibrate_tolerance: float = 1e-3,
    emission_pool: Optional[float] = None,
    pricing: str = "ledger",
    tool_registrations: Optional[Dict[str, Dict[str, str]]] = None,
    agent_owner_map: Optional[Dict[str, str]] = None,
    voice_weights: Optional[Dict[str, float]] = None,
    tool_vetting: Optional[Dict[str, Any]] = None,
    tool_positions: Optional[Dict[str, Dict[str, Any]]] = None,
    # v4.1 gradient trust (ratified 2026-07-09) carry-over inputs; amended
    # 2026-07-10 (rep_supply dropped — β cap gone, rep_shares → drift only):
    rep_shares: Optional[Dict[str, float]] = None,
    tool_credibility: Optional[Dict[str, float]] = None,
    tool_review_book: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a full federated epoch close given a canonical sequence.

    Builds a throwaway replay world from a fresh charter (or from the
    supplied ``seed_world`` if continuing from a prior epoch's
    authoritative state), replays the canonical batch sequence, and
    runs federated_reconcile_epoch.

    ``pricing`` selects how per-node scores are derived:

      - ``"ledger"`` (default, post-phase8): replay applies causal
        events only — NO equilibrate rounds, NO derived-sprout capture.
        A node's score is the ``net_score`` tree recursion (posts +
        signed pro/con children). Score moves only when causal events
        land on/under the node during the epoch. The O(N^2) equilibrate
        leaves the hot path. Cycle memoization is made deterministic by
        evaluating net_score in sorted node-id order (see
        ``snapshot_node_scores_ordered``).

      - ``"equilibrated"``: the pre-phase8 experimental kernel. Replays
        each batch with per-batch equilibration and captures the
        derived sprouts it produces, attributing them to the batch's
        author. Preserved BIT-FOR-BIT for the experimental kernel.

    The returned ``result`` is bit-identical across daemons that
    received the same canonical sequence — which by 5.3's contract
    they will, given the same set of batches.

    Notes
    -----

    Phase 5.4 builds the replay world from a **fresh charter**, which
    is correct as long as epochs don't carry any persistent state
    forward. Phase 5.5+ may want to seed from the prior authoritative
    epoch's snapshot (so cross-epoch lineage works); that's why
    ``seed_world`` is exposed.
    """
    if pricing not in ("ledger", "equilibrated"):
        raise ValueError(f"unknown pricing mode: {pricing!r}")

    if seed_world is None:
        world = build_charter_world(
            bandwidth=bandwidth, embedding_dim=embedding_dim,
        )
    else:
        world = seed_world

    snapshots = EpochSnapshots()

    if pricing == "ledger":
        # Ledger replay: causal events only, no equilibration, no
        # derived-sprout capture. Snapshots use sorted-order net_score
        # so co-parented-cycle memoization is deterministic.
        #
        # Attribution needs each causal sprout event's node_id to be the
        # LIVE (content-addressed) id, but the wire carries solver-side
        # labels. We capture the replay remap and rewrite the events so
        # ``_events_under_node`` (which matches raw event node_ids
        # against live descendant ids) attributes to the real author.
        snapshots.start = snapshot_node_scores_ordered(world)
        all_events: List[Dict[str, Any]] = []
        for batch in canonical.ordered_batches:
            if not batch.events:
                continue
            remap: Dict[str, str] = {}
            apply_events(
                world, batch.events,
                equilibrate_after=False, remap_out=remap,
            )
            # Build attribution copies with node_id rewritten to the
            # live id. We do NOT mutate batch.events (the canonical log
            # is fed downstream to the state-sync tracker verbatim).
            sender_hex = batch.sender_pubkey.hex()
            for ev in batch.events:
                if ev.get("kind") == "sub_claim_sprouted":
                    live = remap.get(ev.get("node_id", ""))
                    if live and live != ev.get("node_id"):
                        ev = dict(ev)
                        ev["solver_node_id"] = ev.get("node_id")
                        ev["node_id"] = live
                ev = _annotate_sender(ev, sender_hex)
                all_events.append(ev)
        snapshots.close = snapshot_node_scores_ordered(world)
    else:
        # Equilibrated (experimental) kernel — unchanged behavior.
        snapshots.record_start(world)
        # Replay each batch as its own apply_events + equilibrate, so
        # equilibration history matches what would have happened with
        # in-order live submission.
        #
        # Attribution gotcha (Phase 5.4): the gossip wire format carries
        # only causal events. Reconcile, however, attributes mint by
        # finding sub_claim_sprouted events under each minting node —
        # which means it needs the *derived* sprouts that equilibration
        # produced too. We capture them here and attribute them to the
        # batch's author, so attribution reproduces deterministically.
        #
        # Determinism: cross-tendency discovery and equilibration are
        # deterministic given world state, so two daemons replaying the
        # same canonical sequence produce the same derived events.
        all_events = []
        for batch in canonical.ordered_batches:
            # Author of this batch's derived events = whoever signed the
            # batch. We use sender_pubkey hex as a stable agent identity.
            # If batch.events carry an explicit author_agent, prefer that
            # (the daemon may run on behalf of multiple agents).
            author = _author_for_batch(batch)
            before_ids = _all_node_ids(world)
            if batch.events:
                apply_events(world, batch.events)
            equilibrate(
                world,
                max_rounds=equilibrate_rounds,
                tolerance=equilibrate_tolerance,
            )
            # Capture derived sprouts post-equilibrate, attributed to
            # this batch's author. The event recorder uses its own
            # sequence numbering; we offset to keep events globally
            # unique-ish in the all_events list (reconcile only matches
            # by node_id and position, not seq).
            recorder = _EventRecorder(agent_id=author)
            recorder.sub_claims_after_equilibrate(world, before_ids)
            derived = [e.to_dict() for e in recorder.events]

            sender_hex = batch.sender_pubkey.hex()
            all_events.extend(_annotate_sender(ev, sender_hex)
                              for ev in batch.events)
            all_events.extend(derived)

        snapshots.record_close(world)

    # Tool substrate: author mint = usage_term (pinned only, v3 —
    # docs/tool_substrate.md Decision 2026-07-08), anchored on the
    # manifest's claim node. Pure function of the replayed world +
    # canonical events + explicitly-passed carry-over registrations,
    # so the bit-identical guarantee holds.
    tool_result = compute_tool_mint(
        world, all_events,
        registrations=tool_registrations,
        agent_owner_map=agent_owner_map,
        voice_weights=voice_weights,
        vetting=tool_vetting,
        positions=tool_positions,
        rep_shares=rep_shares,
        credibility=tool_credibility,
        tool_review_book=tool_review_book,
    )
    # v3 position drift: write the reviewed charter heads onto the replay
    # world so the checkpoint (world_cid) and any world reader carry the
    # drifted positions forward. Coords never feed scores, so the score
    # snapshots above are unaffected; the write is a pure function of
    # (carry, canonical events) — bit-identical everywhere.
    apply_tool_positions(world, tool_result["positions_next"])

    result = federated_reconcile_epoch(
        world, snapshots, all_events,
        apply_gate=apply_gate,
        gate_strength=gate_strength,
        output_decimals=output_decimals,
        emission_pool=emission_pool,
        extra_node_agent_mint=tool_result["node_agent"],
    )
    # Keep tool_mint in the SAME UNITS as agent_mint: when the emission
    # pool normalized agent mints, scale the per-digest display values
    # by the identical factor (usage_term stays raw — it's the input
    # signal, not a payout). Sorted iteration + rounding for the
    # bit-identical guarantee.
    per_digest = tool_result["per_digest"]
    _pool = result.get("emission_pool")
    _raw_total = float(result.get("raw_mint_total") or 0.0)
    if _pool is not None and _raw_total > 0.0:
        _scale = float(_pool) / _raw_total
        per_digest = {
            d: {**e, "mint": round(e["mint"] * _scale, output_decimals)}
            for d, e in sorted(per_digest.items())
        }
    result["tool_mint"] = per_digest
    result["tool_registrations"] = tool_result["registrations_next"]
    result["tool_vetting"] = tool_result["vetting_next"]
    result["tool_positions"] = tool_result["positions_next"]
    # v4.1 gradient-trust carried state (rebuildable-cache contract, same
    # as tool_positions): the driver persists these and feeds them into
    # the next close.
    result["tool_credibility"] = tool_result["credibility_next"]
    result["tool_review_book"] = tool_result["tool_review_book_next"]
    result["epoch_root"] = canonical.epoch_root().hex()
    result["n_batches"] = len(canonical.ordered_batches)
    result["n_events"] = len(all_events)
    result["scope"] = "federated"
    result["authoritative"] = True
    result["pricing"] = pricing

    # The on-chain anchor (Phase 5.5) commits the merkle of an
    # ``authoritative_payload`` dict. Only fields whose keys are
    # deterministic across daemons can live here. Node ids are now
    # fully deterministic (content-addressed children; roots are
    # ``root_<tendency_id>``), but per-node score maps are unbounded
    # in size, so the chain payload still carries only ``agent_mint``
    # + ``agent_novelty`` (keyed by user-supplied agent_id) plus
    # ``epoch_root``. Per-node fields stay in ``result`` for local
    # diagnostics only.
    # Charter version hash: the anchored charter this close ran against. A
    # charter change is a forward-only fork boundary, so the hash belongs in
    # the authoritative payload (imported lazily to keep the import block small
    # and avoid a circular import via adapter).
    from .world_model_substrate.charter_version import charter_hash as _charter_hash

    result["authoritative_payload"] = {
        # schema 3 (fees-only, Decision 2026-07-10): MONEY ONLY — the
        # anchor's mint merkle commits (agent, agent_mint) on a 2-field
        # leaf. ``agent_rep`` is gone: REP is claimed DAO-side on ratified
        # ATN earnings, not federation-minted here. (Schema 2 was the v4.1
        # 3-field agent_rep shape; not reused for this different shape.)
        "schema": 3,
        "epoch_root": result["epoch_root"],
        "agent_mint": result["agent_mint"],
        "agent_novelty": result["agent_novelty"],
        "total_mint": result["total_mint"],
        "total_novelty": result["total_novelty"],
        "output_decimals": result.get("output_decimals", output_decimals),
        "gate_applied": result.get("gate_applied", False),
        "pricing": pricing,
        "charter_hash": _charter_hash(),
        "n_batches": result["n_batches"],
        "n_events": result["n_events"],
    }
    return result


def _annotate_sender(ev: Dict[str, Any], sender_hex: str) -> Dict[str, Any]:
    """Stamp the gossip batch's signing key onto a COPY of the events
    the tool-mint pass reads (tool_used receipts and manifest_meta
    registration sprouts).

    The AGENT is the only authoritative economic entity; the sender key
    is transport plumbing used solely for the wire-level dedup in
    compute_tool_mint (self-attestation via co-hosted sybil agents
    collapses to nothing). It never enters formula outputs, attribution
    maps, or any chain surface. Deterministic: canonical batches carry
    the key, so every daemon stamps identically. batch.events is never
    mutated (the canonical log flows downstream verbatim).
    """
    if ev.get("kind") == "tool_used" or (
        ev.get("kind") == "sub_claim_sprouted" and ev.get("manifest_meta")
    ):
        ev = dict(ev)
        ev["sender"] = sender_hex
    return ev


def _author_for_batch(batch) -> str:
    """Determine which agent_id derived events should be attributed to.

    Preference order:
      1. The author_agent of the first observation_added event in the
         batch (the batch was made on behalf of that user-supplied id).
      2. The sender_pubkey hex (a stable cryptographic identity, used
         when the batch carries no causal observation events).

    Determinism: every honest daemon picks the same author given the
    same canonical batch, so derived-event attribution reproduces.
    """
    for ev in batch.events:
        if ev.get("kind") == "observation_added":
            author = ev.get("author_agent")
            if author:
                return str(author)
    # Fall back: derive a stable identity from the signer.
    return f"pk:{batch.sender_pubkey.hex()[:16]}"


def _round_dict(d: Dict[str, float], decimals: int) -> Dict[str, float]:
    """Round every value in a {str: float} dict to ``decimals`` places.
    Drops zero values so empty contributions don't pollute the map."""
    out: Dict[str, float] = {}
    for k, v in d.items():
        rv = round(float(v), decimals)
        if rv != 0.0:
            out[k] = rv
    return out

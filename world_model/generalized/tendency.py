"""GeneralizedTendency: thesis + budget + frame + tree + behaviour.

A tendency in the generalized model is a *thesis about its world* and
the disposition to keep that thesis satisfied. Concretely:

  - It owns a tree (from world_model.models.tree.Tree). The root of the
    tree is the tendency's foundational claim.
  - It owns a CoordinateFrame whose claims correspond to the nodes of
    its tree (one CoordinateClaim per Node, at the same depth, with
    the node's stake aggregated as the claim's stake).
  - It has a budget (real-valued) that it can spend staking on its own
    tree (defense) or on other tendencies' trees (support if PRO,
    attack if CON).
  - It has a CoordinateProbe to evaluate novelty of incoming
    observations and of other tendencies' nodes.

When asked to act, a tendency:
  1. Runs the novelty probe on each incoming observation against its
     own frame.
  2. Based on the termination, stakes on the appropriate node:
       - INTEGRATED    -> stake PRO on the related claim's node, absorb
       - CONTRADICTS   -> stake CON on the contradicted node
       - DISRUPTS      -> stake CON heavily; consider sprouting child
       - ORTHOGONAL    -> ignore (out of this tendency's scope)
       - MAX_ITERATIONS -> ignore for now
  3. For every other tendency in the world, runs the probe on each of
     that tendency's nodes against its own frame. Same staking logic
     but cross-staked into the other's tree.

The tendency doesn't reason about budget allocation between defense
and cross-staking explicitly; the novelty probe's output (which
tendency's tree it's evaluating against) determines where stake
lands. Budget is a hard cap: if the sum of intended stakes exceeds
budget, they're scaled down proportionally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple
import hashlib
import json
import math

from ..models.tree import Node, Position, Stake, Tree
from ..novelty.core import Stance, Termination
from .coordinate_frame import (
    CoordinateClaim,
    CoordinateFrame,
    CoordinateProbe,
    _coords_of,
)
from .observation import Observation


# ---------------------------------------------------------------------------
# intrinsic_score: count-of-posts + signed children walk (cycle-safe)
# ---------------------------------------------------------------------------


def _intrinsic_score(node: Node, _seen: Optional[set] = None) -> float:
    """Walk the node's full subtree across all child edges (PRO/CON)
    and return its intrinsic score.

      intrinsic_score = len(stakes) + Σ pro_children.intrinsic - Σ con_children.intrinsic

    Cycle protection: a multi-parented node could appear in multiple
    places in a recursive walk. The `_seen` set guards against
    revisiting the same node id within one call.

    NOTE: this is the *global* intrinsic — it walks every child edge
    regardless of which tendency owns it. For a node co-parented
    across trees with different positions, the global walk may
    over-subtract (a child that is CON of `node` in tree T1 may be
    PRO of `node` in T2; the walk subtracts unconditionally). For
    per-tendency reads, prefer `_intrinsic_score_in_tendency`.
    """
    if _seen is None:
        _seen = set()
    if node.id in _seen:
        return 0.0
    _seen.add(node.id)
    direct = float(len(node.stakes))
    pro_sum = sum(_intrinsic_score(c, _seen) for c in node.pro_children)
    con_sum = sum(_intrinsic_score(c, _seen) for c in node.con_children)
    return direct + pro_sum - con_sum


def _intrinsic_score_in_tendency(
    node: Node,
    tendency_id: str,
    _seen: Optional[set] = None,
) -> float:
    """Tendency-aware intrinsic walk.

    For each child of `node`, the child contributes only if it has a
    parent_link `(node.id, position, tendency_id)`. PRO at that edge
    contributes +intrinsic_in_tendency(child); CON contributes
    -intrinsic_in_tendency(child). Direct posts on `node` count
    toward intrinsic in every tendency.

    This is what `prune_veto_negatives` should consult when
    computing a CON child's contribution to its tendency-tree-root
    score, because it correctly ignores edges from other tendencies.
    """
    if _seen is None:
        _seen = set()
    if node.id in _seen:
        return 0.0
    _seen.add(node.id)
    direct = float(len(node.stakes))
    pro_sum = 0.0
    con_sum = 0.0
    # Look at every node in the world that lists `node.id` as a
    # parent in this tendency. We don't have a back-pointer index, so
    # walk the union of pro_children + con_children and filter by
    # parent_link matching (node.id, ?, tendency_id).
    for child in node.pro_children + node.con_children:
        edge_position = None
        for link in child.parents:
            if link.parent_id == node.id and link.tendency_id == tendency_id:
                edge_position = link.position
                break
        if edge_position is None:
            continue
        sub = _intrinsic_score_in_tendency(child, tendency_id, _seen)
        if edge_position == Position.PRO:
            pro_sum += sub
        else:
            con_sum += sub
    return direct + pro_sum - con_sum


# ---------------------------------------------------------------------------
# Tendency
# ---------------------------------------------------------------------------


@dataclass
class GeneralizedTendency:
    """A thesis owning a tree, a frame, a budget, and a probe."""

    id: str
    thesis: str
    anchor: Tuple[float, ...]                # location of root claim in space
    polarity_axis: Tuple[float, ...]         # unit vector for PRO direction
    budget: float = 1.0
    bandwidth: float = 1.0                   # frame's similarity bandwidth

    # Internal state -- built in __post_init__
    tree: Tree = field(init=False)
    frame: CoordinateFrame = field(init=False)
    probe: CoordinateProbe = field(init=False)
    _root_claim: CoordinateClaim = field(init=False)
    _node_to_claim: dict[str, CoordinateClaim] = field(init=False, default_factory=dict)

    # Last computed stakes, keyed by (other_tendency_id, node_id) -> signed stake.
    # Negative = CON, positive = PRO. Used by World to apply stakes.
    last_stakes: dict[tuple[str, str], float] = field(default_factory=dict)

    # Observation ids that have already been positioned as CON children
    # in our tree. Prevents re-sprouting deeper descendants on every
    # round; the CON-position is the persistent record of "this
    # contradicts us."
    _con_positioned: set = field(default_factory=set)

    # Smooth promotion: outbound staking capacity per node, indexed by
    # node_id. Updated each round as a windowed running average of
    # PRO-stake received. A freshly sprouted node has 0 capacity (its
    # voice is silent). As it accumulates PRO support, its capacity
    # rises and it can stake on related claims at proportional
    # magnitude. Tendency roots are initialized with full capacity
    # (= self.budget) so they act normally from the start.
    node_capacity: dict[str, float] = field(default_factory=dict)
    capacity_decay: float = 0.7   # blend factor: cap_new = decay * cap_old + (1-decay) * pro_in
    capacity_reach: float = 0.5   # outbound stake = reach * capacity per round
    capacity_threshold: float = 0.05  # below this, node stays silent
    smooth_promotion: bool = True   # if False, only the root acts; sub-claims stay passive

    # Per-node novelty rate constants (see lindblad/NOVELTY_REFACTOR.md).
    # Per-round update of node.n via:
    #   dn/dt = -gamma_pro * n * pro_rate + gamma_con * (1-n) * con_rate
    #         + epsilon * (1-n)
    # gamma_pro > gamma_con so confirmation reduces uncertainty faster
    # than contradiction restores it. epsilon is a slow drift toward
    # uncertainty when no observations land.
    novelty_gamma_pro: float = 1.0
    novelty_gamma_con: float = 0.5
    novelty_drift: float = 0.01

    # Correctness-as-veto. Deployers tag a tendency as `veto_shaped`
    # to mark it as a hard-veto root: subtrees rooted under its
    # children that drop below a configured intrinsic-score floor get
    # pruned regardless of n. Pair with a higher novelty_gamma_con so
    # CON evidence settles fast on this root. See
    # `prune.prune_veto_negatives` for the asymmetric-pruning
    # mechanic.
    veto_shaped: bool = False
    veto_score_floor: float = -1.0   # subtree pruned if intrinsic_score < this

    def __post_init__(self):
        # Create root claim and tree
        self._root_claim = CoordinateClaim(
            content=self.thesis,
            depth=0,
            stake=1.0,
            anchor=self.anchor,
            polarity_axis=self.polarity_axis,
        )
        self.tree = Tree(root_value=self.thesis)
        # Bind root node's content to the thesis label (already done by Tree)
        # and link it in the index
        self._node_to_claim[self.tree.root_node.id] = self._root_claim
        # Build frame from claim hierarchy
        self.frame = CoordinateFrame(
            claims=[self._root_claim],
            integrated={},
            bandwidth=self.bandwidth,
        )
        self.probe = CoordinateProbe(max_iterations=8, disruption_threshold=0.6)
        # Root node starts with full capacity -- it's the founding voice.
        self.node_capacity[self.tree.root_node.id] = self.budget

    # ----- Tree growth -----

    def _stable_path_to_root(self, parent_node_id: str) -> str:
        """Return a stable path string from this parent up to the root,
        so two trees with the same shape produce the same string.

        The local UUID of the parent is replaced by a chain of
        (position, anchor, axis) hops terminating in
        "ROOT:<tendency_id>". This way two solvers with the same
        tendency id and the same intermediate-claim shape compute the
        same parent input regardless of their roots' local UUIDs.
        """
        parts: list[str] = []
        node_id: Optional[str] = parent_node_id
        while node_id is not None:
            node = self.tree.get_node(node_id)
            if node is None:
                break
            if node.position == Position.ROOT:
                parts.append(f"ROOT:{self.id}")
                break
            claim = self._node_to_claim.get(node_id)
            anchor = list(claim.anchor) if claim and claim.anchor else []
            axis = list(claim.polarity_axis) if claim and claim.polarity_axis else []
            parts.append(f"{node.position.value}|{anchor}|{axis}")
            node_id = node.parent_id
        return ">".join(reversed(parts))

    def _content_address(
        self,
        anchor: Tuple[float, ...],
        polarity_axis: Tuple[float, ...],
    ) -> str:
        """Compute a deterministic node id from coordinate inputs only.

        Under the post-and-coparent refactor, the hash is over
        {anchor, axis}. Two solvers proposing the same coordinate-
        anchored claim get the same node id regardless of which
        parent they hung it under; on merge, the parent links
        accumulate on the shared node (federation-friendly).
        """
        payload = json.dumps(
            {
                "anchor": list(anchor) if anchor else [],
                "axis": list(polarity_axis) if polarity_axis else [],
            },
            sort_keys=True,
        )
        return "n_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def sprout_child(
        self,
        parent_node_id: str,
        position: Position,
        anchor: Tuple[float, ...],
        polarity_axis: Tuple[float, ...],
        observation: Optional[Observation] = None,
        content: str = "",
        world: Optional["World"] = None,
    ) -> Node:
        """Add a child node + matching claim under an existing node.

        The child's polarity_axis defaults to the parent's if none
        provided. Stakes are unit-weight posts; magnitude lives in
        the count of posts and per-node `n`.

        Content-addressed id: hashed over {anchor, axis} only. Two
        solvers proposing the same coordinate-anchored claim get the
        same node id regardless of parent, so federation merges
        naturally accumulate parent edges on a single node.

        If `world` is provided, cross-tendency edge discovery runs:
        for every other tendency whose anchor falls within
        `bandwidth * 1.5` of the new node's anchor, a parent link is
        appended pointing at that tendency's nearest existing node
        (or root). The polarity at the new edge is determined by
        projecting the new node's anchor onto that tendency's
        polarity axis. This is how nodes become "work items"
        bridging multiple tendencies — emergent multi-parenthood
        rather than an explicit type.
        """
        parent_node = self.tree.get_node(parent_node_id)
        if parent_node is None:
            raise ValueError(f"unknown node: {parent_node_id}")
        parent_claim = self._node_to_claim[parent_node_id]

        if not polarity_axis:
            polarity_axis = parent_claim.polarity_axis

        # Hash from coordinates only; parent set is structural metadata
        # appended onto the node, not part of identity.
        new_id = self._content_address(anchor, polarity_axis)

        # If the same content-addressed node already exists in this
        # tree, just append our parent edge if missing and return it.
        # Self-references are skipped (a node can't be its own parent).
        existing = self.tree.get_node(new_id)
        if existing is not None:
            if parent_node_id != existing.id:
                existing.add_parent_link(parent_node_id, position, self.id)
                # Make sure tree's pro/con child lists for the parent
                # include this existing node (in case it was originally
                # sprouted via a different path).
                parent_pro = parent_node.pro_children
                parent_con = parent_node.con_children
                if existing not in parent_pro and existing not in parent_con:
                    if position == Position.PRO:
                        parent_pro.append(existing)
                    else:
                        parent_con.append(existing)
                    parent_node.invalidate_cache()
            self._maybe_add_cross_tendency_edges(existing, anchor, world)
            return existing

        # New node in the tree. The parent link (with position +
        # tendency_id) is appended by tree.add_node -> add_child.
        new_node = Node(
            id=new_id,
            observation_id=observation.id if observation else None,
            tree_id=self.tree.id,
            content=content,
        )
        self.tree.add_node(parent_node_id, new_node, position, tendency_id=self.id)

        # Matching claim under the parent claim
        new_claim = CoordinateClaim(
            content=content or (observation.label if observation else ""),
            depth=parent_claim.depth + 1,
            stake=0.0,
            anchor=anchor,
            polarity_axis=polarity_axis,
        )
        parent_claim.children.append(new_claim)
        self._node_to_claim[new_node.id] = new_claim

        # Refresh frame's claim list (frame is recreated to keep it
        # internally consistent; integrated set is preserved)
        self._refresh_frame()

        # Cross-tendency edge discovery -- replaces the old
        # _sub_claim_staking locality propagation.
        self._maybe_add_cross_tendency_edges(new_node, anchor, world)
        return new_node

    def _maybe_add_cross_tendency_edges(
        self,
        node: Node,
        anchor: Tuple[float, ...],
        world: Optional["World"],
    ) -> None:
        """If a world is provided, walk every other tendency and
        append a parent edge to nodes whose coordinate sits inside
        the locality bandwidth of this node's anchor. The position
        at the new edge is determined by sign of the dot product
        between the anchor and that tendency's polarity axis.

        Idempotent: re-calling with the same node + world is a no-op
        once the edges exist.
        """
        if world is None or not anchor:
            return
        my_id = self.id
        for other_id, other in world.tendencies.items():
            if other_id == my_id:
                continue
            if not other.anchor:
                continue
            d = math.sqrt(sum(
                (a - b) ** 2 for a, b in zip(anchor, other.anchor)
            ))
            if d >= other.bandwidth * 1.5:
                continue
            # Find the nearest existing node in the other tendency's
            # tree to be the parent of our edge. If nothing better,
            # fall back to the other root. Skip self -- a node can't
            # be its own parent (this comes up under co-parenting:
            # `node` may already be indexed in the other tendency's
            # tree from a prior round).
            other_parent_id = other.tree.root_node.id
            best_d = float("inf")
            for cand in other.tree.all_nodes():
                if cand.id == other.tree.root_node.id:
                    continue
                if cand.id == node.id:
                    continue
                claim = other._node_to_claim.get(cand.id)
                if claim is None or not claim.anchor:
                    continue
                cand_d = math.sqrt(sum(
                    (a - b) ** 2 for a, b in zip(anchor, claim.anchor)
                ))
                if cand_d < best_d:
                    best_d = cand_d
                    other_parent_id = cand.id
            # Determine PRO vs CON by polarity-axis projection.
            dot = 0.0
            if other.polarity_axis:
                dot = sum(a * p for a, p in zip(anchor, other.polarity_axis))
            other_position = Position.PRO if dot >= 0 else Position.CON
            # Append the parent edge if not already present.
            had = any(
                p.parent_id == other_parent_id and p.tendency_id == other_id
                for p in node.parents
            )
            if had:
                continue
            node.add_parent_link(other_parent_id, other_position, other_id)
            # Reflect the edge in the other tendency's tree by
            # listing this node as a child of `other_parent_id`.
            other_parent_node = other.tree.get_node(other_parent_id)
            if other_parent_node is None:
                continue
            if (
                node not in other_parent_node.pro_children
                and node not in other_parent_node.con_children
            ):
                if other_position == Position.PRO:
                    other_parent_node.pro_children.append(node)
                else:
                    other_parent_node.con_children.append(node)
                other_parent_node.invalidate_cache()
            # Index the node in the other tendency's tree so
            # tree.get_node finds it.
            other.tree._node_index[node.id] = node

    def _refresh_frame(self) -> None:
        self.frame = CoordinateFrame(
            claims=[self._root_claim],
            integrated=dict(self.frame.integrated),
            bandwidth=self.bandwidth,
            topic_threshold=self.frame.topic_threshold,
            pro_threshold=self.frame.pro_threshold,
            contain_distance=self.frame.contain_distance,
        )

    # ----- Action -----

    def evaluate(
        self,
        observation: Observation,
    ) -> Tuple[Termination, float, Optional[CoordinateClaim]]:
        """Run the novelty probe on an observation against this
        tendency's frame. Returns (termination, composite_novelty,
        most-related-claim).
        """
        result = self.probe.measure(observation, self.frame)
        # Find best-matching claim from the frame
        related = self.frame.find_claims(observation)
        best = related[0][0] if related else None
        return result.termination, result.composite, best  # type: ignore[return-value]

    def act(self, world: "World") -> None:
        """Compute and record this tendency's posts for the current
        observation set and world state.

        Under the post-only refactor, every intent is unit-magnitude
        (+1 or -1). Magnitude lives in the count of posts and in
        per-node `n` rather than in stake weight.

        For each observation:
          - Evaluate stance through the probe.
          - INTEGRATED -> ensure PRO child under the related claim,
            post +1 on it, absorb the obs.
          - CONTRADICTS_ROOT / DISRUPTS -> ensure CON child under the
            contradicted claim, post +1 on it (CON-position drags
            parent score down).
          - ORTHOGONAL / MAX_ITER -> ignore.

        Cross-tendency posts: for each other tendency's node, post
        +1 if INTEGRATED into our frame, -1 if CONTRADICTS/DISRUPTS,
        nothing otherwise.

        Stakes are stored in self.last_stakes as a (target, node) ->
        signed-unit dict; World.apply_stakes writes the actual posts.
        """
        intents: dict[tuple[str, str], float] = {}

        # Capacity update: smooth-promotion bookkeeping. Under the
        # post-only model the input is a count of positive posts.
        self._update_capacities()

        # 1. Stake on / sprout from incoming observations against own tree
        for obs in world.observations.values():
            # If we've already CON-positioned this obs in a prior round,
            # don't re-evaluate -- just re-post on the existing CON
            # child at unit weight.
            if obs.id in self._con_positioned:
                existing_id = self._find_existing_obs_child(obs.id, position=Position.CON)
                if existing_id is not None:
                    intents[(self.id, existing_id)] = intents.get((self.id, existing_id), 0.0) + 1.0
                    continue
            term, novelty, claim = self.evaluate(obs)
            if claim is None:
                continue
            parent_node_id = self._claim_to_node_id(claim)
            if parent_node_id is None:
                continue

            if term == Termination.INTEGRATED:
                child_id = self._ensure_obs_child(parent_node_id, obs, position=Position.PRO, world=world)
                intents[(self.id, child_id)] = intents.get((self.id, child_id), 0.0) + 1.0
                # Absorb only on PRO -- this obs *fits* our worldview.
                self.frame = self.frame.absorb(obs)

            elif term in (Termination.CONTRADICTS_ROOT, Termination.DISRUPTS):
                child_id = self._ensure_obs_child(parent_node_id, obs, position=Position.CON, world=world)
                intents[(self.id, child_id)] = intents.get((self.id, child_id), 0.0) + 1.0
                self._con_positioned.add(obs.id)

        # 2. Cross-tendency posts on other tendencies' nodes.
        # Under the post-only model these are unit-magnitude intents
        # signed by stance: +1 if INTEGRATED, -1 if
        # CONTRADICTS/DISRUPTS, 0 otherwise.
        for other in world.tendencies.values():
            if other.id == self.id:
                continue
            for node in other.tree.all_nodes():
                if node.observation_id is not None:
                    target = world.observations.get(node.observation_id)
                    if target is None:
                        continue
                else:
                    foreign_claim = other._node_to_claim.get(node.id)
                    if foreign_claim is None or not foreign_claim.anchor:
                        continue
                    target = Observation(
                        id=f"_anchor_{other.id}_{node.id}",
                        coords=foreign_claim.anchor,
                        label=f"anchor:{foreign_claim.content}",
                    )
                term, _novelty, _claim = self.evaluate(target)
                if term == Termination.INTEGRATED:
                    signed = +1.0
                elif term in (Termination.CONTRADICTS_ROOT, Termination.DISRUPTS):
                    signed = -1.0
                else:
                    signed = 0.0
                if signed != 0.0:
                    key = (other.id, node.id)
                    intents[key] = intents.get(key, 0.0) + signed

        # 3. Budget as integer post-cap. If the tendency's intent
        # count exceeds budget, scale down proportionally. Default
        # budget=1.0 leaves intents as-is.
        intents = {k: v * self.budget for k, v in intents.items()}
        self.last_stakes = intents

    def _update_capacities(self) -> None:
        """Blend each node's accumulated positive stake into its
        capacity. Capacity is the node's outbound staking budget.

        Update rule: cap_new = decay * cap_old + (1 - decay) * pro_in
        where pro_in = sum of positive stakes currently on the node.

        Root node's capacity is pinned at self.budget (it always has
        full standing). Other nodes start at 0 and earn capacity as
        PRO stake accumulates.
        """
        root_id = self.tree.root_node.id
        for node in self.tree.all_nodes():
            if node.id == root_id:
                self.node_capacity[node.id] = self.budget
                continue
            pro_in = sum(s.weight for s in node.stakes if s.weight > 0)
            old = self.node_capacity.get(node.id, 0.0)
            new = self.capacity_decay * old + (1.0 - self.capacity_decay) * pro_in
            self.node_capacity[node.id] = new

    def update_novelty(self, dt: float = 1.0) -> None:
        """Update per-node persistent novelty n based on the current
        round's stake deltas.

        Discretized form of:
          dn/dt = -gamma_pro * n * pro_rate
                + gamma_con * (1-n) * con_rate
                + epsilon * (1-n)

        Per-round we use stake deltas as proxies for the rates:
          pro_rate_i = positive stakes currently on node i (round-fresh,
                       since apply_stakes wipes prior round)
          con_rate_i = magnitude of CON-position influence: net negative
                       contribution from CON children's net_score,
                       capped at 0 if children sum positively.

        n is clipped to [0, 1] after the update.
        """
        root_id = self.tree.root_node.id
        for node in self.tree.all_nodes():
            if node.id == root_id:
                # Root never decays in novelty -- it's the founding
                # claim and stays "always potentially surprising" in
                # the sense that further evidence can always shift it.
                # Keep n=1.0 anchored at the root.
                node.n = 1.0
                continue
            pro_rate = sum(s.weight for s in node.stakes if s.weight > 0)
            # CON pressure on this node: weight of CON children's net
            # contribution. If this node has CON children with positive
            # net_score, those drive n upward (re-surprise).
            con_rate = sum(max(c.net_score, 0.0) for c in node.con_children)
            n = node.n
            d_n = (
                -self.novelty_gamma_pro * n * pro_rate
                + self.novelty_gamma_con * (1.0 - n) * con_rate
                + self.novelty_drift * (1.0 - n)
            ) * dt
            new_n = n + d_n
            # Clip
            if new_n < 0.0:
                new_n = 0.0
            elif new_n > 1.0:
                new_n = 1.0
            node.n = new_n

    def intrinsic_score(self, node: Node) -> float:
        """How strongly this node is supported by its full subtree
        across ALL tendencies that share it.

        Walks every child edge regardless of which tendency owns it:
          intrinsic_score(node) = len(stakes)
                                + Σ intrinsic_score(c) for c PRO of node anywhere
                                - Σ intrinsic_score(c) for c CON of node anywhere

        For single-parent nodes, this collapses to the existing
        recursion. For multi-parented (work-item) nodes, the same
        intrinsic value gets signed differently when read up each
        parent's tree (see substrate architecture docs).
        """
        return _intrinsic_score(node)

    def _find_existing_obs_child(
        self,
        observation_id: str,
        position: Position,
    ) -> Optional[str]:
        """Search this tendency's tree for a node with the given
        observation_id whose first parent_link's position matches.
        Returns its node id or None. Used for the (legacy) same-
        position re-stake path in `act`.
        """
        for node in self.tree.all_nodes():
            if node.observation_id == observation_id and node.position == position:
                return node.id
        return None

    def _find_obs_node_anywhere(
        self,
        world: Optional["World"],
        observation_id: str,
    ) -> Optional[Node]:
        """Search across all tendencies in `world` for any node bearing
        this observation_id. Used by `_ensure_obs_child` to dedup
        same-observation sprouts that two tendencies might otherwise
        place as separately-hashed nodes (different polarity_axis ->
        different content hash even though it's the same observation).
        Falls back to this tendency's own tree if `world` is None.
        """
        if world is not None:
            for tendency in world.tendencies.values():
                for node in tendency.tree.all_nodes():
                    if node.observation_id == observation_id:
                        return node
        for node in self.tree.all_nodes():
            if node.observation_id == observation_id:
                return node
        return None

    def _ensure_obs_child(
        self,
        parent_node_id: str,
        observation: Observation,
        position: Position,
        world: Optional["World"] = None,
    ) -> str:
        """Find or create a node carrying this observation as a child of
        the given parent at the requested position. Returns the node id.

        Three-step dedup, in order:

          1. Existing node in THIS tendency's tree with the same
             (observation_id, position) -- direct re-use (the legacy
             case).
          2. Existing node anywhere in `world` with this
             observation_id, regardless of its current position --
             append a parent link in this tendency at the requested
             position; do NOT create a new hash. This prevents
             "two parallel S11 nodes" when two tendencies disagree
             on the obs's polarity, and is the surgical fix for the
             sub-child accretion that cancels veto contributions.
          3. Otherwise sprout fresh (existing behavior). Cross-
             tendency edge discovery still runs via `world`.
        """
        existing = self._find_existing_obs_child(observation.id, position)
        if existing is not None:
            return existing

        # Cross-tendency dedup: another tendency may have already
        # sprouted a node for this observation. Reuse it.
        anywhere = self._find_obs_node_anywhere(world, observation.id)
        if anywhere is not None:
            # Make sure this tendency's tree indexes the node, has a
            # parent link at the requested position, and lists it as
            # a child of the requested parent.
            anywhere.add_parent_link(parent_node_id, position, self.id)
            self.tree._node_index[anywhere.id] = anywhere
            parent_node = self.tree.get_node(parent_node_id)
            if parent_node is not None:
                if anywhere not in parent_node.pro_children and anywhere not in parent_node.con_children:
                    if position == Position.PRO:
                        parent_node.pro_children.append(anywhere)
                    else:
                        parent_node.con_children.append(anywhere)
                    parent_node.invalidate_cache()
            # Make sure this tendency has a claim record so `act`
            # downstream lookups don't break.
            if anywhere.id not in self._node_to_claim:
                parent_claim = self._node_to_claim.get(parent_node_id)
                if parent_claim is not None:
                    polarity = parent_claim.polarity_axis
                    if position == Position.CON:
                        polarity = tuple(-u for u in polarity)
                    new_claim = CoordinateClaim(
                        content=observation.label or f"obs:{observation.id[:6]}",
                        depth=parent_claim.depth + 1,
                        stake=0.0,
                        anchor=observation.coords,
                        polarity_axis=polarity,
                    )
                    self._node_to_claim[anywhere.id] = new_claim
            return anywhere.id

        parent_claim = self._node_to_claim[parent_node_id]
        anchor = observation.coords
        polarity = parent_claim.polarity_axis
        if position == Position.CON:
            polarity = tuple(-u for u in polarity)
        new_node = self.sprout_child(
            parent_node_id=parent_node_id,
            position=position,
            anchor=anchor,
            polarity_axis=polarity,
            observation=observation,
            content=observation.label or f"obs:{observation.id[:6]}",
            world=world,
        )
        return new_node.id

    # ----- Mapping helpers -----

    def _claim_to_node_id(self, claim: CoordinateClaim) -> Optional[str]:
        for node_id, c in self._node_to_claim.items():
            if c is claim:
                return node_id
        return None

    def __repr__(self) -> str:
        return (
            f"GeneralizedTendency(id={self.id!r}, thesis={self.thesis!r}, "
            f"budget={self.budget:.2f}, nodes={len(self._node_to_claim)})"
        )

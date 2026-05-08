"""
Hierarchical staking - builds tree depth through two phases:

Phase 1 (Anchor Identification):
    Find observations that directly address the tree's root value.
    These become first-level children of the root.

Phase 2 (Relational Staking):
    For remaining observations, find which existing node they
    support or contradict. This builds tree depth.

Efficiency: O(trees + observations) instead of O(trees × observations)
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from ..models.agent import AgentSet, Tendency
from ..models.tree import Tree, Node, Position
from ..models.observation import Observation, ObservationStore


@dataclass
class AnchorResult:
    """Result of anchor identification for one observation."""
    observation_id: str
    is_anchor: bool
    relevance: float  # 0.0 - 1.0
    reasoning: str


@dataclass
class RelationalResult:
    """Result of relational staking for one observation."""
    observation_id: str
    target_node_id: Optional[str]  # Which node this relates to (None = not relevant)
    position: Optional[Position]   # PRO or CON relative to target
    relevance: float
    reasoning: str


ANCHOR_PROMPT = """You are identifying which observations directly address a core value.

## The Core Value
{root_value}

## Observations
{observations}

## Task

For each observation, determine:
1. Does it DIRECTLY address the core value? (not tangentially related - directly about it)
2. How relevant is it? (0.0 = not at all, 1.0 = central to this value)

An observation is an "anchor" if it's relevance >= 0.6.

## Examples

Core value: "Decentralized coordination and DAO governance"
- "Works at dOrg on DAO tooling" → anchor (directly about DAOs, relevance=0.9)
- "10 years building governance frameworks" → anchor (directly about governance, relevance=0.85)
- "Lives paycheck to paycheck" → not anchor (financial, not about coordination, relevance=0.1)
- "Posted inflammatory messages at 4 AM" → not anchor (interpersonal, relevance=0.3)

## Output Format

Return JSON array with one entry per observation:
```json
[
  {{"observation_id": "...", "is_anchor": true, "relevance": 0.9, "reasoning": "..."}},
  {{"observation_id": "...", "is_anchor": false, "relevance": 0.2, "reasoning": "..."}},
  ...
]
```

Return ONLY the JSON array."""


RELATIONAL_PROMPT = """You are positioning an observation within a value tree.

## The Value Tree

Root: {root_value}

Current nodes:
{existing_nodes}

## The Observation to Position

ID: {observation_id}
Content: {observation_content}

## Task

Determine which existing node (if any) this observation supports or contradicts.

Rules:
1. Look for LOGICAL relationships, not just word similarity
2. An observation can support (PRO) or contradict (CON) a node
3. If it doesn't meaningfully relate to any existing node, return null
4. Prefer relating to specific nodes over the root (builds depth)

## Examples

Existing node: "Works at dOrg on DAO tooling"
Observation: "Posted inflammatory messages to colleague at 4 AM"
→ CON for "Works at dOrg" - undermines the collaborative work implied

Existing node: "10 years building governance frameworks"
Observation: "ViraTrace was rejected by EU for not being profitable"
→ PRO for "10 years building" - evidence of the long, difficult journey

## Output Format

Return JSON:
```json
{{
  "target_node_id": "..." or null,
  "position": "pro" or "con" or null,
  "relevance": 0.0-1.0,
  "reasoning": "..."
}}
```

Return ONLY the JSON object."""


AGENT_STAKES_PROMPT = """Determine how human tendencies view this positioned observation.

## Context

Tree root: {root_value}
Parent node: {parent_content}
Observation: {observation_content}
Position relative to parent: {position}

## Tendencies

For each tendency, determine how much they "care" about this observation (0.0 = irrelevant, 1.0 = highly relevant):

- SURVIVAL: Physical safety, resource acquisition, risk mitigation
- STATUS: Social standing, achievement, recognition
- MEANING: Significance, impact, legacy, purpose
- CONNECTION: Relationships, belonging, community
- AUTONOMY: Independence, self-determination, freedom
- COMFORT: Ease, pleasure, avoiding pain
- CURIOSITY: Knowledge, understanding, exploration

## Output Format

Return JSON array:
```json
[
  {{"tendency": "survival", "weight": 0.0-1.0}},
  {{"tendency": "status", "weight": 0.0-1.0}},
  ...
]
```

Return ONLY the JSON array."""


class HierarchicalStaker:
    """
    Two-phase staking that builds tree depth.

    Phase 1: Identify anchors (observations that directly address root value)
    Phase 2: Position remaining observations relative to existing nodes
    """

    def __init__(self, work_dir: Optional[str] = None):
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.gettempdir())
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def _call_claude(self, prompt: str, timeout: int = 300) -> str:
        """Call Claude CLI with prompt."""
        prompt_file = self.work_dir / f"prompt_{hash(prompt) % 100000}.txt"
        prompt_file.write_text(prompt, encoding="utf-8")

        try:
            if os.name == 'nt':
                cmd = f'type "{prompt_file}" | claude -p --dangerously-skip-permissions'
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True,
                    timeout=timeout, cwd=str(self.work_dir)
                )
            else:
                with open(prompt_file, 'r') as f:
                    result = subprocess.run(
                        ['claude', '-p', '--dangerously-skip-permissions'],
                        stdin=f, capture_output=True, text=True,
                        timeout=timeout, cwd=str(self.work_dir)
                    )

            if result.returncode != 0:
                raise RuntimeError(f"Claude CLI failed: {result.stderr}")

            return result.stdout

        finally:
            try:
                prompt_file.unlink()
            except:
                pass

    def _parse_json_array(self, text: str) -> list:
        """Extract JSON array from response."""
        start = text.find('[')
        end = text.rfind(']') + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON array found: {text[:500]}...")
        return json.loads(text[start:end])

    def _parse_json_object(self, text: str) -> dict:
        """Extract JSON object from response."""
        start = text.find('{')
        end = text.rfind('}') + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON object found: {text[:500]}...")
        return json.loads(text[start:end])

    # =========================================================================
    # Phase 1: Anchor Identification
    # =========================================================================

    def identify_anchors(
        self,
        observations: list[Observation],
        tree: Tree,
        batch_size: int = 30,
    ) -> list[AnchorResult]:
        """
        Identify which observations directly address the tree's root value.

        Args:
            observations: All observations to evaluate
            tree: The tree to find anchors for
            batch_size: How many observations per Claude call

        Returns:
            List of AnchorResult for each observation
        """
        all_results = []

        # Process in batches
        for i in range(0, len(observations), batch_size):
            batch = observations[i:i + batch_size]

            # Format observations for prompt
            obs_text = "\n".join(
                f"- ID: {obs.id}\n  Content: {obs.content}"
                for obs in batch
            )

            prompt = ANCHOR_PROMPT.format(
                root_value=tree.root_value,
                observations=obs_text,
            )

            response = self._call_claude(prompt)
            results = self._parse_json_array(response)

            for item in results:
                all_results.append(AnchorResult(
                    observation_id=item["observation_id"],
                    is_anchor=item.get("is_anchor", False),
                    relevance=float(item.get("relevance", 0.0)),
                    reasoning=item.get("reasoning", ""),
                ))

        return all_results

    # =========================================================================
    # Phase 2: Relational Staking
    # =========================================================================

    def stake_relational(
        self,
        observation: Observation,
        tree: Tree,
        agents: AgentSet,
    ) -> Optional[Node]:
        """
        Position an observation relative to existing nodes in the tree.

        Args:
            observation: The observation to position
            tree: The tree with existing nodes
            agents: Agent set for stake weights

        Returns:
            The created node, or None if not relevant
        """
        # Format existing nodes
        existing_nodes = tree.all_nodes()
        nodes_text = "\n".join(
            f"- ID: {node.id}\n  Content: {node.content}\n  Position: {node.position.value}"
            for node in existing_nodes
        )

        # Get relational positioning
        prompt = RELATIONAL_PROMPT.format(
            root_value=tree.root_value,
            existing_nodes=nodes_text,
            observation_id=observation.id,
            observation_content=observation.content,
        )

        response = self._call_claude(prompt)
        result = self._parse_json_object(response)

        target_node_id = result.get("target_node_id")
        position_str = result.get("position")

        if not target_node_id or not position_str:
            return None  # Not relevant to this tree

        position = Position.PRO if position_str == "pro" else Position.CON
        target_node = tree.get_node(target_node_id)

        if not target_node:
            # Fallback to root if target not found
            target_node = tree.root_node

        # Create the node
        node = Node(
            observation_id=observation.id,
            content=observation.content,
            tree_id=tree.id,
        )

        # Get agent stakes
        stakes = self._get_agent_stakes(observation, tree, target_node, position, agents)
        for tendency, weight in stakes.items():
            if weight > 0.01:
                agent = agents.get(tendency)
                stake_weight = weight * agent.allocation
                node.add_stake(tendency.value, stake_weight)
                agent.stakes_placed += 1

        # Store reasoning
        node.metadata = {
            "relational_reasoning": result.get("reasoning", ""),
            "relevance": result.get("relevance", 0.0),
        }

        # Add to tree
        tree.add_node(target_node.id, node, position)

        return node

    def _get_agent_stakes(
        self,
        observation: Observation,
        tree: Tree,
        parent_node: Node,
        position: Position,
        agents: AgentSet,
    ) -> dict[Tendency, float]:
        """Get stake weights from each agent for this observation."""
        prompt = AGENT_STAKES_PROMPT.format(
            root_value=tree.root_value,
            parent_content=parent_node.content,
            observation_content=observation.content,
            position=position.value,
        )

        response = self._call_claude(prompt, timeout=60)
        results = self._parse_json_array(response)

        stakes = {}
        for item in results:
            tendency = Tendency(item["tendency"].lower())
            stakes[tendency] = float(item.get("weight", 0.0))

        return stakes

    # =========================================================================
    # Full Pipeline
    # =========================================================================

    def stake_all(
        self,
        store: ObservationStore,
        tree: Tree,
        agents: AgentSet,
        anchor_threshold: float = 0.6,
        verbose: bool = True,
    ) -> dict:
        """
        Full two-phase staking pipeline.

        Args:
            store: All observations
            tree: Target tree
            agents: Agent set
            anchor_threshold: Minimum relevance to be an anchor
            verbose: Print progress

        Returns:
            Stats dict with counts
        """
        observations = store.all()

        if verbose:
            print(f"\n{'='*60}")
            print(f"Staking into: {tree.root_value}")
            print(f"{'='*60}")

        # Phase 1: Identify anchors
        if verbose:
            print(f"\nPhase 1: Identifying anchors from {len(observations)} observations...")

        anchor_results = self.identify_anchors(observations, tree)

        anchors = [r for r in anchor_results if r.is_anchor]
        non_anchors = [r for r in anchor_results if not r.is_anchor]

        if verbose:
            print(f"  Found {len(anchors)} anchors")

        # Create anchor nodes (direct children of root)
        anchor_nodes = []
        obs_by_id = {obs.id: obs for obs in observations}

        for result in anchors:
            obs = obs_by_id.get(result.observation_id)
            if not obs:
                continue

            node = Node(
                observation_id=obs.id,
                content=obs.content,
                tree_id=tree.id,
            )

            # Get agent stakes for anchors
            stakes = self._get_agent_stakes(obs, tree, tree.root_node, Position.PRO, agents)
            for tendency, weight in stakes.items():
                if weight > 0.01:
                    agent = agents.get(tendency)
                    stake_weight = weight * agent.allocation
                    node.add_stake(tendency.value, stake_weight)
                    agent.stakes_placed += 1

            node.metadata = {
                "is_anchor": True,
                "anchor_relevance": result.relevance,
                "anchor_reasoning": result.reasoning,
            }

            # Anchors are PRO for the root value (they exemplify it)
            tree.add_node(tree.root_node.id, node, Position.PRO)
            anchor_nodes.append(node)

            if verbose:
                print(f"  + Anchor: {obs.content[:50]}...")

        # Phase 2: Relational staking for non-anchors
        if verbose:
            print(f"\nPhase 2: Relational staking for {len(non_anchors)} remaining observations...")

        relational_count = 0
        skipped_count = 0

        for i, result in enumerate(non_anchors):
            obs = obs_by_id.get(result.observation_id)
            if not obs:
                continue

            # Skip very low relevance observations
            if result.relevance < 0.15:
                skipped_count += 1
                continue

            if verbose and (i + 1) % 10 == 0:
                print(f"  Processing {i + 1}/{len(non_anchors)}...")

            node = self.stake_relational(obs, tree, agents)
            if node:
                relational_count += 1
                if verbose:
                    parent = tree.get_node(node.parent_id)
                    parent_preview = parent.content[:30] if parent else "root"
                    print(f"    {node.position.value.upper()} -> '{parent_preview}...': {obs.content[:40]}...")

        if verbose:
            print(f"\n  Relational nodes created: {relational_count}")
            print(f"  Skipped (low relevance): {skipped_count}")

        # Summary
        stats = {
            "total_observations": len(observations),
            "anchors": len(anchors),
            "relational_nodes": relational_count,
            "skipped": skipped_count,
            "tree_score": tree.score,
            "tree_depth": tree.depth(),
            "total_nodes": len(tree.all_nodes()),
        }

        if verbose:
            print(f"\n{'='*60}")
            print(f"Results:")
            print(f"  Tree score: {tree.score:.3f}")
            print(f"  Tree depth: {tree.depth()}")
            print(f"  Total nodes: {len(tree.all_nodes())}")
            print(f"{'='*60}")

        return stats

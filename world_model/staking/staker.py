"""
Staking mechanism - how agents decide where to stake on trees.

Agents analyze observations in context of a tree's root value and
determine their position (pro/con) and stake weight.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from ..models.agent import AgentSet, Tendency
from ..models.tree import Tree, Node, Position, Stake
from ..models.observation import Observation


@dataclass
class StakeDecision:
    """An agent's decision about an observation relative to a claim."""
    tendency: Tendency
    position: Optional[Position]  # None = no stake (not relevant)
    weight: float                 # 0.0 - 1.0 of agent's allocation
    reasoning: str                # Why this position


STAKING_PROMPT = """You are analyzing how different human tendencies would view an observation in context of a value claim.

## The Claim (Root Value)
{claim}

## The Observation
{observation}

## Human Tendencies to Evaluate

For EACH tendency below, determine:
1. Is this observation RELEVANT to this tendency's optimization target?
2. If relevant, is the observation PRO (supports the claim) or CON (contradicts it) from this tendency's perspective?
3. How strongly relevant? (0.0 = not relevant, 1.0 = highly relevant)

Tendencies:
- SURVIVAL: Physical safety, resource acquisition, risk mitigation
- STATUS: Social standing, achievement, recognition, being valued
- MEANING: Significance, impact, legacy, purpose beyond self
- CONNECTION: Relationships, belonging, community, being known
- AUTONOMY: Independence, self-determination, freedom from constraint
- COMFORT: Ease, pleasure, avoiding pain, reducing friction
- CURIOSITY: Knowledge, understanding, exploration, novelty

## Key Rules

1. Same observation can be PRO from one tendency and CON from another
2. "Not relevant" (null position) is valid - not everything matters to every tendency
3. Weight reflects RELEVANCE, not agreement - high weight means this observation strongly speaks to this tendency
4. Be specific about WHY in reasoning

## Output Format

Return JSON array:
```json
[
  {{"tendency": "SURVIVAL", "position": "pro" | "con" | null, "weight": 0.0-1.0, "reasoning": "..."}},
  {{"tendency": "STATUS", "position": "pro" | "con" | null, "weight": 0.0-1.0, "reasoning": "..."}},
  ...
]
```

Return ONLY the JSON array, no other text."""


class Staker:
    """
    Determines how agents stake on observations relative to claims.

    Uses Claude to analyze the semantic relationship between observations
    and value claims from each tendency's perspective.
    """

    def __init__(self, work_dir: Optional[str] = None):
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.gettempdir())
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def analyze_observation(
        self,
        observation: Observation,
        claim: str,
    ) -> list[StakeDecision]:
        """
        Analyze how each tendency views an observation relative to a claim.

        Args:
            observation: The observation to analyze
            claim: The value claim (e.g., tree root value)

        Returns:
            List of StakeDecision for each tendency
        """
        prompt = STAKING_PROMPT.format(
            claim=claim,
            observation=observation.content
        )

        # Write prompt to temp file
        prompt_file = self.work_dir / f"stake_prompt_{observation.id[:8]}.txt"

        prompt_file.write_text(prompt, encoding="utf-8")

        try:
            # Call Claude CLI
            if os.name == 'nt':  # Windows
                cmd = f'type "{prompt_file}" | claude -p --dangerously-skip-permissions'
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=str(self.work_dir)
                )
            else:  # Unix
                with open(prompt_file, 'r') as f:
                    result = subprocess.run(
                        ['claude', '-p', '--dangerously-skip-permissions'],
                        stdin=f,
                        capture_output=True,
                        text=True,
                        timeout=120,
                        cwd=str(self.work_dir)
                    )

            if result.returncode != 0:
                raise RuntimeError(f"Claude CLI failed: {result.stderr}")

            response_text = result.stdout

        finally:
            try:
                prompt_file.unlink()
            except:
                pass

        # Parse JSON
        decisions = self._parse_decisions(response_text)
        return decisions

    def _parse_decisions(self, text: str) -> list[StakeDecision]:
        """Parse JSON response into StakeDecision objects."""
        start = text.find('[')
        end = text.rfind(']') + 1

        if start == -1 or end == 0:
            raise ValueError(f"No JSON array found in response: {text[:500]}...")

        json_str = text[start:end]
        data = json.loads(json_str)

        decisions = []
        for item in data:
            tendency = Tendency(item["tendency"].lower())
            position = None
            if item.get("position"):
                position = Position.PRO if item["position"] == "pro" else Position.CON

            decisions.append(StakeDecision(
                tendency=tendency,
                position=position,
                weight=float(item.get("weight", 0.0)),
                reasoning=item.get("reasoning", ""),
            ))

        return decisions

    def stake_observation(
        self,
        observation: Observation,
        tree: Tree,
        agents: AgentSet,
        parent_node: Optional[Node] = None,
    ) -> Optional[Node]:
        """
        Create a node for an observation and stake it into a tree.

        Args:
            observation: The observation to stake
            tree: The tree to stake into
            agents: The agent set (for allocations)
            parent_node: Parent node (default: tree root)

        Returns:
            The created node, or None if no agent found the observation relevant
        """
        if parent_node is None:
            parent_node = tree.root_node

        # Get stake decisions from Claude
        claim = parent_node.content if parent_node.content else tree.root_value
        decisions = self.analyze_observation(observation, claim)

        # Filter to relevant decisions (non-null position, non-zero weight)
        relevant = [d for d in decisions if d.position is not None and d.weight > 0.01]

        if not relevant:
            return None  # No agent found this observation relevant

        # Determine dominant position (majority vote weighted by stakes)
        pro_weight = sum(
            d.weight * agents.get(d.tendency).allocation
            for d in relevant if d.position == Position.PRO
        )
        con_weight = sum(
            d.weight * agents.get(d.tendency).allocation
            for d in relevant if d.position == Position.CON
        )

        # Position is determined by stronger vote
        node_position = Position.PRO if pro_weight >= con_weight else Position.CON

        # Create node
        node = Node(
            observation_id=observation.id,
            tree_id=tree.id,
            content=observation.content,
        )

        # Add stakes from each relevant agent
        for decision in relevant:
            # Only stake if agent agrees with node position, OR if position is contested
            # (contested = agents disagree, we show the tension)
            agent = agents.get(decision.tendency)
            stake_weight = decision.weight * agent.allocation

            # Track stake on agent
            agent.stakes_placed += 1

            node.add_stake(decision.tendency.value, stake_weight)

        # Store decision reasoning in metadata
        node.metadata = {
            "decisions": [
                {
                    "tendency": d.tendency.value,
                    "position": d.position.value if d.position else None,
                    "weight": d.weight,
                    "reasoning": d.reasoning,
                }
                for d in decisions
            ],
            "pro_weight": pro_weight,
            "con_weight": con_weight,
        }

        # Add to tree
        tree.add_node(parent_node.id, node, node_position)

        return node


class BatchStaker:
    """
    Efficient staking for multiple observations.

    Batches Claude calls where possible.
    """

    def __init__(self, staker: Optional[Staker] = None):
        self.staker = staker or Staker()

    def stake_observations(
        self,
        observations: list[Observation],
        tree: Tree,
        agents: AgentSet,
        parent_node: Optional[Node] = None,
    ) -> list[Node]:
        """
        Stake multiple observations into a tree.

        Args:
            observations: Observations to stake
            tree: Target tree
            agents: Agent set
            parent_node: Parent for all observations (default: root)

        Returns:
            List of created nodes (may be shorter than input if some not relevant)
        """
        nodes = []
        for obs in observations:
            node = self.staker.stake_observation(obs, tree, agents, parent_node)
            if node:
                nodes.append(node)
        return nodes

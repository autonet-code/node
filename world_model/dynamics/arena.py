"""
Arena - Where agents compete to define a person's worldview.

The Arena orchestrates the adversarial dynamics:
1. Agents PROPOSE trees (claims about what matters)
2. Agents STAKE observations (support own claims, undermine others)
3. RESOLUTION determines winners (whose claims are best supported)
4. REALLOCATION shifts influence to winners

This is where "life" happens - the continuous competition that
shapes the equilibrium we call personality.
"""

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..models.observation import Observation, ObservationStore
from ..models.agent import Agent, AgentSet, Tendency
from ..models.tree import Tree, TreeStore, Node, Position


@dataclass
class Claim:
    """A tree proposed by an agent - their position on what matters."""
    tree: Tree
    proposer: Tendency
    score: float = 0.0

    @property
    def id(self) -> str:
        return self.tree.id


@dataclass
class StakeDecision:
    """An agent's decision about how to stake an observation."""
    observation_id: str
    tree_id: str
    position: Position  # PRO or CON
    weight: float       # How much to stake (0-1)
    staking_agent: str  # Which tendency is staking
    reasoning: str
    is_own_tree: bool   # Staking on own vs competitor tree


@dataclass
class DebateResult:
    """Results of a debate round."""
    claims: list[Claim]
    total_stakes: int
    winner: Optional[Tendency]
    scores: dict[Tendency, float]
    allocation_changes: dict[Tendency, float]


class Arena:
    """
    The arena where agents compete.

    Usage:
        arena = Arena()

        # Full onboarding
        result = arena.run_full_debate(
            observations=store,
            agents=agent_set,
            rounds=3,
        )

        # Or step by step
        claims = arena.proposal_phase(observations, agents)
        arena.staking_phase(observations, claims, agents)
        result = arena.resolution_phase(claims, agents)
    """

    def __init__(self, work_dir: Optional[str] = None):
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.gettempdir())
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def _call_claude(self, prompt: str, timeout: int = 300, retries: int = 2) -> str:
        """Call Claude CLI with prompt and retry logic."""
        import time as time_module
        prompt_file = self.work_dir / f"arena_prompt_{hash(prompt) % 100000}.txt"
        prompt_file.write_text(prompt, encoding="utf-8")

        try:
            last_error = None
            for attempt in range(retries + 1):
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

                except subprocess.TimeoutExpired as e:
                    last_error = e
                    if attempt < retries:
                        time_module.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s
                        continue
                    raise

                except Exception as e:
                    last_error = e
                    if attempt < retries:
                        time_module.sleep(2 ** attempt)
                        continue
                    raise

            # If we get here, all retries failed
            raise last_error or RuntimeError("All retries failed")

        finally:
            try:
                prompt_file.unlink()
            except:
                pass

    def _parse_json(self, text: str, array: bool = True):
        """Extract JSON from response."""
        if array:
            start, end = text.find('['), text.rfind(']') + 1
        else:
            start, end = text.find('{'), text.rfind('}') + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON found: {text[:500]}...")
        return json.loads(text[start:end])

    # =========================================================================
    # Phase 1: Proposal - Agents propose their claims
    # =========================================================================

    def proposal_phase(
        self,
        observations: ObservationStore,
        agents: AgentSet,
        verbose: bool = True,
    ) -> list[Claim]:
        """
        Each agent proposes a tree root claim based on their tendency.

        The claim is the agent's position: "This is what matters."
        """
        if verbose:
            print("\n" + "="*60)
            print("PHASE 1: PROPOSAL")
            print("="*60)

        # Sample observations for context
        obs_sample = observations.all()[:30]
        obs_text = "\n".join(f"- {obs.content}" for obs in obs_sample)

        prompt = f"""You are helping model a person's worldview through competing internal tendencies.

## Context: Observations about this person
{obs_text}
{"... and more" if len(observations) > 30 else ""}

## Task

Each human tendency below should propose a ROOT CLAIM - a statement about what fundamentally matters in life, framed from that tendency's perspective.

The claim should:
1. Be a clear position statement (not a question)
2. Reflect what this tendency optimizes for
3. Be arguable (observations can support or contradict it)
4. Be specific enough to be meaningful for THIS person

Tendencies:
- SURVIVAL: Physical safety, resource acquisition, risk mitigation
- STATUS: Social standing, achievement, recognition
- MEANING: Significance, impact, legacy, purpose beyond self
- CONNECTION: Relationships, belonging, community
- AUTONOMY: Independence, self-determination, freedom
- COMFORT: Ease, pleasure, avoiding pain
- CURIOSITY: Knowledge, understanding, exploration

## Output Format

Return JSON array with one claim per tendency:
```json
[
  {{"tendency": "survival", "claim": "Financial security and stability are foundational to a good life", "reasoning": "..."}},
  {{"tendency": "status", "claim": "...", "reasoning": "..."}},
  ...
]
```

Make each claim specific to what you observe about this person, not generic platitudes.

Return ONLY the JSON array."""

        response = self._call_claude(prompt)
        proposals = self._parse_json(response)

        claims = []
        for prop in proposals:
            tendency = Tendency(prop["tendency"].lower())
            tree = Tree(
                root_value=prop["claim"],
                description=f"Proposed by {tendency.value}: {prop.get('reasoning', '')}",
            )

            claim = Claim(tree=tree, proposer=tendency)
            claims.append(claim)

            if verbose:
                print(f"\n  [{tendency.value.upper()}] proposes:")
                print(f"    \"{prop['claim']}\"")

        return claims

    # =========================================================================
    # Phase 2: Staking - Agents stake observations adversarially
    # =========================================================================

    def staking_phase(
        self,
        observations: ObservationStore,
        claims: list[Claim],
        agents: AgentSet,
        verbose: bool = True,
    ) -> dict[str, list[Node]]:
        """
        Agents stake observations on all claims.

        Key dynamics:
        - Agents stake PRO on their own claims (support)
        - Agents stake CON on competitor claims (undermine)
        - Same observation can be PRO for one claim, CON for another
        """
        if verbose:
            print("\n" + "="*60)
            print("PHASE 2: ADVERSARIAL STAKING")
            print("="*60)

        # Build claims context
        claims_text = "\n".join(
            f"- [{c.proposer.value.upper()}] \"{c.tree.root_value}\""
            for c in claims
        )

        all_obs = observations.all()
        nodes_by_tree: dict[str, list[Node]] = {c.tree.id: [] for c in claims}

        # Process in batches
        batch_size = 20
        for i in range(0, len(all_obs), batch_size):
            batch = all_obs[i:i + batch_size]

            if verbose:
                print(f"\n  Processing observations {i+1}-{i+len(batch)}...")

            decisions = self._get_adversarial_stakes(batch, claims, agents)

            for decision in decisions:
                claim = next((c for c in claims if c.tree.id == decision.tree_id), None)
                if not claim:
                    continue

                # Create node with stake
                obs = next((o for o in batch if o.id == decision.observation_id), None)
                if not obs:
                    continue

                node = Node(
                    observation_id=obs.id,
                    content=obs.content,
                    tree_id=claim.tree.id,
                )

                # Weight = agent's allocation * decision weight
                try:
                    agent = agents.get(Tendency(decision.staking_agent))
                except ValueError:
                    agent = agents.get(claim.proposer)  # Fallback to claim proposer
                stake_weight = decision.weight * agent.allocation
                node.add_stake(agent.tendency.value, stake_weight)

                # Add to tree
                claim.tree.add_node(claim.tree.root_node.id, node, decision.position)
                nodes_by_tree[claim.tree.id].append(node)

                if verbose and len(nodes_by_tree[claim.tree.id]) <= 3:
                    print(f"    {decision.position.value.upper()} on [{claim.proposer.value}]: {obs.content[:40]}...")

        return nodes_by_tree

    def _get_adversarial_stakes(
        self,
        observations: list[Observation],
        claims: list[Claim],
        agents: AgentSet,
    ) -> list[StakeDecision]:
        """Get adversarial staking decisions for a batch of observations."""

        claims_text = "\n".join(
            f"- ID: {c.tree.id}\n  Proposer: {c.proposer.value}\n  Claim: \"{c.tree.root_value}\""
            for c in claims
        )

        obs_text = "\n".join(
            f"- ID: {obs.id}\n  Content: {obs.content}"
            for obs in observations
        )

        prompt = f"""You are simulating adversarial staking in a debate about what matters in life.

## The Claims (each proposed by a different tendency)
{claims_text}

## Observations to Stake
{obs_text}

## Adversarial Staking Rules

For each observation, determine how it should be staked:

1. **Support your own claim**: If you're the MEANING tendency and an observation supports meaning/purpose, stake it PRO on the meaning claim.

2. **Undermine competitors**: If an observation contradicts a competitor's claim, stake it CON on their tree.

3. **Same observation, multiple stakes**: An observation like "Lives paycheck to paycheck" could be:
   - CON on SURVIVAL's claim (evidence of financial insecurity)
   - PRO on MEANING's claim (sacrifice for purpose)
   - PRO on AUTONOMY's claim (chose freedom over salary)

4. **Weight by relevance**: 0.0-1.0 how strongly this observation speaks to this claim

## Output Format

Return JSON array of staking decisions:
```json
[
  {{"observation_id": "...", "tree_id": "...", "position": "pro" or "con", "weight": 0.0-1.0, "staking_agent": "meaning", "reasoning": "..."}},
  ...
]
```

An observation can have MULTIPLE stakes (on different trees).
Skip observations that aren't relevant to any claim.

Return ONLY the JSON array."""

        response = self._call_claude(prompt)  # Uses default 300s timeout with retries
        decisions_raw = self._parse_json(response)

        decisions = []
        for d in decisions_raw:
            claim = next((c for c in claims if c.tree.id == d["tree_id"]), None)
            staking_agent = d.get("staking_agent", "meaning").lower()  # Default to meaning if not specified
            decisions.append(StakeDecision(
                observation_id=d["observation_id"],
                tree_id=d["tree_id"],
                position=Position.PRO if d["position"] == "pro" else Position.CON,
                weight=float(d.get("weight", 0.5)),
                staking_agent=staking_agent,
                reasoning=d.get("reasoning", ""),
                is_own_tree=(claim.proposer.value == staking_agent) if claim else False,
            ))

        return decisions

    # =========================================================================
    # Phase 3: Resolution - Determine winners and reallocate
    # =========================================================================

    def resolution_phase(
        self,
        claims: list[Claim],
        agents: AgentSet,
        learning_rate: float = 0.1,
        verbose: bool = True,
    ) -> DebateResult:
        """
        Resolve the debate: compute scores, determine winners, reallocate.
        """
        if verbose:
            print("\n" + "="*60)
            print("PHASE 3: RESOLUTION")
            print("="*60)

        # Compute final scores for each claim
        scores: dict[Tendency, float] = {}
        for claim in claims:
            claim.score = claim.tree.score
            scores[claim.proposer] = claim.score

            if verbose:
                print(f"\n  [{claim.proposer.value.upper()}] score: {claim.score:.3f}")
                print(f"    Nodes: {len(claim.tree.all_nodes())}")
                print(f"    Depth: {claim.tree.depth()}")

        # Determine winner (highest score)
        winner = max(scores, key=scores.get) if scores else None

        if verbose and winner:
            print(f"\n  WINNER: {winner.value.upper()}")

        # Reallocate based on scores
        allocation_changes = self._reallocate(agents, scores, learning_rate)

        if verbose:
            print("\n  Allocation changes:")
            for tendency, change in sorted(allocation_changes.items(), key=lambda x: -x[1]):
                direction = "+" if change > 0 else ""
                print(f"    {tendency.value}: {direction}{change:.1%}")

        return DebateResult(
            claims=claims,
            total_stakes=sum(len(c.tree.all_nodes()) - 1 for c in claims),  # -1 for root
            winner=winner,
            scores=scores,
            allocation_changes=allocation_changes,
        )

    def _reallocate(
        self,
        agents: AgentSet,
        scores: dict[Tendency, float],
        learning_rate: float,
    ) -> dict[Tendency, float]:
        """
        Shift allocations based on debate scores.

        Winners gain allocation, losers lose it.
        """
        if not scores:
            return {}

        # Normalize scores to sum to 1
        total_score = sum(max(0, s) for s in scores.values())  # Only positive scores
        if total_score == 0:
            return {t: 0.0 for t in scores}

        target_allocations = {
            t: max(0, s) / total_score
            for t, s in scores.items()
        }

        # Blend current allocations toward target
        changes = {}
        for tendency in Tendency:
            current = agents.get(tendency).allocation
            target = target_allocations.get(tendency, current)
            change = (target - current) * learning_rate
            changes[tendency] = change
            agents.get(tendency).allocation = current + change

        # Normalize to ensure sum = 1
        agents.normalize()

        return changes

    # =========================================================================
    # Full Debate Flow
    # =========================================================================

    def run_full_debate(
        self,
        observations: ObservationStore,
        agents: AgentSet,
        rounds: int = 1,
        learning_rate: float = 0.1,
        verbose: bool = True,
    ) -> tuple[TreeStore, DebateResult]:
        """
        Run complete debate: propose -> stake -> resolve.

        Args:
            observations: All observations about the person
            agents: Agent set (allocations will be modified)
            rounds: Number of debate rounds (more = more refined)
            learning_rate: How fast allocations shift
            verbose: Print progress

        Returns:
            (TreeStore with all claims, final DebateResult)
        """
        if verbose:
            print("\n" + "#"*60)
            print("# ARENA: ADVERSARIAL DEBATE")
            print("#"*60)
            print(f"\n  Observations: {len(observations)}")
            print(f"  Agents: {agents}")
            print(f"  Rounds: {rounds}")

        final_result = None
        all_claims = []

        for round_num in range(rounds):
            if verbose:
                print(f"\n{'='*60}")
                print(f"ROUND {round_num + 1}")
                print("="*60)

            # Phase 1: Proposals
            claims = self.proposal_phase(observations, agents, verbose)
            all_claims.extend(claims)

            # Phase 2: Staking
            self.staking_phase(observations, claims, agents, verbose)

            # Phase 3: Resolution
            final_result = self.resolution_phase(claims, agents, learning_rate, verbose)

        # Build tree store from all claims
        trees = TreeStore()
        for claim in all_claims:
            trees.add(claim.tree)

        if verbose:
            print("\n" + "#"*60)
            print("# DEBATE COMPLETE")
            print("#"*60)
            print(f"\n  Final allocations:")
            for agent in sorted(agents.all(), key=lambda a: -a.allocation):
                print(f"    {agent.tendency.value}: {agent.allocation:.1%}")

        return trees, final_result

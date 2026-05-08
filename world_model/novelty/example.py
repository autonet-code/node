"""
Example: Novelty computation in action.

Demonstrates computing novelty of concepts relative to an agent's world model.
Uses semantic embeddings for similarity comparison.
"""

from world_model import (
    Observation,
    ObservationStore,
    Tendency,
    AgentSet,
    Position,
    Node,
    Tree,
    TreeStore,
)
from .novelty import compute_novelty
from .embeddings import cached_similarity, preload_cache, relation_fit_score, analyze_relation


def build_sample_world_model():
    """
    Build a sample world model for a curious technologist.

    This represents someone who values:
    - Autonomy and decentralization
    - Technical understanding
    - Meaning through building
    """

    # Create agent with custom allocations
    agents = AgentSet.with_profile({
        Tendency.CURIOSITY: 0.25,
        Tendency.AUTONOMY: 0.22,
        Tendency.MEANING: 0.20,
        Tendency.SURVIVAL: 0.12,
        Tendency.CONNECTION: 0.10,
        Tendency.STATUS: 0.06,
        Tendency.COMFORT: 0.05,
    })

    # Create trees representing core values
    trees = TreeStore()

    # Tree 1: Decentralization
    decentralization = Tree(root_value="Decentralization enables human flourishing")
    root = decentralization.root_node

    # Add supporting observations
    node1 = Node(content="Centralized systems create single points of failure")
    node1.add_stake("autonomy", 0.3)
    node1.add_stake("survival", 0.2)
    root.add_child(node1, Position.PRO)

    node2 = Node(content="Peer-to-peer networks resist censorship")
    node2.add_stake("autonomy", 0.4)
    root.add_child(node2, Position.PRO)

    node3 = Node(content="Coordination is harder without central authority")
    node3.add_stake("connection", 0.2)
    root.add_child(node3, Position.CON)

    trees.add(decentralization)

    # Tree 2: Building as meaning
    building = Tree(root_value="Creating technology is meaningful work")
    root = building.root_node

    node4 = Node(content="Software can scale impact indefinitely")
    node4.add_stake("meaning", 0.4)
    root.add_child(node4, Position.PRO)

    node5 = Node(content="Open source contributes to collective knowledge")
    node5.add_stake("meaning", 0.3)
    node5.add_stake("connection", 0.2)
    root.add_child(node5, Position.PRO)

    trees.add(building)

    # Tree 3: Understanding over following
    understanding = Tree(root_value="Deep understanding beats surface knowledge")
    root = understanding.root_node

    node6 = Node(content="First principles thinking enables novel solutions")
    node6.add_stake("curiosity", 0.5)
    root.add_child(node6, Position.PRO)

    node7 = Node(content="Experts often miss paradigm shifts")
    node7.add_stake("curiosity", 0.2)
    node7.add_stake("autonomy", 0.2)
    root.add_child(node7, Position.PRO)

    trees.add(understanding)

    # Observations store (concepts already integrated)
    observations = ObservationStore()
    observations.add(Observation(content="Bitcoin enables trustless transactions"))
    observations.add(Observation(content="Learning Rust improved my systems thinking"))
    observations.add(Observation(content="Open source projects create community"))

    return trees, agents, observations


def main():
    print("=" * 60)
    print("NOVELTY COMPUTATION EXAMPLE (with NLI stance detection)")
    print("=" * 60)

    # Quick NLI demo
    print("\n--- NLI Stance Detection Demo ---")
    demo_pairs = [
        ("Decentralization enables human flourishing",
         "Ethereum enables decentralized smart contracts"),
        ("Decentralization enables human flourishing",
         "Centralized platforms are more user-friendly than decentralized ones"),
        ("Creating technology is meaningful work",
         "Software development is a waste of time"),
    ]
    for premise, hypothesis in demo_pairs:
        rel = analyze_relation(premise, hypothesis)
        print(f"\n  Premise:    \"{premise[:50]}...\"")
        print(f"  Hypothesis: \"{hypothesis[:50]}...\"")
        print(f"  Topical similarity: {rel.topical_similarity:.2f}")
        print(f"  Stance: {rel.nli_result.stance} (E:{rel.nli_result.entailment:.2f} C:{rel.nli_result.contradiction:.2f} N:{rel.nli_result.neutral:.2f})")
        print(f"  Fit score: {rel.novelty_fit_score:.2f}")

    # Build world model
    print("\n" + "=" * 60)
    trees, agents, observations = build_sample_world_model()

    # Preload embeddings for all node content (faster batch processing)
    all_node_texts = []
    for tree in trees.all():
        for node in tree.all_nodes():
            if node.content:
                all_node_texts.append(node.content)
    preload_cache(all_node_texts)
    print(f"\nPreloaded {len(all_node_texts)} node embeddings.")

    print("\n--- Agent Profile ---")
    for agent in agents.all():
        print(f"  {agent.tendency.value}: {agent.allocation:.0%}")

    print(f"\n--- World Model: {len(trees)} trees ---")
    for tree in trees.all():
        print(f"  - {tree.root_value}")
        print(f"    Nodes: {len(tree.all_nodes())}, Score: {tree.score:.2f}")

    # Test concepts with varying novelty
    test_concepts = [
        # Low novelty - fits existing worldview
        Observation(content="Ethereum enables decentralized smart contracts"),

        # Medium novelty - related but different
        Observation(content="AI systems can enhance human creativity"),

        # High novelty - challenges existing frames
        Observation(content="Centralized platforms are more user-friendly than decentralized ones"),

        # Paradigm shift - would restructure priorities
        Observation(content="Consciousness cannot emerge from computation"),
    ]

    print("\n" + "=" * 60)
    print("NOVELTY SCORES FOR TEST CONCEPTS")
    print("=" * 60)

    for concept in test_concepts:
        print(f"\n--- Concept: \"{concept.content}\" ---")

        score = compute_novelty(
            concept, trees, agents, observations,
            similarity_fn=relation_fit_score  # Now uses NLI for stance detection
        )

        print(f"\n  COMPOSITE SCORE: {score.composite_score:.3f}")
        print(f"\n  Components:")
        print(f"    Integration Resistance: {score.integration_resistance:.3f}")
        print(f"      (How hard to stake in existing trees)")
        print(f"    Contradiction Depth:    {score.contradiction_depth:.3f}")
        print(f"      (How deep conflicts go)")
        print(f"    Tree Coverage Gap:      {score.tree_coverage_gap:.3f}")
        print(f"      (Fraction of trees that can't accommodate)")
        print(f"    Allocation Disruption:  {score.allocation_disruption:.3f}")
        print(f"      (Would shift tendency priorities)")

        print(f"\n  Classification:")
        if score.is_paradigm_shift:
            print(f"    *** PARADIGM SHIFT - would restructure worldview ***")
        elif score.is_deep_novel:
            print(f"    DEEP NOVEL - challenges foundational assumptions")
        elif score.is_surface_novel:
            print(f"    SURFACE NOVEL - new but easily integrated")
        else:
            print(f"    FAMILIAR - fits existing structures")

        print(f"\n  Stake attempts:")
        for attempt in score.stake_attempts:
            fit_status = "fits" if attempt.fit_score > 0.3 else "doesn't fit"
            print(f"    - Tree '{attempt.tree.root_value[:40]}...': {fit_status} ({attempt.fit_score:.2f})")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()

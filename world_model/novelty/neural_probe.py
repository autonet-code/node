"""
Neural-Enhanced Novelty Probe

Combines:
- Wikidata for graph traversal and adjacency
- Neural embeddings for topical similarity
- NLI model for stance detection (entailment/contradiction/neutral)

This should produce CONTRADICTS terminations when concepts genuinely
oppose claims in the reference frame.
"""

from dataclasses import dataclass, field
from typing import Any, List, Tuple, Optional, Set

try:
    from .core import (
        NoveltyProbe,
        ReferenceFrame,
        Focus,
        ParseResult,
        Termination,
        Stance,
        Claim,
        NoveltyResult,
    )
    from .embeddings import (
        semantic_similarity,
        nli_inference,
        NLIResult,
        cached_similarity,
        preload_cache,
    )
    from .wikidata import (
        best_match,
        get_entity,
        get_ancestry,
        WikidataEntity,
    )
except ImportError:
    from core import (
        NoveltyProbe,
        ReferenceFrame,
        Focus,
        ParseResult,
        Termination,
        Stance,
        Claim,
        NoveltyResult,
    )
    from embeddings import (
        semantic_similarity,
        nli_inference,
        NLIResult,
        cached_similarity,
        preload_cache,
    )
    from wikidata import (
        best_match,
        get_entity,
        get_ancestry,
        WikidataEntity,
    )


# =============================================================================
# Neural Reference Frame
# =============================================================================

@dataclass
class NeuralClaim(Claim):
    """A claim with text content for neural analysis."""
    text: str = ""
    description: str = ""
    qid: Optional[str] = None  # Optional Wikidata backing


@dataclass
class NeuralFrame(ReferenceFrame):
    """
    Reference frame that uses neural models for stance detection.

    Claims are textual statements that can be analyzed via NLI.
    Wikidata is used for graph structure when available.
    """
    claims: List[NeuralClaim] = field(default_factory=list)
    integrated_texts: Set[str] = field(default_factory=set)
    _total_stake: float = 0.0

    # Thresholds
    similarity_threshold: float = 0.6
    contradiction_threshold: float = 0.5
    entailment_threshold: float = 0.5

    def contains(self, content: Any) -> Tuple[bool, float]:
        """
        Check if content is already integrated.

        IMPORTANT: We only check for near-exact matches, NOT topical similarity.
        Two statements can be about the same topic but contradict each other.
        Topical similarity is handled by find_claims + NLI.
        """
        text = self._to_text(content)
        if not text:
            return False, 0.0

        # Direct containment (exact or near-exact match only)
        if text in self.integrated_texts:
            return True, 1.0

        # Only very high similarity counts as "already integrated"
        # (essentially the same statement rephrased)
        for integrated in self.integrated_texts:
            sim = cached_similarity(text, integrated)
            if sim > 0.92:  # Very high threshold - near paraphrase
                print(f"    [contains] Near-paraphrase of '{integrated[:30]}...' (sim={sim:.2f})")
                return True, sim

        # Otherwise, NOT integrated - let NLI determine stance
        return False, 0.0

    def find_claims(self, content: Any) -> List[Tuple[Claim, float]]:
        """Find claims related to this content via semantic similarity."""
        text = self._to_text(content)
        if not text:
            return []

        results = []
        for claim in self.claims:
            sim = cached_similarity(text, claim.text)
            if sim > 0.3:  # Minimum relevance threshold
                results.append((claim, sim))

        return sorted(results, key=lambda x: -x[1])

    def detect_stance(self, content: Any, claim: Claim) -> Tuple[Stance, float]:
        """
        Detect stance using NLI model.

        This is the key neural integration - we use a trained NLI model
        to detect if the content contradicts, supports, or is neutral
        to the claim.
        """
        text = self._to_text(content)
        if not text or not isinstance(claim, NeuralClaim):
            return Stance.NEUTRAL, 0.0

        # Run NLI: claim as premise, content as hypothesis
        # "Given the claim, does the content follow/contradict/neither?"
        nli_result = nli_inference(claim.text, text)

        if nli_result.is_contradiction:
            return Stance.CON, nli_result.contradiction
        elif nli_result.is_entailment:
            return Stance.PRO, nli_result.entailment
        else:
            return Stance.NEUTRAL, nli_result.neutral

    def absorb(self, content: Any) -> "NeuralFrame":
        """Create new frame with content integrated."""
        text = self._to_text(content)
        if not text:
            return self

        new_integrated = self.integrated_texts | {text}

        return NeuralFrame(
            claims=self.claims,
            integrated_texts=new_integrated,
            _total_stake=self._total_stake,
            similarity_threshold=self.similarity_threshold,
            contradiction_threshold=self.contradiction_threshold,
            entailment_threshold=self.entailment_threshold,
        )

    def get_adjacent(self, content: Any) -> List[Any]:
        """Get adjacent concepts via Wikidata if available."""
        text = self._to_text(content)
        if not text:
            return []

        # Try to find in Wikidata for graph adjacency
        match = best_match(text)
        if not match:
            return []

        qid = match[0]
        try:
            entity = get_entity(qid)
            ancestry = get_ancestry(qid, max_depth=5)

            # Return ancestor labels as adjacent concepts
            adjacent = []
            for ancestor_qid in ancestry.ancestors[:5]:
                label = ancestry.labels.get(ancestor_qid, ancestor_qid)
                if label not in self.integrated_texts:
                    adjacent.append(label)

            return adjacent
        except:
            return []

    @property
    def total_stake(self) -> float:
        return max(self._total_stake, 0.01)

    def _to_text(self, content: Any) -> Optional[str]:
        """Convert content to text for neural analysis."""
        if isinstance(content, str):
            return content
        elif hasattr(content, 'label'):
            return content.label
        elif hasattr(content, 'text'):
            return content.text
        return None

    @classmethod
    def from_claims(cls, claim_texts: List[str], stakes: List[float] = None) -> "NeuralFrame":
        """
        Build a frame from textual claims.

        Args:
            claim_texts: List of claim statements
            stakes: Optional stake weights (defaults to uniform)
        """
        if stakes is None:
            stakes = [1.0] * len(claim_texts)

        claims = []
        for i, (text, stake) in enumerate(zip(claim_texts, stakes)):
            claims.append(NeuralClaim(
                content=text,
                depth=0,  # All top-level for now
                stake=stake,
                text=text,
            ))

        total_stake = sum(stakes)

        # Preload embeddings for efficiency
        preload_cache(claim_texts)

        return cls(
            claims=claims,
            integrated_texts=set(claim_texts),
            _total_stake=total_stake,
        )


# =============================================================================
# Neural Novelty Probe
# =============================================================================

@dataclass
class NeuralFetchResult:
    """Data fetched for neural analysis."""
    text: str
    description: str = ""
    qid: Optional[str] = None
    adjacent: List[str] = field(default_factory=list)


class NeuralProbe(NoveltyProbe):
    """
    Novelty probe using neural models for stance detection.

    Key difference from WikidataProbe: uses NLI to detect contradiction,
    which should produce CONTRADICTS terminations for genuinely opposing
    concepts.
    """

    def __init__(
        self,
        max_iterations: int = 10,
        similarity_threshold: float = 0.7,
        contradiction_threshold: float = 0.5,
        disruption_threshold: float = 0.5,
    ):
        super().__init__(max_iterations)
        self.similarity_threshold = similarity_threshold
        self.contradiction_threshold = contradiction_threshold
        self.disruption_threshold = disruption_threshold
        self._visited: Set[str] = set()

    def measure(self, content: Any, frame: ReferenceFrame) -> NoveltyResult:
        """Run the novelty loop with visited tracking."""
        self._visited = set()
        return super().measure(content, frame)

    def fetch(self, focus: Focus, frame: ReferenceFrame) -> Optional[NeuralFetchResult]:
        """Fetch data for the current focus."""
        content = focus.content

        if isinstance(content, str):
            text = content
        elif hasattr(content, 'text'):
            text = content.text
        else:
            return None

        # Track visited
        if text in self._visited:
            return None
        self._visited.add(text)

        # Try to get Wikidata info for richer description
        description = ""
        qid = None
        match = best_match(text)
        if match:
            qid, label, description = match

        # Get adjacent via frame
        adjacent = []
        if isinstance(frame, NeuralFrame):
            all_adjacent = frame.get_adjacent(text)
            adjacent = [a for a in all_adjacent if a not in self._visited]

        return NeuralFetchResult(
            text=text,
            description=description or text,
            qid=qid,
            adjacent=adjacent,
        )

    def parse(
        self,
        data: Optional[NeuralFetchResult],
        focus: Focus,
        frame: ReferenceFrame,
    ) -> ParseResult:
        """
        Parse fetched data using neural models.

        Key: uses NLI for stance detection to find contradictions.
        """
        if data is None:
            return ParseResult.terminate(
                Termination.ORTHOGONAL,
                similarity_to_frame=0.0,
            )

        # Check containment
        is_contained, similarity = frame.contains(data.text)
        if is_contained:
            return ParseResult.terminate(
                Termination.INTEGRATED,
                similarity_to_frame=similarity,
            )

        # Find related claims and check stance via NLI
        related_claims = frame.find_claims(data.text)

        deepest_contradiction = float('inf')
        max_stake_affected = 0.0
        found_contradiction = False

        for claim, relevance in related_claims:
            if relevance < 0.4:  # Skip weakly related
                continue

            stance, confidence = frame.detect_stance(data.text, claim)

            print(f"    [NLI] '{data.text[:30]}...' vs '{claim.text[:30]}...'")
            print(f"          stance={stance.value}, confidence={confidence:.2f}")

            if stance == Stance.CON and confidence > self.contradiction_threshold:
                found_contradiction = True
                if claim.depth < deepest_contradiction:
                    deepest_contradiction = claim.depth
                max_stake_affected = max(max_stake_affected, claim.stake * relevance)

        if found_contradiction:
            return ParseResult.terminate(
                Termination.CONTRADICTS_ROOT,
                contradiction_depth=int(deepest_contradiction) if deepest_contradiction != float('inf') else 0,
                stake_affected=max_stake_affected,
                similarity_to_frame=similarity,
            )

        # Check for high-stake disruption (even without contradiction)
        if related_claims:
            max_stake = max(claim.stake * rel for claim, rel in related_claims)
            if max_stake > self.disruption_threshold * frame.total_stake:
                return ParseResult.terminate(
                    Termination.DISRUPTS,
                    stake_affected=max_stake,
                    similarity_to_frame=similarity,
                )

        # No termination - continue to adjacent
        if data.adjacent:
            next_text = data.adjacent[0]
            next_focus = focus.expand_to(next_text, via=data.text)
            return ParseResult.continue_to(
                focus=next_focus,
                similarity_to_frame=similarity,
            )

        # No adjacent = orthogonal
        return ParseResult.terminate(
            Termination.ORTHOGONAL,
            similarity_to_frame=similarity,
        )

    def _get_max_depth(self, frame: ReferenceFrame) -> int:
        if isinstance(frame, NeuralFrame) and frame.claims:
            return max(c.depth for c in frame.claims)
        return 10


# =============================================================================
# Convenience Functions
# =============================================================================

def measure_neural_novelty(
    concept: str,
    frame_claims: List[str],
    stakes: List[float] = None,
    max_iterations: int = 10,
    verbose: bool = True,
) -> NoveltyResult:
    """
    Measure novelty using neural stance detection.

    Args:
        concept: The concept to measure
        frame_claims: List of claim statements forming the reference frame
        stakes: Optional stake weights for claims
        max_iterations: Max probe iterations
        verbose: Print progress

    Returns:
        NoveltyResult with termination reason and component scores
    """
    if verbose:
        print(f"Building neural frame from {len(frame_claims)} claims...")

    frame = NeuralFrame.from_claims(frame_claims, stakes)

    if verbose:
        print(f"Measuring novelty of: '{concept}'")
        print("-" * 50)

    probe = NeuralProbe(max_iterations=max_iterations)
    result = probe.measure(concept, frame)

    if verbose:
        print(f"\nTermination: {result.termination.value}")
        print(f"Iterations: {result.iterations}")
        print(f"Path: {' -> '.join(result.path) if result.path else '(direct)'}")
        print(f"\nComponents:")
        print(f"  integration_resistance: {result.integration_resistance:.3f}")
        print(f"  contradiction_depth: {result.contradiction_depth:.3f}")
        print(f"  coverage_gap: {result.coverage_gap:.3f}")
        print(f"  allocation_disruption: {result.allocation_disruption:.3f}")
        print(f"\nComposite: {result.composite:.3f}")

    return result


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TEST: Neural Probe with Adversarial Claims")
    print("=" * 60)

    # A frame with clear positions that can be contradicted
    frame_claims = [
        "Centralized systems are more efficient than decentralized ones",
        "Trust in institutions is necessary for economic transactions",
        "Government regulation protects consumers",
        "Traditional banking provides security and stability",
    ]

    # Test concepts that should trigger different terminations
    test_concepts = [
        # Should CONTRADICT (opposes frame claims)
        "Bitcoin enables trustless peer-to-peer transactions without intermediaries",

        # Should INTEGRATE (aligns with frame)
        "Banks provide essential services for the economy",

        # Should be ORTHOGONAL (unrelated)
        "Photosynthesis converts sunlight into chemical energy",

        # Might CONTRADICT or DISRUPT
        "Decentralization eliminates single points of failure",
    ]

    for concept in test_concepts:
        print("\n" + "=" * 60)
        print(f"Concept: {concept[:50]}...")
        print("=" * 60)
        result = measure_neural_novelty(concept, frame_claims)
        print()

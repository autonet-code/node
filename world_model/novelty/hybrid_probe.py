"""
Hybrid Novelty Probe: Wikidata Graph + Neural Semantics

Combines the strengths of both approaches:
- Wikidata provides: graph structure, adjacency, entity labels, descriptions
- Neural models provide: semantic similarity, stance detection via NLI

The key insight: Wikidata gives us the STRUCTURE (what's connected to what),
but the MEANING of those connections requires neural analysis of the text.

Example:
- "Bitcoin" connects to "currency" in Wikidata (graph structure)
- Whether Bitcoin SUPPORTS or CONTRADICTS claims about currency requires
  parsing the descriptions through NLI
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
        cached_similarity,
        nli_inference,
        NLIResult,
        preload_cache,
    )
    from .wikidata import (
        best_match,
        get_entity,
        get_ancestry,
        get_graph_metrics,
        WikidataEntity,
        AncestryPath,
        GraphMetrics,
    )
except ImportError:
    # Running as script, not package
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
        cached_similarity,
        nli_inference,
        NLIResult,
        preload_cache,
    )
    from wikidata import (
        best_match,
        get_entity,
        get_ancestry,
        get_graph_metrics,
        WikidataEntity,
        AncestryPath,
        GraphMetrics,
    )


# =============================================================================
# Hybrid Reference Frame
# =============================================================================

@dataclass
class HybridClaim(Claim):
    """
    A claim backed by both Wikidata structure and textual content.

    The text is what gets parsed through neural models.
    The qid/ancestry provides graph structure.
    """
    qid: Optional[str] = None
    label: str = ""
    description: str = ""
    ancestry: List[str] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """Combined label + description for neural analysis."""
        if self.description:
            return f"{self.label}: {self.description}"
        return self.label


@dataclass
class HybridFrame(ReferenceFrame):
    """
    Reference frame that combines Wikidata graph with neural semantics.

    - Graph structure (Q-IDs, ancestry) for finding related concepts
    - Neural models for determining semantic stance
    """
    claims: List[HybridClaim] = field(default_factory=list)
    integrated_qids: Set[str] = field(default_factory=set)
    integrated_texts: Set[str] = field(default_factory=set)
    _total_stake: float = 0.0

    # Thresholds
    paraphrase_threshold: float = 0.92  # Near-exact match for containment
    relevance_threshold: float = 0.4    # Minimum similarity to consider related
    contradiction_threshold: float = 0.5  # NLI confidence for contradiction

    def contains(self, content: Any) -> Tuple[bool, float]:
        """
        Check if content is already integrated.

        Two ways to be contained:
        1. Same Q-ID (exact Wikidata concept)
        2. Near-paraphrase of existing text (very high similarity)

        We do NOT trigger containment on mere topical similarity -
        that's handled by find_claims + detect_stance.
        """
        qid, text = self._extract(content)

        # Q-ID match = definitely contained
        if qid and qid in self.integrated_qids:
            return True, 1.0

        # Check for near-paraphrase
        if text:
            if text in self.integrated_texts:
                return True, 1.0

            for integrated in self.integrated_texts:
                sim = cached_similarity(text, integrated)
                if sim > self.paraphrase_threshold:
                    return True, sim

        return False, 0.0

    def find_claims(self, content: Any) -> List[Tuple[Claim, float]]:
        """
        Find claims related to this content.

        Uses BOTH:
        - Semantic similarity (neural) on text
        - Graph overlap (Wikidata) on Q-IDs/ancestry
        """
        qid, text = self._extract(content)
        if not text and not qid:
            return []

        results = []

        # Get ancestry if we have a Q-ID
        content_ancestry = set()
        if qid:
            try:
                ancestry = get_ancestry(qid)
                content_ancestry = set(ancestry.ancestors)
            except:
                pass

        for claim in self.claims:
            relevance = 0.0

            # Text-based relevance (neural)
            if text and claim.full_text:
                sim = cached_similarity(text, claim.full_text)
                relevance = max(relevance, sim)

            # Graph-based relevance (Wikidata)
            if qid and claim.qid:
                # Direct Q-ID match
                if qid == claim.qid:
                    relevance = max(relevance, 1.0)
                # Shared ancestry
                elif content_ancestry and claim.ancestry:
                    shared = content_ancestry & set(claim.ancestry)
                    if shared:
                        # Closer shared ancestor = more relevant
                        min_distance = min(
                            claim.ancestry.index(s) if s in claim.ancestry else 100
                            for s in shared
                        )
                        graph_relevance = 1.0 - (min_distance / 10)
                        relevance = max(relevance, graph_relevance)

            if relevance > 0.5:  # Must be meaningfully related to compare
                results.append((claim, relevance))

        return sorted(results, key=lambda x: -x[1])

    def detect_stance(self, content: Any, claim: Claim) -> Tuple[Stance, float]:
        """
        Detect stance using NLI on Wikidata textual content.

        This is where neural models parse the Wikipedia data:
        - claim.full_text = Wikidata label + description
        - content = concept text (also potentially from Wikidata)

        IMPORTANT: We first check topical similarity. NLI models can get confused
        on completely unrelated domains (e.g., "photosynthesis" vs "banking").
        If topics aren't related, they can't meaningfully contradict each other.
        """
        _, text = self._extract(content)

        if not text or not isinstance(claim, HybridClaim):
            return Stance.NEUTRAL, 0.0

        claim_text = claim.full_text
        if not claim_text:
            return Stance.NEUTRAL, 0.0

        # First: check topical similarity
        # If topics are completely unrelated, return NEUTRAL without NLI
        # NLI models can get confused on unrelated domains
        topical_sim = cached_similarity(text, claim_text)
        if topical_sim < 0.55:  # Below this = different domains
            # Can't contradict what's not even related
            # print(f"      [skip NLI] low topical sim: {topical_sim:.2f}")
            return Stance.NEUTRAL, 0.3

        # Run NLI: claim as premise, content as hypothesis
        nli_result = nli_inference(claim_text, text)

        if nli_result.is_contradiction:
            return Stance.CON, nli_result.contradiction
        elif nli_result.is_entailment:
            return Stance.PRO, nli_result.entailment
        else:
            return Stance.NEUTRAL, nli_result.neutral

    def absorb(self, content: Any) -> "HybridFrame":
        """Create new frame with content integrated."""
        qid, text = self._extract(content)

        new_integrated_qids = self.integrated_qids.copy()
        new_integrated_texts = self.integrated_texts.copy()

        if qid:
            new_integrated_qids.add(qid)
        if text:
            new_integrated_texts.add(text)

        return HybridFrame(
            claims=self.claims,
            integrated_qids=new_integrated_qids,
            integrated_texts=new_integrated_texts,
            _total_stake=self._total_stake,
            paraphrase_threshold=self.paraphrase_threshold,
            relevance_threshold=self.relevance_threshold,
            contradiction_threshold=self.contradiction_threshold,
        )

    def get_adjacent(self, content: Any) -> List[Any]:
        """
        Get adjacent concepts via Wikidata graph.

        Returns entity labels (text) for neural processing.
        """
        qid, text = self._extract(content)

        # If no Q-ID, try to find one
        if not qid and text:
            match = best_match(text)
            if match:
                qid = match[0]

        if not qid:
            return []

        try:
            entity = get_entity(qid)
            ancestry = get_ancestry(qid, max_depth=5)

            # Collect adjacent Q-IDs from graph structure
            adjacent_qids = set()
            adjacent_qids.update(entity.subclass_of[:3])
            adjacent_qids.update(entity.instance_of[:3])
            adjacent_qids.update(entity.part_of[:2])
            adjacent_qids.update(ancestry.ancestors[:3])

            # Remove already integrated
            adjacent_qids -= self.integrated_qids

            # Convert to labels for neural processing
            adjacent_texts = []
            for adj_qid in list(adjacent_qids)[:5]:
                try:
                    adj_entity = get_entity(adj_qid)
                    # Return full description for richer semantic analysis
                    if adj_entity.description:
                        adjacent_texts.append(f"{adj_entity.label}: {adj_entity.description}")
                    else:
                        adjacent_texts.append(adj_entity.label)
                except:
                    pass

            return adjacent_texts

        except:
            return []

    @property
    def total_stake(self) -> float:
        return max(self._total_stake, 0.01)

    def _extract(self, content: Any) -> Tuple[Optional[str], Optional[str]]:
        """Extract Q-ID and text from content."""
        qid = None
        text = None

        if isinstance(content, str):
            if content.startswith('Q') and content[1:].isdigit():
                qid = content
                # Get label for text
                try:
                    entity = get_entity(qid)
                    text = f"{entity.label}: {entity.description}" if entity.description else entity.label
                except:
                    text = content
            else:
                text = content
                # Try to find Q-ID
                match = best_match(content)
                if match:
                    qid = match[0]

        elif hasattr(content, 'qid'):
            qid = content.qid
            if hasattr(content, 'label'):
                text = content.label
                if hasattr(content, 'description') and content.description:
                    text = f"{content.label}: {content.description}"

        elif hasattr(content, 'text'):
            text = content.text

        return qid, text

    @classmethod
    def from_concepts(cls, concepts: List[str], stakes: List[float] = None) -> "HybridFrame":
        """
        Build a hybrid frame from concept texts.

        Each concept is looked up in Wikidata to get:
        - Q-ID and graph structure
        - Label and description for neural analysis
        """
        if stakes is None:
            stakes = [1.0] * len(concepts)

        claims = []
        integrated_qids = set()
        integrated_texts = set()
        total_stake = 0.0
        all_texts = []

        for concept, stake in zip(concepts, stakes):
            match = best_match(concept)

            if match:
                qid, label, description = match

                try:
                    entity = get_entity(qid)
                    ancestry = get_ancestry(qid)
                    metrics = get_graph_metrics(qid)

                    full_text = f"{label}: {description}" if description else label

                    claim = HybridClaim(
                        content=concept,
                        depth=min(ancestry.depth, 10),
                        stake=stake * metrics.integration_score,  # Weight by establishment
                        qid=qid,
                        label=label,
                        description=description,
                        ancestry=ancestry.ancestors,
                    )
                    claims.append(claim)
                    integrated_qids.add(qid)
                    integrated_texts.add(full_text)
                    all_texts.append(full_text)
                    total_stake += claim.stake

                except Exception as e:
                    print(f"  Warning: Could not fetch Wikidata for '{concept}': {e}")
                    # Still add as text-only claim
                    claim = HybridClaim(
                        content=concept,
                        depth=5,
                        stake=stake,
                        label=concept,
                        description="",
                    )
                    claims.append(claim)
                    integrated_texts.add(concept)
                    all_texts.append(concept)
                    total_stake += stake
            else:
                print(f"  Warning: No Wikidata match for '{concept}'")
                # Text-only claim
                claim = HybridClaim(
                    content=concept,
                    depth=5,
                    stake=stake,
                    label=concept,
                    description="",
                )
                claims.append(claim)
                integrated_texts.add(concept)
                all_texts.append(concept)
                total_stake += stake

        # Preload embeddings for efficiency
        preload_cache(all_texts)

        return cls(
            claims=claims,
            integrated_qids=integrated_qids,
            integrated_texts=integrated_texts,
            _total_stake=total_stake,
        )

    @classmethod
    def from_claim_texts(cls, claim_texts: List[str], stakes: List[float] = None) -> "HybridFrame":
        """
        Build a frame from textual claims WITHOUT Wikidata lookup.

        Use this when you want precise control over claim wording,
        while still using Wikidata for the concepts being tested.

        The hybrid approach:
        - Frame claims: use exact text you provide (for NLI)
        - Test concepts: looked up in Wikidata (for graph traversal)
        """
        if stakes is None:
            stakes = [1.0] * len(claim_texts)

        claims = []
        integrated_texts = set()
        total_stake = 0.0

        for i, (text, stake) in enumerate(zip(claim_texts, stakes)):
            claim = HybridClaim(
                content=text,
                depth=i,  # Arbitrary depth ordering
                stake=stake,
                qid=None,
                label=text,
                description="",
            )
            claims.append(claim)
            integrated_texts.add(text)
            total_stake += stake

        # Preload embeddings
        preload_cache(claim_texts)

        return cls(
            claims=claims,
            integrated_qids=set(),
            integrated_texts=integrated_texts,
            _total_stake=total_stake,
        )


# =============================================================================
# Hybrid Novelty Probe
# =============================================================================

@dataclass
class HybridFetchResult:
    """Data fetched for hybrid analysis."""
    qid: Optional[str]
    label: str
    description: str
    full_text: str
    entity: Optional[WikidataEntity]
    ancestry: Optional[AncestryPath]
    adjacent: List[str]  # Text labels of adjacent concepts


class HybridProbe(NoveltyProbe):
    """
    Novelty probe combining Wikidata graph with neural semantics.

    The loop:
    1. Fetch: Get Wikidata entity with labels/descriptions
    2. Parse: Use NLI on those texts to detect stance against claims
    3. Expand: Use Wikidata graph for adjacency

    This parses Wikipedia content through neural embeddings.
    """

    def __init__(
        self,
        max_iterations: int = 10,
        contradiction_threshold: float = 0.5,
        disruption_threshold: float = 0.5,
    ):
        super().__init__(max_iterations)
        self.contradiction_threshold = contradiction_threshold
        self.disruption_threshold = disruption_threshold
        self._visited: Set[str] = set()  # Track visited texts
        self._visited_qids: Set[str] = set()  # Track visited Q-IDs

    def measure(self, content: Any, frame: ReferenceFrame) -> NoveltyResult:
        """Run the novelty loop with visited tracking."""
        self._visited = set()
        self._visited_qids = set()
        return super().measure(content, frame)

    def fetch(self, focus: Focus, frame: ReferenceFrame) -> Optional[HybridFetchResult]:
        """
        Fetch data for the current focus.

        Tries to get Wikidata entity, falls back to text-only.
        Returns structured data including text for neural analysis.
        """
        content = focus.content

        # Extract text
        if isinstance(content, str):
            text = content
        elif hasattr(content, 'text'):
            text = content.text
        elif hasattr(content, 'label'):
            text = content.label
        else:
            return None

        # Check if already visited
        if text in self._visited:
            return None
        self._visited.add(text)

        # Try to get Wikidata info
        qid = None
        label = text
        description = ""
        entity = None
        ancestry = None
        adjacent = []

        match = best_match(text)
        if match:
            qid, label, description = match

            if qid in self._visited_qids:
                return None
            self._visited_qids.add(qid)

            try:
                entity = get_entity(qid)
                ancestry = get_ancestry(qid, max_depth=5)

                # Get adjacent from frame
                if isinstance(frame, HybridFrame):
                    all_adjacent = frame.get_adjacent(qid)
                    adjacent = [a for a in all_adjacent
                               if a not in self._visited]
            except Exception as e:
                print(f"  Fetch warning: {e}")

        full_text = f"{label}: {description}" if description else label

        return HybridFetchResult(
            qid=qid,
            label=label,
            description=description,
            full_text=full_text,
            entity=entity,
            ancestry=ancestry,
            adjacent=adjacent,
        )

    def parse(
        self,
        data: Optional[HybridFetchResult],
        focus: Focus,
        frame: ReferenceFrame,
    ) -> ParseResult:
        """
        Parse fetched data using neural models on Wikidata text.

        This is where Wikipedia content meets NLI:
        - data.full_text contains Wikidata label + description
        - Claims have their own Wikidata text
        - NLI determines if they contradict
        """
        if data is None:
            return ParseResult.terminate(
                Termination.ORTHOGONAL,
                similarity_to_frame=0.0,
            )

        # Check containment
        is_contained, similarity = frame.contains(data.full_text)
        if is_contained:
            return ParseResult.terminate(
                Termination.INTEGRATED,
                similarity_to_frame=similarity,
            )

        # Find related claims and check stance via NLI
        related_claims = frame.find_claims(data.full_text)

        deepest_contradiction = float('inf')
        max_stake_affected = 0.0
        found_contradiction = False

        for claim, relevance in related_claims:
            if relevance < 0.4:
                continue

            stance, confidence = frame.detect_stance(data.full_text, claim)

            if isinstance(claim, HybridClaim):
                claim_preview = claim.label[:30]
            else:
                claim_preview = str(claim.content)[:30]

            print(f"    [NLI] '{data.label[:30]}' vs '{claim_preview}'")
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

        # Check for high-stake disruption
        if related_claims:
            max_stake = max(claim.stake * rel for claim, rel in related_claims)
            if max_stake > self.disruption_threshold * frame.total_stake:
                return ParseResult.terminate(
                    Termination.DISRUPTS,
                    stake_affected=max_stake,
                    similarity_to_frame=similarity,
                )

        # Continue to adjacent (using Wikidata graph)
        if data.adjacent:
            next_text = data.adjacent[0]
            next_focus = focus.expand_to(next_text, via=data.label)
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
        if isinstance(frame, HybridFrame) and frame.claims:
            return max(c.depth for c in frame.claims)
        return 10


# =============================================================================
# Convenience Functions
# =============================================================================

def measure_hybrid_novelty(
    concept: str,
    frame_concepts: List[str],
    stakes: List[float] = None,
    max_iterations: int = 10,
    verbose: bool = True,
) -> NoveltyResult:
    """
    Measure novelty using hybrid Wikidata+Neural approach.

    Args:
        concept: The concept to measure
        frame_concepts: List of concepts forming the reference frame
        stakes: Optional stake weights for claims
        max_iterations: Max probe iterations
        verbose: Print progress

    Returns:
        NoveltyResult with termination reason and component scores
    """
    if verbose:
        print(f"Building hybrid frame from {len(frame_concepts)} concepts...")
        print("(Fetching Wikidata + preloading embeddings)")

    frame = HybridFrame.from_concepts(frame_concepts, stakes)

    if verbose:
        print(f"\nFrame built:")
        print(f"  - {len(frame.integrated_qids)} Q-IDs")
        print(f"  - {len(frame.claims)} claims with text")
        print(f"  - Total stake: {frame.total_stake:.2f}")
        print(f"\nMeasuring novelty of: '{concept}'")
        print("-" * 50)

    probe = HybridProbe(max_iterations=max_iterations)
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


def measure_against_claims(
    concept: str,
    claim_texts: List[str],
    stakes: List[float] = None,
    max_iterations: int = 10,
    verbose: bool = True,
) -> NoveltyResult:
    """
    Measure novelty of a concept against textual claims.

    This is the HYBRID approach:
    - Frame: pure textual claims (for precise NLI stance detection)
    - Concept: looked up in Wikidata (for description + graph expansion)

    This parses Wikipedia content (the concept's Wikidata description)
    through neural NLI to compare against your claim statements.

    Args:
        concept: The concept to measure (will be looked up in Wikidata)
        claim_texts: List of claim statements forming the reference frame
        stakes: Optional stake weights
        max_iterations: Max probe iterations
        verbose: Print progress

    Returns:
        NoveltyResult
    """
    if verbose:
        print(f"Building frame from {len(claim_texts)} textual claims...")

    frame = HybridFrame.from_claim_texts(claim_texts, stakes)

    if verbose:
        print(f"Frame built: {len(frame.claims)} claims")
        print(f"\nMeasuring novelty of: '{concept}'")
        print("(Will fetch Wikidata description for neural comparison)")
        print("-" * 50)

    probe = HybridProbe(max_iterations=max_iterations)
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
    print("=" * 70)
    print("HYBRID PROBE: Wikidata Graph + Neural Semantics")
    print("=" * 70)
    print()
    print("This probe parses Wikipedia/Wikidata content through neural models:")
    print("1. Frame claims: textual statements (what the agent believes)")
    print("2. Test concept: looked up in Wikidata (gets description)")
    print("3. NLI compares: does Wikidata description support/contradict claims?")
    print("4. Expansion: uses Wikidata graph to explore adjacent concepts")
    print()

    # Frame: Textual claims about traditional finance
    # These are the agent's beliefs - stated clearly for NLI
    frame_claims = [
        "Centralized systems are more efficient than decentralized ones",
        "Trust in institutions is necessary for economic transactions",
        "Government regulation protects consumers in financial markets",
        "Traditional banking provides security and stability",
    ]

    # Test concepts - will be looked up in Wikidata
    # Their Wikidata descriptions will be parsed through NLI
    test_concepts = [
        # Should CONTRADICT - trustless, decentralized
        ("Bitcoin", "decentralized cryptocurrency, challenges trust-based systems"),

        # Should INTEGRATE - aligns with frame
        ("central bank", "institutional banking, aligned with trust/regulation"),

        # Should be ORTHOGONAL - unrelated to finance
        ("photosynthesis", "biological process, unrelated to frame"),

        # Might CONTRADICT
        ("blockchain", "decentralized ledger, may contradict centralization claims"),
    ]

    print("Reference frame (agent's beliefs):")
    for i, claim in enumerate(frame_claims, 1):
        print(f"  {i}. {claim}")
    print()
    print("=" * 70)

    for concept, expected in test_concepts:
        print(f"\n{'=' * 70}")
        print(f"Testing: {concept}")
        print(f"Expected: {expected}")
        print("=" * 70)

        result = measure_against_claims(concept, frame_claims)

        print(f"\n>>> Result: {result.termination.value} (composite={result.composite:.3f})")

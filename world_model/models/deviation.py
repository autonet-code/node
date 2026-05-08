from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid

from .evidence import EvidencePointer


class DeviationType(Enum):
    """High-level categories of deviation from baseline human"""
    EPISTEMIC = "epistemic"          # How beliefs form, what's trusted
    MOTIVATIONAL = "motivational"    # What drives action, goals, priorities
    RELATIONAL = "relational"        # How others are modeled, trust, attachment
    ATTENTIONAL = "attentional"      # What gets noticed, salience patterns
    AXIOLOGICAL = "axiological"      # Values, what matters, meaning
    SELF_MODEL = "self_model"        # Identity, agency, narrative
    BEHAVIORAL = "behavioral"        # Action patterns, habits, coping
    COGNITIVE = "cognitive"          # Reasoning style, problem-solving


class EdgeType(Enum):
    """Types of relationships between deviation nodes"""
    SUPPORTS = "supports"        # This deviation supports/enables another
    CONFLICTS = "conflicts"      # This deviation is in tension with another
    TRIGGERS = "triggers"        # This deviation activates a behavioral pattern
    CONTEXTUALIZES = "contextualizes"  # This deviation explains when another applies
    CAUSES = "causes"            # This deviation causally leads to another
    REFINES = "refines"          # This is a more specific version of another


@dataclass
class Edge:
    """A relationship between two deviation nodes"""
    target_id: str
    edge_type: EdgeType
    strength: float = 1.0  # How strong is this relationship
    description: str = ""  # Natural language explanation
    bidirectional: bool = False


@dataclass
class DeviationNode:
    """
    A single deviation from baseline human expectations.

    The core unit of the personal world model. Represents one way
    this person differs from what you'd expect of a "typical" person.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # The deviation itself
    dimension: str = ""           # Emergent label, e.g., "institutional trust"
    deviation_type: DeviationType = DeviationType.EPISTEMIC

    # Baseline comparison (key insight: store what we're comparing against)
    baseline_assumption: str = ""  # What would typical person think/do/feel
    deviation_description: str = ""  # How this person differs

    # Quantification
    magnitude: float = 0.5        # How far from baseline (0-1)
    confidence: float = 0.5       # How certain based on evidence (0-1)
    stability: float = 0.5        # How consistent across contexts (0-1)

    # Fractal structure
    parent_id: Optional[str] = None
    children_ids: list[str] = field(default_factory=list)

    # Evidence grounding
    evidence: list[EvidencePointer] = field(default_factory=list)
    contradictions: list[EvidencePointer] = field(default_factory=list)

    # Graph relationships
    edges: list[Edge] = field(default_factory=list)

    # Temporal tracking
    first_observed: datetime = field(default_factory=datetime.now)
    last_reinforced: datetime = field(default_factory=datetime.now)
    observation_count: int = 1

    # Metadata
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    def reinforce(self, evidence: EvidencePointer, confidence_boost: float = 0.1):
        """Called when new evidence supports this deviation"""
        self.evidence.append(evidence)
        self.last_reinforced = datetime.now()
        self.observation_count += 1
        self.confidence = min(1.0, self.confidence + confidence_boost)
        # Stability increases with repeated observations
        self.stability = min(1.0, self.stability + 0.05)

    def contradict(self, evidence: EvidencePointer, confidence_penalty: float = 0.1):
        """Called when evidence contradicts this deviation"""
        self.contradictions.append(evidence)
        # Don't reduce confidence below a floor - contradictions might be contextual
        self.confidence = max(0.2, self.confidence - confidence_penalty)
        # Stability decreases with contradictions
        self.stability = max(0.1, self.stability - 0.1)

    def add_child(self, child_id: str):
        """Add a more specific sub-deviation"""
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)

    def add_edge(self, edge: Edge):
        """Add a relationship to another deviation"""
        self.edges.append(edge)

    def similarity_key(self) -> str:
        """Key for finding similar/duplicate deviations"""
        return f"{self.deviation_type.value}:{self.dimension.lower()}"

    def __repr__(self):
        return f"Deviation({self.dimension}: {self.deviation_description[:50]}...)"

    def to_summary(self, depth: int = 0) -> str:
        """Human-readable summary at specified depth"""
        indent = "  " * depth
        summary = f"{indent}[{self.deviation_type.value}] {self.dimension}\n"
        summary += f"{indent}  Baseline: {self.baseline_assumption[:100]}...\n"
        summary += f"{indent}  Deviation: {self.deviation_description[:100]}...\n"
        summary += f"{indent}  (magnitude={self.magnitude:.2f}, confidence={self.confidence:.2f}, stability={self.stability:.2f})\n"
        return summary

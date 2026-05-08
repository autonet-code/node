"""
Wikidata Integration for Novelty Measurement

Provides direct access to Wikipedia's knowledge graph via Wikidata APIs.
No caching, no local storage - just on-demand graph queries.

The key insight: Wikidata provides PRE-COMPUTED graph metrics that map
directly to our novelty components:

- Incoming references -> Integration resistance (inverse)
- Hierarchy depth (P279 chain) -> Contradiction depth context
- Sitelinks + properties -> Coverage gap (inverse)
- Centrality ratio -> Allocation disruption potential

Two API endpoints used:
1. wbgetentities - Get entity properties, labels, sitelinks
2. SPARQL endpoint - Get graph metrics (incoming/outgoing counts, ancestry)
"""

import urllib.request
import urllib.parse
import json
import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


# =============================================================================
# Data Types
# =============================================================================

@dataclass
class WikidataEntity:
    """A Wikidata entity with its core properties."""
    qid: str
    label: str
    description: str = ""

    # Graph structure
    instance_of: List[str] = field(default_factory=list)      # P31
    subclass_of: List[str] = field(default_factory=list)      # P279
    part_of: List[str] = field(default_factory=list)          # P361
    has_parts: List[str] = field(default_factory=list)        # P527
    uses: List[str] = field(default_factory=list)             # P2283
    facet_of: List[str] = field(default_factory=list)         # P1269

    # Counts (raw metrics)
    sitelinks_count: int = 0
    properties_count: int = 0


@dataclass
class GraphMetrics:
    """Pre-computed graph metrics from Wikidata."""
    qid: str

    # Direct counts
    incoming_refs: int = 0      # How many entities reference this
    outgoing_refs: int = 0      # How many entities this references
    sitelinks: int = 0          # Language editions (global reach)
    properties: int = 0         # Semantic richness

    # Hierarchy
    depth: int = 0              # Steps up P279 chain to root
    subclass_count: int = 0     # Direct taxonomic children

    # Derived metrics
    @property
    def centrality_ratio(self) -> float:
        """Incoming/outgoing ratio. >1 = established, <1 = emerging."""
        if self.outgoing_refs == 0:
            return float(self.incoming_refs) if self.incoming_refs > 0 else 1.0
        return self.incoming_refs / self.outgoing_refs

    @property
    def integration_score(self) -> float:
        """How well-integrated is this concept? (0-1, higher = more integrated)"""
        # Log scale - 841 refs -> ~0.93, 10 refs -> ~0.77, 1 ref -> ~0.5
        if self.incoming_refs == 0:
            return 0.0
        return 1.0 - (1.0 / math.log(self.incoming_refs + math.e))

    @property
    def coverage_score(self) -> float:
        """How broadly covered is this concept? (0-1, higher = broader)"""
        # Sitelinks: 167 -> ~0.9, 50 -> ~0.76, 10 -> ~0.59
        # Properties: 76 -> bonus
        sitelink_score = math.log(self.sitelinks + 1) / math.log(300)  # ~300 max languages
        property_score = min(self.properties / 100, 1.0)  # Normalize to 100 properties
        return min((sitelink_score + property_score * 0.3) / 1.3, 1.0)

    @property
    def establishment_score(self) -> float:
        """How established/foundational is this concept? (0-1)"""
        # Based on centrality ratio and depth
        ratio_score = min(math.log(self.centrality_ratio + 1) / 3, 1.0)
        depth_score = 1.0 - min(self.depth / 20, 1.0)  # Shallower = more foundational
        return (ratio_score + depth_score) / 2


@dataclass
class AncestryPath:
    """Path up the P279 (subclass of) hierarchy."""
    qid: str
    ancestors: List[str] = field(default_factory=list)  # Ordered from immediate parent to root
    labels: Dict[str, str] = field(default_factory=dict)  # QID -> label mapping

    @property
    def depth(self) -> int:
        return len(self.ancestors)

    def shared_ancestor(self, other: "AncestryPath") -> Optional[str]:
        """Find the nearest common ancestor with another path."""
        other_set = set(other.ancestors)
        for ancestor in self.ancestors:
            if ancestor in other_set:
                return ancestor
        return None

    def distance_to(self, other: "AncestryPath") -> int:
        """Semantic distance via shared ancestry. -1 if no common ancestor."""
        shared = self.shared_ancestor(other)
        if shared is None:
            return -1
        idx_self = self.ancestors.index(shared)
        idx_other = other.ancestors.index(shared)
        return idx_self + idx_other


# =============================================================================
# API Functions
# =============================================================================

USER_AGENT = "NoveltyBot/1.0 (https://github.com/novelty-research; research project)"


def _api_request(url: str, params: dict = None) -> dict:
    """Make a request to Wikidata API."""
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"

    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as e:
        raise WikidataError(f"HTTP {e.code}: {e.reason}")
    except urllib.error.URLError as e:
        raise WikidataError(f"URL Error: {e.reason}")
    except json.JSONDecodeError as e:
        raise WikidataError(f"JSON decode error: {e}")


def _sparql_query(query: str) -> dict:
    """Execute a SPARQL query against Wikidata."""
    url = "https://query.wikidata.org/sparql"
    params = {'query': query, 'format': 'json'}
    return _api_request(url, params)


class WikidataError(Exception):
    """Error from Wikidata API."""
    pass


# =============================================================================
# Core Functions
# =============================================================================

def search_concept(text: str, limit: int = 5) -> List[tuple[str, str, str]]:
    """
    Search Wikidata for a concept by text.

    Args:
        text: The concept to search for
        limit: Maximum results to return

    Returns:
        List of (qid, label, description) tuples
    """
    url = "https://www.wikidata.org/w/api.php"
    params = {
        'action': 'wbsearchentities',
        'search': text,
        'language': 'en',
        'limit': limit,
        'format': 'json',
    }

    data = _api_request(url, params)

    results = []
    for item in data.get('search', []):
        qid = item.get('id', '')
        label = item.get('label', '')
        description = item.get('description', '')
        results.append((qid, label, description))

    return results


def best_match(text: str) -> Optional[tuple[str, str, str]]:
    """
    Find the best Wikidata match for a concept.

    Uses heuristics to pick the most relevant result:
    - Prefers exact label matches
    - Prefers results where description doesn't indicate it's a company/person/place
    - Falls back to simplified searches if no results

    Args:
        text: The concept to search for

    Returns:
        (qid, label, description) or None if no results
    """
    # Try original text first
    results = search_concept(text, limit=10)

    # If no results, try simplifications
    if not results:
        # Remove common prefixes
        simplifications = [
            text.replace("Invention of ", "").replace("invention of ", ""),
            text.replace("Development of ", "").replace("development of ", ""),
            text.replace("Evolution of ", "").replace("evolution of ", ""),
            text.replace("Emergence of ", "").replace("emergence of ", ""),
            text.replace("First ", "").replace("first ", ""),
            text.replace("Earliest ", "").replace("earliest ", ""),
        ]
        # Also try removing parenthetical notes
        if "(" in text:
            simplifications.append(text.split("(")[0].strip())

        # Also try extracting key nouns
        words = text.split()
        if len(words) > 2:
            # Try last 2-3 words (often the core concept)
            simplifications.append(" ".join(words[-2:]))
            simplifications.append(" ".join(words[-3:]))

        for simp in simplifications:
            if simp and simp != text:
                results = search_concept(simp, limit=10)
                if results:
                    break

    if not results:
        return None

    text_lower = text.lower()

    # First pass: exact label match
    for qid, label, desc in results:
        if label.lower() == text_lower:
            return (qid, label, desc)

    # Second pass: avoid companies, people, places, academic papers
    avoid_keywords = ['company', 'person', 'village', 'city', 'district', 'county',
                      'album', 'film', 'song', 'scientific article', 'scholarly article']
    for qid, label, desc in results:
        desc_lower = desc.lower()
        if not any(kw in desc_lower for kw in avoid_keywords):
            return (qid, label, desc)

    # Fall back to first result
    return results[0]


def get_entity(qid: str) -> WikidataEntity:
    """
    Get a Wikidata entity with its properties.

    Args:
        qid: Wikidata Q-ID (e.g., "Q11982" for Photosynthesis)

    Returns:
        WikidataEntity with parsed properties
    """
    url = "https://www.wikidata.org/w/api.php"
    params = {
        'action': 'wbgetentities',
        'ids': qid,
        'props': 'labels|descriptions|claims|sitelinks',
        'languages': 'en',
        'format': 'json',
    }

    data = _api_request(url, params)

    if 'error' in data:
        raise WikidataError(data['error'].get('info', 'Unknown error'))

    entity_data = data.get('entities', {}).get(qid, {})

    if not entity_data or 'missing' in entity_data:
        raise WikidataError(f"Entity {qid} not found")

    # Extract label and description
    label = entity_data.get('labels', {}).get('en', {}).get('value', qid)
    description = entity_data.get('descriptions', {}).get('en', {}).get('value', '')

    # Extract claims (properties)
    claims = entity_data.get('claims', {})

    def extract_qids(prop: str) -> List[str]:
        """Extract Q-IDs from a property's claims."""
        qids = []
        for claim in claims.get(prop, []):
            try:
                value = claim['mainsnak']['datavalue']['value']
                if isinstance(value, dict) and 'id' in value:
                    qids.append(value['id'])
            except (KeyError, TypeError):
                pass
        return qids

    return WikidataEntity(
        qid=qid,
        label=label,
        description=description,
        instance_of=extract_qids('P31'),
        subclass_of=extract_qids('P279'),
        part_of=extract_qids('P361'),
        has_parts=extract_qids('P527'),
        uses=extract_qids('P2283'),
        facet_of=extract_qids('P1269'),
        sitelinks_count=len(entity_data.get('sitelinks', {})),
        properties_count=len(claims),
    )


def get_graph_metrics(qid: str) -> GraphMetrics:
    """
    Get graph metrics for an entity via SPARQL.

    Makes separate lightweight SPARQL queries for:
    - Incoming reference count
    - Subclass count
    - Hierarchy depth

    Outgoing count derived from entity properties.

    Args:
        qid: Wikidata Q-ID

    Returns:
        GraphMetrics with all computed values
    """
    # Get entity for sitelinks/properties/outgoing estimate
    entity = get_entity(qid)

    # Outgoing is approximated by properties count (each property points somewhere)
    outgoing = entity.properties_count

    # Get incoming count (simpler query, faster)
    incoming = _get_incoming_count(qid)

    # Get subclass count
    children = _get_subclass_count(qid)

    # Get hierarchy depth
    depth = _get_hierarchy_depth(qid)

    return GraphMetrics(
        qid=qid,
        incoming_refs=incoming,
        outgoing_refs=outgoing,
        sitelinks=entity.sitelinks_count,
        properties=entity.properties_count,
        depth=depth,
        subclass_count=children,
    )


def _get_incoming_count(qid: str) -> int:
    """Count items that reference this entity."""
    query = f"""
    SELECT (COUNT(?item) AS ?count) WHERE {{
        ?item ?p wd:{qid} .
        FILTER(?p != schema:about)
    }}
    """
    try:
        data = _sparql_query(query)
        bindings = data.get('results', {}).get('bindings', [{}])[0]
        return int(bindings.get('count', {}).get('value', 0))
    except:
        return 0


def _get_subclass_count(qid: str) -> int:
    """Count direct subclasses (taxonomic children)."""
    query = f"""
    SELECT (COUNT(?child) AS ?count) WHERE {{
        ?child wdt:P279 wd:{qid} .
    }}
    """
    try:
        data = _sparql_query(query)
        bindings = data.get('results', {}).get('bindings', [{}])[0]
        return int(bindings.get('count', {}).get('value', 0))
    except:
        return 0


def _get_hierarchy_depth(qid: str, max_depth: int = 30) -> int:
    """Get depth by counting P279 ancestors."""
    query = f"""
    SELECT (COUNT(DISTINCT ?ancestor) AS ?depth) WHERE {{
        wd:{qid} wdt:P279+ ?ancestor .
    }}
    """

    try:
        data = _sparql_query(query)
        bindings = data.get('results', {}).get('bindings', [{}])[0]
        return int(bindings.get('depth', {}).get('value', 0))
    except:
        return 0


def get_ancestry(qid: str, max_depth: int = 20) -> AncestryPath:
    """
    Get the P279 (subclass of) ancestry chain.

    Args:
        qid: Starting entity Q-ID
        max_depth: Maximum ancestors to fetch

    Returns:
        AncestryPath with ordered ancestors and labels
    """
    # Get ordered ancestors via property path
    query = f"""
    SELECT ?ancestor ?ancestorLabel WHERE {{
        wd:{qid} wdt:P279+ ?ancestor .
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT {max_depth}
    """

    data = _sparql_query(query)
    bindings = data.get('results', {}).get('bindings', [])

    ancestors = []
    labels = {}

    for binding in bindings:
        ancestor_uri = binding.get('ancestor', {}).get('value', '')
        if ancestor_uri:
            ancestor_qid = ancestor_uri.split('/')[-1]
            if ancestor_qid not in ancestors:
                ancestors.append(ancestor_qid)
                label = binding.get('ancestorLabel', {}).get('value', ancestor_qid)
                labels[ancestor_qid] = label

    return AncestryPath(qid=qid, ancestors=ancestors, labels=labels)


def get_labels(qids: List[str]) -> Dict[str, str]:
    """
    Get labels for multiple Q-IDs in a single request.

    Args:
        qids: List of Q-IDs to look up

    Returns:
        Dict mapping Q-ID to label
    """
    if not qids:
        return {}

    # API limits to 50 IDs per request
    labels = {}

    for i in range(0, len(qids), 50):
        batch = qids[i:i+50]
        url = "https://www.wikidata.org/w/api.php"
        params = {
            'action': 'wbgetentities',
            'ids': '|'.join(batch),
            'props': 'labels',
            'languages': 'en',
            'format': 'json',
        }

        data = _api_request(url, params)

        for qid, entity in data.get('entities', {}).items():
            label = entity.get('labels', {}).get('en', {}).get('value', qid)
            labels[qid] = label

    return labels


# =============================================================================
# Smart Expansion with Wikidata Metadata Filtering
# =============================================================================

@dataclass
class AdjacentNode:
    """An adjacent node with Wikidata-provided relevance signals."""
    qid: str
    label: str
    relation: str       # Property that connects (P31, P279, etc.)
    sitelinks: int      # Notability signal
    statements: int     # Semantic richness
    relevance_score: float = 0.0  # Computed score


def get_adjacent_with_metadata(
    qid: str,
    min_sitelinks: int = 5,
    max_results: int = 10,
) -> List[AdjacentNode]:
    """
    Get adjacent nodes with Wikidata metadata for filtering.

    Uses a single SPARQL query to fetch neighbors + their sitelink counts,
    letting Wikidata do the heavy lifting for relevance filtering.

    Properties queried (by priority):
    - P31 (instance of) - highest signal
    - P279 (subclass of) - taxonomic relation
    - P361 (part of) - structural relation
    - P527 (has parts) - lower signal, often noisy

    Args:
        qid: Source entity Q-ID
        min_sitelinks: Minimum sitelinks to include (filters noise)
        max_results: Maximum nodes to return

    Returns:
        List of AdjacentNode sorted by relevance_score
    """
    # Single SPARQL query gets neighbors + their metadata
    query = f"""
    SELECT ?neighbor ?neighborLabel ?relation ?sitelinks ?statements WHERE {{
      {{
        wd:{qid} ?relation ?neighbor .
        VALUES ?relation {{ wdt:P31 wdt:P279 wdt:P361 wdt:P527 }}
      }} UNION {{
        ?neighbor ?relation wd:{qid} .
        VALUES ?relation {{ wdt:P31 wdt:P279 }}
      }}

      # Get sitelinks count (notability)
      ?neighbor wikibase:sitelinks ?sitelinks .

      # Get statements count (richness)
      ?neighbor wikibase:statements ?statements .

      # Filter by minimum sitelinks
      FILTER(?sitelinks >= {min_sitelinks})

      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    ORDER BY DESC(?sitelinks)
    LIMIT {max_results * 2}
    """

    try:
        data = _sparql_query(query)
        bindings = data.get('results', {}).get('bindings', [])
    except Exception as e:
        print(f"  Adjacent query failed: {e}")
        return []

    # Property priority weights (higher = more meaningful relation)
    property_weights = {
        'P31': 1.0,   # instance of - defines what it IS
        'P279': 0.9,  # subclass of - taxonomic
        'P361': 0.7,  # part of - structural
        'P527': 0.4,  # has parts - often noisy
    }

    results = []
    seen = set()

    for binding in bindings:
        neighbor_uri = binding.get('neighbor', {}).get('value', '')
        if not neighbor_uri or '/Q' not in neighbor_uri:
            continue

        neighbor_qid = neighbor_uri.split('/')[-1]
        if neighbor_qid in seen or neighbor_qid == qid:
            continue
        seen.add(neighbor_qid)

        # Extract relation property
        relation_uri = binding.get('relation', {}).get('value', '')
        relation = relation_uri.split('/')[-1] if relation_uri else 'unknown'

        # Get metadata
        sitelinks = int(binding.get('sitelinks', {}).get('value', 0))
        statements = int(binding.get('statements', {}).get('value', 0))
        label = binding.get('neighborLabel', {}).get('value', neighbor_qid)

        # Compute relevance score from Wikidata metadata
        # Formula: property_weight * log(sitelinks) * sqrt(statements)
        prop_weight = property_weights.get(relation, 0.3)
        relevance = prop_weight * math.log(sitelinks + 1) * math.sqrt(statements + 1)

        results.append(AdjacentNode(
            qid=neighbor_qid,
            label=label,
            relation=relation,
            sitelinks=sitelinks,
            statements=statements,
            relevance_score=relevance,
        ))

    # Sort by relevance and return top results
    results.sort(key=lambda x: -x.relevance_score)
    return results[:max_results]


def get_topically_related(
    qid: str,
    reference_qids: List[str],
    max_results: int = 5,
    min_sitelinks: int = 10,
) -> List[AdjacentNode]:
    """
    Get adjacent nodes filtered by Wikidata notability metrics.

    Note: Ancestry-based topical filtering doesn't work well because
    Wikidata's P279 taxonomies often don't connect semantically related
    concepts (e.g., "bank" and "savings account" have zero ancestry overlap).

    Instead, we rely on:
    1. Sitelinks count (notability/importance filter)
    2. Statements count (semantic richness)
    3. Property type weights (P31/P279 > P527)

    Args:
        qid: Source entity Q-ID
        reference_qids: Q-IDs from reference frame (used for direct connection check)
        max_results: Maximum nodes to return
        min_sitelinks: Minimum sitelinks threshold for notability

    Returns:
        List of AdjacentNode sorted by relevance
    """
    # Get adjacent with metadata filtering
    adjacent = get_adjacent_with_metadata(
        qid,
        min_sitelinks=min_sitelinks,
        max_results=max_results * 2,
    )

    if not adjacent:
        return []

    # If we have reference QIDs, boost nodes that are IN the reference set
    # (direct connection is meaningful even without ancestry overlap)
    if reference_qids:
        ref_set = set(reference_qids)
        for node in adjacent:
            if node.qid in ref_set:
                # Direct match to reference frame - very relevant
                node.relevance_score *= 3.0

    # Re-sort after potential boosts
    adjacent.sort(key=lambda x: -x.relevance_score)
    return adjacent[:max_results]


def filter_by_shared_properties(
    nodes: List[AdjacentNode],
    reference_qid: str,
    min_shared: int = 1,
) -> List[AdjacentNode]:
    """
    Filter nodes that share properties with a reference entity.

    This is an alternative to ancestry overlap - checks if entities
    share common instance_of or facet_of values.

    Args:
        nodes: Candidate nodes to filter
        reference_qid: Reference entity Q-ID
        min_shared: Minimum shared properties required

    Returns:
        Filtered list of nodes
    """
    try:
        ref_entity = get_entity(reference_qid)
        ref_types = set(ref_entity.instance_of + ref_entity.facet_of)
    except:
        return nodes

    if not ref_types:
        return nodes

    filtered = []
    for node in nodes:
        try:
            node_entity = get_entity(node.qid)
            node_types = set(node_entity.instance_of + node_entity.facet_of)

            shared = ref_types & node_types
            if len(shared) >= min_shared:
                node.relevance_score *= (1.0 + 0.3 * len(shared))
                filtered.append(node)
        except:
            pass

    return filtered if filtered else nodes[:len(nodes)//2]


def concept_distance(qid1: str, qid2: str) -> tuple[int, Optional[str]]:
    """
    Compute semantic distance between two concepts via shared ancestry.

    Args:
        qid1: First concept Q-ID
        qid2: Second concept Q-ID

    Returns:
        (distance, shared_ancestor_qid) where distance is -1 if no common ancestor
    """
    ancestry1 = get_ancestry(qid1)
    ancestry2 = get_ancestry(qid2)

    shared = ancestry1.shared_ancestor(ancestry2)
    distance = ancestry1.distance_to(ancestry2)

    return distance, shared


# =============================================================================
# Novelty Integration
# =============================================================================

@dataclass
class WikidataNoveltyInputs:
    """
    Inputs derived from Wikidata for novelty computation.

    These map directly to our four novelty components:
    - integration_resistance: inverse of how connected the concept is
    - depth_context: where in hierarchy contradictions would occur
    - coverage_gap: inverse of global reach
    - disruption_potential: based on centrality and establishment
    """
    qid: str
    label: str

    # Raw metrics
    metrics: GraphMetrics
    ancestry: AncestryPath

    # Pre-computed novelty inputs (0-1 scale, higher = more novel)
    integration_resistance: float = 0.0
    depth_factor: float = 0.0
    coverage_gap: float = 0.0
    disruption_potential: float = 0.0

    @classmethod
    def from_qid(cls, qid: str) -> "WikidataNoveltyInputs":
        """Build novelty inputs from a Q-ID."""
        entity = get_entity(qid)
        metrics = get_graph_metrics(qid)
        ancestry = get_ancestry(qid)

        # Compute novelty-relevant scores (inverted where needed)
        integration_resistance = 1.0 - metrics.integration_score
        coverage_gap = 1.0 - metrics.coverage_score
        disruption_potential = 1.0 - metrics.establishment_score

        # Depth factor: deeper concepts are more specific, less foundational
        # But for NOVELTY, we care about depth relative to reference
        depth_factor = min(metrics.depth / 15, 1.0)  # Normalize to ~15 levels

        return cls(
            qid=qid,
            label=entity.label,
            metrics=metrics,
            ancestry=ancestry,
            integration_resistance=integration_resistance,
            depth_factor=depth_factor,
            coverage_gap=coverage_gap,
            disruption_potential=disruption_potential,
        )

    @classmethod
    def from_text(cls, text: str) -> Optional["WikidataNoveltyInputs"]:
        """Search for concept and build novelty inputs using best match."""
        match = best_match(text)
        if not match:
            return None

        qid, label, _ = match
        return cls.from_qid(qid)


def compare_novelty_inputs(
    concept: WikidataNoveltyInputs,
    reference: WikidataNoveltyInputs,
) -> Dict[str, float]:
    """
    Compare a concept against a reference using Wikidata graph structure.

    Returns novelty component estimates based on:
    - Shared ancestry (for contradiction depth)
    - Relative metrics (for integration/coverage)

    Args:
        concept: The concept being evaluated
        reference: The reference frame concept

    Returns:
        Dict with novelty component estimates
    """
    # Semantic distance via shared ancestry
    distance, shared_ancestor = concept_distance(concept.qid, reference.qid)

    # If they share ancestry, contradiction depth depends on WHERE they diverge
    if distance >= 0 and shared_ancestor:
        # Find depth of shared ancestor in concept's chain
        if shared_ancestor in concept.ancestry.ancestors:
            divergence_depth = concept.ancestry.ancestors.index(shared_ancestor)
        else:
            divergence_depth = 0
        # Shallower divergence = more fundamental disagreement
        contradiction_depth = 1.0 - (divergence_depth / max(len(concept.ancestry.ancestors), 1))
    else:
        # No shared ancestry - completely orthogonal
        contradiction_depth = 0.0  # Can't contradict what's unrelated

    # Integration resistance: relative connectivity
    concept_connectivity = concept.metrics.incoming_refs + concept.metrics.outgoing_refs
    ref_connectivity = reference.metrics.incoming_refs + reference.metrics.outgoing_refs

    if ref_connectivity > 0:
        relative_integration = concept_connectivity / ref_connectivity
        integration_resistance = 1.0 - min(relative_integration, 1.0)
    else:
        integration_resistance = concept.integration_resistance

    # Coverage gap: relative global reach
    concept_reach = concept.metrics.sitelinks + concept.metrics.properties
    ref_reach = reference.metrics.sitelinks + reference.metrics.properties

    if ref_reach > 0:
        relative_coverage = concept_reach / ref_reach
        coverage_gap = 1.0 - min(relative_coverage, 1.0)
    else:
        coverage_gap = concept.coverage_gap

    # Disruption: inverse of establishment relative to reference
    disruption = concept.disruption_potential

    return {
        'integration_resistance': integration_resistance,
        'contradiction_depth': contradiction_depth,
        'coverage_gap': coverage_gap,
        'allocation_disruption': disruption,
        'semantic_distance': distance,
        'shared_ancestor': shared_ancestor,
    }


# =============================================================================
# Convenience / Testing
# =============================================================================

def describe_concept(text_or_qid: str) -> str:
    """Get a human-readable description of a concept's graph position."""

    # Check if it's a Q-ID or text
    if text_or_qid.startswith('Q') and text_or_qid[1:].isdigit():
        qid = text_or_qid
    else:
        results = search_concept(text_or_qid, limit=1)
        if not results:
            return f"Could not find: {text_or_qid}"
        qid = results[0][0]

    inputs = WikidataNoveltyInputs.from_qid(qid)

    lines = [
        f"WIKIDATA: {inputs.label} ({inputs.qid})",
        "=" * 50,
        "",
        "GRAPH METRICS:",
        f"  Incoming refs: {inputs.metrics.incoming_refs}",
        f"  Outgoing refs: {inputs.metrics.outgoing_refs}",
        f"  Sitelinks: {inputs.metrics.sitelinks}",
        f"  Properties: {inputs.metrics.properties}",
        f"  Hierarchy depth: {inputs.metrics.depth}",
        f"  Subclasses: {inputs.metrics.subclass_count}",
        "",
        "DERIVED SCORES:",
        f"  Centrality ratio: {inputs.metrics.centrality_ratio:.2f}",
        f"  Integration score: {inputs.metrics.integration_score:.2f}",
        f"  Coverage score: {inputs.metrics.coverage_score:.2f}",
        f"  Establishment score: {inputs.metrics.establishment_score:.2f}",
        "",
        "NOVELTY INPUTS (higher = more novel):",
        f"  Integration resistance: {inputs.integration_resistance:.2f}",
        f"  Depth factor: {inputs.depth_factor:.2f}",
        f"  Coverage gap: {inputs.coverage_gap:.2f}",
        f"  Disruption potential: {inputs.disruption_potential:.2f}",
        "",
        f"ANCESTRY ({len(inputs.ancestry.ancestors)} levels):",
    ]

    for i, ancestor in enumerate(inputs.ancestry.ancestors[:10]):
        label = inputs.ancestry.labels.get(ancestor, ancestor)
        lines.append(f"  {i+1}. {label} ({ancestor})")

    if len(inputs.ancestry.ancestors) > 10:
        lines.append(f"  ... and {len(inputs.ancestry.ancestors) - 10} more")

    return "\n".join(lines)


def compute_wikidata_novelty(
    concept_text: str,
    reference_text: str = None,
) -> Dict[str, Any]:
    """
    Compute novelty of a concept using Wikidata graph metrics.

    This is the main integration point - computes all four novelty components
    directly from Wikidata's knowledge graph.

    Args:
        concept_text: The concept to evaluate (will be searched in Wikidata)
        reference_text: Optional reference concept. If None, uses absolute metrics.

    Returns:
        Dict with:
        - qid: Wikidata Q-ID
        - label: Concept label
        - components: Dict of four novelty component scores (0-1)
        - composite: Geometric mean of components
        - metrics: Raw GraphMetrics
        - ancestry_depth: Number of ancestors in P279 chain
    """
    concept = WikidataNoveltyInputs.from_text(concept_text)
    if concept is None:
        return {'error': f'Could not find concept: {concept_text}'}

    if reference_text:
        reference = WikidataNoveltyInputs.from_text(reference_text)
        if reference is None:
            return {'error': f'Could not find reference: {reference_text}'}
        comparison = compare_novelty_inputs(concept, reference)
        components = {
            'integration_resistance': comparison['integration_resistance'],
            'contradiction_depth': comparison['contradiction_depth'],
            'coverage_gap': comparison['coverage_gap'],
            'allocation_disruption': comparison['allocation_disruption'],
        }
    else:
        # Absolute metrics (no reference)
        components = {
            'integration_resistance': concept.integration_resistance,
            'contradiction_depth': concept.depth_factor,  # Use depth as proxy
            'coverage_gap': concept.coverage_gap,
            'allocation_disruption': concept.disruption_potential,
        }

    # Compute composite (geometric mean)
    epsilon = 0.01
    values = [v + epsilon for v in components.values()]
    product = 1.0
    for v in values:
        product *= v
    composite = product ** (1.0 / len(values))

    return {
        'qid': concept.qid,
        'label': concept.label,
        'components': components,
        'composite': composite,
        'metrics': {
            'incoming_refs': concept.metrics.incoming_refs,
            'outgoing_refs': concept.metrics.outgoing_refs,
            'sitelinks': concept.metrics.sitelinks,
            'properties': concept.metrics.properties,
            'depth': concept.metrics.depth,
            'centrality_ratio': concept.metrics.centrality_ratio,
        },
        'ancestry_depth': len(concept.ancestry.ancestors),
    }


if __name__ == "__main__":
    # Quick test
    print(describe_concept("Photosynthesis"))
    print()
    print(describe_concept("Big Bang"))

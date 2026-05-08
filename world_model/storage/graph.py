import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..models import DeviationNode, DeviationType, Edge, EdgeType, EvidencePointer, Source


class DeviationGraph:
    """
    Storage and query layer for deviation nodes.

    Maintains the graph of deviations that constitute a person's
    delta from baseline human.
    """

    def __init__(self):
        self.nodes: dict[str, DeviationNode] = {}
        self.sources: dict[str, Source] = {}
        self._dimension_index: dict[str, list[str]] = {}  # dimension -> node_ids
        self._type_index: dict[DeviationType, list[str]] = {}  # type -> node_ids

    def add_node(self, node: DeviationNode) -> None:
        """Add a deviation node to the graph"""
        self.nodes[node.id] = node
        self._index_node(node)

    def add_source(self, source: Source) -> None:
        """Register a source document"""
        self.sources[source.id] = source

    def get_node(self, node_id: str) -> Optional[DeviationNode]:
        """Get a node by ID"""
        return self.nodes.get(node_id)

    def get_by_dimension(self, dimension: str) -> list[DeviationNode]:
        """Get all nodes matching a dimension (case-insensitive partial match)"""
        dimension_lower = dimension.lower()
        results = []
        for dim, node_ids in self._dimension_index.items():
            if dimension_lower in dim.lower():
                results.extend(self.nodes[nid] for nid in node_ids)
        return results

    def get_by_type(self, deviation_type: DeviationType) -> list[DeviationNode]:
        """Get all nodes of a specific type"""
        node_ids = self._type_index.get(deviation_type, [])
        return [self.nodes[nid] for nid in node_ids]

    def get_children(self, node_id: str) -> list[DeviationNode]:
        """Get all child nodes of a given node"""
        node = self.nodes.get(node_id)
        if not node:
            return []
        return [self.nodes[cid] for cid in node.children_ids if cid in self.nodes]

    def get_parent(self, node_id: str) -> Optional[DeviationNode]:
        """Get parent node if exists"""
        node = self.nodes.get(node_id)
        if not node or not node.parent_id:
            return None
        return self.nodes.get(node.parent_id)

    def get_connected(self, node_id: str, edge_type: Optional[EdgeType] = None) -> list[DeviationNode]:
        """Get nodes connected via edges"""
        node = self.nodes.get(node_id)
        if not node:
            return []

        results = []
        for edge in node.edges:
            if edge_type is None or edge.edge_type == edge_type:
                target = self.nodes.get(edge.target_id)
                if target:
                    results.append(target)
        return results

    def get_roots(self) -> list[DeviationNode]:
        """Get top-level nodes (no parent)"""
        return [n for n in self.nodes.values() if n.parent_id is None]

    def get_high_confidence(self, threshold: float = 0.7) -> list[DeviationNode]:
        """Get nodes above confidence threshold"""
        return [n for n in self.nodes.values() if n.confidence >= threshold]

    def get_high_magnitude(self, threshold: float = 0.7) -> list[DeviationNode]:
        """Get most extreme deviations"""
        return [n for n in self.nodes.values() if n.magnitude >= threshold]

    def search(
        self,
        query: str,
        type_filter: Optional[DeviationType] = None,
        min_confidence: float = 0.0,
        min_magnitude: float = 0.0
    ) -> list[DeviationNode]:
        """Search nodes by text and filters"""
        query_lower = query.lower()
        results = []

        for node in self.nodes.values():
            # Apply filters
            if type_filter and node.deviation_type != type_filter:
                continue
            if node.confidence < min_confidence:
                continue
            if node.magnitude < min_magnitude:
                continue

            # Text search
            searchable = f"{node.dimension} {node.baseline_assumption} {node.deviation_description}".lower()
            if query_lower in searchable:
                results.append(node)

        return sorted(results, key=lambda n: n.confidence, reverse=True)

    def to_summary(self, max_depth: int = 2) -> str:
        """Generate human-readable summary of the graph"""
        lines = [f"Deviation Graph: {len(self.nodes)} nodes from {len(self.sources)} sources\n"]

        # Group by type
        by_type: dict[DeviationType, list[DeviationNode]] = {}
        for node in self.nodes.values():
            by_type.setdefault(node.deviation_type, []).append(node)

        for dtype, nodes in sorted(by_type.items(), key=lambda x: x[0].value):
            lines.append(f"\n## {dtype.value.upper()} ({len(nodes)} deviations)")
            # Show top nodes by confidence
            top_nodes = sorted(nodes, key=lambda n: n.confidence, reverse=True)[:5]
            for node in top_nodes:
                lines.append(f"  - {node.dimension} (confidence={node.confidence:.2f}, magnitude={node.magnitude:.2f})")
                lines.append(f"    {node.deviation_description[:100]}...")

        return "\n".join(lines)

    def to_profile(self, depth: int = 1) -> str:
        """Generate a compact profile description"""
        high_conf = sorted(self.get_high_confidence(0.6), key=lambda n: n.magnitude, reverse=True)

        lines = ["# Personal Deviation Profile\n"]
        lines.append("## Core Deviations (high confidence, high magnitude)\n")

        for node in high_conf[:10]:
            lines.append(f"**{node.dimension}** [{node.deviation_type.value}]")
            lines.append(f"- Baseline: {node.baseline_assumption}")
            lines.append(f"- This person: {node.deviation_description}")
            lines.append("")

        return "\n".join(lines)

    def _index_node(self, node: DeviationNode) -> None:
        """Update indices for a node"""
        # Dimension index
        dim_key = node.dimension.lower()
        if dim_key not in self._dimension_index:
            self._dimension_index[dim_key] = []
        self._dimension_index[dim_key].append(node.id)

        # Type index
        if node.deviation_type not in self._type_index:
            self._type_index[node.deviation_type] = []
        self._type_index[node.deviation_type].append(node.id)

    def save(self, path: str) -> None:
        """Save graph to JSON file"""
        data = {
            "nodes": [],
            "sources": []
        }

        for node in self.nodes.values():
            node_dict = {
                "id": node.id,
                "dimension": node.dimension,
                "deviation_type": node.deviation_type.value,
                "baseline_assumption": node.baseline_assumption,
                "deviation_description": node.deviation_description,
                "magnitude": node.magnitude,
                "confidence": node.confidence,
                "stability": node.stability,
                "parent_id": node.parent_id,
                "children_ids": node.children_ids,
                "evidence": [
                    {"source_id": e.source_id, "excerpt": e.excerpt, "context": e.context}
                    for e in node.evidence
                ],
                "contradictions": [
                    {"source_id": e.source_id, "excerpt": e.excerpt, "context": e.context}
                    for e in node.contradictions
                ],
                "edges": [
                    {"target_id": e.target_id, "edge_type": e.edge_type.value, "strength": e.strength}
                    for e in node.edges
                ],
                "first_observed": node.first_observed.isoformat(),
                "last_reinforced": node.last_reinforced.isoformat(),
                "observation_count": node.observation_count,
                "tags": node.tags,
                "notes": node.notes
            }
            data["nodes"].append(node_dict)

        for source in self.sources.values():
            source_dict = {
                "id": source.id,
                "name": source.name,
                "path": source.path,
                "ingested_at": source.ingested_at.isoformat(),
                "metadata": source.metadata
            }
            data["sources"].append(source_dict)

        Path(path).write_text(json.dumps(data, indent=2))

    def load(self, path: str) -> None:
        """Load graph from JSON file"""
        data = json.loads(Path(path).read_text())

        for source_dict in data.get("sources", []):
            source = Source(
                id=source_dict["id"],
                name=source_dict["name"],
                path=source_dict.get("path"),
                ingested_at=datetime.fromisoformat(source_dict["ingested_at"]),
                metadata=source_dict.get("metadata", {})
            )
            self.sources[source.id] = source

        for node_dict in data.get("nodes", []):
            evidence = [
                EvidencePointer(source_id=e["source_id"], excerpt=e["excerpt"], context=e["context"])
                for e in node_dict.get("evidence", [])
            ]
            contradictions = [
                EvidencePointer(source_id=e["source_id"], excerpt=e["excerpt"], context=e["context"])
                for e in node_dict.get("contradictions", [])
            ]
            edges = [
                Edge(target_id=e["target_id"], edge_type=EdgeType(e["edge_type"]), strength=e["strength"])
                for e in node_dict.get("edges", [])
            ]

            node = DeviationNode(
                id=node_dict["id"],
                dimension=node_dict["dimension"],
                deviation_type=DeviationType(node_dict["deviation_type"]),
                baseline_assumption=node_dict["baseline_assumption"],
                deviation_description=node_dict["deviation_description"],
                magnitude=node_dict["magnitude"],
                confidence=node_dict["confidence"],
                stability=node_dict["stability"],
                parent_id=node_dict.get("parent_id"),
                children_ids=node_dict.get("children_ids", []),
                evidence=evidence,
                contradictions=contradictions,
                edges=edges,
                first_observed=datetime.fromisoformat(node_dict["first_observed"]),
                last_reinforced=datetime.fromisoformat(node_dict["last_reinforced"]),
                observation_count=node_dict.get("observation_count", 1),
                tags=node_dict.get("tags", []),
                notes=node_dict.get("notes", "")
            )
            self.add_node(node)

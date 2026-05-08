"""
Persistence layer for complete world models.

A WorldModel bundles:
- ObservationStore: all extracted observations
- AgentSet: the person's tendency allocations
- TreeStore: all value trees with staked nodes

File format is a single JSON file containing all components.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..models.observation import Observation, ObservationStore
from ..models.agent import AgentSet
from ..models.tree import Tree, TreeStore


@dataclass
class WorldModel:
    """
    Complete world model for a person.

    Contains all observations, agents, and value trees.
    """

    # Identity
    name: str = ""                    # Person's name or identifier
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Components
    observations: ObservationStore = field(default_factory=ObservationStore)
    agents: AgentSet = field(default_factory=AgentSet)
    trees: TreeStore = field(default_factory=TreeStore)

    # Metadata
    metadata: dict = field(default_factory=dict)

    def save(self, path: str | Path):
        """Save world model to JSON file."""
        path = Path(path)

        self.updated_at = datetime.now()

        data = {
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "observations": {
                "count": len(self.observations),
                "items": [obs.to_dict() for obs in self.observations.all()]
            },
            "agents": self.agents.to_dict(),
            "trees": {
                "count": len(self.trees),
                "items": {tid: t.to_dict() for tid, t in self.trees.trees.items()}
            },
            "metadata": self.metadata,
        }

        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "WorldModel":
        """Load world model from JSON file."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))

        model = cls(
            name=data.get("name", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {}),
        )

        # Load observations
        for obs_data in data.get("observations", {}).get("items", []):
            obs = Observation.from_dict(obs_data)
            model.observations.add(obs)

        # Load agents
        if "agents" in data:
            model.agents = AgentSet.from_dict(data["agents"])

        # Load trees
        for tree_data in data.get("trees", {}).get("items", {}).values():
            tree = Tree.from_dict(tree_data)
            model.trees.add(tree)

        return model

    def summary(self) -> str:
        """Human-readable summary of the world model."""
        lines = [
            f"WorldModel: {self.name}",
            f"  Created: {self.created_at.strftime('%Y-%m-%d %H:%M')}",
            f"  Updated: {self.updated_at.strftime('%Y-%m-%d %H:%M')}",
            f"  Observations: {len(self.observations)}",
            f"  Trees: {len(self.trees)}",
        ]

        if self.trees.trees:
            lines.append("  Value hierarchies:")
            for tree in self.trees.all():
                depth = tree.depth()
                nodes = len(tree.all_nodes())
                lines.append(f"    - {tree.root_value}: {nodes} nodes, depth {depth}, score {tree.score:.2f}")

        lines.append("  Agent allocations:")
        for agent in sorted(self.agents.all(), key=lambda a: -a.allocation):
            stakes = f" ({agent.stakes_placed} stakes)" if agent.stakes_placed > 0 else ""
            lines.append(f"    - {agent.tendency.value}: {agent.allocation:.0%}{stakes}")

        return "\n".join(lines)


def create_world_model(
    name: str,
    observations_path: Optional[str | Path] = None,
) -> WorldModel:
    """
    Create a new world model, optionally loading existing observations.

    Args:
        name: Person's name or identifier
        observations_path: Path to observations.json (optional)

    Returns:
        New WorldModel instance
    """
    model = WorldModel(name=name)

    if observations_path:
        path = Path(observations_path)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            for obs_data in data.get("observations", []):
                obs = Observation.from_dict(obs_data)
                model.observations.add(obs)

    return model

"""
Autonet Node Implementations

This package provides the node implementations for the Autonet distributed
AI training and inference network.
"""

__version__ = "0.1.0"

from .core import (
    Node,
    NodeRole,
    Constitution,
    DEFAULT_CONSTITUTION,
    create_node,
)
from .proposer import ProposerNode
from .solver import SolverNode
from .coordinator import CoordinatorNode
from .aggregator import AggregatorNode

__all__ = [
    "Node",
    "NodeRole",
    "Constitution",
    "DEFAULT_CONSTITUTION",
    "create_node",
    "ProposerNode",
    "SolverNode",
    "CoordinatorNode",
    "AggregatorNode",
]

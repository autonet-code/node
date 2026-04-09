"""
Autonet Node Implementations

This package provides the node implementations for the Autonet distributed
AI training and inference network.

Autonomous training path only — task-driven nodes (Proposer, Solver,
Coordinator) have been deprecated.
"""

__version__ = "0.1.0"

from .core import (
    Node,
    NodeRole,
    Constitution,
    DEFAULT_CONSTITUTION,
    create_node,
)
from .aggregator import AggregatorNode

__all__ = [
    "Node",
    "NodeRole",
    "Constitution",
    "DEFAULT_CONSTITUTION",
    "create_node",
    "AggregatorNode",
]

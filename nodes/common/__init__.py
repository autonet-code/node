"""Common utilities for Autonet nodes."""

from .blockchain import BlockchainInterface, TransactionResult
from .contracts import ContractRegistry, ContractHandle
from .ipfs import IPFSClient
from .crypto import hash_content, hash_string, verify_signature
from .repops import (
    RepOpsConfig,
    RepOpsContext,
    DeterministicRNG,
    ReproducibleMatMul,
    ReproducibleOptimizer,
    create_repops_context,
    verify_checkpoint_consistency,
    REPOPS_VERSION,
)

__all__ = [
    "BlockchainInterface",
    "TransactionResult",
    "ContractRegistry",
    "ContractHandle",
    "IPFSClient",
    "hash_content",
    "hash_string",
    "verify_signature",
    # RepOps - Reproducible Operators
    "RepOpsConfig",
    "RepOpsContext",
    "DeterministicRNG",
    "ReproducibleMatMul",
    "ReproducibleOptimizer",
    "create_repops_context",
    "verify_checkpoint_consistency",
    "REPOPS_VERSION",
]

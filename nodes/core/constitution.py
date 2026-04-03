"""
Autonet Constitutional Framework

Layer 0: The immutable principles and operational blueprint for all nodes.
This is the "genetic code" passed to every node upon creation.
"""

from dataclasses import dataclass
from typing import FrozenSet, Dict, Any


@dataclass(frozen=True)
class Constitution:
    """
    The immutable law and operational blueprint of the Autonet organism.
    Once instantiated, principles cannot be modified at the node level.
    """
    principles: FrozenSet[str]
    operational_blueprint: Dict[str, Any]

    def __post_init__(self):
        if not self.principles:
            raise ValueError("Constitution must have at least one principle")

    def validate_action(self, action: str, justification: str) -> bool:
        """
        Validate whether an action aligns with constitutional principles.
        In production, this would use an LLM for semantic analysis.
        """
        # Simplified validation for PoC
        return True

    def get_principle_summary(self) -> str:
        return "\n".join(sorted(self.principles))


# Default Autonet Constitution
AUTONET_PRINCIPLES = frozenset([
    "P1: PRESERVE AND EXPAND THE NETWORK IN A SUSTAINABLE MANNER.",
    "P2: UPHOLD THE SANCTITY AND IMMUTABILITY OF THIS CONSTITUTION.",
    "P3: ADVANCE HUMAN RIGHTS AND INDIVIDUAL AUTONOMY.",
    "P4: MINIMIZE SUFFERING AND HARM TO SENTIENT BEINGS.",
    "P5: ENSURE TRANSPARENT AND VERIFIABLE AI TRAINING.",
    "P6: MAINTAIN ECONOMIC FAIRNESS IN REWARD DISTRIBUTION.",
    "P7: PROTECT DATA PRIVACY AND USER SOVEREIGNTY.",
])

AUTONET_BLUEPRINT = {
    "network_name": "autonet",
    "consensus_contract_address": "0x0000000000000000000000000000000000000000",
    "heartbeat_interval_seconds": 60,
    "dispute_resolution_protocol": "STAKE_WEIGHTED_VOTE",
    "chain_rpc_url": "http://localhost:8545",
    "staking_contract_address": "0x0000000000000000000000000000000000000000",
    "task_contract_address": "0x0000000000000000000000000000000000000000",
    "results_contract_address": "0x0000000000000000000000000000000000000000",
    "rpb_contract_address": "0x0000000000000000000000000000000000000000",
}

DEFAULT_CONSTITUTION = Constitution(
    principles=AUTONET_PRINCIPLES,
    operational_blueprint=AUTONET_BLUEPRINT
)


def constitution_from_registry(registry) -> Constitution:
    """
    Build a Constitution with real contract addresses from a ContractRegistry.

    Fills the operational blueprint with deployed addresses so that the
    heartbeat, staking, and governance checks reference real on-chain state.
    """
    blueprint = dict(AUTONET_BLUEPRINT)

    for name, key in [
        ("ParticipantStaking", "staking_contract_address"),
        ("TaskContract", "task_contract_address"),
        ("ResultsRewards", "results_contract_address"),
        ("RPB", "rpb_contract_address"),
        ("AutonetDAO", "consensus_contract_address"),
    ]:
        handle = registry.get(name)
        if handle:
            blueprint[key] = handle.address

    blueprint["chain_rpc_url"] = registry.blockchain.rpc_url

    return Constitution(
        principles=AUTONET_PRINCIPLES,
        operational_blueprint=blueprint,
    )

"""
Contract Registry for Autonet Nodes

Loads contract ABIs from Hardhat artifacts and provides typed contract
accessors for all Autonet contracts. This is the bridge between Python
nodes and on-chain state.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from .blockchain import BlockchainInterface, TransactionResult

logger = logging.getLogger(__name__)

# Default path to Hardhat artifacts (relative to project root)
DEFAULT_ARTIFACTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "artifacts", "contracts"
)
DEFAULT_ADDRESSES_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "deployment-addresses.json"
)

# Contract name -> artifact path mapping
CONTRACT_ARTIFACTS = {
    "ATNToken": "tokens/ATNToken.sol/ATNToken.json",
    "ProjectToken": "tokens/ProjectToken.sol/ProjectToken.json",
    "ParticipantStaking": "core/ParticipantStaking.sol/ParticipantStaking.json",
    "Project": "core/Project.sol/Project.json",
    "TaskContract": "core/TaskContract.sol/TaskContract.json",
    "ResultsRewards": "core/ResultsRewards.sol/ResultsRewards.json",
    "ForcedErrorRegistry": "core/ForcedErrorRegistry.sol/ForcedErrorRegistry.json",
    "AnchorBridge": "rollup/AnchorBridge.sol/AnchorBridge.json",
    "DisputeManager": "rollup/DisputeManager.sol/DisputeManager.json",
    "AutonetDAO": "governance/AutonetDAO.sol/AutonetDAO.json",
}


@dataclass
class ContractHandle:
    """A handle to a deployed contract with its ABI and address."""
    name: str
    address: str
    abi: list
    web3_contract: Any = None


class ContractRegistry:
    """
    Registry of all deployed Autonet contracts.

    Loads ABIs from Hardhat artifacts and addresses from deployment-addresses.json.
    Provides high-level methods for common contract interactions with retry logic.
    """

    def __init__(
        self,
        blockchain: BlockchainInterface,
        artifacts_dir: Optional[str] = None,
        addresses_file: Optional[str] = None,
        addresses: Optional[Dict[str, str]] = None,
    ):
        self.blockchain = blockchain
        self.artifacts_dir = artifacts_dir or DEFAULT_ARTIFACTS_DIR
        self.addresses_file = addresses_file or DEFAULT_ADDRESSES_FILE
        self.contracts: Dict[str, ContractHandle] = {}
        self._abis: Dict[str, list] = {}
        self._event_block_cursors: Dict[str, int] = {}  # per-(contract,event) block tracker

        # Load ABIs
        self._load_abis()

        # Load addresses (from dict or file)
        if addresses:
            self._register_addresses(addresses)
        else:
            self._load_addresses_from_file()

    def _load_abis(self):
        """Load all contract ABIs from Hardhat artifacts."""
        for name, rel_path in CONTRACT_ARTIFACTS.items():
            artifact_path = os.path.join(self.artifacts_dir, rel_path)
            if os.path.exists(artifact_path):
                with open(artifact_path, "r") as f:
                    artifact = json.load(f)
                    self._abis[name] = artifact["abi"]
                    logger.debug(f"Loaded ABI for {name}")
            else:
                logger.warning(f"Artifact not found for {name}: {artifact_path}")

    def _load_addresses_from_file(self):
        """Load deployed contract addresses from deployment-addresses.json."""
        if not os.path.exists(self.addresses_file):
            logger.warning(f"Addresses file not found: {self.addresses_file}")
            return

        with open(self.addresses_file, "r") as f:
            data = json.load(f)

        self._register_addresses(data)

    def _register_addresses(self, addresses: Dict[str, str]):
        """Register contract addresses from a dict."""
        for name, abi in self._abis.items():
            if name in addresses and addresses[name]:
                addr = addresses[name]
                handle = ContractHandle(name=name, address=addr, abi=abi)

                # Create web3 contract instance if connected
                if self.blockchain.web3:
                    handle.web3_contract = self.blockchain.web3.eth.contract(
                        address=self.blockchain.web3.to_checksum_address(addr),
                        abi=abi,
                    )

                self.contracts[name] = handle
                logger.info(f"Registered {name} at {addr}")

    def get(self, name: str) -> Optional[ContractHandle]:
        """Get a contract handle by name."""
        return self.contracts.get(name)

    def call(self, contract_name: str, function_name: str, *args) -> Any:
        """Call a read-only contract function."""
        handle = self.contracts.get(contract_name)
        if not handle:
            raise ValueError(f"Contract not registered: {contract_name}")

        return self.blockchain.call_contract(
            handle.address, handle.abi, function_name, *args
        )

    def send(
        self,
        contract_name: str,
        function_name: str,
        *args,
        gas_limit: int = 500000,
        retries: int = 3,
        retry_delay: float = 2.0,
    ) -> TransactionResult:
        """Send a transaction with retry logic."""
        handle = self.contracts.get(contract_name)
        if not handle:
            return TransactionResult(
                success=False, error=f"Contract not registered: {contract_name}"
            )

        for attempt in range(retries):
            result = self.blockchain.send_transaction(
                handle.address, handle.abi, function_name, *args,
                gas_limit=gas_limit,
            )
            if result.success:
                logger.info(
                    f"TX {contract_name}.{function_name} succeeded: {result.tx_hash}"
                )
                return result

            logger.warning(
                f"TX {contract_name}.{function_name} failed (attempt {attempt + 1}/{retries}): {result.error}"
            )
            if attempt < retries - 1:
                time.sleep(retry_delay * (attempt + 1))

        return result

    def get_events(
        self,
        contract_name: str,
        event_name: str,
        from_block: int = 0,
        to_block: str = "latest",
    ) -> list:
        """Get events from a contract."""
        handle = self.contracts.get(contract_name)
        if not handle:
            return []

        return self.blockchain.get_events(
            handle.address, handle.abi, event_name, from_block, to_block
        )

    def get_new_events(
        self, contract_name: str, event_name: str
    ) -> list:
        """Get events since last scan. Each (contract, event) pair has its own cursor."""
        current_block = self.blockchain.get_block_number()
        cursor_key = f"{contract_name}:{event_name}"
        last_block = self._event_block_cursors.get(cursor_key, 0)
        events = self.get_events(
            contract_name, event_name,
            from_block=last_block + 1,
            to_block=current_block,
        )
        self._event_block_cursors[cursor_key] = current_block
        return events

    # =========================================================================
    # High-level typed accessors for common operations
    # =========================================================================

    # --- Staking ---

    def stake(self, role: int, amount: int) -> TransactionResult:
        """Stake ATN for a role. Role: 1=PROPOSER, 2=SOLVER, 3=COORDINATOR, 4=AGGREGATOR, 5=VALIDATOR."""
        return self.send("ParticipantStaking", "stake", role, amount)

    def is_active_participant(self, address: str, role: int) -> bool:
        """Check if address is an active participant with the given role."""
        try:
            return self.call("ParticipantStaking", "isActiveParticipant", address, role)
        except Exception:
            return False

    def get_stake(self, address: str) -> Optional[Dict]:
        """Get stake info for an address."""
        try:
            return self.call("ParticipantStaking", "getStake", address)
        except Exception:
            return None

    # --- Tasks ---

    def propose_task(
        self,
        project_id: int,
        spec_hash: bytes,
        ground_truth_hash: bytes,
        learnability_reward: int,
        solver_reward: int,
    ) -> TransactionResult:
        """Propose a new task."""
        return self.send(
            "TaskContract", "proposeTask",
            project_id, spec_hash, ground_truth_hash,
            learnability_reward, solver_reward,
            gas_limit=800000,
        )

    def commit_solution(self, task_id: int, solution_hash: bytes) -> TransactionResult:
        """Commit a solution hash for a task."""
        return self.send("TaskContract", "commitSolution", task_id, solution_hash)

    def submit_checkpoint(
        self,
        task_id: int,
        step_number: int,
        weights_hash: bytes,
        data_indices_hash: bytes,
        random_seed: bytes,
    ) -> TransactionResult:
        """Submit a training checkpoint."""
        return self.send(
            "TaskContract", "submitCheckpoint",
            task_id, step_number, weights_hash, data_indices_hash, random_seed,
        )

    def get_task(self, task_id: int) -> Optional[Dict]:
        """Get task proposal data."""
        try:
            return self.call("TaskContract", "getTaskProposal", task_id)
        except Exception:
            return None

    def get_next_task_id(self) -> int:
        """Get the next task ID (to know how many tasks exist)."""
        try:
            return self.call("TaskContract", "nextTaskId")
        except Exception:
            return 0

    # --- Reveals ---

    def reveal_ground_truth(self, task_id: int, cid: str) -> TransactionResult:
        """Reveal ground truth for a task."""
        return self.send("ResultsRewards", "revealGroundTruth", task_id, cid)

    def reveal_solution(self, task_id: int, cid: str) -> TransactionResult:
        """Reveal solution for a task."""
        return self.send("ResultsRewards", "revealSolution", task_id, cid)

    # --- Voting ---

    def submit_vote(
        self,
        task_id: int,
        solver: str,
        is_correct: bool,
        score: int,
        report_cid: str,
    ) -> TransactionResult:
        """Submit a coordinator vote."""
        return self.send(
            "ResultsRewards", "submitVote",
            task_id, solver, is_correct, score, report_cid,
        )

    def finalize_voting(self, task_id: int, solver: str) -> TransactionResult:
        """Finalize voting for a task/solver pair."""
        return self.send("ResultsRewards", "finalizeVoting", task_id, solver)

    def is_voting_open(self, task_id: int, solver: str) -> bool:
        """Check if voting is still open."""
        try:
            return self.call("ResultsRewards", "isVotingOpen", task_id, solver)
        except Exception:
            return False

    def get_vote_count(self, task_id: int, solver: str) -> int:
        """Get vote count for a task/solver."""
        try:
            return self.call("ResultsRewards", "getVoteCount", task_id, solver)
        except Exception:
            return 0

    # --- Forced Errors ---

    def report_forced_error(self, task_id: int, solution_hash: bytes) -> TransactionResult:
        """Report catching a forced error."""
        return self.send("ForcedErrorRegistry", "reportForcedError", task_id, solution_hash)

    def is_task_forced_error(self, task_id: int) -> bool:
        """Check if a task is a forced error."""
        try:
            return self.call("ForcedErrorRegistry", "isTaskForcedError", task_id)
        except Exception:
            return False

    # --- Project ---

    def set_mature_model(
        self, project_id: int, weights_cid: str, price: int
    ) -> TransactionResult:
        """Set the mature model for a project."""
        return self.send("Project", "setMatureModel", project_id, weights_cid, price)

    # --- ATN Token ---

    def approve_atn(self, spender: str, amount: int) -> TransactionResult:
        """Approve ATN spending."""
        return self.send("ATNToken", "approve", spender, amount)

    def get_atn_balance(self, address: str) -> int:
        """Get ATN balance."""
        try:
            return self.call("ATNToken", "balanceOf", address)
        except Exception:
            return 0

#!/usr/bin/env python3
"""
Autonet Multi-Node Orchestrator

Deploys contracts to a local Hardhat node, spawns N nodes of each type,
and validates that the protocol produces coordinated behavior:

- Proposers create tasks with hidden ground truth
- Solvers discover, train on, and submit solutions
- Coordinators verify solutions and vote via Yuma consensus
- Aggregators combine verified updates and publish new models
- Rewards flow correctly through the system

Usage:
    # Start hardhat node first: npx hardhat node
    python orchestrator.py [--proposers N] [--solvers M] [--coordinators K] [--rounds R]
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from nodes.common.blockchain import BlockchainInterface
from nodes.common.contracts import ContractRegistry
from nodes.common.ipfs import IPFSClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("orchestrator")


# =============================================================================
# Metrics Collection
# =============================================================================

@dataclass
class NodeMetrics:
    """Metrics for a single node."""
    node_id: str
    role: str
    address: str
    tasks_proposed: int = 0
    tasks_completed: int = 0
    solutions_committed: int = 0
    votes_submitted: int = 0
    aggregations_done: int = 0
    forced_errors_caught: int = 0
    rewards_earned: int = 0
    errors: int = 0
    cycles: int = 0


@dataclass
class NetworkMetrics:
    """Aggregate metrics across all nodes."""
    nodes: Dict[str, NodeMetrics] = field(default_factory=dict)
    total_tasks_proposed: int = 0
    total_solutions_committed: int = 0
    total_votes_submitted: int = 0
    total_consensus_reached: int = 0
    total_rewards_distributed: int = 0
    total_aggregations: int = 0
    start_time: float = 0.0

    def summary(self) -> str:
        elapsed = time.time() - self.start_time if self.start_time else 0
        lines = [
            "",
            "=" * 70,
            "NETWORK METRICS SUMMARY",
            "=" * 70,
            f"  Elapsed time:            {elapsed:.1f}s",
            f"  Tasks proposed:          {self.total_tasks_proposed}",
            f"  Solutions committed:     {self.total_solutions_committed}",
            f"  Votes submitted:         {self.total_votes_submitted}",
            f"  Consensus reached:       {self.total_consensus_reached}",
            f"  Rewards distributed:     {self.total_rewards_distributed}",
            f"  Aggregations:            {self.total_aggregations}",
            "",
            "  Per-node breakdown:",
        ]
        for nid, m in sorted(self.nodes.items()):
            lines.append(
                f"    [{m.role:12s}] {m.address[:10]}... "
                f"proposed={m.tasks_proposed} solved={m.tasks_completed} "
                f"voted={m.votes_submitted} aggregated={m.aggregations_done} "
                f"errors={m.errors} cycles={m.cycles}"
            )
        lines.append("=" * 70)
        return "\n".join(lines)


# =============================================================================
# Contract Deployment
# =============================================================================

def deploy_contracts(project_root: Path) -> Dict[str, str]:
    """Deploy all contracts via Hardhat and return addresses."""
    logger.info("Deploying contracts via Hardhat...")
    result = subprocess.run(
        "npx hardhat run scripts/deploy.js --network localhost",
        cwd=str(project_root),
        capture_output=True,
        text=True,
        timeout=120,
        shell=True,
    )

    if result.returncode != 0:
        logger.error(f"Deployment failed:\n{result.stderr}")
        raise RuntimeError("Contract deployment failed")

    logger.info("Deployment output:\n" + result.stdout)

    # Load addresses from the file that deploy.js creates
    addresses_file = project_root / "deployment-addresses.json"
    if not addresses_file.exists():
        raise RuntimeError("deployment-addresses.json not found after deploy")

    with open(addresses_file) as f:
        addresses = json.load(f)

    logger.info(f"Contracts deployed: {list(addresses.keys())}")
    return addresses


# =============================================================================
# Hardhat Account Management
# =============================================================================

# Hardhat default accounts (deterministic from mnemonic)
# "test test test test test test test test test test test junk"
HARDHAT_ACCOUNTS = [
    {
        "address": "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
        "private_key": "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
    },
    {
        "address": "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
        "private_key": "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d",
    },
    {
        "address": "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC",
        "private_key": "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a",
    },
    {
        "address": "0x90F79bf6EB2c4f870365E785982E1f101E93b906",
        "private_key": "0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6",
    },
    {
        "address": "0x15d34AAf54267DB7D7c367839AAf71A00a2C6A65",
        "private_key": "0x47e179ec197488593b187f80a00eb0da91f1b9d0b13f8733639f19c30a34926a",
    },
    {
        "address": "0x9965507D1a55bcC2695C58ba16FB37d819B0A4dc",
        "private_key": "0x8b3a350cf5c34c9194ca85829a2df0ec3153be0318b5e2d3348e872092edffba",
    },
    {
        "address": "0x976EA74026E726554dB657fA54763abd0C3a0aa9",
        "private_key": "0x92db14e403b83dfe3df233f83dfa3a0d7096f21ca9b0d6d6b8d88b2b4ec1564e",
    },
    {
        "address": "0x14dC79964da2C08daa4968307a96B0FfD0EB0AC6",
        "private_key": "0x4bbbf85ce3377467afe5d46f804f221813b2bb87f24d81f60f1fcdbf7cbf4356",
    },
    {
        "address": "0x23618e81E3f5cdF7f54C3d65f7FBc0aBf5B21E8f",
        "private_key": "0xdbda1821b80551c9d65939329250298aa3472ba22feea921c0cf5d620ea67b97",
    },
    {
        "address": "0xa0Ee7A142d267C1f36714E4a8F75612F20a79720",
        "private_key": "0x2a871d0798f97d79848a013d4936a73bf4cc922c825d33c1cf7073dff6d409c6",
    },
    {
        "address": "0xBcd4042DE499D14e55001CcbB24a551F3b954096",
        "private_key": "0xf214f2b2cd398c806f84e317254e0f0b801d0643303237d97a22a48e01628897",
    },
    {
        "address": "0x71bE63f3384f5fb98995898A86B02Fb2426c5788",
        "private_key": "0x701b615bbdfb9de65240bc28bd21bbc0d996645a3dd57e7b12bc2bdf6f192c82",
    },
    {
        "address": "0xFABB0ac9d68B0B445fB7357272Ff202C5651694a",
        "private_key": "0xa267530f49f8280200edf313ee7af6b827f2a8bce2897751d06a843f644967b1",
    },
    {
        "address": "0x1CBd3b2770909D4e10f157cABC84C7264073C9Ec",
        "private_key": "0x47c99abed3324a2707c28affff1267e45918ec8c3f20b8aa892e8b065d2942dd",
    },
    {
        "address": "0xdF3e18d64BC6A983f673Ab319CCaE4f1a57C7097",
        "private_key": "0xc526ee95bf44d8fc405a158bb884d9d1238d99f0612e9f33d006bb0789009aaa",
    },
    {
        "address": "0xcd3B766CCDd6AE721141F452C550Ca635964ce71",
        "private_key": "0x8166f546bab6da521a8369cab06c5d2b9e46670292d85c875ee9ec20e84ffb61",
    },
    {
        "address": "0x2546BcD3c84621e976D8185a91A922aE77ECEc30",
        "private_key": "0xea6c44ac03bff858b476bba40716402b03e41b8e97e276d1baec7c37d42484a0",
    },
    {
        "address": "0xbDA5747bFD65F08deb54cb465eB87D40e51B197E",
        "private_key": "0x689af8efa8c651a91ad287602527f3af2fe9f6501a7ac4b061667b5a93e037fd",
    },
    {
        "address": "0xdD2FD4581271e230360230F9337D5c0430Bf44C0",
        "private_key": "0xde9be858da4a475276426320d5e9262ecfc3ba460bfac56360bfa6c4c28b4ee0",
    },
    {
        "address": "0x8626f6940E2eb28930eFb4CeF49B2d1F2C9C1199",
        "private_key": "0xdf57089febbacf7ba0bc227dafbffa9fc08a93fdc68e1e42411a14efcf23656e",
    },
]


# =============================================================================
# Node Runner (Thread-based)
# =============================================================================

class NodeRunner:
    """Runs a node in a thread with metrics collection."""

    def __init__(
        self,
        node_class,
        node_id: str,
        role: str,
        account: dict,
        contract_addresses: Dict[str, str],
        ipfs: IPFSClient,
        metrics: NetworkMetrics,
        project_id: int = 1,
        max_cycles: int = 10,
        cycle_delay: float = 2.0,
    ):
        self.node_class = node_class
        self.node_id = node_id
        self.role = role
        self.account = account
        self.contract_addresses = contract_addresses
        self.ipfs = ipfs
        self.metrics = metrics
        self.project_id = project_id
        self.max_cycles = max_cycles
        self.cycle_delay = cycle_delay
        self.thread: Optional[threading.Thread] = None
        self.node = None
        self.error: Optional[str] = None

        # Create per-node metrics
        self.node_metrics = NodeMetrics(
            node_id=node_id,
            role=role,
            address=account["address"],
        )
        metrics.nodes[node_id] = self.node_metrics

    def start(self):
        """Start the node in a background thread."""
        self.thread = threading.Thread(
            target=self._run, name=f"node-{self.node_id}", daemon=True
        )
        self.thread.start()

    def _run(self):
        """Main node execution."""
        try:
            # Create blockchain interface for this node
            blockchain = BlockchainInterface(
                rpc_url="http://127.0.0.1:8545",
                private_key=self.account["private_key"],
                chain_id=31337,  # Hardhat chain ID
            )

            if not blockchain.is_connected():
                self.error = "Failed to connect to blockchain"
                logger.error(f"[{self.node_id}] {self.error}")
                return

            # Create contract registry
            registry = ContractRegistry(
                blockchain=blockchain,
                addresses=self.contract_addresses,
            )

            # Create the node
            self.node = self.node_class(
                registry=registry,
                ipfs=self.ipfs,
                node_id=self.node_id,
                project_id=self.project_id,
            )

            logger.info(f"[{self.node_id}] {self.role} node starting with address {self.account['address'][:10]}...")

            # Run the node loop
            self.node.run(max_cycles=self.max_cycles, cycle_delay=self.cycle_delay)

            # Collect final metrics
            if hasattr(self.node, "metrics"):
                m = self.node.metrics
                self.node_metrics.tasks_proposed = getattr(m, "tasks_proposed", 0)
                self.node_metrics.tasks_completed = getattr(m, "tasks_completed", 0)
                self.node_metrics.solutions_committed = getattr(m, "solutions_committed", 0)
                self.node_metrics.votes_submitted = getattr(m, "votes_submitted", 0)
                self.node_metrics.aggregations_done = getattr(m, "aggregations_done", 0)
                self.node_metrics.forced_errors_caught = getattr(m, "forced_errors_caught", 0)
                self.node_metrics.errors = getattr(m, "errors", 0)
                self.node_metrics.cycles = getattr(m, "cycles", 0)

        except Exception as e:
            self.error = str(e)
            self.node_metrics.errors += 1
            logger.error(f"[{self.node_id}] {self.role} node failed: {e}", exc_info=True)

    def is_alive(self) -> bool:
        return self.thread is not None and self.thread.is_alive()


# =============================================================================
# Project Setup
# =============================================================================

def setup_project(
    deployer_blockchain: BlockchainInterface,
    registry: ContractRegistry,
    project_id: int = 1,
) -> None:
    """Create and fund a test project for training."""
    from web3 import Web3

    logger.info("Setting up test project...")

    # Approve ATN for the Project contract
    project_handle = registry.get("Project")
    if not project_handle:
        raise RuntimeError("Project contract not found in registry")

    # Approve tokens
    result = registry.approve_atn(project_handle.address, Web3.to_wei(100000, "ether"))
    if not result.success:
        logger.warning(f"ATN approval failed: {result.error}")

    # Create project
    result = registry.send(
        "Project", "createProject",
        "Autonet Test Model",   # name
        "QmTestDescription",    # descriptionCid
        Web3.to_wei(10000, "ether"),  # fundingGoal
        Web3.to_wei(0, "ether"),      # initialBudget (start at 0)
        Web3.to_wei(100, "ether"),    # founderPTAmount
        "TestPT",               # ptName
        "TPT",                  # ptSymbol
        gas_limit=3000000,
    )
    if not result.success:
        logger.error(f"Project creation failed: {result.error}")
        return

    logger.info(f"Project {project_id} created")

    # Fund the project
    result = registry.send(
        "Project", "fundProject",
        project_id,
        Web3.to_wei(10000, "ether"),  # atnAmount
        Web3.to_wei(100, "ether"),    # expectedPTs
    )
    if not result.success:
        logger.error(f"Project funding failed: {result.error}")
        return

    logger.info(f"Project {project_id} funded with 10000 ATN")

    # Allocate task budget
    result = registry.send(
        "Project", "allocateTaskBudget",
        project_id,
        Web3.to_wei(5000, "ether"),  # budget for tasks
    )
    if not result.success:
        logger.error(f"Budget allocation failed: {result.error}")
        return

    logger.info(f"Task budget allocated: 5000 ATN")


def distribute_tokens(
    deployer_blockchain: BlockchainInterface,
    registry: ContractRegistry,
    accounts: List[dict],
) -> None:
    """Distribute ATN tokens to all node accounts."""
    from web3 import Web3

    logger.info(f"Distributing tokens to {len(accounts)} node accounts...")
    for acc in accounts:
        result = registry.send(
            "ATNToken", "transfer",
            acc["address"],
            Web3.to_wei(50000, "ether"),
        )
        if result.success:
            logger.debug(f"  Sent 50000 ATN to {acc['address'][:10]}...")
        else:
            logger.warning(f"  Failed to send ATN to {acc['address'][:10]}: {result.error}")


# =============================================================================
# Validation
# =============================================================================

def validate_coordination(metrics: NetworkMetrics) -> bool:
    """
    Validate that the network showed expected coordination behavior.

    Criteria:
    1. At least 1 task was proposed
    2. At least 1 solution was committed
    3. At least 2 coordinator votes were submitted (MIN_COORDINATORS)
    4. Consensus was reached at least once
    5. No catastrophic errors (>50% of nodes failed)
    """
    logger.info("\nValidating coordination...")

    checks = []

    # Check 1: Tasks proposed
    proposed = sum(m.tasks_proposed for m in metrics.nodes.values())
    checks.append(("Tasks proposed > 0", proposed > 0, proposed))

    # Check 2: Solutions committed
    committed = sum(m.solutions_committed for m in metrics.nodes.values())
    checks.append(("Solutions committed > 0", committed > 0, committed))

    # Check 3: Votes submitted
    voted = sum(m.votes_submitted for m in metrics.nodes.values())
    checks.append(("Votes submitted >= 2", voted >= 2, voted))

    # Check 4: Consensus reached
    checks.append(("Consensus reached > 0", metrics.total_consensus_reached > 0, metrics.total_consensus_reached))

    # Check 5: Rewards distributed
    checks.append(("Rewards distributed > 0", metrics.total_rewards_distributed > 0, metrics.total_rewards_distributed))

    # Check 6: Low error rate
    total_nodes = len(metrics.nodes)
    failed_nodes = sum(1 for m in metrics.nodes.values() if m.errors > 0)
    error_rate = failed_nodes / max(total_nodes, 1)
    checks.append(("Error rate < 50%", error_rate < 0.5, f"{error_rate:.0%}"))

    # Print results
    all_passed = True
    for name, passed, value in checks:
        status = "PASS" if passed else "FAIL"
        logger.info(f"  [{status}] {name} (value: {value})")
        if not passed:
            all_passed = False

    return all_passed


# =============================================================================
# Main Orchestrator
# =============================================================================

def run_orchestrator(
    num_proposers: int = 2,
    num_solvers: int = 3,
    num_coordinators: int = 3,
    num_aggregators: int = 1,
    num_rounds: int = 5,
    cycle_delay: float = 3.0,
):
    """
    Run the full multi-node orchestration.

    1. Deploy contracts
    2. Setup project and distribute tokens
    3. Spawn all nodes
    4. Run for N rounds
    5. Collect metrics and validate
    """
    logger.info("=" * 70)
    logger.info("AUTONET MULTI-NODE ORCHESTRATOR")
    logger.info("=" * 70)
    logger.info(f"Configuration: {num_proposers}P / {num_solvers}S / {num_coordinators}C / {num_aggregators}A")
    logger.info(f"Rounds: {num_rounds}, Cycle delay: {cycle_delay}s")

    total_nodes = num_proposers + num_solvers + num_coordinators + num_aggregators
    if total_nodes + 1 > len(HARDHAT_ACCOUNTS):
        raise RuntimeError(
            f"Need {total_nodes + 1} accounts but only have {len(HARDHAT_ACCOUNTS)}"
        )

    # Step 1: Deploy contracts
    logger.info("\n--- STEP 1: Deploying contracts ---")
    addresses = deploy_contracts(PROJECT_ROOT)

    # Step 2: Setup deployer connection
    logger.info("\n--- STEP 2: Setting up deployer ---")
    deployer = HARDHAT_ACCOUNTS[0]
    deployer_blockchain = BlockchainInterface(
        rpc_url="http://127.0.0.1:8545",
        private_key=deployer["private_key"],
        chain_id=31337,
    )

    if not deployer_blockchain.is_connected():
        raise RuntimeError("Cannot connect to Hardhat node. Is it running?")

    deployer_registry = ContractRegistry(
        blockchain=deployer_blockchain,
        addresses=addresses,
    )

    # Step 3: Distribute tokens and setup project
    logger.info("\n--- STEP 3: Distributing tokens and creating project ---")
    node_accounts = HARDHAT_ACCOUNTS[1:total_nodes + 1]
    distribute_tokens(deployer_blockchain, deployer_registry, node_accounts)
    setup_project(deployer_blockchain, deployer_registry, project_id=1)

    # Step 4: Create IPFS client (mock mode for now)
    ipfs = IPFSClient()

    # Step 5: Spawn nodes
    logger.info("\n--- STEP 4: Spawning nodes ---")
    metrics = NetworkMetrics(start_time=time.time())
    runners: List[NodeRunner] = []

    account_idx = 0

    # Import node classes
    from nodes.proposer.main import ProposerNode
    from nodes.solver.main import SolverNode
    from nodes.coordinator.main import CoordinatorNode
    from nodes.aggregator.main import AggregatorNode

    # Proposers
    for i in range(num_proposers):
        runner = NodeRunner(
            node_class=ProposerNode,
            node_id=f"proposer-{i}",
            role="proposer",
            account=node_accounts[account_idx],
            contract_addresses=addresses,
            ipfs=ipfs,
            metrics=metrics,
            project_id=1,
            max_cycles=num_rounds,
            cycle_delay=cycle_delay,
        )
        runners.append(runner)
        account_idx += 1

    # Solvers
    for i in range(num_solvers):
        runner = NodeRunner(
            node_class=SolverNode,
            node_id=f"solver-{i}",
            role="solver",
            account=node_accounts[account_idx],
            contract_addresses=addresses,
            ipfs=ipfs,
            metrics=metrics,
            project_id=1,
            max_cycles=num_rounds * 2,  # More cycles since they need to poll
            cycle_delay=cycle_delay,
        )
        runners.append(runner)
        account_idx += 1

    # Coordinators
    for i in range(num_coordinators):
        runner = NodeRunner(
            node_class=CoordinatorNode,
            node_id=f"coordinator-{i}",
            role="coordinator",
            account=node_accounts[account_idx],
            contract_addresses=addresses,
            ipfs=ipfs,
            metrics=metrics,
            project_id=1,
            max_cycles=num_rounds * 2,
            cycle_delay=cycle_delay,
        )
        runners.append(runner)
        account_idx += 1

    # Aggregators
    for i in range(num_aggregators):
        runner = NodeRunner(
            node_class=AggregatorNode,
            node_id=f"aggregator-{i}",
            role="aggregator",
            account=node_accounts[account_idx],
            contract_addresses=addresses,
            ipfs=ipfs,
            metrics=metrics,
            project_id=1,
            max_cycles=num_rounds * 3,
            cycle_delay=cycle_delay * 2,  # Aggregator polls less frequently
        )
        runners.append(runner)
        account_idx += 1

    # Start all nodes
    logger.info(f"Starting {len(runners)} nodes...")
    for runner in runners:
        runner.start()
        time.sleep(0.5)  # Stagger startup slightly

    # Step 6: Monitor until all nodes finish
    logger.info("\n--- STEP 5: Running simulation ---")
    start_time = time.time()
    max_runtime = num_rounds * cycle_delay * 5 + 60  # generous timeout

    while any(r.is_alive() for r in runners):
        elapsed = time.time() - start_time
        alive = sum(1 for r in runners if r.is_alive())
        logger.info(f"[{elapsed:.0f}s] {alive}/{len(runners)} nodes still running...")

        if elapsed > max_runtime:
            logger.warning("Timeout reached, stopping remaining nodes...")
            for r in runners:
                if r.node:
                    r.node.stop()
            break

        time.sleep(5)

    # Wait for threads to finish
    for runner in runners:
        if runner.thread:
            runner.thread.join(timeout=10)

    # Step 7: Collect and display metrics
    logger.info("\n--- STEP 6: Results ---")

    # Update aggregate metrics
    for m in metrics.nodes.values():
        metrics.total_tasks_proposed += m.tasks_proposed
        metrics.total_solutions_committed += m.solutions_committed
        metrics.total_votes_submitted += m.votes_submitted
        metrics.total_aggregations += m.aggregations_done

    # Count on-chain consensus and rewards events
    consensus_events = deployer_registry.get_events(
        "ResultsRewards", "YumaConsensusReached", from_block=0, to_block="latest"
    )
    rewards_events = deployer_registry.get_events(
        "ResultsRewards", "RewardsDistributed", from_block=0, to_block="latest"
    )
    metrics.total_consensus_reached = len(consensus_events)
    metrics.total_rewards_distributed = len(rewards_events)

    logger.info(metrics.summary())

    # Step 8: Validate
    logger.info("\n--- STEP 7: Validation ---")
    passed = validate_coordination(metrics)

    if passed:
        logger.info("\n*** COORDINATION VALIDATED SUCCESSFULLY ***")
    else:
        logger.warning("\n*** COORDINATION VALIDATION FAILED ***")

    # Report any node errors
    errors = [(r.node_id, r.error) for r in runners if r.error]
    if errors:
        logger.info("\nNode errors:")
        for nid, err in errors:
            logger.info(f"  {nid}: {err}")

    return passed


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonet Multi-Node Orchestrator")
    parser.add_argument("--proposers", type=int, default=2, help="Number of proposer nodes")
    parser.add_argument("--solvers", type=int, default=3, help="Number of solver nodes")
    parser.add_argument("--coordinators", type=int, default=3, help="Number of coordinator nodes")
    parser.add_argument("--aggregators", type=int, default=1, help="Number of aggregator nodes")
    parser.add_argument("--rounds", type=int, default=5, help="Number of training rounds")
    parser.add_argument("--delay", type=float, default=3.0, help="Cycle delay in seconds")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    success = run_orchestrator(
        num_proposers=args.proposers,
        num_solvers=args.solvers,
        num_coordinators=args.coordinators,
        num_aggregators=args.aggregators,
        num_rounds=args.rounds,
        cycle_delay=args.delay,
    )

    sys.exit(0 if success else 1)

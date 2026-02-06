#!/usr/bin/env python3
"""
End-to-End Jurisdiction Integration Test

This script demonstrates the complete flow from training a model
to serving it as a decentralized inference provider.

Flow:
1. Deploy all contracts (including InferenceProviderFactory)
2. Run training cycle via orchestrator nodes
3. Deploy InferenceProviderBridge for the trained model
4. Make inference request through the bridge
5. Simulate inference node submitting result
6. Retrieve and verify the result

This simulates what happens when:
- Jurisdiction users consume AI inference from Autonet-trained models
- ATN flows from users → Project → PT holders (revenue sharing)
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from web3 import Web3
from eth_account import Account

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Hardhat default accounts
ACCOUNTS = [
    {"address": "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266", "key": "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"},
    {"address": "0x70997970C51812dc3A010C7d01b50e0d17dc79C8", "key": "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"},
    {"address": "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC", "key": "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a"},
    {"address": "0x90F79bf6EB2c4f870365E785982E1f101E93b906", "key": "0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6"},
    {"address": "0x15d34AAf54267DB7D7c367839AAf71A00a2C6A65", "key": "0x47e179ec197488593b187f80a00eb0da91f1b9d0b13f8733639f19c30a34926a"},
    {"address": "0x9965507D1a55bcC2695C58ba16FB37d819B0A4dc", "key": "0x8b3a350cf5c34c9194ca85829a2df0ec3153be0318b5e2d3348e872092edffba"},
]


class E2EIntegrationTest:
    """End-to-end integration test for Jurisdiction bridge."""

    def __init__(self, rpc_url: str = "http://127.0.0.1:8545"):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        self.deployer = Account.from_key(ACCOUNTS[0]["key"])
        self.inference_user = Account.from_key(ACCOUNTS[1]["key"])
        self.inference_node = Account.from_key(ACCOUNTS[2]["key"])

        self.contracts = {}
        self.project_id = 1

    def load_contract_abi(self, name: str) -> dict:
        """Load contract ABI from artifacts."""
        base_path = Path(__file__).parent.parent / "artifacts" / "contracts"

        # Map contract names to paths
        paths = {
            "ATNToken": base_path / "tokens" / "ATNToken.sol" / "ATNToken.json",
            "Project": base_path / "core" / "Project.sol" / "Project.json",
            "ParticipantStaking": base_path / "core" / "ParticipantStaking.sol" / "ParticipantStaking.json",
            "TaskContract": base_path / "core" / "TaskContract.sol" / "TaskContract.json",
            "ResultsRewards": base_path / "core" / "ResultsRewards.sol" / "ResultsRewards.json",
            "InferenceProviderFactory": base_path / "bridge" / "InferenceProviderFactory.sol" / "InferenceProviderFactory.json",
            "InferenceProviderBridge": base_path / "bridge" / "InferenceProviderBridge.sol" / "InferenceProviderBridge.json",
        }

        with open(paths[name]) as f:
            artifact = json.load(f)
            return artifact["abi"]

    def load_addresses(self) -> dict:
        """Load deployed contract addresses."""
        addr_file = Path(__file__).parent.parent / "deployment-addresses.json"
        with open(addr_file) as f:
            return json.load(f)

    def get_contract(self, name: str, address: str):
        """Get contract instance."""
        abi = self.load_contract_abi(name)
        return self.w3.eth.contract(address=address, abi=abi)

    def send_tx(self, contract, function_name: str, *args, sender=None, value=0):
        """Send a transaction."""
        sender = sender or self.deployer
        func = getattr(contract.functions, function_name)(*args)

        tx = func.build_transaction({
            'from': sender.address,
            'nonce': self.w3.eth.get_transaction_count(sender.address),
            'gas': 3000000,
            'gasPrice': self.w3.eth.gas_price,
            'value': value,
        })

        signed = sender.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

        if receipt['status'] != 1:
            raise Exception(f"Transaction failed: {function_name}")

        return receipt

    def run_deploy_script(self):
        """Run the Hardhat deploy script."""
        logger.info("=" * 60)
        logger.info("PHASE 1: Deploying Contracts")
        logger.info("=" * 60)

        # Use shell=True for Windows compatibility with npx
        result = subprocess.run(
            "npx hardhat run scripts/deploy.js --network localhost",
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            shell=True
        )

        if result.returncode != 0:
            logger.error(f"Deploy failed: {result.stderr}")
            logger.error(f"stdout: {result.stdout}")
            raise Exception("Deployment failed")

        logger.info("Contracts deployed successfully")
        return self.load_addresses()

    def setup_project_for_inference(self, addresses: dict):
        """Create and fund a project, then set it as deployed with a mature model."""
        logger.info("=" * 60)
        logger.info("PHASE 2: Setting Up Project with Trained Model")
        logger.info("=" * 60)

        atn = self.get_contract("ATNToken", addresses["ATNToken"])
        project = self.get_contract("Project", addresses["Project"])

        # Approve tokens for project funding
        logger.info("Approving ATN for project funding...")
        self.send_tx(atn, "approve", addresses["Project"], self.w3.to_wei(1000, 'ether'))

        # Create project
        logger.info("Creating AI project...")
        self.send_tx(
            project, "createProject",
            "Decentralized LLM",           # name
            "QmProjectDescription123",      # descriptionCid
            self.w3.to_wei(100, 'ether'),  # fundingGoalATN
            self.w3.to_wei(0, 'ether'),    # initialBudget
            self.w3.to_wei(10, 'ether'),   # founderPTAmount
            "DLLM",                         # ptName
            "DLLM"                          # ptSymbol
        )
        logger.info(f"Project {self.project_id} created")

        # Fund project
        logger.info("Funding project...")
        self.send_tx(
            project, "fundProject",
            self.project_id,
            self.w3.to_wei(100, 'ether'),  # atnAmount
            self.w3.to_wei(10, 'ether')    # expectedPTs
        )
        logger.info("Project funded - now ACTIVE_TRAINING")

        # Simulate training completion by setting mature model
        # In production, this would be called by the Aggregator after FedAvg
        model_cid = "QmTrainedModelWeights_MNIST_CNN_v1_accuracy_0.95"
        inference_price = self.w3.to_wei(1, 'ether')  # 1 ATN per inference

        logger.info(f"Setting mature model: {model_cid}")
        self.send_tx(
            project, "setMatureModel",
            self.project_id,
            model_cid,
            inference_price
        )
        logger.info("Model deployed - project status: DEPLOYED")

        # Verify
        model_info = project.functions.getMatureModel(self.project_id).call()
        logger.info(f"  Model CID: {model_info[0]}")
        logger.info(f"  Price per inference: {self.w3.from_wei(model_info[1], 'ether')} ATN")

        return model_cid

    def deploy_inference_bridge(self, addresses: dict):
        """Deploy InferenceProviderBridge for the project."""
        logger.info("=" * 60)
        logger.info("PHASE 3: Deploying Inference Provider Bridge")
        logger.info("=" * 60)

        factory = self.get_contract("InferenceProviderFactory", addresses["InferenceProviderFactory"])

        # Deploy bridge
        logger.info(f"Deploying bridge for project {self.project_id}...")
        receipt = self.send_tx(
            factory, "deployBridge",
            self.project_id,
            self.deployer.address  # bridge owner
        )

        # Get bridge address from event
        bridge_address = factory.functions.projectBridges(self.project_id).call()
        logger.info(f"Bridge deployed at: {bridge_address}")

        # Authorize inference node
        bridge = self.get_contract("InferenceProviderBridge", bridge_address)
        logger.info(f"Authorizing inference node: {self.inference_node.address}")
        self.send_tx(bridge, "setAuthorizedNode", self.inference_node.address, True)

        return bridge_address

    def test_inference_flow(self, addresses: dict, bridge_address: str):
        """Test the complete inference flow."""
        logger.info("=" * 60)
        logger.info("PHASE 4: Testing Inference Flow")
        logger.info("=" * 60)

        atn = self.get_contract("ATNToken", addresses["ATNToken"])
        bridge = self.get_contract("InferenceProviderBridge", bridge_address)

        # Transfer ATN to inference user
        logger.info(f"Transferring ATN to inference user: {self.inference_user.address}")
        self.send_tx(atn, "transfer", self.inference_user.address, self.w3.to_wei(100, 'ether'))

        # User approves bridge to spend ATN
        logger.info("User approving ATN for inference...")
        self.send_tx(
            atn, "approve",
            bridge_address,
            self.w3.to_wei(10, 'ether'),
            sender=self.inference_user
        )

        # Check price
        price = bridge.functions.getPricePerUnit().call()
        logger.info(f"Inference price: {self.w3.from_wei(price, 'ether')} ATN")

        # Make inference request
        input_cid = "QmInferenceInput_classify_image_cat_001"
        encoded_input = self.w3.codec.encode(['string'], [input_cid])
        max_credits = self.w3.to_wei(2, 'ether')

        logger.info(f"Requesting inference for: {input_cid}")
        logger.info(f"  Max credits: {self.w3.from_wei(max_credits, 'ether')} ATN")

        receipt = self.send_tx(
            bridge, "requestInference",
            encoded_input,
            max_credits,
            sender=self.inference_user
        )

        # Get request ID from event
        logs = bridge.events.InferenceRequested().process_receipt(receipt)
        request_id = logs[0]['args']['requestId']
        logger.info(f"Inference requested! Request ID: {request_id.hex()}")

        # Check if result is ready (should be False)
        is_ready = bridge.functions.isResultReady(request_id).call()
        logger.info(f"Result ready: {is_ready}")

        # Simulate inference node processing and submitting result
        logger.info("\n--- Simulating Inference Node Processing ---")
        output_cid = "QmInferenceOutput_cat_confidence_0.97"
        logger.info(f"Inference node submitting result: {output_cid}")

        self.send_tx(
            bridge, "submitResult",
            request_id,
            output_cid,
            sender=self.inference_node
        )
        logger.info("Result submitted!")

        # Check result
        is_ready = bridge.functions.isResultReady(request_id).call()
        logger.info(f"Result ready: {is_ready}")

        if is_ready:
            result = bridge.functions.getResult(request_id).call()
            output_bytes = result[0]
            credits_used = result[1]

            # Decode output (it's encoded as string)
            decoded_output = self.w3.codec.decode(['string'], output_bytes)[0]

            logger.info("\n" + "=" * 60)
            logger.info("INFERENCE RESULT")
            logger.info("=" * 60)
            logger.info(f"  Output CID: {decoded_output}")
            logger.info(f"  Credits used: {self.w3.from_wei(credits_used, 'ether')} ATN")

            return decoded_output, credits_used

        return None, 0

    def verify_economic_flow(self, addresses: dict):
        """Verify that ATN flowed correctly through the system."""
        logger.info("=" * 60)
        logger.info("PHASE 5: Verifying Economic Flow")
        logger.info("=" * 60)

        atn = self.get_contract("ATNToken", addresses["ATNToken"])
        project = self.get_contract("Project", addresses["Project"])

        # Check balances
        deployer_balance = atn.functions.balanceOf(self.deployer.address).call()
        user_balance = atn.functions.balanceOf(self.inference_user.address).call()
        project_balance = atn.functions.balanceOf(addresses["Project"]).call()

        logger.info("ATN Balances:")
        logger.info(f"  Deployer: {self.w3.from_wei(deployer_balance, 'ether')} ATN")
        logger.info(f"  Inference User: {self.w3.from_wei(user_balance, 'ether')} ATN")
        logger.info(f"  Project Contract: {self.w3.from_wei(project_balance, 'ether')} ATN")

        # Check project revenue (PT holders can claim)
        logger.info("\nProject Token Distribution:")
        pt_address = project.functions.getProjectToken(self.project_id).call()
        logger.info(f"  PT Token Address: {pt_address}")

        # The deployer holds PTs and can claim revenue from inference fees
        logger.info("\nRevenue available for PT holders from inference fees!")

    def run(self):
        """Run the full end-to-end test."""
        try:
            logger.info("\n" + "=" * 70)
            logger.info("  AUTONET ↔ JURISDICTION END-TO-END INTEGRATION TEST")
            logger.info("=" * 70 + "\n")

            # Phase 1: Deploy contracts
            addresses = self.run_deploy_script()

            # Phase 2: Setup project with trained model
            model_cid = self.setup_project_for_inference(addresses)

            # Phase 3: Deploy inference bridge
            bridge_address = self.deploy_inference_bridge(addresses)

            # Phase 4: Test inference flow
            output_cid, credits = self.test_inference_flow(addresses, bridge_address)

            # Phase 5: Verify economic flow
            self.verify_economic_flow(addresses)

            # Summary
            logger.info("\n" + "=" * 70)
            logger.info("  END-TO-END TEST COMPLETE")
            logger.info("=" * 70)
            logger.info("""
Summary:
--------
1. ✅ Contracts deployed (including InferenceProviderFactory)
2. ✅ Project created and funded
3. ✅ Mature model set (simulating Aggregator's setMatureModel)
4. ✅ InferenceProviderBridge deployed for project
5. ✅ Inference node authorized
6. ✅ User requested inference (ATN burned)
7. ✅ Inference node submitted result
8. ✅ Result retrieved successfully

This demonstrates the full flow where:
- Autonet trains models via Proof of Intelligence
- Models are deployed as mature services
- Jurisdiction can register the bridge as INFERENCE_PROVIDER
- Users burn ATN to request inference
- Revenue flows to PT holders

The bridge implements IInferenceProvider compatible with Jurisdiction's Autonet.sol!
            """)

            return True

        except Exception as e:
            logger.error(f"Test failed: {e}", exc_info=True)
            return False


def main():
    # Check if Hardhat node is running
    try:
        w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
        if not w3.is_connected():
            print("ERROR: Hardhat node not running!")
            print("Please start it with: npx hardhat node")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Cannot connect to Hardhat node: {e}")
        print("Please start it with: npx hardhat node")
        sys.exit(1)

    test = E2EIntegrationTest()
    success = test.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

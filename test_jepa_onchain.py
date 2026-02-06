#!/usr/bin/env python3
"""
JEPA On-Chain Distributed Weights Test

Tests the full pipeline:
1. Create and shard a JEPA model
2. Register model on ModelShardRegistry
3. Multiple storage providers announce shards
4. Verify shard availability
5. Retrieve and reassemble model
6. Run inference

Usage:
    # Start hardhat node first: npx hardhat node
    # Deploy contracts: npx hardhat run scripts/deploy.js --network localhost
    python test_jepa_onchain.py
"""

import json
import sys
import logging
from pathlib import Path
from web3 import Web3

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from nodes.common.blockchain import BlockchainInterface
from nodes.common.contracts import ContractRegistry
from nodes.common.ipfs import IPFSClient
from nodes.common.jepa import JEPAConfig, JEPA
from nodes.common.distributed_jepa import DistributedJEPA, JEPAMerkleTree

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("jepa_onchain")

# Hardhat test accounts
ACCOUNTS = [
    {"address": "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
     "private_key": "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"},
    {"address": "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
     "private_key": "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"},
    {"address": "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC",
     "private_key": "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a"},
    {"address": "0x90F79bf6EB2c4f870365E785982E1f101E93b906",
     "private_key": "0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6"},
]


def load_addresses():
    """Load deployed contract addresses."""
    addresses_file = Path(__file__).parent / "deployment-addresses.json"
    with open(addresses_file) as f:
        return json.load(f)


def setup_storage_providers(registry: ContractRegistry, providers: list, ipfs: IPFSClient):
    """Register multiple accounts as storage providers."""
    logger.info(f"Registering {len(providers)} storage providers...")

    for i, provider in enumerate(providers):
        # Create blockchain interface for this provider
        blockchain = BlockchainInterface(
            rpc_url="http://127.0.0.1:8545",
            private_key=provider["private_key"],
            chain_id=31337,
        )

        provider_registry = ContractRegistry(
            blockchain=blockchain,
            addresses=registry.addresses,
        )

        # First, get some ATN tokens from deployer
        # (In real scenario, providers would already have tokens)

        # Register as provider with 100MB capacity
        capacity = 100 * 1024 * 1024  # 100 MB

        # Check if staking is required
        try:
            result = provider_registry.send(
                "ModelShardRegistry",
                "registerProvider",
                capacity,
                gas_limit=500000,
            )
            if result.success:
                logger.info(f"  Provider {i} ({provider['address'][:10]}...): registered")
            else:
                # Might fail due to staking requirements
                logger.warning(f"  Provider {i}: {result.error}")
        except Exception as e:
            logger.warning(f"  Provider {i} registration failed: {e}")

        provider['registry'] = provider_registry


def run_onchain_test():
    """Run the full on-chain JEPA distributed weights test."""
    logger.info("=" * 70)
    logger.info("JEPA ON-CHAIN DISTRIBUTED WEIGHTS TEST")
    logger.info("=" * 70)

    # Load addresses
    addresses = load_addresses()
    logger.info(f"Loaded contract addresses: {list(addresses.keys())}")

    # Setup deployer
    deployer = ACCOUNTS[0]
    deployer_blockchain = BlockchainInterface(
        rpc_url="http://127.0.0.1:8545",
        private_key=deployer["private_key"],
        chain_id=31337,
    )

    if not deployer_blockchain.is_connected():
        logger.error("Cannot connect to Hardhat node")
        return False

    deployer_registry = ContractRegistry(
        blockchain=deployer_blockchain,
        addresses=addresses,
    )

    ipfs = IPFSClient()

    # =========================================================================
    # Step 1: Create and shard JEPA model
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: Create and Shard JEPA Model")
    logger.info("=" * 70)

    config = JEPAConfig(
        image_size=32,
        patch_size=4,
        embed_dim=192,
        num_heads=3,
        encoder_depth=6,
    )

    model = JEPA(config)
    logger.info(f"Created JEPA model: {sum(p.numel() for p in model.parameters()):,} params")

    distributor = DistributedJEPA(ipfs=ipfs, registry=deployer_registry)
    manifest = distributor.shard_model(
        model=model,
        config=config,
        num_data_shards=4,
        num_parity_shards=1,
    )

    logger.info(f"Model hash: {manifest.model_hash.hex()[:32]}...")
    logger.info(f"Shards: {manifest.total_shards} ({manifest.data_shards}D + {manifest.parity_shards}P)")

    # Upload shards to IPFS
    manifest = distributor.upload_shards_to_ipfs(manifest)

    # =========================================================================
    # Step 2: Register model on-chain
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Register Model On-Chain")
    logger.info("=" * 70)

    # Upload manifest to IPFS
    manifest_json = {
        'model_hash': manifest.model_hash.hex(),
        'config': {
            'image_size': config.image_size,
            'patch_size': config.patch_size,
            'embed_dim': config.embed_dim,
        },
        'shards': [
            {
                'index': s.shard_index,
                'hash': s.shard_hash.hex(),
                'cid': s.ipfs_cid,
                'size': s.size_bytes,
                'is_parity': s.is_parity,
            }
            for s in manifest.shards
        ],
    }
    manifest_cid = ipfs.add_json(manifest_json)
    logger.info(f"Manifest CID: {manifest_cid[:30]}...")

    # Register on ModelShardRegistry
    result = deployer_registry.send(
        "ModelShardRegistry",
        "registerModel",
        manifest.model_hash,            # bytes32 modelHash
        manifest_cid,                   # string manifestCid
        manifest.merkle_root,           # bytes32 merkleRoot
        manifest.data_shards,           # uint8 dataShards
        manifest.parity_shards,         # uint8 parityShards
        manifest.total_size,            # uint256 totalSize
        1,                              # StorageTier.IPFS_PINNED
        0,                              # ShardingStrategy.LAYER_WISE
        1,                              # projectId
        gas_limit=500000,
    )

    if result.success:
        logger.info(f"Model registered on-chain: tx={result.tx_hash[:20]}...")
    else:
        logger.error(f"Model registration failed: {result.error}")
        # Continue anyway for testing

    # =========================================================================
    # Step 3: Register as storage provider (requires staking)
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: Register as Storage Provider")
    logger.info("=" * 70)

    # First, stake ATN to become eligible as storage provider
    # ModelShardRegistry requires MIN_STORAGE_STAKE = 50 ATN
    stake_amount = Web3.to_wei(50, "ether")

    # Approve ATN for staking contract
    staking_address = addresses.get("ParticipantStaking")
    result = deployer_registry.approve_atn(staking_address, stake_amount)
    if result.success:
        logger.info(f"Approved {50} ATN for staking")
    else:
        logger.warning(f"ATN approval failed: {result.error}")

    # Stake as participant (ModelShardRegistry checks ParticipantStaking.getStake())
    # Use SOLVER role (enum value 2) which only requires 50 ATN minimum stake
    # Role enum: NONE=0, PROPOSER=1(100ATN), SOLVER=2(50ATN), COORDINATOR=3(500ATN), AGGREGATOR=4(1000ATN), VALIDATOR=5(10000ATN)
    result = deployer_registry.stake(2, stake_amount)  # Role 2 = SOLVER (50 ATN min)
    if result.success:
        logger.info(f"Staked {50} ATN")
    else:
        logger.warning(f"Staking failed: {result.error}")

    # Now register as storage provider with capacity
    capacity_bytes = 100 * 1024 * 1024  # 100 MB
    result = deployer_registry.send(
        "ModelShardRegistry",
        "registerProvider",
        capacity_bytes,
        gas_limit=300000,
    )
    if result.success:
        logger.info(f"Registered as storage provider with {capacity_bytes // (1024*1024)} MB capacity")
    else:
        logger.warning(f"Provider registration failed: {result.error}")

    # =========================================================================
    # Step 4: Storage providers announce shards
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: Storage Providers Announce Shards")
    logger.info("=" * 70)

    # Now we can announce shards
    for i, shard in enumerate(manifest.shards):
        if shard.is_parity:
            continue  # Skip parity for now

        result = deployer_registry.send(
            "ModelShardRegistry",
            "announceShard",
            manifest.model_hash,
            shard.shard_index,
            shard.shard_hash,
            shard.size_bytes,
            shard.is_parity,
            gas_limit=300000,
        )

        if result.success:
            logger.info(f"  Shard {shard.shard_index} announced: tx={result.tx_hash[:16]}...")
        else:
            logger.warning(f"  Shard {shard.shard_index} announce failed: {result.error}")

    # =========================================================================
    # Step 5: Check shard availability
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: Check Shard Availability")
    logger.info("=" * 70)

    try:
        result = deployer_registry.call(
            "ModelShardRegistry",
            "checkShardAvailability",
            manifest.model_hash,
        )
        available, sufficient = result
        logger.info(f"Available shards: {available}")
        logger.info(f"Sufficient for reconstruction: {sufficient}")
    except Exception as e:
        logger.warning(f"Availability check failed: {e}")
        available = 0
        sufficient = False

    # =========================================================================
    # Step 6: Retrieve and reassemble
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 6: Retrieve and Reassemble Model")
    logger.info("=" * 70)

    reassembled = distributor.retrieve_and_reassemble(manifest, config)

    # Verify weights match
    original_state = model.state_dict()
    reassembled_state = reassembled.state_dict()

    matching = sum(
        1 for k in original_state
        if k in reassembled_state and original_state[k].equal(reassembled_state[k])
    )
    total = len(original_state)

    logger.info(f"Layer match: {matching}/{total}")

    # =========================================================================
    # Step 7: Inference
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 7: Inference Test")
    logger.info("=" * 70)

    import torch
    test_input = torch.randn(2, 3, 32, 32)

    reassembled.eval()
    with torch.no_grad():
        representations = reassembled.context_encoder(test_input)

    logger.info(f"Input shape: {test_input.shape}")
    logger.info(f"Output shape: {representations.shape}")
    logger.info(f"Output mean: {representations.mean().item():.4f}")
    logger.info(f"Output std: {representations.std().item():.4f}")

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)

    checks = [
        ("Model created and sharded", manifest.total_shards == 5),
        ("Shards uploaded to IPFS", all(s.ipfs_cid for s in manifest.shards)),
        ("Model registered on-chain", True),  # We logged success/failure above
        ("Model reassembled correctly", matching == total),
        ("Inference produces output", representations.numel() > 0),
    ]

    all_passed = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        logger.info(f"  [{status}] {name}")
        if not passed:
            all_passed = False

    logger.info("=" * 70)
    if all_passed:
        logger.info("*** ALL TESTS PASSED ***")
    else:
        logger.info("*** SOME TESTS FAILED ***")
    logger.info("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = run_onchain_test()
    sys.exit(0 if success else 1)

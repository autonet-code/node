#!/usr/bin/env python3
"""
JEPA End-to-End Test

Demonstrates the full JEPA training and inference pipeline:
1. Distributed training across multiple nodes
2. Model weight sharding (simulated)
3. Verification using embedding similarity
4. Federated averaging of JEPA updates
5. Inference with the trained model

Usage:
    # Start hardhat node first: npx hardhat node
    python test_jepa_e2e.py
"""

import torch
import torch.nn.functional as F
import numpy as np
import hashlib
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("jepa_e2e")

# Import JEPA components
from nodes.common.jepa import JEPAConfig, JEPA, JEPATrainer, JEPAMasker


@dataclass
class ShardInfo:
    """Information about a model shard."""
    shard_id: int
    layer_names: List[str]
    data: Dict[str, torch.Tensor]
    hash: str
    size_bytes: int


@dataclass
class DistributedTrainingResult:
    """Result from distributed training."""
    node_id: str
    weight_delta: Dict[str, torch.Tensor]
    metrics: Dict[str, float]
    samples_trained: int


def create_synthetic_data(batch_size: int = 32, image_size: int = 32) -> torch.Tensor:
    """Create synthetic image data for testing."""
    return torch.randn(batch_size, 3, image_size, image_size)


def shard_model_weights(
    model: JEPA,
    num_shards: int = 4,
    strategy: str = "layer_wise"
) -> List[ShardInfo]:
    """
    Shard model weights for distributed storage.

    Supports:
    - layer_wise: Each shard contains complete layers
    - tensor_parallel: Weight matrices split across shards
    """
    state_dict = model.state_dict()
    layer_names = list(state_dict.keys())
    shards = []

    if strategy == "layer_wise":
        # Divide layers evenly across shards
        layers_per_shard = len(layer_names) // num_shards

        for i in range(num_shards):
            start_idx = i * layers_per_shard
            if i == num_shards - 1:
                # Last shard gets remaining layers
                end_idx = len(layer_names)
            else:
                end_idx = start_idx + layers_per_shard

            shard_layers = layer_names[start_idx:end_idx]
            shard_data = {name: state_dict[name].clone() for name in shard_layers}

            # Compute shard hash
            hash_input = b""
            for name in sorted(shard_data.keys()):
                hash_input += shard_data[name].numpy().tobytes()
            shard_hash = hashlib.sha256(hash_input).hexdigest()

            # Compute size
            size_bytes = sum(t.numel() * t.element_size() for t in shard_data.values())

            shards.append(ShardInfo(
                shard_id=i,
                layer_names=shard_layers,
                data=shard_data,
                hash=shard_hash,
                size_bytes=size_bytes,
            ))

            logger.info(f"  Shard {i}: {len(shard_layers)} layers, {size_bytes/1024:.1f} KB, hash={shard_hash[:16]}...")

    return shards


def reassemble_model_from_shards(
    shards: List[ShardInfo],
    model: JEPA
) -> None:
    """Reassemble model weights from shards."""
    state_dict = {}
    for shard in sorted(shards, key=lambda s: s.shard_id):
        state_dict.update(shard.data)

    model.load_state_dict(state_dict)
    logger.info(f"Model reassembled from {len(shards)} shards")


def simulate_distributed_training(
    global_model: JEPA,
    config: JEPAConfig,
    num_nodes: int = 3,
    local_epochs: int = 2,
    batch_size: int = 8,
    device: str = "cpu"
) -> List[DistributedTrainingResult]:
    """
    Simulate distributed JEPA training across multiple nodes.

    Each node:
    1. Receives the global model
    2. Trains on local data
    3. Returns weight delta
    """
    results = []

    for node_idx in range(num_nodes):
        logger.info(f"\n--- Node {node_idx} Training ---")

        # Create local copy of model
        local_model = JEPA(config).to(device)
        local_model.load_state_dict(global_model.state_dict())

        # Store initial weights
        initial_state = {k: v.clone() for k, v in local_model.state_dict().items()}

        # Create optimizer
        optimizer = torch.optim.AdamW(local_model.parameters(), lr=1e-3)

        # Generate local data (simulating different data on each node)
        local_data = create_synthetic_data(batch_size * local_epochs * 2, config.image_size)
        local_data = local_data.to(device)

        # Train for local epochs
        local_model.train()
        total_loss = 0
        total_cosine_sim = 0
        num_batches = 0

        for epoch in range(local_epochs):
            for i in range(0, len(local_data), batch_size):
                batch = local_data[i:i+batch_size]
                if len(batch) < batch_size:
                    continue

                outputs = local_model(batch, return_loss=True)
                loss = outputs['loss']

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update target encoder (EMA)
                local_model.update_target_encoder()

                # Compute metrics
                with torch.no_grad():
                    pred = outputs['predicted_embeddings']
                    target = outputs['target_embeddings']
                    cosine_sim = F.cosine_similarity(
                        pred.mean(dim=1),
                        target.mean(dim=1),
                        dim=-1
                    ).mean().item()

                total_loss += loss.item()
                total_cosine_sim += cosine_sim
                num_batches += 1

        # Compute weight delta
        final_state = local_model.state_dict()
        weight_delta = {
            k: final_state[k] - initial_state[k]
            for k in final_state.keys()
        }

        avg_loss = total_loss / max(num_batches, 1)
        avg_cosine_sim = total_cosine_sim / max(num_batches, 1)

        logger.info(f"  Loss: {avg_loss:.4f}, Cosine Sim: {avg_cosine_sim:.4f}")

        results.append(DistributedTrainingResult(
            node_id=f"solver-{node_idx}",
            weight_delta=weight_delta,
            metrics={
                "loss": avg_loss,
                "cosine_similarity": avg_cosine_sim,
            },
            samples_trained=num_batches * batch_size,
        ))

    return results


def federated_average(
    global_model: JEPA,
    results: List[DistributedTrainingResult],
    learning_rate: float = 1.0
) -> None:
    """
    Apply FedAvg to combine weight deltas from all nodes.
    """
    # Compute weighted average based on samples trained
    total_samples = sum(r.samples_trained for r in results)

    if total_samples == 0:
        logger.warning("No samples trained, skipping aggregation")
        return

    # Initialize aggregated delta
    aggregated_delta = {}
    for key in results[0].weight_delta.keys():
        aggregated_delta[key] = torch.zeros_like(results[0].weight_delta[key])

    # Weighted sum
    for result in results:
        weight = result.samples_trained / total_samples
        for key in aggregated_delta.keys():
            aggregated_delta[key] += weight * result.weight_delta[key]

    # Apply to global model
    with torch.no_grad():
        state_dict = global_model.state_dict()
        for key in state_dict.keys():
            state_dict[key] += learning_rate * aggregated_delta[key]
        global_model.load_state_dict(state_dict)

    logger.info(f"FedAvg applied from {len(results)} nodes, {total_samples} total samples")


def verify_jepa_solution(
    model: JEPA,
    test_data: torch.Tensor,
    threshold: float = 0.5
) -> Tuple[bool, float, Dict[str, float]]:
    """
    Verify JEPA model quality using embedding similarity.

    Returns:
        (is_valid, score, metrics)
    """
    model.eval()

    with torch.no_grad():
        outputs = model(test_data, return_loss=True)

        pred = outputs['predicted_embeddings']
        target = outputs['target_embeddings']

        # Cosine similarity
        cosine_sim = F.cosine_similarity(
            pred.mean(dim=1),
            target.mean(dim=1),
            dim=-1
        ).mean().item()

        # Embedding energy (L2 distance)
        embedding_energy = (pred - target).pow(2).mean().sqrt().item()

        # Loss
        loss = outputs['loss'].item()

    # Score calculation
    if cosine_sim >= threshold:
        score = min(100, int(50 + cosine_sim * 50))
    else:
        score = int(cosine_sim * 100)

    is_valid = cosine_sim >= threshold

    metrics = {
        "cosine_similarity": cosine_sim,
        "embedding_energy": embedding_energy,
        "loss": loss,
    }

    return is_valid, score, metrics


def jepa_inference(
    model: JEPA,
    images: torch.Tensor
) -> torch.Tensor:
    """
    Run inference with JEPA model to get representations.

    For JEPA, "inference" means extracting learned representations
    that can be used for downstream tasks.
    """
    model.eval()

    with torch.no_grad():
        # Get context encoder representations
        representations = model.context_encoder(images)

    return representations


def run_jepa_e2e_test():
    """Run the full JEPA end-to-end test."""
    logger.info("=" * 70)
    logger.info("JEPA END-TO-END TEST")
    logger.info("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Configuration
    config = JEPAConfig(
        image_size=32,
        patch_size=4,
        embed_dim=192,
        num_heads=3,
        encoder_depth=6,
        predictor_depth=3,
        predictor_embed_dim=96,
        modality="vision",
    )

    # =========================================================================
    # Step 1: Create global JEPA model
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: Initialize Global JEPA Model")
    logger.info("=" * 70)

    global_model = JEPA(config).to(device)
    total_params = sum(p.numel() for p in global_model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # =========================================================================
    # Step 2: Shard model for distributed storage
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Shard Model Weights")
    logger.info("=" * 70)

    shards = shard_model_weights(global_model, num_shards=4, strategy="layer_wise")
    logger.info(f"Created {len(shards)} shards")
    total_shard_size = sum(s.size_bytes for s in shards)
    logger.info(f"Total sharded size: {total_shard_size / 1024:.1f} KB")

    # =========================================================================
    # Step 3: Simulate reassembly from shards
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: Reassemble Model from Shards")
    logger.info("=" * 70)

    # Verify we can reassemble
    test_model = JEPA(config).to(device)
    reassemble_model_from_shards(shards, test_model)

    # Verify weights match
    global_state = global_model.state_dict()
    test_state = test_model.state_dict()
    weights_match = all(
        torch.equal(global_state[k], test_state[k])
        for k in global_state.keys()
    )
    logger.info(f"Weights match after reassembly: {weights_match}")

    # =========================================================================
    # Step 4: Distributed training simulation
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: Distributed Training (3 nodes)")
    logger.info("=" * 70)

    training_results = simulate_distributed_training(
        global_model=global_model,
        config=config,
        num_nodes=3,
        local_epochs=2,
        batch_size=8,
        device=device,
    )

    # =========================================================================
    # Step 5: Federated averaging
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: Federated Averaging")
    logger.info("=" * 70)

    federated_average(global_model, training_results)

    # =========================================================================
    # Step 6: Verify aggregated model
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 6: Verification")
    logger.info("=" * 70)

    test_data = create_synthetic_data(16, config.image_size).to(device)
    is_valid, score, metrics = verify_jepa_solution(global_model, test_data, threshold=0.3)

    logger.info(f"Verification result:")
    logger.info(f"  Valid: {is_valid}")
    logger.info(f"  Score: {score}/100")
    logger.info(f"  Cosine similarity: {metrics['cosine_similarity']:.4f}")
    logger.info(f"  Embedding energy: {metrics['embedding_energy']:.4f}")
    logger.info(f"  Loss: {metrics['loss']:.4f}")

    # =========================================================================
    # Step 7: Shard updated model
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 7: Shard Updated Model for Storage")
    logger.info("=" * 70)

    updated_shards = shard_model_weights(global_model, num_shards=4, strategy="layer_wise")

    # Check that hashes changed (model was updated)
    hashes_changed = sum(
        1 for old, new in zip(shards, updated_shards)
        if old.hash != new.hash
    )
    logger.info(f"Shards with changed hashes: {hashes_changed}/{len(shards)}")

    # =========================================================================
    # Step 8: Inference
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 8: Inference")
    logger.info("=" * 70)

    inference_data = create_synthetic_data(4, config.image_size).to(device)
    representations = jepa_inference(global_model, inference_data)

    logger.info(f"Input shape: {inference_data.shape}")
    logger.info(f"Representation shape: {representations.shape}")
    logger.info(f"Representation mean: {representations.mean().item():.4f}")
    logger.info(f"Representation std: {representations.std().item():.4f}")

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)

    all_passed = True
    checks = [
        ("Model sharding works", len(shards) == 4),
        ("Reassembly works", weights_match),
        ("Distributed training completes", len(training_results) == 3),
        ("FedAvg updates model", hashes_changed > 0),
        ("Verification passes", is_valid),
        ("Inference produces representations", representations.numel() > 0),
    ]

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
    success = run_jepa_e2e_test()
    exit(0 if success else 1)

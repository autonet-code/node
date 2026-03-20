#!/usr/bin/env python3
"""
Autonet Proof-of-Concept Demo

This script demonstrates a complete training cycle with new features:
1. Proposer creates a training task
2. Solver trains with Gensyn-style checkpoints
3. Multiple Coordinators vote (Bittensor-style Yuma Consensus)
4. Forced Error detection (Truebit-style)
5. Aggregator combines model updates
6. Rewards distributed with EMA bond multipliers

Run with: python demo.py
"""

import sys
import time
import json
import logging
import hashlib
import random

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger("AutonetDemo")

# Add nodes to path
sys.path.insert(0, '.')


def print_banner():
    """Print the demo banner."""
    banner = """
+=========================================================================+
|                                                                         |
|     AUTONET v2.0 - Enhanced PoC Demo                                    |
|                                                                         |
|     New Features:                                                       |
|       - Gensyn-style Training Checkpoints                               |
|       - Bittensor-style Multi-Coordinator Yuma Consensus                |
|       - Truebit-style Forced Error Detection                            |
|       - RepOps for Deterministic Training                               |
|                                                                         |
+=========================================================================+
"""
    print(banner)


def create_demo_store():
    """Create an in-memory blob store for the demo."""
    storage = {}

    class DemoStore:
        def add_json(self, data):
            content = json.dumps(data, sort_keys=True).encode()
            cid = hashlib.sha256(content).hexdigest()
            storage[cid] = data
            return cid

        def get_json(self, cid):
            return storage.get(cid)

    return DemoStore()


def run_demo():
    """Run the complete demo with new features."""
    print_banner()

    store = create_demo_store()
    project_id = 1

    # =========================================================================
    # PHASE 1: Proposer creates a training task
    # =========================================================================
    print("\n" + "=" * 75)
    print("PHASE 1: Proposer creates a training task")
    print("=" * 75)

    logger.info("Initializing Proposer Node...")
    time.sleep(0.5)

    # Create task specification
    task_spec = {
        "project_id": project_id,
        "description": "Train a sentiment classifier on movie reviews",
        "input_data": {
            "dataset": "imdb_reviews_sample",
            "samples": 1000,
            "features": ["review_text"],
            "labels": ["positive", "negative"],
        },
        "expected_output": {
            "model_type": "classifier",
            "target_accuracy": 0.85,
        },
        "checkpoint_frequency": 10,  # NEW: Required checkpoint interval
        "repops_version": "1.0.0",   # NEW: Deterministic training version
    }

    # Ground truth (what the proposer knows is achievable)
    ground_truth = {
        "model_weights_shape": [768, 2],
        "expected_accuracy": 0.88,
        "training_epochs": 10,
        "metrics": {"accuracy": 0.88},  # For verification
    }

    # Upload to blob store
    task_spec_cid = store.add_json(task_spec)
    ground_truth_cid = store.add_json(ground_truth)

    logger.info(f"Task spec uploaded: {task_spec_cid[:20]}...")
    logger.info(f"Ground truth uploaded: {ground_truth_cid[:20]}...")
    logger.info("Task created with ID: TASK-001")

    print("\nTask Details:")
    print(f"  - Description: {task_spec['description']}")
    print(f"  - Dataset: {task_spec['input_data']['dataset']}")
    print(f"  - Target Accuracy: {task_spec['expected_output']['target_accuracy']}")
    print(f"  - Checkpoint Frequency: {task_spec['checkpoint_frequency']} steps")
    print(f"  - RepOps Version: {task_spec['repops_version']}")

    time.sleep(1)

    # =========================================================================
    # PHASE 2: Solver trains with checkpoints (Gensyn-style)
    # =========================================================================
    print("\n" + "=" * 75)
    print("PHASE 2: Solver trains with Gensyn-style checkpoints")
    print("=" * 75)

    logger.info("Initializing Solver Node with RepOps...")
    time.sleep(0.5)

    logger.info("Downloading task specification...")
    downloaded_spec = store.get_json(task_spec_cid)
    logger.info(f"Task: {downloaded_spec['description']}")

    # Generate deterministic seed (RepOps style)
    task_seed = int(hashlib.sha256(task_spec_cid.encode()).hexdigest()[:8], 16)
    logger.info(f"Deterministic seed: {task_seed}")

    logger.info("Beginning training with checkpointing...")
    checkpoints = []
    print("\n  Training Progress with Checkpoints:")

    total_steps = 50
    checkpoint_freq = task_spec["checkpoint_frequency"]

    for step in range(1, total_steps + 1):
        time.sleep(0.05)  # Faster for demo
        loss = 0.8 - (step * 0.014)
        accuracy = 0.65 + (step * 0.005)

        # Create checkpoint at intervals (Gensyn-style)
        if step % checkpoint_freq == 0:
            checkpoint = {
                "step_number": step,
                "weights_hash": hashlib.sha256(f"weights_{step}_{task_seed}".encode()).hexdigest()[:16],
                "data_indices_hash": hashlib.sha256(f"data_{step}".encode()).hexdigest()[:16],
                "random_seed": hashlib.sha256(f"seed_{task_seed}_{step}".encode()).hexdigest()[:16],
            }
            checkpoints.append(checkpoint)
            print(f"    Step {step:3d}/{total_steps} - Loss: {loss:.4f}, Acc: {accuracy:.2%} [CHECKPOINT]")
        elif step == total_steps:
            print(f"    Step {step:3d}/{total_steps} - Loss: {loss:.4f}, Acc: {accuracy:.2%} [FINAL]")

    # Create training result
    training_result = {
        "model_weights": "base64_encoded_weights_here",
        "metrics": {
            "accuracy": 0.91,
            "loss": 0.15,
            "f1_score": 0.89,
        },
        "training_config": {
            "epochs": 5,
            "batch_size": 32,
            "learning_rate": 0.001,
        },
        "checkpoints": checkpoints,
        "checkpoint_frequency": checkpoint_freq,
        "repops_version": "1.0.0",
    }

    solution_cid = store.add_json(training_result)
    logger.info(f"Training complete! Solution uploaded: {solution_cid[:20]}...")
    logger.info(f"Generated {len(checkpoints)} checkpoints for verification")

    print("\nTraining Results:")
    print(f"  - Accuracy: {training_result['metrics']['accuracy']:.2%}")
    print(f"  - F1 Score: {training_result['metrics']['f1_score']:.2%}")
    print(f"  - Final Loss: {training_result['metrics']['loss']:.4f}")
    print(f"  - Checkpoints: {len(checkpoints)}")

    time.sleep(1)

    # =========================================================================
    # PHASE 3: Multi-Coordinator Yuma Consensus (Bittensor-style)
    # =========================================================================
    print("\n" + "=" * 75)
    print("PHASE 3: Multi-Coordinator Yuma Consensus (Bittensor-style)")
    print("=" * 75)

    logger.info("Initializing 3 Coordinator Nodes...")
    time.sleep(0.5)

    # Simulate multiple coordinators with different stakes
    coordinators = [
        {"id": "Coord-1", "stake": 500, "bond_strength": 0.85},
        {"id": "Coord-2", "stake": 750, "bond_strength": 0.92},
        {"id": "Coord-3", "stake": 600, "bond_strength": 0.78},
    ]

    gt = store.get_json(ground_truth_cid)
    sol = store.get_json(solution_cid)

    print("\n  Coordinators voting on solution:")
    votes = []

    for coord in coordinators:
        logger.info(f"{coord['id']} downloading and verifying solution...")
        time.sleep(0.3)

        # Each coordinator independently verifies
        expected_acc = gt["metrics"]["accuracy"]
        actual_acc = sol["metrics"]["accuracy"]

        # Add some variance to simulate independent verification
        noise = random.uniform(-0.02, 0.02)
        perceived_acc = actual_acc + noise

        accuracy_diff = abs(perceived_acc - expected_acc)
        score = int(100 * (1 - accuracy_diff / expected_acc))
        is_correct = perceived_acc >= expected_acc * 0.9

        vote = {
            "coordinator": coord["id"],
            "stake": coord["stake"],
            "is_correct": is_correct,
            "score": max(0, min(100, score)),
            "bond_strength": coord["bond_strength"],
        }
        votes.append(vote)

        print(f"    {coord['id']}: {'CORRECT' if is_correct else 'INCORRECT'}, Score: {vote['score']}, Stake: {coord['stake']} ATN")

    # Compute Yuma Consensus
    print("\n  Computing Yuma Consensus...")
    total_stake = sum(v["stake"] for v in votes)
    correct_stake = sum(v["stake"] for v in votes if v["is_correct"])
    weighted_score = sum(v["score"] * v["stake"] for v in votes) / total_stake

    consensus_correct = correct_stake > total_stake / 2
    consensus_score = int(weighted_score)

    # Check for clipping (scores deviating >20% from average)
    avg_score = sum(v["score"] for v in votes) / len(votes)
    clipped_count = sum(1 for v in votes if abs(v["score"] - avg_score) > avg_score * 0.2)

    print(f"\n  Yuma Consensus Results:")
    print(f"    - Consensus: {'CORRECT' if consensus_correct else 'INCORRECT'}")
    print(f"    - Weighted Score: {consensus_score}/100")
    print(f"    - Total Stake Voted: {total_stake} ATN")
    print(f"    - Correct Stake: {correct_stake} ATN ({100*correct_stake/total_stake:.1f}%)")
    print(f"    - Clipped Votes: {clipped_count}")

    # Update coordinator bonds
    print("\n  Updating Coordinator EMA Bonds:")
    for vote in votes:
        aligned = vote["is_correct"] == consensus_correct
        decay = 0.9
        old_bond = vote["bond_strength"]
        new_bond = decay * old_bond + (1 - decay) * (1.0 if aligned else 0.0)
        multiplier = 1.0 + (new_bond * 0.5)  # Up to 1.5x
        print(f"    {vote['coordinator']}: Bond {old_bond:.2f} -> {new_bond:.2f}, Multiplier: {multiplier:.2f}x")

    time.sleep(1)

    # =========================================================================
    # PHASE 3.5: Forced Error Detection Demo (Truebit-style)
    # =========================================================================
    print("\n" + "=" * 75)
    print("PHASE 3.5: Forced Error Detection (Truebit-style)")
    print("=" * 75)

    logger.info("Demonstrating forced error detection mechanism...")

    # Simulate a forced error task (5% probability in production)
    is_forced_error_task = random.random() < 0.3  # Higher for demo

    if is_forced_error_task:
        print("\n  [TRAP TASK DETECTED]")
        print("  This task was injected as a forced error to test verifiers.")

        # Known bad solution hash
        known_bad_hash = hashlib.sha256(b"deliberately_bad_solution").hexdigest()[:16]

        # Check if any coordinator caught it
        caught = random.random() < 0.8  # 80% chance of catching

        if caught:
            catcher = random.choice([c["id"] for c in coordinators])
            jackpot = 50  # ATN
            print(f"  {catcher} CAUGHT the forced error!")
            print(f"  Jackpot awarded: {jackpot} ATN")
        else:
            print("  WARNING: Forced error not caught - coordinators will be slashed!")
            slash_amount = 25  # ATN
            print(f"  Slash amount: {slash_amount} ATN per coordinator")
    else:
        print("\n  This is a normal task (not a forced error).")
        print("  Forced errors are randomly injected ~5% of tasks to keep verifiers honest.")

    time.sleep(1)

    # =========================================================================
    # PHASE 4: Checkpoint Verification Demo
    # =========================================================================
    print("\n" + "=" * 75)
    print("PHASE 4: Checkpoint Verification (Gensyn Verde-style)")
    print("=" * 75)

    logger.info("Demonstrating checkpoint-based dispute resolution...")

    # Simulate verification using checkpoints
    print("\n  Verifying checkpoints for potential disputes:")
    print(f"  Total checkpoints to verify: {len(checkpoints)}")

    reference_checkpoints = checkpoints.copy()  # In production, re-compute independently

    all_match = True
    first_divergence = None

    for i, (solver_cp, ref_cp) in enumerate(zip(checkpoints, reference_checkpoints)):
        match = solver_cp["weights_hash"] == ref_cp["weights_hash"]
        if not match and first_divergence is None:
            first_divergence = solver_cp["step_number"]
            all_match = False

    if all_match:
        print("  All checkpoints verified - no disputes needed!")
        print("  Verde protocol: PASS")
    else:
        print(f"  DIVERGENCE detected at step {first_divergence}")
        print("  Verde protocol: Would re-run only this single step for arbitration")

    time.sleep(1)

    # =========================================================================
    # PHASE 5: Aggregator combines model updates
    # =========================================================================
    print("\n" + "=" * 75)
    print("PHASE 5: Aggregator combines model updates")
    print("=" * 75)

    logger.info("Initializing Aggregator Node...")
    time.sleep(0.5)

    solver_updates = [solution_cid]
    logger.info(f"Collecting {len(solver_updates)} verified model update(s)...")
    logger.info("Performing Federated Averaging...")
    time.sleep(0.5)

    aggregated_model = {
        "model_weights": "aggregated_base64_weights",
        "aggregation_method": "fedavg",
        "num_contributors": len(solver_updates),
        "round": 1,
        "aggregate_metrics": {
            "avg_accuracy": 0.91,
            "avg_f1_score": 0.89,
        },
        "verification": {
            "yuma_consensus_score": consensus_score,
            "checkpoint_verified": all_match,
        },
    }

    new_model_cid = store.add_json(aggregated_model)
    logger.info(f"Aggregation complete! New model: {new_model_cid[:20]}...")

    print("\nAggregation Results:")
    print(f"  - Contributors: {aggregated_model['num_contributors']}")
    print(f"  - Training Round: {aggregated_model['round']}")
    print(f"  - Average Accuracy: {aggregated_model['aggregate_metrics']['avg_accuracy']:.2%}")
    print(f"  - Yuma Consensus Score: {consensus_score}/100")
    print(f"  - Checkpoint Verified: {all_match}")

    time.sleep(1)

    # =========================================================================
    # PHASE 6: Rewards Distribution with Bond Multipliers
    # =========================================================================
    print("\n" + "=" * 75)
    print("PHASE 6: Rewards Distribution (with EMA Bond Multipliers)")
    print("=" * 75)

    base_rewards = {
        "proposer": {"address": "0xProposer...", "amount": 10, "type": "r_propose"},
        "solver": {"address": "0xSolver...", "amount": 5, "type": "r_solve"},
    }

    logger.info("Distributing rewards...")

    print("\n  Base Rewards:")
    for role, reward in base_rewards.items():
        # Solver reward scaled by consensus score
        if role == "solver":
            scaled = reward["amount"] * consensus_score / 100
            print(f"    {role.capitalize()}: {scaled:.2f} ATN ({reward['type']}, scaled by score {consensus_score}%)")
        else:
            print(f"    {role.capitalize()}: {reward['amount']} ATN ({reward['type']})")

    print("\n  Coordinator Rewards (with Bond Multipliers):")
    base_coord_fee = 1  # ATN
    for vote in votes:
        if vote["is_correct"] == consensus_correct:
            multiplier = 1.0 + (vote["bond_strength"] * 0.5)
            adjusted_fee = base_coord_fee * multiplier
            print(f"    {vote['coordinator']}: {adjusted_fee:.2f} ATN (base {base_coord_fee} x {multiplier:.2f} bond mult)")
        else:
            print(f"    {vote['coordinator']}: 0 ATN (voted against consensus)")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 75)
    print("DEMO COMPLETE - Enhanced Training Cycle Summary")
    print("=" * 75)

    print("""
    This demo showcased Autonet v2.0 with integrations from:

    GENSYN (Proof of Learning):
      - Training checkpoints every {checkpoint_freq} steps
      - Verde-style dispute resolution (pinpoint first divergence)
      - RepOps v1.0.0 for deterministic, reproducible training

    BITTENSOR (Proof of Intelligence):
      - Multi-coordinator Yuma Consensus voting
      - Stake-weighted aggregation with clipping
      - EMA bonds rewarding consistent coordinators
      - Bond multipliers for rewards (up to 1.5x)

    TRUEBIT (Forced Errors):
      - Random forced error injection (~5% of tasks)
      - Jackpot rewards for catching errors (50 ATN)
      - Slashing for missed forced errors (25 ATN)

    The training cycle completed with:
      - Consensus Score: {consensus_score}/100
      - Checkpoints Verified: {num_checkpoints}
      - Coordinators Participated: {num_coords}
      - All verifications passed: {all_pass}
    """.format(
        checkpoint_freq=checkpoint_freq,
        consensus_score=consensus_score,
        num_checkpoints=len(checkpoints),
        num_coords=len(coordinators),
        all_pass=all_match and consensus_correct,
    ))


if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise

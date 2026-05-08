"""
Trainer - Manages training epochs with convergence detection and validation.

Handles:
- Multiple epochs of debate
- Convergence detection (stop when allocations stabilize)
- Train/validation split
- Statistical significance testing
- Real-time metrics logging (TensorBoard, wandb, or custom)
"""

import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Protocol
import math

from ..models.observation import Observation, ObservationStore
from ..models.agent import AgentSet, Tendency
from ..models.tree import TreeStore
from .arena import Arena, DebateResult, Claim


# ============================================================================
# Metrics Logging Protocol
# ============================================================================

class MetricsLogger(Protocol):
    """Protocol for metrics logging backends."""

    def log_scalar(self, name: str, value: float, step: int) -> None:
        """Log a scalar metric."""
        ...

    def log_scalars(self, name: str, values: dict[str, float], step: int) -> None:
        """Log multiple scalars (e.g., allocations per agent)."""
        ...

    def close(self) -> None:
        """Cleanup."""
        ...


class ConsoleLogger:
    """Simple console logging."""

    def log_scalar(self, name: str, value: float, step: int) -> None:
        print(f"  [{step}] {name}: {value:.4f}")

    def log_scalars(self, name: str, values: dict[str, float], step: int) -> None:
        vals = ", ".join(f"{k}={v:.3f}" for k, v in values.items())
        print(f"  [{step}] {name}: {vals}")

    def close(self) -> None:
        pass


class JSONLogger:
    """Log metrics to JSON file for later visualization."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.data: list[dict] = []

    def log_scalar(self, name: str, value: float, step: int) -> None:
        self.data.append({"step": step, "name": name, "value": value})

    def log_scalars(self, name: str, values: dict[str, float], step: int) -> None:
        for k, v in values.items():
            self.data.append({"step": step, "name": f"{name}/{k}", "value": v})

    def close(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2))


class TensorBoardLogger:
    """TensorBoard logging (requires tensorboard package)."""

    def __init__(self, log_dir: str):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            self.available = True
        except ImportError:
            print("TensorBoard not available. Install: pip install tensorboard")
            self.available = False
            self.writer = None

    def log_scalar(self, name: str, value: float, step: int) -> None:
        if self.available:
            self.writer.add_scalar(name, value, step)

    def log_scalars(self, name: str, values: dict[str, float], step: int) -> None:
        if self.available:
            self.writer.add_scalars(name, values, step)

    def close(self) -> None:
        if self.available:
            self.writer.close()


class WandbLogger:
    """Weights & Biases logging (requires wandb package)."""

    def __init__(self, project: str, name: Optional[str] = None):
        try:
            import wandb
            wandb.init(project=project, name=name)
            self.wandb = wandb
            self.available = True
        except ImportError:
            print("wandb not available. Install: pip install wandb")
            self.available = False
            self.wandb = None

    def log_scalar(self, name: str, value: float, step: int) -> None:
        if self.available:
            self.wandb.log({name: value}, step=step)

    def log_scalars(self, name: str, values: dict[str, float], step: int) -> None:
        if self.available:
            self.wandb.log({f"{name}/{k}": v for k, v in values.items()}, step=step)

    def close(self) -> None:
        if self.available:
            self.wandb.finish()


# ============================================================================
# Training Configuration
# ============================================================================

@dataclass
class TrainConfig:
    """Training configuration."""

    # Epochs
    max_epochs: int = 10
    min_epochs: int = 2

    # Convergence
    convergence_threshold: float = 0.005  # Stop when max allocation change < this
    patience: int = 3  # Stop after N epochs with no improvement

    # Learning rate
    initial_lr: float = 0.15
    lr_decay: float = 0.9  # Multiply LR by this each epoch
    min_lr: float = 0.05

    # Validation
    validation_split: float = 0.2  # Hold out 20% for validation
    validate_every: int = 1  # Validate every N epochs

    # Regularization
    min_allocation: float = 0.03  # No agent below 3%
    max_allocation: float = 0.50  # No agent above 50%

    # Logging
    log_every: int = 1


@dataclass
class TrainHistory:
    """Training history for analysis."""

    epochs: list[int] = field(default_factory=list)
    allocations: list[dict[str, float]] = field(default_factory=list)
    tree_scores: list[dict[str, float]] = field(default_factory=list)
    winners: list[str] = field(default_factory=list)
    validation_accuracy: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    converged_at: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "epochs": self.epochs,
            "allocations": self.allocations,
            "tree_scores": self.tree_scores,
            "winners": self.winners,
            "validation_accuracy": self.validation_accuracy,
            "learning_rates": self.learning_rates,
            "converged_at": self.converged_at,
        }


# ============================================================================
# Validator
# ============================================================================

@dataclass
class ValidationResult:
    """Results of validation on held-out observations."""

    total: int
    correct: int
    accuracy: float
    baseline_accuracy: float  # Random baseline
    p_value: float  # Statistical significance
    is_significant: bool  # p < 0.05
    predictions: list[dict]  # Details for analysis


class Validator:
    """Validates model on held-out observations."""

    def __init__(self, arena: Arena):
        self.arena = arena

    def validate(
        self,
        test_observations: list[Observation],
        claims: list[Claim],
        agents: AgentSet,
    ) -> ValidationResult:
        """
        Test if model can predict positions of held-out observations.

        For each test observation:
        1. Model predicts which claim it's most relevant to and PRO/CON
        2. Actually stake it and compare
        3. Count correct predictions
        """
        predictions = []
        correct = 0

        for obs in test_observations:
            # Get model's prediction (using current agent allocations as prior)
            predicted_claim, predicted_pos = self._predict(obs, claims, agents)

            # Actually stake and see what happens
            actual_claim, actual_pos = self._actual_stake(obs, claims)

            is_correct = (
                predicted_claim == actual_claim and
                predicted_pos == actual_pos
            )
            if is_correct:
                correct += 1

            predictions.append({
                "observation": obs.content[:50],
                "predicted_claim": predicted_claim,
                "predicted_position": predicted_pos,
                "actual_claim": actual_claim,
                "actual_position": actual_pos,
                "correct": is_correct,
            })

        total = len(test_observations)
        accuracy = correct / total if total > 0 else 0

        # Random baseline: 1/7 chance of right claim * 1/2 chance of right position
        # ≈ 7.1% for random guessing
        baseline = (1 / len(claims)) * 0.5 if claims else 0.5

        # Chi-squared test for significance
        p_value = self._chi_squared_test(correct, total, baseline)

        return ValidationResult(
            total=total,
            correct=correct,
            accuracy=accuracy,
            baseline_accuracy=baseline,
            p_value=p_value,
            is_significant=p_value < 0.05,
            predictions=predictions,
        )

    def _predict(
        self,
        obs: Observation,
        claims: list[Claim],
        agents: AgentSet,
    ) -> tuple[str, str]:
        """
        Predict which claim and position based on current model state.

        Uses agent allocations as prior: claims from high-allocation agents
        are more likely.
        """
        # Simple heuristic: find claim whose root_value is most similar to observation
        # In practice, we'd use embeddings or Claude
        best_claim = None
        best_score = -1

        for claim in claims:
            # Rough relevance: word overlap
            claim_words = set(claim.tree.root_value.lower().split())
            obs_words = set(obs.content.lower().split())
            overlap = len(claim_words & obs_words)

            # Weight by agent allocation
            agent_weight = agents.get(claim.proposer).allocation
            score = overlap * agent_weight

            if score > best_score:
                best_score = score
                best_claim = claim

        if not best_claim:
            best_claim = claims[0] if claims else None

        # Predict position based on claim's current score
        # Positive score → likely PRO, negative → likely CON
        predicted_pos = "pro" if best_claim and best_claim.score >= 0 else "con"

        return (
            best_claim.proposer.value if best_claim else "unknown",
            predicted_pos
        )

    def _actual_stake(
        self,
        obs: Observation,
        claims: list[Claim],
    ) -> tuple[str, str]:
        """Get actual staking result from Claude."""
        # For speed, use a simplified version
        # In production, would call arena._get_adversarial_stakes

        # Heuristic: use keyword matching as proxy
        obs_lower = obs.content.lower()

        # Map keywords to claims
        keyword_map = {
            "meaning": ["purpose", "legacy", "impact", "mission", "governance", "system"],
            "survival": ["money", "financial", "paycheck", "budget", "survive", "resource"],
            "status": ["recognition", "achievement", "reputation", "respect", "position"],
            "connection": ["relationship", "team", "colleague", "friend", "community"],
            "autonomy": ["freedom", "independent", "control", "own", "solo"],
            "comfort": ["easy", "comfortable", "pain", "stress", "difficult"],
            "curiosity": ["understand", "learn", "research", "explore", "question"],
        }

        best_match = "meaning"  # default
        best_count = 0

        for tendency, keywords in keyword_map.items():
            count = sum(1 for kw in keywords if kw in obs_lower)
            if count > best_count:
                best_count = count
                best_match = tendency

        # Position: look for negative indicators
        negative_words = ["fail", "delay", "reject", "crisis", "problem", "difficult", "lost"]
        is_negative = any(w in obs_lower for w in negative_words)
        position = "con" if is_negative else "pro"

        return best_match, position

    def _chi_squared_test(
        self,
        observed_correct: int,
        total: int,
        expected_rate: float,
    ) -> float:
        """
        Chi-squared test: is observed accuracy significantly better than baseline?

        Returns p-value.
        """
        if total == 0:
            return 1.0

        expected_correct = total * expected_rate
        expected_wrong = total * (1 - expected_rate)
        observed_wrong = total - observed_correct

        # Chi-squared statistic
        if expected_correct > 0 and expected_wrong > 0:
            chi2 = (
                ((observed_correct - expected_correct) ** 2) / expected_correct +
                ((observed_wrong - expected_wrong) ** 2) / expected_wrong
            )
        else:
            return 1.0

        # Approximate p-value (1 degree of freedom)
        # Using survival function of chi-squared distribution
        # For simplicity, use lookup table for common values
        if chi2 > 10.83:
            return 0.001
        elif chi2 > 6.63:
            return 0.01
        elif chi2 > 3.84:
            return 0.05
        elif chi2 > 2.71:
            return 0.1
        else:
            return 0.5  # Not significant


# ============================================================================
# Trainer
# ============================================================================

class Trainer:
    """
    Manages training with convergence detection and validation.

    Usage:
        trainer = Trainer(config=TrainConfig(max_epochs=10))
        trees, history = trainer.train(
            observations=store,
            agents=agents,
            logger=TensorBoardLogger("runs/experiment1"),
        )
    """

    def __init__(
        self,
        config: Optional[TrainConfig] = None,
        arena: Optional[Arena] = None,
    ):
        self.config = config or TrainConfig()
        self.arena = arena or Arena()
        self.validator = Validator(self.arena)

    def train(
        self,
        observations: ObservationStore,
        agents: AgentSet,
        logger: Optional[MetricsLogger] = None,
        verbose: bool = True,
    ) -> tuple[TreeStore, TrainHistory]:
        """
        Train the world model with convergence detection.

        Args:
            observations: All observations
            agents: Agent set (will be modified)
            logger: Metrics logger (TensorBoard, wandb, etc.)
            verbose: Print progress

        Returns:
            (final trees, training history)
        """
        logger = logger or ConsoleLogger()
        history = TrainHistory()
        config = self.config

        # Train/validation split
        all_obs = observations.all()
        random.shuffle(all_obs)
        split_idx = int(len(all_obs) * (1 - config.validation_split))
        train_obs = all_obs[:split_idx]
        val_obs = all_obs[split_idx:]

        train_store = ObservationStore()
        for obs in train_obs:
            train_store.add(obs)

        if verbose:
            print(f"\n{'#'*60}")
            print(f"# TRAINING")
            print(f"{'#'*60}")
            print(f"\n  Total observations: {len(all_obs)}")
            print(f"  Training: {len(train_obs)}")
            print(f"  Validation: {len(val_obs)}")
            print(f"  Max epochs: {config.max_epochs}")
            print(f"  Convergence threshold: {config.convergence_threshold:.1%}")

        current_lr = config.initial_lr
        best_val_accuracy = 0
        epochs_without_improvement = 0
        final_trees = None
        final_claims = []

        for epoch in range(1, config.max_epochs + 1):
            if verbose:
                print(f"\n{'='*60}")
                print(f"EPOCH {epoch} (lr={current_lr:.3f})")
                print(f"{'='*60}")

            # Run one debate round
            trees, result = self.arena.run_full_debate(
                observations=train_store,
                agents=agents,
                rounds=1,
                learning_rate=current_lr,
                verbose=verbose,
            )
            final_trees = trees
            final_claims = result.claims

            # Apply regularization (clamp allocations)
            self._regularize_allocations(agents)

            # Record history
            history.epochs.append(epoch)
            history.allocations.append({
                t.value: agents.get(t).allocation
                for t in Tendency
            })
            history.tree_scores.append({
                c.proposer.value: c.score
                for c in result.claims
            })
            history.winners.append(result.winner.value if result.winner else "none")
            history.learning_rates.append(current_lr)

            # Log metrics
            if epoch % config.log_every == 0:
                logger.log_scalars("allocations", history.allocations[-1], epoch)
                logger.log_scalars("tree_scores", history.tree_scores[-1], epoch)
                logger.log_scalar("learning_rate", current_lr, epoch)

                if result.winner:
                    logger.log_scalar("winner_score", result.scores[result.winner], epoch)

            # Validation
            if epoch % config.validate_every == 0 and val_obs:
                val_result = self.validator.validate(val_obs, final_claims, agents)
                history.validation_accuracy.append(val_result.accuracy)

                logger.log_scalar("validation/accuracy", val_result.accuracy, epoch)
                logger.log_scalar("validation/p_value", val_result.p_value, epoch)

                if verbose:
                    sig = "**" if val_result.is_significant else ""
                    print(f"\n  Validation: {val_result.accuracy:.1%} accuracy "
                          f"(baseline: {val_result.baseline_accuracy:.1%}, "
                          f"p={val_result.p_value:.3f}){sig}")

                # Early stopping check
                if val_result.accuracy > best_val_accuracy:
                    best_val_accuracy = val_result.accuracy
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            # Convergence check
            if epoch >= config.min_epochs:
                max_change = max(abs(c) for c in result.allocation_changes.values())
                logger.log_scalar("max_allocation_change", max_change, epoch)

                if max_change < config.convergence_threshold:
                    if verbose:
                        print(f"\n  CONVERGED at epoch {epoch} (max change: {max_change:.3%})")
                    history.converged_at = epoch
                    break

                # Patience check
                if epochs_without_improvement >= config.patience:
                    if verbose:
                        print(f"\n  EARLY STOPPING at epoch {epoch} (no improvement for {config.patience} epochs)")
                    history.converged_at = epoch
                    break

            # Decay learning rate
            current_lr = max(config.min_lr, current_lr * config.lr_decay)

        # Final validation
        if val_obs and final_claims:
            final_val = self.validator.validate(val_obs, final_claims, agents)
            if verbose:
                print(f"\n{'#'*60}")
                print(f"# FINAL VALIDATION")
                print(f"{'#'*60}")
                print(f"\n  Accuracy: {final_val.accuracy:.1%}")
                print(f"  Baseline: {final_val.baseline_accuracy:.1%}")
                print(f"  p-value: {final_val.p_value:.4f}")
                print(f"  Significant: {'YES' if final_val.is_significant else 'NO'}")

        logger.close()

        return final_trees, history

    def _regularize_allocations(self, agents: AgentSet) -> None:
        """Clamp allocations to prevent extremes."""
        config = self.config
        clamped = False

        for agent in agents.all():
            if agent.allocation < config.min_allocation:
                agent.allocation = config.min_allocation
                clamped = True
            elif agent.allocation > config.max_allocation:
                agent.allocation = config.max_allocation
                clamped = True

        if clamped:
            agents.normalize()

"""
Training Data Feed — bridges ATN agent activity to JEPA training.

Subscribes to runtime events, accumulates training data from the agent
framework's JSONL output, and drives training cycles through the unified
train_vljepa_on_task() function.

Data flow:
  Runtime EventBus
    → EXECUTION_COMPLETED event
    → TrainingDataFeed._on_execution_completed()
    → Increments pending_events counter
  AutonetService._main_loop()
    → TrainingDataFeed.run_cycle()
    → Refreshes TextTrainingDataSource (reads JSONL from disk)
    → Calls train_vljepa_on_task() if enough new data
    → Updates BehavioralProfile with training embeddings
    → Returns (weight_delta, metrics) for network submission

Story 3.2 (BACKLOG_TRAINING_DATA.md)
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TrainingFeedConfig:
    """Configuration for the training data feed."""
    # ATN data directory (~/.atn or equivalent)
    data_dir: str = ""
    # Minimum new events before triggering a training cycle
    min_events_for_cycle: int = 5
    # Minimum seconds between training cycles (prevent thrashing)
    min_cycle_interval: float = 60.0
    # TextJEPA model params (from task_spec or defaults)
    vocab_size: int = 260
    max_seq_length: int = 2048
    embed_dim: int = 64
    num_heads: int = 4
    encoder_depth: int = 2
    predictor_depth: int = 1
    predictor_embed_dim: int = 32
    # Training params
    epochs: int = 1
    learning_rate: float = 1e-3
    batch_size: int = 8
    # Behavioral profile
    profile_K: int = 32
    profile_D: int = 0  # 0 = auto from embed_dim
    profile_decay: float = 0.998
    # Privacy
    scrub_pii: bool = True
    exclude_patterns: list[str] = field(default_factory=list)


class TrainingDataFeed:
    """Manages the flow of ATN agent data into JEPA training cycles.

    This class is the core integration point between the agent framework
    (which generates conversational and execution data) and the training
    pipeline (which learns from that data).

    Lifecycle:
        1. Created by AutonetService or AutonetBridge
        2. notify_execution() called when agent executions complete
        3. run_cycle() called periodically by the service main loop
        4. Internally: refreshes data source, trains, updates profile
    """

    def __init__(self, config: TrainingFeedConfig, governance=None):
        """
        Args:
            config: Training feed configuration.
            governance: Optional GovernanceBridge for on-chain attestation.
                        None = offline mode (training still works, just no attestation).
        """
        self.config = config
        self._data_dir = Path(config.data_dir) if config.data_dir else None

        # Counters
        self._pending_events: int = 0
        self._total_events: int = 0
        self._cycles_completed: int = 0
        self._last_cycle_time: float = 0.0

        # Lazily initialized
        self._text_source = None
        self._profile = None
        self._last_segment_count: int = 0

        # Training attestor (Story 4.2) — attests usage on-chain after cycles
        from .training_attestor import TrainingAttestor
        self._attestor = TrainingAttestor(governance=governance)

    @property
    def pending_events(self) -> int:
        return self._pending_events

    @property
    def cycles_completed(self) -> int:
        return self._cycles_completed

    def notify_execution(self, agent_id: str, execution_id: str, status: str) -> None:
        """Notify that an agent execution completed.

        Called by AutonetBridge's EventBus handler when EXECUTION_COMPLETED
        fires. Increments the pending counter so the next run_cycle() knows
        new data is available.
        """
        if status not in ("completed", "failed"):
            return
        self._pending_events += 1
        self._total_events += 1
        logger.debug(
            "Training feed: execution %s/%s (%s), pending=%d",
            agent_id, execution_id, status, self._pending_events,
        )

    def should_train(self) -> bool:
        """Check if conditions are met for a training cycle.

        Returns True if:
        1. Enough new events have accumulated (min_events_for_cycle)
        2. Enough time has passed since last cycle (min_cycle_interval)
        3. We have a valid data directory to read from
        """
        if not self._data_dir:
            return False

        if self._pending_events < self.config.min_events_for_cycle:
            return False

        elapsed = time.time() - self._last_cycle_time
        if elapsed < self.config.min_cycle_interval:
            return False

        return True

    def run_cycle(self, task_spec: Optional[Dict[str, Any]] = None) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Run one training cycle if conditions are met.

        Refreshes the text data source from disk (JSONL files written by
        ConversationStore and ExecutionLog), trains TextJEPA, updates the
        behavioral profile, and returns the weight delta + metrics.

        Args:
            task_spec: Optional task spec override (from proposer/coordinator).
                       If None, uses config defaults.

        Returns:
            (weight_delta, metrics) if training ran, None if skipped.
        """
        if not self.should_train():
            return None

        logger.info(
            "Training feed: starting cycle %d (%d pending events, %d total)",
            self._cycles_completed + 1, self._pending_events, self._total_events,
        )

        try:
            result = self._do_training(task_spec)
            self._pending_events = 0
            self._last_cycle_time = time.time()
            self._cycles_completed += 1

            if result:
                weight_delta, metrics = result
                logger.info(
                    "Training feed: cycle %d complete — loss=%.4f, batches=%d",
                    self._cycles_completed,
                    metrics.get("loss", 0),
                    metrics.get("num_batches", 0),
                )

                # Attest usage on-chain (Story 4.2)
                self._attestor.record_cycle(metrics, cycle_number=self._cycles_completed)

                return result
            else:
                logger.info("Training feed: cycle %d skipped (no data)", self._cycles_completed)
                return None

        except Exception as e:
            logger.error("Training feed: cycle failed: %s", e, exc_info=True)
            self._pending_events = 0
            self._last_cycle_time = time.time()
            return None

    def _do_training(self, task_spec: Optional[Dict[str, Any]] = None) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Internal: run the actual training step."""
        # Lazy imports — nodes packages may not be available
        from .text_data import TextDataConfig, TextTrainingDataSource
        from .ml import train_vljepa_on_task

        # Build / refresh the text data source
        conversations_dir = self._data_dir / "conversations" if self._data_dir else None
        agents_dir = self._data_dir / "agents" if self._data_dir else None

        text_config = TextDataConfig(
            conversations_dir=str(conversations_dir) if conversations_dir else "",
            agents_dir=str(agents_dir) if agents_dir else "",
            max_seq_length=self.config.max_seq_length,
            vocab_size=self.config.vocab_size,
            batch_size=self.config.batch_size,
            scrub_pii=self.config.scrub_pii,
            exclude_patterns=self.config.exclude_patterns,
            shuffle=True,
        )

        if self._text_source is None:
            self._text_source = TextTrainingDataSource(text_config)
        else:
            # Re-create with fresh config (data dir doesn't change but
            # we need a fresh segment load)
            self._text_source = TextTrainingDataSource(text_config)

        # Force reload from disk
        self._text_source.reload()

        # Check if we have any data
        num_segments = len(self._text_source.segments)
        if num_segments == 0:
            logger.debug("Training feed: no text segments found")
            return None

        # Check if data actually changed
        if num_segments == self._last_segment_count and self._cycles_completed > 0:
            logger.debug("Training feed: segment count unchanged (%d), skipping", num_segments)
            return None

        self._last_segment_count = num_segments
        logger.info("Training feed: %d text segments loaded", num_segments)

        # Build task spec from config defaults + any overrides
        spec = {
            "vocab_size": self.config.vocab_size,
            "max_seq_length": self.config.max_seq_length,
            "embed_dim": self.config.embed_dim,
            "num_heads": self.config.num_heads,
            "encoder_depth": self.config.encoder_depth,
            "predictor_depth": self.config.predictor_depth,
            "predictor_embed_dim": self.config.predictor_embed_dim,
        }
        if task_spec:
            spec.update(task_spec)

        # Train
        weight_delta, metrics = train_vljepa_on_task(
            task_spec=spec,
            text_data_source=self._text_source,
            epochs=self.config.epochs,
            learning_rate=self.config.learning_rate,
        )

        # Update behavioral profile
        self._update_profile(metrics)

        return weight_delta, metrics

    def _update_profile(self, metrics: Dict[str, Any]) -> None:
        """Update the behavioral profile after a training cycle."""
        if not self._data_dir:
            return

        try:
            from .behavioral_profile import BehavioralProfile, compute_training_embeddings
            from .text_jepa import TextJEPAConfig, TextJEPATrainer

            profile_D = self.config.profile_D or self.config.embed_dim

            profile_path = str(self._data_dir / "behavioral_profile.pt")

            if self._profile is None:
                self._profile = BehavioralProfile(
                    K=self.config.profile_K,
                    D=profile_D,
                    decay=self.config.profile_decay,
                    profile_path=profile_path,
                )

            # If the text source has data and we have a trained model reference,
            # compute embeddings. For now, use a simple approach: take the
            # mean-pooled training data embeddings.
            if self._text_source and len(self._text_source.segments) > 0:
                # Re-create a trainer to extract embeddings
                config = TextJEPAConfig(
                    vocab_size=self.config.vocab_size,
                    max_seq_length=self.config.max_seq_length,
                    embed_dim=self.config.embed_dim,
                    num_heads=self.config.num_heads,
                    encoder_depth=self.config.encoder_depth,
                    predictor_depth=self.config.predictor_depth,
                    predictor_embed_dim=self.config.predictor_embed_dim,
                )
                trainer = TextJEPATrainer(config, device="cpu")

                # Get a few batches for profile update
                batches = list(self._text_source)[:10]
                if batches:
                    embeddings = compute_training_embeddings(
                        trainer, batches, max_batches=10,
                    )
                    if embeddings.numel() > 0:
                        # compute_training_embeddings returns (N, D)
                        # BehavioralProfile expects (K, D) or (N, K, D)
                        # For text, K=1 (single vector per sample), so we
                        # reshape (N, D) → (N, 1, D) and let profile mean-pool
                        import torch
                        if embeddings.dim() == 2:
                            # Use K=1 for text-only profiles
                            if self._profile.K == 1:
                                embeddings = embeddings.unsqueeze(1)  # (N, 1, D)
                            else:
                                # Mean-pool all samples into a single (K, D)
                                # by repeating the mean across K slots
                                mean_emb = embeddings.mean(dim=0)  # (D,)
                                embeddings = mean_emb.unsqueeze(0).expand(
                                    self._profile.K, -1
                                )  # (K, D)

                        self._profile.update(embeddings)
                        self._profile.save()
                        logger.info(
                            "Behavioral profile updated (count=%d, hash=%s)",
                            self._profile.update_count,
                            self._profile.hash()[:12],
                        )

        except Exception as e:
            logger.warning("Failed to update behavioral profile: %s", e)

    def get_status(self) -> Dict[str, Any]:
        """Return current training feed status for monitoring."""
        status = {
            "pending_events": self._pending_events,
            "total_events": self._total_events,
            "cycles_completed": self._cycles_completed,
            "last_cycle_time": self._last_cycle_time,
            "data_dir": str(self._data_dir) if self._data_dir else None,
            "segment_count": self._last_segment_count,
            "attestation": self._attestor.get_status(),
        }

        if self._profile:
            status["profile"] = self._profile.to_dict()

        return status

    def force_cycle(self, task_spec: Optional[Dict[str, Any]] = None) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Force a training cycle regardless of conditions.

        Used for testing or manual triggering via API.
        """
        logger.info("Training feed: force cycle requested")
        self._pending_events = max(self._pending_events, self.config.min_events_for_cycle)
        self._last_cycle_time = 0  # Reset interval check
        return self.run_cycle(task_spec)

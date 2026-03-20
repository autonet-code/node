"""
Autonet Node Configuration

Centralized config system replacing all hardcoded values.
Loads from YAML file with environment variable overrides.

Priority order (highest wins):
  1. Environment variables (AUTONET_*)
  2. Config file (autonet.yaml)
  3. Defaults

Usage:
    from nodes.common.config import load_config
    cfg = load_config()              # auto-discovers autonet.yaml
    cfg = load_config("path.yaml")   # explicit path
    cfg.device                       # "cpu" or "cuda"
    cfg.rpc_url                      # blockchain RPC
    cfg.training.epochs              # training hyperparams
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Config dataclasses
# =============================================================================


@dataclass
class BlockchainConfig:
    rpc_url: str = "http://127.0.0.1:8545"
    chain_id: int = 31337
    private_key: Optional[str] = None
    gas_limit: int = 500000


@dataclass
class BlobStoreConfig:
    data_dir: str = "~/.autonet/blobs"
    peer_urls: list = field(default_factory=list)
    server_port: int = 9100


@dataclass
class ModelConfig:
    """Model architecture config."""
    architecture: str = "jepa"  # "simplenet", "jepa", "vl_jepa"
    image_size: int = 32
    patch_size: int = 4
    embed_dim: int = 192
    num_heads: int = 3
    encoder_depth: int = 6
    predictor_depth: int = 3
    predictor_embed_dim: int = 96


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 2
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.05
    num_samples: int = 500
    optimizer: str = "adamw"  # "sgd" or "adamw"
    task_type: str = "jepa"  # "supervised" or "jepa"


@dataclass
class StakingConfig:
    """Staking amounts (in ATN, not wei)."""
    proposer: int = 100
    solver: int = 50
    coordinator: int = 500
    aggregator: int = 1000


@dataclass
class NodeConfig:
    """Per-node runtime config."""
    max_cycles: int = 10
    cycle_delay: float = 3.0
    task_mode: str = "ground_truth"  # "ground_truth" or "consensus_truth"
    aggregation_method: str = "fedavg"  # "fedavg" or "trimmed_mean"
    trim_ratio: float = 0.2
    min_updates_for_aggregation: int = 2


@dataclass
class AutonetConfig:
    """Top-level config container."""
    device: str = "auto"  # "auto", "cpu", "cuda"
    data_dir: str = "./data"
    log_level: str = "INFO"
    seed: Optional[int] = None

    blockchain: BlockchainConfig = field(default_factory=BlockchainConfig)
    blob_store: BlobStoreConfig = field(default_factory=BlobStoreConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    staking: StakingConfig = field(default_factory=StakingConfig)
    node: NodeConfig = field(default_factory=NodeConfig)

    def resolve_device(self) -> str:
        """Resolve 'auto' to actual device."""
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"


# =============================================================================
# Loaders
# =============================================================================


def _deep_update(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _apply_env_overrides(raw: dict) -> dict:
    """
    Apply AUTONET_* environment variable overrides.

    Mapping:
        AUTONET_DEVICE          -> device
        AUTONET_RPC_URL         -> blockchain.rpc_url
        AUTONET_CHAIN_ID        -> blockchain.chain_id
        AUTONET_PRIVATE_KEY     -> blockchain.private_key
        AUTONET_BLOB_DIR        -> blob_store.data_dir
        AUTONET_DATA_DIR        -> data_dir
        AUTONET_LOG_LEVEL       -> log_level
        AUTONET_SEED            -> seed
        AUTONET_TASK_TYPE       -> training.task_type
        AUTONET_TASK_MODE       -> node.task_mode
        AUTONET_EPOCHS          -> training.epochs
        AUTONET_BATCH_SIZE      -> training.batch_size
        AUTONET_LEARNING_RATE   -> training.learning_rate
        AUTONET_NUM_SAMPLES     -> training.num_samples
        AUTONET_IMAGE_SIZE      -> model.image_size
        AUTONET_ARCHITECTURE    -> model.architecture
    """
    env_map = {
        "AUTONET_DEVICE": ("device", str),
        "AUTONET_RPC_URL": ("blockchain.rpc_url", str),
        "AUTONET_CHAIN_ID": ("blockchain.chain_id", int),
        "AUTONET_PRIVATE_KEY": ("blockchain.private_key", str),
        "AUTONET_BLOB_DIR": ("blob_store.data_dir", str),
        "AUTONET_DATA_DIR": ("data_dir", str),
        "AUTONET_LOG_LEVEL": ("log_level", str),
        "AUTONET_SEED": ("seed", int),
        "AUTONET_TASK_TYPE": ("training.task_type", str),
        "AUTONET_TASK_MODE": ("node.task_mode", str),
        "AUTONET_EPOCHS": ("training.epochs", int),
        "AUTONET_BATCH_SIZE": ("training.batch_size", int),
        "AUTONET_LEARNING_RATE": ("training.learning_rate", float),
        "AUTONET_NUM_SAMPLES": ("training.num_samples", int),
        "AUTONET_IMAGE_SIZE": ("model.image_size", int),
        "AUTONET_ARCHITECTURE": ("model.architecture", str),
    }

    for env_var, (path, cast) in env_map.items():
        value = os.environ.get(env_var)
        if value is not None:
            parts = path.split(".")
            target = raw
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = cast(value)
            logger.debug(f"Config override from env: {env_var} -> {path}={value}")

    return raw


def _dict_to_config(raw: dict) -> AutonetConfig:
    """Convert a raw dict to typed AutonetConfig."""
    cfg = AutonetConfig()

    # Top-level scalars
    for key in ("device", "data_dir", "log_level", "seed"):
        if key in raw:
            setattr(cfg, key, raw[key])

    # Sub-configs
    sub_map = {
        "blockchain": (BlockchainConfig, cfg.blockchain),
        "blob_store": (BlobStoreConfig, cfg.blob_store),
        "model": (ModelConfig, cfg.model),
        "training": (TrainingConfig, cfg.training),
        "staking": (StakingConfig, cfg.staking),
        "node": (NodeConfig, cfg.node),
    }

    for section, (cls, instance) in sub_map.items():
        if section in raw and isinstance(raw[section], dict):
            for key, value in raw[section].items():
                if hasattr(instance, key):
                    setattr(instance, key, value)

    return cfg


def load_config(path: Optional[str] = None) -> AutonetConfig:
    """
    Load config from YAML file + env overrides.

    Args:
        path: Explicit path to config file. If None, searches:
              1. AUTONET_CONFIG env var
              2. ./autonet.yaml
              3. ~/.autonet/config.yaml
              Falls back to defaults if no file found.

    Returns:
        AutonetConfig with all values resolved.
    """
    raw = {}

    # Find config file
    if path is None:
        path = os.environ.get("AUTONET_CONFIG")

    if path is None:
        candidates = [
            Path("autonet.yaml"),
            Path.home() / ".autonet" / "config.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                path = str(candidate)
                break

    # Load YAML if found
    if path and Path(path).exists():
        try:
            import yaml
            with open(path) as f:
                raw = yaml.safe_load(f) or {}
            logger.info(f"Loaded config from {path}")
        except ImportError:
            logger.warning("pyyaml not installed, using defaults + env overrides")
        except Exception as e:
            logger.warning(f"Failed to load config from {path}: {e}")

    # Apply env overrides
    _apply_env_overrides(raw)

    # Convert to typed config
    cfg = _dict_to_config(raw)

    logger.info(
        f"Config: device={cfg.device}, arch={cfg.model.architecture}, "
        f"task_type={cfg.training.task_type}, rpc={cfg.blockchain.rpc_url}"
    )

    return cfg

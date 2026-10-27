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
    # Address of the deployed Substrate.sol. Required for chain
    # anchoring + authoritative mint submission (Phase 10.3b/c).
    # Empty = chain submission disabled (offline mode).
    substrate_address: str = ""
    # Deployed ToolBond.sol. The federated close reads the staked-review
    # ratings and reviewer shares from its logs, pinned to the previous
    # anchor (docs/multi_jurisdiction.md decision 10). Empty => zero
    # ratings and no reviewer shares (seeded from registry.json).
    tool_bond_address: str = ""
    # The jurisdiction's RepToken (Decision 2026-07-10): the close weights
    # reviews and voice by REP read from it, pinned to the previous anchor.
    # Empty => genesis regime (no REP weighting). Seeded from registry.json.
    rep_token_address: str = ""
    # The jurisdiction's DAO Registry (key-value store). The close reads
    # the charter text (jurisdiction.autonet.charter) from its
    # RegistryUpdated logs, pinned to the previous anchor, to detect a
    # charter change (docs/multi_jurisdiction.md decision 5). Empty => no
    # pinned charter read. Seeded from registry.json ("dao_registry").
    dao_registry_address: str = ""
    # The jurisdiction's DAO Governor. Epoch anchors go through it as
    # proposals (Decision 2026-10-16). Seeded from registry.json ("dao").
    governor_address: str = ""
    # The block the jurisdiction's contracts were deployed at: every close
    # log scan starts here (Decision 2026-10-26), so no daemon infers it
    # from reads a non-archive node may fail. 0 = find it by search
    # (needs an archive node). Seeded from registry.json ("deploy_block").
    deploy_block: int = 0
    # The jurisdiction's AnchorElection (Decision 2026-10-26): epoch anchors
    # are single-choice elections among drawn agents' options. Empty =>
    # the older yes/no anchor proposal. Seeded from registry.json
    # ("anchor_election"; on chain jurisdiction.<id>.anchor_election).
    anchor_election_address: str = ""


@dataclass
class BlobStoreConfig:
    data_dir: str = "~/.autonet/blobs"
    peer_urls: list = field(default_factory=list)
    server_port: int = 9100


@dataclass
class ModelConfig:
    """Model architecture config."""
    architecture: str = "jepa"  # "simplenet", "jepa", "vl_jepa", "backbone"
    image_size: int = 32
    patch_size: int = 4
    embed_dim: int = 192
    num_heads: int = 3
    encoder_depth: int = 6
    predictor_depth: int = 3
    predictor_embed_dim: int = 96
    # LLM backbone (when architecture="backbone")
    backbone_model_id: str = ""  # HuggingFace repo ID, e.g. "Qwen/Qwen3-1.7B"


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
class P2PConfig:
    """P2P networking configuration."""
    listen_port: int = 0           # 0 = random port
    listen_host: str = "0.0.0.0"
    bootstrap_peers: list = field(default_factory=list)  # multiaddr strings
    # Addresses peers should dial us on, replacing the listen addresses we
    # advertise (a NAT'd or tunneled daemon's public /ip4 or /dns4 addr).
    announce_addrs: list = field(default_factory=list)
    bootstrap_redial_interval: int = 30      # seconds; 0 = dial once at start
    # "tcp" or "ws": one transport per network (py-libp2p runs one per host).
    # "ws" lets relays sit behind an HTTPS proxy and HTTPS-only machines
    # dial out through HTTP_PROXY / HTTPS_PROXY (nodes/common/ws_proxy.py).
    transport: str = "tcp"
    enable_quic: bool = False
    # Without announce_addrs: map the listen port on the home router (UPnP)
    # and announce our public address once a peer confirms it can connect.
    enable_upnp: bool = True
    auto_reach: bool = True
    reach_interval: int = 300                # seconds between reachability checks
    capability_advertise_interval: int = 60  # seconds
    ping_interval: int = 30                  # seconds


@dataclass
class InferenceP2PConfig:
    """P2P settings specific to inference and embedding exchange."""
    embedding_exchange_timeout: float = 5.0   # seconds to wait for embedding response
    max_embedding_peers: int = 4              # max peers to broadcast embeddings to
    shard_announce_enabled: bool = True       # announce locally-stored shards to peers
    blob_fetch_timeout: float = 30.0          # seconds to wait for P2P blob fetch
    blob_announce_on_store: bool = True       # announce blobs when added to store


@dataclass
class CaptureConfig:
    """Data capture configuration (Epic 5)."""
    enabled_sources: list = field(default_factory=list)  # ["screen", "audio", "browser"]
    fps_cap: int = 2                     # Max frames per second
    resolution: list = field(default_factory=lambda: [224, 224])  # [width, height]
    screen_monitor: int = 1              # mss monitor index (1 = primary)
    browser_relay_url: str = "ws://127.0.0.1:9222/training"  # CDP relay training endpoint
    browser_exclude_patterns: list = field(default_factory=list)  # URL patterns to skip
    browser_scrub_pii: bool = True       # Scrub PII from browser-captured text


@dataclass
class PrivacyConfig:
    """Privacy controls for captured data (Epic 5)."""
    exclude_apps: list = field(default_factory=list)   # Window titles to skip
    blur_regions: list = field(default_factory=list)    # [{"x": 0, "y": 0, "w": 100, "h": 100}]
    scrub_pii: bool = True               # Redact emails, phones, credit cards


@dataclass
class ResourceConfig:
    """Resource limits for the node (Epic 6)."""
    max_cpu_percent: float = 80.0        # Pause training above this (system-wide)
    max_memory_mb: int = 4096            # Pause when THE DAEMON's RSS exceeds this
    min_free_memory_mb: int = 512        # Pause when system free memory dips below
    active_hours_start: int = 0          # 0-23 hour (0 = always active)
    active_hours_end: int = 24           # 0-24 hour (24 = always active)
    check_interval: float = 10.0         # Seconds between checks


@dataclass
class UpdateConfig:
    """Auto-update configuration (Epic 6)."""
    check_interval: int = 3600           # Seconds between update checks
    update_source: str = "git"           # "git", "http", or "blob_store"
    update_url: str = ""                 # URL for http source
    auto_apply: bool = False             # Auto-apply updates


@dataclass
class GuildConfig:
    """Guild formation configuration."""
    guild_id: Optional[int] = None           # Guild this node belongs to (None = not in a guild)
    auto_join: bool = False                  # Auto-join recommended guild on startup
    module_ids: list = field(default_factory=list)  # Module IDs this node can train
    aggregation_level: str = "flat"          # "flat" (legacy), "guild", or "network"
    min_guild_updates: int = 2               # Min updates before guild-level aggregation
    reputation_weight: float = 0.3           # Weight of guild reputation in network aggregation
    member_count_weight: float = 0.7         # Weight of member count in network aggregation


@dataclass
class NodeConfig:
    """Per-node runtime config."""
    # 0 = run forever. A daemon reads autonet.yaml only from its cwd, so
    # these are what every installed daemon runs with: the old 10 stopped
    # the service (and its epoch closes) after ~40s.
    max_cycles: int = 0
    cycle_delay: float = 30.0
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
    # Fixed-emission economics: tokens minted per second of epoch duration,
    # split pro-rata by post-gate contribution at epoch close. None = legacy
    # uncapped mint. Consensus-relevant: all daemons in a network must agree.
    epoch_emission_rate: Optional[float] = None
    # Candle close (anti last-second-spam; docs/epoch_economics.md): epochs
    # run the full min+window, then close at a retroactively drawn cutoff
    # inside the window; post-cutoff events roll forward. 0 = disabled
    # (plain interval epochs). Consensus-relevant like the rate.
    epoch_min_seconds: float = 0.0
    epoch_candle_window_seconds: float = 0.0
    # "<chain_id>:<DAO address>" (docs/multi_jurisdiction.md decision 2):
    # the gossip topic and the key this jurisdiction's world data lives
    # under. Set by the atn bridge from the resolved jurisdiction. Empty
    # only in local, chain-free runs.
    jurisdiction_key: str = ""

    blockchain: BlockchainConfig = field(default_factory=BlockchainConfig)
    blob_store: BlobStoreConfig = field(default_factory=BlobStoreConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    node: NodeConfig = field(default_factory=NodeConfig)
    p2p: P2PConfig = field(default_factory=P2PConfig)
    guild: GuildConfig = field(default_factory=GuildConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    update: UpdateConfig = field(default_factory=UpdateConfig)

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
        AUTONET_P2P_PORT        -> p2p.listen_port
        AUTONET_P2P_HOST        -> p2p.listen_host
        AUTONET_P2P_BOOTSTRAP   -> p2p.bootstrap_peers (comma list)
        AUTONET_P2P_ANNOUNCE    -> p2p.announce_addrs (comma list)
    """
    env_map = {
        "AUTONET_DEVICE": ("device", str),
        "AUTONET_RPC_URL": ("blockchain.rpc_url", str),
        "AUTONET_CHAIN_ID": ("blockchain.chain_id", int),
        "AUTONET_PRIVATE_KEY": ("blockchain.private_key", str),
        "AUTONET_SUBSTRATE_ADDRESS": ("blockchain.substrate_address", str),
        "AUTONET_TOOL_BOND_ADDRESS": ("blockchain.tool_bond_address", str),
        "AUTONET_REP_TOKEN_ADDRESS": ("blockchain.rep_token_address", str),
        "AUTONET_DAO_REGISTRY_ADDRESS": ("blockchain.dao_registry_address", str),
        "AUTONET_GOVERNOR_ADDRESS": ("blockchain.governor_address", str),
        "AUTONET_DEPLOY_BLOCK": ("blockchain.deploy_block", int),
        "AUTONET_ANCHOR_ELECTION_ADDRESS": ("blockchain.anchor_election_address", str),
        "AUTONET_BLOB_DIR": ("blob_store.data_dir", str),
        "AUTONET_DATA_DIR": ("data_dir", str),
        "AUTONET_LOG_LEVEL": ("log_level", str),
        "AUTONET_P2P_PORT": ("p2p.listen_port", int),
        "AUTONET_P2P_HOST": ("p2p.listen_host", str),
        "AUTONET_P2P_TRANSPORT": ("p2p.transport", str),
        "AUTONET_P2P_BOOTSTRAP": ("p2p.bootstrap_peers", _csv_list),
        "AUTONET_P2P_ANNOUNCE": ("p2p.announce_addrs", _csv_list),
        "AUTONET_SEED": ("seed", int),
        "AUTONET_TASK_TYPE": ("training.task_type", str),
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


def _csv_list(value: str) -> list:
    return [part.strip() for part in value.split(",") if part.strip()]


def _dict_to_config(raw: dict) -> AutonetConfig:
    """Convert a raw dict to typed AutonetConfig."""
    cfg = AutonetConfig()

    # Top-level scalars
    for key in ("device", "data_dir", "log_level", "seed", "epoch_emission_rate",
                "epoch_min_seconds", "epoch_candle_window_seconds"):
        if key in raw:
            setattr(cfg, key, raw[key])

    # Sub-configs
    sub_map = {
        "blockchain": (BlockchainConfig, cfg.blockchain),
        "blob_store": (BlobStoreConfig, cfg.blob_store),
        "model": (ModelConfig, cfg.model),
        "training": (TrainingConfig, cfg.training),
        "node": (NodeConfig, cfg.node),
        "p2p": (P2PConfig, cfg.p2p),
        "guild": (GuildConfig, cfg.guild),
        "capture": (CaptureConfig, cfg.capture),
        "privacy": (PrivacyConfig, cfg.privacy),
        "resources": (ResourceConfig, cfg.resources),
        "update": (UpdateConfig, cfg.update),
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

    # Seed the substrate address from registry.json when yaml/env left
    # it empty. registry.json is the single source of truth the atn
    # config loader and publish_network_config.py already share, so a
    # contract redeploy only has to touch one file for the chain
    # candle seed and chain submission to follow.
    if not cfg.blockchain.substrate_address:
        seeded = _registry_substrate_address(path)
        if seeded:
            cfg.blockchain.substrate_address = seeded
            logger.info(
                "blockchain.substrate_address seeded from registry.json: %s…",
                seeded[:10],
            )
    if not cfg.blockchain.tool_bond_address:
        cfg.blockchain.tool_bond_address = _registry_substrate_address(
            path, key="tool_bond")
    if not cfg.blockchain.rep_token_address:
        cfg.blockchain.rep_token_address = _registry_substrate_address(
            path, key="rep_token")
    if not cfg.blockchain.dao_registry_address:
        cfg.blockchain.dao_registry_address = _registry_substrate_address(
            path, key="dao_registry")
    if not cfg.blockchain.governor_address:
        cfg.blockchain.governor_address = _registry_substrate_address(
            path, key="dao")
    if not cfg.blockchain.anchor_election_address:
        cfg.blockchain.anchor_election_address = _registry_substrate_address(
            path, key="anchor_election")
    if not cfg.blockchain.deploy_block:
        try:
            cfg.blockchain.deploy_block = int(
                _registry_substrate_address(path, key="deploy_block") or 0)
        except (TypeError, ValueError):
            cfg.blockchain.deploy_block = 0

    logger.info(
        f"Config: device={cfg.device}, arch={cfg.model.architecture}, "
        f"task_type={cfg.training.task_type}, rpc={cfg.blockchain.rpc_url}"
    )

    return cfg


def _registry_substrate_address(
    config_path: Optional[str], key: str = "substrate",
) -> str:
    """Resolve the deployed Substrate.sol address (or another
    ``contracts.<key>`` entry, e.g. ``tool_bond``) from registry.json.

    The copy packaged with atn comes first: it is the one the atn daemon
    registers agents against (atn.config._load_registry_seed), so the
    federated close must anchor to the same Substrate. Stale copies next to
    the config file (e.g. ~/.atn/registry.json from an older release) or in
    the CWD used to win here and silently pointed anchoring at a retired
    Substrate after a redeploy. They remain fallbacks only.
    """
    import json as _json

    candidates = []
    # The copy packaged with atn (the only one a pip install has).
    candidates.append(Path(__file__).resolve().parents[2] / "atn" / "registry.json")
    candidates.append(Path(__file__).resolve().parents[2] / "registry.json")
    if config_path:
        candidates.append(Path(config_path).resolve().parent / "registry.json")
    candidates.append(Path("registry.json"))
    for candidate in candidates:
        try:
            if not candidate.exists():
                continue
            data = _json.loads(candidate.read_text(encoding="utf-8"))
            for entry in data.get("jurisdictions", {}).values():
                addr = entry.get("contracts", {}).get(key, "")
                if addr:
                    return str(addr)
        except Exception as e:
            logger.warning("registry.json read failed at %s: %s", candidate, e)
    return ""

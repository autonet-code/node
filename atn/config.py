"""Configuration loading for ATN.

Reads ~/.atn/config.yaml (or a user-specified path).  Supports environment
variable interpolation via ${VAR_NAME} syntax.

Config layout:
    data_dir:    ~/.atn              # global state, pidfiles
    agents_dir:  ./agents            # agent directories (project-local, gitignored)

    orchestrator:
      provider: anthropic            # which provider the orchestrator uses
      model: claude-sonnet-4-20250514   # which model (overrides provider default)

    providers:
      anthropic:
        api_key: ${ANTHROPIC_API_KEY}
        default_model: claude-sonnet-4-20250514
      openai:
        api_key: ${OPENAI_API_KEY}
        default_model: gpt-4o
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import json

import yaml

log = logging.getLogger(__name__)

_DEFAULT_DIR = Path.home() / ".atn"
_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider."""
    name: str
    api_key: str = ""
    default_model: str = ""
    base_url: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConnectorConfig:
    """Configuration for a single MCP connector."""
    name: str
    mode: str = "local"           # "local" | "npx" | "uvx"
    package: str = ""             # npm/pypi package name (npx/uvx modes)
    entry: str = "server.py"      # entry point (local mode only)
    command: str = ""             # explicit command override
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    env_required: list[str] = field(default_factory=list)


@dataclass
class VoiceConfig:
    """Configuration for the voice service.

    Requires ``pip install atn[voice]`` at minimum.
    """
    enabled: bool = False       # voice service active on startup
    backend: str = "kokoro"     # TTS backend: kokoro, edge, elevenlabs, piper
    ptt_keys: list[str] = field(default_factory=lambda: ["page down", "insert"])
    mute_key: str = "page up"
    voice_volume: float = 1.0
    tools_volume: float = 0.55
    effects_volume: float = 0.35
    narrate_tools: bool = True
    announcements: list[str] = field(default_factory=lambda: [
        "agent_runs", "agent_created", "agent_completed", "delegate_lifecycle"
    ])
    output_device: str | None = None
    input_device: str | None = None
    kokoro_model_dir: str | None = None   # directory containing kokoro-v1.0.onnx
    piper_module_dir: str | None = None   # directory containing piper voice module


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator's LLM usage."""
    provider: str = ""          # provider name (e.g. "anthropic", "gemini")
    model: str = ""             # model override (e.g. "claude-sonnet-4-20250514")


@dataclass
class RPBConfig:
    """Configuration for RPB (Recursive Principial Body) network participation.

    Controls the decentralized training service, blockchain connection,
    and network participation.  All fields are optional — the framework
    works fully without any network participation.

    Previously named AutonetConfig; "Autonet" is the first jurisdiction.

    Defaults to the Autonet jurisdiction on Etherlink Shadownet.  The
    Governor address, RPC URL, and chain ID are hardcoded in
    ``atn.jurisdiction`` and used when no user override is provided.
    This means a fresh ``pip install autonet-computer`` connects to the
    Autonet jurisdiction automatically — no configuration required for
    contract discovery.
    """
    enabled: bool = False               # Whether the autonet service starts
    config_path: str = ""               # Path to autonet.yaml (auto-discovered if empty)
    # Blockchain connection — defaults from atn.jurisdiction
    rpc_url: str = ""                   # Defaults to jurisdiction.RPC_URL
    chain_id: int = 0                   # Defaults to jurisdiction.CHAIN_ID
    private_key: str = ""               # Hex private key for signing attestation txns
    # Wallet is managed externally (MetaMask etc.) — we just track the address
    wallet_address: str = ""            # Connected wallet address (empty = not connected)
    # Jurisdiction entry point — defaults from atn.jurisdiction
    dao_address: str = ""               # Defaults to jurisdiction.GOVERNOR_ADDRESS
    # RPB-specific fields
    jurisdiction_id: str = "autonet"    # First jurisdiction
    rpb_contract_address: str = ""      # Discovered from Registry
    registry_address: str = ""          # Discovered from RepToken.registryAddress()
    token_address: str = ""             # Discovered from Governor.token()
    economy_address: str = ""           # Discovered from RepToken.economyAddress()
    timelock_address: str = ""          # Discovered from Governor.timelock()
    min_alignment_threshold: float = 0.5
    generate_keypairs: bool = True

    def __post_init__(self) -> None:
        """Apply jurisdiction defaults for empty fields."""
        from .jurisdiction import GOVERNOR_ADDRESS, RPC_URL, CHAIN_ID
        if not self.dao_address:
            self.dao_address = GOVERNOR_ADDRESS
        if not self.rpc_url:
            self.rpc_url = RPC_URL
        if not self.chain_id:
            self.chain_id = CHAIN_ID


# Backward-compat alias
AutonetConfig = RPBConfig


@dataclass
class TraceLoggingConfig:
    """Configuration for structured agent trace logging.

    Traces are written to ``trace_dir`` (default: ``{data_dir}/traces/``) as
    content-addressed JSON files and serve as training data for VL-JEPA.

    Example config.yaml snippet::

        trace_logging:
          enabled: true
          trace_dir: ~/.atn/traces   # optional override
          include_user_data: false   # set true to include orchestrator sessions
          min_turns: 1               # quality filter: skip near-empty sessions
    """
    enabled: bool = False
    trace_dir: str = ""             # empty → {data_dir}/traces/
    include_user_data: bool = False  # consent gate (Epic 3 Story 3.1)
    min_turns: int = 1              # minimum assistant turns for quality filter


@dataclass
class ATNConfig:
    """Top-level ATN configuration."""
    data_dir: Path = field(default_factory=lambda: _DEFAULT_DIR)
    agents_dir: Path = field(default_factory=lambda: Path("agents"))
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    autonet: AutonetConfig = field(default_factory=AutonetConfig)
    trace_logging: TraceLoggingConfig = field(default_factory=TraceLoggingConfig)
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    connectors: dict[str, ConnectorConfig] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def rpb(self) -> AutonetConfig:
        """Alias — RPB config is the autonet config."""
        return self.autonet


# ---------------------------------------------------------------------------
# Registry seed — repo-level jurisdiction defaults
# ---------------------------------------------------------------------------

def _load_registry_seed(jurisdiction_id: str = "autonet") -> dict[str, Any]:
    """Load network + contract defaults from registry.json in the repo root.

    Returns a flat dict suitable for merging into the RPBConfig builder.
    Empty dict if the file is missing or the jurisdiction isn't listed.
    """
    # registry.json lives at the repo root (parent of atn/)
    registry_path = Path(__file__).resolve().parent.parent / "registry.json"
    if not registry_path.is_file():
        return {}
    try:
        data = json.loads(registry_path.read_text(encoding="utf-8"))
        entry = data.get("jurisdictions", {}).get(jurisdiction_id)
        if not entry:
            return {}
        seed: dict[str, Any] = {}
        seed["jurisdiction_id"] = jurisdiction_id
        net = entry.get("network", {})
        if net.get("rpc_url"):
            seed["rpc_url"] = net["rpc_url"]
        if net.get("chain_id"):
            seed["chain_id"] = net["chain_id"]
        contracts = entry.get("contracts", {})
        if contracts.get("dao"):
            seed["dao_address"] = contracts["dao"]
        if contracts.get("rpb"):
            seed["rpb_contract_address"] = contracts["rpb"]
        log.info("Registry seed for '%s': dao=%s, rpb=%s",
                 jurisdiction_id,
                 seed.get("dao_address", "")[:10] or "(none)",
                 seed.get("rpb_contract_address", "")[:10] or "(none)")
        return seed
    except Exception:
        log.warning("Failed to load registry.json", exc_info=True)
        return {}


# ---------------------------------------------------------------------------
# Env-var interpolation
# ---------------------------------------------------------------------------

def _resolve_env(value: Any) -> Any:
    """Recursively resolve ${VAR} references in strings."""
    if isinstance(value, str):
        def _sub(m: re.Match) -> str:
            var = m.group(1)
            env_val = os.environ.get(var, "")
            if not env_val:
                log.warning("Environment variable %s is not set", var)
            return env_val
        return _ENV_PATTERN.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# .env loading (no external dependency)
# ---------------------------------------------------------------------------

def _load_dotenv(env_path: Path | None = None) -> int:
    """Load key=value pairs from a .env file into os.environ.

    Skips blank lines and comments (#).  Strips optional quotes around
    values.  Does NOT override variables that are already set.

    Returns the number of variables loaded.
    """
    if env_path is None:
        env_path = _DEFAULT_DIR / ".env"
    if not env_path.is_file():
        return 0

    loaded = 0
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Strip surrounding quotes (single or double)
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            if key and key not in os.environ:
                os.environ[key] = value
                loaded += 1
    except Exception:
        log.warning("Failed to load .env from %s", env_path, exc_info=True)
    if loaded:
        log.info("Loaded %d env variable(s) from %s", loaded, env_path)
    return loaded


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _expand_path(p: str | Path) -> Path:
    """Expand ~ and env vars in a path string."""
    return Path(os.path.expandvars(os.path.expanduser(str(p))))


def load_config(path: Path | None = None) -> ATNConfig:
    """Load configuration from a YAML file.

    Falls back to sensible defaults if the file doesn't exist.
    """
    if path is None:
        path = _DEFAULT_DIR / "config.yaml"

    # Load ~/.atn/.env before config so ${VAR} interpolation can use .env values
    _load_dotenv()

    config = ATNConfig()

    if not path.exists():
        log.info("No config file at %s — using defaults", path)
        return config

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        log.exception("Failed to read config from %s", path)
        return config

    config.raw = raw
    config_dir = path.parent  # resolve relative paths against config file location

    if "data_dir" in raw:
        config.data_dir = _expand_path(raw["data_dir"])
        if not config.data_dir.is_absolute():
            config.data_dir = (config_dir / config.data_dir).resolve()
    if "agents_dir" in raw:
        config.agents_dir = _expand_path(raw["agents_dir"])
        if not config.agents_dir.is_absolute():
            config.agents_dir = (config_dir / config.agents_dir).resolve()

    # Orchestrator
    orch_raw = raw.get("orchestrator", {})
    if isinstance(orch_raw, dict):
        config.orchestrator = OrchestratorConfig(
            provider=orch_raw.get("provider", ""),
            model=orch_raw.get("model", ""),
        )

    # Voice
    voice_raw = raw.get("voice", {})
    if isinstance(voice_raw, dict):
        config.voice = VoiceConfig(
            enabled=voice_raw.get("enabled", False),
            backend=voice_raw.get("backend", "kokoro"),
            ptt_keys=voice_raw.get("ptt_keys", ["page down", "insert"]),
            mute_key=voice_raw.get("mute_key", "page up"),
            voice_volume=voice_raw.get("voice_volume", 1.0),
            tools_volume=voice_raw.get("tools_volume", 0.55),
            effects_volume=voice_raw.get("effects_volume", 0.35),
            narrate_tools=voice_raw.get("narrate_tools", True),
            announcements=voice_raw.get("announcements", [
                "agent_runs", "agent_created", "agent_completed", "delegate_lifecycle"
            ]),
            output_device=voice_raw.get("output_device"),
            input_device=voice_raw.get("input_device"),
            kokoro_model_dir=voice_raw.get("kokoro_model_dir"),
            piper_module_dir=voice_raw.get("piper_module_dir"),
        )

    # Autonet / RPB network layer
    # Merge priority (lowest to highest):
    #   registry.json (repo-level defaults) < blockchain < rpb < autonet
    autonet_raw = raw.get("autonet", {})
    if not isinstance(autonet_raw, dict):
        autonet_raw = {}
    rpb_raw = raw.get("rpb", {})
    if not isinstance(rpb_raw, dict):
        rpb_raw = {}
    blockchain_raw = raw.get("blockchain", {})
    if not isinstance(blockchain_raw, dict):
        blockchain_raw = {}
    # Seed from registry.json (repo-level jurisdiction registry)
    jurisdiction_id = (
        autonet_raw.get("jurisdiction_id")
        or rpb_raw.get("jurisdiction_id")
        or "autonet"
    )
    merged = _load_registry_seed(jurisdiction_id)
    # Layer blockchain section on top
    if blockchain_raw.get("rpc_url"):
        merged["rpc_url"] = blockchain_raw["rpc_url"]
    if blockchain_raw.get("chain_id"):
        merged["chain_id"] = blockchain_raw["chain_id"]
    # Pull dao_address from blockchain.contracts.AutonetDAO
    bc_contracts = blockchain_raw.get("contracts", {})
    if isinstance(bc_contracts, dict) and bc_contracts.get("AutonetDAO"):
        merged["dao_address"] = bc_contracts["AutonetDAO"]
    merged.update(rpb_raw)
    merged.update(autonet_raw)
    resolved = _resolve_env(merged)
    # Enable automatically if dao_address + rpc_url are present
    enabled = resolved.get("enabled", False)
    if not enabled and resolved.get("dao_address") and resolved.get("rpc_url"):
        enabled = True
    config.autonet = RPBConfig(
        enabled=enabled,
        config_path=resolved.get("config_path", ""),
        rpc_url=resolved.get("rpc_url", ""),
        chain_id=resolved.get("chain_id", 0),
        private_key=resolved.get("private_key", ""),
        wallet_address=resolved.get("wallet_address", ""),
        dao_address=resolved.get("dao_address", ""),
        jurisdiction_id=resolved.get("jurisdiction_id", "autonet"),
        rpb_contract_address=resolved.get("rpb_contract_address", ""),
        registry_address=resolved.get("registry_address", ""),
        token_address=resolved.get("token_address", ""),
        economy_address=resolved.get("economy_address", ""),
        timelock_address=resolved.get("timelock_address", ""),
        min_alignment_threshold=resolved.get("min_alignment_threshold", 0.5),
        generate_keypairs=resolved.get("generate_keypairs", True),
    )

    # Trace logging
    trace_raw = raw.get("trace_logging", {})
    if isinstance(trace_raw, dict):
        config.trace_logging = TraceLoggingConfig(
            enabled=trace_raw.get("enabled", False),
            trace_dir=trace_raw.get("trace_dir", ""),
            include_user_data=trace_raw.get("include_user_data", False),
            min_turns=trace_raw.get("min_turns", 1),
        )

    # Connectors
    for name, craw in raw.get("connectors", {}).items():
        if not isinstance(craw, dict):
            continue
        resolved = _resolve_env(craw)
        config.connectors[name] = ConnectorConfig(
            name=name,
            mode=resolved.get("mode", "local"),
            package=resolved.get("package", ""),
            entry=resolved.get("entry", "server.py"),
            command=resolved.get("command", ""),
            args=resolved.get("args", []),
            env=resolved.get("env", {}),
            env_required=resolved.get("env_required", []),
        )

    # Providers
    for name, praw in raw.get("providers", {}).items():
        if not isinstance(praw, dict):
            continue
        resolved = _resolve_env(praw)
        known_keys = {"api_key", "default_model", "base_url"}
        config.providers[name] = ProviderConfig(
            name=name,
            api_key=resolved.get("api_key", ""),
            default_model=resolved.get("default_model", ""),
            base_url=resolved.get("base_url", ""),
            extra={k: v for k, v in resolved.items() if k not in known_keys},
        )

    # Warm the module cache (used by scheduler for periodic health checks)
    from . import _cache
    _cache.warm_left()

    return config


# ---------------------------------------------------------------------------
# Config persistence — save/remove connectors in config.yaml
# ---------------------------------------------------------------------------

def save_connector_to_config(
    connector_id: str,
    spec_dict: dict[str, Any],
    config_path: Path | None = None,
) -> None:
    """Add or update a connector entry in config.yaml.

    Reads the existing YAML, updates the connectors section, writes back.
    Creates the file if it doesn't exist.
    """
    config_path = config_path or (_DEFAULT_DIR / "config.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    raw: dict[str, Any] = {}
    if config_path.exists():
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception:
            log.warning("Failed to read config for connector save: %s", config_path)
            raw = {}

    if "connectors" not in raw:
        raw["connectors"] = {}

    raw["connectors"][connector_id] = spec_dict

    config_path.write_text(
        yaml.dump(raw, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    log.info("Connector '%s' saved to %s", connector_id, config_path)


def save_provider_to_config(
    provider_id: str,
    spec_dict: dict[str, Any],
    config_path: Path | None = None,
) -> None:
    """Add or update a provider entry in config.yaml.

    Reads the existing YAML, updates the providers section, writes back.
    Creates the file if it doesn't exist.
    """
    config_path = config_path or (_DEFAULT_DIR / "config.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    raw: dict[str, Any] = {}
    if config_path.exists():
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception:
            log.warning("Failed to read config for provider save: %s", config_path)
            raw = {}

    if "providers" not in raw:
        raw["providers"] = {}

    raw["providers"][provider_id] = spec_dict

    config_path.write_text(
        yaml.dump(raw, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    log.info("Provider '%s' saved to %s", provider_id, config_path)


def remove_provider_from_config(
    provider_id: str,
    config_path: Path | None = None,
) -> bool:
    """Remove a provider entry from config.yaml.

    Returns True if the provider was found and removed, False otherwise.
    """
    config_path = config_path or (_DEFAULT_DIR / "config.yaml")

    if not config_path.exists():
        return False

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        log.warning("Failed to read config for provider removal: %s", config_path)
        return False

    providers = raw.get("providers", {})
    if provider_id not in providers:
        return False

    del providers[provider_id]
    if not providers:
        raw.pop("providers", None)

    config_path.write_text(
        yaml.dump(raw, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    log.info("Provider '%s' removed from %s", provider_id, config_path)
    return True


def save_orchestrator_model_to_config(
    model: str,
    config_path: Path | None = None,
) -> None:
    """Persist the orchestrator model choice to config.yaml.

    Reads the existing YAML, updates orchestrator.model, writes back.
    Creates the file / section if it doesn't exist.
    """
    config_path = config_path or (_DEFAULT_DIR / "config.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    raw: dict[str, Any] = {}
    if config_path.exists():
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception:
            log.warning("Failed to read config for orchestrator model save: %s", config_path)
            raw = {}

    if "orchestrator" not in raw:
        raw["orchestrator"] = {}

    raw["orchestrator"]["model"] = model

    config_path.write_text(
        yaml.dump(raw, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    log.info("Orchestrator model '%s' saved to %s", model, config_path)


def remove_connector_from_config(
    connector_id: str,
    config_path: Path | None = None,
) -> bool:
    """Remove a connector entry from config.yaml.

    Returns True if the connector was found and removed, False otherwise.
    """
    config_path = config_path or (_DEFAULT_DIR / "config.yaml")

    if not config_path.exists():
        return False

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        log.warning("Failed to read config for connector removal: %s", config_path)
        return False

    connectors = raw.get("connectors", {})
    if connector_id not in connectors:
        return False

    del connectors[connector_id]
    if not connectors:
        raw.pop("connectors", None)

    config_path.write_text(
        yaml.dump(raw, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    log.info("Connector '%s' removed from %s", connector_id, config_path)
    return True

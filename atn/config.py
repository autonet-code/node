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
class AutonetConfig:
    """Configuration for the Autonet network layer.

    Controls the decentralized training service, blockchain connection,
    and network participation.  All fields are optional — the framework
    works fully without any network participation.
    """
    enabled: bool = False               # Whether the autonet service starts
    config_path: str = ""               # Path to autonet.yaml (auto-discovered if empty)
    # Blockchain connection (overrides autonet.yaml if set)
    rpc_url: str = ""                   # e.g. "https://node.shadownet.etherlink.com"
    chain_id: int = 0                   # e.g. 127823 for Etherlink Shadownet
    # Wallet is managed externally (MetaMask etc.) — we just track the address
    wallet_address: str = ""            # Connected wallet address (empty = not connected)


@dataclass
class ATNConfig:
    """Top-level ATN configuration."""
    data_dir: Path = field(default_factory=lambda: _DEFAULT_DIR)
    agents_dir: Path = field(default_factory=lambda: Path("agents"))
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    autonet: AutonetConfig = field(default_factory=AutonetConfig)
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    connectors: dict[str, ConnectorConfig] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


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

    # Autonet network layer
    autonet_raw = raw.get("autonet", {})
    if isinstance(autonet_raw, dict):
        resolved = _resolve_env(autonet_raw)
        config.autonet = AutonetConfig(
            enabled=resolved.get("enabled", False),
            config_path=resolved.get("config_path", ""),
            rpc_url=resolved.get("rpc_url", ""),
            chain_id=resolved.get("chain_id", 0),
            wallet_address=resolved.get("wallet_address", ""),
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

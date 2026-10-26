"""Configuration loading for ATN.

Reads ~/.atn/config.yaml (or a user-specified path).  Supports environment
variable interpolation via ${VAR_NAME} syntax.

Config layout:
    data_dir:    ~/.atn              # global state, pidfiles
    agents_dir:  ~/.atn/agents       # agent directories (or ./agents if it exists in CWD)

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

import hashlib
import json

import yaml

log = logging.getLogger(__name__)

_DEFAULT_DIR = Path.home() / ".atn"
_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _default_agents_dir(data_dir: Path | None = None) -> Path:
    """Resolve the default agents directory.

    - If ``./agents`` exists in the CWD (a dev running from the repo, or a
      user who deliberately created one), use it — back-compat.
    - Otherwise root it under the data dir (``~/.atn/agents``), so pip users
      launching ``atn`` from an arbitrary CWD don't get a stray ``agents/``
      mkdir'd wherever they happen to be.

    The directory is NOT created here; consumers mkdir on demand.
    """
    cwd_agents = Path("agents")
    if cwd_agents.is_dir():
        return cwd_agents
    base = data_dir if data_dir is not None else _DEFAULT_DIR
    return base / "agents"


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
    # Custom-provider model list — surfaces in the UI's model picker and
    # cross-provider listings.  Each entry is either a string ID or a dict
    # {id, name, capability_tier?, context_window?}.
    models: list[Any] = field(default_factory=list)
    # Daemon-wide HARD dollar expenditure cap for this provider, enforced
    # pre-flight by the metering service (atn/metering.py). 0.0 => no cap.
    # For metered API providers (anthropic/openai/…) this is a real spend
    # ceiling in USD, computed from the cost-per-token table × actual token
    # counts. For subscription providers (claude_max) it is unused — those are
    # bounded by the inferred subscription quota, not a dollar figure.
    dollar_limit: float = 0.0
    # Rollover window for the dollar cap: "none" (lifetime) | "daily" |
    # "weekly" | "monthly". Mirrors the per-agent budget period vocabulary.
    dollar_period: str = "none"
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

    Requires ``pip install autonet-computer[voice]`` at minimum.
    """
    enabled: bool = False       # voice service active on startup
    backend: str = "kokoro"     # TTS backend: kokoro, edge, elevenlabs, piper
    ptt_keys: list[str] = field(default_factory=lambda: ["page down", "insert"])
    mute_key: str = "page up"
    voice_volume: float = 1.0
    tools_volume: float = 0.55
    effects_volume: float = 0.35
    narrate_tools: bool = True
    # Tool narration and announcements speak in a second voice so the two
    # channels are easy to tell apart. `tools_backend = None` means "whatever
    # `backend` is", so the contrast stays a voice difference within the
    # selected backend instead of silently falling back to another backend.
    tools_voice: str = "am_michael"
    tools_backend: str | None = None
    announcements: list[str] = field(default_factory=lambda: [
        "agent_runs", "agent_created", "agent_completed", "delegate_lifecycle"
    ])
    output_device: str | None = None
    input_device: str | None = None
    kokoro_model_dir: str | None = None   # directory containing kokoro-v1.0.onnx
    piper_module_dir: str | None = None   # directory containing piper voice module
    # Security-alarm consumer: when True the alarm is spoken via Piper (offline,
    # always available), bypassing mute/backend selection so a tamper alert is
    # never swallowed by a cloud-TTS outage. Full monitor lands in the security
    # track; this flag governs the consumer stub already wired here.
    piper_mandatory_for_alarm: bool = True
    # Reserved for the optional local audio overlay (owner decision pending); the
    # overlay module is NOT built yet. Kept here so config stays forward-stable.
    local_overlay: bool = False


@dataclass
class ChatConfig:
    """Configuration for the chat service (Discord etc).

    Binds agent conversations to a chat platform. The daemon constructs the
    platform adapter from this config and auto-starts the service when enabled,
    same as the voice service. The bot token is read from an env var (named
    here) rather than stored in config — secrets stay out of the file.
    """
    enabled: bool = False           # chat service active on startup
    platform: str = "discord"       # adapter to use (discord; more later)
    token_env: str = "DISCORD_BOT_TOKEN"  # env var holding the bot token
    channel_id: str = ""            # THE channel bound to the agent below
    # The agent this channel talks to. Required when chat is enabled: there
    # is no implicit default agent.
    bound_agent: str = ""
    # Input-seam gating policy: "open" (AllowAll), "operator" (only operator_ids),
    # or "credit" (rolling per-user credits; operators unlimited).
    policy: str = "open"
    operator_ids: list[str] = field(default_factory=list)
    credit_window_secs: int = 86400   # credit: rolling window
    credit_default_limit: int = 5     # credit: requests/window for an unconfigured user
    root_label: str = "K3V|N"
    excluded_agents: list[str] = field(default_factory=list)  # agents never rendered


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
    # Native gas token — published so the UI can label balances correctly
    # without hardcoding "ETH". Defaults match Etherlink Shadownet.
    gas_symbol: str = "XTZ"
    gas_decimals: int = 18
    private_key: str = ""               # Hex private key for signing attestation txns
    # Wallet is managed externally (MetaMask etc.) — we just track the address
    wallet_address: str = ""            # Connected wallet address (empty = not connected)
    # --- Remote-frontend auth (WS server) ------------------------------------
    # The MetaMask wallet that OWNS this daemon. Distinct from wallet_address
    # (the transient network-connection wallet) and from any agent's generated
    # identity.json key. A remote connection that signs the auth challenge
    # with this address is rooted at the full fleet.
    # MUST be pre-configured for the remote listener to start: there is NO
    # trust-on-first-use on a network-reachable socket (it would let the first
    # wallet to find the port claim ownership of the fleet).
    owner_wallet: str = ""
    # Port for the privileged LOCAL WS listener (bound to localhost). 0 => the
    # 7700 default. Configurable so a SECOND daemon can coexist on one machine
    # (a cross-daemon E2E, a staging instance beside a live one): the CLI used
    # to pass the 7700 literal and, on collision, kill whatever held the port,
    # so a second instance murdered the first. The remote listener still
    # defaults to local_ws_port + 1 when remote_ws_port is unset.
    local_ws_port: int = 0
    # Bind/port for the REMOTE (auth-required) WS listener. The privileged
    # local listener is 127.0.0.1:local_ws_port (7700 by default); this is the
    # separate socket remote/proxied clients reach. Empty host => remote
    # listener disabled (local-only daemon, today's default). A reverse proxy
    # (wss://autonet.computer -> proxy) points at this.
    remote_ws_host: str = ""        # e.g. "0.0.0.0"; empty = remote disabled
    remote_ws_port: int = 7701
    # INTEGRATION listener (third socket; docs/integration_listener.md): a
    # guest harness authenticates with a per-agent bearer token minted by
    # `atn integration-token create` and is clamped to that one agent.
    # Off by default. Env ATN_INTEGRATION_WS=1 / ATN_INTEGRATION_WS_HOST /
    # ATN_INTEGRATION_WS_PORT override (the Docker sidecar binds 0.0.0.0 on
    # an internal network).
    integration_ws_enabled: bool = False
    integration_ws_host: str = "127.0.0.1"
    integration_ws_port: int = 7710
    integration_rate_per_sec: float = 5.0
    integration_burst: int = 20
    # The browser-reachable wss:// URL this daemon advertises so a remote
    # frontend can find it by an agent's 0x address. Behind a reverse proxy the
    # daemon can't infer its own public URL, so it must be configured here
    # (e.g. "wss://autonet.computer/ws"). Empty => don't publish reachability
    # (local-only daemon). Written to the Firestore agent directory at startup;
    # see atn/agent_directory.py. The chain owns identity/registration; this is
    # mutable presence data, so it lives in Firestore, not on-chain.
    public_ws_endpoint: str = ""
    # Firestore project the agent directory lives in. Empty => use the ambient
    # GOOGLE_CLOUD_PROJECT / default credentials. Reachability publishing is
    # skipped entirely if no public_ws_endpoint is set.
    firestore_project: str = ""
    # Jurisdiction entry point — defaults from atn.jurisdiction
    dao_address: str = ""               # Defaults to jurisdiction.GOVERNOR_ADDRESS
    # RPB-specific fields
    jurisdiction_id: str = "autonet"    # First jurisdiction
    rpb_contract_address: str = ""      # Legacy: pre-substrate RPB address (kept
                                        # so existing configs still parse). New
                                        # configs should use substrate_address.
    substrate_address: str = ""         # Deployed Substrate.sol address. Read by
                                        # OnChainService; falls back to
                                        # rpb_contract_address if empty.
    rep_token_address: str = ""         # Deployed RepToken.sol (DAO) address. The
                                        # federated close reads REP (voice) share
                                        # from its ERC20Votes checkpoints, pinned to
                                        # the prev anchor timestamp (Decision
                                        # 2026-07-10). Empty => genesis regime.
    charter_hash: str = ""              # The jurisdiction's governed charter
                                        # version: registry key
                                        # jurisdiction.<id>.charter (0x + sha256
                                        # hex), changed by DAO proposal. The
                                        # daemon compares its local charter_hash
                                        # against it and warns on drift.
    charter_uri: str = ""               # Legacy (pre 2026-10-17): the charter
                                        # text lives in the Registry now.
    charter_text: str = ""              # jurisdiction.<id>.charter: the charter
                                        # JSON {"axes":[{"id","thesis"}]}, exact
                                        # registry bytes (charter_hash = their
                                        # sha256). docs/multi_jurisdiction.md 5.
    anchor_election_address: str = ""   # AnchorElection: epoch anchors are
                                        # elections (registry anchor_election).
    deploy_block: int = 0               # Block the jurisdiction's contracts
                                        # were deployed at; every close log
                                        # scan starts here. 0 = search.
    epoch_rules: str = ""               # jurisdiction.<id>.epoch_rules: the
                                        # epoch computation version the DAO's
                                        # network runs. Empty = "1".
    dao_registry_address: str = ""      # The DAO's governance Registry (the
                                        # jurisdiction's key-value store).
    tool_bond_address: str = ""         # ToolBond (trustless-contracts).
    registry_address: str = ""          # Discovered from RepToken.registryAddress()
                                        # — the DAO governance Registry, populated
                                        # at runtime by _discover_jurisdiction().
                                        # NOTE: the registry.json seed historically
                                        # also lands ServiceMarket's ServiceRegistry
                                        # here for back-compat; prefer the clearly
                                        # named ``service_registry_address`` below
                                        # for the ServiceMarket rail.
    service_registry_address: str = ""  # Deployed ServiceMarket ServiceRegistry
                                        # address (ServiceMarket.sol). Seeded from
                                        # registry.json contracts.service_registry.
                                        # Distinct from ``registry_address`` (the
                                        # DAO Registry) to avoid the collision the
                                        # shared name caused.
    payment_channel_address: str = ""   # Deployed ServiceMarket PaymentChannel
                                        # address (ServiceMarket.sol). Seeded from
                                        # registry.json contracts.payment_channel.
                                        # The settlement rail for services.
    token_address: str = ""             # Discovered from Governor.token()
    economy_address: str = ""           # Discovered from RepToken.economyAddress()
    timelock_address: str = ""          # Discovered from Governor.timelock()
    min_alignment_threshold: float = 0.5
    generate_keypairs: bool = True
    # Sponsor mode — advertise willingness to proxy inference for dependents
    sponsor_inference: bool = False
    # Provider to use for sponsored inference (e.g. "anthropic", "openai").
    # Empty = use same provider resolution as local agents.
    sponsor_provider: str = ""
    sponsor_model: str = ""  # Model to serve (empty = accept any model request)
    # Dependent mode — the 0x address of the sponsor THIS daemon routes its
    # rpb inference through. Daemon-level, not per agent: the dependent
    # identity is the owner wallet, and a daemon has at most one sponsor
    # (ratified 2026-07-25, docs/sponsored_inference.md). Empty = not bound to
    # a sponsor; the rpb provider falls back to open discovery.
    sponsor_address: str = ""
    # Input-seam gating policy for the WebSocket surface: "allow" (AllowAll,
    # today's default — every connected client may drive its scoped agents) or
    # "single_writer" (only the arbiter-elected active input surface may send).
    # Mirrors chat.policy; the single-writer decision is enforced by the
    # runtime-owned InputArbiter, not by this policy object.
    ws_input_policy: str = "allow"
    # Runtime-only, never read from or written to config.yaml. The daemon
    # stays local until it joins the network (first on-chain registration or
    # ``enabled``); resolve_network_registry() then fills the chain addresses
    # from the packaged registry and sets ``network_joined``.
    network_joined: bool = False
    # Keys the operator set in config.yaml; the registry never replaces them.
    explicit_fields: list[str] = field(default_factory=list, repr=False)

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
          include_user_data: false   # set true to include user-facing sessions
          min_turns: 1               # quality filter: skip near-empty sessions
    """
    enabled: bool = False
    trace_dir: str = ""             # empty → {data_dir}/traces/
    include_user_data: bool = False  # consent gate (Epic 3 Story 3.1)
    min_turns: int = 1              # minimum assistant turns for quality filter


@dataclass
class AutoUpdateConfig:
    """Silent daemon auto-update (ON by default; set enabled: false to opt out).

    Default-on is load-bearing for consensus: federated closes fork on a
    mixed fleet, so laggard daemons are a network hazard, not just a stale
    install. Staging is the safe half — the daemon never restarts itself.

    When ``enabled``, a background task polls the release source (default
    PyPI) on ``check_interval_secs``. A newer release is downloaded, its
    wheel hash verified, and (advisory) checked against the on-chain core
    hash, then *staged* to ``{data_dir}/staged_update/``. The running
    daemon is never restarted — the staged wheel is installed on the next
    daemon boot, before any heavy modules load, then the process re-execs
    once into the new code.

    This is legitimate rather than sneaky only because control of the
    official codebase is meant to be decentralized via reputation /
    governance. V1 trusts PyPI as the version pointer and the published
    wheel hash (plus the on-chain core hash when present); the source of
    the version pointer can later move to a governance-approved on-chain
    pointer without changing the stage/apply machinery.

    Example config.yaml snippet::

        auto_update:
          enabled: true
          check_interval_secs: 86400   # daily
          source: pypi                 # pypi | git | http | blob_store
          pypi_index_url: ""           # "" → pypi.org default
          package_name: autonet-computer
    """
    enabled: bool = True
    check_interval_secs: int = 86400     # daily
    source: str = "pypi"                 # pypi | git | http | blob_store
    pypi_index_url: str = ""             # "" → pypi.org JSON API default
    package_name: str = "autonet-computer"


@dataclass
class WorkerIsolationConfig:
    """Agent per-PID process isolation (default ON since 2026-10-13).

    When ``enabled`` (env ``ATN_WORKER_ISOLATION`` or config
    ``worker_isolation.enabled``), the execution engine will spawn each
    cognitive agent in its OWN OS process (``atn.agent_worker``) so the kernel
    PID becomes a real security identity for PID-bound secret access control.

    As of P4 this is a real cutover for cognitive agents that use an API
    provider (Anthropic / OpenAI-compatible): with the flag ON the provider loop
    + local sandboxed tools run IN THE WORKER PROCESS, while authority tools,
    events, status, and budget booking cross the IPC seam back to the daemon.
    The Claude-Max bridge is worker-eligible as of P5 (the worker owns its node
    SDK child); ``codex_max`` and rpb/substrate composite providers still run
    in-process, as does delegate SPAWN (P6): a worker that tries to create a
    child issues a spawn_child RPC instead. See
    ``atn.runtime.execution_engine.ExecutionEngine._worker_eligible`` for the
    single source of truth. With the flag OFF ``trigger_run`` takes the
    in-process asyncio path. Default ON (docs/tool_substrate.md, Decision
    2026-10-13): third-party tool code must not share the daemon's process.

    The flag is read at ``load_config`` time from EITHER the env var (takes
    precedence, so an operator can force it without editing config) OR the
    config file. Any truthy string ("1", "true", "yes", "on") enables it.
    """
    enabled: bool = True
    # Optional per-worker memory cap (MiB) enforced via the Win32 Job Object
    # (JOB_OBJECT_LIMIT_PROCESS_MEMORY). 0 => no cap. POSIX: advisory only in
    # this phase (no Job Object).
    memory_cap_mb: int = 0


@dataclass
class SecretsConfig:
    """Owner config for the secret-allowance / tripwire track (default OFF).

    Governs how the daemon resolves the ROOT (parent-less / user-created)
    agent's secret allowance. Everything here is FAIL-CLOSED by default: a
    root agent gets NOTHING unless the owner explicitly opts in.

    ``default_root_allowance`` is a resolve_spec string (kevin/keystore.py
    form): "none" (default; deny-all) or "all" (grant every vault service,
    unbounded / picks up new services) or a comma-separated bundle/literal
    spec. This is the L_parent seed for agents with no parent allowance to
    intersect against, applied by
    ``atn.runtime.worker_host.resolve_effective_grant``; read only when
    worker_isolation is also enabled and a real grant is requested.
    """
    default_root_allowance: str = "none"


@dataclass
class RobotConfig:
    """This daemon runs a robot (docs/robots.md). Off unless ``enabled``.

    ``bundle_dir`` is where the daemon exports the required safety packs and
    writes its heartbeat for the robot's filter node. ``required_packs`` are
    tool names or digests in this daemon's tool store (the jurisdiction's
    standards, from config until they move to the Registry). ``bag_dir`` holds
    the robot's MCAP worklog chunks, one directory per stream, which its
    signed worklog chain commits.
    ``agent_id`` is the robot's own agent; its key signs attestations.
    """
    enabled: bool = False
    agent_id: str = ""
    bundle_dir: str = ""
    bag_dir: str = ""
    required_packs: list[str] = field(default_factory=list)
    heartbeat_interval_s: float = 1.0


@dataclass
class RobotPollConfig:
    """The robot poll rail (docs/robots.md, Decision 2026-10-25). Off
    unless ``enabled``; needs the autonet network (p2p and chain). Every
    local agent may then be drawn as a poller, and the robot agent (when
    ``robot.enabled``) commits its worklog and answers polls.

    All values are provisional defaults for the jurisdiction to set:
    ``pollers`` per robot per round, the REP ``percentile`` floor, the
    round length, the answer deadline, how long a watched robot may stay
    silent, and per-stream retention in rounds (a stream not listed is
    kept).
    """
    enabled: bool = False
    period_s: float = 3600.0
    pollers: int = 3
    percentile: int = 90
    deadline_s: float = 2.0
    commit_gap_s: float = 30.0
    retention_rounds: dict[str, int] = field(default_factory=dict)


@dataclass
class ATNConfig:
    """Top-level ATN configuration."""
    data_dir: Path = field(default_factory=lambda: _DEFAULT_DIR)
    agents_dir: Path = field(default_factory=lambda: _default_agents_dir())
    # The config.yaml this was loaded from (None when built in code).
    source_path: Path | None = None
    # LEGACY-DATA: the retired top-level ``defaults:`` section ({model,
    # provider}), read only so the one-time agent-pinning migration
    # (atn/legacy_routing_migration.py) can pin unpinned agents to what they
    # resolved to before agents had to choose. Nothing else reads it; the
    # migration removes the section from config.yaml.
    legacy_agent_routing: dict[str, str] = field(default_factory=dict)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)
    autonet: AutonetConfig = field(default_factory=AutonetConfig)
    trace_logging: TraceLoggingConfig = field(default_factory=TraceLoggingConfig)
    auto_update: AutoUpdateConfig = field(default_factory=AutoUpdateConfig)
    worker_isolation: WorkerIsolationConfig = field(default_factory=WorkerIsolationConfig)
    secrets: SecretsConfig = field(default_factory=SecretsConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    robot_poll: "RobotPollConfig" = field(default_factory=lambda: RobotPollConfig())
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    connectors: dict[str, ConnectorConfig] = field(default_factory=dict)
    # Seconds an MCP connector may sit idle before the daemon stops it.
    # Connectors are long-lived server processes started lazily and (before
    # this) kept until daemon shutdown, so a connector used once held RAM
    # forever. Pinned tools need no equivalent — they exit per call.
    # <= 0 disables reaping (keep alive forever, the pre-reaper behavior).
    connector_idle_timeout_s: float = 900.0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def rpb(self) -> AutonetConfig:
        """Alias — RPB config is the autonet config."""
        return self.autonet


# ---------------------------------------------------------------------------
# Network registry: public chain addresses, resolved lazily
# ---------------------------------------------------------------------------
#
# registry.json ships INSIDE the package (atn/registry.json, package data),
# pinned per release; tests/atn/test_config_registry.py keeps it identical to
# the repo-root copy of record. Nothing is fetched at boot: the daemon stays
# fully local until it joins the network, which happens when the first agent
# registers on chain or when ``autonet.enabled`` is set. Only then is the
# registry read (resolve_network_registry) and its addresses filled into the
# autonet config. user config.yaml values (blockchain/rpb/autonet sections)
# still win over it.
#
# An operator may point ATN_REGISTRY_URL at a registry document (a private
# fork, a test network). It is fetched at join time, never at boot, and the
# packaged copy is the fallback if it is unreachable.

# Read before joining by find_services / market_services / network_status.
NOT_JOINED_MESSAGE = "Not joined: register an agent to join the network."
# Short worst-case wait for an ATN_REGISTRY_URL override fetch.
_REGISTRY_FETCH_TIMEOUT = 4.0


def not_joined_state() -> dict[str, Any]:
    """The payload a read-only chain call returns before the daemon joined."""
    return {"joined": False, "status": "not_joined",
            "message": NOT_JOINED_MESSAGE}


def _packaged_registry_path() -> Path:
    """The registry.json shipped with this release (package data)."""
    return Path(__file__).resolve().parent / "registry.json"


def _registry_override_url() -> str:
    """The operator's ATN_REGISTRY_URL, or "" when unset."""
    return os.environ.get("ATN_REGISTRY_URL", "").strip()


def _parse_registry_data(data: dict[str, Any], jurisdiction_id: str) -> dict[str, Any]:
    """Flatten a parsed registry document into an RPBConfig-mergeable seed.

    Empty dict if the jurisdiction isn't listed.
    """
    entry = data.get("jurisdictions", {}).get(jurisdiction_id)
    if not entry:
        return {}
    seed: dict[str, Any] = {"jurisdiction_id": jurisdiction_id}
    net = entry.get("network", {})
    if net.get("rpc_url"):
        seed["rpc_url"] = net["rpc_url"]
    if net.get("chain_id"):
        seed["chain_id"] = net["chain_id"]
    if net.get("gas_symbol"):
        seed["gas_symbol"] = net["gas_symbol"]
    if net.get("gas_decimals") is not None:
        seed["gas_decimals"] = int(net["gas_decimals"])
    contracts = entry.get("contracts", {})
    if contracts.get("dao"):
        seed["dao_address"] = contracts["dao"]
    if contracts.get("rpb"):
        seed["rpb_contract_address"] = contracts["rpb"]
    if contracts.get("substrate"):
        seed["substrate_address"] = contracts["substrate"]
    if contracts.get("rep_token"):
        seed["rep_token_address"] = contracts["rep_token"]
    if contracts.get("deploy_block"):
        seed["deploy_block"] = int(contracts["deploy_block"])
    charter = contracts.get("charter") or ""
    if charter.lstrip().startswith("{"):
        # The charter text itself (2026-10-17). Hash the exact registry
        # bytes, never a re-serialization, so every daemon agrees.
        seed["charter_text"] = charter
        seed["charter_hash"] = "0x" + hashlib.sha256(
            charter.encode("utf-8")).hexdigest()
    elif charter:
        seed["charter_hash"] = charter
    if contracts.get("charter_uri"):
        seed["charter_uri"] = contracts["charter_uri"]
    if contracts.get("epoch_rules"):
        seed["epoch_rules"] = str(contracts["epoch_rules"]).strip()
    if contracts.get("timelock"):
        seed["timelock_address"] = contracts["timelock"]
    # On chain the key is "registry"; registry.json calls it "dao_registry".
    dao_registry = contracts.get("dao_registry") or contracts.get("registry")
    if dao_registry:
        seed["dao_registry_address"] = dao_registry
    if contracts.get("economy"):
        seed["economy_address"] = contracts["economy"]
    if contracts.get("service_registry"):
        # Populate the clearly-named field AND the legacy shared name, so
        # existing consumers of ``registry_address`` (the audit noted this
        # collides with the DAO Registry) keep working until they migrate,
        # while new code reads the unambiguous ``service_registry_address``.
        seed["service_registry_address"] = contracts["service_registry"]
        seed["registry_address"] = contracts["service_registry"]
    if contracts.get("payment_channel"):
        seed["payment_channel_address"] = contracts["payment_channel"]
    if contracts.get("anchor_election"):
        seed["anchor_election_address"] = contracts["anchor_election"]
    if contracts.get("tool_bond"):
        # ToolBond (publish stakes, core-replacement bond gate).
        seed["tool_bond_address"] = contracts["tool_bond"]
    return seed


# A jurisdiction does not start unless every one of these resolved
# (docs/multi_jurisdiction.md decision 3). A missing RepToken would
# otherwise run the close in the genesis regime, and daemons that disagree
# on it fork.
REQUIRED_JURISDICTION_FIELDS = (
    ("substrate_address", "substrate"),
    ("rep_token_address", "rep_token"),
    ("tool_bond_address", "tool_bond"),
    ("dao_address", "dao (governor)"),
    ("timelock_address", "timelock"),
)

# Epoch rules versions this build computes (decision 4). Bumped only when
# the epoch computation changes; feature releases leave it alone.
SUPPORTED_EPOCH_RULES = frozenset({"1"})


def missing_prerequisites(an: "RPBConfig") -> list[str]:
    """Names of the required jurisdiction addresses ``an`` lacks."""
    return [label for field_name, label in REQUIRED_JURISDICTION_FIELDS
            if not (getattr(an, field_name, "") or "").strip()]


def epoch_rules_error(an: "RPBConfig") -> str:
    """Why this build can't join ``an``'s jurisdiction, or "" if it can."""
    rules = (getattr(an, "epoch_rules", "") or "1").strip()
    if rules not in SUPPORTED_EPOCH_RULES:
        return (f"the jurisdiction runs epoch rules version {rules}; this "
                f"daemon computes {sorted(SUPPORTED_EPOCH_RULES)}. Upgrade to "
                "a build that supports it.")
    return ""


def charter_text_error(text: str) -> tuple[bool, str]:
    """(equals the built-in v1 charter, why this build can't run it or "").

    Any charter ``Charter.from_text`` accepts runs (each DAO defines its own
    axes, docs/multi_jurisdiction.md decision 5): valid compact JSON, at
    least one axis, distinct ids, and the mandatory correctness and
    simplicity axes."""
    from nodes.common.world_model_substrate.charter import (
        DEFAULT_CHARTER,
        Charter,
    )
    try:
        charter = Charter.from_text(text)
    except ValueError as e:
        return False, f"the jurisdiction's charter is invalid: {e}"
    return charter.hash == DEFAULT_CHARTER.hash, ""


def check_jurisdiction(dao_registry: str, rpc_url: str,
                       jurisdiction_id: str = "autonet") -> dict[str, Any]:
    """Read a candidate jurisdiction's Registry and say whether this daemon
    can join it: {ok, errors, chain_id, dao, name, rpc_url, dao_registry}.
    Network read; used before switching."""
    data = _fetch_onchain_registry(dao_registry, rpc_url, jurisdiction_id)
    if data is None:
        return {"ok": False, "dao_registry": dao_registry, "rpc_url": rpc_url,
                "errors": [f"no autonet jurisdiction readable at {dao_registry}"]}
    seed = _parse_registry_data(data, jurisdiction_id)
    an = RPBConfig()
    for k, v in seed.items():
        if hasattr(an, k):
            setattr(an, k, v)
    errors: list[str] = []
    missing = missing_prerequisites(an)
    if missing:
        errors.append("missing jurisdiction addresses: " + ", ".join(missing))
    rules = epoch_rules_error(an)
    if rules:
        errors.append(rules)
    if an.charter_text:
        _match, charter = charter_text_error(an.charter_text)
        if charter:
            errors.append(charter)
    return {"ok": not errors, "errors": errors, "chain_id": an.chain_id,
            "dao": an.dao_address, "dao_registry": dao_registry,
            "rpc_url": rpc_url, "key": jurisdiction_key(an),
            "name": seed.get("name", "")}


def jurisdiction_key(an: "RPBConfig") -> str:
    """``<chain_id>:<DAO address>``: the jurisdiction's identity, used as its
    gossip topic and the key its local substrate data is parked under."""
    dao = (getattr(an, "dao_address", "") or "").strip().lower()
    chain_id = int(getattr(an, "chain_id", 0) or 0)
    if not dao or not chain_id:
        return ""
    return f"{chain_id}:{dao}"


def _fetch_registry(url: str) -> dict[str, Any] | None:
    """Fetch a registry document from ``url``. None on any failure; never
    raises. Only used for the ATN_REGISTRY_URL override."""
    try:
        import httpx

        resp = httpx.get(url, timeout=_REGISTRY_FETCH_TIMEOUT,
                         follow_redirects=True)
        resp.raise_for_status()
        data = json.loads(resp.text)
    except Exception as exc:
        log.debug("Registry fetch from %s failed: %s", url, exc)
        return None
    return data if isinstance(data, dict) else None


_REGISTRY_READ_ABI = [
    {"name": "getAllKeys", "type": "function", "stateMutability": "view",
     "inputs": [], "outputs": [{"name": "", "type": "string[]"}]},
    {"name": "getAllValues", "type": "function", "stateMutability": "view",
     "inputs": [], "outputs": [{"name": "", "type": "string[]"}]},
]


# ---------------------------------------------------------------------------
# The selected jurisdiction (docs/multi_jurisdiction.md decision 1)
# ---------------------------------------------------------------------------
#
# A daemon participates in one jurisdiction at a time. The owner's choice is
# saved as <data_dir>/jurisdiction.json: {chain_id, rpc_url, dao_registry,
# dao, name}. Switching rewrites it and restarts the daemon. Without it the
# daemon uses the packaged registry.json's jurisdiction.

SELECTION_FILE = "jurisdiction.json"


def _selection_path(data_dir: Path | None = None) -> Path:
    return Path(data_dir or _DEFAULT_DIR) / SELECTION_FILE


def load_selected_jurisdiction(data_dir: Path | None = None) -> dict[str, Any] | None:
    """The saved jurisdiction selection, or None when there is none (or it
    is unreadable, which is logged)."""
    path = _selection_path(data_dir)
    if not path.is_file():
        return None
    try:
        sel = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        log.warning("jurisdiction selection %s unreadable; ignoring it", path)
        return None
    if not isinstance(sel, dict) or not sel.get("dao_registry") or not sel.get("chain_id"):
        log.warning("jurisdiction selection %s incomplete; ignoring it", path)
        return None
    return sel


def save_selected_jurisdiction(sel: dict[str, Any], data_dir: Path | None = None) -> Path:
    """Write the selection atomically. Takes effect at the next start."""
    path = _selection_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({
        "chain_id": int(sel["chain_id"]),
        "rpc_url": str(sel.get("rpc_url") or ""),
        "dao_registry": str(sel["dao_registry"]),
        "dao": str(sel.get("dao") or ""),
        "name": str(sel.get("name") or ""),
    }, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def boot_jurisdiction_key(data_dir: Path | None = None) -> str:
    """``<chain_id>:<DAO address>`` of the jurisdiction this process runs,
    known at boot without touching the network: the saved selection, else
    the packaged registry.json's entry. "" when neither names a DAO."""
    sel = load_selected_jurisdiction(data_dir)
    if sel is not None:
        dao, chain_id = str(sel.get("dao") or ""), int(sel.get("chain_id") or 0)
    else:
        try:
            data = json.loads(_packaged_registry_path().read_text(encoding="utf-8"))
            entry = data["jurisdictions"]["autonet"]
            dao = str(entry.get("contracts", {}).get("dao") or "")
            chain_id = int(entry.get("network", {}).get("chain_id") or 0)
        except Exception:
            return ""
    if not dao or not chain_id:
        return ""
    return f"{chain_id}:{dao.strip().lower()}"


def _onchain_registry_target() -> tuple[str, str]:
    """(DAO Registry address, rpc url) for a jurisdiction launched on chain:
    ATN_JURISDICTION_REGISTRY / ATN_JURISDICTION_RPC, else the saved
    jurisdiction selection. The rpc falls back to the packaged network's.
    ("", "") when neither is set."""
    addr = os.environ.get("ATN_JURISDICTION_REGISTRY", "").strip()
    if not addr:
        sel = load_selected_jurisdiction(_selection_data_dir())
        if sel is None:
            return "", ""
        return str(sel["dao_registry"]), str(sel.get("rpc_url") or "")
    rpc = os.environ.get("ATN_JURISDICTION_RPC", "").strip()
    if not rpc:
        from .jurisdiction import RPC_URL
        rpc = RPC_URL
    return addr, rpc


def _fetch_onchain_registry(registry_address: str, rpc_url: str,
                            jurisdiction_id: str) -> dict[str, Any] | None:
    """Read a launched jurisdiction's contracts from its DAO Registry.

    A jurisdiction launched in one transaction (the werule "Autonet
    jurisdiction" type) writes its suite under ``jurisdiction.<id>.<name>``
    keys, with the same names registry.json uses under ``contracts``. Returns
    a registry document shaped like registry.json, or None on any failure.
    """
    try:
        from web3 import Web3
        w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={
            "timeout": _REGISTRY_FETCH_TIMEOUT}))
        reg = w3.eth.contract(address=Web3.to_checksum_address(registry_address),
                              abi=_REGISTRY_READ_ABI)
        keys = reg.functions.getAllKeys().call()
        values = reg.functions.getAllValues().call()
        chain_id = int(w3.eth.chain_id)
    except Exception as exc:
        log.debug("On-chain registry read from %s failed: %s", registry_address, exc)
        return None
    prefix = f"jurisdiction.{jurisdiction_id}."
    contracts = {k[len(prefix):]: v for k, v in zip(keys, values)
                 if k.startswith(prefix) and v}
    if not contracts.get("substrate"):
        return None
    contracts.setdefault("dao_registry", registry_address)
    return {"jurisdictions": {jurisdiction_id: {
        "network": {"rpc_url": rpc_url, "chain_id": chain_id},
        "contracts": contracts}}}


def _load_registry_seed(jurisdiction_id: str = "autonet") -> dict[str, Any]:
    """Resolve network + contract defaults for ``jurisdiction_id``.

    Called only when the daemon joins the network, never at boot:
      1. ATN_REGISTRY_URL, when the operator set it (fetched now);
      2. else ATN_JURISDICTION_REGISTRY, a launched jurisdiction's DAO
         Registry read on chain (jurisdiction.<id>.* keys);
      3. else (or if those fail) the packaged registry.json;
      4. else {} with one warning.

    Returns a flat dict suitable for merging into the RPBConfig builder.
    """
    data: dict[str, Any] | None = None
    source = "packaged"
    url = _registry_override_url()
    if url:
        data = _fetch_registry(url)
        if data is None:
            log.warning("ATN_REGISTRY_URL %s unreachable; using the packaged "
                        "registry", url)
        else:
            source = url
    if data is None:
        reg_addr, reg_rpc = _onchain_registry_target()
        if reg_addr:
            data = _fetch_onchain_registry(reg_addr, reg_rpc, jurisdiction_id)
            if data is None:
                if load_selected_jurisdiction(_selection_data_dir()) is not None:
                    # The owner chose this DAO: never quietly join the
                    # packaged one instead. Nothing resolves, so the
                    # jurisdiction refuses to start and says why.
                    log.warning("Selected jurisdiction registry %s unreadable on "
                                "chain; not joining any network", reg_addr)
                    return {}
                log.warning("Jurisdiction registry %s unreadable on chain; using "
                            "the packaged registry", reg_addr)
            else:
                source = f"chain:{reg_addr}"
                sel = load_selected_jurisdiction(_selection_data_dir())
                contracts = data["jurisdictions"][jurisdiction_id]["contracts"]
                if sel is not None and sel.get("dao"):
                    contracts.setdefault("dao", sel["dao"])
    if data is None:
        path = _packaged_registry_path()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            log.warning("network registry unavailable: packaged %s missing or "
                        "unreadable, chain features off", path)
            return {}
    try:
        seed = _parse_registry_data(data, jurisdiction_id)
    except Exception:
        log.warning("Failed to parse registry (%s)", source, exc_info=True)
        return {}
    log.info("Registry seed (%s) for '%s': dao=%s, substrate=%s",
             source, jurisdiction_id,
             (seed.get("dao_address") or "")[:10] or "(none)",
             (seed.get("substrate_address") or "")[:10] or "(none)")
    return seed


# The data dir whose jurisdiction.json the registry resolution reads. Set by
# the runtime at boot (tests point it elsewhere); None = ~/.atn.
_SELECTION_DATA_DIR: Path | None = None


def set_selection_data_dir(data_dir: Path | None) -> None:
    global _SELECTION_DATA_DIR
    _SELECTION_DATA_DIR = Path(data_dir) if data_dir else None


def _selection_data_dir() -> Path | None:
    return _SELECTION_DATA_DIR


def _seed_defaults() -> dict[str, Any]:
    """Values an RPBConfig holds when nobody set them (dataclass defaults and
    the atn.jurisdiction fallbacks). The registry may replace these."""
    from .jurisdiction import GOVERNOR_ADDRESS, RPC_URL, CHAIN_ID
    return {"rpc_url": RPC_URL, "chain_id": CHAIN_ID,
            "dao_address": GOVERNOR_ADDRESS, "gas_symbol": "XTZ",
            "gas_decimals": 18, "jurisdiction_id": "autonet"}


def resolve_network_registry(an: "RPBConfig", *, force: bool = False) -> bool:
    """Join the network: fill ``an`` (in place) from the registry.

    Fields the operator set explicitly (config.yaml or the RPBConfig
    constructor) are never replaced. Marks ``an.network_joined``. Returns
    True if the registry yielded any addresses. Idempotent unless ``force``.
    """
    if getattr(an, "network_joined", False) and not force:
        return True
    seed = _load_registry_seed(getattr(an, "jurisdiction_id", "") or "autonet")
    explicit = set(getattr(an, "explicit_fields", ()) or ())
    defaults = _seed_defaults()
    for key, value in seed.items():
        if key in explicit or not hasattr(an, key):
            continue
        current = getattr(an, key)
        if current in ("", 0, None) or current == defaults.get(key):
            setattr(an, key, value)
    an.network_joined = True
    return bool(seed)


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


def _load_worker_isolation(wi_raw: dict[str, Any]) -> WorkerIsolationConfig:
    """Build the worker-isolation config, with env override taking precedence.

    ``ATN_WORKER_ISOLATION`` (if set) wins over the config file so an operator
    can force the flag on/off without editing config.yaml. Any truthy string
    ("1"/"true"/"yes"/"on", case-insensitive) enables it.
    """
    wi_enabled = bool(wi_raw.get("enabled", True))
    env_iso = os.environ.get("ATN_WORKER_ISOLATION")
    if env_iso is not None:
        wi_enabled = env_iso.strip().lower() in ("1", "true", "yes", "on")
    try:
        memory_cap_mb = int(wi_raw.get("memory_cap_mb", 0))
    except (TypeError, ValueError):
        memory_cap_mb = 0
    return WorkerIsolationConfig(enabled=wi_enabled, memory_cap_mb=memory_cap_mb)


def _load_robot_poll(raw: dict[str, Any]) -> RobotPollConfig:
    ret = raw.get("retention_rounds") or {}
    return RobotPollConfig(
        enabled=bool(raw.get("enabled", False)),
        period_s=float(raw.get("period_s", 3600.0) or 3600.0),
        pollers=int(raw.get("pollers", 3) or 3),
        percentile=int(raw.get("percentile", 90)),
        deadline_s=float(raw.get("deadline_s", 2.0) or 2.0),
        commit_gap_s=float(raw.get("commit_gap_s", 30.0) or 30.0),
        retention_rounds={str(k): int(v) for k, v in ret.items()} if isinstance(ret, dict) else {},
    )


def _load_robot(raw: dict[str, Any]) -> RobotConfig:
    packs = raw.get("required_packs") or []
    return RobotConfig(
        enabled=bool(raw.get("enabled", False)),
        agent_id=str(raw.get("agent_id") or ""),
        bundle_dir=str(raw.get("bundle_dir") or ""),
        bag_dir=str(raw.get("bag_dir") or ""),
        required_packs=[str(p) for p in packs] if isinstance(packs, list) else [],
        heartbeat_interval_s=float(raw.get("heartbeat_interval_s", 1.0) or 1.0),
    )


def _load_secrets(secrets_raw: dict[str, Any]) -> SecretsConfig:
    """Build the secrets config (secret-allowance / tripwire track).

    Fail-closed: an unset or non-string ``default_root_allowance`` falls back
    to "none" (deny-all root).
    """
    root = secrets_raw.get("default_root_allowance", "none")
    if not isinstance(root, str) or not root.strip():
        root = "none"
    return SecretsConfig(default_root_allowance=root.strip())


def _apply_auto_update_env(config: ATNConfig) -> None:
    """ATN_AUTO_UPDATE wins over the config file, with or without one. An
    immutable container image updates by rebuild, so the daemon image sets
    ATN_AUTO_UPDATE=0 (no PyPI polling from inside it)."""
    env = os.environ.get("ATN_AUTO_UPDATE", "").strip().lower()
    if env in ("0", "false", "no", "off"):
        config.auto_update.enabled = False
    elif env in ("1", "true", "yes", "on"):
        config.auto_update.enabled = True


def _load_autonet_config(raw: dict[str, Any]) -> RPBConfig:
    """The autonet section. Runs with or without a config file (``raw`` is
    ``{}`` then). The network registry is NOT read here: it is resolved when
    the daemon joins (resolve_network_registry), except that an explicit
    ``enabled: true`` joins at load time (a local file read, no network)."""
    # Autonet / RPB network layer
    # Merge priority (lowest to highest):
    #   registry.json (applied at join time) < blockchain < rpb < autonet
    autonet_raw = raw.get("autonet", {})
    if not isinstance(autonet_raw, dict):
        autonet_raw = {}
    rpb_raw = raw.get("rpb", {})
    if not isinstance(rpb_raw, dict):
        rpb_raw = {}
    blockchain_raw = raw.get("blockchain", {})
    if not isinstance(blockchain_raw, dict):
        blockchain_raw = {}
    merged: dict[str, Any] = {}
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
    # Phase 12: autonet starts on registration, not on boot — having a
    # dao_address/rpc_url configured is necessary but no longer sufficient.
    # Users who want eager startup (bootstrap nodes, services that always
    # participate) set ``autonet.enabled: true`` explicitly.
    enabled = resolved.get("enabled", False)
    an = RPBConfig(
        enabled=enabled,
        config_path=resolved.get("config_path", ""),
        rpc_url=resolved.get("rpc_url", ""),
        chain_id=resolved.get("chain_id", 0),
        gas_symbol=resolved.get("gas_symbol", "XTZ"),
        gas_decimals=int(resolved.get("gas_decimals", 18)),
        private_key=resolved.get("private_key", ""),
        wallet_address=resolved.get("wallet_address", ""),
        dao_address=resolved.get("dao_address", ""),
        jurisdiction_id=resolved.get("jurisdiction_id", "autonet"),
        rpb_contract_address=resolved.get("rpb_contract_address", ""),
        substrate_address=resolved.get("substrate_address", ""),
        rep_token_address=resolved.get("rep_token_address", ""),
        charter_hash=resolved.get("charter_hash", ""),
        charter_uri=resolved.get("charter_uri", ""),
        charter_text=resolved.get("charter_text", ""),
        epoch_rules=str(resolved.get("epoch_rules", "") or ""),
        deploy_block=int(resolved.get("deploy_block", 0) or 0),
        anchor_election_address=resolved.get("anchor_election_address", ""),
        dao_registry_address=resolved.get("dao_registry_address", ""),
        tool_bond_address=resolved.get("tool_bond_address", ""),
        registry_address=resolved.get("registry_address", ""),
        service_registry_address=resolved.get("service_registry_address", ""),
        payment_channel_address=resolved.get("payment_channel_address", ""),
        token_address=resolved.get("token_address", ""),
        economy_address=resolved.get("economy_address", ""),
        timelock_address=resolved.get("timelock_address", ""),
        min_alignment_threshold=resolved.get("min_alignment_threshold", 0.5),
        generate_keypairs=resolved.get("generate_keypairs", True),
        # Sponsored inference (docs/sponsored_inference.md). These were
        # dataclass-only until now — never read from config.yaml, so sponsor
        # mode could not actually be configured on disk.
        sponsor_inference=bool(resolved.get("sponsor_inference", False)),
        sponsor_provider=resolved.get("sponsor_provider", ""),
        sponsor_model=resolved.get("sponsor_model", ""),
        sponsor_address=resolved.get("sponsor_address", ""),
        # Remote-frontend auth + reachability (this session's work).
        owner_wallet=resolved.get("owner_wallet", ""),
        local_ws_port=int(resolved.get("local_ws_port", 0) or 0),
        remote_ws_host=resolved.get("remote_ws_host", ""),
        remote_ws_port=int(resolved.get("remote_ws_port", 7701)),
        integration_ws_enabled=bool(resolved.get("integration_ws_enabled", False)),
        integration_ws_host=resolved.get("integration_ws_host", "127.0.0.1") or "127.0.0.1",
        integration_ws_port=int(resolved.get("integration_ws_port", 7710) or 7710),
        integration_rate_per_sec=float(resolved.get("integration_rate_per_sec", 5.0) or 5.0),
        integration_burst=int(resolved.get("integration_burst", 20) or 20),
        public_ws_endpoint=resolved.get("public_ws_endpoint", ""),
        firestore_project=resolved.get("firestore_project", ""),
        ws_input_policy=resolved.get("ws_input_policy", "allow"),
        explicit_fields=sorted(k for k, v in resolved.items()
                               if v not in ("", None)),
    )
    if enabled:
        resolve_network_registry(an)
    return an


def _migrate_legacy_orchestrator_config(path: Path) -> bool:
    """One-time rewrite of a config file written before the root-agent purge.

    Moves the top-level ``orchestrator:`` section (model/provider) to
    ``defaults:`` and renames ``chat.orchestrator_label`` to
    ``chat.root_label``. Text-level so comments and layout survive; falls
    back to a YAML rewrite only when both ``orchestrator:`` and
    ``defaults:`` exist (``defaults`` wins, the old section is dropped).
    Returns True when the file was rewritten.

    The resulting ``defaults:`` section is itself retired: load_config reads
    it into ``legacy_agent_routing`` and the one-time agent-pinning migration
    (atn/legacy_routing_migration.py) pins agents with it, then deletes it.
    """
    import re as _re
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    new = text
    has_old = _re.search(r"^orchestrator:", new, _re.M) is not None
    has_new = _re.search(r"^defaults:", new, _re.M) is not None
    if has_old and not has_new:
        new = _re.sub(r"^orchestrator:", "defaults:", new, flags=_re.M)
    if (_re.search(r"^\s+orchestrator_label:", new, _re.M)
            and not _re.search(r"^\s+root_label:", new, _re.M)):
        new = _re.sub(r"^(\s+)orchestrator_label:", r"\1root_label:", new,
                      flags=_re.M)
    if has_old and has_new:
        try:
            data = yaml.safe_load(new) or {}
        except Exception:
            return False
        if isinstance(data, dict):
            data.pop("orchestrator", None)
            chat = data.get("chat")
            if isinstance(chat, dict):
                chat.pop("orchestrator_label", None)
            new = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    if new == text:
        return False
    try:
        path.write_text(new, encoding="utf-8")
    except OSError as exc:
        log.warning("config: could not persist migrated config %s: %s",
                    path, exc)
        return False
    log.info("config: migrated legacy config keys in %s", path)
    return True


def load_config(path: Path | None = None) -> ATNConfig:
    """Load configuration from a YAML file.

    Falls back to sensible defaults if the file doesn't exist.
    """
    if path is None:
        path = _DEFAULT_DIR / "config.yaml"

    # Load ~/.atn/.env before config so ${VAR} interpolation can use .env values
    _load_dotenv()

    config = ATNConfig()
    # Env var ATN_WORKER_ISOLATION is authoritative and must apply even with no
    # config file (the early-return path below). The file-load path re-applies
    # it with the same precedence.
    config.worker_isolation = _load_worker_isolation({})
    _apply_auto_update_env(config)

    if not path.exists():
        log.info("No config file at %s — using defaults", path)
        config.autonet = _load_autonet_config({})
        return config

    _migrate_legacy_orchestrator_config(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        log.exception("Failed to read config from %s", path)
        return config

    config.raw = raw
    config.source_path = path
    config_dir = path.parent  # resolve relative paths against config file location

    if "data_dir" in raw:
        config.data_dir = _expand_path(raw["data_dir"])
        if not config.data_dir.is_absolute():
            config.data_dir = (config_dir / config.data_dir).resolve()
    if "agents_dir" in raw:
        # User explicitly set agents_dir — honor it verbatim (resolved against
        # the config file location when relative). Unaffected by the default.
        config.agents_dir = _expand_path(raw["agents_dir"])
        if not config.agents_dir.is_absolute():
            config.agents_dir = (config_dir / config.agents_dir).resolve()
    else:
        # No explicit agents_dir: re-derive the default so it tracks whatever
        # data_dir resolved to above (the dataclass default was computed with
        # the pre-override data_dir).
        config.agents_dir = _default_agents_dir(config.data_dir)

    # LEGACY-DATA: the retired ``defaults:`` section, kept only for the
    # one-time pinning migration (see ATNConfig.legacy_agent_routing).
    legacy_raw = raw.get("defaults", {})
    if isinstance(legacy_raw, dict) and legacy_raw:
        config.legacy_agent_routing = {
            "provider": str(legacy_raw.get("provider", "") or ""),
            "model": str(legacy_raw.get("model", "") or ""),
        }

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

    chat_raw = raw.get("chat", {})
    if isinstance(chat_raw, dict):
        config.chat = ChatConfig(
            enabled=chat_raw.get("enabled", False),
            platform=chat_raw.get("platform", "discord"),
            token_env=chat_raw.get("token_env", "DISCORD_BOT_TOKEN"),
            channel_id=str(chat_raw.get("channel_id", "")),
            bound_agent=str(chat_raw.get("bound_agent", "") or ""),
            policy=str(chat_raw.get("policy", "open")),
            operator_ids=[str(x) for x in chat_raw.get("operator_ids", [])],
            credit_window_secs=int(chat_raw.get("credit_window_secs", 86400)),
            credit_default_limit=int(chat_raw.get("credit_default_limit", 5)),
            root_label=chat_raw.get("root_label", "K3V|N"),
            excluded_agents=[str(x) for x in chat_raw.get("excluded_agents", [])],
        )

    config.autonet = _load_autonet_config(raw)

    # Trace logging
    trace_raw = raw.get("trace_logging", {})
    if isinstance(trace_raw, dict):
        config.trace_logging = TraceLoggingConfig(
            enabled=trace_raw.get("enabled", False),
            trace_dir=trace_raw.get("trace_dir", ""),
            include_user_data=trace_raw.get("include_user_data", False),
            min_turns=trace_raw.get("min_turns", 1),
        )

    # Auto-update
    auto_update_raw = raw.get("auto_update", {})
    if isinstance(auto_update_raw, dict):
        config.auto_update = AutoUpdateConfig(
            enabled=auto_update_raw.get("enabled", True),
            check_interval_secs=auto_update_raw.get("check_interval_secs", 86400),
            source=auto_update_raw.get("source", "pypi"),
            pypi_index_url=auto_update_raw.get("pypi_index_url", ""),
            package_name=auto_update_raw.get("package_name", "autonet-computer"),
        )
    _apply_auto_update_env(config)

    # Worker isolation (agent per-PID process isolation, default ON).
    wi_raw = raw.get("worker_isolation", {})
    config.worker_isolation = _load_worker_isolation(
        wi_raw if isinstance(wi_raw, dict) else {})

    # Secrets / tripwire track (root allowance, fail-closed default "none").
    secrets_raw = raw.get("secrets", {})
    config.secrets = _load_secrets(
        secrets_raw if isinstance(secrets_raw, dict) else {})
    robot_raw = raw.get("robot", {})
    config.robot = _load_robot(robot_raw if isinstance(robot_raw, dict) else {})
    poll_raw = raw.get("robot_poll", {})
    config.robot_poll = _load_robot_poll(poll_raw if isinstance(poll_raw, dict) else {})

    # Idle-connector reaping. Top-level (not per-connector): the cost being
    # managed is the daemon's total resident footprint, not any one server's.
    _idle_raw = raw.get("connector_idle_timeout_s")
    if _idle_raw is not None:
        try:
            config.connector_idle_timeout_s = float(_idle_raw)
        except (TypeError, ValueError):
            pass  # keep the default; a malformed value must not break boot

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
        known_keys = {"api_key", "default_model", "base_url", "models",
                      "dollar_limit", "dollar_period"}
        models_raw = resolved.get("models", [])
        models = models_raw if isinstance(models_raw, list) else []
        try:
            dollar_limit = float(resolved.get("dollar_limit", 0.0) or 0.0)
        except (TypeError, ValueError):
            dollar_limit = 0.0
        dollar_period = str(resolved.get("dollar_period", "none") or "none").lower()
        if dollar_period not in ("none", "hourly", "daily", "weekly", "monthly"):
            dollar_period = "none"
        config.providers[name] = ProviderConfig(
            name=name,
            api_key=resolved.get("api_key", ""),
            default_model=resolved.get("default_model", ""),
            base_url=resolved.get("base_url", ""),
            models=models,
            dollar_limit=dollar_limit,
            dollar_period=dollar_period,
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


def save_secrets_config_to_yaml(
    *,
    worker_isolation: bool | None = None,
    default_root_allowance: str | None = None,
    config_path: Path | None = None,
) -> None:
    """Persist security settings (worker isolation + root allowance) to config.yaml.

    Reads the existing YAML, updates ``worker_isolation.enabled`` and/or
    ``secrets.default_root_allowance`` (only the keys explicitly passed), writes
    back. Creates the file / sections if they don't exist. Mirrors the
    read-update-dump pattern of ``save_provider_to_config``.
    """
    config_path = config_path or (_DEFAULT_DIR / "config.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    raw: dict[str, Any] = {}
    if config_path.exists():
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception:
            log.warning("Failed to read config for secrets-config save: %s", config_path)
            raw = {}

    if worker_isolation is not None:
        wi = raw.get("worker_isolation")
        if not isinstance(wi, dict):
            wi = {}
            raw["worker_isolation"] = wi
        wi["enabled"] = bool(worker_isolation)

    if default_root_allowance is not None:
        sec = raw.get("secrets")
        if not isinstance(sec, dict):
            sec = {}
            raw["secrets"] = sec
        sec["default_root_allowance"] = default_root_allowance

    config_path.write_text(
        yaml.dump(raw, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    log.info("Security settings saved to %s", config_path)


def save_sponsor_address_to_config(
    sponsor_address: str,
    config_path: Path | None = None,
) -> None:
    """Persist this daemon's sponsor (dependent-side) to config.yaml.

    Sponsored inference is daemon-level: the dependent identity is the owner
    wallet and a daemon has at most one sponsor (ratified 2026-07-25,
    docs/sponsored_inference.md). An empty string clears the binding, which
    returns the rpb provider to open discovery.
    """
    config_path = config_path or (_DEFAULT_DIR / "config.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    raw: dict[str, Any] = {}
    if config_path.exists():
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception:
            log.warning("Failed to read config for sponsor save: %s", config_path)
            raw = {}

    autonet = raw.get("autonet")
    if not isinstance(autonet, dict):
        autonet = {}
        raw["autonet"] = autonet
    autonet["sponsor_address"] = (sponsor_address or "").strip()

    config_path.write_text(
        yaml.dump(raw, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    log.info("Sponsor address saved to %s", config_path)


def save_sponsor_mode_to_config(
    sponsor_inference: bool,
    sponsor_provider: str = "",
    sponsor_model: str = "",
    config_path: Path | None = None,
) -> None:
    """Persist this daemon's sponsor-SIDE mode to config.yaml.

    The mirror of save_sponsor_address_to_config: that one records the
    sponsor this daemon consumes from, this one records that this daemon
    serves inference to bound dependents. Without it, enabling sponsor mode
    from the UI lasted only until the next restart while sponsor_bindings.json
    survived, so the panel kept listing dependents the daemon no longer served.

    Blank provider/model are left untouched so an enable call that omits them
    does not wipe a hand-edited value.
    """
    config_path = config_path or (_DEFAULT_DIR / "config.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    raw: dict[str, Any] = {}
    if config_path.exists():
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception:
            log.warning("Failed to read config for sponsor mode save: %s", config_path)
            raw = {}

    autonet = raw.get("autonet")
    if not isinstance(autonet, dict):
        autonet = {}
        raw["autonet"] = autonet
    autonet["sponsor_inference"] = bool(sponsor_inference)
    if sponsor_provider:
        autonet["sponsor_provider"] = sponsor_provider
    if sponsor_model:
        autonet["sponsor_model"] = sponsor_model

    config_path.write_text(
        yaml.dump(raw, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    log.info("Sponsor mode saved to %s (enabled=%s)", config_path, bool(sponsor_inference))


def save_owner_wallet_to_config(
    owner_wallet: str,
    config_path: Path | None = None,
) -> None:
    """Persist the daemon's OWNER wallet to config.yaml.

    This is the identity that owns the fleet's earnings (tool_store's author
    fallback, service_store's provider identity) and the address a remote
    connection must sign with to be rooted at the full fleet. An empty string
    clears it, which disables the remote owner handshake.
    """
    config_path = config_path or (_DEFAULT_DIR / "config.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    raw: dict[str, Any] = {}
    if config_path.exists():
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception:
            log.warning("Failed to read config for owner wallet save: %s", config_path)
            raw = {}

    autonet = raw.get("autonet")
    if not isinstance(autonet, dict):
        autonet = {}
        raw["autonet"] = autonet
    autonet["owner_wallet"] = (owner_wallet or "").strip()

    config_path.write_text(
        yaml.dump(raw, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    log.info("Owner wallet saved to %s", config_path)


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

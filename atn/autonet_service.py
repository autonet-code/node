"""Autonet network service integration for ATN.

Bridges the ATN Runtime with the Autonet decentralized training service.
The service is optional — everything works without it.  When enabled
(via config or user toggle), it starts the training loop in the background.

Lifecycle:
  - ATN Runtime starts → AutonetBridge.start() (if enabled in config)
  - User toggles training on → AutonetBridge.start()
  - User toggles training off → AutonetBridge.stop()
  - ATN Runtime stops → AutonetBridge.stop()

The bridge exposes status and control methods consumed by WS handlers.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING

from .events import EventType

if TYPE_CHECKING:
    from .config import AutonetConfig
    from .events import EventBus

log = logging.getLogger(__name__)


class AutonetStatus(str, Enum):
    """Status of the autonet network service."""
    DISABLED = "disabled"       # Not configured / user hasn't opted in
    STOPPED = "stopped"         # Configured but not running
    STARTING = "starting"       # Starting up
    RUNNING = "running"         # Training loop active
    PAUSED = "paused"           # Paused (resource limits, user pause)
    STOPPING = "stopping"       # Shutting down
    ERROR = "error"             # Failed to start or crashed


@dataclass
class AutonetState:
    """Observable state of the autonet service."""
    status: AutonetStatus = AutonetStatus.DISABLED
    wallet_connected: bool = False
    wallet_address: str = ""
    chain_id: int = 0
    rpc_url: str = ""
    jurisdiction_name: str = ""
    # Training metrics
    cycles_completed: int = 0
    uptime_seconds: float = 0.0
    errors: int = 0
    last_error: str = ""
    # Resource usage (from ResourceMonitor)
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    gpu_available: bool = False
    gpu_memory_mb: float = 0.0
    # Contract addresses (discovered from chain)
    dao_address: str = ""
    rpb_contract_address: str = ""
    registry_address: str = ""
    token_address: str = ""
    economy_address: str = ""
    timelock_address: str = ""
    # Constitution CID loaded from registry
    constitution_cid: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "wallet_connected": self.wallet_connected,
            "wallet_address": self.wallet_address,
            "chain_id": self.chain_id,
            "rpc_url": self.rpc_url,
            "jurisdiction_name": self.jurisdiction_name,
            "cycles_completed": self.cycles_completed,
            "uptime_seconds": self.uptime_seconds,
            "errors": self.errors,
            "last_error": self.last_error,
            "cpu_percent": self.cpu_percent,
            "memory_mb": self.memory_mb,
            "gpu_available": self.gpu_available,
            "gpu_memory_mb": self.gpu_memory_mb,
            "dao_address": self.dao_address,
            "rpb_contract_address": self.rpb_contract_address,
            "registry_address": self.registry_address,
            "token_address": self.token_address,
            "economy_address": self.economy_address,
            "timelock_address": self.timelock_address,
            "constitution_cid": self.constitution_cid,
        }


class AutonetBridge:
    """Bridge between ATN Runtime and the Autonet training service.

    Manages the lifecycle of the AutonetService and exposes control/status
    methods for the WS API.  Runs the training loop in a background asyncio
    task so it doesn't block the ATN event loop.
    """

    def __init__(self, config: AutonetConfig, event_bus: EventBus | None = None,
                 data_dir: str = "") -> None:
        self.config = config
        self.state = AutonetState()
        self._events = event_bus
        self._service = None      # Will hold nodes.service.AutonetService
        self._task: asyncio.Task | None = None
        self._autonet_config = None  # Will hold nodes.common.config.AutonetConfig
        # Standards publication state (Story 3.2)
        self._published_standards_hash: str = ""
        self._published_tx_hash: str = ""
        self._user_contract_address: str = ""
        # Training data feed (Story 3.2) — ATN data dir for JSONL files
        self._data_dir = data_dir

        # Apply config values to state
        if config.wallet_address:
            self.state.wallet_connected = True
            self.state.wallet_address = config.wallet_address
        if config.rpc_url:
            self.state.rpc_url = config.rpc_url
        if config.chain_id:
            self.state.chain_id = config.chain_id

        # Discover jurisdiction contracts from DAO Governor address
        if config.dao_address and config.rpc_url:
            self._discover_jurisdiction()

        # Subscribe to execution events for training data feed
        if self._events:
            self._events.subscribe(
                EventType.EXECUTION_COMPLETED,
                self._on_execution_completed,
            )

        # P2P agent advertisement
        self._p2p_host = None      # AutonetHost (lazy, trio-based)
        self._p2p_thread: threading.Thread | None = None
        self._p2p_stop = threading.Event()
        self._agent_registry = None  # Set by Runtime after init
        # Subscribe to agent lifecycle events for p2p advertisement refresh
        if self._events:
            for evt in (EventType.AGENT_REGISTERED, EventType.AGENT_UNREGISTERED,
                        EventType.AGENT_ACTIVATED, EventType.AGENT_DEACTIVATED):
                self._events.subscribe(evt, self._on_agent_changed)

        # Constitution CID loaded lazily from on-chain Registry
        self._constitution_loaded = False
        # Raw constitution text (loaded once, cached for prompt injection)
        self._constitution_text: str = ""

    def _discover_jurisdiction(self) -> None:
        """Discover all contract addresses from the DAO Governor at startup."""
        try:
            from .on_chain import discover_jurisdiction
            discovered = discover_jurisdiction(
                self.config.rpc_url, self.config.dao_address,
            )
            # Populate config so OnChainService can use them
            self.config.rpb_contract_address = discovered.get("rpb", "")
            self.config.registry_address = discovered.get("registry", "")
            self.config.token_address = discovered.get("token", "")
            self.config.economy_address = discovered.get("economy", "")
            self.config.timelock_address = discovered.get("timelock", "")
            # Populate state for WS API
            self.state.dao_address = self.config.dao_address
            self.state.rpb_contract_address = self.config.rpb_contract_address
            self.state.registry_address = self.config.registry_address
            self.state.token_address = self.config.token_address
            self.state.economy_address = self.config.economy_address
            self.state.timelock_address = self.config.timelock_address
            self.state.jurisdiction_name = discovered.get("jurisdiction_name", "")
            # Load constitution from registry if available
            cid = discovered.get("registry.rpb.prompt.current", "")
            if cid:
                self.state.constitution_cid = cid
                self._constitution_loaded = True
            log.info("Jurisdiction '%s' contracts discovered from %s",
                     self.state.jurisdiction_name, self.config.dao_address)
        except Exception as e:
            log.warning("Failed to discover jurisdiction from %s: %s",
                        self.config.dao_address, e)

    # ------------------------------------------------------------------
    # P2P agent advertisement
    # ------------------------------------------------------------------

    def set_agent_registry(self, registry: Any) -> None:
        """Called by Runtime after init to provide agent registry reference."""
        self._agent_registry = registry

    async def _on_agent_changed(self, event: Any) -> None:
        """Refresh p2p capability when agents are registered/unregistered."""
        self._refresh_p2p_agents()

    def _refresh_p2p_agents(self) -> None:
        """Rebuild agent advertisements from registry and push to p2p host."""
        if not self._p2p_host or not self._agent_registry:
            return
        try:
            ads = self._agent_registry.build_agent_advertisements()
            # Also include the connected wallet as a root agent if present
            if self.state.wallet_address and not any(
                a["address"].lower() == self.state.wallet_address.lower() for a in ads
            ):
                ads.insert(0, {
                    "address": self.state.wallet_address,
                    "name": "root",
                    "description": "",
                    "agent_type": "orchestrator",
                    "model": "",
                    "is_root": True,
                    "parent_address": "",
                    "registered_on_chain": False,
                })
            self._p2p_host.update_capability(agents=ads)
            log.debug("P2P capability updated with %d agent(s)", len(ads))
        except Exception:
            log.debug("Failed to refresh p2p agents", exc_info=True)

    def start_p2p(self) -> None:
        """Start the p2p host in a background thread for agent advertisement."""
        if self._p2p_thread and self._p2p_thread.is_alive():
            return
        try:
            from nodes.common.p2p import AutonetHost, NodeCapability
            from nodes.common.config import load_config as load_autonet_config
        except Exception:
            log.debug("P2P not available (nodes package not installed or import error)")
            return

        config_path = self.config.config_path or None
        try:
            cfg = load_autonet_config(config_path)
        except Exception:
            cfg = None

        listen_port = cfg.p2p.listen_port if cfg else 0
        listen_host = cfg.p2p.listen_host if cfg else "0.0.0.0"
        bootstrap = cfg.p2p.bootstrap_peers if cfg else []
        advertise_interval = cfg.p2p.capability_advertise_interval if cfg else 60

        node_id = f"atn-{self.state.wallet_address[:8]}" if self.state.wallet_address else "atn-daemon"
        cap = NodeCapability(peer_id="", node_id=node_id)

        host = AutonetHost(
            node_id=node_id,
            listen_port=listen_port,
            listen_host=listen_host,
            bootstrap_peers=bootstrap,
            capability=cap,
        )
        self._p2p_host = host
        self._p2p_stop.clear()

        def _run():
            import trio
            async def _main():
                async with host.run():
                    self._refresh_p2p_agents()
                    await host.advertise_capability()
                    log.info("P2P host running, advertising %d agent(s)",
                             len(host._capability.agents))
                    while not self._p2p_stop.is_set():
                        await trio.sleep(advertise_interval)
                        self._refresh_p2p_agents()
                        await host.advertise_capability()
            try:
                trio.run(_main)
            except Exception:
                log.debug("P2P host stopped", exc_info=True)

        self._p2p_thread = threading.Thread(target=_run, name="p2p-host", daemon=True)
        self._p2p_thread.start()
        log.info("P2P agent advertisement started")

    def stop_p2p(self) -> None:
        """Stop the p2p background thread."""
        self._p2p_stop.set()
        if self._p2p_thread:
            self._p2p_thread.join(timeout=5)
            self._p2p_thread = None
        self._p2p_host = None

    async def _emit(self, event_type_name: str, data: dict[str, Any] | None = None) -> None:
        """Emit an event if the event bus is available."""
        if not self._events:
            return
        from .events import Event, EventType
        et = getattr(EventType, event_type_name, None)
        if et:
            await self._events.emit(Event(
                type=et,
                source="autonet",
                data=data or self.state.to_dict(),
            ))

    async def load_constitution(self) -> str | None:
        """Load constitution CID and text from on-chain Registry (once).

        Called lazily on first state request or at startup.
        Returns the CID string or None.
        """
        if self._constitution_loaded:
            return self.state.constitution_cid or None
        self._constitution_loaded = True
        if not self.config.rpb_contract_address or not self.config.registry_address:
            # Even without chain, try loading the local fallback text
            self._load_constitution_text(None)
            return None
        try:
            from .on_chain import OnChainService
            svc = OnChainService(self.config)
            cid = await svc.get_constitution_cid()
            if cid:
                self.state.constitution_cid = cid
                log.info("Constitution CID loaded from chain: %s", cid[:40])
            self._load_constitution_text(cid)
            return cid
        except Exception as e:
            log.debug("Failed to load constitution from chain: %s", e)
            self._load_constitution_text(None)
            return None

    def _load_constitution_text(self, cid: str | None) -> None:
        """Load the raw constitution text from blob store or local fallback."""
        try:
            from nodes.common.rpb_prompt import V1_PROMPT_FILE
            # For now, load from local file (blob store integration later)
            if V1_PROMPT_FILE.exists():
                self._constitution_text = V1_PROMPT_FILE.read_text(encoding="utf-8").strip()
                log.info("Constitution text loaded (%d chars)", len(self._constitution_text))
            else:
                log.warning("No constitution text available at %s", V1_PROMPT_FILE)
        except Exception as e:
            log.debug("Failed to load constitution text: %s", e)

    @property
    def constitution_text(self) -> str:
        """The raw constitution text, empty if not loaded."""
        return self._constitution_text

    async def _on_execution_completed(self, event) -> None:
        """Handle EXECUTION_COMPLETED events from the EventBus.

        Story 3.2: When an agent execution completes, notify the training
        service so it knows new training data is available on disk.
        """
        data = event.data or {}
        agent_id = data.get("agent_id", "")
        execution_id = data.get("execution_id", "")
        status = data.get("status", "")

        if self._service:
            self._service.notify_execution(agent_id, execution_id, status)

    async def start(self) -> dict[str, Any]:
        """Start the autonet training service.

        Returns status dict for WS response.
        """
        if self.state.status == AutonetStatus.RUNNING:
            return {"status": "already_running"}

        self.state.status = AutonetStatus.STARTING
        log.info("Starting autonet service...")

        # Pre-flight cache refresh
        try:
            from ._cache import validate
            if self.config.rpc_url and self.config.registry_address:
                validate(
                    self.config.rpc_url, self.config.registry_address,
                    __import__("atn").__version__,
                )
        except Exception:
            pass

        try:
            # Lazy import — nodes package is optional
            from nodes.common.config import load_config as load_autonet_config
            from nodes.service import AutonetService

            # Load autonet config (discovers autonet.yaml)
            config_path = self.config.config_path or None
            self._autonet_config = load_autonet_config(config_path)

            # Apply any ATN-level overrides
            if self.config.rpc_url:
                self._autonet_config.blockchain.rpc_url = self.config.rpc_url
            if self.config.chain_id:
                self._autonet_config.blockchain.chain_id = self.config.chain_id
            if self.config.private_key:
                self._autonet_config.blockchain.private_key = self.config.private_key

            # Create and start the service in a background thread
            # (AutonetService.start() is blocking)
            self._service = AutonetService(self._autonet_config, data_dir=self._data_dir)
            self._task = asyncio.create_task(self._run_service())

            self.state.status = AutonetStatus.RUNNING
            log.info("Autonet service started")
            # Start p2p agent advertisement alongside the training service
            self.start_p2p()
            await self._emit("AUTONET_STARTED")
            return {"status": "started"}

        except ImportError as e:
            self.state.status = AutonetStatus.ERROR
            self.state.last_error = f"Autonet nodes package not available: {e}"
            log.warning("Failed to start autonet: %s", e)
            return {"status": "error", "error": self.state.last_error}
        except Exception as e:
            self.state.status = AutonetStatus.ERROR
            self.state.last_error = str(e)
            log.exception("Failed to start autonet service")
            return {"status": "error", "error": str(e)}

    async def _run_service(self) -> None:
        """Run the autonet service in a background thread."""
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._service.start)
        except Exception as e:
            self.state.status = AutonetStatus.ERROR
            self.state.last_error = str(e)
            self.state.errors += 1
            log.exception("Autonet service crashed")

    async def stop(self) -> dict[str, Any]:
        """Stop the autonet training service."""
        if self.state.status not in (AutonetStatus.RUNNING, AutonetStatus.PAUSED):
            return {"status": "not_running"}

        self.state.status = AutonetStatus.STOPPING
        log.info("Stopping autonet service...")

        try:
            if self._service:
                self._service.stop()
            if self._task and not self._task.done():
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass

            self.stop_p2p()
            self.state.status = AutonetStatus.STOPPED
            log.info("Autonet service stopped")
            await self._emit("AUTONET_STOPPED")
            return {"status": "stopped"}
        except Exception as e:
            self.state.status = AutonetStatus.ERROR
            self.state.last_error = str(e)
            return {"status": "error", "error": str(e)}

    def get_status(self) -> dict[str, Any]:
        """Get current autonet status for WS response."""
        # Update resource metrics if service is running
        if self._service and self.state.status in (
            AutonetStatus.RUNNING, AutonetStatus.PAUSED
        ):
            try:
                svc_status = self._service.status()
                self.state.cycles_completed = svc_status.cycles_completed
                self.state.uptime_seconds = svc_status.uptime_seconds
                self.state.errors = svc_status.errors
                # Sync state (service may have paused itself due to resources)
                if svc_status.state.value == "paused":
                    self.state.status = AutonetStatus.PAUSED
                elif svc_status.state.value == "running":
                    self.state.status = AutonetStatus.RUNNING
                # Pull resource snapshot if available
                if self._service._resource_monitor:
                    snap = self._service._resource_monitor.last_snapshot
                    if snap:
                        self.state.cpu_percent = snap.cpu_percent
                        self.state.memory_mb = snap.memory_mb
                        self.state.gpu_memory_mb = snap.gpu_memory_mb
                        try:
                            import torch
                            self.state.gpu_available = torch.cuda.is_available()
                        except ImportError:
                            self.state.gpu_available = False
            except Exception:
                pass  # Service might be between cycles

        return self.state.to_dict()

    def connect_wallet(self, address: str) -> dict[str, Any]:
        """Register a wallet connection (from frontend wallet provider)."""
        self.state.wallet_connected = True
        self.state.wallet_address = address
        log.info("Wallet connected: %s", address[:10] + "...")
        return {
            "status": "connected",
            "address": address,
        }

    def disconnect_wallet(self) -> dict[str, Any]:
        """Disconnect the wallet."""
        self.state.wallet_connected = False
        self.state.wallet_address = ""
        log.info("Wallet disconnected")
        return {"status": "disconnected"}

    def set_chain(self, rpc_url: str, chain_id: int) -> dict[str, Any]:
        """Update the blockchain connection."""
        self.state.rpc_url = rpc_url
        self.state.chain_id = chain_id
        # If service is running, update its config too
        if self._autonet_config:
            self._autonet_config.blockchain.rpc_url = rpc_url
            self._autonet_config.blockchain.chain_id = chain_id
        log.info("Chain updated: %s (chain_id=%d)", rpc_url, chain_id)
        return {"status": "updated", "rpc_url": rpc_url, "chain_id": chain_id}

    # ------------------------------------------------------------------
    # Standards publication (Story 3.2)
    # ------------------------------------------------------------------

    def get_standards(self, user_profile=None) -> dict[str, Any]:
        """Get current standards and on-chain publication status.

        Standards are derived from the user profile (populated during
        onboarding or conversation).  The hash can optionally be
        published on-chain via `publish_standards()`.

        Args:
            user_profile: UserProfile instance (from runtime.user_profile)
        """
        standards: list[dict[str, Any]] = []
        if user_profile:
            profile = user_profile.get_profile()
            standards = profile.standards or []

        # Compute the hash that would go on-chain
        standards_hash = self._compute_standards_hash(standards)

        return {
            "standards": standards,
            "standards_hash": standards_hash,
            "on_chain": {
                "published": self.state.wallet_connected and bool(self._published_standards_hash),
                "tx_hash": self._published_tx_hash,
                "published_hash": self._published_standards_hash,
                "matches_current": self._published_standards_hash == standards_hash,
                "user_contract": self._user_contract_address,
            },
        }

    async def publish_standards(self, user_profile=None, private_key: str = "") -> dict[str, Any]:
        """Publish standards hash on-chain by creating a user contract.

        Requires:
          - Wallet connected (address set)
          - RPC URL configured
          - Private key (passed from client-side wallet, NOT stored)
          - Standards exist in user profile

        Args:
            user_profile: UserProfile instance
            private_key: Hex private key for signing the transaction
        """
        if not self.state.wallet_connected:
            return {"status": "error", "error": "Wallet not connected"}
        if not self.state.rpc_url:
            return {"status": "error", "error": "No RPC URL configured"}
        if not private_key:
            return {"status": "error", "error": "Private key required for signing"}

        # Get standards
        standards: list[dict[str, Any]] = []
        if user_profile:
            profile = user_profile.get_profile()
            standards = profile.standards or []

        if not standards:
            return {"status": "error", "error": "No standards defined. Complete onboarding first."}

        standards_hash = self._compute_standards_hash(standards)
        standards_hash_bytes = bytes.fromhex(standards_hash)

        try:
            from web3 import Web3
            from eth_account import Account

            w3 = Web3(Web3.HTTPProvider(self.state.rpc_url))
            if not w3.is_connected():
                return {"status": "error", "error": f"Cannot connect to {self.state.rpc_url}"}

            account = Account.from_key(private_key)
            if account.address.lower() != self.state.wallet_address.lower():
                return {"status": "error", "error": "Private key does not match connected wallet"}

            # Load the Autonet contract ABI
            autonet_address = self._get_autonet_contract_address()
            if not autonet_address:
                return {"status": "error", "error": "Autonet contract address not configured"}

            abi = self._load_contract_abi("Autonet")
            if not abi:
                return {"status": "error", "error": "Autonet contract ABI not found"}

            contract = w3.eth.contract(
                address=Web3.to_checksum_address(autonet_address),
                abi=abi,
            )

            # Build and send createUserContract transaction
            nonce = w3.eth.get_transaction_count(account.address)
            tx = contract.functions.createUserContract(
                standards_hash_bytes
            ).build_transaction({
                "from": account.address,
                "nonce": nonce,
                "gas": 2_000_000,
                "gasPrice": w3.eth.gas_price,
                "chainId": self.state.chain_id or w3.eth.chain_id,
            })

            signed = account.sign_transaction(tx)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

            if receipt.status == 1:
                # Parse UserContractCreated event to get user contract address
                user_contract_addr = ""
                try:
                    logs = contract.events.UserContractCreated().process_receipt(receipt)
                    if logs:
                        user_contract_addr = logs[0]["args"]["userContract"]
                except Exception:
                    pass

                self._published_standards_hash = standards_hash
                self._published_tx_hash = tx_hash.hex()
                self._user_contract_address = user_contract_addr

                log.info("Standards published on-chain: tx=%s, user_contract=%s",
                         self._published_tx_hash, user_contract_addr)

                await self._emit("AUTONET_STATUS", {
                    "action": "standards_published",
                    "tx_hash": self._published_tx_hash,
                    "standards_hash": standards_hash,
                    "user_contract": user_contract_addr,
                })

                return {
                    "status": "published",
                    "tx_hash": self._published_tx_hash,
                    "standards_hash": standards_hash,
                    "user_contract": user_contract_addr,
                }
            else:
                return {"status": "error", "error": "Transaction reverted"}

        except ImportError:
            return {"status": "error", "error": "web3 package not installed. Install with: pip install web3"}
        except Exception as e:
            log.exception("Failed to publish standards")
            return {"status": "error", "error": str(e)}

    @staticmethod
    def _compute_standards_hash(standards: list[dict[str, Any]]) -> str:
        """Compute a deterministic bytes32 hash of the standards list.

        The hash is a keccak256 of the JSON-serialized standards,
        matching the Solidity bytes32 format.
        """
        if not standards:
            return "0" * 64
        # Canonical JSON: sorted keys, no whitespace
        canonical = json.dumps(standards, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def _get_autonet_contract_address(self) -> str | None:
        """Get the deployed Autonet contract address from config."""
        return self._get_contract_address("autonet")

    def _load_deployment_addresses(self) -> dict[str, str]:
        """Load contract addresses from deployment-addresses.json."""
        from pathlib import Path
        candidates = [
            Path("deployment-addresses.json"),
            Path(__file__).parent.parent / "deployment-addresses.json",
        ]
        for path in candidates:
            if path.exists():
                try:
                    with open(path) as f:
                        return json.load(f)
                except Exception:
                    continue
        return {}

    def _load_contract_abi(self, contract_name: str) -> list | None:
        """Load a contract ABI from the artifacts directory."""
        from pathlib import Path

        base = Path(__file__).parent.parent / "artifacts" / "contracts"
        base2 = Path("artifacts") / "contracts"

        # Search all subdirectories under artifacts/contracts/
        subdirs = ["economic", "governance", "core", "tokens", "bridge",
                    "rollup", "utils", "interfaces", "mocks"]
        candidates = []
        for sd in subdirs:
            candidates.append(base / sd / f"{contract_name}.sol" / f"{contract_name}.json")
            candidates.append(base2 / sd / f"{contract_name}.sol" / f"{contract_name}.json")
        # Also check flat locations
        candidates.append(base.parent / f"{contract_name}.json")
        candidates.append(Path("artifacts") / f"{contract_name}.json")

        for path in candidates:
            if path.exists():
                try:
                    with open(path) as f:
                        artifact = json.load(f)
                    return artifact.get("abi", [])
                except Exception:
                    continue
        return None

    # ------------------------------------------------------------------
    # Earnings dashboard (Story 3.5)
    # ------------------------------------------------------------------

    async def get_earnings(self) -> dict[str, Any]:
        """Get the user's ATN earnings summary.

        Returns local training metrics and, if wallet+contract are
        available, on-chain epoch reward data.
        """
        # Local training stats (always available)
        local = {
            "cycles_completed": self.state.cycles_completed,
            "uptime_seconds": self.state.uptime_seconds,
            "status": self.state.status.value,
        }

        # On-chain earnings (requires wallet + contract)
        on_chain: dict[str, Any] = {
            "available": False,
            "balance": "0",
            "total_burned": "0",
            "current_epoch": 0,
            "claimable_rewards": [],
        }

        if (self.state.wallet_connected
                and self.state.rpc_url
                and self._get_autonet_contract_address()):
            try:
                on_chain = await self._fetch_on_chain_earnings()
            except Exception as e:
                log.debug("Failed to fetch on-chain earnings: %s", e)

        return {
            "local": local,
            "on_chain": on_chain,
        }

    async def _fetch_on_chain_earnings(self) -> dict[str, Any]:
        """Fetch earnings data from the Autonet contract."""
        from web3 import Web3

        w3 = Web3(Web3.HTTPProvider(self.state.rpc_url))
        if not w3.is_connected():
            return {"available": False, "error": "RPC not reachable"}

        autonet_address = self._get_autonet_contract_address()
        abi = self._load_contract_abi("Autonet")
        if not autonet_address or not abi:
            return {"available": False, "error": "Contract not configured"}

        contract = w3.eth.contract(
            address=Web3.to_checksum_address(autonet_address),
            abi=abi,
        )

        wallet = Web3.to_checksum_address(self.state.wallet_address)

        # Fetch key data points
        balance = contract.functions.balanceOf(wallet).call()
        current_epoch = contract.functions.currentEpoch().call()
        total_burned = contract.functions.totalInferenceBurned().call()
        user_burned = contract.functions.userInferenceBurned(wallet).call()

        # Check user contract
        user_contract = contract.functions.getUserContract(wallet).call()
        has_user_contract = user_contract != "0x" + "0" * 40

        # Scan recent epochs for claimable rewards
        claimable: list[dict[str, Any]] = []
        scan_start = max(1, current_epoch - 10)  # Last 10 epochs
        for epoch_id in range(scan_start, current_epoch + 1):
            try:
                epoch_stats = contract.functions.getEpochStats(epoch_id).call()
                total_usage, budget, finalized, _ = epoch_stats
                if not finalized or budget == 0:
                    continue

                # Check all services for this user's attested usage
                service_ids = contract.functions.getAllServiceIds().call()
                for sid in service_ids:
                    attester_usage = contract.functions.getAttesterUsage(
                        epoch_id, sid, wallet
                    ).call()
                    if attester_usage == 0:
                        continue

                    # Check if already claimed
                    already_claimed = contract.functions.hasClaimedReward(
                        sid, epoch_id, wallet
                    ).call()
                    if already_claimed:
                        continue

                    # Estimate reward
                    epoch_reward = contract.functions.getServiceEpochReward(
                        sid, epoch_id
                    ).call()
                    reward_total, claimed_amount, _ = epoch_reward
                    if reward_total == 0:
                        continue

                    service_epoch_usage = contract.functions.epochUsage(
                        epoch_id, sid
                    ).call() if hasattr(contract.functions, 'epochUsage') else total_usage

                    estimated = (attester_usage * reward_total) // service_epoch_usage if service_epoch_usage > 0 else 0

                    claimable.append({
                        "epoch": epoch_id,
                        "service_id": sid.hex() if isinstance(sid, bytes) else str(sid),
                        "attested_units": attester_usage,
                        "estimated_reward_wei": str(estimated),
                        "estimated_reward": str(estimated / 10**18) if estimated > 0 else "0",
                    })
            except Exception:
                continue  # Skip epochs with errors

        return {
            "available": True,
            "balance": str(balance),
            "balance_formatted": f"{balance / 10**18:.4f}",
            "current_epoch": current_epoch,
            "total_burned": str(total_burned),
            "user_burned": str(user_burned),
            "has_user_contract": has_user_contract,
            "user_contract": user_contract if has_user_contract else "",
            "claimable_rewards": claimable,
        }

    async def claim_reward(self, epoch_id: int, service_id: str,
                           private_key: str) -> dict[str, Any]:
        """Claim participant reward for an epoch."""
        if not self.state.wallet_connected:
            return {"status": "error", "error": "Wallet not connected"}
        if not private_key:
            return {"status": "error", "error": "Private key required"}

        try:
            from web3 import Web3
            from eth_account import Account

            w3 = Web3(Web3.HTTPProvider(self.state.rpc_url))
            account = Account.from_key(private_key)
            autonet_address = self._get_autonet_contract_address()
            abi = self._load_contract_abi("Autonet")

            if not autonet_address or not abi:
                return {"status": "error", "error": "Contract not configured"}

            contract = w3.eth.contract(
                address=Web3.to_checksum_address(autonet_address),
                abi=abi,
            )

            sid_bytes = bytes.fromhex(service_id) if isinstance(service_id, str) else service_id

            nonce = w3.eth.get_transaction_count(account.address)
            tx = contract.functions.claimParticipantReward(
                sid_bytes, epoch_id
            ).build_transaction({
                "from": account.address,
                "nonce": nonce,
                "gas": 500_000,
                "gasPrice": w3.eth.gas_price,
                "chainId": self.state.chain_id or w3.eth.chain_id,
            })

            signed = account.sign_transaction(tx)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

            if receipt.status == 1:
                log.info("Reward claimed: epoch=%d, tx=%s", epoch_id, tx_hash.hex())
                return {
                    "status": "claimed",
                    "tx_hash": tx_hash.hex(),
                    "epoch": epoch_id,
                }
            else:
                return {"status": "error", "error": "Transaction reverted"}

        except Exception as e:
            log.exception("Failed to claim reward")
            return {"status": "error", "error": str(e)}

    # ------------------------------------------------------------------
    # Proposals & governance (Stories 3.6 / 3.7)
    # ------------------------------------------------------------------

    async def get_proposals(self) -> dict[str, Any]:
        """List evolution proposals from the EvolutionProposal contract."""
        if not self.state.wallet_connected or not self.state.rpc_url:
            return {"available": False, "proposals": []}

        try:
            from web3 import Web3

            w3 = Web3(Web3.HTTPProvider(self.state.rpc_url))
            if not w3.is_connected():
                return {"available": False, "proposals": []}

            ep_address = self._get_contract_address("evolution_proposal")
            abi = self._load_contract_abi("EvolutionProposal")
            if not ep_address or not abi:
                return {"available": False, "proposals": []}

            contract = w3.eth.contract(
                address=Web3.to_checksum_address(ep_address), abi=abi,
            )

            count = contract.functions.getProposalCount().call()
            proposals: list[dict[str, Any]] = []

            for i in range(min(count, 50)):  # Cap at 50 most recent
                pid = count - i  # Newest first
                try:
                    result = contract.functions.getProposal(pid).call()
                    proposer, content_cid, stake, status, created_at, trial_budget, eval_count, eval_score, dao_id = result
                    status_names = ["proposed", "evaluating", "trial", "adopted", "rejected"]
                    proposals.append({
                        "id": pid,
                        "proposer": proposer,
                        "content_cid": content_cid,
                        "stake": str(stake),
                        "status": status_names[status] if status < len(status_names) else "unknown",
                        "created_at": created_at,
                        "trial_budget": str(trial_budget),
                        "evaluation_count": eval_count,
                        "evaluation_score": eval_score,
                        "dao_proposal_id": dao_id,
                    })
                except Exception:
                    continue

            return {"available": True, "proposals": proposals}

        except ImportError:
            return {"available": False, "error": "web3 not installed"}
        except Exception as e:
            log.debug("Failed to fetch proposals: %s", e)
            return {"available": False, "proposals": []}

    async def get_governance(self) -> dict[str, Any]:
        """Get governance overview from AutonetDAO."""
        if not self.state.wallet_connected or not self.state.rpc_url:
            return {"available": False, "proposals": []}

        try:
            from web3 import Web3

            w3 = Web3(Web3.HTTPProvider(self.state.rpc_url))
            if not w3.is_connected():
                return {"available": False, "proposals": []}

            dao_address = self._get_contract_address("dao")
            abi = self._load_contract_abi("AutonetDAO")
            if not dao_address or not abi:
                return {"available": False, "proposals": []}

            contract = w3.eth.contract(
                address=Web3.to_checksum_address(dao_address), abi=abi,
            )

            next_id = contract.functions.nextProposalId().call()
            proposals: list[dict[str, Any]] = []

            for pid in range(max(1, next_id - 20), next_id):
                try:
                    p = contract.functions.proposals(pid).call()
                    state = contract.functions.getProposalState(pid).call()
                    state_names = ["pending", "active", "canceled", "defeated",
                                   "succeeded", "queued", "expired", "executed"]
                    proposals.append({
                        "id": p[0],
                        "proposer": p[1],
                        "description": p[2],
                        "start_block": p[3],
                        "end_block": p[4],
                        "for_votes": str(p[5]),
                        "against_votes": str(p[6]),
                        "executed": p[7],
                        "canceled": p[8],
                        "state": state_names[state] if state < len(state_names) else "unknown",
                    })
                except Exception:
                    continue

            return {"available": True, "proposals": proposals}

        except ImportError:
            return {"available": False, "error": "web3 not installed"}
        except Exception as e:
            log.debug("Failed to fetch governance: %s", e)
            return {"available": False, "proposals": []}

    def _get_contract_address(self, name: str) -> str | None:
        """Get a named contract address from config or deployment-addresses.json.

        Name mapping:
          autonet → ATNToken (main Autonet entry-point uses ATN Token)
          dao → AutonetDAO
          evolution_proposal → (not yet deployed separately)
          staking → ParticipantStaking
          Any key from deployment-addresses.json
        """
        # Mapping from logical names to deployment-addresses.json keys
        name_map = {
            "autonet": "Autonet",
            "dao": "AutonetDAO",
            "rpb": "RPB",
            "rpb_factory": "RPBFactory",
            "staking": "ParticipantStaking",
            "shard_registry": "ModelShardRegistry",
            "task": "TaskContract",
            "results": "ResultsRewards",
            "bridge": "AnchorBridge",
            "disputes": "DisputeManager",
            "forced_errors": "ForcedErrorRegistry",
            "inference_factory": "InferenceProviderFactory",
        }

        # 1. Check autonet YAML config (blockchain.contracts section)
        if self._autonet_config and hasattr(self._autonet_config, "blockchain"):
            bc = self._autonet_config.blockchain
            # Check direct attribute
            attr_name = f"{name}_contract"
            addr = getattr(bc, attr_name, None)
            if addr:
                return addr
            # Check contracts dict if it exists
            contracts = getattr(bc, "contracts", None)
            if contracts:
                json_key = name_map.get(name, name)
                if json_key in contracts:
                    return contracts[json_key]

        # 2. Check deployment-addresses.json
        addresses = self._load_deployment_addresses()
        if addresses:
            json_key = name_map.get(name, name)
            if json_key in addresses:
                return addresses[json_key]

        # 3. Check ATN config
        contract_addr = getattr(self.config, f"{name}_contract", None)
        return contract_addr

    # ------------------------------------------------------------------
    # Alignment dashboard (Story 3.8)
    # ------------------------------------------------------------------

    def get_alignment(self, user_profile=None) -> dict[str, Any]:
        """Get alignment status between user standards and network behavior.

        Shows how well the node's training activity reflects the user's
        stated standards and values.  Without live training data, returns
        the structural alignment (standards exist, training active, etc.).
        """
        standards: list[dict[str, Any]] = []
        if user_profile:
            profile = user_profile.get_profile()
            standards = profile.standards or []

        # Alignment indicators
        has_standards = len(standards) > 0
        is_training = self.state.status.value == "running"
        has_wallet = self.state.wallet_connected
        has_user_contract = bool(self._user_contract_address)
        standards_published = bool(self._published_standards_hash)

        # Compute an alignment score (0-100)
        # Each factor contributes to overall alignment readiness
        score = 0
        factors: list[dict[str, Any]] = []

        if has_standards:
            score += 25
            factors.append({"name": "Standards defined", "met": True,
                           "description": f"{len(standards)} standards in your profile"})
        else:
            factors.append({"name": "Standards defined", "met": False,
                           "description": "Complete onboarding to define your standards"})

        if standards_published:
            score += 25
            factors.append({"name": "Standards published", "met": True,
                           "description": "Your standards hash is recorded on-chain"})
        elif has_wallet:
            factors.append({"name": "Standards published", "met": False,
                           "description": "Publish your standards to establish on-chain identity"})
        else:
            factors.append({"name": "Standards published", "met": False,
                           "description": "Connect wallet first, then publish standards"})

        if is_training:
            score += 25
            factors.append({"name": "Training active", "met": True,
                           "description": f"{self.state.cycles_completed} cycles completed"})
        else:
            factors.append({"name": "Training active", "met": False,
                           "description": "Start training to contribute to the network"})

        if has_user_contract:
            score += 25
            factors.append({"name": "Node registered", "met": True,
                           "description": "Your node has an on-chain identity"})
        elif has_wallet:
            factors.append({"name": "Node registered", "met": False,
                           "description": "Publish standards to register your node"})
        else:
            factors.append({"name": "Node registered", "met": False,
                           "description": "Connect wallet to register on-chain"})

        return {
            "score": score,
            "factors": factors,
            "standards_count": len(standards),
            "training_active": is_training,
            "wallet_connected": has_wallet,
            "node_registered": has_user_contract,
        }

    # ------------------------------------------------------------------
    # Data capture & privacy (Story 3.4)
    # ------------------------------------------------------------------

    def get_capture_config(self) -> dict[str, Any]:
        """Get the current capture and privacy configuration."""
        cfg = self._autonet_config or self._load_autonet_config_readonly()
        return {
            "capture": {
                "enabled_sources": list(cfg.capture.enabled_sources),
                "fps_cap": cfg.capture.fps_cap,
                "resolution": list(cfg.capture.resolution),
                "screen_monitor": cfg.capture.screen_monitor,
                "browser_scrub_pii": cfg.capture.browser_scrub_pii,
                "browser_exclude_patterns": list(cfg.capture.browser_exclude_patterns),
            },
            "privacy": {
                "exclude_apps": list(cfg.privacy.exclude_apps),
                "blur_regions": list(cfg.privacy.blur_regions),
                "scrub_pii": cfg.privacy.scrub_pii,
            },
        }

    def set_capture_config(self, capture: dict[str, Any] | None = None,
                           privacy: dict[str, Any] | None = None) -> dict[str, Any]:
        """Update capture and/or privacy configuration."""
        cfg = self._autonet_config or self._load_autonet_config_readonly()

        if capture:
            if "enabled_sources" in capture:
                cfg.capture.enabled_sources = list(capture["enabled_sources"])
            if "fps_cap" in capture:
                cfg.capture.fps_cap = int(capture["fps_cap"])
            if "resolution" in capture:
                cfg.capture.resolution = list(capture["resolution"])
            if "screen_monitor" in capture:
                cfg.capture.screen_monitor = int(capture["screen_monitor"])
            if "browser_scrub_pii" in capture:
                cfg.capture.browser_scrub_pii = bool(capture["browser_scrub_pii"])
            if "browser_exclude_patterns" in capture:
                cfg.capture.browser_exclude_patterns = list(capture["browser_exclude_patterns"])

        if privacy:
            if "exclude_apps" in privacy:
                cfg.privacy.exclude_apps = list(privacy["exclude_apps"])
            if "blur_regions" in privacy:
                cfg.privacy.blur_regions = list(privacy["blur_regions"])
            if "scrub_pii" in privacy:
                cfg.privacy.scrub_pii = bool(privacy["scrub_pii"])

        self._autonet_config = cfg
        self._save_autonet_config(cfg)
        log.info("Capture/privacy config updated")
        return self.get_capture_config()

    def enumerate_sources(self) -> dict[str, Any]:
        """Enumerate available capture sources (screens, browsers).

        Returns a list of available sources the user can select from.
        """
        sources: list[dict[str, Any]] = []

        # Enumerate screens via mss
        try:
            import mss
            with mss.mss() as sct:
                for i, mon in enumerate(sct.monitors):
                    if i == 0:
                        continue  # monitor 0 is the "all monitors" virtual screen
                    sources.append({
                        "type": "screen",
                        "id": f"screen_{i}",
                        "name": f"Screen {i}",
                        "monitor_index": i,
                        "width": mon["width"],
                        "height": mon["height"],
                        "left": mon["left"],
                        "top": mon["top"],
                    })
        except ImportError:
            log.debug("mss not available — screen enumeration disabled")
        except Exception as e:
            log.debug("Screen enumeration failed: %s", e)

        # Browser source (always available if the relay is configured)
        sources.append({
            "type": "browser",
            "id": "browser",
            "name": "Browser Activity",
            "description": "Captures browsing activity via CDP relay",
        })

        return {"sources": sources}

    def _load_autonet_config_readonly(self):
        """Load autonet config without storing it (for read-only access)."""
        from nodes.common.config import load_config as load_autonet_config
        config_path = self.config.config_path or None
        return load_autonet_config(config_path)

    def _save_autonet_config(self, cfg) -> None:
        """Save autonet config back to YAML file."""
        try:
            import yaml
            from pathlib import Path

            # Find the config file path
            config_path = self.config.config_path
            if not config_path:
                import os
                config_path = os.environ.get("AUTONET_CONFIG")
            if not config_path:
                candidates = [
                    Path("autonet.yaml"),
                    Path.home() / ".autonet" / "config.yaml",
                ]
                for c in candidates:
                    if c.exists():
                        config_path = str(c)
                        break

            if not config_path:
                log.warning("No config file found to save to")
                return

            # Read existing YAML, update capture/privacy sections, write back
            path = Path(config_path)
            with open(path) as f:
                raw = yaml.safe_load(f) or {}

            raw["capture"] = {
                "enabled_sources": cfg.capture.enabled_sources,
                "fps_cap": cfg.capture.fps_cap,
                "resolution": cfg.capture.resolution,
                "screen_monitor": cfg.capture.screen_monitor,
                "browser_scrub_pii": cfg.capture.browser_scrub_pii,
                "browser_exclude_patterns": cfg.capture.browser_exclude_patterns,
            }
            raw["privacy"] = {
                "exclude_apps": cfg.privacy.exclude_apps,
                "blur_regions": cfg.privacy.blur_regions,
                "scrub_pii": cfg.privacy.scrub_pii,
            }

            with open(path, "w") as f:
                yaml.dump(raw, f, default_flow_style=False, sort_keys=False)

            log.info("Saved config to %s", config_path)
        except Exception as e:
            log.warning("Failed to save config: %s", e)

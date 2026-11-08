"""
Autonet Node Service — Managed Daemon Process.

Top-level entry point that runs a solver node as a managed background service.
Handles signal-based lifecycle (SIGTERM/SIGINT) and graceful shutdown. Updates are the atn daemon's (PyPI, atn/cli.py).

Story 6.1: Solver node as daemon background service.
"""

import logging
import signal
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from .common.config import AutonetConfig, load_config

logger = logging.getLogger(__name__)


class ServiceState(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"


@dataclass
class ServiceStatus:
    """Snapshot of the service's current state."""
    state: ServiceState
    uptime_seconds: float = 0.0
    cycles_completed: int = 0
    errors: int = 0
    paused_reason: Optional[str] = None


class AutonetService:
    """
    Managed service wrapper around an Autonet solver node.

    Lifecycle:
        1. start() — initialize subsystems, enter main loop
        2. Main loop: run the consensus cycle
        3. stop() — graceful shutdown of all subsystems

    Signal handling:
        SIGTERM/SIGINT → stop()
    """

    def __init__(
        self,
        config: Optional[AutonetConfig] = None,
        data_dir: Optional[str] = None,
        *,
        configure_logging: bool = True,
    ):
        self.config = config or load_config()
        # False lets an embedding host keep its own logging setup:
        # start() then never touches the root logger.
        self._configure_logging = configure_logging
        self._state = ServiceState.STOPPED
        self._start_time: float = 0.0
        self._cycles: int = 0
        self._errors: int = 0
        self._shutdown_requested = False

        # Subsystems (lazy-init in start())
        self._node = None  # Solver or other node type

        self._data_dir = data_dir  # ATN data dir (~/.atn), passed from AutonetBridge

        # The tool library service (epoch tick, event plumbing, search),
        # one per jurisdiction key.
        self._library = None
        # Event gossip for substrate federation across daemons.
        # Constructed by attach_event_gossip() after the libp2p host
        # is ready (Phase 10.2).
        self._event_gossip = None
        # Federated-close driver: runs canonical_order +
        # federated_epoch_close on every epoch close, and decides
        # whether this daemon is the chosen on-chain submitter for
        # that epoch (Phase 10.3a). Subscribers fed the result land
        # in self._federated_close_subscribers.
        self._federated_close_driver = None
        self._federated_close_subscribers: list = []
        # Optional callback(new_charter_text): the jurisdiction's charter
        # changed (pinned to the previous anchor) and the close refused to
        # run on the old axes. The atn host restarts the daemon onto it.
        self.on_charter_change = None
        # Chain submission driver (Phase 10.3b/c): attached by
        # attach_chain_submission() after the federated-close driver
        # is up. Owns the EpochAnchorer (one) and per-agent
        # AuthoritativeChainSubmitter cache.
        self._chain_submission_driver = None
        # The epoch scheduler ticks the library's epochs.
        self._epoch_scheduler = None
        # External subscribers receive epoch close records. Installed
        # before start() so they're attached when the library service is
        # constructed.
        self._external_epoch_subscribers: list = []

        # Blob store for uploading weight deltas
        self._blob_store = None

    @property
    def state(self) -> ServiceState:
        return self._state

    def status(self) -> ServiceStatus:
        """Get current service status."""
        uptime = time.time() - self._start_time if self._start_time else 0.0
        return ServiceStatus(
            state=self._state,
            uptime_seconds=uptime,
            cycles_completed=self._cycles,
            errors=self._errors,
        )

    def start(self):
        """
        Initialize subsystems and enter the main loop.

        This blocks until stop() is called (via signal or explicit call).
        """
        if self._state != ServiceState.STOPPED:
            logger.warning(f"Cannot start: service is {self._state.value}")
            return

        self._state = ServiceState.STARTING
        self._start_time = time.time()
        self._shutdown_requested = False

        # Configure logging (global root-logger setup; an embedding host
        # opts out with configure_logging=False).
        if self._configure_logging:
            log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
            logging.basicConfig(level=log_level, format="%(asctime)s %(name)s %(levelname)s %(message)s")

        logger.info("AutonetService starting...")

        # Install signal handlers
        self._install_signal_handlers()

        # Initialize subsystems
        try:
            # The tool library service (always available).
            self._init_library_service()
            if self._library is not None:
                self._init_epoch_scheduler()

            # Initialize blob store for weight delta uploads
            self._init_blob_store()

            logger.info(
                f"AutonetService initialized: "
                f"device={self.config.device}, "
                f"arch={self.config.model.architecture}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize subsystems: {e}")
            self._state = ServiceState.STOPPED
            self._errors += 1
            return

        self._state = ServiceState.RUNNING
        logger.info("AutonetService running")

        # Main loop
        self._main_loop()

    def stop(self):
        """Request graceful shutdown."""
        if self._state in (ServiceState.STOPPED, ServiceState.STOPPING):
            return

        logger.info("AutonetService stopping...")
        self._state = ServiceState.STOPPING
        self._shutdown_requested = True

        # Stop the solver node if running
        if self._node and hasattr(self._node, "stop"):
            try:
                self._node.stop()
            except Exception as e:
                logger.error(f"Error stopping node: {e}")

        # Force a final epoch close so any in-flight reward window
        # is finalized rather than dropped silently.
        if self._epoch_scheduler is not None:
            try:
                self._epoch_scheduler.force_tick()
            except Exception as e:
                logger.error(f"Error in final epoch tick: {e}")

        if self._library is not None:
            try:
                # Close any epoch the scheduler didn't reach.
                if self._library.current_epoch_id is not None:
                    self._library.close_epoch()
                self._library.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down the library service: {e}")

        self._state = ServiceState.STOPPED
        uptime = time.time() - self._start_time
        logger.info(
            f"AutonetService stopped after {uptime:.0f}s "
            f"({self._cycles} cycles, {self._errors} errors)"
        )

    def _main_loop(self):
        """Core service loop: run the cycle, sleep, repeat."""
        cycle_delay = self.config.node.cycle_delay
        max_cycles = self.config.node.max_cycles

        while not self._shutdown_requested:
            # Check cycle limit
            if max_cycles and self._cycles >= max_cycles:
                logger.info(f"Reached max_cycles ({max_cycles}), stopping")
                break

            try:
                # No CPU gate: this cycle is consensus work (substrate feed,
                # gossip outbox, the epoch scheduler's close tick), and a
                # busy machine skipping it missed closes the network made.
                # The gate was for JEPA training, which is retired.
                self._run_cycle()
                self._cycles += 1

                # Sleep between cycles
                time.sleep(cycle_delay)

            except Exception as e:
                logger.error(f"Error in service cycle {self._cycles}: {e}", exc_info=True)
                self._errors += 1
                time.sleep(cycle_delay)

        # Clean shutdown
        self.stop()

    def _run_cycle(self):
        """Execute one training/work cycle."""
        # If a solver node is attached, delegate to it
        if self._node and hasattr(self._node, "_run_cycle"):
            self._node._run_cycle()
            return

        # Local events held while no peer was subscribed go out now.
        if self._event_gossip is not None:
            try:
                self._event_gossip.flush_outbox()
            except Exception as e:
                logger.warning("Gossip outbox flush failed: %s", e)

        # Epoch scheduler: tick to close+open reward windows on its
        # configured interval. Phase 4 will surface the close result
        # to the chain.
        if self._epoch_scheduler:
            try:
                self._epoch_scheduler.maybe_tick()
            except Exception as e:
                logger.warning("Epoch scheduler tick failed: %s", e)

        logger.debug(f"Service cycle {self._cycles}")

    def _init_blob_store(self):
        """Initialize the blob store for weight delta uploads."""
        try:
            from .common.blob_store import BlobStore

            blob_dir = self.config.blob_store.data_dir if hasattr(self.config, "blob_store") else ""
            if not blob_dir:
                blob_dir = "~/.autonet/blobs"

            peer_urls = []
            if hasattr(self.config, "blob_store") and hasattr(self.config.blob_store, "peer_urls"):
                peer_urls = self.config.blob_store.peer_urls or []

            self._blob_store = BlobStore(data_dir=blob_dir, peer_urls=peer_urls)
            logger.info("Blob store initialized at %s", blob_dir)

        except Exception as e:
            logger.warning("Failed to initialize blob store: %s", e)
            self._blob_store = None

    def _install_signal_handlers(self):
        """Install SIGTERM and SIGINT handlers for graceful shutdown."""
        def _handler(signum, frame):
            sig_name = signal.Signals(signum).name
            logger.info(f"Received {sig_name}, initiating shutdown...")
            self._shutdown_requested = True

        try:
            signal.signal(signal.SIGTERM, _handler)
            signal.signal(signal.SIGINT, _handler)
        except (OSError, ValueError):
            # signal handlers can only be set in main thread
            logger.debug("Could not install signal handlers (not main thread?)")

    def _init_library_service(self):
        """Initialize the tool library service: one per jurisdiction key,
        living for the whole process."""
        try:
            from .common.library_service import LibraryService
            # The jurisdiction key "<chain_id>:<DAO address>" names the
            # gossip topic and the state dir, so each jurisdiction's data
            # is parked under its own key. "default" only for local,
            # chain-free runs.
            rpb_address = (getattr(self.config, "jurisdiction_key", "")
                           or "default")
            self._library = LibraryService(rpb_address=str(rpb_address))
            # Attach external subscribers registered before start().
            for handler in self._external_epoch_subscribers:
                self._library.subscribe_epoch_closed(handler)
            logger.info(
                "Library service initialized for rpb=%s (%d subscribers)",
                rpb_address, len(self._external_epoch_subscribers),
            )
        except Exception as e:
            logger.warning("Failed to initialize the library service: %s", e)
            self._library = None

    def add_epoch_close_subscriber(self, handler) -> None:
        """Register a handler that will be invoked on every epoch
        close. Safe to call before start(): handlers attach to the
        library service when it's constructed in start()."""
        self._external_epoch_subscribers.append(handler)
        if self._library is not None:
            self._library.subscribe_epoch_closed(handler)

    def _init_epoch_scheduler(self):
        """Initialize the wall-clock epoch scheduler for reward windows.

        Phase 3 of native integration. The scheduler doesn't run its
        own thread by default — instead the daemon's main loop calls
        ``maybe_tick`` each cycle. Phase 4 will hand the close result
        off to a chain-reporting handler; for now we just log it.
        """
        try:
            from .common.epoch_scheduler import (
                EpochScheduler,
                EpochSchedulerConfig,
            )
            interval = float(getattr(self.config, "epoch_interval_seconds", 60.0))

            def _on_close(result: Dict[str, Any]) -> None:
                # Empty local epochs (an idle or follower daemon) stay at
                # debug so they don't fill the console every minute.
                logger.log(
                    logging.INFO if result.get("n_events") else logging.DEBUG,
                    "Epoch closed: id=%s, n_events=%d, total_mint=%.4f, "
                    "agents=%d",
                    result.get("epoch_id"),
                    result.get("n_events", 0),
                    result.get("total_mint", 0.0),
                    len(result.get("agent_mint", {})),
                )

            candle_min = float(getattr(self.config, "epoch_min_seconds", 0.0) or 0.0)
            candle_window = float(
                getattr(self.config, "epoch_candle_window_seconds", 0.0) or 0.0
            )
            # Chain-derived candle seed (federation-safe cutoff). Falls
            # back to the scheduler's local seed when chain config is
            # absent or the RPC is unreachable.
            seed_source = None
            if candle_min > 0:
                try:
                    from .common.candle_seed import build_chain_candle_seed
                    bc = self.config.blockchain
                    chain_seed = build_chain_candle_seed(
                        substrate_address=getattr(bc, "substrate_address", "") or "",
                        rpc_url=bc.rpc_url,
                        chain_id=bc.chain_id,
                    )
                    if chain_seed is not None:
                        seed_source = chain_seed.seed_for
                        logger.info("Candle cut will use chain-derived seed")
                except Exception as e:
                    logger.warning("chain candle seed unavailable: %s", e)

            self._epoch_scheduler = EpochScheduler(
                library=self._library,
                config=EpochSchedulerConfig(
                    interval_seconds=interval,
                    candle_min_seconds=candle_min,
                    candle_window_seconds=candle_window,
                ),
                on_close=_on_close,
                candle_seed_source=seed_source,
            )
            if candle_min > 0:
                logger.info(
                    "Epoch scheduler initialized (candle: min=%.0fs window=%.0fs)",
                    candle_min, candle_window,
                )
            else:
                logger.info("Epoch scheduler initialized (interval=%.0fs)", interval)
        except Exception as e:
            logger.warning("Failed to initialize epoch scheduler: %s", e)
            self._epoch_scheduler = None

    def attach_event_gossip(self, p2p_host: Any) -> bool:
        """Wire the substrate event-gossip transport over libp2p.

        Caller is responsible for waiting until ``p2p_host._ready_event``
        is set (the host's trio loop is up and accepting publishes).
        This method then constructs an ``EventGossip`` per RPB and
        binds it to the library service's local-event stream.

        Returns True on success, False otherwise.
        """
        if self._library is None:
            logger.debug("attach_event_gossip: library service not yet initialized")
            return False
        if p2p_host is None:
            logger.debug("attach_event_gossip: no p2p_host")
            return False
        if self._event_gossip is not None:
            return True  # already wired
        try:
            from .common.event_gossip import (
                EventGossip,
                Keypair,
                LibP2PTransport,
            )
        except Exception as e:
            logger.warning("attach_event_gossip: import failed: %s", e)
            return False

        try:
            from pathlib import Path
            keypair = self._load_or_create_gossip_keypair()
            transport = LibP2PTransport(p2p_host)
            self._event_gossip = EventGossip(
                rpb_address=str(self._library.rpb_address),
                keypair=keypair,
                transport=transport,
                library=self._library,
                seq_path=(Path(self._data_dir) / "world" / "gossip" / "sender_seq.json"
                          if self._data_dir else None),
                batch_log_path=(Path(self._data_dir) / "world" / "gossip" / "batches.jsonl"
                                if self._data_dir else None),
            )
            logger.info(
                "Event gossip attached for rpb=%s (sender_pubkey=%s...)",
                self._library.rpb_address,
                keypair.public_key.hex()[:16],
            )
            self._init_federated_close_driver()
            self._wire_batch_catchup(p2p_host)
            return True
        except Exception as e:
            logger.warning("attach_event_gossip: %s", e, exc_info=True)
            self._event_gossip = None
            return False

    def _wire_batch_catchup(self, p2p_host: Any) -> None:
        """Serve our batch log to peers catching up, and let the close ask
        peers for the batches we missed (issue #41). The close runs off the
        host's trio thread, so its request crosses with from_thread."""
        gossip = self._event_gossip
        if gossip is None or not hasattr(p2p_host, "set_batch_source"):
            return
        p2p_host.set_batch_source(gossip.topic, gossip.batch_log.serve)
        driver = self._federated_close_driver
        if driver is None:
            return

        def catchup(floor: float) -> int:
            import trio
            token = getattr(p2p_host, "_trio_token", None)
            if token is None:
                return 0
            request = gossip.catchup_request(floor)

            async def ask():
                return await p2p_host.request_batches(request)
            try:
                wires = trio.from_thread.run(ask, trio_token=token)
            except RuntimeError:   # already on the trio thread: skip
                return 0
            return gossip.ingest_catchup(wires, floor)

        driver.catchup = catchup

    def _init_federated_close_driver(self) -> None:
        """Construct the federated-close driver and subscribe it to
        library epoch-close events.

        Runs after EventGossip is attached. Each local epoch close
        triggers ``driver.run(local_close_result)`` which builds the
        canonical order from gossiped batches, runs
        ``federated_epoch_close``, and produces a deterministic
        authoritative result. Subscribers (chain submitter, etc.)
        receive the FederatedCloseResult.
        """
        if self._library is None or self._event_gossip is None:
            return
        try:
            from .common.federated_close_driver import FederatedCloseDriver
        except Exception as e:
            logger.warning("federated_close_driver import failed: %s", e)
            return
        # The carry-over files live in the library service's state dir.
        # Missing state dir: in-memory only.
        tool_regs_path = None
        tool_rounds_path = None
        tool_positions_path = None
        try:
            state_dir = getattr(self._library, "state_dir", None)
            if state_dir is not None:
                tool_regs_path = state_dir / "tool_registrations.json"
                tool_rounds_path = state_dir / "tool_rounds.json"
                # Positions carry the own_share estimates, a close INPUT.
                # A daemon that restarts without them closes against an
                # empty prior while its peers carry the accumulated one
                # (a deterministic fork). Same rebuildable-cache contract
                # as the registration map above.
                tool_positions_path = state_dir / "tool_positions.json"
        except Exception:
            tool_regs_path = None
            tool_rounds_path = None
            tool_positions_path = None
        self._federated_close_driver = FederatedCloseDriver(
            gossip=self._event_gossip,
            tool_registrations_path=tool_regs_path,
            tool_rounds_path=tool_rounds_path,
            tool_positions_path=tool_positions_path,
        )
        # Refuse (log + skip) any close this daemon can't compute
        # identically to its peers: no chain read access below, or no
        # carry-over files past genesis. See participation_blockers().
        self._federated_close_driver.enforce_preconditions = True

        def _charter_changed(text: str) -> None:
            cb = getattr(self, "on_charter_change", None)
            if cb is not None:
                cb(text)
            else:
                logger.error(
                    "the jurisdiction changed its charter; restart this "
                    "daemon to run the new charter (closes are halted)")

        self._federated_close_driver.on_charter_change = _charter_changed

        # Household voice + fee-recycled emission sourcing. Wired here —
        # NOT in attach_chain_submission — because these are CLOSE
        # INPUTS every honest daemon must derive identically: they gate
        # on chain READ access (substrate_address + rpc_url) only. A
        # read-only daemon that skipped this would close with
        # weights=None / floor pool while submitters close with real
        # values — a deterministic fork. No private key involved.
        try:
            bc = getattr(self.config, "blockchain", None)
            substrate_addr = getattr(bc, "substrate_address", "") or ""
            rpc_url = getattr(bc, "rpc_url", "") or ""
            # REP (voice) now lives DAO-side in RepToken (Decision
            # 2026-07-10). Empty => genesis regime: fees-only pool, no rep.
            rep_token_addr = getattr(bc, "rep_token_address", "") or ""
            # ToolBond (docs/multi_jurisdiction.md decision 10): ratings
            # and the reviewer share, pinned on the same snapshot.
            tool_bond_addr = getattr(bc, "tool_bond_address", "") or ""
            # DAO Registry: the charter in force, pinned to the same anchor
            # (docs/multi_jurisdiction.md decision 5).
            dao_registry_addr = getattr(bc, "dao_registry_address", "") or ""
            deploy_block = int(getattr(bc, "deploy_block", 0) or 0) or None
            # The DAO Governor: its proposal lifecycle bounds the epoch
            # length (Decision 2026-10-26, item 4).
            governor_addr = getattr(bc, "governor_address", "") or ""
        except Exception:
            substrate_addr, rpc_url, rep_token_addr = "", "", ""
            tool_bond_addr = ""
            dao_registry_addr = ""
            deploy_block = None
            governor_addr = ""
        if substrate_addr and rpc_url:
            def _voice_source(pin=None):
                from .common.voice_state import read_voice_state
                return read_voice_state(
                    substrate_addr, rpc_url, rep_token_address=rep_token_addr,
                    tool_bond_address=tool_bond_addr,
                    dao_registry_address=dao_registry_addr,
                    deploy_block=deploy_block,
                    governor_address=governor_addr,
                    pin=pin,
                )

            self._federated_close_driver.voice_source = _voice_source

            # Pipelined elections (Decision 2026-10-26): the latest anchors,
            # so the driver can close ahead of them and rewind on a
            # different winner.
            def _chain_source(tail: int = 16):
                from web3 import Web3
                from .common.voice_state import _VOICE_ABI, _anchor_cutoff
                w3 = Web3(Web3.HTTPProvider(rpc_url))
                c = w3.eth.contract(
                    address=Web3.to_checksum_address(substrate_addr),
                    abi=_VOICE_ABI)
                n = int(c.functions.anchorCount().call())
                anchors = []
                for i in range(max(0, n - tail), n):
                    a = c.functions.getAnchor(i).call()
                    anchors.append({"number": i + 1, "epoch_id": str(a[0]),
                                    "root": bytes(a[1]).hex(),
                                    "cutoff": _anchor_cutoff(a)})
                return {"anchor_count": n, "anchors": anchors}

            self._federated_close_driver.chain_source = _chain_source

            # Volume-ended epochs (Decision 2026-10-27): fees burned between
            # two times, from Substrate's on-chain checkpoints.
            _burn_contract: list = []

            def _burned_source(from_ts: int, to_ts: int) -> int:
                if not _burn_contract:
                    from web3 import Web3
                    w3 = Web3(Web3.HTTPProvider(rpc_url))
                    _burn_contract.append(w3.eth.contract(
                        address=Web3.to_checksum_address(substrate_addr),
                        abi=[{"name": "burnedBetween", "type": "function",
                              "stateMutability": "view",
                              "inputs": [{"name": "fromTs", "type": "uint256"},
                                         {"name": "toTs", "type": "uint256"}],
                              "outputs": [{"name": "", "type": "uint256"}]}]))
                return int(_burn_contract[0].functions.burnedBetween(
                    int(from_ts), int(to_ts)).call())

            self._federated_close_driver.burned_source = _burned_source

            # Cheap anchorCount read: the voice refresh (every input pinned
            # to the latest anchor) reruns only when a new anchor lands.
            def _anchor_count():
                from web3 import Web3
                from .common.voice_state import _VOICE_ABI
                w3 = Web3(Web3.HTTPProvider(rpc_url))
                return int(w3.eth.contract(
                    address=Web3.to_checksum_address(substrate_addr),
                    abi=_VOICE_ABI,
                ).functions.anchorCount().call())

            self._federated_close_driver.anchor_count_source = _anchor_count

            # Joiner bootstrap (payload schema 4): a daemon that first
            # boots after anchors exist has no carry-over files, and they
            # are only written by a close. The driver fetches the latest
            # anchor's carry_cid bundle from peers and verifies it against
            # the on-chain payloadHash before installing it; it observes
            # until that succeeds. Blobs come from the chain-submission
            # resolver (libp2p in the daemon), looked up per call because
            # it is attached after this driver.
            def _carry_source():
                from web3 import Web3
                from .common.anchored_state import (
                    CarryOverFetch,
                    fetch_carry_over_from_chain,
                )
                from .common.voice_state import _VOICE_ABI
                csd = getattr(self, "_chain_submission_driver", None)
                resolver = getattr(csd, "blob_resolver", None)
                if resolver is None:
                    logger.info(
                        "carry-over fetch: no blob resolver attached yet")
                    return CarryOverFetch(status="unavailable")
                w3 = Web3(Web3.HTTPProvider(rpc_url))
                contract = w3.eth.contract(
                    address=Web3.to_checksum_address(substrate_addr),
                    abi=_VOICE_ABI,
                )
                return fetch_carry_over_from_chain(contract, resolver)

            self._federated_close_driver.carry_source = _carry_source

            logger.info(
                "Voice/emission source wired (substrate=%s, rep_token=%s)",
                substrate_addr, rep_token_addr or "(none)",
            )

        def _on_local_close(local_result):
            try:
                fed = self._federated_close_driver.run(local_result)
            except Exception as e:
                logger.error(
                    "federated close driver failed: %s", e, exc_info=True,
                )
                return
            if fed is None:
                return
            for sub in list(self._federated_close_subscribers):
                try:
                    sub(fed)
                except Exception as e:
                    logger.error(
                        "federated-close subscriber failed: %s", e, exc_info=True,
                    )

        self._library.subscribe_epoch_closed(_on_local_close)
        logger.info("Federated-close driver wired into the library service")

    def add_federated_close_subscriber(self, handler: Any) -> None:
        """Register a callable invoked with the FederatedCloseResult
        each time a federated close completes. Used by chain wiring
        (Phase 10.3b/c) to anchor + submit per-agent mints.
        """
        self._federated_close_subscribers.append(handler)

    def attach_chain_submission(
        self,
        agent_chain_resolver: Any,
        *,
        blob_resolver: Any = None,
        owner_wallet: str = "",
    ) -> bool:
        """Wire on-chain anchor + per-agent mint submission.

        Requires Phase 10.3a (federated-close driver) to be wired
        first. Reads chain config from
        ``self.config.blockchain.substrate_address`` etc.; if any
        required field is empty, this is a no-op.

        ``agent_chain_resolver`` is a zero-arg callable returning a
        list of ``AgentChainIdentity(address, private_key)`` for every
        currently-registered agent on this daemon. The atn-side
        autonet_service walks ``agent_registry.list_agents()`` to
        produce this.

        Returns True when chain submission is now active.
        """
        if self._federated_close_driver is None:
            logger.warning(
                "attach_chain_submission: federated-close driver not wired"
            )
            return False
        try:
            from .common.chain_submission_driver import (
                ChainSubmissionConfig,
                ChainSubmissionDriver,
            )
        except Exception as e:
            logger.warning("chain_submission_driver import failed: %s", e)
            return False

        bc = self.config.blockchain
        cfg = ChainSubmissionConfig(
            substrate_address=getattr(bc, "substrate_address", "") or "",
            rpc_url=bc.rpc_url,
            chain_id=bc.chain_id,
            daemon_private_key=bc.private_key or "",
            governor_address=getattr(bc, "governor_address", "") or "",
            anchor_election_address=getattr(
                bc, "anchor_election_address", "") or "",
            rep_claim_address=getattr(bc, "rep_claim_address", "") or "",
            owner_wallet=owner_wallet,
        )
        from pathlib import Path
        try:
            driver = ChainSubmissionDriver(
                config=cfg,
                agent_chain_resolver=agent_chain_resolver,
                blob_resolver=blob_resolver,
                elections_path=(Path(self._data_dir) / "world" / "open_elections.json"
                                if self._data_dir else None),
            )
        except Exception as e:
            logger.warning("ChainSubmissionDriver build failed: %s", e)
            return False
        if not driver.enabled:
            logger.info("Chain submission disabled (no substrate_address or key)")
            return False

        self._chain_submission_driver = driver
        self.add_federated_close_subscriber(driver.handle_federated_close)
        # (Voice/emission sourcing is deliberately NOT wired here: it
        # gates on chain READ access and is attached with the close
        # driver itself — see the driver construction above — so
        # read-only daemons derive the same close inputs as submitters.)
        logger.info(
            "Chain submission attached (substrate=%s)", cfg.substrate_address,
        )
        return True

    def _load_or_create_gossip_keypair(self):
        """Load this daemon's persistent ed25519 gossip keypair, or
        create one and persist it under the data dir.

        The key is per-daemon (not per-agent) — it identifies the
        gossip-batch sender, separate from any on-chain agent identity.
        """
        from .common.event_gossip import Keypair
        import json
        from pathlib import Path
        if not self._data_dir:
            return Keypair.generate()
        # _data_dir may arrive as either str or Path depending on caller.
        key_dir = Path(self._data_dir) / "world" / "gossip"
        key_dir.mkdir(parents=True, exist_ok=True)
        key_path = key_dir / "keypair.json"
        if key_path.exists():
            try:
                data = json.loads(key_path.read_text(encoding="utf-8"))
                return Keypair(
                    public_key=bytes.fromhex(data["public_key"]),
                    private_key=bytes.fromhex(data["private_key"]),
                )
            except Exception as e:
                logger.warning("gossip keypair load failed (regenerating): %s", e)
        kp = Keypair.generate()
        key_path.write_text(
            json.dumps({
                "public_key": kp.public_key.hex(),
                "private_key": kp.private_key.hex(),
            }),
            encoding="utf-8",
        )
        return kp

    @staticmethod
    def _get_version() -> str:
        """Get the current node software version."""
        try:
            from nodes import __version__
            return __version__
        except (ImportError, AttributeError):
            return "0.1.0"

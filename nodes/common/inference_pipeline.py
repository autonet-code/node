"""
Inference Request Routing Through Module Pipeline.

Routes inference requests through guild-hosted modules using
the P2P activation relay protocol. Each module in the pipeline
is hosted by a different node (potentially in different guilds).

Story 9.1: Inference request routing through module pipeline.
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .intelligence_tiers import InferenceTier, InferenceTierManager
from .model_module import MODULE_NAMES, ModuleSpec

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """An inference request to be routed through the pipeline."""
    request_id: str
    tier: InferenceTier
    input_data: Dict[str, Any]   # Named inputs (e.g. {"images": tensor})
    requester_id: str
    timestamp: float = 0.0
    max_credits: float = 0.0     # ATN budget

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


@dataclass
class InferenceResult:
    """Result from the inference pipeline."""
    request_id: str
    outputs: Dict[str, Any]      # Named outputs from the last module
    modules_executed: List[str]
    total_latency_ms: float = 0.0
    credits_used: float = 0.0
    success: bool = True
    error: str = ""


@dataclass
class ModuleRoute:
    """Routing info for a single module in the pipeline."""
    module_name: str
    node_id: str                 # Node hosting this module
    guild_id: Optional[int] = None
    latency_ms: float = 0.0     # Expected latency to this node


class InferencePipeline:
    """
    Routes inference requests through the distributed module pipeline.

    For each request:
    1. Determine modules needed based on intelligence tier
    2. Find the best node for each module (lowest latency)
    3. Execute modules in sequence, forwarding activations via P2P
    4. Return final output to the requester

    The pipeline uses the P2P activation relay protocol
    (/autonet/activations/1.0.0) for inter-node tensor transfer.

    Usage:
        pipeline = InferencePipeline(tier_manager, peer_router)
        result = pipeline.execute(request)
    """

    def __init__(
        self,
        tier_manager: Optional[InferenceTierManager] = None,
        peer_router: Optional[Any] = None,  # PeerLatencyTracker from p2p module
    ):
        self.tier_manager = tier_manager or InferenceTierManager()
        self.peer_router = peer_router
        self._stats = PipelineStats()

    def plan_route(self, request: InferenceRequest) -> List[ModuleRoute]:
        """
        Plan the routing for an inference request.

        Selects the best node for each module based on:
        - Module availability (which nodes host which modules)
        - Network latency (prefer nearby nodes)
        - Guild membership (prefer intra-guild routing)

        Returns ordered list of ModuleRoute for pipeline execution.
        """
        modules = self.tier_manager.get_modules(request.tier)
        routes = []

        for module_name in modules:
            route = self._select_node_for_module(module_name)
            routes.append(route)

        logger.debug(
            f"Route planned for request {request.request_id}: "
            f"{[r.module_name + '@' + r.node_id for r in routes]}"
        )
        return routes

    def execute(self, request: InferenceRequest) -> InferenceResult:
        """
        Execute an inference request through the module pipeline.

        Steps:
        1. Plan route (select nodes for each module)
        2. For each module in order:
           a. Send activations to the hosting node
           b. Receive output activations
        3. Return final output

        If peer_router is None, executes locally (single-node mode).
        """
        start_time = time.time()
        self._stats.requests_total += 1

        try:
            routes = self.plan_route(request)
            modules_executed = []
            current_activations = request.input_data

            for route in routes:
                module_start = time.time()

                # Execute module (local or remote)
                output = self._execute_module(route, current_activations)

                if output is None:
                    self._stats.requests_failed += 1
                    return InferenceResult(
                        request_id=request.request_id,
                        outputs={},
                        modules_executed=modules_executed,
                        success=False,
                        error=f"Module {route.module_name} failed",
                    )

                current_activations.update(output)
                modules_executed.append(route.module_name)

                module_ms = (time.time() - module_start) * 1000
                logger.debug(
                    f"Module {route.module_name} executed in {module_ms:.1f}ms"
                )

            total_ms = (time.time() - start_time) * 1000
            credits = self.tier_manager.compute_cost(request.tier)

            self._stats.requests_completed += 1
            self._stats.total_latency_ms += total_ms

            return InferenceResult(
                request_id=request.request_id,
                outputs=current_activations,
                modules_executed=modules_executed,
                total_latency_ms=total_ms,
                credits_used=credits,
            )

        except Exception as e:
            self._stats.requests_failed += 1
            logger.error(f"Pipeline error: {e}", exc_info=True)
            return InferenceResult(
                request_id=request.request_id,
                outputs={},
                modules_executed=[],
                success=False,
                error=str(e),
            )

    def _execute_module(
        self,
        route: ModuleRoute,
        activations: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a single module, either locally or via P2P relay.

        In single-node mode (no peer_router), uses the local model.
        In distributed mode, sends activations via the P2P activation
        relay protocol and receives the output.
        """
        if self.peer_router is None or route.node_id == "local":
            # Local execution
            return self._execute_local(route.module_name, activations)
        else:
            # Remote execution via P2P
            return self._execute_remote(route, activations)

    def _execute_local(
        self,
        module_name: str,
        activations: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Execute a module locally using the attached model."""
        # This is a stub — in production, the InferenceNode holds
        # the ModularVLJEPA model and calls run_module() here.
        logger.debug(f"Local execution of {module_name} (stub)")
        return activations  # Pass through in stub mode

    def _execute_remote(
        self,
        route: ModuleRoute,
        activations: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a module remotely via P2P activation relay.

        Uses the /autonet/activations/1.0.0 protocol to:
        1. Serialize activations to bytes
        2. Send to the hosting node
        3. Receive output activations
        4. Deserialize
        """
        # This is a stub — in production, this uses the P2P host's
        # activation relay protocol (from p2p.py).
        logger.debug(
            f"Remote execution of {route.module_name} "
            f"on node {route.node_id} (stub)"
        )
        return activations  # Pass through in stub mode

    def _select_node_for_module(self, module_name: str) -> ModuleRoute:
        """
        Select the best node to execute a module.

        Uses the peer router to find the lowest-latency node
        that hosts the required module. Falls back to local.
        """
        if self.peer_router is not None:
            try:
                # In production: query peer_router for capable nodes
                # peers = self.peer_router.select_peers_for_route([module_name])
                # if peers:
                #     return ModuleRoute(module_name=module_name, node_id=peers[0].peer_id)
                pass
            except Exception as e:
                logger.debug(f"Peer selection failed for {module_name}: {e}")

        return ModuleRoute(module_name=module_name, node_id="local")

    @property
    def stats(self) -> "PipelineStats":
        return self._stats


@dataclass
class PipelineStats:
    """Statistics for the inference pipeline."""
    requests_total: int = 0
    requests_completed: int = 0
    requests_failed: int = 0
    total_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        if self.requests_completed == 0:
            return 0.0
        return self.total_latency_ms / self.requests_completed

    @property
    def success_rate(self) -> float:
        if self.requests_total == 0:
            return 0.0
        return self.requests_completed / self.requests_total

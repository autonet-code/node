"""
Autonet Inference Node

Loads published models and serves inference requests.
Watches MatureModelUpdated events to stay current, and
InferenceRequested events to process queries.
"""

import logging
import time
import torch
from dataclasses import dataclass
from typing import Optional

from ..common.contracts import ContractRegistry
from ..common.blob_store import BlobStore
from ..common.ml import SimpleNet, load_weights_into_model
from ..common.governance import GovernanceBridge
from ..common.inference_attestor import InferenceAttestor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class InferenceMetrics:
    """Metrics tracked by the inference node."""
    tasks_proposed: int = 0
    tasks_completed: int = 0
    solutions_committed: int = 0
    votes_submitted: int = 0
    aggregations_done: int = 0
    forced_errors_caught: int = 0
    errors: int = 0
    cycles: int = 0

    # Inference-specific
    models_loaded: int = 0
    requests_served: int = 0
    requests_failed: int = 0


class InferenceNode:
    """
    Autonomous inference node that:
    1. Watches MatureModelSet events for new models
    2. Downloads and loads model weights from blob store
    3. Serves inference requests
    4. Runs forward pass and submits results
    """

    def __init__(
        self,
        registry: ContractRegistry,
        store: BlobStore,
        node_id: str,
        config=None,
    ):
        self.registry = registry
        self.store = store
        self.node_id = node_id

        self.metrics = InferenceMetrics()
        self.running = False

        # Current loaded model
        self.model: Optional[torch.nn.Module] = None
        self.model_hash: Optional[str] = None

        # Track processed requests to avoid duplicates
        self._processed_requests = set()

        # Governance bridge and inference attestor
        self.governance = GovernanceBridge(registry, node_id)
        self.attestor = InferenceAttestor(
            self.governance, store,
            tokens_per_unit=1000,
            flush_threshold=5000,
        )

        logger.info(f"InferenceNode initialized: {node_id}")

    def run(self, max_cycles: Optional[int] = None, cycle_delay: float = 2.0):
        """Main execution loop."""
        self.running = True
        logger.info(f"InferenceNode {self.node_id} starting...")

        cycle = 0
        while self.running and (max_cycles is None or cycle < max_cycles):
            try:
                self._run_cycle()
                self.metrics.cycles += 1
                cycle += 1

                if self.running:
                    time.sleep(cycle_delay)

            except Exception as e:
                logger.error(f"Error in cycle {cycle}: {e}", exc_info=True)
                self.metrics.errors += 1
                time.sleep(cycle_delay)

        logger.info(
            f"InferenceNode {self.node_id} stopped after {cycle} cycles. "
            f"Models loaded: {self.metrics.models_loaded}, "
            f"Requests served: {self.metrics.requests_served}"
        )

    def stop(self):
        """Stop the node and flush any pending attestations."""
        self.running = False
        self.attestor.flush()
        logger.info(f"Stopping InferenceNode {self.node_id}...")

    def _run_cycle(self):
        """Execute one cycle: check for model updates, process inference requests."""
        self._check_model_updates()
        self._process_inference_requests()

    def _check_model_updates(self):
        """Watch for MatureModelSet events and load new models."""
        try:
            events = self.registry.get_new_events("RPB", "MatureModelSet")

            for event in events:
                weights_hash = event["args"]["weightsCid"]
                if not weights_hash or weights_hash == self.model_hash:
                    continue

                logger.info(
                    f"[{self.node_id}] New model published: {weights_hash[:16]}... "
                    f"Loading weights..."
                )

                weights_data = self.store.get_json(weights_hash)
                if not weights_data:
                    logger.error(
                        f"[{self.node_id}] Failed to download model: {weights_hash[:16]}..."
                    )
                    self.metrics.errors += 1
                    continue

                try:
                    model = SimpleNet()
                    self.model = load_weights_into_model(model, weights_data)
                    self.model_hash = weights_hash
                    self.metrics.models_loaded += 1
                    logger.info(
                        f"[{self.node_id}] Model loaded successfully "
                        f"(hash={weights_hash[:16]}...)"
                    )
                except Exception as e:
                    logger.error(f"[{self.node_id}] Failed to load model: {e}")
                    self.metrics.errors += 1

        except Exception as e:
            logger.error(f"Error checking model updates: {e}", exc_info=True)
            self.metrics.errors += 1

    def _process_inference_requests(self):
        """Watch for InferenceTaskSubmitted events and serve predictions."""
        if not self.model:
            return

        try:
            events = self.registry.get_new_events("TaskContract", "InferenceTaskSubmitted")

            for event in events:
                args = event["args"]
                inference_id = args["inferenceId"]

                if inference_id in self._processed_requests:
                    continue

                input_hash = args.get("inputCidHash", "")
                logger.info(
                    f"[{self.node_id}] Inference request {inference_id.hex()[:16]}: "
                    f"input={input_hash.hex()[:16] if isinstance(input_hash, bytes) else str(input_hash)[:16]}..."
                )

                self._serve_inference(inference_id, input_hash)

        except Exception as e:
            logger.error(f"Error processing inference requests: {e}", exc_info=True)
            self.metrics.errors += 1

    def _serve_inference(self, inference_id: bytes, input_hash):
        """Run inference on input and submit result."""
        try:
            input_hash_str = input_hash.hex() if isinstance(input_hash, bytes) else str(input_hash)
            # Download input data
            input_data = self.store.get_json(input_hash_str)
            if not input_data:
                logger.error(
                    f"[{self.node_id}] Failed to download input: {input_hash_str[:16]}..."
                )
                self.metrics.requests_failed += 1
                return

            # Prepare input tensor
            tensor = self._prepare_input(input_data)
            if tensor is None:
                logger.error(f"[{self.node_id}] Failed to prepare input tensor")
                self.metrics.requests_failed += 1
                return

            # Run inference
            with torch.no_grad():
                output = self.model(tensor)

            # Format output
            predictions = output.argmax(dim=1).tolist()
            probabilities = torch.softmax(output, dim=1).tolist()

            inference_id_hex = inference_id.hex() if isinstance(inference_id, bytes) else str(inference_id)
            result = {
                "inference_id": inference_id_hex,
                "predictions": predictions,
                "probabilities": probabilities,
                "model_hash": self.model_hash,
                "provider": self.node_id,
                "timestamp": int(time.time()),
            }

            # Store result in blob store
            output_hash = self.store.add_json(result)

            logger.info(
                f"[{self.node_id}] Inference result stored: "
                f"id={inference_id_hex[:16]}, prediction={predictions}, "
                f"output={output_hash[:16]}"
            )
            self.metrics.requests_served += 1
            self.metrics.tasks_completed += 1
            self._processed_requests.add(inference_id)

            # Attest inference usage (estimate token count from tensor size)
            input_size = tensor.numel()
            self.attestor.record_inference(
                provider="native",
                input_tokens=input_size,
                output_tokens=len(predictions) * 10,
                model=self.model_hash or "unknown",
                request_id=inference_id_hex,
            )

        except Exception as e:
            logger.error(f"[{self.node_id}] Inference error: {e}", exc_info=True)
            self.metrics.requests_failed += 1
            self.metrics.errors += 1

    def _prepare_input(self, input_data: dict) -> Optional[torch.Tensor]:
        """Convert input data dict to a PyTorch tensor for inference."""
        try:
            if "pixels" in input_data:
                # Raw pixel data
                tensor = torch.tensor(input_data["pixels"]).float()
                # Ensure correct shape: [batch, channels, height, width]
                if tensor.dim() == 2:
                    # Single grayscale image [H, W] -> [1, 1, H, W]
                    tensor = tensor.unsqueeze(0).unsqueeze(0)
                elif tensor.dim() == 3:
                    # [C, H, W] -> [1, C, H, W]
                    tensor = tensor.unsqueeze(0)
                return tensor

            elif "tensor" in input_data:
                # Pre-formatted tensor
                return torch.tensor(input_data["tensor"]).float()

            else:
                logger.warning(
                    f"[{self.node_id}] Unknown input format: {list(input_data.keys())}"
                )
                return None

        except Exception as e:
            logger.error(f"[{self.node_id}] Input preparation failed: {e}")
            return None

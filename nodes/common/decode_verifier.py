"""
Decode Trust Verification for Native Inference.

Ensures that decode nodes produce consistent, deterministic output
from the same latent plan (K-vector). Prevents decode nodes from
substituting semantics during local decoding.

Verification strategy:
  1. Given the same K-vector input, the decoder must produce the same tokens
  2. Multiple verifiers decode independently and compare
  3. Mismatches trigger a dispute (slash the malicious node)

Story 9.4: Decode trust verification (deterministic output matching).
"""

import hashlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class VerificationResult(Enum):
    VERIFIED = "verified"
    MISMATCH = "mismatch"
    INSUFFICIENT_VERIFIERS = "insufficient_verifiers"
    ERROR = "error"


@dataclass
class DecodeOutput:
    """Output from a single decode node."""
    node_id: str
    generated_ids: List[int]        # Token IDs
    latent_plan_hash: str           # SHA-256 of the input K-vector
    output_hash: str = ""           # SHA-256 of generated_ids
    temperature: float = 1.0
    seed: int = 0                   # RNG seed for deterministic decoding

    def __post_init__(self):
        if not self.output_hash:
            self.output_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Hash the generated token IDs for comparison."""
        h = hashlib.sha256()
        for token_id in self.generated_ids:
            h.update(token_id.to_bytes(4, "big"))
        return h.hexdigest()


@dataclass
class VerificationReport:
    """Result of verifying decode outputs from multiple nodes."""
    result: VerificationResult
    latent_plan_hash: str
    outputs: List[DecodeOutput]
    majority_hash: Optional[str] = None
    agreement_ratio: float = 0.0
    dissenting_nodes: List[str] = field(default_factory=list)

    @property
    def is_verified(self) -> bool:
        return self.result == VerificationResult.VERIFIED


class DecodeVerifier:
    """
    Verifies deterministic decode output from multiple nodes.

    For trustless inference, the latent plan (K-vector) is computed
    by the network. Decoding happens locally on the requester's node.
    But if decoding is outsourced, we need to verify the output.

    Protocol:
        1. Send same K-vector to N decode nodes (with fixed seed)
        2. Collect outputs
        3. Majority vote: if >threshold agree, output is trusted
        4. Dissenters are flagged for dispute

    Usage:
        verifier = DecodeVerifier(min_verifiers=3, agreement_threshold=0.67)
        verifier.submit_output(output_from_node_1)
        verifier.submit_output(output_from_node_2)
        verifier.submit_output(output_from_node_3)
        report = verifier.verify()
    """

    def __init__(
        self,
        min_verifiers: int = 2,
        agreement_threshold: float = 0.67,
    ):
        self.min_verifiers = min_verifiers
        self.agreement_threshold = agreement_threshold
        self._outputs: List[DecodeOutput] = []

    def submit_output(self, output: DecodeOutput) -> None:
        """Submit a decode output from a node."""
        self._outputs.append(output)
        logger.debug(
            f"Decode output submitted: node={output.node_id}, "
            f"hash={output.output_hash[:16]}..."
        )

    def verify(self) -> VerificationReport:
        """
        Verify submitted outputs using majority voting.

        Returns a VerificationReport with the result.
        """
        if len(self._outputs) < self.min_verifiers:
            return VerificationReport(
                result=VerificationResult.INSUFFICIENT_VERIFIERS,
                latent_plan_hash=self._outputs[0].latent_plan_hash if self._outputs else "",
                outputs=self._outputs,
                agreement_ratio=0.0,
            )

        # Verify all outputs reference the same latent plan
        plan_hashes = set(o.latent_plan_hash for o in self._outputs)
        if len(plan_hashes) > 1:
            logger.error(f"Outputs reference different latent plans: {plan_hashes}")
            return VerificationReport(
                result=VerificationResult.ERROR,
                latent_plan_hash="mixed",
                outputs=self._outputs,
            )

        latent_plan_hash = self._outputs[0].latent_plan_hash

        # Count votes by output hash
        vote_counts: Dict[str, int] = {}
        for output in self._outputs:
            vote_counts[output.output_hash] = vote_counts.get(output.output_hash, 0) + 1

        # Find majority
        total = len(self._outputs)
        majority_hash = max(vote_counts, key=vote_counts.get)
        majority_count = vote_counts[majority_hash]
        agreement_ratio = majority_count / total

        # Find dissenters
        dissenting_nodes = [
            o.node_id for o in self._outputs
            if o.output_hash != majority_hash
        ]

        if agreement_ratio >= self.agreement_threshold:
            result = VerificationResult.VERIFIED
            if dissenting_nodes:
                logger.warning(
                    f"Decode verified ({agreement_ratio:.0%}) but "
                    f"{len(dissenting_nodes)} dissenting nodes: {dissenting_nodes}"
                )
            else:
                logger.debug(f"Decode verified: 100% agreement from {total} nodes")
        else:
            result = VerificationResult.MISMATCH
            logger.error(
                f"Decode MISMATCH: only {agreement_ratio:.0%} agreement "
                f"({majority_count}/{total})"
            )

        return VerificationReport(
            result=result,
            latent_plan_hash=latent_plan_hash,
            outputs=self._outputs,
            majority_hash=majority_hash,
            agreement_ratio=agreement_ratio,
            dissenting_nodes=dissenting_nodes,
        )

    def reset(self) -> None:
        """Clear submitted outputs for a new verification round."""
        self._outputs.clear()


def hash_latent_plan(tensor) -> str:
    """
    Compute SHA-256 hash of a latent plan tensor.

    Used to identify the K-vector input for decode verification.

    Args:
        tensor: torch.Tensor of shape [B, K, D] or [K, D]

    Returns:
        Hex digest string
    """
    h = hashlib.sha256()
    # Ensure we hash the raw bytes deterministically
    data = tensor.detach().cpu().numpy().tobytes()
    h.update(data)
    return h.hexdigest()

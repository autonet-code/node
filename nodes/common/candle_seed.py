"""Chain-derived randomness for the candle epoch cutoff.

The candle close (docs/epoch_economics.md) needs a seed with two
properties:

  1. **Shared** — every daemon must derive the same cutoff, or they
     partition epoch events differently and federated close diverges.
  2. **Unknowable while the epoch is open** — otherwise the cutoff is
     predictable and last-second sniping comes back.

The local hash seed in ``EpochScheduler._draw_candle_cut`` has neither
property across daemons. This module provides both from public chain
data:

    seed = sha256( latestAnchorHash(at B)  ||  hash(B) )

where ``B`` is the FIRST block whose timestamp >= the window's end.
``hash(B)`` does not exist until the window has fully elapsed, so the
seed cannot be known early; both inputs are on-chain, so every daemon
that agrees on the window end derives the same bytes.

Remaining dependency (documented, not solved here): daemons must agree
on the window end itself. Today epoch open times are local wall clocks
— close enough on a synced fleet for dev (block granularity absorbs
sub-second skew), but mainnet federation should anchor epoch opens to
chain time. A VRF is the further upgrade path if block-producer grinding
ever becomes a realistic concern on the target chain.

Fail-open design: any error returns None and the scheduler falls back
to its local seed — a single-daemon network keeps working with no RPC.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def first_block_at_or_after(w3: Any, ts: float) -> Optional[Any]:
    """Binary-search the chain for the first block with
    ``block.timestamp >= ts``. Returns None if the chain hasn't
    reached ``ts`` yet (the window is still open in chain time).
    """
    latest = w3.eth.get_block("latest")
    if latest.timestamp < ts:
        return None
    lo, hi = 0, int(latest.number)
    # Invariant: block(hi).timestamp >= ts; find the smallest such.
    while lo < hi:
        mid = (lo + hi) // 2
        if w3.eth.get_block(mid).timestamp >= ts:
            hi = mid
        else:
            lo = mid + 1
    return w3.eth.get_block(lo)


class ChainCandleSeed:
    """Callable seed source for ``EpochScheduler``.

    ``seed_for(window_end_ts)`` returns 32 deterministic bytes once the
    chain has produced a block past the window end, else None.
    """

    def __init__(self, contract: Any, w3: Any):
        """``contract``: web3 contract for Substrate.sol (needs the
        ``latestAnchorHash`` view). ``w3``: the matching Web3."""
        self._contract = contract
        self._w3 = w3

    def seed_for(self, window_end_ts: float) -> Optional[bytes]:
        try:
            block = first_block_at_or_after(self._w3, window_end_ts)
            if block is None:
                return None
            # Read the anchor hash AS OF that block so daemons sampling
            # at different wall times still read the same value, even
            # if a new anchor lands between their reads.
            try:
                anchor_hash = self._contract.functions.latestAnchorHash().call(
                    block_identifier=int(block.number),
                )
            except Exception:
                # Some RPCs reject historical state reads; the anchor
                # hash changes at most once per epoch, so a head read
                # is almost always identical. Logged for forensics.
                logger.debug(
                    "historical latestAnchorHash read failed; using head",
                )
                anchor_hash = self._contract.functions.latestAnchorHash().call()
            seed = hashlib.sha256(
                bytes(anchor_hash) + bytes(block.hash)
            ).digest()
            logger.info(
                "candle seed from chain: block=%d ts=%d anchor=%s",
                block.number, block.timestamp, bytes(anchor_hash).hex()[:16],
            )
            return seed
        except Exception as e:
            logger.warning("chain candle seed unavailable: %s", e)
            return None


def build_chain_candle_seed(
    *,
    substrate_address: str,
    rpc_url: str,
    chain_id: int = 31337,
) -> Optional[ChainCandleSeed]:
    """Production factory: dial the RPC and load the Substrate ABI.
    Returns None when config is incomplete or the artifact is missing.
    """
    if not substrate_address or not rpc_url:
        return None
    try:
        import json
        from pathlib import Path

        from web3 import Web3

        artifact = Path(
            "C:/code/autonet/artifacts/contracts/core/Substrate.sol/Substrate.json"
        )
        abi = json.loads(artifact.read_text(encoding="utf-8"))["abi"]
        w3 = Web3(Web3.HTTPProvider(rpc_url))
        contract = w3.eth.contract(
            address=w3.to_checksum_address(substrate_address), abi=abi,
        )
        return ChainCandleSeed(contract, w3)
    except Exception as e:
        logger.warning("chain candle seed setup failed: %s", e)
        return None

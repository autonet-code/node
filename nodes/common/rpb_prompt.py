"""
RPB (Recursive Principled Body) Constitutional Prompt — V1

The V1 constitutional text is the Universal Declaration of Human Rights,
stored in constitution/v1_udhr.txt. The raw text is pushed into the
training feed config as plane-1 text and can be deployed to the blob store
as a content-addressed document (chain-anchorable later).

The RPB *evaluator* was deleted 2026-07-05 (its on-chain consumer,
EvolutionProposal.updateRPBPrompt, was removed long before). Nothing
evaluates proposals against this prose today — it is a founding-values
artifact that informs future governance votes. This module now only
loads and deploys the RAW text; there is no evaluation wrapping.

Usage:
    from nodes.common.rpb_prompt import deploy_v1_prompt, load_constitution_text

    # Deploy (once, by governance):
    content_hash = deploy_v1_prompt(blob_store)

    # Load the raw constitution text:
    text = load_constitution_text(blob_store, content_hash)
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Path to the v1 constitutional prompt text file
CONSTITUTION_DIR = Path(__file__).parent.parent.parent / "constitution"
V1_PROMPT_FILE = CONSTITUTION_DIR / "v1_udhr.txt"


def load_constitution_text(
    blob_store: Optional[object] = None,
    content_hash: Optional[str] = None,
) -> Optional[str]:
    """
    Load the RAW constitutional text.

    Tries in order:
    1. Blob store (by hash) — production path
    2. Local file (constitution/v1_udhr.txt) — development fallback

    Returns:
        The raw constitution text (stripped), or None if unavailable.
    """
    raw_text = None

    # Try blob store first
    if blob_store is not None and content_hash is not None:
        try:
            data = blob_store.get_bytes(content_hash)
            if data is not None:
                raw_text = data.decode("utf-8")
                logger.debug(f"Loaded constitution text from blob store: {content_hash[:16]}...")
        except Exception as e:
            logger.warning(f"Failed to load constitution from blob store: {e}")

    # Fall back to local file
    if raw_text is None and V1_PROMPT_FILE.exists():
        try:
            raw_text = V1_PROMPT_FILE.read_text(encoding="utf-8")
            logger.debug(f"Loaded constitution text from {V1_PROMPT_FILE}")
        except Exception as e:
            logger.warning(f"Failed to read local constitution file: {e}")

    if raw_text is None:
        return None

    return raw_text.strip()


def deploy_v1_prompt(blob_store) -> Optional[str]:
    """
    Store the v1 constitutional text in the blob store.

    Reads constitution/v1_udhr.txt, stores it, and returns the content
    hash. This hash is chain-anchorable later.

    Args:
        blob_store: BlobStore instance

    Returns:
        Content hash (SHA256 hex), or None if the file doesn't exist.
    """
    if not V1_PROMPT_FILE.exists():
        logger.error(
            f"Constitution text file not found: {V1_PROMPT_FILE}\n"
            f"Place the Universal Declaration of Human Rights text there."
        )
        return None

    text = V1_PROMPT_FILE.read_text(encoding="utf-8")
    content_hash = blob_store.add_bytes(text.encode("utf-8"))
    logger.info(
        f"V1 constitution text deployed: {content_hash[:16]}... "
        f"({len(text)} bytes)"
    )
    return content_hash


def get_prompt_file_path() -> Path:
    """Return the path where the v1 constitutional text should be placed."""
    return V1_PROMPT_FILE

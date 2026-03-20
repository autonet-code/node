"""
RPB (Recursive Principled Body) Constitutional Prompt — V1

The V1 constitutional prompt is the Universal Declaration of Human Rights.
The text is stored in constitution/v1_udhr.txt and deployed to the blob
store as a content-addressed document. Its hash is then registered on-chain
via EvolutionProposal.updateRPBPrompt(cid).

Nodes fetch the prompt CID from the chain, retrieve the content from the
blob store, and use it as the system prompt when evaluating evolution
proposals through an AI provider.

Usage:
    from nodes.common.rpb_prompt import deploy_v1_prompt, load_prompt_text

    # Deploy (once, by governance):
    content_hash = deploy_v1_prompt(blob_store)

    # Load (by RPBEvaluator):
    text = load_prompt_text(blob_store, content_hash)
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Path to the v1 constitutional prompt text file
CONSTITUTION_DIR = Path(__file__).parent.parent.parent / "constitution"
V1_PROMPT_FILE = CONSTITUTION_DIR / "v1_udhr.txt"

# Evaluation framing — wraps the constitutional text when used as a system prompt
EVALUATION_PREAMBLE = """\
You are an evaluator for the Autonet decentralized network. Your role is to
assess evolution proposals against the constitutional principles below.

For each proposal, determine:
1. Whether it aligns with or violates the constitutional principles
2. Your confidence level (0-100%) in the assessment
3. Specific principles that support or conflict with the proposal
4. Risks and benefits of adoption

Constitutional Principles:
=========================

"""

EVALUATION_SUFFIX = """

=========================

Evaluate the following proposal against these principles. Be rigorous but
not obstructionist. The constitution sets boundaries, not preferences.
Proposals that do not conflict with any principle should be approved.
Proposals that clearly violate a principle should be rejected with
specific citations. Ambiguous cases should note the tension and express
moderate confidence.
"""


def load_prompt_text(
    blob_store: Optional[object] = None,
    content_hash: Optional[str] = None,
) -> Optional[str]:
    """
    Load the constitutional prompt text.

    Tries in order:
    1. Blob store (by hash) — production path
    2. Local file (constitution/v1_udhr.txt) — development fallback

    Returns:
        The full evaluation prompt (preamble + constitution + suffix),
        or None if no prompt is available.
    """
    raw_text = None

    # Try blob store first
    if blob_store is not None and content_hash is not None:
        try:
            data = blob_store.get_bytes(content_hash)
            if data is not None:
                raw_text = data.decode("utf-8")
                logger.debug(f"Loaded constitutional prompt from blob store: {content_hash[:16]}...")
        except Exception as e:
            logger.warning(f"Failed to load prompt from blob store: {e}")

    # Fall back to local file
    if raw_text is None and V1_PROMPT_FILE.exists():
        try:
            raw_text = V1_PROMPT_FILE.read_text(encoding="utf-8")
            logger.debug(f"Loaded constitutional prompt from {V1_PROMPT_FILE}")
        except Exception as e:
            logger.warning(f"Failed to read local prompt file: {e}")

    if raw_text is None:
        return None

    return EVALUATION_PREAMBLE + raw_text.strip() + EVALUATION_SUFFIX


def deploy_v1_prompt(blob_store) -> Optional[str]:
    """
    Store the v1 constitutional prompt in the blob store.

    Reads constitution/v1_udhr.txt, stores it, and returns the content
    hash. This hash should then be registered on-chain via
    EvolutionProposal.updateRPBPrompt(hash).

    Args:
        blob_store: BlobStore instance

    Returns:
        Content hash (SHA256 hex), or None if the file doesn't exist.
    """
    if not V1_PROMPT_FILE.exists():
        logger.error(
            f"Constitutional prompt file not found: {V1_PROMPT_FILE}\n"
            f"Place the Universal Declaration of Human Rights text there."
        )
        return None

    text = V1_PROMPT_FILE.read_text(encoding="utf-8")
    content_hash = blob_store.add_bytes(text.encode("utf-8"))
    logger.info(
        f"V1 constitutional prompt deployed: {content_hash[:16]}... "
        f"({len(text)} bytes)"
    )
    return content_hash


def get_prompt_file_path() -> Path:
    """Return the path where the v1 constitutional prompt should be placed."""
    return V1_PROMPT_FILE

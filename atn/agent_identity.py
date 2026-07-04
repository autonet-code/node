"""Agent identity management - keypairs, addresses, lineage verification."""

import hashlib
import secrets
from typing import Tuple

from .models import AgentIdentity


def generate_keypair() -> Tuple[str, str]:
    """Generate a new keypair for an agent.

    Returns (private_key_hex, public_key_hex).
    Uses secp256k1 if available, falls back to Ed25519-like derivation.
    """
    try:
        from eth_account import Account
        account = Account.create()
        return account.key.hex(), account.address
    except ImportError:
        # Fallback: generate raw keypair
        private_key = secrets.token_hex(32)
        # Derive a deterministic "address" from the private key
        public_key = hashlib.sha256(bytes.fromhex(private_key)).hexdigest()
        address = "0x" + hashlib.sha256(public_key.encode()).hexdigest()[:40]
        return private_key, address


def derive_address(public_key_hex: str) -> str:
    """Derive 0x address from a public key."""
    return "0x" + hashlib.sha256(bytes.fromhex(public_key_hex)).hexdigest()[:40]


def compute_lineage_hash(
    agent_id: str,
    parent_lineage_hash: str | None,
    charter_text: str,
    public_key: str,
) -> str:
    """Compute the lineage hash for an agent.

    child_hash = sha256(parent_hash + charter + public_key)
    Root agents have parent_hash = "".
    """
    components = f"{parent_lineage_hash or ''}:{charter_text[:256]}:{public_key}"
    return hashlib.sha256(components.encode()).hexdigest()


def generate_agent_identity(
    agent_id: str,
    system_prompt: str = "",
    parent_identity: AgentIdentity | None = None,
    owner_root: str = "",
) -> Tuple[AgentIdentity, str]:
    """Generate a full agent identity.

    Returns (identity, private_key_hex).
    The private key should be stored securely by the caller.

    ``owner_root``: the owner wallet's 0x address, used as the lineage
    root for PARENTLESS agents so a rootless fleet still hangs from the
    human's identity, never from an installation (docs/tool_substrate.md,
    Owner-rooted registration). Free bonus, not a requirement: unknown
    owner → empty root, exactly as before. Authoritative fleet ownership
    lives on chain via sponsored registration, not in this hash.
    """
    private_key, address = generate_keypair()
    public_key = address  # In the fallback case, address serves as public key

    parent_hash = parent_identity.lineage_hash if parent_identity else (owner_root or None)
    lineage = compute_lineage_hash(agent_id, parent_hash, system_prompt, public_key)

    identity = AgentIdentity(
        public_key=public_key,
        address=address,
        lineage_hash=lineage,
        parent_lineage_hash=parent_hash,
    )

    return identity, private_key


def verify_lineage(identity: AgentIdentity, parent_identity: AgentIdentity | None, charter_text: str) -> bool:
    """Verify that an agent's lineage hash is valid."""
    expected = compute_lineage_hash(
        "",  # agent_id not used in hash
        parent_identity.lineage_hash if parent_identity else None,
        charter_text,
        identity.public_key,
    )
    return expected == identity.lineage_hash

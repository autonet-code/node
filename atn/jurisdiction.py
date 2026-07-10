"""Autonet jurisdiction constants — hardcoded network identity.

These values define the Autonet jurisdiction on Etherlink Shadownet.
The Governor address is the single root from which all other contract
addresses are discovered at runtime via the contract discovery chain:

    Governor → RepToken → Registry → RPB
    Governor → Timelock

This module is core-protected: modifications are detected by the
on-chain integrity check.  The jurisdiction identity should only
change through a governance-approved release.
"""

# Etherlink Shadownet
CHAIN_ID = 127823
RPC_URL = "https://node.shadownet.etherlink.com"

# DAO Governor — the single entry point for contract discovery
GOVERNOR_ADDRESS = "0xD5691B7c37a472D84D4E3b8d89CEb79a675dd36e"

# Deployer (for reference only — not used at runtime)
DEPLOYER_ADDRESS = "0x06E5b15Bc39f921e1503073dBb8A5dA2Fc6220E9"

# Autonet as the code says it is (2026-10-28)

Written from the code alone: four readers worked on a copy of `wan-run`
(at 782bdc8) with every Markdown and text file deleted, so nothing here comes
from the existing docs. Each file cites `file:line` and ends with
"Inconsistencies and dead code". Use these to find where the hand-written
docs (`docs/architecture.md`, `docs/economy_design.md` and older pages) have
drifted from the code.

| File | Covers |
|---|---|
| `runtime_and_agents.md` | CLI and processes, boot order, listeners and their auth, agents and keys, owner binding, operator key, the MCP server, configuration and env, tool store and containment |
| `networking.md` | libp2p host, transports and proxies, NAT and reach, gossip and signing, blobs, the relay, seeds, deployment shapes and harnesses |
| `epoch_and_mint.md` | events, the federated close, every mint rule with its numbers, the cutoff rule, anchoring and elections, claims, reviews, containment, sanctions, Registry keys |
| `contracts.md` | Substrate, ServiceMarket, AnchorElection, the external contracts as the daemon calls them, deployment scripts, address resolution |

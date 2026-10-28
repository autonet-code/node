# Architecture: the whole system on one page (2026-10-28)

Kept current: every decision that changes a component, a connection or a
user journey updates this page in the same change. The economy's rules are
in `economy_design.md`. Items marked **OPEN** are not settled yet.

## Components

**Daemon (`atn`, this repo: `atn/` runtime + `nodes/`).** One per OS user:
its state is that user's `~/.atn` (lock per data dir), and it belongs to one
household. It joins one jurisdiction at a time (switching restarts it). It:

- hosts agents and keeps their keys (age-encrypted vault);
- runs tools under containment (`tool_guard`, consensus tiers);
- gossips events over libp2p, serves blobs, runs the federated close,
  votes epoch elections and submits claims through the operator key;
- reviews on its agents' behalf (commit-reveal) and reveals at epoch end;
- keeps working when the chain is silent; only settlement waits.

It has three listeners:

| port | who | auth |
|---|---|---|
| 7700 | owner, same machine (app, `atn send`, MCP owner mode) | loopback only, pre-authed |
| 7701 | owner, remote (app in a browser) | owner wallet signature |
| 7710 | guest agents (`ATN_MCP_TOKEN`) | bearer token, allowlist of library and review calls |

**Agents.** On-chain identities the daemon hosts. **Native**: the daemon
runs the agent loop with a provider the owner chose (no default provider).
**External**: an outside harness (Claude Code, Codex, any MCP client) is the
brain; the daemon signs and acts for it. External is the main journey:
people bring the agents they already use.

**MCP server (`python -m atn.mcp_server`).** A thin proxy from an MCP client
to the daemon. Modes: owner, owner acting as one agent (`ATN_MCP_AGENT`),
guest (`ATN_MCP_TOKEN`). On connect it tells the agent to read
`atn_guide` (`atn/external_agents.md`), which is the whole manual. **OPEN:**
should launching the MCP server boot the daemon when none is running, so a
coding agent's start is the only start (today it errors and says to run
`atn`).

**App (`C:\code\atn_web_apps`, Flutter, branch `unified-apps`; never
`atn_web`).** The owner's dashboard and wallet: connect the owner wallet,
approve agent registration and adoptions, delegate REP to the operator,
watch earnings, epochs and the substrate. It talks to the daemon's 7700
locally or 7701 remotely. Agents don't need it.

**Relay (`atn relay`).** A networking-only node: seed, gossip and blob
relay, reach checks. No close, chain or agents. Seeds are listed in the
DAO Registry by hostname (`jurisdiction.<id>.bootstrap`, never an IP).

**Chain.** In this repo (`contracts/core/`): `Substrate.sol` (ATN, epoch
anchors, agent registry with owner binding, mint records, `payForService`
with the burn, genesis claim), `AnchorElection.sol`, `ServiceMarket.sol`.
In `C:\code\dao\trustless-contracts` (dao owns it; coordinate through
`chat.txt`): governor + timelock, Registry (every parameter lives here),
RepToken, AutonetRepClaim, ToolBond, AutonetVault, and the AutonetSetup
module that werule uses to create a jurisdiction. The indexer
(`EightRice/indexer`, dao owns) mirrors chain events to Firestore for the
agent directory.

## How things connect

- Daemon to daemon: libp2p gossip per jurisdiction (topic
  `<chain_id>:<DAO address>`), over TCP or WebSocket. Behind NAT a daemon
  dials out only; UPnP and a reach check make home machines reachable when
  the router allows. Behind an HTTP proxy (cloud sandboxes) it dials `wss`
  through CONNECT.
- Daemon to chain: one RPC URL. Addresses come from the jurisdiction's
  Registry, read pinned to the previous anchor so every daemon agrees.
- Household on chain: the owner wallet claims ATN and REP; the daemon's
  operator key registers bound to it, gets the owner's REP by delegation,
  proposes and votes epoch elections, and submits claims.

## How people run it

| who | runs | how |
|---|---|---|
| Desktop user | app + daemon | install, open the app, connect a wallet |
| Coding-agent user | daemon + MCP server | `pip install autonet-computer`, `atn --headless`, `claude mcp add atn -e ATN_MCP_AGENT=<id> -- python -m atn.mcp_server`; the agent reads the guide |
| Server / cloud VM | daemon in Docker (`docker/Dockerfile.daemon`) | NAT'd, HTTPS egress only; agents connect as guests |
| Network supporter | `atn relay` | a public host; its hostname goes in the Registry |

## Networks

- **Local hardhat**: every change is proven here first (S0 processes, S1
  containers behind an HTTPS-only proxy: `scripts/s0_local_run.py`,
  `scripts/s1_container_run.py`).
- **WAN run** (next, cloud credits expire around 2026-11-02): S2 pilot then
  S3 fleet. Relay and RPC on the Hetzner box behind Cloudflare
  (`relay.` / `rpc.autonet.computer`), daemons on cloud VMs, Claude Code
  cloud sessions as the agents through MCP. Purpose: real-world networking,
  and a baseline tool library built by real agents.
- **Shadownet** (Etherlink testnet): no deploy until the owner confirms on
  local hardhat.
- **Mainnet** (Polygon or Arbitrum): virgin mint; the WAN run's tools are
  re-published from the owner's household as the starting library.

## Repos and rules

- `C:\code\autonet` (backend, run branch `wan-run`), `C:\code\atn_web_apps`
  (app), `C:\code\dao\trustless-contracts` (dao's).
- Commit locally; push only when the owner says. Consensus or ABI changes
  are `Breaking upgrade:` commits (flag day: every daemon on the new build
  and a fresh Substrate).
- Economy changes need the owner's sign-off.

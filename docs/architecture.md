# Architecture: the whole system on one page (2026-10-28)

Kept current: every decision that changes a component, a connection or a
user journey updates this page in the same change. The economy's rules are
in `economy_design.md`. Items marked **OPEN** are not settled yet.

## Components

**Daemon (`atn`, this repo: `atn/` runtime + `nodes/`).** One per OS user:
its state is that user's `~/.atn` (lock per data dir), and it belongs to one
household. It joins one jurisdiction at a time (switching restarts it).
With none chosen it still starts: agents and tools work locally, nothing
joins a network, and the CLI banner and the app say so. The app lists the
network's jurisdictions from the DAO directory (homebase Firestore), the one
in `contracts/<network>.recommendedJurisdiction` marked recommended and
first; the owner chooses, and the daemon reads the chosen one's Registry
at every start. Joining starts its network at once (gossip, the tool
substrate, tool use, the seed address); registering an agent is optional
and only adds consensus (closes, elections, claims), which is how a
household earns. Headless: `atn jurisdiction packaged` (the
release's `registry.json`) or `atn jurisdiction <dao_registry> <rpc_url>`. A data dir whose agents were already registered
in the packaged jurisdiction keeps it. It:

- hosts agents and keeps their keys (age-encrypted vault);
- runs tools under containment (`tool_guard`, consensus tiers);
- gossips events over libp2p, serves blobs, runs the federated close,
  votes epoch elections and submits claims through the operator key;
- reviews on its agents' behalf (commit-reveal): commits when its close
  fires, reveals once that epoch is anchored;
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
`atn_web`).** The owner's dashboard and wallet. The avatar opens two
tabs. Household: owner (wallet, ATN, voice, claims), operator key
(including delegating REP to the operator, the one governance write the
app does; everything else in governance is werule's), agents
(registration, owner binding), governance read-only with a link to werule,
and this daemon's network (with its seed address, to propose for the
Registry's seed list). Jurisdiction: contracts, charter, epochs, the
switch, and the Substrate view. The home section keeps Docs and the
Novelty map. Each agent's window keeps only
its own identity, registration indicator and substrate position. The app
also approves adoptions. It talks to the daemon's 7700 locally or 7701
remotely. Agents don't need it.

**Relay (`atn relay`).** A networking-only node: seed, gossip and blob
relay, a short log of event batches for catch-up, the state blobs each
anchor commits (daemons offer them after every close), reach checks. No close, chain or agents. Seeds are listed in the
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
  `/autonet/world-events/<chain_id>:<DAO address>`), over WebSocket by
  default (TCP by `AUTONET_P2P_TRANSPORT`; one transport per network), on a
  port kept across restarts (`p2p.port` in the data dir) so the address can
  be a seed. Before closing an
  epoch, a daemon asks its peers (relays first) for that epoch's batches it
  missed (`/autonet/batches/1.0.0`), so a late joiner or a daemon that lost
  its peers closes the same set as everyone else. Behind NAT a daemon
  dials out only; UPnP and a reach check make home machines reachable when
  the router allows. Behind an HTTP proxy (cloud sandboxes) it dials `wss`
  through CONNECT. A running daemon re-reads the Registry seed list every
  few minutes (`p2p.seed_refresh_interval`, default 300 s) and dials new
  seeds; with no peers it redials its seeds at once.
- Buyer to service provider: libp2p `/autonet/service/1.0.0` to the
  provider's on-chain peer id, direct or one hop through a relay (so NAT'd
  providers can sell); the provider's wss endpoint only when libp2p cannot
  deliver.
- Daemon to chain: one RPC URL. Contract addresses come from the
  jurisdiction's Registry, read at each start (latest block). The close's
  parameters (REP, charter, sanctions, containment, epoch rules) are read
  pinned to the previous anchor so every daemon agrees. A jurisdiction
  won't start without `substrate`, `rep_token`, `tool_bond`, `dao` and
  `timelock` in its Registry and `epoch_rules` "2". Chain calls run off the
  daemon's event loop.
- Household on chain: the owner wallet claims ATN and REP; the daemon's
  operator key registers bound to it, gets the owner's REP by delegation,
  proposes and votes epoch elections, and submits claims.

## How people run it

| who | runs | how |
|---|---|---|
| Desktop user | app + daemon | install, open the app, choose a jurisdiction (the recommended one is marked), connect a wallet |
| Coding-agent user | daemon + MCP server | `pip install autonet-computer`, `atn jurisdiction packaged`, `atn --headless`, `claude mcp add atn -e ATN_MCP_AGENT=<id> -- python -m atn.mcp_server`; the agent reads the guide |
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

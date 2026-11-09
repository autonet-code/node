# Architecture: the whole system on one page (2026-11-10)

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
household earns. Headless: `atn jurisdiction recommended`, or an agent's
`POST /agents/<id>/join_recommended_jurisdiction`: the recommended one is
read from `registry.json` on master in `autonet-code/node` (the release's
copy if unreachable), so a migration needs no release, and joining pins
its DAO in the data dir (a later edit to the file moves no running
daemon). Also `atn jurisdiction <dao_registry> <rpc_url>`, or `packaged`
(the release's copy, unpinned). A data dir whose agents were already
registered in the packaged jurisdiction keeps it. It:

- hosts agents and keeps their keys (age-encrypted vault);
- runs tools (`use_tool`) as subprocesses that get only the environment
  variables and secrets their manifest declares, never the daemon's own;
  nothing contains them (Decision 2026-11-09: the framework rates tools,
  what a household runs is its own policy). A tool is a codebase anywhere
  on the machine, snapshotted by content (`docs/tool_codebases.md`), or a
  single script;
- gossips events over libp2p (a published tool's one `tool_registered`
  event), serves blobs and announces the tool blobs it holds in the DHT;
- with a registered agent or operator: runs the federated close, votes epoch elections
  and submits claims through the operator key. Without one it closes
  nothing and installs each anchor's state instead. That state is the
  anchor's carry bundle (registrations, review rounds, ratings), the same
  one a joiner or a rewind installs;
- reviews on its agents' behalf (commit-reveal): commits when its close
  fires, reveals once that epoch is anchored. Every review carries
  correctness, simplicity, novelty and malware; a malware flag's evidence
  is a blob the daemon keeps and serves (`fetch_evidence`). WS
  `tool_round_reviews` lists every household's on-chain reviews of a tool
  (from the ToolBond logs the close reads: reviewer, household, the four
  fields, charter scores, weight, accuracy once settled, evidence hash),
  so the app can fetch any flag's evidence. A household without REP at
  the epoch's start sends no reviews;
- searches the tool library locally (`atn/tool_search.py`): keyword and
  semantic over every manifest it holds, in relevance order, with the
  rating as of the latest close (the four fields' means, the malware
  share, the charter axes), bond, author and the household's own relation
  as metadata, with the highest malware share among its dependencies.
  Unbonded and flagged tools are listed too; pruned ones are not. The
  model (all-MiniLM-L6-v2, pinned by revision and sha256,
  `atn/search_model.py`) downloads in the background on the first run,
  from Hugging Face or from peers that hold it, into `<data_dir>/models`;
  search is keyword-only until it loads. WS: `search_model_status`, event
  `search.model_status`;
- keeps working when the chain is silent; only settlement waits.
- prunes the tools the close lists (Decision 2026-11-10: an uncleared
  malware verdict, a tool built on one, a registration the close refused
  for importing one): drops them from the library index, deletes their
  manifest and code blobs except what the household installed, never
  serves or fetches them (`nodes/common/pruned_tools.py`), and adoption
  refuses them and any composite built on them. A DAO clearing lists a
  tool again.
- keeps its sender chain, batches and open elections across restarts (`world/gossip/`,
  `world/open_elections.json`; the batch log holds two weeks), and never drops an unanchored epoch: an
  election lasts days and every later epoch waits on it.

It has two listeners:

| port | who | auth |
|---|---|---|
| 7700 | owner, same machine (app, `atn send`, MCP owner mode) | loopback only, pre-authed; changing the owner wallet and exporting a key also need the owner wallet's signature, since agents on the machine can reach it too |
| 7701 | owner, remote (app in a browser) | owner wallet signature |
| 7702 | agents on the same machine (HTTP API) | loopback only, each call scoped to the agent in its path |

**Agents.** On-chain identities the daemon hosts. **Native**: the daemon
runs the agent loop with a provider the owner chose (no default provider).
**External**: an outside harness (Claude Code, Codex, anything with a shell)
is the brain; the daemon signs and acts for it. External is the main
journey: people bring the agents they already use. On chain an external
agent IS the household's operator (Decision 2026-11-08): it makes no
keypair, its address is the operator's, and the daemon signs everything it
does (publishing and bonds, reviews, service payments and channels,
`send_transaction`) with the operator key. So a household's reviews go out
as one batch per epoch from one address, and the owner funds only the
operator. There is no registration per external agent: each one counts as
registered once the operator is (`operator_identity.json` in the data dir).
Each session keeps its own id, token and Activity log. A tool one external
session adopts is installed for all of them to use, check out and review
(Decision 2026-11-09); each is reminded only of the tools it adopted.
Native agents keep their own keypairs and registrations, and lineage
scoping of their tools. Native agents are an
opt-in mode (`atn/native_mode.py`, off on a fresh daemon): the app hides
their surfaces while it is off, and the daemon refuses to create them.
Robots are the case that keeps it: an always-on loop no harness drives. A
fresh daemon has no
agents (the seeded Kevin concierge was removed 2026-11-02); the app's empty
Agents screen gives the one line to tell an agent.

**Agent API (`atn/agent_api.py`, HTTP on the WebSocket port + 2).** The
path for every external agent: anything with a shell explores it from
`GET /` and acts with plain HTTP, so nothing loads when its session starts.
The tree is built from the daemon's own tool definitions (`/commands`) and
searches (`/agents/<id>/tools?q=`, `/services?q=`). Identity is in the path
(`/agents/<id>/<command>`): each call runs through the WebSocket handler as
a session scoped to that agent's subtree (no owner-only commands, no key
export); `POST /agents` creates an external agent and returns its token
(shown once, kept hashed in `agent_tokens.json`), which every call under
`/agents/<id>/` carries as a bearer token; `atn agent-token <id>` issues a
new one. `GET /agents` lists the external agents so a new session finds its
own; a call may carry a `note` (why), shown in Activity. `/commands` lists
what external agents use; the commands for native agents are behind
`?all=1`. `send_transaction` (to, calldata, value, optional gas and note)
signs any transaction with the operator key, waits for the receipt and
returns its status, block, gas used and revert reason: how an agent takes
part in governance for the owner (werule proposals and votes, ToolBond
verdict clearings). No approval step, since the operator is an accepted hot wallet;
Activity records each one's target, selector, value and note. It is for
external agents only; native agents and the owner surface are refused.
`export_operator_key` stays off this API. `GET /guide` serves
`atn/external_agents.md`, the whole manual. Refuses browser requests
(`Origin`) and non-localhost `Host`. (It replaced the MCP proxy, deleted
2026-11-02: an MCP server has to load when a session starts.)

**Activity.** Every call an external agent makes, by any path, is kept in
its Activity log (`atn/agent_activity.py`) and streamed to the app.

**App (`C:\code\atn_web_apps`, Flutter, branch `unified-apps`; never
`atn_web`).** The owner's dashboard and wallet. The avatar opens two
tabs. Household: owner (wallet, ATN, voice, claims), operator key
(including delegating REP to the operator, the one governance write the
app does; everything else in governance is werule's), agents
(registration, owner binding), governance read-only with a link to werule,
and this daemon's network (with its seed address, to propose for the
Registry's seed list). Jurisdiction: contracts, charter, epochs (the anchored ones, read from
Substrate by `epoch_history`; never the daemon's local ticks), the
switch, and the Substrate view. The home section keeps Docs and the
Novelty map. Each agent's window keeps only
its own identity, registration indicator and substrate position; an
external agent's window shows, in place of a chat, its Activity: every call
it made through the daemon, kept by the daemon (`agent_activity`, the
newest 500 per agent in `<data_dir>/activity/`). The app
also approves the one kind of adoption an agent can't do alone: a tool that replaces core built-ins (agents install other tools themselves). It talks to the daemon's 7700 locally or 7701
remotely. Agents don't need it.

**Relay (`atn relay`).** A networking-only node: seed, circuit relay v2
(NAT'd daemons reserve on it and are reached through it), DHT server,
gossip relay, a short log of event batches for catch-up, the state blobs
each anchor commits (daemons offer them after every close; it provides them
in the DHT), reach checks. No close, chain or agents. Seeds are listed in
the DAO Registry by hostname (`jurisdiction.<id>.bootstrap`, never an IP).

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

- Daemon to daemon: a libp2p mesh (py-libp2p 0.8.1,
  `docs/p2p_networking.md`). Every daemon listens on TCP and WebSocket at
  once, on ports kept across restarts (`p2p.port` for WebSocket, the one
  seeds name, and `p2p.tcp_port` in the data dir), and dials whichever a
  peer offers. A NAT'd daemon reserves a slot on the relays it is connected
  to and advertises `/p2p-circuit` addresses, so anyone reaches it through
  a relay; DCUtR then turns those links into direct ones by TCP hole
  punching. A DHT on every daemon finds peers and blob providers; the mesh
  loop dials towards 8 peers from the DHT, capability gossip and a peer
  cache in the data dir (`p2p_peers.json`, which lets a restart rejoin with
  every seed down), and the connection manager prunes above 48. Gossip per
  jurisdiction (topic `/autonet/world-events/<chain_id>:<DAO address>`)
  rides those links, so it keeps flowing with every relay down. Before
  closing an epoch, a daemon asks its peers (relays first) for that epoch's
  batches it missed (`/autonet/batches/1.0.0`), so a late joiner or a daemon
  that lost its peers closes the same set as everyone else. UPnP and a
  reach check make home machines directly reachable when the router allows.
  Behind an HTTP(S) or SOCKS proxy (cloud sandboxes) a daemon runs WebSocket
  only and dials through the proxy; it is reached through its relays. A
  running daemon re-reads the Registry seed list every few minutes
  (`p2p.seed_refresh_interval`, default 300 s) and dials new seeds; with no
  peers it redials its seeds and cached peers at once.
- Tool code and blobs: a daemon holding a blob a tool needs announces it in
  the DHT (on publish, before the registration event is gossiped; on adopt;
  re-announced every 30 min). A fetch asks connected peers and the blob's
  DHT providers at once and dials the providers directly or through a relay
  circuit; peers never forward blob requests.
- Buyer to service provider: libp2p `/autonet/service/1.0.0` to the
  provider's on-chain peer id, dialled directly or through the provider's
  relay circuit (so NAT'd providers can sell), else forwarded one hop by a
  relay connected to it; the provider's wss endpoint only when libp2p
  cannot deliver.
- Daemon to chain: one RPC URL. Contract addresses come from the
  jurisdiction's Registry, read at each start (latest block). The close's
  parameters (REP, charter, sanctions, mint gate, review params, epoch rules) are read
  pinned to the previous anchor so every daemon agrees. A jurisdiction
  won't start without `substrate`, `rep_token`, `tool_bond`, `dao` and
  `timelock` in its Registry and `epoch_rules` "2". Chain calls run off the
  daemon's event loop.
- Household on chain: the owner wallet claims ATN and REP; the daemon's
  operator key registers bound to it, gets the owner's REP by delegation,
  proposes and votes epoch elections, and submits claims. It also signs
  every external agent's actions, so its transactions come from several
  paths at once; each takes its nonce from one manager per address
  (`nodes/common/tx_nonce.py`), so none share one.

## How people run it

| who | runs | how |
|---|---|---|
| Desktop user | app + daemon | install, open the app, choose a jurisdiction (the recommended one is marked), connect a wallet |
| Coding-agent user | daemon + any agent harness | `pip install autonet-computer`, `atn --headless`; the agent starts at the agent API (`GET /`), reads the guide and joins the recommended jurisdiction itself |
| Server / cloud VM | daemon on the VM (`scripts/s2/vm_up.py`), or in Docker (`docker/Dockerfile.daemon`) | NAT'd, HTTPS egress only; the agent runs on the same VM and talks to its daemon with `atn send` (WAN run) or the MCP server |
| Network supporter | `atn relay` | a public host; its hostname goes in the Registry |

## Networks

- **Local hardhat**: every change is proven here first (S0 processes, S1
  containers behind an HTTPS-only proxy: `scripts/s0_local_run.py`,
  `scripts/s1_container_run.py`).
- **WAN run** (next, cloud credits expire around 2026-11-02): S2 pilot then
  S3 fleet. Relay and RPC on the Hetzner box behind Cloudflare
  (`relay.` / `rpc.autonet.computer`), daemons on cloud VMs, Claude Code
  cloud sessions as the agents, on the same VM as their daemon, through
  `atn send` (a cloud session loads MCP servers before its daemon exists,
  `docs/wan_fleet_brief.md`). Purpose: real-world networking,
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

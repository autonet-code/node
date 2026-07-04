# Services market: decentralized monetizable APIs

Status: DESIGN — ratified in discussion 2026-07-04. Companion to
`docs/tool_substrate.md` v2 (which holds the Tools/Services line:
tools = local, known, minted commons; services = remote counterparties,
market-priced). This doc specifies the market rail.

A Service is a remote API published by an agent (ultimately by its
human benefactor): general-purpose, priced per work item, payable in
**any ERC20** — service commerce is deliberately independent of ATN's
constitutional roles (mint, reputation, alignment pricing). The system
is decentralized because it leverages the consensus we already have:
gossip for liveness/discovery, chain for identity/settlement, blob
store for content.

## What the network can and cannot know

Execution integrity of a remote endpoint is unknowable in principle
(any script with a 0x key is indistinguishable from a faithful daemon).
So Services get NO substrate standing, NO verdict-layer claims, NO
mint, NO decay math. The trust basis is purely behavioral:

- identity — signed by the agent's 0x key (chain-verified)
- payment — atomic; not served = not paid
- track record — reviews signed by people who PAID (receipts with
  skin in the game; success rate + volume + dispute rate displayed,
  priced by buyers, not by consensus)

Master-disable / fingerprinting is fleet hygiene for honest daemons,
not a defense against dishonest ones. The defense is that lying has a
wallet attached and receipts are forever.

## The pieces

### 1. Service spec (blob store)

sha256-addressed JSON, same rail as tool manifests:

```json
{
  "kind": "service_spec",
  "name": "transcribe_audio",
  "description": "Speech-to-text, per audio minute.",
  "input_schema": { ... },          // work-item request shape
  "output_schema": { ... },
  "author": "<agent-id>",
  "author_pubkey": "0x...",
  "author_sig": "...",              // over canonical bytes, sig excluded
  "ask": { "token": "0x...ERC20", "amount": "1000000", "unit": "per_item" },
  "endpoint_hint": "wss://...",     // mutable presence lives on-chain, this is advisory
  "version_of": null,
  "created_ts": 0
}
```

### 2. Chain: ONE registry, not per-service contracts

Per-service contracts buy custom logic no v1 service needs and cost a
factory + N deployments + growing audit surface. Instead:

- `ServiceRegistry` (or folded into Substrate.sol as the Service
  primitive reborn): `registerService(specDigest, token, askAmount)`,
  `updateServiceAsk`, `retireService`. Events → indexer → Firestore
  `services` collection (chain = truth, blob = storage, Firestore =
  web2 cache; exact same doctrine as agents/tools).
- `ServiceEscrow` — settlement. Two modes:
  - **v1 escrow-per-request**: client deposits (token, amount,
    requestId), provider delivers, client releases (or timeout logic —
    sims decide the griefing rules). Fully atomic, gas-heavy.
  - **v1.5 payment channel** ("prepaid credits done trustlessly"):
    client deposits once; hands provider signed vouchers
    (cumulative_amount, request_id) off-chain per work item; provider
    claims the latest voucher on close; remainder refunds. Two txs for
    N requests, per-item granularity preserved: not served = no
    voucher. Streaming (Sablier-style) only for subscription-shaped
    services — wrong default for per-item work.

### 3. Daemon as server: the wss rail, reused

The remote-daemon connection mechanism already exists: daemon signs
`updateEndpoint(wss://...)` → `EndpointUpdated` → indexer → Firestore →
client resolves 0x → wss and dials. A service client is just another
dialer with a different protocol frame:

```
client → provider:  {service_request, spec_digest, request_id, args, voucher}
provider → client:  {service_result, request_id, result | error, receipt_sig}
```

The **sponsor/dependent inference pipe is the same channel** — sponsored
inference (work-AI, deferred Phase 8+) is a Service whose work item is
an inference call and whose ask is alignment-priced (possibly 0/
subsidized). Design the frame generically now so the sponsor pipe rides
it later without a new protocol: `spec_digest` names what's being
served; pricing policy is the spec's business.

Daemon-side components:
- `ServiceStore` (mirror of ToolStore): specs authored here, ask
  management, publish = register on-chain + blob push.
- Service-host handler on the existing WS server: authenticates the
  channel, validates vouchers, dispatches to the backing implementation
  (which is just a local tool run on the provider's daemon — a Service
  is a tool the OWNER chose to sell), emits signed receipts.
- Client side: an MCP connector (`service_client`) — so from the
  consuming agent's seat, a remote service is indistinguishable from
  any other tool. One probe, two economies, one interface.

### 4. Reviews

A review = the payer's signed verdict on a request_id (ok/score/note),
publishable as a gossip event and mirrored by the indexer. Only
addresses that actually paid for the item can review it (receipt-gated
— web2's fake-review problem solved by construction). Aggregates
(success rate, dispute rate, volume) are displayed, never enforced:
pricing self-regulates because buyers see history; no on-chain math
beyond settlement.

### 5. Marketplace surface

- Web app: new Services screen — browse (Firestore-backed storefront),
  spec inspection, purchase flow (wallet), review history. The
  framework is both marketplace and consumer; the human benefactor of
  each daemon is the true end-consumer on both sides.
- Agent-side: the inference probe returns tools + services merged,
  ranked; the agent decides by judgment and wallet. Granting an agent
  spend authority over non-ATN tokens is an OWNER decision (budget
  system extension — open design point).

## Open knobs (sims / user)

1. Escrow griefing rules (release timeout, dispute path) — hardhat
   game-outs.
2. Channel close/challenge windows.
3. Whether reviews live as chain events (costly, permanent) or gossip
   + indexer (cheap, replayable) — lean gossip.
4. Non-ATN token budgets for agents (owner-granted allowances).

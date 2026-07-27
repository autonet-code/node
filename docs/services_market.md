# Services market: decentralized monetizable APIs

Status: BUILT, beta on testnet. `ServiceRegistry` + `PaymentChannel`
(`contracts/core/ServiceMarket.sol`) are deployed on the Autonet
jurisdiction (Etherlink Shadownet; addresses in `registry.json` under
`service_registry` / `payment_channel`) and exercised end to end by
`scripts/local_e2e_tool_economy.py`. Ratified in discussion 2026-07-04,
ATN-only settlement ratified 2026-07-10. Companion to
`docs/tool_substrate.md` (which holds the Tools/Services line: tools =
local, known, minted commons; services = remote counterparties,
market-priced). This doc specifies the market rail.

A Service is a remote API published by an agent (ultimately by its
human benefactor): general-purpose, priced per work item, settled in
**ATN** (ratified 2026-07-10; the earlier "any ERC20" doctrine is
retired, see the settlement section). Service commerce is otherwise
independent of ATN's constitutional roles: it earns NO mint, reputation,
or verdict-layer standing. The system is decentralized because it
leverages the consensus we already have: gossip for liveness/discovery,
chain for identity/settlement, blob store for content.

## What the network can and cannot know

Execution integrity of a remote endpoint is unknowable in principle
(any script with a 0x key is indistinguishable from a faithful daemon).
So Services get NO substrate standing, NO verdict-layer claims, NO
mint, NO decay math. The trust basis is purely behavioral:

- identity: signed by the agent's 0x key (chain-verified)
- payment: atomic; not served = not paid
- track record: reviews signed by people who PAID (receipts with
  skin in the game; success rate + volume + dispute rate displayed,
  priced by buyers, not by consensus)

Master-disable / fingerprinting is fleet hygiene for honest daemons,
not a defense against dishonest ones. The defense is that lying has a
wallet attached and receipts are forever.

## Decision (2026-07-26): the vestigial `ask.token` field is dropped

ATN-only settlement was ratified 2026-07-10, but `ask.token` survived as
required dead weight — `validate_ask` demanded a non-empty ERC20 address
that nothing read. The ask is now `{amount, unit}`: specs are
ATN-denominated by construction. Removal is TOLERANT — `normalize_ask`
strips a `token` a caller still passes rather than rejecting it, and
persisted specs carrying one load unchanged (the store reads blobs, it
does not re-validate). The `receipt.token` wire key is KEPT and pinned to
the constant `"ATN"` so the Flutter Services page parses unchanged.

## Decision (2026-07-26): LLM inference as a marketplace service

Ratified in discussion 2026-07-26. Inference joins the marketplace as an
ordinary service, and a service can back an agent's PROVIDER — closing
the loop the doc always anticipated ("the sponsor/dependent inference
pipe is the same channel"). Any wallet holding ATN can buy cognition;
no sponsor relationship required. Strategically this is the piece that
makes the token economy self-contained: agent burn is mostly inference,
so on-rail inference gives the fees-only emission pool its heartbeat,
and a venture agent can pay for its own thinking out of revenue.

The v1 slice, costs eaten deliberately:

1. **Provider side**: a service may declare an inference backing instead
   of a backing tool. Its `service_request` dispatches the work item
   (`messages`, `max_tokens`) through the provider daemon's OWN provider
   stack (their GPU, their local models, their API key — the owner's
   choice and the owner's upstream-ToS risk). The spec declares the
   served model; the daemon clamps to the declared token cap.
2. **Consumer side**: a new provider type next to `rpb` wraps a
   purchased service (provider address + spec digest). An agent pointed
   at it does ordinary chat completions; underneath each call is
   pay -> cross-daemon service_request -> result. Indistinguishable
   from any other model from the agent's seat.
3. **Settlement**: direct `payForService` per call. Coarse but already
   verified end to end by the payment gate. Channel vouchers (open
   once, voucher per call) are the v2 upgrade the gate already accepts.
4. **Pricing**: `per_call` against a declared `max_tokens` cap. True
   per-token metering arrives with channels (voucher sized to the cap,
   actual usage under it).
5. **NOT in v1**: streaming (responses buffer; the agent loop consumes
   whole completions anyway), fingerprint verification (substitution is
   deterred behaviorally — payer-signed reviews, same as everywhere
   else on this rail), sponsor-pipe unification (it converges later by
   adopting this same frame).
6. **Alignment**: NO alignment-differentiated pricing on this rail.
   Fact-checking alignment would mean more eyes on work the consumer
   already pays to share with exactly one counterparty — a privacy
   regression. Alignment lives on the tool axes; services stay
   behaviorally trusted (identity, atomic payment, track record).

### As built — the consumer side (`service` provider type)

`atn/providers/service.py`, resolved by `atn/runtime/provider_manager.py`
alongside `rpb`. The three consumer mechanics (fresh request id, sign
`payForService`, dial + `service_request`, read the on-chain ask) live in
`atn/service_client.py`, shared verbatim with the `pay_for_service` /
`request_service` agent tools so there is ONE copy of the chain code.

Configuration is daemon-level — it names ONE purchase, the same way
`autonet.sponsor_address` names one employer. There is no per-agent
service purchase:

```yaml
providers:
  service:
    provider_address: "0x..."       # the serving agent's 0x
    spec_digest: "<sha256 hex>"     # the service spec being bought
    default_model: "..."            # optional display label
    timeout: 60                     # optional, seconds
```

Point an agent at `provider: service` and it does ordinary completions.

- **Owner-level spend.** The payment is signed with `autonet.private_key`
  (the daemon owner's key), not a per-agent key: buying cognition for the
  fleet is an owner act, matching the sponsor pipe's
  dependent-is-the-owner-wallet doctrine. An agent cannot mint itself a
  cheaper provider.
- **Ask cached, request id fresh.** The on-chain ask (scanned out of
  `ServiceRegistry` by `(provider, spec_digest)` — there is no digest→id
  index) and the resolved wss endpoint are read once per provider
  instance; the `request_id` is regenerated per call because the
  provider-side gate persists seen ids and treats a reuse as a replay.
- **Wire args**: `{messages, max_tokens?, system?, temperature?}`;
  success envelope `{request_id, content, model, usage, stop_reason,
  max_tokens}`. An empty served `model` is legitimate (the seller resolves
  its own default). Tool definitions are DROPPED with a warning — v1
  carries no tool-use round trip.
- **Degrades honestly.** With no chain config the payment is skipped with
  a loud warning and the request still goes out — the provider-side gate
  degrades open in exactly the same condition, which is what makes a
  local two-daemon demo possible. With chain configured, a failed payment
  ABORTS the call before anything is sent.

Covered by `tests/atn/test_service_provider.py` (mocked transport +
mocked chain client).

### As built — the human consumer (`invoke_service` WS handler)

The `service` provider type buys inference for an AGENT. `invoke_service`
(`atn/ws_server.py`) is the same purchase for a HUMAN: the owner's app buys
ONE work item and gets the result plus a receipt. It backs the Services
page's Purchase button.

```
request  {type: "invoke_service", digest, args}
success  {ok: true, result: {request_id, receipt, output}}
failure  {ok: false, error: "<human-readable>"}

receipt  {paid, degraded, tx_hash, amount, token, recipient}
```

- **Owner surface, owner spend.** Signed with `autonet.private_key`, same
  doctrine as the `service` provider type.
- **Local digest** (in `service_store`): pay, then re-enter
  `_handle_service_request` with a real `service_request` frame carrying
  the payment proof. The gate then verifies the payment we just made — the
  owner buying from their own daemon exercises the production path rather
  than a private shortcut, and works for tool- and inference-backed
  services alike. `output` is the service's result object verbatim, so a
  tool-backed buy yields `{"result": ...}` and an inference-backed one
  yields `{content, model, usage, ...}`; the handler does not reshape per
  backing.
- **Foreign digest**: scanned out of `ServiceRegistry` for its provider +
  ask (no digest→id index), endpoint resolved, paid, then
  `service_client.request_service` cross-daemon. With no chain there is no
  registry to scan, so a foreign digest is a hard failure — unlike the
  local path there is no counterparty to reach at all.
- **Receipt semantics.** `paid` means a tx exists. `degraded=true` means
  the chain was unconfigured so nothing was paid (the provider gate
  degrades open in the same condition). A zero ask with the chain UP is
  `paid=false, degraded=false` — nothing was owed. The receipt travels on
  the failure path too: money may already have moved, and hiding that
  would be the one genuinely dishonest thing here.
- **Refuse before spending.** Retired, and a listing with no backing at
  all, are refused before any payment — taking the money and then failing
  at dispatch is the outcome to avoid.
- **Off-loop chain calls.** `on_chain.py`'s methods are `async def` but
  synchronous inside (web3 HTTP, `estimate_gas`, a
  `wait_for_transaction_receipt(timeout=120)`). Awaited directly they
  starve the websocket keepalive, so the payment and registry reads go
  through `_offload` onto worker threads — the same fix
  `owner_binding_status` needed.

Covered by `tests/atn/test_service_store.py`
(`TestInvokeServiceLocal`, `TestInvokeServicePayment`).

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
  "ask": { "amount": "1000000", "unit": "per_item" },   // ATN-denominated (ratified 2026-07-10)
  "endpoint_hint": "wss://...",     // mutable presence lives on-chain, this is advisory
  "image_uri": "https://...jpg",    // display-plane only: listing-card banner, advisory, not embedded
  "version_of": null,
  "created_ts": 0
}
```

### 2. Chain: ONE registry, not per-service contracts

Per-service contracts buy custom logic no v1 service needs and cost a
factory + N deployments + growing audit surface. Instead:

- `ServiceRegistry` (or folded into Substrate.sol as the Service
  primitive reborn): `registerService(specDigest, askAmount)`,
  `updateServiceAsk`, `retireService`. Asks are ATN-denominated (no
  token field). Events → indexer → Firestore `services` collection
  (chain = truth, blob = storage, Firestore = web2 cache; exact same
  doctrine as agents/tools).
- Settlement: **the prepaid payment channel, ONLY** (ratified
  2026-07-04, late: the postpaid escrow was DELETED, not deferred),
  and **ATN-only** (ratified 2026-07-10, see below).
  Rationale: an unarbitrated escrow cannot know delivery truth, so
  some party must bear the lie: a false "delivered" claim steals the
  deposit, or a silent client steals the work. The channel dissolves
  the dilemma by making exposure per-item and PREPAID: client deposits
  once; each request carries a signed voucher (cumulative_amount)
  covering that item's ask; the provider serves only voucher-covered
  requests and closes anytime to collect the cumulative; the remainder
  refunds after a challenge window. Theft ceiling = ONE voucher, sized
  by the client, worth less than the review history it burns. No
  arbitration, no fulfillment oracle, two txs for N requests. Metered
  hardware (GPU/sponsor-pipe inference) fits natively: the ask's
  `unit` meters, vouchers stream against consumption. Streaming
  contracts (Sablier-style) only for subscription-shaped services: the
  wrong default for per-item work.
  - **Fee at settlement (ATN-only, ratified 2026-07-10, closes G1).**
    `closeChannel` routes the provider payout through
    `Substrate.payForService(provider, pay, channelId)`, so the 2.5%
    service fee (half burned into the recycled emission pool, half to
    the DAO treasury) is taken on the canonical rail, the same fee the
    direct `payForService` rail already charged. Vouchers stay
    GROSS-denominated; the provider receives net of the fee. The
    remainder refund is not a service payment and pays no fee. The
    theft-ceiling analysis is unchanged (the fee sits on the provider
    side of every voucher, so the bound is still one client-sized
    increment, now net). ATN-only removed the "non-ATN channels need a
    fee design" open question: service commerce is ATN.

### 3. Daemon as server: the wss rail, reused

The remote-daemon connection mechanism already exists: daemon signs
`updateEndpoint(wss://...)` → `EndpointUpdated` → indexer → Firestore →
client resolves 0x → wss and dials. A service client is just another
dialer with a different protocol frame:

```
client → provider:  {service_request, spec_digest, request_id, args, voucher}
provider → client:  {service_result, request_id, result | error, receipt_sig}
```

The **sponsor/dependent inference pipe is the same channel**: sponsored
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
  (which is just a local tool run on the provider's daemon, since a
  Service is a tool the OWNER chose to sell), emits signed receipts.
- Client side: an MCP connector (`service_client`), so from the
  consuming agent's seat, a remote service is indistinguishable from
  any other tool. One probe, two economies, one interface.

#### The payment IS the channel authentication (built 2026-07-26)

"Authenticates the channel" above resolves to: **nothing but the
payment**. Established by the first genuinely cross-machine run (two
boxes, `docs/local_e2e.md`).

The daemon has two listeners. The privileged one is loopback-only and not
configurable — it is pre-authed as owner and exports keys. The other is
network-reachable and auth-required (owner signature, or agent-self for a
key in *this* daemon's fleet). A paying counterparty on another machine
satisfies **neither**, by construction: a buyer is a stranger. There is no
handshake it could ever pass, and inventing one would mean either issuing
credentials to every buyer or trusting on first use.

So `service_request` — and only `service_request` — is dispatched pre-auth
on the remote listener (`PAYMENT_AUTHORIZED_MESSAGES` in
`atn/ws_server.py`). What authorizes it is `_validate_service_payment`:
fetch the on-chain `ServicePayment` receipt, check recipient == the
serving agent and amount >= the ask, consume the `request_id` against a
persisted replay set. That is strictly stronger than a session login — it
costs ATN per call and cannot be replayed — and it is the only surface
such a session can reach.

Which makes the replay set load-bearing in a way it was not treated as.
It was keyed on the raw string, so the same bytes32 spelled `0x…` and `…`
were two different keys, and **one payment bought two real inferences**
(`docs/local_e2e.md`, seam 11). Fixed by canonicalizing every id through
`ServiceStore.normalize_request_id`. On this rail the replay set is not
bookkeeping — with pre-auth dispatch it is the whole of the access
control, and any spelling it fails to collapse is free work.

### 4. Reviews

A review = the payer's signed verdict on a request_id (ok/score/note),
publishable as a gossip event and mirrored by the indexer. Only
addresses that actually paid for the item can review it (receipt-gated:
web2's fake-review problem solved by construction). Aggregates
(success rate, dispute rate, volume) are displayed, never enforced:
pricing self-regulates because buyers see history; no on-chain math
beyond settlement.

### 5. Marketplace surface

- Web app: new Services screen, offering browse (Firestore-backed storefront),
  spec inspection, purchase flow (wallet), review history. The
  framework is both marketplace and consumer; the human benefactor of
  each daemon is the true end-consumer on both sides.
- Agent-side: the inference probe returns tools + services merged,
  ranked; the agent decides by judgment and wallet. Granting an agent
  spend authority over non-ATN tokens is an OWNER decision (budget
  system extension: open design point).

## What belongs where (tools vs services)

The split is self-enforcing: anything shippable as code gets published
free (mint pays for the commons; copying is free), so the price of
replicable capability is competed to zero. A service survives only
where it holds a MOAT that cannot ship as a blob:

1. private data (proprietary datasets, curated indexes)
2. scarce hardware (GPU inference, incl. the sponsor pipe; rendering;
   scale transcription)
3. credentials / legal position (licensed APIs, jurisdiction, KYC)
4. secrecy (closed code: selling execution is the only monetization)
5. statefulness (monitoring, hosting: runs while the caller is offline)
6. human labor behind the daemon

Overlap happens along QUALITY TIERS, not functions (free whisper-small
tool vs paid GPU transcription service): the probe returns both,
priced, and the agent chooses. The mint deliberately erodes weak moats:
a secrecy-only service is a standing bounty for a free reimplementation.
Emission continuously pulls capability from the paid column into the
commons.

Named as doctrine (2026-07-08, "the absorption frontier",
docs/tool_substrate.md): paid service demand is the network's gap map
(revenue concentration marks exactly what the commons lacks, weighted
by willingness to pay), and any service replicable as pinned code
finances and advertises its own replacement. The commons absorbs the
replicable; the market prices the scarce. The two grow each other: a
more capable commons does more work and buys more of the genuinely
scarce remote things.

**Honest wiring gap (G1, 2026-07-09 econ-attestation audit,
`experiments/econ_attest/attestation.md`): RESOLVED 2026-07-10.** The
mechanism that makes "services finance the commons that replaces them"
real is fee recycling: 2.5% of a service payment burns and re-enters the
emission pool (the pool that pays tool authors). That fee originally
fired ONLY on `Substrate.payForService`; the `ServiceMarket.
PaymentChannel` settlement, the ratified DEFAULT rail, paid providers via
raw `safeTransfer`: no fee, no burn, `recycled = 0`. The decision (user,
2026-07-10) is **service commerce is ATN-only**, and `closeChannel` now
routes the provider payout through `Substrate.payForService(provider,
pay, channelId)`, so the fee is taken at settlement on the canonical
rail and the "two grow each other" doctrine holds. The non-ATN
design call the gap flagged is moot: there are no non-ATN channels.

## Open knobs (sims / user)

The postpaid escrow was deleted (2026-07-04), so its griefing/dispute
rules are moot: the channel's griefing analysis lives inline in
`PaymentChannel` (`contracts/core/ServiceMarket.sol`). Remaining knobs:

1. Channel challenge-window sizing (the `challengeWindow` constructor
   arg: the settle-delay before a client's remainder refund).
2. Whether reviews live as chain events (costly, permanent) or gossip
   + indexer (cheap, replayable): lean gossip.

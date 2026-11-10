# Services market: decentralized monetizable APIs

Status: BUILT, beta on testnet, with two pieces still only specified.
BUILT: the on-chain registry (register / update ask / retire), direct
`payForService` settlement, the provider-side payment gate, and the
`service` consumer provider type. SPECIFIED, NOT BUILT: receipt-gated
reviews (§4). The voucher-per-call client flow is BUILT for agents'
`request_service` (2026-10-28, #45: `atn/service_channels.py`, one channel
per payer and provider; a voucher counts once the provider's signed reply
says it kept the payment, or when the outcome is unknown; refunds swept
after the window; the provider closes to collect, Decision 2026-11-10);
the `service` inference provider still pays each call up front.
`ServiceRegistry` + `PaymentChannel`
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

## Decision (2026-11-10): not served = not paid, signed replies, providers collect

An audit of the whole rail (2026-11-10) found the payment and delivery
paths broke the principles above in ways each piece hid. Fixed:

- **Not served = not paid, on both sides.** The provider's gate reserves a
  channel voucher (a copy sent meanwhile is refused, and its request id is
  used up) and keeps it only when the dispatch succeeds; a failure or a
  crash releases it. Everything a request can be refused for without its
  payment (unknown or retired service, no backing, malformed or oversized
  inference input, an upstream provider the daemon cannot build) is
  checked before the gate, so such a refusal leaves the payment unspent.
  Every reply says `charged` (whether this request's payment was kept).
  The buyer writes its voucher down as pending before sending, counts it
  when the provider's reply says `charged: true` or when no answer from
  the provider came back after the request may have reached it, and drops
  it on `charged: false`. The buyer's book is never behind the provider's,
  and ahead by at most one voucher when the outcome is unknown. A direct
  `payForService` payment is on chain already, so it is charged whatever
  the dispatch did.
- **Replies are signed.** The provider signs every reply about its
  services (request id, spec digest, and the reply's `ok`, `result`,
  `error` and `charged`) with the key that registered the service, the
  address the buyer paid (`service_client.sign_reply`, `receipt_sig`).
  The buyer takes nothing else as served: an unsigned or wrongly signed
  reply is not ok, its `charged` is not believed, and a relayed one moves
  on to the next relay. Only relays (connected seeds and peers advertising
  the relay role) forward requests, since a request's arguments are plain
  to whoever forwards them; a peer's advertised capability is stored
  under the sending peer's id, so no peer can claim or strip another's
  relay role.
- **Providers collect.** The provider's daemon closes a channel with its
  best voucher when the remaining deposit is below the on-chain price,
  when the channel has been idle for `autonet.channel_idle_close_s`
  (default 24 h), or during a close the client started; never while a
  request on it is being served. The buyer notices a channel the provider
  closed, takes the remainder back after the window and opens a new one.
  A ServiceMarket write whose gas estimate reverts is not sent; a client
  close past its window is dropped; a failed `startClose` is sent again.
- **The household buys as its operator** everywhere: agents, the owner's
  `invoke_service` in the app, and the daemon-level `providers.service`
  purchase (the configured daemon key, else the vault-held operator key).
- **Inference names its upstream.** The `inference` block is `{provider,
  model, max_tokens_cap, max_input_chars}`: provider and model are
  required, serving uses exactly that provider (never a guess from the
  model name), and `max_input_chars` (default 64,000, the messages as
  compact JSON plus the system prompt) bounds the input the per-call price
  covers.
- **Used-up request ids** are kept for 30 days (`REPLAY_HORIZON_S`),
  appended one line each to `served_requests.jsonl` and compacted when
  stale lines pile up; the gate refuses a direct payment older than that
  (the `ServicePayment` block time), so a forgotten id can never be
  replayed.
- **The request log** names only the payer the gate verified, counts
  refused attempts without attributing them, keeps its summary as rows
  are written (`requests_summary.json`) and rotates at 4 MiB.
- `request_service` takes the numeric `service_id` from `find_services`
  (digest and provider read from the chain), or `spec_digest` with
  `provider_address`. Sanctions read an unbound address's owner again
  after a minute. libp2p service requests have their own bounded thread
  pool (16).
- Proven by `scripts/local_e2e_service_provider.py` (a failed dispatch is
  not charged, the provider's own close, the buyer's refund) and
  `scripts/local_e2e_cross_daemon_inference.py` (a paid turn accepted
  through the provider's signed reply).

## Decision (2026-11-10): services are published and found the way tools are

Every daemon knows the jurisdiction's services the way it knows its tools,
and serves them to the app and its agents itself; nothing reads Firestore
for them (the indexer's `services` mirror is unused).

- **Gossip.** A spec has no code, so it travels whole: the provider's
  daemon announces each registered spec on
  `/autonet/services/<jurisdiction>` (`nodes/common/event_gossip.py`
  `topic_for_services`) as soon as the chain numbers it, and again every
  30 minutes. The topic is not consensus (services earn no mint, so
  nothing enters the epoch-close). Relays subscribe to it, so specs cross
  daemons that reach each other only through a relay.
- **The chain admits.** Each daemon's catalog (`atn/service_catalog.py`)
  follows the `ServiceRegistry`: one enumeration on the first sync, then
  every minute the entries numbered past the last known and the
  `ServiceAskUpdated` / `ServiceRetired` logs in 1000-block chunks (every
  known entry is read again when an RPC refuses logs). A gossiped spec is
  kept only when an entry names its digest, so spam costs gas and a
  registered address; accepting one costs no chain read.
- **Genuine entries.** Anyone can register a copy of a public spec under
  another address and price. An entry counts only when its provider is
  the spec's signer: specs are signed with, and name as `author_pubkey`,
  the key that registers them (the operator's for the owner surface and
  external agents).
- **Missed specs.** An active entry whose spec no announcement brought is
  fetched by digest from any daemon holding it, retried with a doubling
  backoff (60 s up to an hour). The tool manifests' fetch got the same
  retry, plus a re-queue from the anchored registrations.
- **Search.** `find_services` (agents) and `market_services` (app) rank
  the catalog with the tools' search (BM25 over name, description and
  input-schema property names, fused with the local model), instead of
  reading every registry entry and fetching every spec per search.
  `lookup_service_ask` and the owner's purchase find the entry through the
  catalog and read it fresh from the chain.

## Decision (2026-11-10): the chain is the only reference for price, status and payee

A service's price, whether it is active and the address paid are read
from its `ServiceRegistry` entry and nowhere else. The spec carries none
of them: `validate_service_spec` refuses a spec with an `ask` field, and
the price is set at `registerService` and changed in place by
`updateServiceAsk` (any Agent with the selling grant can send it through
`send_transaction`; the digest does not change). Before this, the
provider's gate charged the price written in the spec while buyers read
the chain's, so an on-chain price change split the two, and
`ServiceStore.update_ask` (a new spec version per price) is removed.

- The daemon keeps the number each registration returned
  (`ServiceRecord.service_id`, persisted) and reads that entry
  (`WebSocketBridge._chain_service`). A record without one is matched by
  digest only among the entries of the address that signs for its
  author: anyone can register a copy of a public spec, at price 0.
- The gate refuses a request when the entry cannot be read or is
  inactive, charges the entry's `askAmount` and verifies payment to the
  entry's `provider`. The spec's `author_pubkey` only names the signer of
  the spec.
- The owner surface (author `user`) registers, retires and is paid under
  the operator's address: `_service_signing_key("user")` is the operator
  key. It used the configured daemon key, empty in a normal install, so
  an owner-published service never reached the chain.
- The buyer's `service` provider scans the registry once for the entry's
  number and reads the price from that entry on every call.
- Proven by `scripts/local_e2e_service_provider.py` stage 8: after
  `updateServiceAsk`, the next call pays the new price and the gate
  verifies it, with no new spec.

## Decision (2026-07-26): the vestigial `ask.token` field is dropped

(Superseded 2026-11-10: the spec carries no price at all.)

ATN-only settlement was ratified 2026-07-10, but `ask.token` survived as
required dead weight: `validate_ask` demanded a non-empty ERC20 address
that nothing read. The ask is now `{amount, unit}`: specs are
ATN-denominated by construction. Removal is TOLERANT: `normalize_ask`
strips a `token` a caller still passes rather than rejecting it, and
persisted specs carrying one load unchanged (the store reads blobs, it
does not re-validate). The `receipt.token` wire key is KEPT and pinned to
the constant `"ATN"` so the Flutter Services page parses unchanged.

## Decision (2026-07-26): LLM inference as a marketplace service

Ratified in discussion 2026-07-26. Inference joins the marketplace as an
ordinary service, and a service can back an agent's PROVIDER, closing
the loop the doc always anticipated ("the sponsor/dependent inference
pipe is the same channel"). Any wallet holding ATN can buy cognition;
no sponsor relationship required. Strategically this is the piece that
makes the token economy self-contained: agent burn is mostly inference,
so on-rail inference gives the fees-only emission pool its heartbeat,
and an agent can pay for its own thinking out of revenue.

The v1 slice, costs eaten deliberately:

1. **Provider side**: a service may declare an inference backing instead
   of a backing tool. Its `service_request` dispatches the work item
   (`messages`, `max_tokens`) through the provider daemon's OWN provider
   stack (their GPU, their local models, their API key: the owner's
   choice and the owner's upstream-ToS risk). The spec declares the
   upstream provider and the served model (both required since
   2026-11-10, never guessed from the model name); the daemon clamps to
   the declared token cap and refuses input above `max_input_chars`.
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
   deterred behaviorally by payer-signed reviews, same as everywhere
   else on this rail), sponsor-pipe unification (it converges later by
   adopting this same frame).
6. **Alignment**: NO alignment-differentiated pricing on this rail.
   Fact-checking alignment would mean more eyes on work the consumer
   already pays to share with exactly one counterparty, a privacy
   regression. Alignment lives on the tool axes; services stay
   behaviorally trusted (identity, atomic payment, track record).

### As built: the consumer side (`service` provider type)

`atn/providers/service.py`, resolved by `atn/runtime/provider_manager.py`
alongside `rpb`. The three consumer mechanics (fresh request id, sign
`payForService`, dial + `service_request`, read the on-chain ask) live in
`atn/service_client.py`, shared verbatim with the `pay_for_service` /
`request_service` agent tools so there is ONE copy of the chain code.

Configuration is daemon-level: it names ONE purchase, the same way
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

- **Operator spend.** The payment is signed with the operator key (the
  configured daemon key, else the vault-held operator key): the household
  buys as its operator (Decision 2026-11-08), not with a per-agent key.
  An agent cannot mint itself a cheaper provider.
- **Price read per call, request id fresh.** The service's
  `ServiceRegistry` number is scanned out once by `(provider, spec_digest)`
  (there is no digest→id index); the price is read from that entry on
  every call (Decision 2026-11-10), and the wss endpoint is cached. The
  `request_id` is regenerated per call because the provider-side gate
  persists seen ids and treats a reuse as a replay.
- **Wire args**: `{messages, max_tokens?, system?, temperature?}`;
  success envelope `{request_id, content, model, usage, stop_reason,
  max_tokens}`. An empty served `model` is legitimate (the seller resolves
  its own default). Tool definitions are DROPPED with a warning: v1
  carries no tool-use round trip.
- **Degrades honestly.** With no chain config the payment is skipped with
  a loud warning and the request still goes out. The provider-side gate
  degrades open in the same condition, but ONLY for a request that arrived
  on the LOCAL listener, which is what makes a local two-daemon demo
  possible. A REMOTE request with no chain config is refused: that
  listener dispatches `service_request` pre-auth precisely because the
  payment is the credential, so degrading open there would hand any peer
  who knows a digest unlimited free work (inference off the owner's
  provider stack included) on nothing worse than an RPC outage. With
  chain configured, a failed payment ABORTS the call before anything is
  sent.

Covered by `tests/atn/test_service_provider.py` (mocked transport +
mocked chain client).

### As built: the human consumer (`invoke_service` WS handler)

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

- **Owner surface, operator spend.** Signed with the operator key, same
  as the `service` provider type: the household buys as its operator.
- **Local digest** (in `service_store`): pay, then re-enter
  `_handle_service_request` with a real `service_request` frame carrying
  the payment proof. The gate then verifies the payment we just made: the
  owner buying from their own daemon exercises the production path rather
  than a private shortcut, and works for tool- and inference-backed
  services alike. `output` is the service's result object verbatim, so a
  tool-backed buy yields `{"result": ...}` and an inference-backed one
  yields `{content, model, usage, ...}`; the handler does not reshape per
  backing.
- **Foreign digest**: the catalog names its genuine registry entry (the
  spec's signer's), read fresh for provider and price, endpoint resolved,
  paid, then `service_client.request_service` cross-daemon. With no chain
  there is no registry, so a foreign digest is a hard failure. Unlike the
  local path there is no counterparty to reach at all.
- **Receipt semantics.** `paid` means a tx exists. `degraded=true` means
  the chain was unconfigured so nothing was paid (the provider gate
  degrades open in the same condition). A zero ask with the chain UP is
  `paid=false, degraded=false`, because nothing was owed. The receipt travels on
  the failure path too: money may already have moved, and hiding that
  would be the one genuinely dishonest thing here.
- **Refuse before spending.** Retired, and a listing with no backing at
  all, are refused before any payment: taking the money and then failing
  at dispatch is the outcome to avoid.
- **Off-loop chain calls.** `on_chain.py`'s methods are `async def` but
  synchronous inside (web3 HTTP, `estimate_gas`, a
  `wait_for_transaction_receipt(timeout=120)`). `OnChainService` and
  `ServiceMarketClient` are wrapped by `_off_the_loop` (#49), so every
  async method runs on a worker thread and no caller starves the
  websocket keepalive.

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
  "author_pubkey": "0x...",         // the spec's signer; the payee is the registry entry's provider
  "author_sig": "...",              // over canonical bytes, sig excluded
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
  token field). Every daemon follows these entries in its own catalog
  and gets the specs over gossip (Decision 2026-11-10, above); the
  indexer's Firestore `services` mirror is not read.
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
- `ServiceStore` (mirror of ToolStore): specs authored here, the
  on-chain service number kept, publish = register on-chain + blob push.
  The price lives on chain only.
- Service-host handler on the existing WS server: authenticates the
  channel, validates vouchers, dispatches to the backing implementation
  (which is just a local tool run on the provider's daemon, since a
  Service is a tool the OWNER chose to sell), emits signed receipts.
- Client side: an MCP connector (`service_client`), so from the
  consuming agent's seat, a remote service is indistinguishable from
  any other tool. One probe, two economies, one interface.

#### How requests travel: libp2p first, wss second (2026-10-28, #36)

A provider on a cloud VM or a home connection usually has no public
address, so it cannot publish a dialable wss endpoint. `request_service`
(`atn/service_client.py`) therefore tries libp2p first:

1. **libp2p** (`/autonet/service/1.0.0`), when this daemon's host is
   running. The provider is found by the peer id its agent bound on chain
   (`Substrate.getAgentPeerId`; stable across restarts via
   `<data_dir>/p2p.key`). Connected to that peer: one stream, direct.
   Otherwise the request goes to connected relays (connected seeds and
   peers advertising the relay role), one at a time, each asked to forward
   it one hop; a relay that is not connected to the provider says so at
   once and the next is tried, as is the next after a reply the buyer
   rejects as unsigned (Decision 2026-11-10). `nodes/relay.py` has no
   service handler of its own.
2. **wss endpoint**, only when libp2p could not deliver the request (no
   host, no peer id on chain, no path, or an older provider without the
   protocol). Once a request reached the provider, its reply, its error or
   its timeout is final: the request is never sent a second way.

Why libp2p first: it is the only path that reaches a NAT'd provider, and
the Noise handshake proves the far end holds the peer id bound on chain,
while a wss URL proves nothing about who answers. The wss rail stays for
providers that publish one and for buyers whose daemon has no host.

The payload is the same `service_request` frame the WS path sends
(payment proof, payer signature), and the provider hands it to the same
`_handle_service_request` with `local=False`, so the payment gate, the
payer-signature check and the replay guard are shared. The relay carries
bytes only; all trust stays end to end: the provider signs its reply, and
the buyer checks the signature against the address it paid. Default wait: 300 s
(`DEFAULT_REQUEST_TIMEOUT`, per-call `timeout`); a relay waits at most
`--service-timeout` (default 600 s, `AUTONET_RELAY_SERVICE_TIMEOUT`) or the
buyer's timeout if shorter.

#### The payment IS the channel authentication (built 2026-07-26)

"Authenticates the channel" above resolves to: **nothing but the
payment**. Established by the first genuinely cross-machine run (two
boxes, `docs/local_e2e.md`).

The daemon has two listeners. The privileged one is loopback-only and not
configurable, and it is pre-authed as owner and exports keys. The other is
network-reachable and auth-required (owner signature, or agent-self for a
key in *this* daemon's fleet). A paying counterparty on another machine
satisfies **neither**, by construction: a buyer is a stranger. There is no
handshake it could ever pass, and inventing one would mean either issuing
credentials to every buyer or trusting on first use.

So `service_request`, and only `service_request`, is dispatched pre-auth
on the remote listener (`PAYMENT_AUTHORIZED_MESSAGES` in
`atn/ws_server.py`). What authorizes it is `_validate_service_payment`:
fetch the on-chain `ServicePayment` receipt, check recipient == the
service's on-chain provider and amount >= its on-chain ask, consume the
`request_id` against a
persisted replay set. That is strictly stronger than a session login (it
costs ATN per call and cannot be replayed), and it is the only surface
such a session can reach.

Which makes the replay set load-bearing in a way it was not treated as.
It was keyed on the raw string, so the same bytes32 spelled `0x…` and `…`
were two different keys, and **one payment bought two real inferences**
(`docs/local_e2e.md`, seam 11). Fixed by canonicalizing every id through
`ServiceStore.normalize_request_id`. On this rail the replay set is not
bookkeeping: with pre-auth dispatch it is the whole of the access
control, and any spelling it fails to collapse is free work.

The receipt alone also did not say who may redeem it: a `request_id` read
off the chain (the `ServicePayment` event is public) could be sent by
anyone before the payer, who then got "replay". Since 2026-10-26 a paid
request carries `payer_signature`: the paying key's signature over
keccak(`autonet:service_request:v1`, request id, spec digest, sha256 of
the args' canonical JSON, provider address)
(`service_client.sign_request`). The gate recovers the signer and
requires it to be the event's payer, so a copied id is worthless and the
args are bound to the payment. Unsigned paid requests are refused. A
failed signature check does not consume the id.

### 4. Reviews (SPECIFIED, NOT BUILT)

Nothing below this paragraph is implemented. There is no review event,
no review store and no review surface in the app: the strings
`review` in `atn/service_store.py` are forward-looking comments on the
request log a review rail would later ride.

What ships instead as the trust surface, and is what the Services page
actually renders:

- the on-chain provider record (`serviceEarnings`, `agentMintTotal`,
  registration tenure), read by the `provider_record` WS handler. It is
  a consequence of payments, so it is costly to inflate: a provider can
  pay itself, at the cost of the fee each time.
- the provider-local success rate and volume from
  `ServiceStore.summary()`, returned alongside `list_services`: requests
  that passed the payment gate, attributed only to the payer it verified.

The specified design, for when it is built: a review = the payer's
signed verdict on a request_id (ok/score/note), publishable as a gossip
event and mirrored by the indexer. Only addresses that actually paid for
the item can review it (receipt-gated: web2's fake-review problem solved
by construction). Aggregates (success rate, dispute rate, volume) are
displayed, never enforced: pricing self-regulates because buyers see
history; no on-chain math beyond settlement.

### 5. Marketplace surface

- Web app: new Services screen, offering browse (the daemon's catalog,
  `market_services`),
  spec inspection, purchase flow (wallet), and (once §4 is built) review
  history; today the trust strip shows the provider record plus the
  provider-local success rate. The
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

1. Channel challenge-window sizing: the settle-delay before a client's
   remainder refund. Governed since 2026-10-16: the constructor sets the
   initial value and the DAO timelock (the Substrate's governor) changes
   it with `setChallengeWindow`. Each channel records the window it
   opened with, so a change never moves an open channel's deadline.
2. Whether reviews live as chain events (costly, permanent) or gossip
   + indexer (cheap, replayable): lean gossip.

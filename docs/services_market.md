# Services market: decentralized monetizable APIs

Status: BUILT on the `services-redesign` branch (Decision 2026-11-12
below, which supersedes the parts of the earlier decisions it names).
`ServiceRegistry` + `PaymentChannel` (`contracts/core/ServiceMarket.sol`)
are the chain half; the daemon half is `atn/service_spec.py`,
`atn/service_prices.py`, `atn/service_store.py`, `atn/service_publish.py`,
`atn/service_catalog.py`, `atn/service_channels.py`,
`atn/service_sessions.py` with `nodes/common/service_session.py`,
`atn/service_runner.py`, `atn/channel_settlement.py` and the reviews in
`nodes/common/service_reviews.py`. Companion to `docs/tool_substrate.md`
(tools = local, known, minted commons; services = remote counterparties,
market-priced).

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
- track record: reviews by people who PAID the provider on chain (skin in
  the game; displayed, priced by buyers, not by consensus)

Master-disable / fingerprinting is fleet hygiene for honest daemons,
not a defense against dishonest ones. The defense is that lying has a
wallet attached and payments are on chain for good.

## Decision (2026-11-12): a service is a tool sold by request or by session

Ratified with the owner in discussion 2026-11-12, over a mockup of the
registration form. Supersedes:
the inference backing (Decision 2026-07-26), the unit of work as free
text and the on-chain price (Decisions 2026-11-10 "the unit of work" and
"the chain is the only reference for price"), the receipts as the proof
behind reviews (Decision 2026-11-10 "service reviews"), and "stream" as
the name of what is now a session.

1. **Every service is backed by a tool** registered on the provider's
   daemon. It can stay unpublished, so its code never leaves the machine.
   The service spec is a separate document from the tool's manifest: it
   says how the tool is sold, not what it is, so a changed listing never
   changes the tool's digest, and one tool can back several services. In
   a codebase the specs live in `autonet-services/<name>.json`, which the
   tool's snapshot always leaves out.
2. **Two ways to use a service** (`mode`). `request`: each request gets
   one reply; the tool runs once per request; priced per request.
   `session`: the buyer keeps a two-way connection open and speaks the
   protocol the tool serves; the tool's manifest declares it in a
   `session` block (the protocol's name, version and published
   specification pinned by sha256, a cap on sessions at once, one process
   shared by all sessions or one per session, and which token counts it
   reports). A session is priced per measure: seconds open, gigabytes
   transferred, input tokens, output tokens, each per a quantity the spec
   names. What a buyer sends is the tool's input schema (a request
   service) or the protocol's specification (a session service).
3. **Quality, not quantity, makes a different service.** A different model
   is another service, set apart by `settings` (public in the spec,
   passed to the tool as `AUTONET_SERVICE_SETTINGS`). How much a buyer
   gets is the voucher size, which the provider sets.
4. **Prices are off chain.** The provider signs a price list per service
   (`atn/service_prices.py`) and gossips it with the spec; search shows
   it. A new list replaces the old by sequence number: no transaction, no
   new spec. A session holds the prices quoted when it opened. The
   registry's `askAmount` stays as an optional price nothing enforces:
   buyers make up their own minds. Input and output tokens are priced
   separately because the mix varies too much between uses for one
   average price.
5. **The daemon serves without cognition.** Once a service is published
   its daemon runs the tool (`atn/service_runner.py`: the port in
   `AUTONET_SERVICE_PORT`, usage posted to `AUTONET_USAGE_URL`), restarts
   it when it dies, refuses buyers before payment when it is down or full,
   and serves anyone whose payment checks out. The service lives in the
   daemon (network, payments, cut-off); the tool only does the work. A
   tool priced in tokens reports each connection's counts after each
   reply, and the daemon cuts the connection when the credits do not cover
   them; at most one reply overshoots, which the provider prices in.
   Failures are recorded for later notifications to the Agent or owner;
   nothing notifies yet. Search shows no liveness: a refused buyer pays
   nothing and tries the next provider.
6. **Credits per provider.** A buyer's credits with a provider are one
   payment channel for all its services, like API credits. The Agent
   opens them with a deposit and tops them up itself (`open_credits`,
   `top_up_credits`); nothing does either automatically. Vouchers pay
   ahead by the provider's voucher size, so a buyer never has more than
   one voucher at risk; the provider serves while what it served stays
   within the running total, and a request it refuses or fails costs
   nothing. A one-off request to a provider the buyer holds no credits
   with is paid directly (one `payForService` transaction).
7. **The contract pays out without closing.** `PaymentChannel.settle`
   cashes in the running total and leaves the channel open; `topUp` adds
   to it. The provider's daemon settles automatically (a threshold, at
   least daily, and when the buyer starts closing, which keeps the buyer's
   refund deadline). Every payout emits `ChannelPaid`, so fees burn as the
   provider earns and the payment is on chain for reviews.
8. **Reviews need a payment to the provider, not a receipt.** A review
   names its service; it counts once the reviewer has paid that provider
   on chain (a direct payment, or a payout from its channel), whichever of
   the provider's services it used. Nothing proves the reviewer used that
   service rather than another of the provider's, but it costs a real
   payment to that provider either way. The signed receipts are gone;
   replies stay signed by the provider (`reply_sig`) so a relay cannot
   make one up.
9. **Removed:** the inference backing (an LLM service is a session to a
   tool speaking an OpenAI-compatible protocol, or a request service),
   "model provider", the per-reply token cap, the hardware field, the
   free-text unit, the endpoint hint, the `pay_for_service` and
   `close_service_channel` commands and the buyer-side quote check (a
   dishonest provider can overcount instead; the buyer notices what it
   pays against what it gets, and reviews).

## Decision (2026-11-10): service reviews in the epoch-close

(The receipts below are superseded by Decision 2026-11-12, item 8: a
review counts once its reviewer has paid the provider on chain.)

A buyer rates a service it paid for, and other buyers read the rating
before they buy. Decided by the owner; built in
`nodes/common/service_reviews.py` (consensus side) and
`atn/service_receipts.py` (the buyer's daemon).

- **Fields.** A service review has the fields of a tool review:
  correctness (-1..1, delivered as described), simplicity (0..1, read as
  the buyer's convenience: time, preparation, documents asked for),
  novelty (0..1, the reviewer's own judgment of how much of its frame the
  service makes it revise), malware (yes or no, the tools' NIST SP 800-53
  "malicious code" definition, which fits services: for example an
  inference model primed to make the buyer's agent leak data or run
  harmful commands), and the optional charter axes, scored as tools score
  them (an unscored axis counts at the Registry's `unscored_axis_bps`
  pinned to the snapshot, as for tools). Physical harm falls
  under correctness or the charter. A malware yes needs evidence: the
  reviewer's text plus the receipt and the provider's signed reply, kept
  as a content-addressed blob in the reviewer's tool-store blobs, which
  peers read with `fetch_evidence` like a tool flag's evidence.
- **Authority from payment, not REP.** Only the address that paid for a
  request may review it, a review counts only once that payment is on
  chain, every counted review weighs the same, and the provider cannot
  review itself. Reviews enforce nothing: no mint, no REP, no slashing,
  nothing on chain beyond what the anchor commits.
- **Receipts.** The provider's reply signature (`receipt_sig`) is a
  receipt: keccak of the ABI encoding of (`autonet:service_receipt:v1`,
  request id, spec digest, provider address, the payer the gate verified,
  `ok`, `charged`, the amount charged, the payment channel of a voucher
  payment or 0 for a direct one, sha256 of the canonical
  `{"error", "result"}` JSON), signed EIP-191. The reply carries `payer`,
  `amount` and `channel_id` next to `charged`. The result itself is not
  covered, so a
  review proves the receipt without revealing what the provider returned.
  The buyer's verification (`reply_from_provider`) checks the same hash,
  and `request_service` hands every authentic reply's receipt to the
  runtime's `ReceiptStore`, which keeps the receipt and the whole signed
  reply in a blob store no peer is served from.
- **The event.** `review_service` (agent tool; the app calls it as the
  owner) takes a request id (or a receipt), the four fields, optional
  charter scores, a note (up to 500 characters, public) and the malware
  evidence. It signs the review with the payer's key (an external agent's
  and the owner surface's is the operator's) and hands one
  `service_review` event to the library service, so it rides the open
  epoch's batch:

  ```
  {"kind": "service_review", "v": 1,
   "receipt": {request_id, spec_digest, provider, payer, ok, charged,
               amount, channel_id, result_hash, sig},
   "review": {jurisdiction, correctness, simplicity, novelty, malware,
              axes, evidence_hash, note, ts},
   "reviewer": "0x...", "sig": "0x..."}
  ```

  Scores are in hundredths as on ToolBond's wire. The reviewer signs
  keccak(`autonet:service_review:v1`, the receipt hash, sha256 of the
  review's canonical JSON).
- **A valid event.** Its receipt is well formed, says `charged: true`
  and is signed by the provider it names; the reviewer is the receipt's
  payer, signed the review, and is not the provider; the review names this
  jurisdiction (the batch's rpb address), has every field in range,
  evidence with a malware yes, and is dated at most an hour after its
  batch.
- **Proof on chain, pinned to the snapshot.** The close reads, like REP
  and the ToolBond rounds, from logs up to the snapshot block
  (`voice_state.read_service_snapshot`): the ServiceRegistry's
  `ServiceRegistered` pairs, Substrate's `ServicePayment` events, the
  PaymentChannel's `ChannelOpened` and `ChannelClosed`, and the Registry's
  `unscored_axis_bps`. A review of a (provider, digest) pair the registry
  never registered is dropped, new or carried; the entry's provider is the
  pair's provider and the receipt must be signed by it. A review counts
  only once its payment is on chain: for a direct payment, a
  `ServicePayment` with the receipt's request id from the payer to the
  provider whose amount covers the receipt's; for a voucher, the receipt's
  channel (client the payer, provider the provider) closed with a settled
  amount covering the amounts of the reviews drawing on it, earliest
  first. Until then it waits in the carried state (`paid: false`) and
  counts at the first close whose snapshot has the payment. So a
  self-rating through addresses the provider controls costs the fee and
  gas of a real payment per payer, like the provider record. Before the
  first anchor the snapshot is empty and no review counts.
- **Counting.** One review per (provider, spec digest, payer): a later
  review (larger signed `ts`, then larger review hash) replaces the
  earlier, so a replayed old review changes nothing. A service's rating is
  over its payers' latest paid reviews: the mean of each field, the share
  of payers flagging malware, each charter axis's mean and their mean
  together (`value`), the number of payers, the total their receipts
  charged, and how many reviews wait for their payment (`pending`). Payers
  of the provider's own household and sanctioned
  households (both from the pinned inputs) are kept but not counted.
  Ratings are keyed by (provider address, spec digest), so a copy
  registered under another address never shares a rating. Integer sums and
  sorted iteration make every daemon compute the same bytes. The carried
  state is bounded: 1000 reviews per service, paid ones kept first (so
  reviews that cost nothing cannot push out paid ones), of which at most
  100 wait for a payment, and 256 services per provider.
- **Anchored state.** Two carry maps, `service_reviews`
  (`<provider>:<digest>` -> payer -> latest review) and `service_ratings`
  (`<provider>:<digest>` -> rating), ride the carry bundle next to the tool
  maps, so a follower or a joiner gets them from the anchor's `carry_cid`.
  A data dir without them starts them empty.
- **Resending.** The buyer's daemon sends a review again after each close
  that lacks it, for 14 days (`ReceiptStore.resend_missing`): a review of
  a service registered after the snapshot, or one whose batch was lost,
  gets in once it can.
- **Search.** `ServiceCatalog.search` rows (`market_services`,
  `find_services`) carry `rating` for genuine entries, read from the close
  driver's `service_rating(provider, digest)` through the catalog's
  `rating_provider`. `service_receipts` (WS) lists the household's
  receipts and the reviews sent from them.
- **Same inputs everywhere.** The snapshot depends on the ServiceRegistry
  and PaymentChannel addresses, so every daemon of a jurisdiction must read
  the same ones (`blockchain.service_registry_address` and
  `payment_channel_address`, seeded like the other contracts).
- Proven by `tests/test_service_reviews.py`,
  `tests/atn/test_service_receipts.py` and
  `scripts/local_e2e_service_provider.py` stage 12 (the child reviews a
  channel-paid request; a close pinned before the seller's channel close
  holds it waiting, one pinned after counts it, its carry bundle rates the
  service and the buyer's search shows it).

## Decision (2026-11-10): a stream between buyer and seller

(Renamed a session and re-based on credits by Decision 2026-11-12: the
tool behind the service names the protocol, and the daemon runs it.)

A service call was one request and one reply. Services also need a two-way
stream between the buyer's and the seller's daemons: inference streaming,
a video call, anything. Decided by the owner:

1. **The application is the parties' business.** The spec names the
   application protocol, `protocol: {id, description}` (`id` a name and a
   version number, "openai-chat/1", "webrtc-sdp/1"; validated in
   `atn/service_spec.py`). The network gives the two daemons a stream and
   nothing more.
2. **Direct first, then hole punching, a relay last.** The stream rides
   the mesh (`docs/p2p_networking.md`): a connection already open, else the
   provider's direct addresses, else its relay circuit (circuit relay v2),
   which DCUtR then upgrades to a direct, hole-punched TCP connection where
   both NATs allow it (symmetric NATs, common on mobile carriers, do not).
   Relays are not paid and relayed connections are unlimited by default,
   like torrent seeding, unless the relay's operator caps them (`atn relay
   --circuit-max-duration`, `--circuit-max-bytes`, `--max-circuits`).
3. **Providers are encouraged to be reachable.** `list_services` returns
   `reachability` (`reachable`, the addresses, advice when buyers reach the
   daemon only through a relay), and catalog rows carry `reachable` per
   provider (its libp2p capability, or a relay's account of it; null until
   known).
4. **Vouchers travel alongside the stream**, and the provider serves only
   while they cover the usage.

### As built

- **Handshake** (`/autonet/service-stream/1.0.0`,
  `nodes/common/service_stream.py`). The buyer's first frame is the
  `service_request` frame of the request/reply rail with `"type":
  "service_stream"`: spec digest, request id, args, a channel voucher and
  `payer_signature`. The provider runs the same gate
  (`WebSocketBridge._validate_service_payment`: on-chain price, payee and
  active flag, the voucher, the payer's signature, the replay guard), after
  the checks that need no payment (known, not retired, a stream service,
  a backend), and answers with one reply signed like a service reply
  (`receipt_sig` over `ok`, `result`, `error`, `charged`). `result.units`
  is what the opening voucher pays for: its increment over the channel
  divided by the on-chain price (or the gate's own count when it gives
  one). A buyer takes a reply only when it is signed by the address it
  pays. Streams are paid with channel vouchers; a `tx_hash` opening is
  refused before the gate, so it costs nothing.
- **Vouchers on the stream.** The provider meters what it serves
  (`ServiceStream.consume(units)`). When coverage runs low it sends
  `{"type": "voucher_needed", "units": n}`; the buyer answers `{"type":
  "voucher", ...}`: a payment proof under a fresh request id whose `args`
  are `{"stream": <opening request id>, "units": n}`, signed by the payer.
  The provider checks it with the same gate and requires the opening's
  payer. Usage no voucher covers within 30 s closes the stream; a buyer
  that will pay no more says `{"type": "no_voucher"}` and still gets what
  it paid for.
- **Not served = not paid.** A voucher is reserved when accepted and kept
  (`ServiceStore.keep_channel_voucher`, logged with the verified payer)
  only once usage enters the units it covers; the others are released when
  the stream ends, also on a cancellation. Each outcome goes back in a
  signed `{"type": "receipt", "charged": ...}`; the buyer's channel book
  counts a voucher on `charged: true` or when no receipt came, and drops it
  on `charged: false`. A closing buyer waits up to 5 s for the receipts.
- **The voucher interface** (for the metered channels): the network layer
  calls `voucher_source(units) -> fields | None` for the next voucher
  (`request_id`, `args`, `channel_id`, `cumulative_amount`, `signature`,
  `payer_signature`), `on_receipt(request_id, charged)` with each outcome,
  and `verify(reply, request_id)` on every reply and receipt. The daemon
  fills it from the household's channel (`service_channels.voucher_for` for
  ask x units), one voucher at a time per channel: a stream holds the
  channel's lock while one of its vouchers is outstanding.
- **The daemon's ends** (`atn/service_streams.py`). The provider pipes an
  accepted stream to the service's backend, a local address in
  `backing_tool` (`tcp://127.0.0.1:<port>`, loopback only), where the owner
  runs the application. The pipe meters by the spec's `unit`: a time unit
  (second, minute, hour) once per period while the stream is open, a size
  unit (byte, KB, MB, GB, KiB, MiB, GiB) per started unit carried either
  way, anything else once for the whole stream. The buyer's agent calls
  `open_service_stream` (agent API, services group) and gets a 127.0.0.1
  port for one local connection; the stream lasts while that connection
  is open.
- Proven between real libp2p hosts: a direct stream, a reachable provider
  dialled directly instead of through the relay, a stream through a relay
  on `tcp` and `ws` (also through `atn relay` itself), an unpaid or
  wrongly signed stream refused by the real gate, vouchers following
  metered usage through the gate with an unused one released on both
  sides (`tests/test_service_stream.py`, `tests/atn/test_service_streams.py`).
- **Not built.** Hole punching (above). The app does not yet show
  `reachable` or the reachability advice, nor open streams for the owner;
  the owner's `invoke_service` stays request/reply. Concurrent streams and
  request/reply calls to one provider share one channel and wait for each
  other's outstanding voucher.

## Decision (2026-11-10): the unit of work; metered on channels

(Superseded by Decision 2026-11-12, items 2 to 4 and 6: measures with
quantities, prices in a signed list, credits per provider.)

Decided by the owner.

- **Holding ATN is the household's credit** in the jurisdiction's service
  economy, and a payment channel is its prepaid balance with one provider.
  Buying avoids a transaction per request: a channel costs one deposit to
  open and one settlement to close, however many requests come between.
- **The spec declares the unit of work.** `unit` (required) is a text saying
  what one work item is: "one completion up to 4,000 output tokens", "one
  minute of RTX 5090 time", "one ice cream". It is the meaning of the price,
  not a price: the price per unit stays on chain only (`askAmount`, Decision
  "the chain is the only reference"). `hardware` (optional) names the
  hardware of a compute service, so a buyer can check its throughput
  against what that hardware documents. Both are signed with the spec (a
  new unit is a new spec), bounded to 200 characters each, embedded for the
  semantic half of search, and shown by `find_services`, `market_services`
  and `list_services`. `register_service` (agent tool and WS handler)
  refuses a registration without `unit`.
- **The buyer pays as usage accrues**: it signs the running total per unit
  consumed, so neither side risks more than one unit. For the request/reply
  services there are today, one request is one unit, and the voucher per
  request already does this. `service_channels.voucher_for(..., units=N)`
  signs N units in one step (amount N x price), for metering several units
  inside one long interaction; that needs the buyer-seller stream, built
  separately. The provider's gate still takes one price per voucher.
- **The `service` model provider pays on the channel**
  (`atn/providers/service.py`), the daemon-level purchase and a parent's
  per-agent binding alike, exactly as the `request_service` agent tool
  pays: one channel per payer and provider, opened with ten calls' worth,
  and a voucher for one unit per call, written down as pending before the
  request goes out, counted when the provider's signed reply says
  `charged: true` or when the outcome is unknown, and dropped on
  `charged: false`. A failed call is not paid. The daemon-level purchase
  pays as the operator. The provider shares the daemon's `ChannelBook` and
  its lock per channel (`service_channels.channel_lock`) with the agent
  tools, so two buys never sign the same running total. With the chain
  configured and no PaymentChannel, a call is refused before anything is
  sent; there is no fallback to a direct `payForService`.
- Proven by `scripts/local_e2e_service_provider.py` (the child's provider
  opens one channel and pays a voucher per call; a price changed on chain
  applies to the next voucher; a replayed request id and a voucher that
  adds nothing are refused; a failed call costs nothing through the
  provider or through the agent tool on the same channel; the seller
  collects once the deposit no longer covers the price; the buyer takes
  the remainder back) and `scripts/local_e2e_cross_daemon_inference.py` (retired 2026-11-12; a
  real agent turn paid with a voucher from the child's wallet, both books
  equal, the provider's kept voucher collected on chain net of the fee).

## Decision (2026-11-10): not served = not paid, signed replies, providers collect

(Receipts and the voucher per request are superseded by Decision
2026-11-12: vouchers pay ahead, a failed request's charge is taken back,
providers settle without closing.)

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
  services with the key that registered the service, the address the
  buyer paid (`service_client.sign_reply`, `receipt_sig`). The signed
  message is a receipt over the request, the payer, `ok`, `charged`, the
  amount and the result's hash (Decision "service reviews in the
  epoch-close" above).
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
  `scripts/local_e2e_cross_daemon_inference.py` (retired 2026-11-12; a paid turn accepted
  through the provider's signed reply).

## Decision (2026-11-10): services are published and found the way tools are

Every daemon knows the jurisdiction's services the way it knows its tools,
and serves them to the app and its agents itself; nothing reads Firestore
for them (the indexer's `services` mirror is unused).

- **Gossip.** A spec has no code, so it travels whole: the provider's
  daemon announces each registered spec on
  `/autonet/services/<jurisdiction>` (`nodes/common/event_gossip.py`
  `topic_for_services`) as soon as the chain numbers it, and again every
  30 minutes. The topic is not consensus (services earn no mint, so no
  spec enters the epoch-close; reviews do, on the epoch batches, see the
  Decision on service reviews). Relays subscribe to it, so specs cross
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

(The price is superseded by Decision 2026-11-12, item 4: the chain keeps
status and payee; prices are the provider's signed list, with an optional
unenforced registry price.)

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

(Superseded by Decision 2026-11-12, item 9: no inference backing; an LLM
service is a tool sold by session or by request.)

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
   (Superseded 2026-11-10: the `service` provider pays a voucher per call
   on the payment channel, Decision "the unit of work; metered on
   channels".)
4. **Pricing**: `per_call` against a declared `max_tokens` cap. True
   per-token metering arrives with channels (voucher sized to the cap,
   actual usage under it). (Since 2026-11-10 the spec's `unit` says what
   one call buys, e.g. "one completion up to 4,000 output tokens"; metering
   inside one interaction waits for the buyer-seller stream.)
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
alongside `rpb`. The consumer mechanics (fresh request id, dial +
`service_request`, read the on-chain ask) live in `atn/service_client.py`,
and the channel (book, voucher, lock) in `atn/service_channels.py`, both
shared verbatim with the `request_service` agent tool so there is ONE copy
of the chain code.

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

- **Operator spend.** The channel is opened and every voucher signed
  with the operator key (the configured daemon key, else the vault-held
  operator key): the household buys as its operator (Decision
  2026-11-08), not with a per-agent key. An agent cannot mint itself a
  cheaper provider.
- **Paid on the channel** (Decision 2026-11-10): a voucher for one unit
  per call on the payer's one channel to the provider, counted or dropped
  by the provider's signed `charged`; a call the provider refuses or fails
  is not paid.
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
  chain configured, a call that cannot be paid (no key, no PaymentChannel,
  no channel opened) ABORTS before anything is sent.

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
  "name": "llama_70b_chat",
  "description": "Chat completions from Llama 3.1 70B Instruct.",
  "mode": "session",                          // or "request"
  "settings": {"model": "llama3.1:70b-instruct-q4_K_M"},
  "measures": [{"measure": "input_tokens", "quantity": 1000000},
               {"measure": "output_tokens", "quantity": 1000000}],
  "protocol": {"id": "openai-chat/1", "spec": "<sha256>",
               "spec_url": "https://..."},   // copied from the tool's manifest
  "input_schema": { ... },                    // the opening arguments, or a request's
  "author": "<agent-id>",
  "author_pubkey": "0x...",                   // the spec's signer and payee
  "author_sig": "...",
  "image_uri": "https://...jpg",
  "version_of": null,
  "created_ts": 0
}
```

The price list beside it (signed by the provider, gossiped with the spec):

```json
{"kind": "service_prices", "service": "<spec digest>", "provider": "0x...",
 "prices": {"input_tokens": 100000, "output_tokens": 400000},
 "voucher": 500000, "seq": 3, "ts": 0, "sig": "..."}
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
  refunds at once after the provider's close, and after a challenge
  window (the provider's time to collect) after the client's. Theft ceiling = ONE voucher, sized
  by the client, worth less than the review history it burns. No
  arbitration, no fulfillment oracle, two txs for N requests. Metered
  hardware (GPU/sponsor-pipe inference) fits natively: the spec's
  `unit` says what one work item is, and the vouchers grow by one price
  per unit consumed (Decision 2026-11-10). Streaming
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

### 4. Reviews (BUILT 2026-11-10)

A review is the payer's signed rating of one paid request, proven by the
provider's signed receipt and the payment on chain, carried as a
`service_review` event in the epoch batches and counted by the epoch-close
into per-service ratings in
the anchored carry bundle (Decision "service reviews in the epoch-close"
at the top: fields, receipt and event formats, counting, bounds). Only
addresses that paid can review (receipt-gated, so the fake-review problem
of open review sites does not arise between strangers), each payer counts
once, and ratings are displayed, never enforced: buyers see the history
and price accordingly.

- Buyer side: `atn/service_receipts.py` (`ReceiptStore`: receipts,
  private reply blobs, reviews, malware evidence) and the
  `review_service` agent tool; `service_receipts` (WS) lists them.
- Consensus side: `nodes/common/service_reviews.py` (hashes, validation,
  `close_service_reviews`), called from `federated_epoch_close`; the
  driver carries `service_reviews` and `service_ratings` and answers
  `service_rating(provider, digest)`.
- Search: catalog rows carry `rating`.

The trust surface beside the ratings:

- the on-chain provider record (`serviceEarnings`, `agentMintTotal`,
  registration tenure), read by the `provider_record` WS handler. It is
  a consequence of payments, so it is costly to inflate: a provider can
  pay itself, at the cost of the fee each time.
- the provider-local success rate and volume from
  `ServiceStore.summary()`, returned alongside `list_services`: requests
  that passed the payment gate, attributed only to the payer it verified.

### 5. Marketplace surface

- Web app: new Services screen, offering browse (the daemon's catalog,
  `market_services`),
  spec inspection, purchase flow (wallet), and the rating from paid
  reviews (each `market_services` row's `rating`, §4); the trust strip
  shows the provider record plus the provider-local success rate. The
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
   remainder refund, after the client's close (a provider's close
   collects and refunds at once, 2026-11-11). Governed since 2026-10-16: the constructor sets the
   initial value and the DAO timelock (the Substrate's governor) changes
   it with `setChallengeWindow`. Each channel records the window it
   opened with, so a change never moves an open channel's deadline.
2. Where reviews live: settled 2026-11-10, as events in the epoch
   batches, rated by the close and carried in the anchored bundle; no
   chain events and no indexer mirror.

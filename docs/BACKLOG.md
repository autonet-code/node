# Autonet Backlog

Consolidated 2026-07-06; updated 2026-07-09 after substrate v4.1. This is
THE board; update in place as items land or get ratified.

State: **substrate v4.1 "gradient trust" BUILT 2026-07-09** (close +
contract + surfaces; UNCOMMITTED, sim-validated in
`experiments/econ_attest/`, spec = `docs/tool_substrate.md` Decision
2026-07-09). The vet GATE is retired (tools mint from first attested
use); mint is rep/ATN-SPLIT under a supply-pegged β cap (zero-rep usage
mints ATN, not reputation); drift weight = rep_share × credibility with
NO ε floor + continuous reversal-aware credibility docking; vets survive
as inspection reviews; `recordTrainingForEpoch(amount, repAmount,
epochIdHash, proof)` with a 3-field merkle leaf + `TrainingRecorded`
gains `repAmount`. PENDING (v4.1):
- **FLAG-DAY redeploy** — v4.1 changes close output/CID + contract ABI +
  merkle leaf; needs a clean genesis on a fresh Substrate + all daemons
  on the v4.1 build before the next federated close (supersedes the v3
  flag-day window below).
- **G1 — channel fee decision (USER):** the 2.5% burn/recycle fires only
  on `Substrate.payForService`; `ServiceMarket.PaymentChannel` settles
  via raw `safeTransfer` (no fee, recycled=0). "Services finance the
  commons" is unwired on the default rail. Fix: take the fee at channel
  settlement (or route payouts through payForService); non-ATN channels
  need a design call (fee is ATN-denominated).
- **S₀ recalibration at launch (USER):** `BETA_S0=5000` (50 epochs × 100
  pool) is the sim's engineering pick over a short 300-epoch horizon;
  retune to the real epoch cadence + pool size before mainnet.
- **v4.1 params seeded-not-blessed:** `BETA_MIN=0.05`, `CRED_DELTA=0.7`
  (δ EMERGED, others SEEDED), `CRED_MASS_FLOOR=3.0`, `CRED_FLOOR=0.1`,
  `CRED_RECOVERY=0.10`, pool 100. δ=0.7 and the necessity of a
  non-constant supply-pegged β emerged from sims; the rest are
  conventional, tunable post-launch.

Prior state: **substrate v3 shipped and PUBLIC** (master pushed 2026-07-08).
Reviews replace debates (`docs/tool_substrate.md`, Decision 2026-07-08):
mint = usage alone, per-axis reviews drift tool positions, work units
left consensus, debate rail retired/dormant. Both economy E2Es
re-verified green ON v3 against a live chain (tool economy 7/7 — mint =
log1p(1) exactly; venture loop 8/8). Also landed: Secrets tab + vault
audit trail, the unified Substrate visualization, Tools Local/Substrate
catalog, review step wired into the agentic loop, register()
idempotency fix, and the README-as-living-paper ("Autonet — The
Recursive Principial Body") now serving both app whitepaper surfaces.

**V3 OPS (time-sensitive):**
- **FLAG-DAY (updated post-genesis 2026-07-08 evening): daemons need
  the CURRENT MASTER build (v0.6.0 unreleased — release pending user
  go), not v0.5.x** — payForService rename + reputation-source voice
  + vault minter changed contract ABI and close inputs; v0.5.x builds
  cannot talk to the genesis Substrate. Restart onto master (or the
  next release) + re-register agents (now with owner-binding popups).
- Deploy Firestore rules for `substrate_viz` (world-readable,
  client-unwritable) + run the substrate publisher next to a daemon
  (`scripts/indexer/` is gitignored ops — lives outside the repo).
- Chrome visual pass on the unified Substrate view (organic-feel knobs:
  `organic_blob.dart` constants, `_kDescendFraction`,
  `_kOpennessSpeed`).
- Secrets broker E2E (vault setup + `ATN_WORKER_ISOLATION=1`) still
  unrun.

**TO FIX / HARDEN:**
- **The v3 sybil trade — MITIGATION BUILT 2026-07-08 (balance-weighted
  voice, spec addendum in `docs/tool_substrate.md`):** callers now
  collapse to HOUSEHOLDS (proven owner wallet) before log1p damping,
  and each household's usage/review credit scales by
  `ε + household_ATN/supply` (linear ⇒ balance-splitting is
  weight-neutral; zero-balance identities carry at most ε = 0.05
  PROVISIONAL, needs blessing). Wired: close math
  (`federated_reconcile.py`), chain sourcing
  (`voice_state.py` → driver `voice_source` hook), `fleet_voice` WS +
  Owner page in atn_web (wallet dropdown). Tests:
  `tests/test_voice_weights.py` (10) + all close families green.
  Reads are PINNED to the previous epoch's anchor block
  (`getAnchor(...).blockNumber`) and served WITHOUT archive state:
  Substrate.sol ATN now carries IVotes-mechanism checkpoints
  (`balanceOfAt`/`atnTotalSupplyAt`, Trace208, no delegation layer —
  undelegated getPastVotes would read 0) and the agent/owner maps
  derive from event logs up to the snapshot. Chain tests
  `tests/test_voice_snapshot.py` (pin property verified); stale
  phase5_6/phase7_2 mint fixtures migrated to v3 tool events.
  REMAINING: (a) the ε floor is the residual sybil surface — bounded,
  but the dormant CON/bust rail stays the named backstop; (b) covert
  harm still has ONLY vetting as its dedicated defense — vet params
  (below) remain load-bearing. NOTE: close output changed + CONTRACT
  changed (checkpoints) — the shadownet redeploy now carries this too;
  same flag-day window as v3 (fleet is still one daemon).
- ~~world_persistence artifact_digest gap~~ — FIXED 2026-07-08:
  digest mirrored into node.metadata (serializer carries it) +
  `lift_artifact_digests` rehydrates after every restore_world
  (world_persistence + both state_sync paths); roundtrip test pins it.
  NOTE: metadata now carries the digest → checkpoint content/world_cid
  changes — rides the open flag-day window.
- Agentic-loop review guidance lives only in tool descriptions +
  `_TOOL_CATEGORY_NOTES`; agents with a custom `system_prompt` bypass
  the category notes (they still get the injected closing review turn).

**USER-ONLY DECISIONS (nothing proceeds without these):**
1. ~~Emission rate~~ — shape RATIFIED + BUILT 2026-07-08:
   **fee-recycled emission** (`docs/epoch_economics.md` Decision) —
   pool = BASE floor + burned service fees recycled (wash-proof by
   conservation; treasury share to the DAO). Remaining user-only:
   bless the PROVISIONAL values — SERVICE_FEE_BPS (2.5%),
   FEE_TREASURY_BPS (50% of fee), BASE_EMISSION_PER_EPOCH (100 ATN).
   Fee params are on-chain constants (redeploy to change); BASE is
   consensus-flag-day.
2. repPerReputation rate on ReputationMirror (default 1:1, timelock-
   governed) — the direction/quality exchange rate; flagged as the
   most consequential unbuilt decision; wants sims.
3. Vet params blessing: sim says N=3 (code has 2.0); royalty×weight
   slash fix (rubber-stamper out-earns careful vetter in deep pools).
   Post-v3 these are LOAD-BEARING — vetting is the only dedicated
   defense against covert harm (see sybil trade above). Note the
   tool-economy sim memo is quarantined (models v2 mint); re-run the
   sweep against usage-only mint before citing absolute numbers.
4. Channel challengeWindow (E2E uses 3600s).
5. ~~Push master to origin~~ — DONE 2026-07-08 (no tag, no release).
   atn_web commits remain local-only.
6. CLAUDE.md is gitignored — machine-onboarding contract doesn't
   travel with clones. Commit / genericize / keep local?
7. ~~Charter-anchor governor handoff~~ — DONE 2026-07-08 (genesis v2):
   CharterAnchor governor = jurisdiction timelock; charter v1 ANCHORED
   on-chain via executed DAO proposal.
8. Admin agent: merged but placeholder budget (100k tokens/day) —
   tune + enable; it starts spending on a 30m schedule once live.
9. Microsoft Store: pay $19, fill the two msix identity placeholders;
   create WINGET_GH_TOKEN; first winget submission via wingetcreate
   new. (RELEASING.md in atn_web has the checklist.)
10. ~~ReputationMirror activation~~ — DONE 2026-07-08 (genesis v2):
    setMinter(mirror) EXECUTED via DAO proposal; mirror can mint.

**RATIFIED, NOT BUILT:**
- ~~ATN = money, reputation = voice~~ — BUILT + DEPLOYED 2026-07-08
  (genesis): voice weights read soulbound reputation at the pinned
  block (reputationOfAt checkpoints); AutonetVault (renamed from
  ParityVault) live on shadownet as Substrate's immutable minter,
  buy-only, parity via jurisdiction Registry entries; vault mint moves
  balance, never voice. Genesis addresses in registry.json. Historical
  design entry follows:
- **ATN = money, reputation = voice (2026-07-08 evening) — the
  jurisdiction merge, attempt 6.** Once ATN is buyable, balance-based
  voice = purchasable review power (pumped ratings would rig
  probe_tools discovery). Decision: switch the close's voice-weight
  source from ATN balance to SOULBOUND REPUTATION (earn-only, minted
  1:1 with ATN, non-transferable) — same household collapse, same
  pinned-anchor reads (needs rep-at-block derivation: checkpoint or
  TrainingRecorded logs). ATN stays purely economic. ParityVault
  (buy-only v1): CoinOffering clone minting Substrate ATN via an
  IMMUTABLE minter address in Substrate's constructor (no-admin-keys
  preserved); payment tokens + rates = jurisdiction Registry entries
  (`jurisdiction.parity.<token>`), timelock-governed; ERC20 proceeds →
  DAO treasury. Rides the owed redeploy (payForService rename + clean
  genesis via sacrificial smoke instance). BLOCKED on user: make the
  trustless-contracts DAO factory deploy a jurisdiction WITH a
  coin-offering-shaped satellite slot; then the genesis event chains
  #7 (CharterAnchor governor handoff) + #10 (ReputationMirror
  activation) + the ParityVault minter grant.
- Tokenized vault shares (2026-07-06): ERC20 per vault, book-value
  redemption (burn → unvested principal + accrued revenue),
  checkpointed halt votes, CoinOffering dividend pattern ported.
  Boats project re-conceived as the flagship venture agent.
- Verifier rewards migration: mint-funded bootstrap → bps-of-raise
  fee once volume exists.
- Capability-gap pricing: coverage-atlas gaps as mint multiplier
  (doctrine only, never implemented).

**ENGINEERING FOLLOW-UPS (flagged by builders, none blocking):**
- Jurisdiction governance timing: the genesis deploy's VOTING_PERIOD
  constant (50400, "~1 week in blocks") lands on-chain as 3,024,000s
  (~35 days — timestamp clock, units mismatch). Make votingDelay/
  votingPeriod/timelock minDelay env-configurable in
  deploy-autonet-jurisdiction.js and use short values on shadownet.
  RESOLVED same evening: durations made env-configurable (seconds,
  no hidden x60), jurisdiction REDEPLOYED with 0/120s/0 (genesis v2,
  all addresses rotated - registry.json is truth), and the genesis
  proposal (mirror minter grant + charter v1 anchor) went
  propose->vote->queue->execute LIVE in ~3 minutes. Mirror IS a
  RepToken minter; charter v1 IS anchored. Etherlink log-range cap
  requires PROPOSAL_ID/FROM_BLOCK on the proposal scripts.
- **Owner binding is UNWIRED in the live path (2026-07-08):** the daemon
  registers agents via plain `registerAgent` (ownerless, `agentOwner=0x0`);
  `registerAgentBound`/`rotateOwner` exist only in the contract + tests +
  E2E script. Balance-weighted voice therefore falls back to
  household=agent-id for every real agent — the human-wallet household
  model is built but unreachable from the product. Wire: frontend collects
  the EIP-712 OwnerBinding signature from the connected wallet during
  registration (fold into the existing popup flow), daemon submits
  registerAgentBound; Owner page gets a "rebind fleet to this wallet"
  surface (rotateOwner) — also the key-loss recovery UI.
- Rename `payForInference` → `payForService` (next redeploy): it settles
  services/venture revenue generally; "inference" is residue from the
  decentralized-inference framing and invites conflation with the
  sponsor/dependent provider pipe (which never touches ATN).
- TrialRunner: transport seam → wire to live WS service-request
  client; OnChainService needs a generic contract-call so attestTrial
  submits directly (currently returns calldata).
- Orchestrator drawer: add the 120s stall timeout the chat surface
  got (silent-stall symmetry).
- Adopted tools: OS-level isolated runner (vault track) as the wall
  behind tool_guard's audit-hook tripwire; macOS isolation untested.
- Indexer: mirror the new events — ToolRegistered, VentureCreated,
  CharterAnchored, service registry — into Firestore collections
  (`tools`/`services` collections are documented as future, not
  implemented).
- MCP server quirk: any Claude Code session with the atn MCP
  configured silently stands up a full runtime on :7700 — should
  attach to an existing daemon when one is running, self-host only
  otherwise.
- Registrations carry-over GC (someday): the consensus map grows
  monotonically; mechanical GC = drop digests with zero usage for N
  epochs, re-registration re-admits. Not a design item.
- ~~Shadownet redeploy~~ — DONE 2026-07-08 (user-gated, user said go):
  Substrate `0x112F617De881ef0e94B010d9756aa8aef2794208` (checkpointed
  ATN + owner binding + endpoint surface), CharterAnchor
  `0x2B239bBE...c47C` (governor = deployer pending #7 handoff; charter
  v1 anchored, hash 5756ed3a...), ServiceRegistry `0x080346d2...6400`.
  PaymentChannel NOT deployed (challengeWindow = decision #4).
  registry.json + Firestore settings/network updated, indexer@shadownet
  restarted, testnet smoke green (anchor + dual mint verified on-chain),
  live read_voice_state verified (snapshot-pinned, ε floor pre-mint).
  REMAINING user-side: restart daemon(s) onto this build, re-register
  agents (rpb_reconcile_registrations flips them local-unregistered).
- atn_web: Flutter SDK one-line patch (VS 2026 generator) dies on
  flutter upgrade; CMAKE_POLICY_VERSION_MINIMUM=3.5 needed per local
  windows build until firebase fixes its CMakeLists.

**EXPERIMENTS:**
- Phase 9 (equilibration at depth) — still unrun, still the
  pre-committed FINAL test of the equilibrated kernel. (Post-v3 the
  live path doesn't read equilibration, but the pre-commitment
  stands: run it or formally retire it via a dated decision.)
- Phase 11 (proposed, unregistered): minimum-model-tier gap bare vs
  substrate-assisted, claim = gap widens as tool corpus grows — the
  decentralization mandate made falsifiable. Phase-10 harness reuses.

**THE ACTUAL CRITICAL PATH (non-code):** one real customer / demand
beachhead — a paid service category with real buyers where crypto
settlement is an advantage (hardware/compute rental for agents = best
named candidate). Show one other human the venture-loop E2E; fund one
tiny real venture. The v3 flywheel (reviews → ranking → usage → mint)
cannot spin with one participant. The network begins the moment it
isn't just its author.

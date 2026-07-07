# Autonet Backlog

Consolidated 2026-07-06; updated 2026-07-08 after substrate v3. This is
THE board; update in place as items land or get ratified.

State: **substrate v3 shipped and PUBLIC** (master pushed 2026-07-08).
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
- **FLAG-DAY: restart every daemon onto the v3 build BEFORE the next
  federated close** — v3 changed the close output/CID; a mixed fleet
  forks. The restart also collapses the duplicated harness records
  (13× per boot pre-fix) and activates `economy_graph`/`tool_reviews`
  for the new UI surfaces.
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
- world_persistence: checkpoint doesn't persist `artifact_digest` →
  viz `kind` tagging and ratings-lift ranking degrade after a daemon
  restart until the next close reapplies positions (carry-over maps
  survive; the anchor-node attribute doesn't).
- Agentic-loop review guidance lives only in tool descriptions +
  `_TOOL_CATEGORY_NOTES`; agents with a custom `system_prompt` bypass
  the category notes (they still get the injected closing review turn).

**USER-ONLY DECISIONS (nothing proceeds without these):**
1. Emission rate — fixed pool vs floating; fixed recommended. Sets
   inflation + backer dilution modelability. (Still unset; every mint
   number is provisional until this is blessed.)
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
7. Charter-anchor governor handoff: deploy the jurisdiction DAO suite
   and point CharterAnchor's governor at the timelock (genesis charter
   hash 5756ed3a...).
8. Admin agent: merged but placeholder budget (100k tokens/day) —
   tune + enable; it starts spending on a 30m schedule once live.
9. Microsoft Store: pay $19, fill the two msix identity placeholders;
   create WINGET_GH_TOKEN; first winget submission via wingetcreate
   new. (RELEASING.md in atn_web has the checklist.)
10. ReputationMirror activation: governor executes setMinter(mirror)
    — deploy script prints the calldata.

**RATIFIED, NOT BUILT:**
- Tokenized vault shares (2026-07-06): ERC20 per vault, book-value
  redemption (burn → unvested principal + accrued revenue),
  checkpointed halt votes, CoinOffering dividend pattern ported.
  Boats project re-conceived as the flagship venture agent.
- Verifier rewards migration: mint-funded bootstrap → bps-of-raise
  fee once volume exists.
- Capability-gap pricing: coverage-atlas gaps as mint multiplier
  (doctrine only, never implemented).

**ENGINEERING FOLLOW-UPS (flagged by builders, none blocking):**
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
- Shadownet: deployed Substrate.sol predates the entire tool/vault/
  anchor surface; redeploy is a user-gated batch (user removed the
  old task; it returns when they say so).
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

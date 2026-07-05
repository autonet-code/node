# Autonet Backlog

Consolidated 2026-07-06 after the bring-it-home sprint. This is THE
board; update in place as items land or get ratified.

State: economy + framework MVP-complete and merged to master (NOT
pushed). Venture-loop E2E 8/8, tool-economy E2E 7/7. Repo tidied,
README rewritten, docs indexed. See [[venture-vault-design]] and
[[tool-substrate-refactor]] for the full ledgers.

**USER-ONLY DECISIONS (nothing proceeds without these):**
1. Emission rate — fixed pool vs floating; fixed recommended. Sets
   inflation + backer dilution modelability.
2. repPerReputation rate on ReputationMirror (default 1:1, timelock-
   governed) — the direction/quality exchange rate; flagged as the
   most consequential unbuilt decision; wants sims.
3. Vet params blessing: sim says N=3 (code has 2.0); royalty×weight
   slash fix (rubber-stamper out-earns careful vetter in deep pools).
4. Channel challengeWindow (E2E uses 3600s).
5. Push master to origin (the entire economy is local-only commits).
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
  (doctrine in CLAUDE.md, never implemented).

**ENGINEERING FOLLOW-UPS (flagged by builders, none blocking):**
- TrialRunner: transport seam → wire to live WS service-request
  client; OnChainService needs a generic contract-call so attestTrial
  submits directly (currently returns calldata).
- Orchestrator drawer: add the 120s stall timeout the chat surface
  got (silent-stall symmetry).
- world_persistence: checkpoint doesn't persist artifact_digest →
  tool STANDING doesn't survive daemon restart on the local
  projection (carry-over maps do).
- Adopted tools: OS-level isolated runner (vault track) as the wall
  behind tool_guard's audit-hook tripwire; macOS isolation untested.
- Indexer (EightRice/indexer, apps/autonet): mirror the new events —
  ToolRegistered, VentureCreated, CharterAnchored, service registry —
  into Firestore collections.
- Shadownet: deployed Substrate.sol predates the entire tool/vault/
  anchor surface; redeploy is a user-gated batch (user removed the
  old task; it returns when they say so).
- atn_web: Flutter SDK one-line patch (VS 2026 generator) dies on
  flutter upgrade; CMAKE_POLICY_VERSION_MINIMUM=3.5 needed per local
  windows build until firebase fixes its CMakeLists.

**EXPERIMENTS:**
- Phase 9 (equilibration at depth) — still unrun, still the
  pre-committed FINAL test of the equilibrated kernel.
- Phase 11 (proposed, unregistered): minimum-model-tier gap bare vs
  substrate-assisted, claim = gap widens as tool corpus grows — the
  decentralization mandate made falsifiable. Phase-10 harness reuses.

**THE ACTUAL CRITICAL PATH (non-code):** one real customer / demand
beachhead — a paid service category with real buyers where crypto
settlement is an advantage (hardware/compute rental for agents = best
named candidate). Show one other human the venture-loop E2E; fund one
tiny real venture. The network begins the moment it isn't just its
author.

# Autonet V1 Backlog

Status: DRAFT
References: PLAN.md, emergent_alignment.md, trustless-contracts, ATN, daemon

V1 launches in Scenario B: centralized inference, native model training in
background, alignment framework and economic loop functional from day one.

---

## Priority: What makes V1 a product

The primary product is the alignment framework + economic loop + agent
framework. Native model training is parallel, not blocking.

```
Track 1 (product):  Economic Layer → Governance Integration → Agent + UI
Track 2 (infra):    P2P Layer → Data Pipeline → Node Packaging
Track 3 (parallel): Native Model Training → Guild Formation → Inference
```

---

## Track 1: The Product

### Epic 1: Economic Layer (trustless-contracts are closest to ready)

Foundation: `Autonet.sol`, `Registry.sol`, `RepToken.sol` in trustless-contracts.

| # | Story | Size | Status | Notes |
|---|-------|------|--------|-------|
| 1.1 | Deterministic decay curve on epoch budgets | M | **DONE** | `configureEmission(initial, decayRateBps, epochDuration)` in Autonet.sol. Exponential: `budget(N) = initial * (rate/10000)^(N-1)`. Configurable retention rate (default 99%). |
| 1.2 | Automate epoch lifecycle (start/finalize without admin) | M | **DONE** | `advanceEpoch()` permissionless after duration elapsed. Auto-triggers on `attestUsage()`. No admin needed. |
| 1.3 | Bridge centralized inference usage to `attestUsage()` | L | **DONE** | `InferenceAttestor` in `nodes/common/inference_attestor.py`. Records per-call tokens, batches, auto-flushes to `attestUsage()`. Wired into InferenceNode. Blob store receipts for auditability. 18 tests. |
| 1.4 | Training activity → `attestUsage()` | M | **DONE** | GovernanceBridge.attest_task_completion() wired into solver + aggregator. Per-attester tracking in epochAttesterUsage. |
| 1.5 | Epoch rewards claimable by participants | M | **DONE** | `claimParticipantReward(serviceId, epochId)` in Autonet.sol. GovernanceBridge.claim_epoch_rewards() in nodes. Proportional to attested usage. |
| 1.6 | Training participation → `claimReputationFromEconomy()` | M | **DONE** | GovernanceBridge.claim_reputation() calls RepToken. Wired into solver + aggregator. train → attest → earn ATN → earn RepToken → vote. |
| 1.7 | Dynamic training pricing (capability gap → reward multiplier) | L | **DONE** | `CapabilityScorecard.sol`: per-module scores, EMA updates, reward multipliers. Autonet._finalizeEpoch() applies multipliers. Low-capability modules get 3x, saturated get 0.5x. |
| 1.8 | Alignment pricing function (from emergent_alignment paper) | L | **DONE** | `AlignmentPricing` in `nodes/common/alignment_pricing.py`. Geometric mean of 3 pairwise similarities. Subsidized/neutral/premium tiers. Training incentives via capability gap → reward multiplier. Advisory in V1. 30 tests. |
| 1.9 | Rollup deployment on established L1 | L | | Autonet as rollup anchored in Ethereum for sybil resistance |

### Epic 2: Governance Integration

Connect fractal governance to the economic layer and node operations.

| # | Story | Size | Status | Notes |
|---|-------|------|--------|-------|
| 2.1 | Evolution proposal contract (`EvolutionProposal.sol`) | L | **DONE** | `submitProposal(cid, stake)` → lifecycle Proposed→Evaluating→Trial→Adopted/Rejected. Stake refund on rejection minus 5% fee. 48 Solidity tests. |
| 2.2 | Trial funding from network budget | M | **DONE** | `fundTrial(proposalId, budget)` transfers ATN. `recordTrialContribution()` tracks participants. `adoptProposal()` returns stake + enables reward claiming. |
| 2.3 | RPB constitutional prompt stored in Registry | S | **DONE** | `updateRPBPrompt(cid)` timelock-protected. Versioned: `rpb.prompt.v1`, `rpb.prompt.current`, `rpb.prompt.version` in Registry. History permanent on-chain. |
| 2.4 | Node-side RPB evaluator (AI provider abstraction) | M | **DONE** | `RPBEvaluator` class loads prompt from Registry, evaluates via `AIProvider` interface (abstract). `PlaceholderAIProvider` for dev. `RPBRecommendation` structured output. |
| 2.5 | Consensus on RPB recommendations (extend Yuma) | L | **DONE** | `RPBConsensus` class calls permissionless `resolveEvaluation()`. Weighted confidence voting: approval = sum(approve_confidence) / sum(all_confidence). Quorum + time-gated. |
| 2.6 | RPB prompt governance (DAO proposal to amend prompt) | M | **DONE** | `updateRPBPrompt()` restricted to `onlyTimelock`. `linkDAOProposal()` connects evolution proposals to HomebaseDAO votes. Higher quorum enforced at DAO level. |
| 2.7 | Draft v1 RPB constitutional prompt | M | **DONE** | Universal Declaration of Human Rights in `constitution/v1_udhr.txt`. `rpb_prompt.py` handles deploy to blob store + load with evaluation framing. RPBEvaluator wired to resolve CID → content via blob store with local file fallback. |
| 2.8 | Contribution spectrum rewards (compute, diagnosis, proposal, validation) | M | **DONE** | `ContributionWeights`: compute=1x, diagnosis=1.5x, proposal=3x, validation=1.2x. `claimTrialReward()` distributes proportional to weighted units. Configurable via `setContributionWeights()`. |
| 2.9 | Governance heartbeat enforcement in daemon | M | **DONE** | `GovernanceBridge.check_heartbeat()` monitors `HeartbeatEmitted` events. Configurable interval (60s default). Work halts if missed. Already wired in prior session. |

### Epic 3: Agent Framework + UI

ATN (`c:\code\atn`) is the most developed component. Integrate with governance.

| # | Story | Size | Notes |
|---|-------|------|-------|
| 3.1 | Wallet connection in ATN (create or import) | M | Key management. User owns wallet. |
| 3.2 | Standards publication during onboarding | M | User defines personal standards, published on-chain (as in emergent_alignment paper) |
| 3.3 | "Enable network training" toggle in settings | S | Opt-in background training |
| 3.4 | Data source selector (OBS-style: pick apps/screens to share) | M | User controls what activity becomes training data |
| 3.5 | Earnings dashboard (ATN earned, rounds, reputation) | M | Pull from on-chain epoch data + local metrics |
| 3.6 | Proposal submission UI (describe, stake, track lifecycle) | M | Natural language description → CID → stake → submit |
| 3.7 | Governance UI (vote on proposals, view RPB recommendations) | M | Integrate with DAO voting |
| 3.8 | Alignment pricing visibility (show alignment score for operations) | S | Advisory in V1 — user sees the score even if pricing isn't enforced yet |
| 3.9 | First-run onboarding: explain the system, set expectations | S | Model may not be useful yet. Training earns tokens. Evolution is open. |

---

## Track 2: Infrastructure

### Epic 4: P2P Communication Layer

Nodes need to talk to each other. Blockchain for consensus, P2P for data.

| # | Story | Size | Status | Notes |
|---|-------|------|--------|-------|
| 4.1 | libp2p integration: DHT discovery + NAT traversal | L | **DONE** | `AutonetHost` wraps py-libp2p. TCP + optional QUIC transport, Noise encryption, UPnP NAT traversal. Bootstrap peer support. 4 lifecycle tests. |
| 4.2 | Latency-aware peer selection | M | **DONE** | `PeerLatencyTracker` using libp2p PingService. EMA-smoothed RTT (alpha=0.3). `select_fastest_peers()`, `select_peers_for_route()`. 3 tests. |
| 4.3 | Weight delta transfer over P2P (not just blob store) | M | **DONE** | Protocol `/autonet/weights/1.0.0`. Length-prefixed framing with SHA-256 integrity check. Chunked writes for Noise layer compatibility (65KB limit). 1 integration test. |
| 4.4 | Activation relay for pipeline-parallel inference | L | **DONE** | Protocol `/autonet/activations/1.0.0`. Framed: `[total_len][meta_len][metadata_json][tensor_bytes]`. Handles 300KB+ tensors via chunked I/O. 1 integration test. |
| 4.5 | Guild-local gossip (intra-jurisdiction coordination) | M | **DONE** | GossipSub (lazy init). `GuildMembership` abstraction. `join_guild()`, `publish_guild_message()`, `leave_guild()`. Topic: `/autonet/guild/{id}`. 2 tests. |
| 4.6 | Cross-guild routing for inference pipeline | M | **DONE** | `select_peers_for_route(module_ids)` picks lowest-latency peer per module across guilds. `get_guild_for_module()` lookup. 1 test. |
| 4.7 | Node capability advertisement via DHT | S | **DONE** | `NodeCapability` dataclass with JSON serialization. Protocol `/autonet/capability/1.0.0`. `advertise_capability()` pushes to connected peers. `find_capable_peers(role, min_gpu, module_id)`. 3 tests. |

### Epic 5: Data Pipeline

User activity → local training data → weight deltas.

| # | Story | Size | Status | Notes |
|---|-------|------|--------|-------|
| 5.1 | Screen/app capture service in daemon | L | **DONE** | `CaptureSource` ABC + `ScreenCaptureSource` (mss) + `AudioCaptureSource` in `nodes/common/capture.py`. `CaptureManager` with FPS-capped round-robin. Frame dataclass with image_data bytes. 22 tests. |
| 5.2 | Browser extension for text extraction | L | **DONE** | Expanded ATN's Agent CDP Relay extension: `training-capture.js` content script extracts page text on load/mutation/SPA-navigation. Service worker forwards `training_frame` messages through relay's new `/training` WebSocket endpoint. Autonet's `BrowserCaptureSource` connects to relay, receives text frames with PII scrubbing via `scrub_text()`. Wired into `CaptureManager.create_sources_from_config()` as `"browser"` source type. Config: `browser_relay_url`, `browser_exclude_patterns`, `browser_scrub_pii`. Privacy filter extended to check `url` metadata. 22 tests. |
| 5.3 | Local data preprocessing (frames → JEPA training batches) | M | **DONE** | `FramePreprocessor` in `nodes/common/preprocessing.py`. PNG→PIL→Resize→Normalize (ImageNet stats)→[B,C,H,W] tensors. `extract_patches()` utility. `TrainingBatch` dataclass. Handles RGBA/grayscale conversion. 14 tests. |
| 5.4 | Audio capture (optional modality) | M | **DONE** | `AudioCaptureSource` stub in `nodes/common/capture.py`. Modality="audio", start/stop lifecycle, yields no frames until real audio backend wired. 4 tests. |
| 5.5 | Data source configuration persistence | S | **DONE** | `CaptureConfig` dataclass in `nodes/common/config.py`. YAML persistence in `autonet.yaml` (sources list with type/name/enabled, fps_cap, buffer_size). `load_config()`/`save_config()`. |
| 5.6 | Privacy controls: exclude list, blur regions, scrub PII | M | **DONE** | `PrivacyFilter` in `nodes/common/privacy.py`. App exclusion (case-insensitive source_id + window_title), blur regions (PIL GaussianBlur), PII scrubbing (email/phone/CC/SSN regex). `scrub_text()` standalone. `PrivacyConfig` dataclass. 16 tests. |

### Epic 6: Node Packaging

Get the thing installable.

| # | Story | Size | Status | Notes |
|---|-------|------|--------|-------|
| 6.1 | Solver node as daemon background service | L | **DONE** | `AutonetService` in `nodes/service.py`. Signal-based lifecycle (SIGTERM/SIGINT), main loop with resource + update checks. `nodes/__main__.py` entry point (`python -m nodes`). `ServiceState` enum. `__version__ = "0.1.0"`. 6 tests. |
| 6.2 | Resource limit configuration (CPU%, memory, active hours) | M | **DONE** | `ResourceMonitor` in `nodes/common/resource_monitor.py`. psutil CPU/memory, torch.cuda GPU. Active hours (normal + overnight ranges). Rate-limited checks. `ResourceConfig` dataclass. 7 tests. |
| 6.3 | Environment config (YAML: RPC, blob store, device, model) | S | **DONE** | `AutonetConfig` with all subsystem configs in `nodes/common/config.py`. `autonet.yaml` with capture, privacy, resources, update, guild sections. `load_config()`/`save_config()`. |
| 6.4 | Single installer bundling ATN + autonet node | L | | Platform-specific (Windows, Mac, Linux) |
| 6.5 | Auto-update mechanism for node software | M | **DONE** | `AutonetUpdater` in `nodes/common/updater.py`. Git (fetch+rev-parse), HTTP (version.json+SHA-256), blob_store sources. `UpdateInfo` with has_update property. `UpdateConfig` dataclass. 6 tests. |

---

## Track 3: Native Model Training (Parallel)

Aspirational. Runs alongside Track 1-2. Not blocking the product.

### Epic 7: Wire Real Training

| # | Story | Size | Status | Notes |
|---|-------|------|--------|-------|
| 7.1 | Solver loads global JEPA model from blob store | M | **DONE** | Deserialize real weights on startup |
| 7.2 | Replace mock training with `train_jepa_on_task()` | M | **DONE** | Remove fake hash generation. Real local training. |
| 7.3 | Real weight delta computation and upload | M | **DONE** | `delta = new - old`, serialize, store |
| 7.4 | CPU-only mode with reduced model/batch | S | **DONE** | Env var `AUTONET_DEVICE=cpu|cuda` |
| 7.5 | Proposer generates real task specs (hyperparams, data config) | M | **DONE** | Replace placeholder specs |
| 7.6 | Coordinator verifies real weight deltas | M | **DONE** | Cosine similarity, energy checks on real data |
| 7.7 | Aggregator performs real FedAvg on deltas | M | **DONE** | Wire to real blob store data |
| 7.8 | End-to-end real training round (multi-node test) | L | **DONE** | `tests/test_e2e_real_training.py`: full pipeline (train → aggregate → infer → attest → price). Multi-round FedAvg. Blob store integrity. 7 tests. |

### Epic 8: Guild Formation

Jurisdictions that specialize in training specific model modules.

| # | Story | Size | Status | Notes |
|---|-------|------|--------|-------|
| 8.1 | Guild creation contract (jurisdiction + module assignment) | L | **DONE** | `GuildRegistry.sol` in `contracts/governance/`. `createGuild(name, moduleIds[])`, `joinGuild()`, `leaveGuild()`, `setGuildAggregator()`. Module exclusivity (one guild per module). ATN staking for creators/members. 35 Hardhat tests. |
| 8.2 | Intra-guild aggregation (guild aggregates member deltas) | M | **DONE** | `_cycle_guild_aggregation()` in aggregator. Filters updates by guild membership, runs FedAvg/trimmed_mean on guild-only deltas. Config: `guild.aggregation_level = "guild"`. Reports guild metrics after aggregation. |
| 8.3 | Network-level aggregation across guilds | M | **DONE** | `_cycle_network_aggregation()` in aggregator. Collects guild-level aggregates, weights by member count + reputation via `GuildManager.get_guild_weights_for_aggregation()`. Config: `guild.aggregation_level = "network"`. Falls back to flat when no guild aggregates. |
| 8.4 | Guild reputation and competition metrics | M | **DONE** | EMA-smoothed reputation in `GuildRegistry.sol` (same pattern as CapabilityScorecard). `recordGuildMetrics()`, `getGuildRanking()`, `getGuildRewardMultiplier()`. Linear 0.5x-1.5x multiplier. Reporter access control. |
| 8.5 | Guild membership and specialization matching | M | **DONE** | `GuildManager` in `nodes/common/guild_manager.py`. `recommend_guild(capabilities)` scores by module overlap + reputation. `auto_join()`, `monitor_guild_health()`, `report_guild_metrics()`. `GuildConfig` in config.py, guild section in autonet.yaml. 39 Python tests. |

### Epic 9: Native Inference

When the model becomes useful, serve it.

| # | Story | Size | Status | Notes |
|---|-------|------|--------|-------|
| 9.1 | Inference request routing through module pipeline | L | **DONE** | `InferencePipeline` in `nodes/common/inference_pipeline.py`. `plan_route()` maps tier→modules→guild peers. `execute()` runs local/stub. `InferenceRequest`/`InferenceResult`/`ModuleRoute` dataclasses. `PipelineStats` with avg_latency/success_rate. 8 tests. |
| 9.2 | Protocol-level token burn on native inference | M | **DONE** | `burnForInference(amount, tier)` in Autonet.sol. `_burn()` on caller, per-tier tracking via `tierInferenceBurned` mapping. `InferenceBurned` event. `getTierBurnStats()` view. Rejects tier 0/4+, zero amount. 11 Hardhat tests. |
| 9.3 | Intelligence tiers (more modules = deeper model = higher price) | M | **DONE** | `InferenceTier` IntEnum (1-3) in `nodes/common/intelligence_tiers.py`. `TIER_MODULES`: T1=encoder, T2=encoder+reasoning (4), T3=full pipeline (5). `TierPricing` with base_cost + per_token. `InferenceTierManager` with compute_cost, recommend_tier. 10 tests. |
| 9.4 | Decode trust verification (deterministic output matching) | M | **DONE** | `DecodeVerifier` in `nodes/common/decode_verifier.py`. `DecodeOutput` with auto SHA-256 hash of token IDs. Majority voting with configurable agreement_threshold. `VerificationResult` enum (VERIFIED/MISMATCH/INSUFFICIENT/ERROR). Mixed plan hash detection. 7 tests. |

---

## Critical Path

```
                    Track 1 (product)
                    ┌─────────────────────────────┐
Epic 1 (economic) → Epic 2 (governance) → Epic 3 (UI) → V1 SHIP
                    └─────────────────────────────┘
                              ↑
                    Track 2 (infra)
                    ┌─────────────────────────────┐
                    Epic 4 (p2p) → Epic 5 (data) → Epic 6 (packaging)
                    └─────────────────────────────┘

                    Track 3 (parallel, not blocking)
                    ┌─────────────────────────────┐
                    Epic 7 (training) → Epic 8 (guilds) → Epic 9 (inference)
                    └─────────────────────────────┘
```

Track 1 and Track 2 converge at Epic 6 (packaging). Track 3 runs
independently and joins when the native model reaches useful quality.

---

## What's NOT in V1

Explicitly deferred (described in PLAN.md for future reference):

- Cross-jurisdiction model sharing
- Differential privacy on weight deltas
- Intelligence tier pricing enforcement (advisory pricing implemented, on-chain enforcement deferred)
- EMA blending with KL divergence monitoring
- Inter-autonet cooperation (blockchain-level interop)
- Native model self-evaluation (RPB Phase 3)
- Formal module boundary discovery (empirical testing)

---

## Size Legend

| Size | Meaning |
|------|---------|
| S | Focused session — single file or small change |
| M | A few sessions — multiple files, some design needed |
| L | Significant — new contracts, new subsystems, integration work |

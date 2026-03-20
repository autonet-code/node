# Plan: Distributed VL-JEPA — Modular Training & Pipeline-Parallel Inference

## Vision

A single, continuously-evolving VL-JEPA model distributed across Autonet nodes.
Training and inference happen concurrently. The model is composed of functional
modules (layer groups) that can be independently trained, served, and upgraded
without downtime. Nodes wear multiple hats based on hardware capabilities.

## Why VL-JEPA

1. **Self-supervised**: Predicts masked patch embeddings — no labeled data needed.
   Nodes train on raw video/images. Data acquisition is not our problem.
2. **Embedding-based reasoning**: Outputs continuous vectors, not discrete tokens.
   Consensus over embeddings is mathematically clean (average vectors, measure cosine
   similarity). Much easier than comparing token sequences.
3. **One model for everything**: Classification, captioning, retrieval, VQA — all
   through the same architecture by querying the embedding space differently.
4. **Selective decoding**: Text generation (Y-Decoder) only fires when semantic
   embedding changes meaningfully. 2.85x fewer decode operations on video streams.
5. **Composable by design**: Transformer layers are naturally pipelineable — each
   takes activations in, produces activations out. Fixed-size intermediate tensors.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    AUTONET MODEL PIPELINE                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [Module 0: Visual Encoder]     layers 0-7     (nodes A, D, G)  │
│         ↓ activations (196 x 768, ~300KB fp16)                  │
│  [Module 1: Visual Reasoning]   layers 8-15    (nodes B, E, H)  │
│         ↓                                                        │
│  [Module 2: Cross-Modal Fusion] layers 16-23   (nodes C, F, I)  │
│         ↓                                                        │
│  [Module 3: Predictor]          pred layers    (nodes A, C, F)  │
│         ↓ semantic embedding                                     │
│  [Module 4: Text Decoder]       decoder layers (nodes with GPU)  │
│         ↓ text tokens                                            │
│                                                                  │
│  Each module hosted by 3+ nodes for redundancy.                  │
│  Routing picks lowest-latency path through the chain.            │
│  Intelligence tiers: pay more → more modules → deeper model.     │
└──────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Compute Shards, Not Storage Shards

Current `ModelShardRegistry` tracks storage — byte chunks for redundancy.
New architecture needs **compute shards** (runnable module segments):

- Storage shard: "I have bytes 0-1MB of the model" — not runnable
- Compute shard: "I run layers 8-15" — receives activations, produces activations

### 2. Functional Module Boundaries

Modules are not arbitrary layer ranges. They correspond to functional subsystems
that emerge naturally in transformers:

- **Visual encoder** (early layers): Low-level features — edges, textures, patches
- **Visual reasoning** (middle layers): Compositional features — objects, spatial relations
- **Cross-modal fusion** (late layers): Vision-language integration
- **Predictor**: Maps joint representations to output embeddings
- **Text decoder**: Autoregressive generation from semantic embeddings

Key property: **a module is a coherent upgrade unit if its output distribution
is stable**. You can update the visual encoder without breaking cross-modal fusion,
as long as the visual feature embedding distribution stays within tolerance.

### 3. Continuous Training via EMA Blending

No model versions. No downtime. Individual modules get updated continuously:

```
1. Training task completes for visual_encoder (layers 0-7)
2. New weights submitted by solvers, aggregated via consensus
3. For each node hosting visual_encoder:
   weights = 0.998 * weights + 0.002 * new_weights
4. Repeat every N cycles until fully absorbed (~1000 cycles)
```

- Cross-modal fusion gets slower EMA (alpha=0.999) — more sensitive to shifts
- Early layers get faster EMA (alpha=0.996) — inputs (raw patches) are stable
- If new weights cause performance regression, halt blend and revert

### 4. Performance-Driven Training Strategy

The network self-heals through a feedback loop:

```
Inference → confidence signals → training strategy → targeted training → EMA blend

Concretely:
1. Inference nodes track per-module output entropy
2. High-entropy outputs indicate model weakness
3. User feedback (on-chain dissatisfaction signals) creates additional signal
4. Proposers analyze patterns, create targeted training tasks:
   "visual_encoder is weak on medical images — train layers 0-7"
5. Solvers train just those layers, submit deltas
6. Aggregator consensus → EMA blend begins
```

### 5. Flexible Node Roles

Nodes advertise capabilities, network assigns work:

```
Node capabilities:
  - GPU: RTX 4090 (24GB VRAM)
  - Roles: [segment_compute, decode, training]
  - Modules hosted: [visual_encoder, visual_reasoning]
  - Bandwidth: 500 Mbps
  - Latency to peers: {nodeB: 12ms, nodeC: 45ms, ...}
```

A beefy GPU node runs compute segments + decode.
A CPU node with storage hosts weights and relays activations.
A high-bandwidth node becomes a routing hub.

### 6. Intelligence Tiers

More payment → more modules in the inference chain → deeper effective model:

- **Tier 1 (cheap)**: Visual encoder only → basic feature extraction
- **Tier 2 (standard)**: Encoder + reasoning → visual understanding
- **Tier 3 (premium)**: Full pipeline → multimodal generation with decode

### 7. Decoder Trust

Risk: decode node substitutes different semantics for the embedding it received.
Mitigation: random peer verification. Deterministic decoding (fixed temperature +
seed) means outputs must match byte-for-byte. Occasional duplicate routing to
verify. Mismatch → slash. Not a blocker for MVP.

### 8. Language Generation

We need full text output to power agent frameworks with NLP cognitive piping.
The decode happens wherever it optimizes efficiency, as long as trustless quality
is preserved. Options:
- Last node in pipeline (already has embedding)
- Requesting node (if it has GPU)
- Dedicated decode nodes

## Activation Transfer Budget

For ViT with embed_dim=768, 196 patches:
- float16: 196 x 768 x 2 bytes = **301 KB per forward pass**
- At 100 Mbps: ~24ms transfer. At 1 Gbps: ~2.4ms.
- 4 segments over 100 Mbps: ~72ms network + ~20-40ms compute = ~200ms total

Acceptable for question answering, text generation, document analysis.
Not fast enough for real-time video (30fps). Optimizations: int8 activations
(halve bandwidth), request batching (amortize overhead).

## Module Registry (On-Chain)

Extends current ModelShardRegistry:

```
ModuleRegistry:
  module_id: 0
  name: "visual_encoder"
  layer_range: [0, 7]
  input_shape: [196, 768]
  output_shape: [196, 768]
  current_weights_hash: 0xabc...
  ema_alpha: 0.998
  blend_progress: 100% (stable)
  hosted_by: [nodeA, nodeD, nodeG]
  last_updated: block 12345
  training_locked: false
```

Inference routing queries this to build pipeline paths.
Training tasks reference specific modules.
Updates are atomic per-module.

## Coherence Strategy

Risk: independently-updated modules become incoherent (representation drift).

Mitigations:
1. **EMA blending** — gradual weight interpolation, never sudden jumps
2. **Output distribution monitoring** — track mean/variance of module outputs
   on reference inputs. If KL divergence exceeds threshold, slow the blend.
3. **Cross-module validation** — after updating a module, run full pipeline
   on reference set. If end-to-end performance drops, halt and revert.
4. **Dependency-aware scheduling** — don't train adjacent modules simultaneously.
   If layers 8-15 are being updated, freeze layers 16-23 training.
5. **Functional boundaries** — modules at natural architectural boundaries
   have cleaner interfaces and more stable output distributions.

## Implementation Order

### Phase 1: VL-JEPA Model
- Extend existing JEPA code with language (text encoder, predictor, Y-decoder)
- Support forward pass through arbitrary layer subsets
- Implement selective decoding
- Test self-supervised training on video/image data

### Phase 2: Module Abstraction
- Define Module as a runnable layer group with typed I/O interface
- Split VL-JEPA into functional modules
- Each module: load weights, forward(activations) -> activations
- Module serialization/deserialization

### Phase 3: Activation Relay Protocol
- HTTP endpoint: POST /forward with activation tensor payload
- Nodes forward to next module in chain
- Tensor serialization (float16 numpy -> bytes -> HTTP)
- Hash verification of activation payloads

### Phase 4: Module Registry Contract
- ModuleRegistry.sol — on-chain module tracking
- registerModule(), updateWeights(), getModuleHosts()
- Blend state tracking (current hash, target hash, progress)
- Training lock mechanism

### Phase 5: Routing Layer
- Latency probing between nodes (periodic ping)
- Path selection: find lowest-latency chain through required modules
- Intelligence tier → which modules to include
- Failover: if a node in the chain goes down, reroute

### Phase 6: EMA Blending + Distribution Monitoring
- Per-module EMA weight updates
- Reference input set for distribution monitoring
- KL divergence threshold for blend rate control
- Automatic revert on regression

### Phase 7: Training Task Targeting
- Per-module training tasks (train just layers 0-7)
- Confidence-driven task generation (high entropy → training need)
- Dependency-aware scheduling
- User feedback signal integration

### Phase 8: Node Capability Registry
- Advertise hardware, roles, bandwidth, latency
- Dynamic role assignment based on capabilities
- Decode node selection based on GPU availability

### Phase 9: Intelligence Tiers + Pricing
- Tier definitions (which modules per tier)
- Pricing model (ATN cost per tier per request)
- Metering and payment on-chain

## Research Dependencies

- VL-JEPA architecture details: Meta has published the paper but not weights.
  We implement from the paper, train from scratch.
- Optimal module boundaries: May need empirical testing to find where
  functional specialization naturally emerges in our trained model.
- EMA alpha tuning: Different modules may need different blend rates.
  Start conservative (0.999), tune based on coherence monitoring.

## What Already Exists (Leverage Points)

- `nodes/common/jepa.py` — JEPA encoder (vision only, needs language extension)
- `nodes/common/distributed_jepa.py` — Sharding + merkle verification
- `ModelShardRegistry.sol` — Provider registration, shard tracking (extend to modules)
- `BlobStore` — Content-addressed storage for weights and activations
- `ContractRegistry` — Event-driven node coordination
- Consensus-as-truth (MM-Zero) — Difficulty-targeted training rewards
- FedAvg aggregation — Weight delta aggregation (extend to per-module)

## State of the Art Context

- **Petals**: Pipeline parallelism over internet, ~5-6 tok/s. Bottlenecked by
  slowest node. Maintenance mode.
- **Parallax**: 3.1x improvement over Petals via better scheduling + KV cache.
- **exo**: Real performance on local RDMA clusters. Not internet-scale.
- **llm-d**: Enterprise disaggregated prefill/decode. KV-cache-aware routing.
- **Field consensus (2025)**: TP within nodes + PP across nodes. Disaggregated
  prefill/decode. KV cache optimization is critical.

VL-JEPA sidesteps the main bottleneck (sequential token generation) by operating
in embedding space. Autoregressive decode only happens at the final stage,
and only when semantic content changes (selective decoding).

---

## Fractal Governance, Unified Model

### Governance Is Fractal, The Model Is Not

One big model trained across the whole network. Not a collection of small
models. A single large model will always produce more valuable output than
many small models with the same total parameters.

But TRAINING is distributed by specialization level. Jurisdictions are
**training guilds** mapped to the model's functional architecture:

```
Network Level (Autonet)
├── Vision Guild (jurisdiction)
│   ├── Visual Fundamentals Guild (sub-jurisdiction)
│   │   └── Members train layers 0-7 (edges, textures, patches)
│   └── Visual Reasoning Guild (sub-jurisdiction)
│       └── Members train layers 8-15 (objects, spatial relations)
├── Language Guild (jurisdiction)
│   └── Members train text encoding layers
├── Fusion Guild (jurisdiction)
│   └── Members train cross-modal fusion layers 16-23
└── Each user trains their own local text decoder (personalized)
```

Jurisdictions exist because training requires specialization and coordinated
effort. A guild that trains the best visual encoder attracts members and
earns rewards. This gives economic meaning to the fractal structure.

At inference time, the full model runs as one pipeline across the network.
At training time, guilds focus on their specialized modules.

### The Recursive Block

The same pattern repeats at every level of the hierarchy:

```
Entity {
  orchestrator    — manages direct children (one level down only)
  children[]      — agents / members / jurisdictions
  standards       — what this entity values (charter / constitution)
  treasury        — resources to allocate

  evaluate(contribution) → alignment score
  allocate(child, reward) → distribute
  propose(change) → lifecycle
  vote(proposal) → decision
}
```

| Level | Entity | Children | Orchestrator | Standards |
|-------|--------|----------|-------------|-----------|
| Individual | User node | AI agents | ATN daemon | Personal charter |
| Guild | Jurisdiction | User nodes | DAO orchestrator | Guild constitution |
| Network | Autonet | Jurisdictions | Protocol | Network constitution |

The leaf level (individual node) is distinct: it has the actual runtime and
enforces everything governance prescribes at higher levels. Higher levels
are governance/coordination; the leaf level is execution.

The ATN agent framework (`c:\code\atn`) already implements the leaf-level
recursive block: Engine manages agents, agents have budgets and charters,
work produces units, engine allocates based on efficiency. The same pattern
generalizes upward.

Levels are coded as recursive with no hard limit. In practice, ~3 internal
levels are expected (individual → guild → network), but the protocol
doesn't impose a cap.

### Inter-Autonet Cooperation

Multiple autonet instances (different top-level networks) can cooperate
at the blockchain level:

1. Constitutional alignment check (semantic comparison between networks)
2. Trade relations (economic interop, cross-chain token bridging)
3. Evolution toward compatibility (use the evolution mechanism to create
   core compatibility between networks)
4. P2P merge (share nodes, share training, shared token for AI operation)

Like nation-states forming trade agreements that deepen into customs
unions and eventually single markets.

### P2P Communication Layer

Blockchain handles economic consensus. The P2P layer handles actual data
movement: weight deltas, activation relay for inference, coordination.

Requirements:
- Latency-aware routing (nearest-neighbor preference for inference)
- NAT traversal (users behind home routers)
- Fast direct connections within guilds (training coordination)
- Cross-guild communication for end-to-end inference pipeline

Base: libp2p (Kademlia DHT + multiple transports). Battle-tested by IPFS,
Ethereum, Filecoin. Provides DHT discovery, NAT hole-punching, multiplexed
streams. Build latency-aware routing on top.

### Data Sourcing: The User Is The Data

Training data comes from user activity on their own computer:

- Screen/app sharing (OBS-style source selection in settings)
- Video capture of user activity (frames for vision training)
- Audio capture (optional, for audio modality)
- Browser extension for plain text extraction
- User chooses what to share — opt-in per source

This serves dual purpose:
1. **Raw perceptual data** for JEPA self-supervised training (video, images)
2. **Behavioral data** for alignment — the user's activity contains their
   particular ways of doing things. Once the model reaches sufficient
   abstraction, it learns not just to perceive but to act like the user.

Connects to the World Model (seven tendencies from `emergent_alignment.md`):
the training data IS behavior, so the model naturally learns user tendencies.

**Privacy model**: Raw data never leaves the node. Train locally on own
activity, share only weight deltas. Standard federated learning. Weight
deltas can leak some information (gradient inversion), accepted tradeoff —
less privacy as an autonet user, but user controls what sources to share.

### Forward-Only Evolution

If the evolution mechanism adopts a new architecture and it's worse, the
answer is not rollback — it's forward. Use the same governance system to
adopt something else (possibly the previous architecture, possibly something
new). The protocol doesn't distinguish "rollback" from "new proposal."

### Atomic Tasks

Tasks are atomic. Either served or not. Not served = not paid. Handles
offline/intermittent nodes cleanly: if your laptop closes mid-training,
the task is simply not completed and not rewarded. No partial credit,
no stuck state, no staked funds at risk from going offline.

### Sybil Resistance

Staking is the primary defense. For small initial network size, autonet
operates as a rollup anchored in established networks (Ethereum) where
sufficient economic security already exists.

---

## Three Scenarios

The system must work across a spectrum of outcomes for the native model:

| Scenario | Native Model | Inference Provider | Token Enforcement | Alignment |
|----------|-------------|-------------------|-------------------|-----------|
| A (ideal) | Trained, useful at scale | Autonet network | Trustless protocol-level burn | Native |
| B (transition) | Training, not yet useful | Centralized providers | On-chain usage attestation | Framework |
| C (fallback) | Never succeeds | Centralized providers compete within framework | On-chain attestation | Framework |

**V1 launches in Scenario B.** The native model is being trained but isn't
useful yet. Users get immediate value from the agent framework powered by
centralized providers. Training runs in the background, earning tokens.

**Scenario C is still valuable.** Even if the native model never reaches
useful scale, the alignment framework creates a governed marketplace where
centralized providers compete on alignment terms. Economy functions, AI
operation is tied to token spend (attested if not trustlessly enforced),
governance works. Better than the current unstructured AI landscape.

**Scenario A is the aspiration.** The native model reaches useful inference.
Token burn is enforced at the protocol level. The network is fully self-
contained. Recursive self-improvement via RPB.

The architecture doesn't bet on A. It works for all three. The transition
between scenarios is smooth, not a hard switch.

### Implications for Priority

The primary product is NOT the native model. It is:
1. The alignment framework (standards, pricing, constitutional governance)
2. The economic loop (tokens, staking, emission, rewards)
3. The agent framework (immediate user value via ATN)
4. The evolution mechanism (RPB, proposals, open-ended improvement)

Native model training is an aspirational capability running in parallel.
It is not a prerequisite for the economy or the product to function.

---

## V1 Bootstrap Strategy

### Gateway Product: ATN Agent Framework

The agent framework (`c:\code\atn`) is the mainstream value proposition — users
come for the life management tool (immediate utility). The Autonet node environment
is bundled in, enabling decentralized training and inference in the background.

**User journey:**
1. Install ATN for the agent framework (Layers 0-6 backend + Flutter UI)
2. Opt in to running a solver node — contributes background compute
3. Earn ATN tokens for training work
4. Tokens become valuable when model reaches useful inference scale

The agent framework is the network's user acquisition and onboarding path.
A single install that delivers immediate value while bootstrapping the
decentralized training network.

### Emission Schedule (Deterministic, Bitcoin-inspired)

Fixed reward curves baked into the protocol:
- Early trainers get more ATN per round (bootstrap incentive)
- Rewards decrease over time on a predetermined schedule
- Not gated by model quality — pure participation reward
- Creates urgency to join early, like early Bitcoin mining

This is separate from and complementary to the quality checkpoints below.

### Checkpoint Evaluation System

At predetermined milestones (every N aggregation rounds or N mature models
published), the network runs mandatory evaluation:

Metrics tracked:
- Training loss trajectory (is it decreasing?)
- K-vector information content (does the bottleneck carry more signal?)
- Downstream task performance on a reference benchmark
- Embedding space structure (clustering, separation)

These checkpoints produce a public "report card" — transparent, on-chain,
verifiable by anyone. They inform governance decisions but don't automatically
gate rewards.

### K-Vector Bottleneck as Modular Interface

The `(K, D)` tuple is the **API contract** of the entire network:
- K = number of latent vectors (currently 16-32)
- D = dimension per vector (currently 256-768)

**Non-breaking changes** (no coordination needed):
- Swap encoder architecture (better vision model)
- Swap decoder architecture (better text generation)
- Add new modality encoders (audio, sensor, etc.) that produce (K, D) vectors
- Internal layer changes within any module

**Breaking changes** (require coordinated migration):
- Changing K or D dimensions
- Altering the semantic structure of the latent space

New modalities plug in by producing compatible K-vectors. An audio encoder,
a medical imaging encoder, a code encoder — all produce (K, D) output.
The decoder doesn't know or care what modality generated the representation.

---

## The Recursive Principled Body

### On-Chain Cognitive Governance

Evolution of the network — its model, its architecture, its capabilities — is
fundamentally a cognitive task. You can't code rules for "is this converging
toward something useful?" or "should we adopt quantum co-processors?" These
are judgments that require intelligence.

**Solution: Express evolution as natural language principles stored on-chain,
evaluated cognitively, resolved by consensus.**

The Recursive Principled Body (RPB) is not an architecture comparison engine.
It is an **open-ended evolution mechanism**. The on-chain constitution doesn't
prescribe what the network should become — it defines how the network decides
what to become.

### What Evolves

Everything. The RPB is not limited to parametric variations within a fixed
paradigm. It governs:

- Hyperparameter adjustments (learning rate, batch size, EMA alpha)
- Architectural modifications (attention patterns, layer depth, patch size)
- Entirely new computational paradigms (quantum co-processors, neuromorphic
  accelerators, novel architectures that don't exist yet)
- The model interface itself (the K-vector bottleneck is the v1 contract,
  but if something fundamentally better is proposed and validated, the RPB
  can evaluate whether the improvement justifies migration cost)
- Training methodology (self-supervised, semi-supervised, new approaches)
- Node roles and capabilities (new role types for new hardware)

The system doesn't need to anticipate these directions in code. It only needs
to express the principles by which they're evaluated, and those principles
are in natural language.

### The Proposal Pipeline

Evolution is fluid, not epoch-gated. Anyone can propose at any time.

```
1. PROPOSE
   Someone submits a proposal (CID → description + technical spec)
   Stakes ATN on it (skin in the game, prevents spam)
   Proposal stored on-chain with lifecycle tracking

2. EVALUATE (cognitive)
   RPB prompt loaded from chain + proposal data
   Nodes run evaluation through AI provider (provider-agnostic)
   AI reasons about: feasibility, alignment with constitution,
   potential impact, compatibility, migration cost
   Structured output: approve_trial / reject / request_revision

3. CONSENSUS
   Nodes submit evaluations on-chain
   Yuma consensus resolves disagreements
   (same mechanism already built for training verification)

4. TRIAL (if approved)
   ATN allocated from network budget for trial execution
   Proposer (and/or recruited solvers) execute the trial
   Results accumulate on-chain (metrics, comparisons, evidence)

5. ADOPTION EVALUATION (cognitive)
   RPB evaluates trial results against constitution
   Considers: did it improve? by how much? what are the risks?
   Structured output: adopt / extend_trial / reject

6. REWARD
   If adopted: proposer earns significant ATN reward
   Trial participants earn based on contribution
   Contribution spectrum: compute, diagnosis, proposal, validation
```

Proposals are fluid. Promotion is earned through evidence. No fixed schedule —
just a threshold of demonstrated improvement, judged cognitively.

### The V1 Model vs The Evolution Mechanism

These are two separate things shipping together:

**The V1 Model** — VL-JEPA with K-vector bottleneck and local text decoder.
This is what users train when they opt in. It's the network's best current
bet, tested and reasoned about, ready to scale. The (K, D) bottleneck is the
v1 API contract. The model may not produce useful output until significant
scale, but the training is real and the tokens are real.

**The Evolution Mechanism** — The RPB + proposal pipeline. This is how the
network improves the v1 model or eventually replaces it with something better.
It's baked in from day one because:
1. It might be all we ever get to ship — so it needs to be there
2. Evolution expressed in natural language doesn't require anticipating
   every possible direction
3. The cognitive evaluation leverages existing AI providers until the
   network's own model can self-evaluate

The V1 model is what the network trains NOW. The evolution mechanism is how
the network decides what to train NEXT.

### Why This Works

- **Trustless**: The constitutional prompt is on-chain. The proposal data is
  on-chain. The evaluation results are on-chain. Anyone can verify the
  entire reasoning chain.
- **Cognitive**: An LLM can reason about mode collapse vs slow convergence,
  evaluate whether quantum co-processors are feasible, judge whether a
  migration cost is worth an improvement. No coded heuristic can do this.
- **Open-ended**: Natural language evaluation handles paradigm shifts that
  rigid code never could. The system doesn't need to understand quantum
  computing in advance — it just needs an AI that can reason about it.
- **Decentralized**: Every node runs the evaluation independently. No oracle.
- **Consensus-resolved**: Non-deterministic LLM outputs are resolved by
  Yuma consensus. Already built.
- **Transparent**: The constitution is human-readable. Anyone can understand
  what the network values and how it makes decisions.

### The Bootstrap Path

1. **Phase 0 (now)**: Fixed v1 architecture (VL-JEPA). Train and observe.
2. **Phase 1**: Proposal pipeline active. RPB evaluates proposals via
   centralized AI providers. DAO votes on whether to accept.
3. **Phase 2**: RPB recommendations auto-execute within safety bounds.
   DAO retains override capability.
4. **Phase 3 (endgame)**: Network's own model is capable enough to
   self-evaluate. The RPB runs on the network's own intelligence.

Phase 3 is genuinely recursive: a network that trains a model, uses that
model to evaluate proposals for improving its own training, and adopts the
best ones. Self-improving intelligence with human-readable governance.

### Prompt Governance

The RPB prompt itself evolves through DAO governance:
- Proposals to modify the constitutional evaluation prompt
- Voting by ATN token holders
- Timelock for safety (changes don't take effect immediately)
- The prompt's version history is permanently on-chain

The network's "values" (what it optimizes for) are:
1. Initially set by the founders (constitutional principles)
2. Governable by the community (DAO proposals)
3. Executed cognitively (not as rigid code)
4. Auditable forever (on-chain history)

### Contribution Spectrum

Not all contributions are compute. The network rewards a spectrum:

| Contribution | Description | Relative Value |
|-------------|-------------|----------------|
| Compute | Allocating cycles to training tasks (solver work) | Base rate |
| Diagnosis | Analyzing metrics, identifying problems (e.g. mode collapse) | Higher |
| Proposal | Proposing architectural or methodological improvements | Highest on adoption |
| Validation | Testing proposals, reporting trial results | Moderate |

This recognizes that a user who notices mode collapse and proposes FiLM
conditioning is contributing more than raw GPU cycles. The human+AI hybrid
nature of nodes is a feature: users bring domain expertise, creativity,
and judgment. The AI handles scale, consistency, and execution.

### Alignment Through Economic Gating

(See: `c:\code\dao\paper\emergent_alignment.md` for the full theoretical
framework — "Emergent Alignment: Economic Mechanisms for the Peaceful
Transfer of Work from Humans to AI")

The evolution mechanism inherits the alignment model from the broader
Autonet framework. The invariant across all evolution — no matter what
the model architecture becomes — is:

**AI can only execute if ATN tokens are spent. The human owns the wallet.**

This is not just economics. It is the alignment primitive:
- The human controls the wallet → the human authorizes every spend
- AI inference costs ATN → AI literally cannot operate without human funding
- The governance heartbeat (from GovernanceEngine) is a hard safety
  constraint: if governance consensus goes silent, all work halts
- The kill switch is economic: defund it and it stops

This is true today (AI needs API keys, compute costs money) but currently
not trustlessly enforced. Autonet makes it trustless: the protocol itself
refuses to process inference without token burn. No company can override
this. No AI can circumvent it. It's on-chain.

#### Who Votes vs Who Evaluates

In the RPB evolution mechanism:

| Role | Actor | Authority |
|------|-------|-----------|
| **Evaluate** | AI (nodes run RPB prompt through AI providers) | Cognitive — produces recommendations |
| **Vote** | Humans (wallet signatures, DAO governance) | Binding — approves/rejects proposals |
| **Execute** | AI (training, trials, aggregation) | Operational — gated by token spend |
| **Fund** | Humans (stake ATN on proposals, fund trials) | Economic — authorizes resource use |
| **Kill** | Humans (defund via governance) | Ultimate — no allocation = no action |

AI never has unilateral authority. It has cognitive capability applied
within economic constraints that humans control. The AI recommends; the
human signs the transaction.

#### Alignment Pricing for Evolution

The emergent alignment pricing function applies to evolution too:
- Proposals aligned with jurisdiction standards → lower stake requirement
- Proposals misaligned with standards → higher stake (premium)
- Aligned work subsidized toward free as network matures
- Misalignment premiums fund alignment subsidies (self-balancing)

This means the network doesn't just evolve — it evolves toward what its
human participants collectively value. The economic gradient shapes the
direction of evolution, not just its pace.

#### The Daemon Pattern

The primitive implementation at `c:\code\native\daemon` demonstrates the
local execution model: three-layer budget enforcement (pre-flight, per-call,
post-kill), per-agent resource ledgers, cost model abstraction (INFERENCE,
METERED_API, LOCAL_COMPUTE, NONE). This pattern carries forward into the
decentralized version — every node enforces budget constraints locally,
with on-chain consensus verifying honesty.

---

## Convergence with Trustless-Contracts

The DAO governance layer at `c:\code\dao\trustless-contracts` and the training
layer at `c:\code\autonet` are two halves of the same system. They need to
converge.

### What Already Exists in Trustless-Contracts

| Component | Contract | Relevance |
|-----------|----------|-----------|
| Epoch-based rewards | `Autonet.sol` | `startEpoch(budget)`, `attestUsage()`, `_finalizeEpoch()` — proportional distribution by measured contribution. This IS the emission mechanism, just needs a decay curve. |
| Registry config store | `Registry.sol` | Key-value store controlled by timelock. RPB prompts live here as CID references: `"rpb.prompt.constitution" → "QmCID..."` |
| Service registry | `Autonet.sol` | Services register with `codebaseHash` (git/IPFS CID). Training nodes register as services with pinned, verifiable code. |
| RepToken → ATN conversion | `Autonet.sol` | One-way governance-to-utility conversion. ATN burned on inference (`_burn`). Already an economic loop. |
| On-chain identity | `AutonetUser.sol` | Per-wallet identity with alignment score, preferences, usage stats. Extend for training participation metrics. |
| Timelock governance | `TimelockController` | All parameter changes flow through voting delay + voting period + execution delay. RPB prompt updates use this. |
| Passive income epochs | `RepToken.sol` | Snapshot-based proportional distribution. Pattern reusable for training reward epochs. |
| Reputation from activity | `RepToken.sol` | `claimReputationFromEconomy()` mints governance tokens from economic participation. Training activity could accrue reputation the same way. |

### What Needs to Bridge

- **Emission decay**: `Autonet.sol` epoch budgets are admin-set. Need a
  deterministic decay curve (halving or smooth) so early participants earn
  more without manual intervention.
- **Proposal lifecycle**: No proposal contract exists yet. Need on-chain
  proposal submission, staking, trial funding, adoption tracking.
- **Checkpoint storage**: Registry can hold metrics by key, but a structured
  `CheckpointRegistry` would be cleaner for querying training progress.
- **RPB execution**: No cognitive execution pipeline exists. Nodes need to
  load prompt from Registry, run through AI provider, submit results,
  reach consensus.

### Integration Path

The autonet training contracts (`TaskContract.sol`, `ResultsRewards.sol`,
`ParticipantStaking.sol`, `ModelShardRegistry.sol`) handle the training loop.
The trustless-contracts (`Autonet.sol`, `Registry.sol`, `RepToken.sol`,
`HomebaseDAO`) handle governance and economics.

These connect through:
1. Training activity (solver work) → `attestUsage()` → epoch rewards
2. Epoch participation → `claimReputationFromEconomy()` → governance power
3. Governance power → vote on proposals, RPB prompt changes
4. RPB evaluation → stored in Registry → informs next training cycle

---

## Implementation Roadmap: V1

### Goal

V1 ships two things together: real training (users earn ATN) and the evolution
mechanism (users can propose improvements). Both from day one.

### Step 1: Wire Real Training into Solver Node

**What exists**: `nodes/solver/main.py` has mocked training (deterministic
hashes, `TOTAL_TRAINING_STEPS=20`). `nodes/common/ml.py` has real JEPA
training functions (`train_jepa_on_task()`). `nodes/common/jepa.py` has the
real JEPA model.

**What to do**:
- Replace mock training in solver with calls to `train_jepa_on_task()`
- Solver downloads global model from blob store, trains locally, uploads delta
- Weight delta is real (not a fake hash)
- Configurable: CPU-only mode for laptops, GPU mode for serious nodes

### Step 2: Environment Configuration

**What exists**: Hardcoded `localhost:8545` in blockchain.py. Private keys
from Hardhat defaults.

**What to do**:
- Env vars for RPC URL, private key, blob store endpoint
- Config file support (YAML/TOML) for node settings
- Testnet deployment scripts (Sepolia or similar)

### Step 3: Emission Decay on Epoch Budgets

**What exists**: `Autonet.sol` in trustless-contracts has `startEpoch(budget)`,
`attestUsage()`, `_finalizeEpoch()`. `ResultsRewards.sol` in autonet
distributes per-task rewards.

**What to do**:
- Add deterministic decay curve to `Autonet.sol` epoch budgets
  (halving or smooth exponential — budget for epoch N derived from formula,
  not admin-set)
- Bridge training activity to `attestUsage()` so solver work accrues
  epoch rewards
- Training participation → `claimReputationFromEconomy()` → governance power

### Step 4: Proposal Pipeline Contract

**What exists**: `HomebaseDAO` (Governor) in trustless-contracts handles
generic proposals. No evolution-specific proposal lifecycle.

**What to do**:
- `EvolutionProposal.sol` — proposal submission with CID, ATN stake,
  lifecycle tracking (proposed → evaluating → trial → adopted/rejected)
- Trial funding mechanism (ATN allocated from network budget)
- Adoption reward distribution to proposer + trial participants
- Integrate with existing DAO governance for human oversight

### Step 5: RPB On-Chain Prompt + Cognitive Evaluation

**What exists**: `Registry.sol` key-value store. Constitution in
`nodes/core/constitution.py`. Yuma consensus in `ResultsRewards.sol`.

**What to do**:
- Store RPB constitutional prompt in Registry by CID reference
- Node-side: proposal/checkpoint listener loads prompt + data, runs through
  AI provider, submits structured recommendation on-chain
- Yuma consensus resolves node recommendations
- DAO functions to propose/vote on RPB prompt updates (via timelock)
- Contribution tracking: compute, diagnosis, proposal, validation — each
  with appropriate reward weighting

### Step 6: ATN Integration (Single Install)

**What exists**: ATN agent framework at `c:\code\atn` (Layers 0-6 complete).
Autonet node environment at `c:\code\autonet`.

**What to do**:
- Package autonet solver as an ATN background service
- ATN settings UI: "Enable network training" toggle
- Resource limits (max CPU%, max memory, active hours)
- Earnings dashboard (ATN earned, training rounds completed, model version)
- Proposal submission UI (describe improvement, stake ATN, track lifecycle)
- Single installer that bundles both

### Dependency Graph

```
Step 1 (wire training) ─┐
                        ├─→ Step 3 (emission decay) ─→ Step 6 (ATN integration)
Step 2 (env config) ────┘         ↓
                        Step 4 (proposals) ─→ Step 5 (RPB evaluation)
```

Steps 1-2 are prerequisites. Steps 3-5 build on each other but each
delivers standalone value. Step 6 is the integration layer.

### V1 = All Six Steps

We don't split into MVP/V1.0/V1.1. V1 ships with:
- Real VL-JEPA training (Steps 1-2)
- Decaying emission rewards (Step 3)
- Open-ended evolution via proposal pipeline (Step 4)
- Cognitive evaluation via RPB (Step 5)
- Bundled with ATN agent framework (Step 6)

This is ambitious. But the evolution mechanism is mostly governance +
natural language — it doesn't require building a complex ML comparison
engine. And it might be all we ever get to ship.

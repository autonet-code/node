# Training Data Pipeline: Agent Framework → VL-JEPA

## Architecture Analysis

### The Latency Symmetry

The same architectural property serves both directions:

```
INFERENCE:  network encoders → K-vectors (one round-trip) → LOCAL DECODER (autoregressive)
TRAINING:   LOCAL ENCODER (backprop on user data) → weight deltas (one round-trip) → NETWORK AGGREGATION
```

Autoregressive decoding can't be distributed (N tokens = N round-trips at network
latency). Backprop through encoder layers can't be distributed either (layer N
depends on layer N-1 activations). JEPA makes both local:

- Inference: only the encoder pipeline runs on the network; the decoder is local
- Training: only the weight deltas cross the network; backprop is local

### Three Data Streams, Two Training Paths

User activity produces three streams of data:

| Stream | Source | Format | Volume |
|--------|--------|--------|--------|
| **Text** | Agent conversations, tool calls, execution records | JSONL (role, content, timestamp) | High — every interaction |
| **Visual** | Screen capture, browser frames | PNG/JPEG → [B,3,H,W] float32 | Medium — opt-in, 2-5 fps |
| **Audio** | Voice input (STT), ambient (future) | int16 numpy @ 16kHz | Low — opt-in, push-to-talk |

These map to VL-JEPA's architecture, which already has:

- `VisionTransformerEncoder` — processes image patches (2D positional embeddings)
- `TextEncoder` — processes token sequences (1D positional embeddings)
- `CrossModalFusion` — cross-attention between visual and text
- `SemanticPredictor` — produces K-vectors from fused representation
- `TextDecoder` — local autoregressive generation from K-vectors

**The question is whether these need separate training loops or one unified loop.**

### Answer: One Loop, Multiple Data Configurations

They do NOT need separate loops. VL-JEPA's `self_supervised_loss()` already handles
the case where both modalities are present. The missing piece is handling the case
where only one modality is available in a given batch:

**Text-only batches** (agent framework output):
- `images` → zero tensor (or learned "no-image" embedding)
- `token_ids` → conversation text, byte-tokenized
- Loss: predict masked text embeddings from context (text JEPA)
- The cross-modal fusion degrades gracefully: text attends to zeros, effectively
  becoming self-attention on text

**Visual-only batches** (screen capture, browser frames):
- `images` → captured frames, preprocessed [B,3,224,224]
- `token_ids` → zero/padding (or learned "no-text" embedding)
- Loss: predict masked visual patches from context (I-JEPA, already working)

**Multimodal batches** (user looking at screen while talking to agent):
- `images` → concurrent screen capture
- `token_ids` → concurrent conversation text
- Loss: predict masked patches AND masked tokens from cross-modal context
- This is the richest signal — model learns to relate visual context to actions

This is elegant because:
1. The solver doesn't need different code paths per modality
2. The proposer just specifies which data sources are included in a task
3. Verification works the same way (cosine similarity of K-vectors)
4. FedAvg works the same way (average weight deltas)
5. The model naturally learns cross-modal associations when both are present

### What the Model Learns from Agent Framework Text

The self-supervised objective for text JEPA:

```
Input:  [user: "check my calendar for tomorrow"] [assistant: "I'll use..."] [MASK] [MASK] [tool: calendar_read] [result: "3 meetings"]
Target: predict embeddings for masked tokens from visible context
```

This teaches the model:
- **Action prediction**: given user intent + context, predict what tool/action comes next
- **State understanding**: given partial conversation, reconstruct the full situation
- **Behavioral patterns**: correlations between user phrasing and agent responses

Unlike a standard language model (predict next token), JEPA predicts in **embedding
space**. The model learns abstract representations of "what happens next" rather than
specific token sequences. This is more robust to paraphrasing and generalizes better
across users.

### Text Masking Strategy

Visual JEPA masks rectangular blocks of patches (spatial locality). Text needs an
analogous strategy that respects conversational structure:

**Option A — Span masking (like SpanBERT)**:
- Mask contiguous spans of 3-15 tokens
- Preserves local context, tests global understanding
- Simple, well-studied

**Option B — Turn masking (conversation-aware)**:
- Mask entire conversation turns (user or assistant)
- Forces cross-turn prediction (given user question, predict assistant action)
- More aligned with behavioral learning

**Option C — Role masking (structural)**:
- Mask all tool calls, or all results, or all assistant text
- Forces the model to predict actions from intent (or intent from actions)
- Most aligned with the training goal

**Recommendation**: Start with span masking (simpler, well-understood), add turn
masking as a training curriculum once the pipeline is working.

### Audio: Not a Separate Path

Audio from voice interactions is already converted to text via STT (faster-whisper)
before it enters the agent framework. The transcript is tagged `"🎤 [Voice Input]"`
and injected as a regular conversation turn. So:

- Voice text already flows through the text pipeline — no separate audio training
  needed initially
- Raw audio (waveform) could eventually feed a future audio encoder, but this is
  Phase 2 at earliest — the text transcription captures the semantic content

### Visual: Parallel Stream, Same Loop

Screen capture and browser frames are already preprocessed by `FramePreprocessor`
into `[B,3,224,224]` tensors normalized with ImageNet stats. The existing I-JEPA
training path in `train_jepa_on_task()` handles this — it just needs a data source
swap from CIFAR/FakeData to the actual capture buffer.

When both text and visual data are available simultaneously (user working with agent
while screen is captured), they should be paired into multimodal batches. The
`CrossModalFusion` module handles the rest.

---

## Implementation Backlog

### Epic 1: Text Training Data Adapter

**Goal**: Agent framework JSONL → VL-JEPA text encoder training batches

#### Story 1.1: TextTrainingDataSource

Create `nodes/common/text_data.py`:
- Read conversation JSONL from `~/.atn/conversations/`
- Read execution records from `~/.atn/agents/*/execution.jsonl`
- Byte-level tokenization (matching VLJEPAConfig.vocab_size=260)
- Sliding window chunking to `max_seq_length` (512 default, should be configurable)
- Produces `(token_ids: Tensor[B, S], attention_mask: Tensor[B, S])`
- Privacy: respect exclusion patterns from autonet.yaml privacy config
- Consent: only read data when user has opted into training

**Interface**:
```python
class TextTrainingDataSource:
    def __init__(self, data_dir: Path, config: VLJEPAConfig, privacy_config):
        ...
    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        # yields {"token_ids": [B, S], "attention_mask": [B, S]}
    def __len__(self) -> int:
        ...
```

**Acceptance criteria**:
- Reads real ATN conversation data
- Byte tokenization matches VLJEPAConfig vocab (4 special + 256 bytes)
- Handles empty/corrupt JSONL gracefully
- Respects privacy exclusion list
- Unit tests with synthetic JSONL

#### Story 1.2: Text Masking Strategy

Add `TextMasker` to `nodes/common/jepa.py` (alongside `JEPAMasker`):
- Span masking: randomly select contiguous spans of 3-15 tokens to mask
- ~15-25% of tokens masked per sequence (configurable)
- Returns `(context_mask, target_masks)` same interface as `JEPAMasker`
- 1D positions instead of 2D grid (text is sequential, not spatial)

**Acceptance criteria**:
- Masks produce valid context/target splits
- Masked ratio stays within configured bounds
- Works with variable-length sequences (attention_mask respected)
- Unit tests

#### Story 1.3: TextEncoder JEPA Training Mode

Extend the JEPA training path to handle text input:
- `JEPAConfig(modality="text")` → use `TextEncoder` instead of `VisionTransformerEncoder`
- `TextEncoder` needs masking support (analogous to VisionTransformerEncoder's mask param)
- Token embedding + 1D positional embedding + mask → context embeddings
- Predictor gets context embeddings + target indices → predicted target embeddings
- Target encoder (EMA) provides supervision signal
- Loss: smooth L1 in embedding space (same as visual JEPA)

The TextEncoder in `vl_jepa.py` already has the right structure but lacks:
- Mask parameter support (only processes full sequences currently)
- Integration with JEPAPredictor (needs 1D positional embeddings in predictor)

**Acceptance criteria**:
- Text-only JEPA training runs end-to-end
- Loss decreases over epochs on held-out text
- Cosine similarity metric works for text embeddings
- Weight deltas are compatible with FedAvg (same dict structure)

---

### Epic 2: Unified VL-JEPA Training Loop

**Goal**: Single training function handles text-only, visual-only, and multimodal batches

#### Story 2.1: Multimodal Data Loader

Create `nodes/common/multimodal_data.py`:
- Combines `TextTrainingDataSource` with visual capture data
- Time-aligns text and visual data when both are available
- Produces batches that may have:
  - Both modalities (text + concurrent screen frame)
  - Text only (agent interaction without screen capture)
  - Visual only (screen capture without concurrent agent interaction)
- Missing modality → zero tensor with a modality-present flag

**Interface**:
```python
class MultimodalDataLoader:
    def __init__(self, text_source, visual_source, config):
        ...
    def __iter__(self) -> Iterator[dict]:
        # yields {
        #   "images": [B, 3, H, W] or zeros,
        #   "token_ids": [B, S] or zeros,
        #   "attention_mask": [B, S],
        #   "has_visual": [B] bool,
        #   "has_text": [B] bool,
        # }
```

**Acceptance criteria**:
- Correctly pairs temporally-aligned text+visual data
- Fills zeros for missing modality
- Shuffles across modality combinations
- Handles unbalanced data (text >> visual or vice versa)

#### Story 2.2: train_vljepa_on_task()

New training function in `nodes/common/ml.py` (alongside `train_jepa_on_task()`):
- Uses VL-JEPA model instead of vision-only JEPA
- Accepts multimodal batches from MultimodalDataLoader
- Training step:
  1. Text masking (TextMasker) + visual masking (JEPAMasker)
  2. Encode context through text encoder + visual encoder
  3. Cross-modal fusion on visible context
  4. Predict target embeddings (both text and visual targets)
  5. Loss against target encoder outputs
  6. Backprop through context encoders + predictor + fusion
  7. EMA update target encoder
- Returns weight deltas compatible with FedAvg
- Falls back gracefully when only one modality is present

**Acceptance criteria**:
- Trains on text-only batches (visual = zeros)
- Trains on visual-only batches (text = zeros)
- Trains on multimodal batches
- Loss decreases on all three batch types
- Weight deltas are FedAvg-compatible
- Metrics include per-modality cosine similarity

#### Story 2.3: Verification for Text/Multimodal

Extend `verify_jepa_solution()` in `ml.py`:
- Accept text and/or visual validation data
- Compute cosine similarity in shared embedding space
- Verification threshold applies to K-vector similarity regardless of source modality
- Coordinators don't need to know which modalities were used

**Acceptance criteria**:
- Verification works for text-only, visual-only, and multimodal solutions
- Same cosine similarity threshold applies uniformly
- Compatible with existing Yuma consensus voting

---

### Epic 3: Wire Agent Framework to Training Pipeline

**Goal**: When user opts in, agent activity feeds local JEPA training in real time

#### Story 3.1: Training Data Consent & Opt-In

Add consent mechanism to ATN:
- New config field: `autonet.train_on_agent_data: bool` (default false)
- UI toggle in network page (alongside existing training switch)
- Separate from screen capture opt-in (user may share agent data but not screen)
- When enabled, ConversationStore and ExecutionLog emit events that the
  training data adapter subscribes to

**Acceptance criteria**:
- Toggle persists in config.yaml
- Training only reads agent data when explicitly enabled
- Can be toggled independently of screen/browser capture
- Clear UI indication of what data is being used

#### Story 3.2: Real-Time Training Data Feed

Bridge ATN event bus to training data pipeline:
- Subscribe to `EXECUTION_COMPLETED` events
- Subscribe to `STEP_COMPLETED` events (for cognitive step outputs)
- Extract conversation text + tool call data
- Buffer into training batches (accumulate N interactions before training step)
- Feed to `TextTrainingDataSource` ring buffer

**Acceptance criteria**:
- New agent interactions appear in training data within one batch cycle
- Buffering prevents training on every single message (wasteful)
- Old data ages out (ring buffer or time window)
- Works with training service running or stopped (buffer persists)

#### Story 3.3: Visual Capture → Training Pipeline

Wire existing capture infrastructure to VL-JEPA training:
- `FramePreprocessor` already produces `[B, 3, 224, 224]` tensors
- Route preprocessed frames to `MultimodalDataLoader`
- Time-align with concurrent conversation data
- Respect fps_cap and resolution settings from capture config

**Acceptance criteria**:
- Screen frames flow into visual training batches
- Browser relay frames flow into visual training batches
- Time alignment with conversation data works (±2 second window)
- fps_cap respected (no training on more frames than configured)

#### Story 3.4: Solver Integration

Update solver node to use new training functions:
- Task spec includes `modalities: ["text", "visual"]` field
- Solver calls `train_vljepa_on_task()` instead of `train_jepa_on_task()` when
  VL-JEPA config is provided
- Data comes from local agent framework (not CIFAR/FakeData)
- Weight deltas flow through existing commit-reveal pipeline

**Acceptance criteria**:
- Solver trains VL-JEPA on real agent data
- Commit-reveal protocol works with VL-JEPA weight deltas
- Coordinator verification works on VL-JEPA outputs
- Aggregator FedAvg works on VL-JEPA weight deltas
- Full loop: train → commit → verify → reward → aggregate → publish

---

### Epic 4: Economic Gating

**Goal**: Training is gated on wallet + stake + jurisdiction membership

#### Story 4.1: Gate Training on Wallet

Modify `AutonetBridge.start()`:
- Require `wallet_connected == True` before starting training
- Require valid RPC connection to chain
- Check wallet has sufficient ATN for solver stake (50 ATN)
- If not staked, call `ParticipantStaking.stake()` before starting
- Return clear error messages when prerequisites aren't met

**Acceptance criteria**:
- Can't start training without wallet
- Can't start training without stake
- Auto-stakes if wallet has sufficient balance
- Clear error if balance insufficient

#### Story 4.2: Attestation on Agent-Data Training

Wire training completions to epoch attestation:
- After each successful training cycle, call `attestUsage(serviceId, units)`
- Units = number of training steps completed (not number of conversations read)
- Service ID = registered training service for user's jurisdiction
- Attestation flows to `Autonet.sol` epoch tracking

**Acceptance criteria**:
- Training cycles produce on-chain attestation
- Attestation count matches actual training work
- Epoch rewards claimable after attestation
- Works with existing emission schedule

#### Story 4.3: Jurisdiction Join Flow

Add jurisdiction discovery and joining:
- Query `GuildRegistry.sol` for available jurisdictions
- Show jurisdictions in UI with their specialization (text, visual, multimodal)
- User selects jurisdiction → joins via contract call
- Jurisdiction membership determines which guild aggregates your deltas

**Acceptance criteria**:
- UI shows available jurisdictions
- User can join a jurisdiction
- Training tasks are scoped to jurisdiction
- Aggregation happens within jurisdiction first, then cross-jurisdiction

---

### Epic 5: Alignment-Based Inference Pricing

**Goal**: Inference pricing is driven by demonstrated behavioral alignment over
time, not point-in-time configuration. Nodes that consistently do aligned work
get cheaper (potentially free) inference. Misaligned work pays a premium that
funds the aligned subsidy. Pricing can't exist until inference exists — during
bootstrap, nodes accumulate behavioral profiles that will determine their
pricing tier when inference activates.

#### Story 5.1: Behavioral Semantic Profile Accumulation

Build the alignment track record during training, before inference exists:
- After each training cycle, compute mean-pooled K-vectors from the node's
  agent interaction embeddings (the text/multimodal data that was just trained on)
- Update a local **behavioral EMA** (exponential moving average):
  ```
  profile_t = decay * profile_{t-1} + (1 - decay) * current_embeddings
  ```
  With `decay = 0.998` and daily updates, it takes ~500 days for old behavior
  to decay to 1/e. This prevents gaming — you can't flip your agent prompts
  today and get cheap inference tomorrow.
- Persist profile locally (it's a single `[K, D]` tensor, ~50KB)
- Publish **profile hash** on-chain per epoch (not the profile itself — privacy)
- The profile hash links to the training attestation, creating a verifiable
  chain: "this node trained on data that produced this behavioral signature"

**What the profile captures**:
- Semantic distribution of agent interactions (healthcare, education, finance, etc.)
- Tool usage patterns (what kinds of actions the node's agents take)
- Conversation topic distribution (what users ask about)
- Goal alignment signals (from user profile standards)

**What the profile does NOT reveal**:
- Individual conversations or queries
- Specific tool call contents
- Personal information (PII scrubbed before training)

**Anti-gaming properties**:
- EMA with slow decay means months of consistent behavior required
- Profile is derived from actual training data (weight deltas), not agent config
- You can't fake training data — coordinators verify weight delta quality
- Switching agent system prompts changes future behavior but doesn't erase history

**Acceptance criteria**:
- Profile accumulates over training cycles (EMA update verified)
- Profile persists across restarts
- Profile hash published on-chain per epoch
- Profile is deterministic (same training data → same profile update)
- Unit tests verify EMA decay behavior over simulated epochs

#### Story 5.2: Inference-Ready Governance Threshold

Inference activation is a governance decision, not a hardcoded threshold:
- New proposal type in `EvolutionProposal.sol`: `INFERENCE_ACTIVATION`
- Proposal includes:
  - Benchmark suite (CID of evaluation dataset)
  - Minimum quality metrics (cosine similarity, perplexity, task accuracy)
  - Jurisdiction scope (which jurisdictions can serve inference)
- RPB evaluator assesses proposal (Phase 1: external AI, Phase 3: self-evaluation)
- Jurisdiction coordinators vote via existing Yuma consensus
- If adopted: `InferencePipeline` activates for that jurisdiction

Different jurisdictions may activate inference at different times based on their
specialization. A text-heavy jurisdiction may go live before a multimodal one.

**Acceptance criteria**:
- New proposal type registered in EvolutionProposal contract
- Proposal includes benchmark CID and quality thresholds
- Voting follows existing Yuma consensus path
- Activation is per-jurisdiction (not global)
- Inference service checks activation status before serving

#### Story 5.3: K-NN Alignment Scoring at Inference Time

When a node requests inference, compute dynamic pricing:
- Load the requesting node's behavioral profile (accumulated EMA from 5.1)
- Load the jurisdiction's standards embedding (from Registry, published on-chain)
- Encode the inference request's semantic content (via text encoder → K-vectors)
- Compute alignment as **k-NN distance** in embedding space between:
  1. Node's behavioral profile ↔ jurisdiction standards (long-term alignment)
  2. Request semantics ↔ jurisdiction standards (task-level alignment)
  3. Node's behavioral profile ↔ request semantics (consistency check —
     is this node doing what it usually does, or something unusual?)
- Final alignment score = geometric mean of all three distances
  (matches existing `AlignmentPricing` formula structure)

**Pricing tiers**:
```
alignment > 0.8  → subsidized (network pays part/all of inference cost)
alignment 0.5-0.8 → base cost (node pays full ATN burn)
alignment < 0.5  → premium (node pays base + surcharge)
```

Premium revenue flows to a jurisdiction-level subsidy treasury.
Subsidy draws from that treasury.

**Key property**: Alignment is demonstrated over time through work done, not
through what agents you have configured at any given moment. A node that has
been doing healthcare-related agent work for 6 months gets subsidized healthcare
inference. If they suddenly request financial trading inference, their behavioral
profile doesn't match the request (distance 3 is high), so they pay full price
even if their jurisdiction alignment is fine.

**Acceptance criteria**:
- K-NN distance computation works in embedding space
- Geometric mean of three alignment distances matches paper formula
- Pricing tiers produce correct burn amounts
- Premium surcharges route to jurisdiction subsidy treasury
- Subsidy draws reduce burn amount for aligned nodes
- Integration tests with simulated node profiles

#### Story 5.4: Subsidy/Premium Treasury Mechanics

On-chain treasury that balances aligned subsidies with misaligned premiums:
- Each jurisdiction has a `subsidy_treasury` balance
- Premium surcharges (from misaligned inference) deposit to treasury
- Subsidies (for aligned inference) withdraw from treasury
- If treasury is empty, subsidized nodes pay base cost (no free inference)
- If treasury is full (cap), premium rate decreases (self-balancing)

**Self-balancing properties**:
- More misaligned inference → more premium revenue → bigger subsidy pool
- More aligned inference → more subsidy draws → smaller pool → premiums stay
- Equilibrium: subsidy pool size reflects the alignment ratio of the jurisdiction
- Jurisdictions with mostly aligned nodes have small treasuries (little premium
  revenue, little subsidy needed)
- Jurisdictions with mixed alignment have larger treasuries (active flow)

**Treasury parameters** (governance-configurable per jurisdiction):
- `max_subsidy_rate`: Maximum fraction of inference cost the network covers (e.g., 0.9 = 90% subsidy max)
- `treasury_cap`: Maximum treasury balance (prevents unbounded accumulation)
- `premium_multiplier`: How much extra misaligned nodes pay (e.g., 1.5x = 50% surcharge)

**Acceptance criteria**:
- Treasury contract holds and disburses funds correctly
- Subsidy rate scales with treasury balance (empty = no subsidy)
- Premium deposits tracked per-node for audit
- Treasury balance queryable from dashboard
- Governance can update parameters via proposal
- Self-balancing verified in simulation (treasury converges to equilibrium)

#### Story 5.5: Alignment Dashboard

Show users their alignment status and pricing implications:
- Current behavioral profile summary (top semantic clusters, not raw embeddings)
- Alignment score vs. jurisdiction standards
- Historical alignment trajectory (line chart over epochs)
- Estimated inference pricing tier based on current profile
- Comparison: "If you maintain current behavior, your inference cost in 30/90/180
  days will be approximately X ATN per request"

This replaces the simpler "earnings display" — the dashboard now shows not just
what you've earned but what your behavioral track record means for future costs.

**Acceptance criteria**:
- Dashboard shows alignment score with breakdown (3 distance components)
- Historical trajectory visible (per-epoch data points)
- Pricing tier estimate based on current profile + treasury state
- Updates after each training cycle
- Works before inference is active (shows projected tier)

---

### Epic 6: Byte Tokenizer Scaling

**Goal**: Handle real conversation lengths beyond 512 tokens

#### Story 6.1: Increase max_seq_length

The current `VLJEPAConfig.max_seq_length = 512` is ~512 characters in byte
tokenization. A typical agent conversation turn is 200-2000 characters. Options:

- Increase to 2048 (covers most single turns, 4x memory)
- Increase to 4096 (covers multi-turn context, 16x memory)
- Use chunked encoding: split long sequences into 512-token chunks, encode
  separately, pool/concatenate

**Recommendation**: Start with 2048 (practical for consumer GPUs), add chunking
later for longer context.

**Acceptance criteria**:
- TextEncoder handles sequences up to configured max_seq_length
- Positional embeddings scale to new length
- Memory usage stays within consumer GPU budget (8-16GB)
- Training stability verified (longer sequences may need adjusted learning rate)

#### Story 6.2: Vocabulary Expansion (Optional, Future)

The byte-level vocab (260) is simple but inefficient for English text (~4x more
tokens than BPE). Consider:

- BPE tokenizer trained on agent interaction data (domain-specific)
- SentencePiece with vocab ~8000 (balances efficiency and simplicity)
- Keep byte-level as fallback for non-text content

**NOTE**: This is optional. Byte-level works, just uses longer sequences. The
simplicity of no-external-tokenizer may be worth the sequence length cost.

---

## Dependency Graph

```
Epic 1 (Text Data Adapter)
  ├── Story 1.1: TextTrainingDataSource
  ├── Story 1.2: TextMasker (depends on 1.1 for testing)
  └── Story 1.3: TextEncoder JEPA mode (depends on 1.2)
       │
Epic 2 (Unified Training Loop)
  ├── Story 2.1: MultimodalDataLoader (depends on 1.1)
  ├── Story 2.2: train_vljepa_on_task (depends on 1.3, 2.1)
  └── Story 2.3: Multimodal verification (depends on 2.2)
       │
Epic 3 (Wire Agent Framework)
  ├── Story 3.1: Consent & opt-in (independent)
  ├── Story 3.2: Real-time data feed (depends on 1.1, 3.1)
  ├── Story 3.3: Visual capture wiring (depends on 2.1)
  └── Story 3.4: Solver integration (depends on 2.2, 3.2, 3.3)
       │
Epic 4 (Economic Gating)
  ├── Story 4.1: Gate on wallet (independent, can parallelize)
  ├── Story 4.2: Attestation (depends on 3.4)
  └── Story 4.3: Jurisdiction join (depends on 4.1)
       │
Epic 5 (Alignment-Based Inference Pricing)
  ├── Story 5.1: Behavioral profile accumulation (depends on 3.4 — needs training
  │              cycles to accumulate from; CAN START as soon as solver trains on
  │              real agent data)
  ├── Story 5.2: Inference-ready governance (depends on 4.3 — needs jurisdictions)
  ├── Story 5.3: K-NN alignment scoring (depends on 5.1, 5.2 — needs profiles +
  │              active inference)
  ├── Story 5.4: Subsidy/premium treasury (depends on 5.3 — needs pricing tiers)
  └── Story 5.5: Alignment dashboard (depends on 5.1 — can show profile before
                 inference is active)

Epic 6 (Tokenizer Scaling)  ← can run in parallel with Epics 1-3
  ├── Story 6.1: Increase max_seq_length
  └── Story 6.2: Vocabulary expansion (optional)
```

## Critical Paths

### Path A: Training MVP (agent data → model → tokens)

The shortest path to agent framework data training the model and earning tokens:

```
1.1 → 1.2 → 1.3 → 2.2 → 3.2 → 3.4 → 4.2
```

Text data adapter → text masking → text JEPA training → unified training
function → real-time data feed → solver integration → on-chain attestation.

Visual and multimodal support (2.1, 3.3) can come after the text-only path
is working. Economic gating (4.1, 4.3) can be parallelized.

### Path B: Alignment accumulation (start building track records early)

As soon as Path A reaches 3.4 (solver trains on real agent data), start:

```
3.4 → 5.1 → 5.5
```

Behavioral profile accumulation → alignment dashboard. Nodes begin building
their behavioral track records immediately. By the time inference is ready,
early adopters have months/years of demonstrated alignment — their pricing
tier is already established.

### Path C: Inference activation (when model is ready)

This path can't start until the model reaches useful quality, which is a
governance decision:

```
5.2 → 5.3 → 5.4
```

Governance votes to enable inference → K-NN pricing activates → treasury
mechanics go live. At this point, behavioral profiles from Path B determine
each node's pricing tier.

### Why Path B Matters

Path B is the bootstrap incentive. Early participants:
1. Earn ATN tokens through training (Path A)
2. Accumulate long behavioral track records (Path B)
3. When inference activates (Path C), they have the best alignment scores
4. Best alignment = cheapest inference = most value from their earned tokens
5. Late joiners start with no track record → pay base/premium rates

This creates a natural first-mover advantage that doesn't require vesting
contracts or hardcoded exchange rates. The advantage is earned through
demonstrated behavior, not through being early per se.

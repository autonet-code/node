# VL-JEPA Validation Findings

Tracking empirical results from validating the VL-JEPA architecture for Autonet's
distributed inference pipeline. The core question: **can a K-vector latent plan
bottleneck transmit enough image-specific information for useful text generation?**

## Model Configuration (All Runs)

| Parameter | Value |
|-----------|-------|
| embed_dim | 256 |
| num_heads | 8 |
| encoder_depth | 4 |
| text_encoder_depth | 4 |
| fusion_depth | 4 |
| semantic_predictor_depth | 2 |
| decoder_depth | 4 |
| K (num_latent_vectors) | 16 |
| image_size | 64x64 |
| patch_size | 8 |
| vocab_size | 260 |

## Dataset

- Source: COCO captions (40,000 image-caption pairs cached)
- Split: 18,000 train / 2,000 test
- Preprocessing: resize to 64x64, ImageNet normalization

---

## Test 0: Synthetic Validation (Colored Shapes)

**Goal**: Verify the architecture can learn *anything* through the K-vector bottleneck.

**Setup**: 18-class colored shapes dataset (6 colors x 3 shapes). Captions like
"a red circle". Trivially structured — tests pure information flow, not capacity.

**Result: PASS**
- 100% accuracy on 18-class classification
- Greedy decoding produces correct shape+color descriptions
- Confirms the architecture *can* transmit information through the bottleneck

**Conclusion**: The architecture works in principle. Failures on real data are about
capacity/conditioning, not fundamental design flaws.

---

## Test 1: Real Image Captioning — Run 1 (Baseline)

**Goal**: Test on real COCO images with natural language captions.

**Architecture**: Baseline VL-JEPA (cross-attention only decoder conditioning).

**Training**: 30 epochs, 4,000 samples, batch_size=32

**Result: FAIL — Mode Collapse**
- Greedy: 1 unique caption out of all test images (100% collapse)
- Generated the same sentence regardless of input image
- Sampled (t=0.7): Diverse but random — no correlation with input images

**Diagnosis**: Too few samples and epochs. Scaled up for Run 2.

---

## Test 2: Real Image Captioning — Run 2 (Scaled Up)

**Goal**: Same architecture, more data and training.

**Architecture**: Same baseline VL-JEPA. 16,930,048 parameters.

**Training**: 100 epochs, 20,000 samples, batch_size=32, lr=3e-4, cosine annealing.
Label smoothing=0.1, diversity_weight=0.5, warmup=10 epochs.

**Result: FAIL — Same Mode Collapse**
- Final CE: 1.0619
- Greedy: 1/2000 unique captions (0%)
  - All: "A man is standing on a snowboard in a snowy mou..."
- Sampled (t=0.7): 1869/2000 unique (93%), but only 7.1% content word match
- Unique words: 13 (greedy) vs 2776 (reference)

**Diagnosis**: The autoregressive decoder learns a strong LM prior P(text) and
completely ignores the latent plan conditioning. More data doesn't help because
the decoder has no incentive to attend to the plan — it can minimize CE loss by
memorizing the most common COCO caption pattern.

---

## Test 3: Classification Bottleneck Diagnostic

**Goal**: Bypass the decoder entirely. Test whether the latent plan *contains*
image-specific information, independent of decoder conditioning issues.

**Setup**: Replace decoder with `mean_pool(latent_plan) -> Linear -> 20 classes`.
20 COCO categories extracted via keyword matching from captions (person, dog, cat,
vehicle, bird, horse, food, airplane, boat, train, elephant, giraffe, zebra, sheep,
cow, bear, surfboard, sports, furniture, outdoor).

**Script**: `validate_classification.py`

**Result: PARTIAL (32.1% accuracy)**
- Random baseline: 5% (20 classes)
- Lift: 6.4x over random
- Some categories well-classified (e.g., person, vehicle)
- Others confused (e.g., outdoor <-> furniture)

**Conclusion**: The latent plan *does* carry image-specific information — significantly
above chance. The captioning failure is primarily a **decoder conditioning problem**,
not a bottleneck information flow problem. However, 32.1% is not high enough to
claim the bottleneck is fully sufficient; both issues likely contribute.

**Decision tree applied**: Classification = PARTIAL -> try stronger conditioning first.

---

## Test 4: Real Image Captioning — Run 3 (Stronger Conditioning)

**Goal**: Fix decoder conditioning by making it harder to ignore the latent plan.

**Architecture changes** (additive to baseline):
1. `plan_bias`: Additive bias from plan to cross-attention output
2. `cross_gate`: Learnable scalar gate on cross-attention (sigmoid-initialized high)
3. `plan_dropout`: 20% dropout on plan during training to prevent co-adaptation

**Parameters**: 16,995,844 (+65K from Run 2)

**Training**: Same hyperparameters as Run 2 (100 epochs, 20K samples).

**Result: FAIL — No Improvement**
- Final CE: 1.0637 (vs 1.0619 in Run 2)
- Greedy: 1/2000 unique captions (0%)
  - All: "A man is sitting on a bench next a statue of ma..."
  - (Different collapsed caption than Run 2, but same pathology)
- Sampled (t=0.7): 1871/2000 unique (94%), 6.9% content match
- Training time: 5147s

**Comparison to Run 2:**

| Metric | Run 2 | Run 3 | Change |
|--------|-------|-------|--------|
| Final CE | 1.0619 | 1.0637 | ~0 |
| Greedy unique | 0% | 0% | none |
| Sampled unique | 93% | 94% | ~0 |
| Sampled content | 7.1% | 6.9% | ~0 |
| Params | 16.9M | 17.0M | +65K |

**Root cause analysis**: All three changes (plan_bias, cross_gate, plan_dropout) are
**additive/residual** — the decoder can learn near-zero weights on these pathways and
continue ignoring the plan. The fundamental problem is the decoder has two information
sources (self-attention = LM prior, cross-attention = plan) and it learns to rely
entirely on the LM prior because that's sufficient to minimize CE loss on COCO.

**Decision tree applied**: Stronger conditioning = FAIL -> architectural refactoring needed.

---

## Test 5: Real Image Captioning — Run 4 (Architectural Refactoring)

**Goal**: Make it architecturally impossible for the decoder to ignore the latent plan.

**Architecture changes** (three mechanisms, replacing Run 3's changes):

### 1. Prefix Tokens
K latent plan vectors are projected and **prepended to the decoder sequence** before
self-attention. The causal mask is modified so all positions can attend to the K
prefix positions. This is the mechanism used by Flamingo, BLIP-2, and PaLM-E.

Unlike cross-attention (which the decoder can learn to ignore), prefix tokens are
**in the self-attention stream** — every subsequent token's self-attention computation
is conditioned on them. The decoder would have to learn to actively suppress
16 prefix positions in every layer to ignore them.

### 2. FiLM Conditioning (Feature-wise Linear Modulation)
Each decoder block applies scale (gamma) and shift (beta) derived from the mean-pooled
latent plan to the layer norm outputs. Two FiLM layers per block: one before
self-attention, one before MLP.

This is **multiplicative** — the decoder literally cannot produce features without
the plan's involvement. Initialized to identity (scale=1, shift=0) for training
stability.

### 3. Word Dropout (30%)
During training, entire token embeddings are zeroed out with 30% probability. This
**weakens the LM prior** by making the self-attention token sequence unreliable.
The decoder must rely on the plan (via prefix tokens and FiLM) to compensate for
missing token information.

**Parameters**: 18,052,608 (+1.1M from Run 2, +6% from FiLM and prefix projection)

**Key code locations** (`nodes/common/vl_jepa.py`):
- `CausalSelfAttention.forward()`: `n_prefix` parameter, mask modification
- `FiLMLayer`: New class, scale/shift from conditioning vector
- `DecoderBlock.forward()`: Two FiLM layers (pre-self-attn, pre-MLP) + n_prefix passthrough
- `TextDecoder.forward()`: Prefix concatenation, word dropout, plan_cond extraction
- `TextDecoder.generate()`: Adjusted for prefix positions

**Training**: Same hyperparameters as Runs 2-3 (100 epochs, 20K samples, lr=3e-4).

**Result: FAIL — Same mode collapse as Runs 2-3**
- Final CE: 1.0899 (higher than Run 2's 1.0619 — word dropout makes the task harder)
- Greedy: 1/2000 unique captions (0%)
  - All: "A man is standing in the water next to a boat."
  - (Yet another collapsed caption, different from Runs 2-3)
- Sampled (t=0.7): 1890/2000 unique (94%), 7.1% content match
- Training time: 5082s, 18,052,608 params

**Comparison across all runs:**

| Metric | Run 2 | Run 3 | Run 4 | Change |
|--------|-------|-------|-------|--------|
| Final CE | 1.0619 | 1.0637 | 1.0899 | +2.6% |
| Greedy unique | 0% | 0% | 0% | none |
| Sampled unique | 93% | 94% | 94% | ~0 |
| Sampled content | 7.1% | 6.9% | 7.1% | ~0 |
| Params | 16.9M | 17.0M | 18.1M | +6% |

**Analysis**: Despite using the strongest known conditioning mechanisms (prefix tokens
used by Flamingo/BLIP-2, multiplicative FiLM, and word dropout to weaken the LM prior),
the decoder still collapses to a single greedy output. The slightly higher CE confirms
word dropout is making the task harder, but the decoder still finds a single-caption
optimum. The sampled metrics are statistically identical across all runs.

**Conclusion**: The problem is **not decoder conditioning** — it's the K-vector
bottleneck itself. At this model scale (embed_dim=256, K=16, 64x64 images, ~18M params),
the latent plan does not carry enough image-specific information for the decoder to
produce varied, image-appropriate captions. The classification test showed 32.1%
accuracy on 20 categories (6.4x above chance) — enough to know the bottleneck carries
*some* signal, but not enough to differentiate individual images.

---

## Decision Tree (Completed)

```
Start
  |
  v
[Synthetic validation] --PASS--> [Real image captioning]
  |                                      |
  FAIL -> fundamental design flaw     [Mode collapse?]
                                         |
                                      YES -> [Classification bottleneck test]
                                               |
                                            [>5% random?]
                                               |
                                         YES (PARTIAL, 32.1%)
                                               |
                                         [Stronger conditioning (Run 3)]
                                               |
                                            [Fixed?]
                                               |
                                         NO (identical to Run 2)
                                               |
                                         [Architectural refactoring (Run 4)]
                                               |
                                            [Fixed?]
                                               |
                                         NO (identical to Runs 2-3)
                                               |
                                         CONCLUSION: Bottleneck capacity
                                         insufficient at this model scale.
                                         Need to increase K, embed_dim,
                                         image resolution, or model depth.
```

## Key Insights

1. **Mode collapse in autoregressive decoders is a well-known failure mode** when
   conditioning is weak. The decoder learns P(text) instead of P(text|image).

2. **Additive/residual conditioning changes don't work** because the decoder can
   learn near-zero weights on those pathways. The fix must be structural.

3. **Even structural conditioning changes (prefix+FiLM+dropout) don't work** when
   the underlying conditioning signal doesn't carry enough information. You can't
   force the decoder to use signal that isn't there.

4. **The latent plan carries category-level information** (6.4x above chance on
   20-class classification) but not enough for instance-level captioning.

5. **For Autonet's use case** (distributed inference), the K-vector bottleneck is
   attractive because it's a fixed-size tensor (~K x D x 2 bytes in fp16 = 8KB for
   K=16, D=256). This is orders of magnitude smaller than transmitting raw activations.
   But it must actually carry enough information to generate useful text.

6. **The architecture works in principle** (synthetic shapes = 100%) — the failure
   is about capacity vs. complexity. Real COCO images have far more visual detail
   than colored shapes, requiring more representational capacity.

## Next Steps to Explore

1. **Scale up the bottleneck**: Increase K (32 or 64), embed_dim (512 or 768),
   or image resolution (128x128 or 224x224). Trade bandwidth for information.
2. **Scale up the model**: Deeper encoders (8-12 layers), more heads, larger FFN.
   The current ~18M params may be too small for real-world vision-language.
3. **Pretrained visual encoder**: Use a frozen CLIP or ViT encoder instead of
   training from scratch. Most successful VLMs (BLIP-2, LLaVA) use pretrained vision.
4. **Contrastive bottleneck training**: Train the bottleneck with a contrastive
   loss (like CLIP) rather than purely through the reconstruction gradient.
5. **Reconsider the pipeline design**: For Autonet, transmitting full activations
   (~300KB for 196x768 fp16) may be acceptable given modern bandwidth. The K-vector
   bottleneck may be an over-optimization.

## Files Reference

| File | Purpose |
|------|---------|
| `nodes/common/vl_jepa.py` | Core VL-JEPA architecture |
| `validate_real_images.py` | Captioning training + eval (Runs 1-4) |
| `validate_classification.py` | Classification bottleneck diagnostic |
| `training_run.log` | Run 1 results (30 epochs, 4K samples) |
| `training_run2.log` | Run 2 results (100 epochs, 20K samples, baseline) |
| `training_run3.log` | Run 3 results (100 epochs, stronger conditioning) |
| `training_run4.log` | Run 4 results (100 epochs, prefix+FiLM+dropout) |

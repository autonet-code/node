"""
Latent Plan Bottleneck Validation

Tests whether the VL-JEPA pipeline can learn to map images to text
through the K-vector latent plan bottleneck.

Dataset: Synthetic colored shapes with templated captions.
         6 colors x 3 shapes = 18 distinct classes.

Success: generated captions name the correct shape and color.
         Random guessing = 5.6%. Architecture works if >50%.

Usage:
    python validate_architecture.py
"""

import random
import math
import time
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from nodes.common.vl_jepa import VLJEPAConfig, VLJEPA
from nodes.common.tokenizer import SimpleTokenizer

# ---------------------------------------------------------------------------
# Config — small model, fast training
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_CONFIG = VLJEPAConfig(
    embed_dim=128,
    num_heads=4,
    encoder_depth=2,
    text_encoder_depth=2,
    fusion_depth=2,
    semantic_predictor_depth=2,
    decoder_depth=2,
    num_latent_vectors=8,
    max_seq_length=48,
    image_size=32,
    patch_size=8,       # 32/8 = 4 -> 16 patches
    in_channels=3,
    vocab_size=260,
)

TRAIN_SAMPLES = 540     # 18 classes x 30 each
TEST_SAMPLES = 90       # 18 classes x 5 each
BATCH_SIZE = 18
EPOCHS = 100
LR = 3e-4

# ---------------------------------------------------------------------------
# Synthetic data: colored shapes
# ---------------------------------------------------------------------------

COLORS = {
    "red":    [1.0, 0.0, 0.0],
    "green":  [0.0, 0.8, 0.0],
    "blue":   [0.0, 0.0, 1.0],
    "yellow": [1.0, 1.0, 0.0],
    "cyan":   [0.0, 1.0, 1.0],
    "white":  [1.0, 1.0, 1.0],
}

SHAPES = ["circle", "square", "triangle"]


def make_shape_image(shape: str, color_name: str, size: int = 32) -> torch.Tensor:
    """Create a (3, H, W) tensor with a colored shape on dark background."""
    rgb = COLORS[color_name]
    img = torch.full((3, size, size), 0.15)

    y, x = torch.meshgrid(
        torch.linspace(-1, 1, size),
        torch.linspace(-1, 1, size),
        indexing="ij",
    )

    if shape == "circle":
        mask = (x ** 2 + y ** 2) < 0.55 ** 2
    elif shape == "square":
        mask = (x.abs() < 0.45) & (y.abs() < 0.45)
    elif shape == "triangle":
        # Upward-pointing triangle
        mask = (y > -0.4) & (y < 0.5) & (x.abs() < (0.5 - y) * 0.55)
    else:
        raise ValueError(shape)

    # Paint shape
    for c in range(3):
        img[c][mask] = rgb[c]

    # Small noise so samples aren't identical
    img = img + torch.randn_like(img) * 0.04
    return img.clamp(0, 1)


def generate_dataset(n_samples: int, seed: int = 42):
    """Return (images, captions) lists."""
    rng = random.Random(seed)
    color_names = list(COLORS.keys())

    images, captions = [], []
    for _ in range(n_samples):
        color = rng.choice(color_names)
        shape = rng.choice(SHAPES)
        images.append(make_shape_image(shape, color, MODEL_CONFIG.image_size))
        captions.append(f"a {color} {shape}")

    return images, captions


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    print(f"Device:     {DEVICE}")
    print(f"Model:      embed={MODEL_CONFIG.embed_dim}, encoder_depth={MODEL_CONFIG.encoder_depth}, "
          f"K={MODEL_CONFIG.num_latent_vectors}, "
          f"img={MODEL_CONFIG.image_size}x{MODEL_CONFIG.image_size}")

    # Build model
    model = VLJEPA(MODEL_CONFIG).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,} (trainable)")

    tokenizer = SimpleTokenizer(vocab_size=MODEL_CONFIG.vocab_size)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    # Data
    print("\nGenerating synthetic dataset ...")
    train_imgs, train_caps = generate_dataset(TRAIN_SAMPLES, seed=42)
    test_imgs, test_caps = generate_dataset(TEST_SAMPLES, seed=99)
    print(f"Train: {len(train_imgs)}   Test: {len(test_imgs)}")
    print(f"Sample captions: {train_caps[:5]}")

    # Fixed query for all samples: "describe"
    q_ids, q_mask = tokenizer.batch_encode(
        ["describe"] * BATCH_SIZE, max_length=MODEL_CONFIG.max_seq_length,
    )
    q_ids = q_ids.to(DEVICE)
    q_mask = q_mask.to(DEVICE)

    # Training loop
    print(f"\nTraining for {EPOCHS} epochs ...")
    t0 = time.time()
    model.train()

    for epoch in range(EPOCHS):
        indices = list(range(len(train_imgs)))
        random.shuffle(indices)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(indices), BATCH_SIZE):
            batch_idx = indices[i : i + BATCH_SIZE]
            bs = len(batch_idx)

            imgs = torch.stack([train_imgs[j] for j in batch_idx]).to(DEVICE)
            caps = [train_caps[j] for j in batch_idx]

            tgt_ids, _ = tokenizer.batch_encode(
                caps, max_length=MODEL_CONFIG.max_seq_length,
            )
            tgt_ids = tgt_ids.to(DEVICE)

            # Use pre-built query (trim to actual batch size if last batch smaller)
            qi = q_ids[:bs]
            qm = q_mask[:bs]

            output = model(
                images=imgs,
                token_ids=qi,
                attention_mask=qm,
                target_token_ids=tgt_ids,
            )

            logits = output["decoder_logits"]  # (B, S, vocab)
            # Next-token prediction: logits[:, :-1] -> targets[:, 1:]
            loss = loss_fn(
                logits[:, :-1].contiguous().view(-1, MODEL_CONFIG.vocab_size),
                tgt_ids[:, 1:].contiguous().view(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg = epoch_loss / n_batches

        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1:3d}/{EPOCHS}   loss={avg:.4f}   "
                  f"[{elapsed:.0f}s elapsed]")

    train_time = time.time() - t0
    print(f"\nTraining finished in {train_time:.1f}s")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EVALUATION — generating captions on held-out images")
    print("=" * 60)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(0, len(test_imgs), BATCH_SIZE):
            batch_imgs = test_imgs[i : i + BATCH_SIZE]
            batch_caps = test_caps[i : i + BATCH_SIZE]
            bs = len(batch_imgs)

            imgs = torch.stack(batch_imgs).to(DEVICE)

            qi_eval, qm_eval = tokenizer.batch_encode(
                ["describe"] * bs,
                max_length=MODEL_CONFIG.max_seq_length,
            )
            qi_eval = qi_eval.to(DEVICE)
            qm_eval = qm_eval.to(DEVICE)

            gen_ids = model.generate(
                images=imgs,
                token_ids=qi_eval,
                attention_mask=qm_eval,
                max_len=32,
                temperature=0.0,  # greedy
            )

            for j in range(bs):
                gen_text = tokenizer.decode(gen_ids[j].cpu().tolist())
                expected = batch_caps[j]

                # Check if generated text contains the right color and shape
                parts = expected.split()
                exp_color = parts[1]
                exp_shape = parts[2]
                gen_lower = gen_text.lower()

                match = (exp_color in gen_lower) and (exp_shape in gen_lower)

                if total < 30:
                    tag = " OK " if match else "MISS"
                    print(f"  [{tag}] expected: {expected:20s}  "
                          f"got: {gen_text!r}")

                if match:
                    correct += 1
                total += 1

    acc = correct / total * 100 if total else 0
    print(f"\nAccuracy: {correct}/{total} = {acc:.1f}%")
    print(f"(random baseline: {100/18:.1f}%)")
    print()

    if acc > 80:
        print("RESULT: PASS")
        print("  The latent plan bottleneck carries sufficient information.")
        print("  The architecture can learn image -> K vectors -> text.")
    elif acc > 40:
        print("RESULT: PARTIAL")
        print("  Some information survives the bottleneck.")
        print("  May need larger K, wider embeddings, or more training.")
    elif acc > 15:
        print("RESULT: WEAK")
        print("  Learning signal detected but output quality is poor.")
        print("  Architecture may work at larger scale.")
    else:
        print("RESULT: FAIL")
        print("  Bottleneck is not carrying useful information.")
        print("  Architecture may need fundamental changes.")

    return acc


if __name__ == "__main__":
    train()

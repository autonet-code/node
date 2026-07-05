"""
Real-Image Validation: COCO Captioning

Trains the VL-JEPA pipeline end-to-end on real photographs with human
captions. This is the true proof-of-concept test: can the latent plan
bottleneck carry enough information about a real visual scene for the
decoder to produce topically relevant text?

Dataset: COCO Karpathy (82K images, 5 captions each) via HuggingFace
Model:   ~17M param VL-JEPA (small but real)
Target:  Generated captions that are topically correct, not random

Anti-collapse measures:
- Label smoothing to prevent over-confident predictions
- Latent plan diversity loss to penalize all images mapping to same point
- Warmup + cosine annealing schedule
- Temperature > 0 at eval for diverse generation

Usage:
    python validate_real_images.py
    python validate_real_images.py --max_samples 20000 --epochs 100
"""

import argparse
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from PIL import Image

from nodes.common.vl_jepa import VLJEPAConfig, VLJEPA
from nodes.common.tokenizer import SimpleTokenizer


def latent_plan_diversity_loss(latent_plan: torch.Tensor) -> torch.Tensor:
    """
    Penalize batch-level collapse of latent plans.

    If all images in a batch produce nearly identical latent plans, the
    bottleneck isn't encoding image-specific information. This loss pushes
    different images' latent plans apart.

    Args:
        latent_plan: (B, K, D) — latent plans for a batch

    Returns:
        Scalar loss: lower when latent plans are more diverse across batch
    """
    B = latent_plan.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=latent_plan.device)

    # Mean-pool over K vectors to get per-image representation (B, D)
    pooled = latent_plan.mean(dim=1)

    # Normalize to unit sphere
    pooled = F.normalize(pooled, dim=-1)

    # Cosine similarity matrix (B, B)
    sim = pooled @ pooled.T

    # We want off-diagonal entries to be low (images should be different)
    # Mask diagonal
    mask = ~torch.eye(B, dtype=torch.bool, device=sim.device)
    off_diag = sim[mask]

    # Loss = mean of off-diagonal similarities (want to minimize)
    return off_diag.mean()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_config():
    """Model sized for real images — bigger than synthetic test, still trainable."""
    return VLJEPAConfig(
        embed_dim=256,
        num_heads=8,
        encoder_depth=4,
        text_encoder_depth=4,
        fusion_depth=4,
        semantic_predictor_depth=2,
        decoder_depth=4,
        num_latent_vectors=16,
        max_seq_length=128,     # real captions are longer
        image_size=64,          # downscaled for speed
        patch_size=8,           # 64/8 = 8 -> 64 patches
        in_channels=3,
        vocab_size=260,
    )


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_coco_captions(max_samples=None, test_ratio=0.1, seed=42):
    """
    Load COCO Karpathy captions + download images from COCO CDN.

    Returns:
        train_data: list of (PIL.Image, caption_str)
        test_data:  list of (PIL.Image, caption_str)
    """
    import io
    import os
    import pickle
    import requests
    import sys
    from datasets import load_dataset
    from concurrent.futures import ThreadPoolExecutor, as_completed

    n_download = max_samples * 2 if max_samples else 40000
    cache_tag = f"coco_pairs_{n_download}"
    cache_path = os.path.join("data", f"{cache_tag}.pkl")
    os.makedirs("data", exist_ok=True)

    # Also check old cache (smaller downloads are a subset)
    old_cache = os.path.join("data", "coco_pairs_cache.pkl")
    if not os.path.exists(cache_path) and os.path.exists(old_cache):
        print(f"Found old cache at {old_cache}, loading ...")
        with open(old_cache, "rb") as f:
            old_pairs = pickle.load(f)
        print(f"  Old cache has {len(old_pairs)} pairs")
        if len(old_pairs) >= (max_samples or 0):
            cache_path = old_cache

    if os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path} ...")
        with open(cache_path, "rb") as f:
            pairs = pickle.load(f)
        print(f"  Loaded {len(pairs)} cached pairs")
    else:
        print("Loading COCO Karpathy captions ...")
        ds = load_dataset("yerevann/coco-karpathy", split="train")
        print(f"  {len(ds)} rows loaded")

        rng_ds = random.Random(seed)
        indices = list(range(len(ds)))
        rng_ds.shuffle(indices)
        indices = indices[:n_download]

        url_cap_pairs = []
        for idx in indices:
            row = ds[idx]
            captions = row["sentences"]
            url = row["url"]
            if not captions or not url:
                continue
            cap = rng_ds.choice(captions)
            if len(cap) > 200:
                cap = cap[:200]
            url_cap_pairs.append((url, cap))

        print(f"Downloading {len(url_cap_pairs)} images from COCO CDN ...")

        def download_one(url_cap):
            url, cap = url_cap
            try:
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                # Resize immediately to save memory
                img = img.resize((128, 128), Image.BICUBIC)
                return (img, cap)
            except Exception:
                return None

        pairs = []
        with ThreadPoolExecutor(max_workers=32) as pool:
            futures = [pool.submit(download_one, uc) for uc in url_cap_pairs]
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                if result is not None:
                    pairs.append(result)
                if (i + 1) % 200 == 0:
                    pct = (i + 1) / len(futures) * 100
                    print(f"  Progress: {i+1}/{len(futures)} ({pct:.0f}%) "
                          f"- {len(pairs)} images OK",
                          flush=True)

        print(f"  Downloaded {len(pairs)}/{len(url_cap_pairs)} images")

        # Cache for next run
        with open(cache_path, "wb") as f:
            pickle.dump(pairs, f)
        print(f"  Cached to {cache_path}")

    rng = random.Random(seed)
    rng.shuffle(pairs)

    if max_samples and max_samples < len(pairs):
        pairs = pairs[:max_samples]

    n_test = max(50, int(len(pairs) * test_ratio))
    test_data = pairs[:n_test]
    train_data = pairs[n_test:]

    print(f"Dataset: train={len(train_data)}, test={len(test_data)}")
    for i in range(min(5, len(train_data))):
        print(f"  Sample {i}: {train_data[i][1]!r}")

    return train_data, test_data


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def make_transform(image_size):
    """Standard image transform: resize, center crop, normalize to [0, 1]."""
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),  # -> [0, 1] float tensor
    ])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    config = make_config()

    print(f"Device:     {DEVICE}")
    print(f"Model:      embed={config.embed_dim}, encoder_depth={config.encoder_depth}, "
          f"decoder_depth={config.decoder_depth}, K={config.num_latent_vectors}, "
          f"img={config.image_size}x{config.image_size}")

    model = VLJEPA(config).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,} (trainable)")

    tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
    transform = make_transform(config.image_size)

    # Load data
    train_data, test_data = load_coco_captions(
        max_samples=args.max_samples,
        seed=args.seed,
    )

    # Preprocess all images to tensors (once, up front)
    print("\nPreprocessing images ...")
    t_pre = time.time()
    train_tensors = []
    for img, cap in train_data:
        img_rgb = img.convert("RGB")
        train_tensors.append((transform(img_rgb), cap))

    test_tensors = []
    for img, cap in test_data:
        img_rgb = img.convert("RGB")
        test_tensors.append((transform(img_rgb), cap))

    print(f"Preprocessing done in {time.time() - t_pre:.1f}s")

    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Warmup + cosine annealing
    warmup_epochs = min(10, args.epochs // 5)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Label smoothing to prevent over-confident predictions → reduces collapse
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_id,
        label_smoothing=0.1,
    )

    diversity_weight = 0.5  # weight for latent plan diversity loss

    batch_size = args.batch_size
    n_batches_per_epoch = (len(train_tensors) + batch_size - 1) // batch_size

    # Fixed query
    query_text = "describe the image"

    print(f"\nTraining for {args.epochs} epochs "
          f"({n_batches_per_epoch} batches/epoch, batch_size={batch_size})")
    print(f"  Warmup: {warmup_epochs} epochs, label_smoothing=0.1, "
          f"diversity_weight={diversity_weight}")
    t0 = time.time()
    model.train()

    for epoch in range(args.epochs):
        random.shuffle(train_tensors)
        epoch_loss = 0.0
        epoch_div_loss = 0.0
        n_batches = 0

        for i in range(0, len(train_tensors), batch_size):
            batch = train_tensors[i : i + batch_size]
            bs = len(batch)

            imgs = torch.stack([b[0] for b in batch]).to(DEVICE)
            caps = [b[1] for b in batch]

            # Encode query
            q_ids, q_mask = tokenizer.batch_encode(
                [query_text] * bs, max_length=config.max_seq_length,
            )
            q_ids = q_ids.to(DEVICE)
            q_mask = q_mask.to(DEVICE)

            # Encode target captions
            tgt_ids, _ = tokenizer.batch_encode(
                caps, max_length=config.max_seq_length,
            )
            tgt_ids = tgt_ids.to(DEVICE)

            output = model(
                images=imgs,
                token_ids=q_ids,
                attention_mask=q_mask,
                target_token_ids=tgt_ids,
            )

            logits = output["decoder_logits"]
            ce_loss = loss_fn(
                logits[:, :-1].contiguous().view(-1, config.vocab_size),
                tgt_ids[:, 1:].contiguous().view(-1),
            )

            # Diversity loss: penalize latent plan collapse
            div_loss = latent_plan_diversity_loss(output["latent_plan"])
            loss = ce_loss + diversity_weight * div_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += ce_loss.item()
            epoch_div_loss += div_loss.item()
            n_batches += 1

        scheduler.step()
        avg = epoch_loss / n_batches
        avg_div = epoch_div_loss / n_batches
        elapsed = time.time() - t0

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch+1:3d}/{args.epochs}   "
                  f"ce={avg:.4f}  div={avg_div:.4f}  "
                  f"lr={lr_now:.2e}  [{elapsed:.0f}s elapsed]")

    train_time = time.time() - t0
    print(f"\nTraining finished in {train_time:.1f}s")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EVALUATION — generating captions on held-out images")
    print("=" * 60)

    model.eval()

    # Run evaluation at two temperatures: greedy (0.0) and sampling (0.7)
    for eval_temp, eval_label in [(0.0, "GREEDY"), (0.7, "SAMPLED (t=0.7)")]:
        print(f"\n--- {eval_label} ---")

        results = []
        with torch.no_grad():
            for i in range(0, len(test_tensors), batch_size):
                batch = test_tensors[i : i + batch_size]
                bs = len(batch)

                imgs = torch.stack([b[0] for b in batch]).to(DEVICE)
                ref_caps = [b[1] for b in batch]

                q_ids, q_mask = tokenizer.batch_encode(
                    [query_text] * bs, max_length=config.max_seq_length,
                )
                q_ids = q_ids.to(DEVICE)
                q_mask = q_mask.to(DEVICE)

                gen_ids = model.generate(
                    images=imgs,
                    token_ids=q_ids,
                    attention_mask=q_mask,
                    max_len=80,
                    temperature=eval_temp,
                )

                for j in range(bs):
                    gen_text = tokenizer.decode(gen_ids[j].cpu().tolist())
                    results.append((ref_caps[j], gen_text))

        # Display results
        print(f"\n{'Reference':<50s} | Generated")
        print("-" * 100)
        for ref, gen in results[:30]:
            ref_short = (ref[:47] + "...") if len(ref) > 50 else ref
            gen_short = (gen[:47] + "...") if len(gen) > 50 else gen
            # Sanitize for Windows console encoding
            ref_short = ref_short.encode("ascii", errors="replace").decode("ascii")
            gen_short = gen_short.encode("ascii", errors="replace").decode("ascii")
            print(f"{ref_short:<50s} | {gen_short}")

        # --------------------------------------------------------------
        # Quality metrics
        # --------------------------------------------------------------
        print(f"\n  QUALITY METRICS ({eval_label})")
        print("  " + "-" * 50)

        all_gen_words = set()
        all_ref_words = set()
        gen_lengths = []
        for ref, gen in results:
            all_gen_words.update(gen.lower().split())
            all_ref_words.update(ref.lower().split())
            gen_lengths.append(len(gen.split()))

        print(f"  Unique words generated: {len(all_gen_words)}")
        print(f"  Unique words in references: {len(all_ref_words)}")
        print(f"  Avg generated length: {sum(gen_lengths)/len(gen_lengths):.1f} words")

        unique_gens = len(set(gen for _, gen in results))
        print(f"  Unique captions: {unique_gens}/{len(results)} "
              f"({unique_gens/len(results)*100:.0f}%)")

        overlaps = []
        for ref, gen in results:
            ref_words = set(ref.lower().split())
            gen_words = set(gen.lower().split())
            if ref_words:
                overlap = len(ref_words & gen_words) / len(ref_words)
                overlaps.append(overlap)

        avg_overlap = sum(overlaps) / len(overlaps) * 100 if overlaps else 0
        print(f"  Avg word overlap with reference: {avg_overlap:.1f}%")

        common_visual_words = {
            "man", "woman", "girl", "boy", "child", "people", "dog", "cat",
            "car", "bike", "ball", "water", "tree", "street", "field",
            "white", "black", "red", "blue", "green", "yellow",
            "two", "three", "group", "young", "old", "small", "large",
            "sitting", "standing", "walking", "running", "playing",
            "wearing", "holding",
        }
        content_matches = 0
        for ref, gen in results:
            ref_content = set(ref.lower().split()) & common_visual_words
            gen_content = set(gen.lower().split()) & common_visual_words
            if ref_content and (ref_content & gen_content):
                content_matches += 1

        content_rate = content_matches / len(results) * 100 if results else 0
        print(f"  Content word matches: {content_matches}/{len(results)} "
              f"({content_rate:.1f}%)")

    # Final verdict (based on last eval run — sampled)
    print("\n" + "=" * 60)
    if avg_overlap > 30 and content_rate > 30:
        print("RESULT: PASS — captions are topically relevant")
        print("  The latent plan bottleneck works on real images.")
    elif avg_overlap > 15 or content_rate > 15:
        print("RESULT: PARTIAL — some relevance detected")
        print("  Architecture shows promise but needs more capacity or training.")
    elif unique_gens > len(results) * 0.3:
        print("RESULT: WEAK — diverse but not accurate")
        print("  Model generates varied text but hasn't learned vision-language mapping.")
    else:
        print("RESULT: FAIL — collapsed or random output")
        print("  The model is not learning from visual content.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VL-JEPA real image validation")
    parser.add_argument("--max_samples", type=int, default=20000,
                        help="Max image-caption pairs to use (default: 20000)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs (default: 100)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()
    train(args)

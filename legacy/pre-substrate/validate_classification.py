"""
Classification Test: Does the latent plan carry image-specific information?

Replaces the autoregressive text decoder with a simple linear classifier
on the latent plan. This removes the confound of the decoder's LM prior
and directly tests whether the visual_encoder -> text_encoder -> fusion ->
semantic_predictor pipeline encodes useful visual information into the
K-vector latent plan.

Categories are extracted from COCO captions via keyword matching.
Success criterion: accuracy significantly above random baseline.

Usage:
    python validate_classification.py
"""

import argparse
import os
import pickle
import random
import re
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image

from nodes.common.vl_jepa import VLJEPAConfig, VLJEPA
from nodes.common.tokenizer import SimpleTokenizer


# ---------------------------------------------------------------------------
# Category extraction from captions
# ---------------------------------------------------------------------------

CATEGORIES = {
    "person": [
        r"\bman\b", r"\bwoman\b", r"\bboy\b", r"\bgirl\b", r"\bchild\b",
        r"\bpeople\b", r"\bperson\b", r"\bkid\b", r"\bplayer\b", r"\brider\b",
    ],
    "dog": [r"\bdog\b", r"\bdogs\b", r"\bpuppy\b"],
    "cat": [r"\bcat\b", r"\bcats\b", r"\bkitten\b"],
    "vehicle": [
        r"\bcar\b", r"\bcars\b", r"\btruck\b", r"\bbus\b", r"\bvan\b",
        r"\bmotorcycle\b", r"\bbicycle\b", r"\bbike\b",
    ],
    "bird": [r"\bbird\b", r"\bbirds\b"],
    "horse": [r"\bhorse\b", r"\bhorses\b"],
    "food": [
        r"\bpizza\b", r"\bcake\b", r"\bsandwich\b", r"\bdonut\b",
        r"\bfood\b", r"\bmeal\b", r"\bplate\b", r"\bdish\b",
    ],
    "airplane": [r"\bairplane\b", r"\bplane\b", r"\bjet\b"],
    "boat": [r"\bboat\b", r"\bboats\b", r"\bship\b"],
    "train": [r"\btrain\b", r"\btrains\b", r"\blocomotive\b"],
    "elephant": [r"\belephant\b", r"\belephants\b"],
    "giraffe": [r"\bgiraffe\b", r"\bgiraffes\b"],
    "zebra": [r"\bzebra\b", r"\bzebras\b"],
    "sheep": [r"\bsheep\b", r"\blamb\b"],
    "cow": [r"\bcow\b", r"\bcows\b", r"\bcattle\b"],
    "bear": [r"\bbear\b", r"\bbears\b"],
    "surfboard": [r"\bsurfboard\b", r"\bsurfing\b", r"\bsurf\b"],
    "sports": [
        r"\bbaseball\b", r"\btennis\b", r"\bsoccer\b", r"\bfootball\b",
        r"\bskiing\b", r"\bski\b", r"\bsnowboard\b", r"\bskate\b",
    ],
    "furniture": [
        r"\bcouch\b", r"\bchair\b", r"\btable\b", r"\bdesk\b",
        r"\bbench\b", r"\bbed\b", r"\bsofa\b",
    ],
    "outdoor": [
        r"\bstreet\b", r"\bsidewalk\b", r"\bfield\b", r"\bpark\b",
        r"\bbeach\b", r"\bocean\b", r"\bmountain\b", r"\btree\b",
    ],
}


def extract_category(caption: str) -> str | None:
    """Extract the dominant category from a caption. Returns None if ambiguous."""
    caption_lower = caption.lower()
    matches = []
    for cat_name, patterns in CATEGORIES.items():
        for pat in patterns:
            if re.search(pat, caption_lower):
                matches.append(cat_name)
                break
    if len(matches) == 1:
        return matches[0]
    # If multiple categories match, skip (ambiguous)
    # If no categories match, skip
    return None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_config():
    return VLJEPAConfig(
        embed_dim=256,
        num_heads=8,
        encoder_depth=4,
        text_encoder_depth=4,
        fusion_depth=4,
        semantic_predictor_depth=2,
        decoder_depth=4,
        num_latent_vectors=16,
        max_seq_length=128,
        image_size=64,
        patch_size=8,
        in_channels=3,
        vocab_size=260,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(args):
    config = make_config()
    print(f"Device:     {DEVICE}")
    print(f"Model:      embed={config.embed_dim}, K={config.num_latent_vectors}, "
          f"img={config.image_size}x{config.image_size}")

    # --- Load cached COCO data ---
    cache_path = os.path.join("data", "coco_pairs_40000.pkl")
    if not os.path.exists(cache_path):
        print(f"ERROR: cached data not found at {cache_path}")
        print("Run validate_real_images.py first to download COCO images.")
        return

    print(f"Loading cached dataset from {cache_path} ...")
    with open(cache_path, "rb") as f:
        pairs = pickle.load(f)
    print(f"  Loaded {len(pairs)} cached pairs")

    # --- Extract categories ---
    print("\nExtracting categories from captions ...")
    labeled = []  # (PIL.Image, category_index)
    cat_names = sorted(CATEGORIES.keys())
    cat_to_idx = {name: i for i, name in enumerate(cat_names)}
    cat_counts = {name: 0 for name in cat_names}

    for img, caption in pairs:
        cat = extract_category(caption)
        if cat is not None:
            labeled.append((img, cat_to_idx[cat]))
            cat_counts[cat] += 1

    num_classes = len(cat_names)
    print(f"  {len(labeled)}/{len(pairs)} images have unambiguous category labels")
    print(f"  {num_classes} categories:")
    for name in cat_names:
        print(f"    {name:15s}: {cat_counts[name]:5d}")

    # --- Split train/test ---
    rng = random.Random(42)
    rng.shuffle(labeled)

    max_samples = args.max_samples or len(labeled)
    labeled = labeled[:max_samples]

    test_size = max(200, int(len(labeled) * 0.1))
    train_data = labeled[test_size:]
    test_data = labeled[:test_size]
    print(f"\nDataset: train={len(train_data)}, test={len(test_data)}")

    # --- Preprocess images ---
    print("\nPreprocessing images ...")
    t0_prep = time.time()
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_imgs = torch.stack([transform(d[0]) for d in train_data])
    train_labels = torch.tensor([d[1] for d in train_data], dtype=torch.long)
    test_imgs = torch.stack([transform(d[0]) for d in test_data])
    test_labels = torch.tensor([d[1] for d in test_data], dtype=torch.long)
    print(f"Preprocessing done in {time.time() - t0_prep:.1f}s")

    # --- Build model ---
    model = VLJEPA(config).to(DEVICE)
    tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)

    # Classification head: mean_pool(latent_plan) -> Linear -> num_classes
    classifier = nn.Linear(config.embed_dim, num_classes).to(DEVICE)

    # Count params
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cls_params = sum(p.numel() for p in classifier.parameters())
    print(f"VL-JEPA parameters: {model_params:,}")
    print(f"Classifier parameters: {cls_params:,}")
    print(f"Total trainable: {model_params + cls_params:,}")

    # --- Fixed query tokens ---
    q_ids, q_mask = tokenizer.batch_encode(
        ["classify"] * args.batch_size, max_length=config.max_seq_length,
    )
    q_ids = q_ids.to(DEVICE)
    q_mask = q_mask.to(DEVICE)

    # --- Optimizer ---
    all_params = list(model.parameters()) + list(classifier.parameters())
    optimizer = optim.AdamW(all_params, lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    # --- Training loop ---
    print(f"\nTraining for {args.epochs} epochs (batch_size={args.batch_size})")
    random_baseline = 100.0 / num_classes
    print(f"Random baseline: {random_baseline:.1f}%")

    t0 = time.time()
    model.train()
    classifier.train()

    for epoch in range(args.epochs):
        indices = list(range(len(train_imgs)))
        random.shuffle(indices)
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for i in range(0, len(indices), args.batch_size):
            batch_idx = indices[i : i + args.batch_size]
            bs = len(batch_idx)

            imgs = train_imgs[batch_idx].to(DEVICE)
            labels = train_labels[batch_idx].to(DEVICE)

            qi = q_ids[:bs]
            qm = q_mask[:bs]

            # Forward through VL-JEPA pipeline (no decoder)
            output = model(images=imgs, token_ids=qi, attention_mask=qm)
            latent_plan = output["latent_plan"]  # (B, K, D)

            # Mean-pool over K vectors -> (B, D)
            pooled = latent_plan.mean(dim=1)

            # Classify
            logits = classifier(pooled)  # (B, num_classes)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            preds = logits.argmax(dim=-1)
            epoch_correct += (preds == labels).sum().item()
            epoch_total += bs

        scheduler.step()
        avg_loss = epoch_loss / (len(indices) // args.batch_size + 1)
        train_acc = epoch_correct / epoch_total * 100

        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1:3d}/{args.epochs}   "
                  f"loss={avg_loss:.4f}  train_acc={train_acc:.1f}%  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  [{elapsed:.0f}s]")

    train_time = time.time() - t0
    print(f"\nTraining finished in {train_time:.1f}s")

    # --- Evaluation ---
    print("\n" + "=" * 60)
    print("EVALUATION on held-out test set")
    print("=" * 60)

    model.eval()
    classifier.eval()

    correct = 0
    total = 0
    per_class_correct = {name: 0 for name in cat_names}
    per_class_total = {name: 0 for name in cat_names}
    confusion = {}

    with torch.no_grad():
        for i in range(0, len(test_imgs), args.batch_size):
            imgs = test_imgs[i : i + args.batch_size].to(DEVICE)
            labels = test_labels[i : i + args.batch_size].to(DEVICE)
            bs = imgs.shape[0]

            qi, qm = tokenizer.batch_encode(
                ["classify"] * bs, max_length=config.max_seq_length,
            )
            qi = qi.to(DEVICE)
            qm = qm.to(DEVICE)

            output = model(images=imgs, token_ids=qi, attention_mask=qm)
            latent_plan = output["latent_plan"]
            pooled = latent_plan.mean(dim=1)
            logits = classifier(pooled)
            preds = logits.argmax(dim=-1)

            for j in range(bs):
                true_cat = cat_names[labels[j].item()]
                pred_cat = cat_names[preds[j].item()]
                per_class_total[true_cat] += 1
                if preds[j] == labels[j]:
                    correct += 1
                    per_class_correct[true_cat] += 1

                # Track confusion
                key = (true_cat, pred_cat)
                confusion[key] = confusion.get(key, 0) + 1

                total += 1

    overall_acc = correct / total * 100 if total else 0
    print(f"\nOverall accuracy: {correct}/{total} = {overall_acc:.1f}%")
    print(f"Random baseline:  {random_baseline:.1f}%")
    print(f"Lift over random: {overall_acc / random_baseline:.1f}x")

    print(f"\nPer-class accuracy:")
    for name in cat_names:
        t = per_class_total[name]
        c = per_class_correct[name]
        acc = c / t * 100 if t > 0 else 0
        bar = "#" * int(acc / 5)
        print(f"  {name:15s}: {c:3d}/{t:3d} = {acc:5.1f}%  {bar}")

    # Show top confusions
    print(f"\nTop confusions (true -> predicted):")
    sorted_conf = sorted(
        ((k, v) for k, v in confusion.items() if k[0] != k[1]),
        key=lambda x: -x[1],
    )
    for (true_cat, pred_cat), count in sorted_conf[:15]:
        print(f"  {true_cat:15s} -> {pred_cat:15s}: {count}")

    # Verdict
    print("\n" + "=" * 60)
    if overall_acc > 50:
        print(f"RESULT: PASS (accuracy {overall_acc:.1f}% >> {random_baseline:.1f}% random)")
        print("  The latent plan carries significant image-specific information.")
        print("  The captioning failure is a decoder conditioning problem,")
        print("  not a bottleneck information flow problem.")
    elif overall_acc > random_baseline * 2:
        print(f"RESULT: PARTIAL (accuracy {overall_acc:.1f}% > {random_baseline:.1f}% random)")
        print("  Some information flows through the bottleneck but it's weak.")
        print("  May need larger model or architectural changes.")
    else:
        print(f"RESULT: FAIL (accuracy {overall_acc:.1f}% ~ {random_baseline:.1f}% random)")
        print("  The latent plan does NOT carry useful visual information.")
        print("  The problem is upstream of the decoder — the pipeline itself")
        print("  is not encoding image content into the K-vector bottleneck.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    train(args)

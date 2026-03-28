"""
Gate 2 Validation: Rollout Degradation Curves

Experiment:
    "Rollout degradation curves — <50% quality drop at T=5"

Methodology:
    1. Initialise a VL-JEPA model (small preset by default).
    2. Generate an initial latent plan from a synthetic trace.
    3. Run VL-JEPA rollout at K = 1, 2, 3, 5, 8, 10, 16, 32 steps.
    4. Measure 6 quality metrics at each rollout step:
         a. Cosine similarity to step-1 prediction (relative quality proxy)
         b. L2 distance from step-1 prediction
         c. Embedding variance (mean variance of plan across embed_dim)
         d. Spectral rank (effective rank of K × D plan matrix)
         e. Action probe accuracy (linear probe trained on step-1 hidden states)
         f. Mutual information estimate (linear regression R² as proxy)
    5. Normalise each metric so that T=1 = 1.0.
    6. Compute composite quality score as the mean of normalised metrics.
    7. Gate PASSES if quality(T=5) ≥ 0.5 × quality(T=1).

Output:
    - Console table of raw and normalised metrics at each rollout depth.
    - JSON data file (--output flag) with the full degradation curves.
    - Exit code 0 on PASS, 1 on FAIL.

Usage::

    python nodes/common/gate2_degradation.py
    python nodes/common/gate2_degradation.py --checkpoint path/to/vljepa.pt --output curves.json
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from nodes.common.trace_encoder import TraceEncoder, TraceEncoderConfig
from nodes.common.vl_jepa import VLJEPA, get_vljepa_preset
from nodes.common.tokenizer import SimpleTokenizer


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class Gate2Config:
    """Configuration for the Gate 2 degradation experiment."""
    model_preset: str = "small"
    checkpoint_path: Optional[str] = None

    # Rollout depths to evaluate
    rollout_steps: List[int] = field(
        default_factory=lambda: [1, 2, 3, 5, 8, 10, 16, 32]
    )

    # Number of initial plans to average over (reduces variance)
    num_trials: int = 16

    # Probe for action-accuracy metric (trained on T=1 hidden states)
    probe_epochs: int = 30
    probe_lr: float = 1e-3
    probe_hidden_dim: int = 128

    # Gate threshold
    quality_retention_threshold: float = 0.50   # T=5 must retain >= 50% of T=1 quality

    random_seed: int = 42
    output_json: Optional[str] = None


# ===========================================================================
# Metric helpers
# ===========================================================================

def _plan_spread(plan: torch.Tensor) -> float:
    """Mean squared deviation of K latent vectors from their centroid. (B,K,D) → float"""
    centroid = plan.mean(dim=1, keepdim=True)  # (B, 1, D)
    return float((plan - centroid).pow(2).mean().item())


def _effective_rank(plan: torch.Tensor) -> float:
    """Spectral effective rank = exp(entropy of normalised singular values). (B,K,D) → float"""
    B, K, D = plan.shape
    flat = plan.reshape(B * K, D)
    flat = flat - flat.mean(dim=0, keepdim=True)

    try:
        s = torch.linalg.svdvals(flat)
    except Exception:
        _, s, _ = torch.svd(flat)

    s = s / (s.sum() + 1e-8)
    entropy = -(s * (s + 1e-8).log()).sum()
    return float(entropy.exp().item())


def _embedding_variance(plan: torch.Tensor) -> float:
    """Mean variance of plan across the embedding dimension. (B,K,D) → float"""
    return float(plan.var(dim=-1).mean().item())


def _cosine_sim_to_ref(plan: torch.Tensor, ref: torch.Tensor) -> float:
    """Mean cosine similarity between plan and ref, both (B,K,D). → float"""
    p_mean = plan.mean(dim=1)    # (B, D)
    r_mean = ref.mean(dim=1)     # (B, D)
    return float(F.cosine_similarity(p_mean, r_mean, dim=-1).mean().item())


def _l2_distance_to_ref(plan: torch.Tensor, ref: torch.Tensor) -> float:
    """Mean L2 distance between plan and ref. (B,K,D) → float"""
    return float((plan - ref).norm(dim=-1).mean().item())


# ===========================================================================
# Action-accuracy probe (trained on T=1 features)
# ===========================================================================

class _LinearProbe(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _flatten_plan(plan: torch.Tensor) -> torch.Tensor:
    """Flatten (B, K, D) → (B, K*D)."""
    return plan.reshape(plan.shape[0], -1)


def _train_action_probe(
    X: torch.Tensor,   # (N, feat_dim) — features at T=1
    y: torch.Tensor,   # (N,) — synthetic action class labels
    epochs: int,
    lr: float,
) -> _LinearProbe:
    """Train a linear probe to predict action classes from T=1 latent plans."""
    n_classes = int(y.max().item()) + 1
    feat_dim = X.shape[-1]
    probe = _LinearProbe(feat_dim, n_classes)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)

    for _ in range(epochs):
        logits = probe(X)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return probe


@torch.no_grad()
def _probe_accuracy(probe: _LinearProbe, X: torch.Tensor, y: torch.Tensor) -> float:
    probe.eval()
    preds = probe(X).argmax(-1)
    return float((preds == y).float().mean().item())


# ===========================================================================
# Mutual information estimate (linear regression R²)
# ===========================================================================

def _mutual_information_proxy(
    features_t: torch.Tensor,   # (N, feat_dim) at step T
    features_1: torch.Tensor,   # (N, feat_dim) at step 1
) -> float:
    """Estimate MI as the R² of a linear regression from features_1 to features_t.

    High R² means features_t is still linearly predictable from features_1,
    i.e. the information from step 1 is largely preserved.  Falls to 0 when
    the rollout has decorrelated completely.
    """
    X = features_1  # (N, D)
    Y = features_t  # (N, D)

    # Per-dimension ordinary least-squares: Y_d = X β_d
    # Closed form: β = (XᵀX)⁻¹Xᵀy, R² = corr(Ŷ, Y)²
    # Average R² across output dimensions as the MI proxy.
    try:
        # Add bias column
        ones = torch.ones(X.shape[0], 1)
        X_b = torch.cat([X, ones], dim=-1)  # (N, D+1)

        # Solve via least squares
        beta, _, _, _ = torch.linalg.lstsq(X_b, Y, rcond=None)
        Y_hat = X_b @ beta   # (N, D)

        # Per-dim R²
        ss_res = ((Y - Y_hat) ** 2).sum(dim=0)
        ss_tot = ((Y - Y.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
        r2 = 1.0 - ss_res / (ss_tot + 1e-8)
        return float(r2.clamp(0.0, 1.0).mean().item())
    except Exception:
        return 0.0


# ===========================================================================
# Main experiment
# ===========================================================================

@dataclass
class StepMetrics:
    """Raw and normalised metrics at a single rollout step."""
    step: int
    cosine_sim: float          # raw cosine similarity to T=1 reference
    l2_distance: float         # raw L2 distance from T=1 reference
    embedding_variance: float
    spectral_rank: float
    probe_accuracy: float
    mi_proxy: float
    # Normalised (each metric divided by its T=1 value → 1.0 at T=1)
    norm_cosine_sim: float = 0.0
    norm_l2_inv: float = 0.0   # 1 / (1 + L2), normalised
    norm_variance: float = 0.0
    norm_rank: float = 0.0
    norm_probe: float = 0.0
    norm_mi: float = 0.0
    # Composite quality score (mean of normalised metrics)
    composite_quality: float = 0.0


def run_gate2(config: Gate2Config) -> Dict[str, Any]:
    """Run the Gate 2 degradation experiment and return structured results."""
    print("=" * 70)
    print("Gate 2 — Rollout Degradation Curves")
    print("=" * 70)

    rng = torch.Generator()
    rng.manual_seed(config.random_seed)

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print(f"\n[1] Loading VLJEPA ({config.model_preset})...")
    vljepa_cfg = get_vljepa_preset(config.model_preset)
    model = VLJEPA(vljepa_cfg)
    model.eval()

    if config.checkpoint_path is not None:
        ckpt = torch.load(config.checkpoint_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"   Loaded checkpoint: {config.checkpoint_path}")
    else:
        print("   Using random weights (curves reflect untrained dynamics)")

    D = vljepa_cfg.embed_dim
    K = vljepa_cfg.num_latent_vectors
    B = config.num_trials

    # ------------------------------------------------------------------
    # 2. Generate initial latent plans
    # ------------------------------------------------------------------
    print(f"\n[2] Generating {B} random initial latent plans (B={B}, K={K}, D={D})...")
    # In a full experiment these would come from real traces.
    # For the gate framework, random plans test that the rollout dynamics
    # don't collapse immediately even at T=1.
    initial_plans = torch.randn(B, K, D)  # (B, K, D)

    # ------------------------------------------------------------------
    # 3. Run rollout at each step depth and collect raw predictions
    # ------------------------------------------------------------------
    max_steps = max(config.rollout_steps)
    print(f"\n[3] Running rollout up to T={max_steps} steps...")

    with torch.no_grad():
        rollout = model.rollout(initial_plans, num_steps=max_steps)

    predictions: List[torch.Tensor] = rollout["predictions"]  # list of (B, K, D)
    # predictions[0] = T=1 plan, predictions[1] = T=2, …

    # ------------------------------------------------------------------
    # 4. Compute features for the probe (based on T=1 predictions)
    # ------------------------------------------------------------------
    print("\n[4] Training action-accuracy probe on T=1 features...")
    ref_plan = predictions[0]                          # (B, K, D) — T=1 reference
    feat_1 = _flatten_plan(ref_plan)                  # (B, K*D)

    # Synthetic labels: B action classes (each trial has its own "true" class)
    # In a real experiment these come from the trace labels.
    # Here we assign class = trial_index % 10 as a synthetic target.
    n_action_classes = min(10, B)
    y_action = torch.arange(B) % n_action_classes     # (B,)

    probe = _train_action_probe(feat_1, y_action, config.probe_epochs, config.probe_lr)

    # ------------------------------------------------------------------
    # 5. Compute metrics at each requested rollout depth
    # ------------------------------------------------------------------
    print("\n[5] Computing metrics at each rollout depth...")

    step_metrics: Dict[int, StepMetrics] = {}

    for t in sorted(config.rollout_steps):
        plan_t = predictions[t - 1]   # 0-indexed: T=1 → index 0
        feat_t = _flatten_plan(plan_t)

        cos_sim = _cosine_sim_to_ref(plan_t, ref_plan)
        l2_dist = _l2_distance_to_ref(plan_t, ref_plan)
        emb_var = _embedding_variance(plan_t)
        spec_rk = _effective_rank(plan_t)
        probe_acc = _probe_accuracy(probe, feat_t, y_action)
        mi_prx = _mutual_information_proxy(feat_t, feat_1)

        step_metrics[t] = StepMetrics(
            step=t,
            cosine_sim=cos_sim,
            l2_distance=l2_dist,
            embedding_variance=emb_var,
            spectral_rank=spec_rk,
            probe_accuracy=probe_acc,
            mi_proxy=mi_prx,
        )

    # ------------------------------------------------------------------
    # 6. Normalise metrics relative to T=1
    # ------------------------------------------------------------------
    m1 = step_metrics[1]
    # L2: convert to similarity via 1/(1+L2); normalise by T=1 value
    l2_to_sim = lambda l2: 1.0 / (1.0 + l2)
    sim_1 = l2_to_sim(m1.l2_distance)

    for t, m in step_metrics.items():
        norm_cos = m.cosine_sim / (m1.cosine_sim + 1e-8)
        norm_l2  = l2_to_sim(m.l2_distance) / (sim_1 + 1e-8)
        norm_var = m.embedding_variance / (m1.embedding_variance + 1e-8)
        norm_rk  = m.spectral_rank / (m1.spectral_rank + 1e-8)
        # Probe accuracy: normalise to baseline; clamp to [0, 1]
        norm_prb = m.probe_accuracy / (m1.probe_accuracy + 1e-8)
        norm_mi  = m.mi_proxy / (m1.mi_proxy + 1e-8) if m1.mi_proxy > 1e-8 else 1.0

        # Clamp all normalised metrics to [0, 1]
        norms = [
            min(1.0, max(0.0, norm_cos)),
            min(1.0, max(0.0, norm_l2)),
            min(1.0, max(0.0, norm_var)),
            min(1.0, max(0.0, norm_rk)),
            min(1.0, max(0.0, norm_prb)),
            min(1.0, max(0.0, norm_mi)),
        ]
        composite = sum(norms) / len(norms)

        m.norm_cosine_sim = norms[0]
        m.norm_l2_inv     = norms[1]
        m.norm_variance   = norms[2]
        m.norm_rank       = norms[3]
        m.norm_probe      = norms[4]
        m.norm_mi         = norms[5]
        m.composite_quality = composite

    # ------------------------------------------------------------------
    # 7. Report
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Raw Metrics")
    print("=" * 70)
    hdr = f"{'T':>4}  {'cos_sim':>8} {'L2_dist':>8} {'emb_var':>8} "
    hdr += f"{'rank':>6} {'probe':>7} {'MI_prx':>7}"
    print(hdr)
    print("-" * 70)
    for t in sorted(step_metrics):
        m = step_metrics[t]
        print(
            f"{t:>4}  {m.cosine_sim:>8.4f} {m.l2_distance:>8.4f}"
            f" {m.embedding_variance:>8.4f} {m.spectral_rank:>6.2f}"
            f" {m.probe_accuracy:>7.3f} {m.mi_proxy:>7.3f}"
        )

    print("\n" + "=" * 70)
    print("Normalised Metrics (relative to T=1, clamped to [0, 1])")
    print("=" * 70)
    hdr2 = f"{'T':>4}  {'cos':>6} {'L2':>6} {'var':>6} {'rank':>6}"
    hdr2 += f" {'probe':>7} {'MI':>6}  {'quality':>8}"
    print(hdr2)
    print("-" * 70)
    for t in sorted(step_metrics):
        m = step_metrics[t]
        gate_flag = "  ← GATE" if t == 5 else ""
        print(
            f"{t:>4}  {m.norm_cosine_sim:>6.3f} {m.norm_l2_inv:>6.3f}"
            f" {m.norm_variance:>6.3f} {m.norm_rank:>6.3f}"
            f" {m.norm_probe:>7.3f} {m.norm_mi:>6.3f}"
            f"  {m.composite_quality:>8.3f}{gate_flag}"
        )

    # ------------------------------------------------------------------
    # 8. Gate verdict
    # ------------------------------------------------------------------
    q1 = step_metrics[1].composite_quality     # = 1.0 by construction
    q5 = step_metrics[5].composite_quality
    retention = q5 / (q1 + 1e-8)

    print("\n" + "=" * 70)
    threshold = config.quality_retention_threshold
    gate_pass = retention >= threshold

    print(f"Quality at T=1 : {q1:.4f}")
    print(f"Quality at T=5 : {q5:.4f}")
    print(f"Retention ratio: {retention:.4f}  (threshold: {threshold:.2f})")
    verdict = "PASS" if gate_pass else "FAIL"
    print(f"\nGate 2: {verdict}  (retention={retention:.3f} {'≥' if gate_pass else '<'} {threshold:.2f})")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 9. Save results
    # ------------------------------------------------------------------
    results = {
        "verdict": verdict,
        "quality_retention_threshold": threshold,
        "quality_at_t1": q1,
        "quality_at_t5": q5,
        "retention_ratio": retention,
        "model_preset": config.model_preset,
        "num_trials": B,
        "embed_dim": D,
        "num_latent_vectors": K,
        "degradation_curves": {
            "steps": sorted(step_metrics.keys()),
            "raw": {
                "cosine_sim":         [step_metrics[t].cosine_sim for t in sorted(step_metrics)],
                "l2_distance":        [step_metrics[t].l2_distance for t in sorted(step_metrics)],
                "embedding_variance": [step_metrics[t].embedding_variance for t in sorted(step_metrics)],
                "spectral_rank":      [step_metrics[t].spectral_rank for t in sorted(step_metrics)],
                "probe_accuracy":     [step_metrics[t].probe_accuracy for t in sorted(step_metrics)],
                "mi_proxy":           [step_metrics[t].mi_proxy for t in sorted(step_metrics)],
            },
            "normalised": {
                "cosine_sim":         [step_metrics[t].norm_cosine_sim for t in sorted(step_metrics)],
                "l2_inv":             [step_metrics[t].norm_l2_inv for t in sorted(step_metrics)],
                "embedding_variance": [step_metrics[t].norm_variance for t in sorted(step_metrics)],
                "spectral_rank":      [step_metrics[t].norm_rank for t in sorted(step_metrics)],
                "probe_accuracy":     [step_metrics[t].norm_probe for t in sorted(step_metrics)],
                "mi_proxy":           [step_metrics[t].norm_mi for t in sorted(step_metrics)],
                "composite_quality":  [step_metrics[t].composite_quality for t in sorted(step_metrics)],
            },
        },
    }

    if config.output_json:
        out_path = Path(config.output_json)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nDegradation curves saved to {out_path}")

    # Optional: try to plot with matplotlib (don't fail if not installed)
    _try_plot(step_metrics, config.output_json)

    return results


def _try_plot(step_metrics: Dict[int, "StepMetrics"], output_json: Optional[str]) -> None:
    """Save a degradation-curve plot if matplotlib is available."""
    try:
        import matplotlib  # noqa: PLC0415
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        return

    steps = sorted(step_metrics.keys())
    composite = [step_metrics[t].composite_quality for t in steps]
    cosine    = [step_metrics[t].norm_cosine_sim for t in steps]
    variance  = [step_metrics[t].norm_variance for t in steps]
    rank      = [step_metrics[t].norm_rank for t in steps]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(steps, composite, "k-o", label="Composite quality", linewidth=2)
    ax.axhline(0.5, color="red", linestyle="--", label="Gate threshold (50%)")
    ax.axvline(5,   color="orange", linestyle="--", label="T=5")
    ax.set_xlabel("Rollout steps (T)")
    ax.set_ylabel("Normalised quality (1.0 = T=1)")
    ax.set_title("Gate 2: Composite Quality Degradation")
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(steps, cosine,   "b-s", label="Cosine similarity")
    ax2.plot(steps, variance, "g-^", label="Embedding variance")
    ax2.plot(steps, rank,     "m-D", label="Spectral rank")
    ax2.axhline(0.5, color="red", linestyle="--", label="50% threshold")
    ax2.axvline(5,   color="orange", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Rollout steps (T)")
    ax2.set_ylabel("Normalised metric (1.0 = T=1)")
    ax2.set_title("Per-Metric Degradation Curves")
    ax2.legend()
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_json:
        plot_path = Path(output_json).with_suffix(".png")
    else:
        plot_path = Path("gate2_degradation_curves.png")

    plt.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"Degradation plot saved to {plot_path}")


# ===========================================================================
# CLI entry point
# ===========================================================================

def _parse_args() -> Gate2Config:
    parser = argparse.ArgumentParser(
        description="Gate 2: Rollout degradation curves (<50% quality drop at T=5)"
    )
    parser.add_argument("--preset", default="small",
                        choices=["small", "medium", "large"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--trials", type=int, default=16,
                        help="Number of initial plans to average over (default: 16)")
    parser.add_argument("--output", default=None,
                        help="Save results JSON to this path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return Gate2Config(
        model_preset=args.preset,
        checkpoint_path=args.checkpoint,
        num_trials=args.trials,
        output_json=args.output,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    cfg = _parse_args()
    results = run_gate2(cfg)
    sys.exit(0 if results["verdict"] == "PASS" else 1)

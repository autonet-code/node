"""
Gate 1 Validation: MLP Probes on Frozen JEPA Hidden States

Experiment:
    "MLP probes on frozen JEPA must predict tool calls at >85% accuracy"

Methodology:
    1. Generate diverse synthetic agent traces with known tool calls.
    2. Encode each trace through the frozen TraceEncoder + VL-JEPA pipeline
       to obtain latent plan embeddings (the "JEPA hidden states").
    3. Train three probes (linear and MLP) on those frozen embeddings:
         a. Tool-name classification   (multi-class)
         b. Has-arguments binary probe (binary)
         c. Success/failure prediction (binary)
    4. Report per-probe accuracy with 95% bootstrap confidence intervals.
    5. Gate PASSES if the tool-name MLP probe accuracy > 85%.

Usage::

    # Run with random-weight model (gate will almost certainly fail — expected):
    python nodes/common/gate1_probe.py

    # Load a trained checkpoint before running:
    python nodes/common/gate1_probe.py --checkpoint path/to/vljepa.pt

The script prints a structured result table and exits 0 on PASS, 1 on FAIL.
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
from torch.optim import Adam

# Resolve project root so imports work from any CWD
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
class Gate1Config:
    """Configuration for the Gate 1 probe experiment."""
    # Model
    model_preset: str = "small"          # "small" | "medium" | "large"
    checkpoint_path: Optional[str] = None

    # Synthetic data
    num_traces: int = 800                # training + test traces
    test_fraction: float = 0.2
    random_seed: int = 42

    # Probe training
    probe_epochs: int = 50
    probe_lr: float = 1e-3
    probe_batch_size: int = 64
    mlp_hidden_dim: int = 256

    # Bootstrap CI
    n_bootstrap: int = 1000
    ci_alpha: float = 0.05               # 95% CI

    # Gate threshold
    gate_accuracy_threshold: float = 0.85

    # Output
    output_json: Optional[str] = None    # path to save results; None = stdout only


# ===========================================================================
# Synthetic data generation
# ===========================================================================

TOOL_NAMES = [
    "grep", "read", "write", "bash", "glob",
    "edit", "webfetch", "websearch", "agent", "task_stop",
]

_TOOL_TEMPLATES = {
    "grep": lambda i: {"pattern": f"pattern_{i}", "path": "./"},
    "read": lambda i: {"file_path": f"/tmp/file_{i}.txt"},
    "write": lambda i: {"file_path": f"/tmp/out_{i}.py", "content": "x = 1"},
    "bash": lambda i: {"command": f"ls -la /tmp/{i}"},
    "glob": lambda i: {"pattern": f"**/*.py"},
    "edit": lambda i: {"file_path": f"foo.py", "old_string": f"a{i}", "new_string": f"b{i}"},
    "webfetch": lambda i: {"url": f"https://example.com/{i}"},
    "websearch": lambda i: {"query": f"search term {i}"},
    "agent": lambda i: {"description": f"sub-task {i}", "prompt": "do something"},
    "task_stop": lambda _: {},
}


def generate_synthetic_traces(
    num_traces: int,
    rng: torch.Generator,
) -> List[Dict[str, Any]]:
    """Generate *num_traces* synthetic ATN-format traces with known labels.

    Each trace has 2–5 turns; one turn contains a tool call from TOOL_NAMES.
    Label fields are embedded in the trace under a ``_labels`` key (stripped
    before encoding, used only to build the probe dataset).
    """
    traces = []
    for i in range(num_traces):
        tool_idx = int(torch.randint(len(TOOL_NAMES), (1,), generator=rng).item())
        tool = TOOL_NAMES[tool_idx]
        args = _TOOL_TEMPLATES[tool](i)
        has_args = len(args) > 0
        success = bool(torch.rand(1, generator=rng).item() > 0.3)  # 70% success rate

        num_turns = 2 + int(torch.randint(4, (1,), generator=rng).item())
        turns = [
            {
                "role": "user",
                "content": f"Task {i}: use {tool} to accomplish objective {i}.",
                "tool_calls": [],
            }
        ]
        # Tool-call turn
        turns.append({
            "role": "assistant",
            "content": f"Executing {tool} with args {json.dumps(args)}.",
            "tool_calls": [
                {
                    "tool": tool,
                    "args": args,
                    "result": f"output_{i}" if success else "",
                    "success": success,
                }
            ],
        })
        # Pad with observation turns if needed
        for j in range(num_turns - 2):
            turns.append({
                "role": "assistant",
                "content": f"Observation {j}: step complete.",
                "tool_calls": [],
            })

        traces.append({
            "session_id": f"gate1-{i:05d}",
            "agent_id": "probe-agent",
            "agent_type": "solver",
            "timestamp": "2026-03-28T00:00:00Z",
            "turns": turns,
            "outcome": {
                "success": success,
                "task_completed": success,
                "errors": [] if success else [f"error_{i}"],
            },
            # Internal label — stripped before passing to encoder
            "_labels": {
                "tool_name": tool_idx,
                "has_args": int(has_args),
                "success": int(success),
            },
        })
    return traces


# ===========================================================================
# Feature extraction (frozen JEPA pipeline)
# ===========================================================================

@torch.no_grad()
def extract_features(
    traces: List[Dict[str, Any]],
    encoder: TraceEncoder,
    model: VLJEPA,
    tokenizer: SimpleTokenizer,
    query: str = "what tool should be used next?",
    max_query_len: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run all traces through the frozen pipeline and collect latent-plan embeddings.

    Returns:
        features:      (N, K*D) flattened latent plan vectors
        tool_labels:   (N,)  LongTensor — tool-name class index
        args_labels:   (N,)  LongTensor — has-arguments binary (0/1)
        success_labels:(N,)  LongTensor — success binary (0/1)
    """
    encoder.eval()
    model.eval()

    D = model.config.embed_dim
    K = model.config.num_latent_vectors
    N_patches = model.config.num_patches

    all_features: List[torch.Tensor] = []
    all_tool: List[int] = []
    all_args: List[int] = []
    all_suc: List[int] = []

    # Pre-encode the shared query (same for all traces — isolates the trace signal)
    tok_ids, tok_mask = tokenizer.batch_encode([query], max_length=max_query_len)

    for trace in traces:
        labels = trace.pop("_labels")
        clean_trace = {k: v for k, v in trace.items()}
        trace["_labels"] = labels  # restore

        encoded = encoder.encode_trace(clean_trace)
        if encoded is None:
            continue  # filtered out

        turn_embs = encoded["embeddings"]   # (max_turns, D)

        # Pad/trim to num_patches for visual context
        if turn_embs.shape[0] >= N_patches:
            visual_ctx = turn_embs[:N_patches].unsqueeze(0)   # (1, N, D)
        else:
            pad = torch.zeros(N_patches - turn_embs.shape[0], D)
            visual_ctx = torch.cat([turn_embs, pad], dim=0).unsqueeze(0)

        text_embs = model.encode_text(tok_ids, tok_mask)     # (1, S, D)
        fused = model.fuse(visual_ctx, text_embs)            # (1, ?, D)
        latent_plan = model.predict_semantic(fused)          # (1, K, D)

        # Flatten K vectors → probe input
        feat = latent_plan.squeeze(0).reshape(-1)  # (K*D,)
        all_features.append(feat)
        all_tool.append(labels["tool_name"])
        all_args.append(labels["has_args"])
        all_suc.append(labels["success"])

    features = torch.stack(all_features, dim=0)              # (N, K*D)
    return (
        features,
        torch.tensor(all_tool, dtype=torch.long),
        torch.tensor(all_args, dtype=torch.long),
        torch.tensor(all_suc, dtype=torch.long),
    )


# ===========================================================================
# Probe models
# ===========================================================================

class LinearProbe(nn.Module):
    """Single linear layer probe."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MLPProbe(nn.Module):
    """Two-layer MLP probe with BatchNorm and ReLU."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ===========================================================================
# Training and evaluation
# ===========================================================================

def train_probe(
    probe: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int,
    lr: float,
    batch_size: int,
) -> List[float]:
    """Train a probe with cross-entropy loss; return per-epoch train accuracy."""
    probe.train()
    optimizer = Adam(probe.parameters(), lr=lr)
    n = X_train.shape[0]
    history: List[float] = []

    for _ in range(epochs):
        perm = torch.randperm(n)
        X_train = X_train[perm]
        y_train = y_train[perm]

        correct = total = 0
        for start in range(0, n, batch_size):
            xb = X_train[start : start + batch_size]
            yb = y_train[start : start + batch_size]
            logits = probe(xb)
            loss = F.cross_entropy(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct += (logits.argmax(-1) == yb).sum().item()
            total += yb.shape[0]

        history.append(correct / total if total > 0 else 0.0)

    return history


@torch.no_grad()
def evaluate_probe(
    probe: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """Return top-1 accuracy on the given dataset."""
    probe.eval()
    preds = probe(X).argmax(-1)
    return float((preds == y).float().mean().item())


def bootstrap_ci(
    probe: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    n_bootstrap: int,
    alpha: float,
    rng: torch.Generator,
) -> Tuple[float, float, float]:
    """Return (point_estimate, ci_lower, ci_upper) via percentile bootstrap."""
    n = X.shape[0]
    point = evaluate_probe(probe, X, y)
    boot_accs: List[float] = []
    for _ in range(n_bootstrap):
        idx = torch.randint(n, (n,), generator=rng)
        boot_accs.append(evaluate_probe(probe, X[idx], y[idx]))

    boot_accs.sort()
    lo = boot_accs[int(math.floor(alpha / 2 * n_bootstrap))]
    hi = boot_accs[int(math.ceil((1 - alpha / 2) * n_bootstrap)) - 1]
    return point, lo, hi


@dataclass
class ProbeResult:
    """Result for a single probe."""
    probe_name: str          # e.g. "tool_name_mlp"
    task: str                # "tool_name" | "has_args" | "success"
    probe_type: str          # "linear" | "mlp"
    accuracy: float
    ci_lower: float
    ci_upper: float
    num_classes: int
    num_train: int
    num_test: int
    train_epochs: int
    passed_gate: bool        # only meaningful for tool_name probes


# ===========================================================================
# Main experiment
# ===========================================================================

def run_gate1(config: Gate1Config) -> Dict[str, Any]:
    """Run the full Gate 1 experiment and return a structured results dict."""
    print("=" * 70)
    print("Gate 1 — MLP Probes on Frozen JEPA Hidden States")
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

    embed_dim = vljepa_cfg.embed_dim

    if config.checkpoint_path is not None:
        ckpt = torch.load(config.checkpoint_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"   Loaded checkpoint from {config.checkpoint_path}")
    else:
        print("   Using random weights (gate will likely fail — expected for untrained model)")

    encoder_cfg = TraceEncoderConfig(embed_dim=embed_dim)
    encoder = TraceEncoder(encoder_cfg)
    encoder.eval()

    tokenizer = SimpleTokenizer(vocab_size=vljepa_cfg.vocab_size)

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad_(False)
    for param in encoder.parameters():
        param.requires_grad_(False)

    # ------------------------------------------------------------------
    # 2. Generate synthetic traces
    # ------------------------------------------------------------------
    print(f"\n[2] Generating {config.num_traces} synthetic traces...")
    all_traces = generate_synthetic_traces(config.num_traces, rng)

    n_test = max(1, int(config.num_traces * config.test_fraction))
    n_train = config.num_traces - n_test

    # Shuffle
    perm = torch.randperm(len(all_traces), generator=rng).tolist()
    shuffled = [all_traces[i] for i in perm]
    train_traces = shuffled[:n_train]
    test_traces = shuffled[n_train:]

    print(f"   Train: {n_train}  Test: {n_test}")

    # ------------------------------------------------------------------
    # 3. Extract frozen JEPA features
    # ------------------------------------------------------------------
    print("\n[3] Extracting frozen JEPA latent-plan embeddings...")
    X_tr, y_tool_tr, y_args_tr, y_suc_tr = extract_features(
        train_traces, encoder, model, tokenizer
    )
    X_te, y_tool_te, y_args_te, y_suc_te = extract_features(
        test_traces, encoder, model, tokenizer
    )
    feat_dim = X_tr.shape[-1]
    print(f"   Feature dim: {feat_dim}  (K={model.config.num_latent_vectors} × D={embed_dim})")
    print(f"   Train samples: {X_tr.shape[0]}  Test samples: {X_te.shape[0]}")

    # ------------------------------------------------------------------
    # 4. Train & evaluate probes
    # ------------------------------------------------------------------
    print("\n[4] Training probes...")

    probe_specs = [
        # (task_label, X_tr, y_tr, X_te, y_te, num_classes)
        ("tool_name", X_tr, y_tool_tr, X_te, y_tool_te, len(TOOL_NAMES)),
        ("has_args",  X_tr, y_args_tr, X_te, y_args_te, 2),
        ("success",   X_tr, y_suc_tr,  X_te, y_suc_te,  2),
    ]

    probe_results: List[ProbeResult] = []
    gate_pass = False

    for task, xtr, ytr, xte, yte, n_cls in probe_specs:
        for probe_type in ("linear", "mlp"):
            name = f"{task}_{probe_type}"
            print(f"   {name} ...", end="", flush=True)

            if probe_type == "linear":
                probe = LinearProbe(feat_dim, n_cls)
            else:
                probe = MLPProbe(feat_dim, config.mlp_hidden_dim, n_cls)

            train_probe(
                probe, xtr, ytr,
                epochs=config.probe_epochs,
                lr=config.probe_lr,
                batch_size=config.probe_batch_size,
            )

            acc, ci_lo, ci_hi = bootstrap_ci(
                probe, xte, yte,
                n_bootstrap=config.n_bootstrap,
                alpha=config.ci_alpha,
                rng=rng,
            )

            is_gate_probe = (task == "tool_name" and probe_type == "mlp")
            passed = is_gate_probe and acc >= config.gate_accuracy_threshold
            if passed:
                gate_pass = True

            pr = ProbeResult(
                probe_name=name,
                task=task,
                probe_type=probe_type,
                accuracy=acc,
                ci_lower=ci_lo,
                ci_upper=ci_hi,
                num_classes=n_cls,
                num_train=xtr.shape[0],
                num_test=xte.shape[0],
                train_epochs=config.probe_epochs,
                passed_gate=passed,
            )
            probe_results.append(pr)
            print(
                f" acc={acc:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]"
                f"{'  ← GATE PROBE' if is_gate_probe else ''}"
            )

    # ------------------------------------------------------------------
    # 5. Report
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"{'Probe':<30} {'Acc':>7} {'CI lo':>7} {'CI hi':>7} {'n_cls':>6}")
    print("-" * 70)
    for r in probe_results:
        marker = " ← GATE" if r.probe_name == "tool_name_mlp" else ""
        print(
            f"{r.probe_name:<30} {r.accuracy:>7.3f} {r.ci_lower:>7.3f}"
            f" {r.ci_upper:>7.3f} {r.num_classes:>6}{marker}"
        )

    print("\n" + "=" * 70)
    gate_probe = next(r for r in probe_results if r.probe_name == "tool_name_mlp")
    threshold = config.gate_accuracy_threshold

    if gate_pass:
        verdict = "PASS"
        print(
            f"Gate 1: PASS  "
            f"(tool_name MLP accuracy={gate_probe.accuracy:.3f} >= {threshold:.2f})"
        )
    else:
        verdict = "FAIL"
        print(
            f"Gate 1: FAIL  "
            f"(tool_name MLP accuracy={gate_probe.accuracy:.3f} < {threshold:.2f})"
        )
    print("=" * 70)

    results = {
        "verdict": verdict,
        "gate_threshold": threshold,
        "gate_probe_accuracy": gate_probe.accuracy,
        "gate_probe_ci": [gate_probe.ci_lower, gate_probe.ci_upper],
        "probes": [
            {
                "name": r.probe_name,
                "task": r.task,
                "probe_type": r.probe_type,
                "accuracy": r.accuracy,
                "ci": [r.ci_lower, r.ci_upper],
                "num_classes": r.num_classes,
            }
            for r in probe_results
        ],
        "config": {
            "model_preset": config.model_preset,
            "num_traces": config.num_traces,
            "probe_epochs": config.probe_epochs,
            "n_bootstrap": config.n_bootstrap,
        },
    }

    if config.output_json:
        out_path = Path(config.output_json)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {out_path}")

    return results


# ===========================================================================
# CLI entry point
# ===========================================================================

def _parse_args() -> Gate1Config:
    parser = argparse.ArgumentParser(
        description="Gate 1: MLP probes on frozen JEPA hidden states"
    )
    parser.add_argument("--preset", default="small",
                        choices=["small", "medium", "large"],
                        help="VLJEPA model size preset (default: small)")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to a VLJEPA checkpoint (.pt file)")
    parser.add_argument("--num-traces", type=int, default=800,
                        help="Number of synthetic traces to generate (default: 800)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Probe training epochs (default: 50)")
    parser.add_argument("--output", default=None,
                        help="Save results as JSON to this path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return Gate1Config(
        model_preset=args.preset,
        checkpoint_path=args.checkpoint,
        num_traces=args.num_traces,
        probe_epochs=args.epochs,
        output_json=args.output,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    cfg = _parse_args()
    results = run_gate1(cfg)
    sys.exit(0 if results["verdict"] == "PASS" else 1)

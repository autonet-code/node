"""
Model Module Abstraction for Distributed VL-JEPA

Splits VL-JEPA into independently-runnable modules, each with a typed I/O
interface. This is the foundation for pipeline-parallel inference: each module
can be hosted on a different node and forward activations to the next.

Key abstractions:
- ModuleSpec: Metadata describing a module's inputs, outputs, and layer range
- ModelModule: A runnable subset of the model with serialization support
- ModularVLJEPA: The full model split into named modules with a pipeline runner

Module boundaries correspond to functional subsystems:
  0: visual_encoder    — PatchEmbed + ViT layers → visual features     [network]
  1: text_encoder      — Token embed + Transformer → text features     [network]
  2: cross_modal_fusion — Cross-attention fusing visual + text          [network]
  3: semantic_predictor — K query tokens → latent plan (B, K, D)       [network]
  4: text_decoder       — Autoregressive generation from latent plan    [local]

The network produces a latent plan in one round-trip. The decoder runs locally
on the requesting node, so output length doesn't affect network latency.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .vl_jepa import (
    VLJEPA,
    VLJEPAConfig,
    TextEncoder,
    CrossModalFusion,
    SemanticPredictor,
    TextDecoder,
)
from .jepa import VisionTransformerEncoder

logger = logging.getLogger(__name__)

# Module names (canonical ordering for pipeline execution)
MODULE_NAMES = [
    "visual_encoder",
    "text_encoder",
    "cross_modal_fusion",
    "semantic_predictor",
    "text_decoder",
]


@dataclass
class ModuleSpec:
    """Specification for a runnable model module."""

    module_id: int
    name: str  # e.g. "visual_encoder", "cross_modal_fusion"
    input_names: List[str]  # named inputs (e.g. ["images"] or ["visual_embeds", "text_embeds"])
    output_names: List[str]  # named outputs (e.g. ["visual_embeds"])
    input_shapes: Dict[str, Tuple[int, ...]]  # shapes without batch dim
    output_shapes: Dict[str, Tuple[int, ...]]  # shapes without batch dim
    layer_range: Optional[Tuple[int, int]] = None  # for encoder sub-splits
    ema_alpha: float = 0.998
    local_only: bool = False  # if True, runs on requesting node (not network)
    weights_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module_id": self.module_id,
            "name": self.name,
            "input_names": self.input_names,
            "output_names": self.output_names,
            "input_shapes": {k: list(v) for k, v in self.input_shapes.items()},
            "output_shapes": {k: list(v) for k, v in self.output_shapes.items()},
            "layer_range": list(self.layer_range) if self.layer_range else None,
            "ema_alpha": self.ema_alpha,
            "local_only": self.local_only,
            "weights_hash": self.weights_hash,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModuleSpec":
        return cls(
            module_id=d["module_id"],
            name=d["name"],
            input_names=d["input_names"],
            output_names=d["output_names"],
            input_shapes={k: tuple(v) for k, v in d["input_shapes"].items()},
            output_shapes={k: tuple(v) for k, v in d["output_shapes"].items()},
            layer_range=tuple(d["layer_range"]) if d.get("layer_range") else None,
            ema_alpha=d.get("ema_alpha", 0.998),
            local_only=d.get("local_only", False),
            weights_hash=d.get("weights_hash"),
        )


class ModelModule:
    """
    A runnable subset of VL-JEPA that can be hosted on a node.

    Each module wraps an nn.Module (the actual layers) and a ModuleSpec
    (the metadata). It knows how to:
    - Run a forward pass given named inputs
    - Serialize/deserialize weights
    - Compute a content hash of current weights
    - Blend new weights via EMA
    """

    def __init__(self, spec: ModuleSpec, layers: nn.Module):
        self.spec = spec
        self.layers = layers
        self.device = "cpu"

    def to(self, device: str) -> "ModelModule":
        self.device = device
        self.layers = self.layers.to(device)
        return self

    def forward(self, **inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run this module's layers on the provided inputs.

        Dispatches based on module name since each module type has
        a different forward signature.
        """
        name = self.spec.name

        if name == "visual_encoder":
            images = inputs["images"]
            out = self.layers(images)
            return {"visual_embeds": out}

        elif name == "text_encoder":
            token_ids = inputs["token_ids"]
            attention_mask = inputs.get("attention_mask")
            out = self.layers(token_ids, attention_mask)
            return {"text_embeds": out}

        elif name == "cross_modal_fusion":
            visual_embeds = inputs["visual_embeds"]
            text_embeds = inputs["text_embeds"]
            out = self.layers(visual_embeds, text_embeds)
            return {"fused_repr": out}

        elif name == "semantic_predictor":
            fused_repr = inputs["fused_repr"]
            out = self.layers(fused_repr)
            return {"latent_plan": out}

        elif name == "text_decoder":
            latent_plan = inputs["latent_plan"]
            target_token_ids = inputs.get("target_token_ids")
            if target_token_ids is not None:
                logits = self.layers(latent_plan, target_token_ids)
                return {"decoder_logits": logits}
            else:
                max_len = inputs.get("max_len", torch.tensor(128)).item() if isinstance(inputs.get("max_len"), torch.Tensor) else inputs.get("max_len", 128)
                temperature = inputs.get("temperature", torch.tensor(1.0)).item() if isinstance(inputs.get("temperature"), torch.Tensor) else inputs.get("temperature", 1.0)
                generated = self.layers.generate(
                    latent_plan,
                    max_len=int(max_len),
                    temperature=float(temperature),
                )
                return {"generated_ids": generated}

        else:
            raise ValueError(f"Unknown module: {name}")

    def get_weights_hash(self) -> str:
        """Content hash of current weights (SHA-256)."""
        h = hashlib.sha256()
        for name, param in sorted(self.layers.state_dict().items()):
            h.update(name.encode())
            h.update(param.cpu().numpy().tobytes())
        digest = h.hexdigest()
        self.spec.weights_hash = digest
        return digest

    def save_weights(self) -> Dict[str, torch.Tensor]:
        """Serialize weights to dict (CPU tensors)."""
        return {k: v.cpu().clone() for k, v in self.layers.state_dict().items()}

    def load_weights(self, weights: Dict[str, torch.Tensor]):
        """Load weights from serialized dict."""
        self.layers.load_state_dict(weights)
        if self.device != "cpu":
            self.layers = self.layers.to(self.device)

    @torch.no_grad()
    def ema_blend(self, new_weights: Dict[str, torch.Tensor], alpha: Optional[float] = None):
        """
        Blend new weights with current weights using EMA.

        new_state = alpha * old_state + (1 - alpha) * new_weights

        Higher alpha = slower blending (more conservative).
        """
        alpha = alpha if alpha is not None else self.spec.ema_alpha
        current = self.layers.state_dict()
        for key in current:
            if key in new_weights:
                current[key].mul_(alpha).add_(new_weights[key].to(current[key].device), alpha=1 - alpha)
        self.layers.load_state_dict(current)

    def param_count(self) -> int:
        """Total number of parameters in this module."""
        return sum(p.numel() for p in self.layers.parameters())

    def __repr__(self) -> str:
        return (
            f"ModelModule(name={self.spec.name!r}, "
            f"params={self.param_count():,}, "
            f"device={self.device})"
        )


class ModularVLJEPA:
    """
    VL-JEPA split into independently-runnable modules.

    This is the bridge between the monolithic VLJEPA model and the distributed
    inference pipeline. It:
    - Splits the model into 5 functional modules
    - Provides per-module forward execution
    - Provides a full pipeline runner (for local testing / single-node)
    - Verifies that pipeline output matches monolithic forward
    """

    def __init__(self, config: Optional[VLJEPAConfig] = None, model: Optional[VLJEPA] = None):
        """
        Args:
            config: Model configuration. Ignored if model is provided.
            model: Pre-built VLJEPA model. If None, one is created from config.
        """
        if model is not None:
            self.model = model
            self.config = model.config
        else:
            self.config = config or VLJEPAConfig()
            self.model = VLJEPA(self.config)

        self.modules: Dict[str, ModelModule] = {}
        self._split_into_modules()

    def _split_into_modules(self):
        """Split the full model into functional modules."""
        cfg = self.config
        D = cfg.embed_dim
        K = cfg.num_latent_vectors
        N = cfg.num_patches  # 196 for 224/16
        S = cfg.max_seq_length

        # Module 0: visual_encoder [network]
        self.modules["visual_encoder"] = ModelModule(
            spec=ModuleSpec(
                module_id=0,
                name="visual_encoder",
                input_names=["images"],
                output_names=["visual_embeds"],
                input_shapes={"images": (cfg.in_channels, cfg.image_size, cfg.image_size)},
                output_shapes={"visual_embeds": (N, D)},
                ema_alpha=0.996,  # Early layers get faster updates
            ),
            layers=self.model.visual_encoder,
        )

        # Module 1: text_encoder [network]
        self.modules["text_encoder"] = ModelModule(
            spec=ModuleSpec(
                module_id=1,
                name="text_encoder",
                input_names=["token_ids", "attention_mask"],
                output_names=["text_embeds"],
                input_shapes={"token_ids": (S,), "attention_mask": (S,)},
                output_shapes={"text_embeds": (S, D)},
                ema_alpha=0.996,
            ),
            layers=self.model.text_encoder,
        )

        # Module 2: cross_modal_fusion [network]
        self.modules["cross_modal_fusion"] = ModelModule(
            spec=ModuleSpec(
                module_id=2,
                name="cross_modal_fusion",
                input_names=["visual_embeds", "text_embeds"],
                output_names=["fused_repr"],
                input_shapes={
                    "visual_embeds": (N, D),
                    "text_embeds": (S, D),
                },
                output_shapes={"fused_repr": (S, D)},
                ema_alpha=0.999,  # Most sensitive to shifts — slow blend
            ),
            layers=self.model.cross_modal_fusion,
        )

        # Module 3: semantic_predictor [network]
        # Output is the latent plan (B, K, D) — this is what gets
        # transferred to the requesting node for local decoding
        self.modules["semantic_predictor"] = ModelModule(
            spec=ModuleSpec(
                module_id=3,
                name="semantic_predictor",
                input_names=["fused_repr"],
                output_names=["latent_plan"],
                input_shapes={"fused_repr": (S, D)},
                output_shapes={"latent_plan": (K, D)},
                ema_alpha=0.998,
            ),
            layers=self.model.semantic_predictor,
        )

        # Module 4: text_decoder [LOCAL — runs on requesting node]
        # Receives the latent plan over the network, decodes locally.
        # Output length doesn't affect network latency.
        self.modules["text_decoder"] = ModelModule(
            spec=ModuleSpec(
                module_id=4,
                name="text_decoder",
                input_names=["latent_plan", "target_token_ids"],
                output_names=["decoder_logits"],
                input_shapes={
                    "latent_plan": (K, D),
                    "target_token_ids": (S,),
                },
                output_shapes={"decoder_logits": (S, cfg.vocab_size)},
                ema_alpha=0.998,
                local_only=True,
            ),
            layers=self.model.text_decoder,
        )

    def get_module(self, name: str) -> ModelModule:
        """Get a specific module by name."""
        if name not in self.modules:
            raise KeyError(f"No module named {name!r}. Available: {list(self.modules.keys())}")
        return self.modules[name]

    def get_module_specs(self) -> List[ModuleSpec]:
        """Get specs for all modules in pipeline order."""
        return [self.modules[name].spec for name in MODULE_NAMES]

    def run_module(self, module_name: str, **inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run a single module (for distributed inference)."""
        return self.get_module(module_name).forward(**inputs)

    def run_pipeline(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_token_ids: Optional[torch.Tensor] = None,
        modules: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Run a subset of modules in sequence (for local testing).

        If modules is None, runs the full pipeline. Pass a subset for
        intelligence tier simulation (e.g. ["visual_encoder"] for tier 1).

        Args:
            images: (B, C, H, W)
            token_ids: (B, S) query text tokens
            attention_mask: (B, S) optional
            target_token_ids: (B, St) for decoder training, or None for generation
            modules: List of module names to run (default: all)

        Returns:
            Dict of all intermediate and final outputs
        """
        run_modules = modules or MODULE_NAMES
        results: Dict[str, torch.Tensor] = {}

        for name in run_modules:
            if name == "visual_encoder":
                out = self.run_module("visual_encoder", images=images)
                results.update(out)

            elif name == "text_encoder":
                inputs_dict = {"token_ids": token_ids}
                if attention_mask is not None:
                    inputs_dict["attention_mask"] = attention_mask
                out = self.run_module("text_encoder", **inputs_dict)
                results.update(out)

            elif name == "cross_modal_fusion":
                out = self.run_module(
                    "cross_modal_fusion",
                    visual_embeds=results["visual_embeds"],
                    text_embeds=results["text_embeds"],
                )
                results.update(out)

            elif name == "semantic_predictor":
                out = self.run_module(
                    "semantic_predictor",
                    fused_repr=results["fused_repr"],
                )
                results.update(out)

            elif name == "text_decoder":
                if target_token_ids is not None:
                    out = self.run_module(
                        "text_decoder",
                        latent_plan=results["latent_plan"],
                        target_token_ids=target_token_ids,
                    )
                else:
                    out = self.run_module(
                        "text_decoder",
                        latent_plan=results["latent_plan"],
                    )
                results.update(out)

        return results

    def to(self, device: str) -> "ModularVLJEPA":
        """Move all modules to a device."""
        for module in self.modules.values():
            module.to(device)
        return self

    def total_params(self) -> int:
        """Total parameters across all modules."""
        return sum(m.param_count() for m in self.modules.values())

    def module_summary(self) -> str:
        """Human-readable summary of all modules."""
        lines = ["ModularVLJEPA Module Summary", "=" * 60]
        total = 0
        for name in MODULE_NAMES:
            m = self.modules[name]
            count = m.param_count()
            total += count
            location = "LOCAL" if m.spec.local_only else "network"
            lines.append(
                f"  [{m.spec.module_id}] {name:25s} "
                f"{count:>10,} params  "
                f"EMA α={m.spec.ema_alpha}  [{location}]"
            )
        lines.append("-" * 60)
        lines.append(f"  Total: {total:>10,} params")
        lines.append(f"  Latent plan: {self.config.num_latent_vectors} vectors × "
                      f"{self.config.embed_dim}D = {self.config.latent_plan_bytes:,} bytes")
        return "\n".join(lines)

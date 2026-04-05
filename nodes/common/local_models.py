"""
Local Model Registry — catalog of open-weights LLMs usable as JEPA encoder backbone.

Probes host hardware (RAM, VRAM, disk) and filters the catalog to models the
host can actually run.  Each model entry describes its resource requirements
so the UI can surface only viable options.

The selected model is used as a frozen feature extractor: its hidden states
become the input to a small trainable projection head that produces the K*D
latent vectors for the distributed JEPA training loop.

Story 7.1: Local open-weights backbone selection.
"""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    """Specification for a downloadable open-weights model."""
    # Identity
    model_id: str               # HuggingFace repo ID (e.g. "meta-llama/Llama-3.2-1B")
    name: str                   # Human-readable display name
    family: str                 # Model family (llama, mistral, qwen, gemma, phi)
    param_count: str            # e.g. "1B", "3B", "7B", "14B"

    # Architecture details needed for projection head
    hidden_dim: int             # Hidden state dimensionality
    num_layers: int             # Total transformer layers
    vocab_size: int             # Tokenizer vocab size
    context_length: int         # Max sequence length

    # Resource requirements
    ram_gb: float               # Minimum system RAM (model loaded in RAM)
    vram_gb: float              # Minimum VRAM for GPU inference (0 = CPU-only ok)
    disk_gb: float              # Approximate download size
    quantized: bool = False     # Whether this is a quantized variant

    # Extraction config
    extract_layer: int = -1     # Which layer to extract hidden states from (-1 = auto: 2/3 depth)
    pooling: str = "mean"       # How to pool sequence: "mean", "cls", "last"

    # Tags for filtering
    tags: list[str] = field(default_factory=list)


# The catalog.  Entries are ordered smallest → largest within each family.
# vram_gb=0 means the model can run on CPU (slowly).
# ram_gb is the minimum to load the model weights + some headroom.

MODEL_CATALOG: list[ModelSpec] = [
    # --- Llama 3.2 ---
    ModelSpec(
        model_id="meta-llama/Llama-3.2-1B",
        name="Llama 3.2 1B",
        family="llama",
        param_count="1B",
        hidden_dim=2048,
        num_layers=16,
        vocab_size=128256,
        context_length=131072,
        ram_gb=4,
        vram_gb=3,
        disk_gb=2.5,
        tags=["small", "fast"],
    ),
    ModelSpec(
        model_id="meta-llama/Llama-3.2-3B",
        name="Llama 3.2 3B",
        family="llama",
        param_count="3B",
        hidden_dim=3072,
        num_layers=28,
        vocab_size=128256,
        context_length=131072,
        ram_gb=8,
        vram_gb=6,
        disk_gb=6,
        tags=["balanced"],
    ),
    ModelSpec(
        model_id="meta-llama/Llama-3.1-8B",
        name="Llama 3.1 8B",
        family="llama",
        param_count="8B",
        hidden_dim=4096,
        num_layers=32,
        vocab_size=128256,
        context_length=131072,
        ram_gb=18,
        vram_gb=16,
        disk_gb=15,
        tags=["quality"],
    ),

    # --- Qwen 3 ---
    ModelSpec(
        model_id="Qwen/Qwen3-0.6B",
        name="Qwen 3 0.6B",
        family="qwen",
        param_count="0.6B",
        hidden_dim=1024,
        num_layers=28,
        vocab_size=151936,
        context_length=32768,
        ram_gb=3,
        vram_gb=2,
        disk_gb=1.5,
        tags=["tiny", "fast"],
    ),
    ModelSpec(
        model_id="Qwen/Qwen3-1.7B",
        name="Qwen 3 1.7B",
        family="qwen",
        param_count="1.7B",
        hidden_dim=2048,
        num_layers=28,
        vocab_size=151936,
        context_length=32768,
        ram_gb=5,
        vram_gb=4,
        disk_gb=3.5,
        tags=["small", "fast"],
    ),
    ModelSpec(
        model_id="Qwen/Qwen3-4B",
        name="Qwen 3 4B",
        family="qwen",
        param_count="4B",
        hidden_dim=2560,
        num_layers=36,
        vocab_size=151936,
        context_length=32768,
        ram_gb=10,
        vram_gb=8,
        disk_gb=8,
        tags=["balanced"],
    ),
    ModelSpec(
        model_id="Qwen/Qwen3-8B",
        name="Qwen 3 8B",
        family="qwen",
        param_count="8B",
        hidden_dim=4096,
        num_layers=36,
        vocab_size=151936,
        context_length=32768,
        ram_gb=18,
        vram_gb=16,
        disk_gb=16,
        tags=["quality"],
    ),

    # --- Gemma ---
    ModelSpec(
        model_id="google/gemma-3-1b-pt",
        name="Gemma 3 1B",
        family="gemma",
        param_count="1B",
        hidden_dim=1536,
        num_layers=26,
        vocab_size=262144,
        context_length=32768,
        ram_gb=4,
        vram_gb=3,
        disk_gb=2,
        tags=["small", "fast"],
    ),
    ModelSpec(
        model_id="google/gemma-3-4b-pt",
        name="Gemma 3 4B",
        family="gemma",
        param_count="4B",
        hidden_dim=2560,
        num_layers=34,
        vocab_size=262144,
        context_length=131072,
        ram_gb=10,
        vram_gb=8,
        disk_gb=8,
        tags=["balanced"],
    ),

    # --- Mistral ---
    ModelSpec(
        model_id="mistralai/Mistral-7B-v0.3",
        name="Mistral 7B v0.3",
        family="mistral",
        param_count="7B",
        hidden_dim=4096,
        num_layers=32,
        vocab_size=32768,
        context_length=32768,
        ram_gb=16,
        vram_gb=14,
        disk_gb=14,
        tags=["quality"],
    ),

    # --- Phi ---
    ModelSpec(
        model_id="microsoft/Phi-4-mini-instruct",
        name="Phi 4 Mini 3.8B",
        family="phi",
        param_count="3.8B",
        hidden_dim=3072,
        num_layers=32,
        vocab_size=100352,
        context_length=131072,
        ram_gb=9,
        vram_gb=8,
        disk_gb=7.5,
        tags=["balanced", "instruct"],
    ),
]

# Index by model_id for fast lookup
_CATALOG_INDEX: dict[str, ModelSpec] = {m.model_id: m for m in MODEL_CATALOG}


def get_model_spec(model_id: str) -> Optional[ModelSpec]:
    """Look up a model spec by HuggingFace repo ID."""
    return _CATALOG_INDEX.get(model_id)


def list_all_models() -> list[ModelSpec]:
    """Return the full catalog."""
    return list(MODEL_CATALOG)


# ---------------------------------------------------------------------------
# Host capability probe
# ---------------------------------------------------------------------------

@dataclass
class HostCapability:
    """Hardware capabilities of the local host."""
    total_ram_gb: float = 0.0
    available_ram_gb: float = 0.0
    gpu_available: bool = False
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0
    disk_free_gb: float = 0.0
    # MPS (Apple Silicon) support
    mps_available: bool = False


def probe_host() -> HostCapability:
    """Probe the local host for hardware capabilities.

    Best-effort: returns zeros for anything it can't detect.
    """
    cap = HostCapability()

    # RAM
    try:
        import psutil
        mem = psutil.virtual_memory()
        cap.total_ram_gb = mem.total / (1024 ** 3)
        cap.available_ram_gb = mem.available / (1024 ** 3)
    except ImportError:
        logger.debug("psutil not available, RAM detection skipped")

    # Disk (check the default HF cache location)
    try:
        hf_cache = Path.home() / ".cache" / "huggingface"
        hf_cache.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(str(hf_cache))
        cap.disk_free_gb = usage.free / (1024 ** 3)
    except Exception:
        # Fallback to cwd
        try:
            usage = shutil.disk_usage(".")
            cap.disk_free_gb = usage.free / (1024 ** 3)
        except Exception:
            pass

    # GPU (CUDA)
    try:
        import torch
        if torch.cuda.is_available():
            cap.gpu_available = True
            cap.gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            cap.gpu_vram_gb = props.total_memory / (1024 ** 3)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            cap.mps_available = True
            cap.gpu_name = "Apple Silicon (MPS)"
            # MPS shares system RAM; report half of total as usable
            cap.gpu_vram_gb = cap.total_ram_gb * 0.5
    except ImportError:
        logger.debug("torch not available, GPU detection skipped")

    return cap


def compatible_models(
    cap: Optional[HostCapability] = None,
    include_cpu_only: bool = True,
) -> list[ModelSpec]:
    """Return models the host can run, sorted smallest → largest.

    Args:
        cap: Pre-probed capability, or None to probe now.
        include_cpu_only: If True, include models that can run on CPU
                          even if they'd be slow (vram_gb > available GPU VRAM
                          but ram_gb <= available RAM).
    """
    if cap is None:
        cap = probe_host()

    result = []
    for spec in MODEL_CATALOG:
        # Check disk space
        if cap.disk_free_gb > 0 and spec.disk_gb > cap.disk_free_gb:
            continue

        # Check if GPU can handle it
        gpu_ok = False
        if cap.gpu_available or cap.mps_available:
            if spec.vram_gb <= cap.gpu_vram_gb:
                gpu_ok = True

        # Check if CPU can handle it (needs enough RAM)
        cpu_ok = False
        if cap.total_ram_gb > 0 and spec.ram_gb <= cap.total_ram_gb:
            cpu_ok = True

        if gpu_ok or (include_cpu_only and cpu_ok):
            result.append(spec)

    return result


def recommend_model(cap: Optional[HostCapability] = None) -> Optional[ModelSpec]:
    """Pick the best model for this host.

    Strategy: largest model that fits in GPU VRAM. If no GPU, largest that
    fits in RAM with a preference for models tagged "fast".
    """
    if cap is None:
        cap = probe_host()

    candidates = compatible_models(cap, include_cpu_only=True)
    if not candidates:
        return None

    # Prefer GPU-runnable models
    gpu_candidates = [
        m for m in candidates
        if (cap.gpu_available or cap.mps_available)
        and m.vram_gb <= cap.gpu_vram_gb
    ]

    if gpu_candidates:
        # Largest that fits in VRAM (sort by VRAM requirement descending)
        gpu_candidates.sort(key=lambda m: m.vram_gb, reverse=True)
        return gpu_candidates[0]

    # CPU fallback: prefer smaller/faster models
    fast = [m for m in candidates if "fast" in m.tags]
    if fast:
        fast.sort(key=lambda m: m.ram_gb)
        return fast[0]
    candidates.sort(key=lambda m: m.ram_gb)
    return candidates[0]


# ---------------------------------------------------------------------------
# Status / serialization for WS API
# ---------------------------------------------------------------------------

def host_status() -> Dict[str, Any]:
    """Return host capability and model compatibility for the WS API."""
    cap = probe_host()
    models = compatible_models(cap)
    recommended = recommend_model(cap)

    return {
        "host": {
            "total_ram_gb": round(cap.total_ram_gb, 1),
            "available_ram_gb": round(cap.available_ram_gb, 1),
            "gpu_available": cap.gpu_available,
            "gpu_name": cap.gpu_name,
            "gpu_vram_gb": round(cap.gpu_vram_gb, 1),
            "mps_available": cap.mps_available,
            "disk_free_gb": round(cap.disk_free_gb, 1),
        },
        "compatible_models": [
            {
                "model_id": m.model_id,
                "name": m.name,
                "family": m.family,
                "param_count": m.param_count,
                "hidden_dim": m.hidden_dim,
                "ram_gb": m.ram_gb,
                "vram_gb": m.vram_gb,
                "disk_gb": m.disk_gb,
                "tags": m.tags,
                "gpu_runnable": (
                    (cap.gpu_available or cap.mps_available)
                    and m.vram_gb <= cap.gpu_vram_gb
                ),
            }
            for m in models
        ],
        "recommended": recommended.model_id if recommended else None,
    }

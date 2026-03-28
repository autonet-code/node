"""
TraceEncoder: Agent Trace → VL-JEPA Embedding Sequences

Converts structured ATN agent traces into embedding sequences suitable for
VL-JEPA training.  Each trace (a session of tool calls, reasoning, and
results) becomes a variable-length sequence of turn embeddings that can
be used to train the VL-JEPA via next-turn prediction in embedding space.

Trace JSON format (from AGENT_TRACE_TRAINING.md):
    {
        "session_id": "...",
        "agent_id": "...",
        "agent_type": "orchestrator | solver | probe | ...",
        "timestamp": "...",
        "turns": [
            {
                "role": "system | user | assistant",
                "content": "text content",
                "tool_calls": [
                    {"tool": "grep", "args": {...}, "result": "...", "success": true}
                ]
            }
        ],
        "outcome": {
            "success": true,
            "task_completed": true,
            "errors": []
        }
    }

Pipeline:
    JSON trace
        → quality filter (min turns, error status)
        → per-turn encoding:
            TextEncoder   (turn.content)    → text_emb   (embed_dim,)
            ActionEncoder (turn.tool_calls) → action_emb (embed_dim,)
            ResultEncoder (tool results)    → result_emb (embed_dim,)
            TurnFuser     (fuse 3 signals)  → turn_emb   (embed_dim,)
        → padding / truncation to max_turns
        → output: (max_turns, embed_dim) + outcome embedding (embed_dim,)

Sub-encoders:
    TextEncoder   — lightweight transformer encoder for free-form text
    ActionEncoder — encoder for serialised tool calls (name + JSON args)
    ResultEncoder — encoder for tool results (stdout, stderr, output)
    TurnFuser     — self-attention fusion over the three per-turn signals

Integration:
    embed_dim defaults to 384, matching VLJEPAConfig.embed_dim so that trace
    embeddings are compatible with the JEPA training objective.  Scale embed_dim
    alongside VL-JEPA when the model is scaled up.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from .jepa import TransformerBlock, get_1d_sincos_pos_embed
from .tokenizer import SimpleTokenizer, PAD_ID

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class TraceEncoderConfig:
    """Configuration for TraceEncoder.

    embed_dim **must** match VLJEPAConfig.embed_dim (default: 384) so that
    encoded trace embeddings are compatible with the VL-JEPA training space.
    Scale both together when upgrading model size.
    """

    # Embedding dimension — keep in sync with VLJEPAConfig.embed_dim
    embed_dim: int = 384
    num_heads: int = 6
    mlp_ratio: float = 4.0

    # Sub-encoder transformer depths (lighter than the main VL-JEPA encoder)
    text_encoder_depth: int = 4
    action_encoder_depth: int = 3
    result_encoder_depth: int = 3

    # TurnFuser: self-attention layers over the (text, action, result) slots
    fuser_depth: int = 2

    # Tokenizer / sequence lengths (byte-level: 4 special + 256 raw bytes)
    vocab_size: int = 260
    max_text_length: int = 512    # tokens for turn.content
    max_action_length: int = 256  # tokens for serialised tool calls
    max_result_length: int = 256  # tokens for tool results

    # Trace-level sequence
    max_turns: int = 64  # traces longer than this are truncated

    # Quality filtering
    min_turns: int = 2             # skip sessions with fewer turns than this
    include_errored: bool = False  # if False, skip sessions with errors/failures
    dedup_threshold: float = 0.95  # cosine sim above this → treat as duplicate

    # Compute device
    device: str = "cpu"


# =============================================================================
# Shared backbone: byte-level sequence encoder → mean-pooled vector
# =============================================================================


class _SequenceEncoder(nn.Module):
    """Lightweight byte-level transformer encoder.

    Encodes a padded token sequence (byte-level IDs from SimpleTokenizer) and
    returns a **mean-pooled** vector of shape (B, embed_dim).  Used as the
    backbone for TextEncoder, ActionEncoder, and ResultEncoder — each is an
    independent instance with its own weights so it can specialise to its
    modality.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        depth: int,
        max_seq_length: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_length = max_seq_length

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_seq_length, embed_dim)
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        pos = get_1d_sincos_pos_embed(self.embed_dim, self.max_seq_length)
        self.pos_embed.data.copy_(
            torch.from_numpy(pos).float().unsqueeze(0)
        )
        nn.init.normal_(self.token_embed.weight, std=0.02)

    def forward(
        self,
        token_ids: torch.Tensor,                   # (B, S) LongTensor
        attention_mask: Optional[torch.Tensor] = None,  # (B, S) BoolTensor
    ) -> torch.Tensor:
        """Return mean-pooled embedding (B, embed_dim)."""
        B, S = token_ids.shape

        x = self.token_embed(token_ids)       # (B, S, D)
        x = x + self.pos_embed[:, :S, :]

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Mean-pool over real (non-padding) positions
        if attention_mask is not None:
            mask_f = attention_mask.float().unsqueeze(-1)  # (B, S, 1)
            x = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        else:
            x = x.mean(dim=1)

        return x  # (B, embed_dim)


# =============================================================================
# Sub-encoders
# =============================================================================


class TextEncoder(nn.Module):
    """Encode free-form text content (reasoning, descriptions) to embeddings.

    Maps the 'content' field of each turn to a single vector in the shared
    embedding space via byte-level tokenisation and mean-pooling.
    """

    def __init__(self, config: TraceEncoderConfig):
        super().__init__()
        self.config = config
        self.tokenizer = SimpleTokenizer(config.vocab_size)
        self.encoder = _SequenceEncoder(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            depth=config.text_encoder_depth,
            max_seq_length=config.max_text_length,
            mlp_ratio=config.mlp_ratio,
        )

    def encode_text(self, text: str, device: torch.device) -> torch.Tensor:
        """Encode a single string → (1, embed_dim)."""
        if not text:
            return torch.zeros(1, self.config.embed_dim, device=device)
        token_ids, attention_mask = self.tokenizer.batch_encode(
            [text], max_length=self.config.max_text_length
        )
        return self.encoder(token_ids.to(device), attention_mask.to(device))

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a pre-tokenised batch → (B, embed_dim)."""
        return self.encoder(token_ids, attention_mask)


class ActionEncoder(nn.Module):
    """Encode tool calls (tool name + structured args) to embeddings.

    Serialises tool calls to a compact text representation:
        grep({"pattern":"foo","path":"./"}) | read({"file":"bar.py"})

    Then encodes with a dedicated transformer so it can specialise for
    action-type representations (what to do, with what arguments).
    """

    def __init__(self, config: TraceEncoderConfig):
        super().__init__()
        self.config = config
        self.tokenizer = SimpleTokenizer(config.vocab_size)
        self.encoder = _SequenceEncoder(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            depth=config.action_encoder_depth,
            max_seq_length=config.max_action_length,
            mlp_ratio=config.mlp_ratio,
        )

    @staticmethod
    def _serialise(tool_calls: List[Dict[str, Any]]) -> str:
        """Serialise tool calls to compact text (tool_name(json_args))."""
        parts = []
        for tc in tool_calls:
            tool = tc.get("tool", "")
            args = tc.get("args", {})
            parts.append(f"{tool}({json.dumps(args, separators=(',', ':'))})")
        return " | ".join(parts)

    def encode_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        device: torch.device,
    ) -> torch.Tensor:
        """Encode a list of tool call dicts → (1, embed_dim)."""
        if not tool_calls:
            return torch.zeros(1, self.config.embed_dim, device=device)
        text = self._serialise(tool_calls)
        token_ids, attention_mask = self.tokenizer.batch_encode(
            [text], max_length=self.config.max_action_length
        )
        return self.encoder(token_ids.to(device), attention_mask.to(device))

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a pre-tokenised batch → (B, embed_dim)."""
        return self.encoder(token_ids, attention_mask)


class ResultEncoder(nn.Module):
    """Encode tool results (stdout, stderr, structured output) to embeddings.

    Serialises result payloads from tool call records:
        [ok] found 3 matches | [err] FileNotFoundError: no such file

    Encodes with a dedicated transformer so it can specialise for
    observation-type representations (what the world returned).
    """

    def __init__(self, config: TraceEncoderConfig):
        super().__init__()
        self.config = config
        self.tokenizer = SimpleTokenizer(config.vocab_size)
        self.encoder = _SequenceEncoder(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            depth=config.result_encoder_depth,
            max_seq_length=config.max_result_length,
            mlp_ratio=config.mlp_ratio,
        )

    @staticmethod
    def _serialise(tool_calls: List[Dict[str, Any]]) -> str:
        """Serialise result payloads with success status prefix."""
        parts = []
        for tc in tool_calls:
            if "result" not in tc:
                continue
            result = tc["result"]
            status = "ok" if tc.get("success", True) else "err"
            if isinstance(result, (dict, list)):
                result = json.dumps(result, separators=(",", ":"))
            parts.append(f"[{status}] {result}")
        return " | ".join(parts)

    def encode_results(
        self,
        tool_calls: List[Dict[str, Any]],
        device: torch.device,
    ) -> torch.Tensor:
        """Encode tool results from tool call records → (1, embed_dim)."""
        calls_with_results = [tc for tc in tool_calls if "result" in tc]
        if not calls_with_results:
            return torch.zeros(1, self.config.embed_dim, device=device)
        text = self._serialise(calls_with_results)
        token_ids, attention_mask = self.tokenizer.batch_encode(
            [text], max_length=self.config.max_result_length
        )
        return self.encoder(token_ids.to(device), attention_mask.to(device))

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a pre-tokenised batch → (B, embed_dim)."""
        return self.encoder(token_ids, attention_mask)


# =============================================================================
# TurnFuser — fuse text / action / result into a single turn embedding
# =============================================================================


class TurnFuser(nn.Module):
    """Fuse text, action, and result embeddings into a single turn embedding.

    Architecture:
        1. Three learnable role-type embeddings are added to the modality
           vectors to give the self-attention positional identity information.
        2. A small self-attention block refines the 3-slot sequence.
        3. Mean-pooling over slots collapses to a single (embed_dim,) vector.

    When an action or result embedding is absent (zero vector), the slot
    still participates in attention but carries no semantic signal, so
    the fusion gracefully degrades to the available modalities.
    """

    def __init__(self, config: TraceEncoderConfig):
        super().__init__()
        D = config.embed_dim

        # Learnable role-type embeddings: slot 0=text, 1=action, 2=result
        self.role_embed = nn.Parameter(torch.zeros(3, D))

        # Self-attention over the 3-slot sequence
        self.blocks = nn.ModuleList([
            TransformerBlock(D, config.num_heads, config.mlp_ratio)
            for _ in range(config.fuser_depth)
        ])
        self.norm = nn.LayerNorm(D)

        nn.init.normal_(self.role_embed, std=0.02)

    def forward(
        self,
        text_emb: torch.Tensor,    # (B, D)
        action_emb: torch.Tensor,  # (B, D) — zeros if no tool calls
        result_emb: torch.Tensor,  # (B, D) — zeros if no results
    ) -> torch.Tensor:
        """
        Args:
            text_emb:   (B, D) — encoded turn text content
            action_emb: (B, D) — encoded tool calls (zeros if none)
            result_emb: (B, D) — encoded tool results (zeros if none)

        Returns:
            Fused turn embedding (B, D)
        """
        # Stack to (B, 3, D) and add per-slot role embeddings
        slots = torch.stack([text_emb, action_emb, result_emb], dim=1)  # (B, 3, D)
        slots = slots + self.role_embed.unsqueeze(0)                      # (B, 3, D)

        # Self-attention over the three modality slots
        for block in self.blocks:
            slots = block(slots)
        slots = self.norm(slots)

        # Mean-pool over the three slots → single turn vector
        return slots.mean(dim=1)  # (B, D)


# =============================================================================
# OutcomeEncoder — encode session outcome as a supervision signal
# =============================================================================


class OutcomeEncoder(nn.Module):
    """Encode the session outcome to an embedding for supervision.

    Combines:
        1. A learned embedding for the (success, task_completed) combination
           — four discrete states captured by an Embedding(4, D).
        2. A text encoding of any error messages via a shallow _SequenceEncoder.

    Both signals are fused with a linear projection → (1, embed_dim).
    """

    def __init__(self, config: TraceEncoderConfig):
        super().__init__()
        self.config = config
        D = config.embed_dim

        # 4 outcome states:
        #   0 = (success=F, task_completed=F)
        #   1 = (success=F, task_completed=T)
        #   2 = (success=T, task_completed=F)
        #   3 = (success=T, task_completed=T)
        self.outcome_embed = nn.Embedding(4, D)

        # Shallow encoder for error message text
        self.error_encoder = _SequenceEncoder(
            vocab_size=config.vocab_size,
            embed_dim=D,
            num_heads=config.num_heads,
            depth=2,  # very shallow: errors are typically short
            max_seq_length=config.max_text_length,
            mlp_ratio=config.mlp_ratio,
        )
        self.tokenizer = SimpleTokenizer(config.vocab_size)

        # Fuse structured + error-text signals
        self.proj = nn.Linear(D * 2, D)
        nn.init.normal_(self.outcome_embed.weight, std=0.02)

    def forward(
        self,
        success: bool,
        task_completed: bool,
        errors: List[Any],
        device: torch.device,
    ) -> torch.Tensor:
        """Encode session outcome → (1, embed_dim).

        Args:
            success:        whether the session succeeded
            task_completed: whether the assigned task was completed
            errors:         list of error strings / objects
            device:         target device for the output tensor
        """
        # Structured outcome index
        outcome_idx = int(bool(success)) * 2 + int(bool(task_completed))
        idx_t = torch.tensor([outcome_idx], dtype=torch.long, device=device)
        structured = self.outcome_embed(idx_t)  # (1, D)

        # Error messages — concatenate and encode if present
        error_text = " | ".join(str(e) for e in errors) if errors else ""
        if error_text:
            token_ids, attention_mask = self.tokenizer.batch_encode(
                [error_text], max_length=self.config.max_text_length
            )
            error_emb = self.error_encoder(
                token_ids.to(device), attention_mask.to(device)
            )  # (1, D)
        else:
            error_emb = torch.zeros(1, self.config.embed_dim, device=device)

        # Fuse structured + error embeddings
        fused = torch.cat([structured, error_emb], dim=-1)  # (1, 2D)
        return self.proj(fused)  # (1, D)


# =============================================================================
# TraceEncoder — main module
# =============================================================================


class TraceEncoder(nn.Module):
    """Encode ATN agent traces to VL-JEPA-compatible embedding sequences.

    This is an nn.Module because the sub-encoders carry learnable weights.
    Training updates them via the VL-JEPA objective (next-turn prediction
    in embedding space) so that the encoding space co-evolves with the
    main model.

    Input:  dict — raw JSON trace (see module docstring for format)
    Output: dict with:
        embeddings:        (max_turns, embed_dim) float tensor — padded sequence
        turn_mask:         (max_turns,) bool tensor — True for real turns
        outcome_embedding: (embed_dim,) float tensor — supervision signal
        metadata:          dict — session_id, agent_id, num_turns, success, …

    Returns None from encode_trace() if the trace fails quality filtering.
    """

    def __init__(self, config: Optional[TraceEncoderConfig] = None):
        super().__init__()
        self.config = config or TraceEncoderConfig()

        self.text_encoder = TextEncoder(self.config)
        self.action_encoder = ActionEncoder(self.config)
        self.result_encoder = ResultEncoder(self.config)
        self.turn_fuser = TurnFuser(self.config)
        self.outcome_encoder = OutcomeEncoder(self.config)

    @property
    def device(self) -> torch.device:
        """Device the model parameters currently live on."""
        return next(self.parameters()).device

    # -------------------------------------------------------------------------
    # Quality filtering
    # -------------------------------------------------------------------------

    def filter_trace(self, trace: Dict[str, Any]) -> Tuple[bool, str]:
        """Apply quality filter to a raw trace dict.

        Returns:
            (should_include, reason) — reason is empty string when included.
        """
        turns = trace.get("turns", [])
        outcome = trace.get("outcome", {})

        if len(turns) < self.config.min_turns:
            return False, f"too few turns ({len(turns)} < {self.config.min_turns})"

        if not self.config.include_errored:
            errors = outcome.get("errors", [])
            success = outcome.get("success", True)
            if errors or not success:
                return False, "session has errors/failures (include_errored=False)"

        return True, ""

    # -------------------------------------------------------------------------
    # Per-turn encoding
    # -------------------------------------------------------------------------

    def encode_turn(
        self,
        turn: Dict[str, Any],
        device: torch.device,
    ) -> torch.Tensor:
        """Encode a single turn dict → (embed_dim,) tensor.

        Encodes the text content, tool calls (actions), and tool results
        separately, then fuses all three signals into one vector.
        """
        content = turn.get("content") or ""
        tool_calls = turn.get("tool_calls") or []

        text_emb = self.text_encoder.encode_text(content, device)          # (1, D)
        action_emb = self.action_encoder.encode_tool_calls(tool_calls, device)  # (1, D)
        result_emb = self.result_encoder.encode_results(tool_calls, device)     # (1, D)

        turn_emb = self.turn_fuser(text_emb, action_emb, result_emb)  # (1, D)
        return turn_emb.squeeze(0)  # (D,)

    # -------------------------------------------------------------------------
    # Full trace encoding
    # -------------------------------------------------------------------------

    def encode_trace(
        self,
        trace: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Encode a full trace to training-ready tensors.

        Args:
            trace: Raw JSON trace dict in ATN format.

        Returns:
            Dict with keys: embeddings, turn_mask, outcome_embedding, metadata.
            Returns None if the trace fails quality filtering.
        """
        should_include, reason = self.filter_trace(trace)
        if not should_include:
            logger.debug(
                "Skipping trace %s: %s", trace.get("session_id", "?"), reason
            )
            return None

        device = self.device
        turns = trace.get("turns", [])[:self.config.max_turns]
        outcome = trace.get("outcome", {})
        T = len(turns)

        # Encode each turn, then stack and pad to max_turns
        turn_embs = [self.encode_turn(t, device) for t in turns]  # list of (D,)
        stacked = torch.stack(turn_embs, dim=0)                    # (T, D)

        embeddings = torch.zeros(
            self.config.max_turns, self.config.embed_dim, device=device
        )
        embeddings[:T] = stacked

        turn_mask = torch.zeros(self.config.max_turns, dtype=torch.bool, device=device)
        turn_mask[:T] = True

        # Encode outcome as supervision signal
        outcome_embedding = self.outcome_encoder(
            success=bool(outcome.get("success", True)),
            task_completed=bool(outcome.get("task_completed", True)),
            errors=outcome.get("errors", []),
            device=device,
        ).squeeze(0)  # (D,)

        metadata = {
            "session_id": trace.get("session_id", ""),
            "agent_id": trace.get("agent_id", ""),
            "agent_type": trace.get("agent_type", ""),
            "timestamp": trace.get("timestamp", ""),
            "num_turns": T,
            "success": bool(outcome.get("success", True)),
            "task_completed": bool(outcome.get("task_completed", True)),
        }

        return {
            "embeddings": embeddings,              # (max_turns, D)
            "turn_mask": turn_mask,                # (max_turns,)
            "outcome_embedding": outcome_embedding, # (D,)
            "metadata": metadata,
        }

    @torch.no_grad()
    def compute_trace_signature(
        self, trace: Dict[str, Any]
    ) -> Optional[torch.Tensor]:
        """Compute a single representative embedding for a trace.

        Used for deduplication: returns the mean of turn embeddings (D,),
        or None if the trace fails quality filtering.
        """
        encoded = self.encode_trace(trace)
        if encoded is None:
            return None
        embs = encoded["embeddings"]   # (max_turns, D)
        mask = encoded["turn_mask"]    # (max_turns,)
        real_embs = embs[mask]         # (T, D)
        return real_embs.mean(dim=0)   # (D,)


# =============================================================================
# TraceDataset — torch Dataset for DataLoader integration
# =============================================================================


class TraceDataset(Dataset):
    """PyTorch Dataset for agent trace training data.

    Wraps a list of raw trace dicts, applies quality filtering at
    construction time, and encodes traces on demand at __getitem__.

    The dataset stores raw dicts to avoid holding all embeddings in memory
    simultaneously.  Encoding happens in the DataLoader worker processes
    when num_workers > 0.

    Usage:
        config = TraceEncoderConfig(embed_dim=384, max_turns=64)
        encoder = TraceEncoder(config)
        dataset = TraceDataset(traces, encoder)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        for batch in loader:
            emb = batch["embeddings"]      # (B, max_turns, embed_dim)
            mask = batch["turn_mask"]      # (B, max_turns) bool
            outcome = batch["outcome_embedding"]  # (B, embed_dim)
    """

    def __init__(
        self,
        traces: List[Dict[str, Any]],
        encoder: TraceEncoder,
        run_dedup: bool = False,
    ):
        """
        Args:
            traces:    List of raw trace dicts in ATN JSON format.
            encoder:   TraceEncoder instance used for filtering and encoding.
            run_dedup: If True, perform embedding-based deduplication at init
                       (encodes every trace once — slow for large datasets).
                       If False, only structural/fast filtering is applied.
        """
        self.encoder = encoder

        # Structural quality filter (fast — no neural encoding required)
        filtered: List[Dict[str, Any]] = []
        n_skipped = 0
        for trace in traces:
            ok, reason = encoder.filter_trace(trace)
            if ok:
                filtered.append(trace)
            else:
                n_skipped += 1
                logger.debug(
                    "Filtered trace %s: %s",
                    trace.get("session_id", "?"),
                    reason,
                )
        logger.info(
            "TraceDataset: %d/%d traces passed quality filter (%d skipped)",
            len(filtered), len(traces), n_skipped,
        )

        if run_dedup:
            filtered = self._deduplicate(filtered)
            logger.info(
                "TraceDataset: %d traces after embedding-based deduplication",
                len(filtered),
            )

        self._traces = filtered

    # -------------------------------------------------------------------------
    # Embedding-based deduplication
    # -------------------------------------------------------------------------

    def _deduplicate(
        self, traces: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove near-duplicate traces using mean-embedding cosine similarity.

        Iterates traces in order; keeps a trace only if its mean embedding
        has cosine similarity < dedup_threshold with all previously kept traces.
        """
        threshold = self.encoder.config.dedup_threshold
        kept: List[Dict[str, Any]] = []
        signatures: List[torch.Tensor] = []

        self.encoder.eval()
        for trace in traces:
            sig = self.encoder.compute_trace_signature(trace)
            if sig is None:
                continue  # filtered out

            if signatures:
                cached = torch.stack(signatures, dim=0)  # (N, D)
                sims = F.cosine_similarity(
                    sig.unsqueeze(0).expand(len(signatures), -1),
                    cached,
                    dim=-1,
                )  # (N,)
                if sims.max().item() >= threshold:
                    continue  # near-duplicate, skip

            kept.append(trace)
            signatures.append(sig.detach())

        n_removed = len(traces) - len(kept)
        if n_removed:
            logger.debug("Dedup removed %d/%d traces", n_removed, len(traces))
        return kept

    # -------------------------------------------------------------------------
    # Class-method constructors
    # -------------------------------------------------------------------------

    @classmethod
    def from_jsonl(
        cls,
        path: str,
        encoder: TraceEncoder,
        run_dedup: bool = False,
    ) -> "TraceDataset":
        """Load traces from a JSON Lines file (one trace per line)."""
        traces: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed JSON at line %d: %s", lineno, exc)
        logger.info("Loaded %d traces from %s", len(traces), path)
        return cls(traces, encoder, run_dedup=run_dedup)

    @classmethod
    def from_directory(
        cls,
        directory: str,
        encoder: TraceEncoder,
        run_dedup: bool = False,
        glob_pattern: str = "**/*.json",
    ) -> "TraceDataset":
        """Load traces from all JSON files matching glob_pattern in directory.

        Each file may contain a single trace dict or a list of trace dicts.
        """
        traces: List[Dict[str, Any]] = []
        root = Path(directory)
        for p in sorted(root.glob(glob_pattern)):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    traces.extend(data)
                elif isinstance(data, dict):
                    traces.append(data)
                else:
                    logger.warning("Unexpected JSON structure in %s, skipping", p)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Skipping %s: %s", p, exc)
        logger.info(
            "Loaded %d traces from %d files in %s",
            len(traces), sum(1 for _ in root.glob(glob_pattern)), directory,
        )
        return cls(traces, encoder, run_dedup=run_dedup)

    # -------------------------------------------------------------------------
    # Dataset protocol
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._traces)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Encode trace on demand and return training tensors.

        Returns:
            Dict with:
                embeddings:        (max_turns, embed_dim)  float32
                turn_mask:         (max_turns,)            bool
                outcome_embedding: (embed_dim,)            float32

        Note: metadata is excluded to keep collation simple; access via
        dataset.get_metadata(idx) if needed.
        """
        trace = self._traces[idx]
        encoded = self.encoder.encode_trace(trace)

        if encoded is None:
            # Defensive fallback — shouldn't occur since traces are pre-filtered
            D = self.encoder.config.embed_dim
            T = self.encoder.config.max_turns
            return {
                "embeddings": torch.zeros(T, D),
                "turn_mask": torch.zeros(T, dtype=torch.bool),
                "outcome_embedding": torch.zeros(D),
            }

        return {
            "embeddings": encoded["embeddings"],
            "turn_mask": encoded["turn_mask"],
            "outcome_embedding": encoded["outcome_embedding"],
        }

    def get_metadata(self, idx: int) -> Dict[str, Any]:
        """Return the raw metadata dict for a trace (without encoding)."""
        trace = self._traces[idx]
        outcome = trace.get("outcome", {})
        return {
            "session_id": trace.get("session_id", ""),
            "agent_id": trace.get("agent_id", ""),
            "agent_type": trace.get("agent_type", ""),
            "timestamp": trace.get("timestamp", ""),
            "num_turns": len(trace.get("turns", [])),
            "success": bool(outcome.get("success", True)),
            "task_completed": bool(outcome.get("task_completed", True)),
        }

    def get_raw_trace(self, idx: int) -> Dict[str, Any]:
        """Return the raw trace dict at index idx."""
        return self._traces[idx]

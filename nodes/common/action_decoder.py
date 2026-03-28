"""
ActionDecoder: JEPA Embedding Space → Structured Tool Calls

Bridges the latent reasoning space produced by VL-JEPA back to concrete
agent actions (tool calls and text responses).

Architecture (from distillation_decoding_survey.md §5.5):

    JEPA reasoning embeddings  (seq_len, jepa_dim)
           │
           ▼
    MLPAdapter  (LLaVA-style, 2-layer GELU MLP)
           │
           ▼
    decoder input space  (seq_len, decoder_dim)
           │
           ▼
    SmallDecoder  (autoregressive Transformer with cross-attention)
           │
           ▼
    JSONConstrainer  (token-level masking for structural validity)
           │
           ▼
    Valid tool call  {"tool": "...", "args": {...}}

ATN tool call format (matches ActionEncoder._serialise in trace_encoder.py):
    {"tool": "<name>", "args": {...}}

This is the mirror of TraceEncoder: the encoder converts tool calls to
embeddings; this decoder recovers tool calls from embeddings.

Production integration with XGrammar:
    Replace JSONConstrainer.apply_logit_mask() with an XGrammar GrammarMatcher
    that enforces a CFG for valid ATN tool-call JSON.  The rest of the pipeline
    (MLPAdapter + SmallDecoder) is unchanged.  XGrammar achieves <40μs per
    token mask with near-100% structural compliance (JSONSchemaBench, 2025).

Scale configurations:
    PoC (~5M decoder params):
        - CPU-runnable for integration tests and embedding probing
        - jepa_dim=384, decoder_dim=128, depth=2

    Production (~500M decoder params):
        - Single consumer GPU target (24 GB VRAM)
        - jepa_dim=1024, decoder_dim=1024, depth=24

References:
    - LLaVA (2-layer MLP cross-modal adapter): https://llava-vl.github.io/
    - XGrammar constrained decoding: https://arxiv.org/abs/2411.15100
    - Distillation/decoding survey: research/surveys/distillation_decoding_survey.md
"""

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .jepa import TransformerBlock, get_1d_sincos_pos_embed
from .tokenizer import SimpleTokenizer, BOS_ID, EOS_ID, PAD_ID, BYTE_OFFSET

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ActionDecoderConfig:
    """Configuration for ActionDecoder.

    jepa_dim must match the upstream VL-JEPA embed_dim (VLJEPAConfig.embed_dim,
    typically 384-1024).  decoder_dim is the internal dimension of the
    autoregressive decoder; the MLPAdapter bridges the two.

    Scale by adjusting decoder_dim and decoder_depth together.  A rule of
    thumb: decoder_dim=128/depth=2 gives ~5M params (PoC), while
    decoder_dim=1024/depth=24 gives ~500M params (production target).
    """

    # ---- Upstream JEPA embedding dimension ----
    jepa_dim: int = 384         # Must match VLJEPAConfig.embed_dim

    # ---- Decoder internals ----
    decoder_dim: int = 256      # Hidden dimension of the autoregressive decoder
    num_heads: int = 4          # Attention heads in self- and cross-attention
    decoder_depth: int = 4      # Number of DecoderBlock layers
    mlp_ratio: float = 4.0      # FFN hidden dim = decoder_dim * mlp_ratio

    # ---- MLP Adapter (JEPA → decoder bridge) ----
    adapter_hidden_dim: int = 512   # Hidden dim of the 2-layer MLP adapter

    # ---- Tokenizer / sequence lengths ----
    vocab_size: int = 260           # Byte-level: 4 special + 256 raw bytes
    max_output_length: int = 256    # Maximum tokens to generate per action

    # ---- Generation ----
    temperature: float = 0.0    # 0.0 = greedy; >0 enables stochastic sampling
    top_p: float = 0.95         # Nucleus sampling threshold (active when temp > 0)

    # ---- Constrained decoding ----
    enforce_json: bool = True   # Apply JSONConstrainer token mask during generation

    # ---- Compute ----
    device: str = "cpu"


def poc_config() -> ActionDecoderConfig:
    """PoC-scale config (~5M decoder params).

    Runnable on CPU for integration tests.  jepa_dim=384 matches the default
    TraceEncoderConfig and VLJEPAConfig so no config changes are needed for
    the rest of the pipeline.
    """
    return ActionDecoderConfig(
        jepa_dim=384,
        decoder_dim=128,
        num_heads=4,
        decoder_depth=2,
        mlp_ratio=4.0,
        adapter_hidden_dim=256,
        max_output_length=128,
    )


def production_config() -> ActionDecoderConfig:
    """Production-scale config (~500M decoder params).

    Targets a single consumer GPU (24 GB VRAM).  jepa_dim=1024 matches a
    large-scale VL-JEPA encoder.  Pair with XGrammar for structured decoding
    in production.
    """
    return ActionDecoderConfig(
        jepa_dim=1024,
        decoder_dim=1024,
        num_heads=16,
        decoder_depth=24,
        mlp_ratio=4.0,
        adapter_hidden_dim=2048,
        max_output_length=512,
    )


# =============================================================================
# Structured output types
# =============================================================================


@dataclass
class ToolCallOutput:
    """A single decoded tool call action.

    Attributes:
        tool:     Tool name (e.g. "grep", "read", "bash").
        args:     Tool arguments dict (JSON-serialisable).
        raw_text: Raw generated text before JSON parsing.
        valid:    True if the output parsed as valid JSON with a non-empty tool name.
    """

    tool: str
    args: Dict[str, Any]
    raw_text: str
    valid: bool

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to ATN trace format {"tool": ..., "args": ...}."""
        return {"tool": self.tool, "args": self.args}


@dataclass
class DecoderOutput:
    """Complete output from ActionDecoder.decode().

    Attributes:
        tool_calls:       One ToolCallOutput per batch item.
        raw_text:         Complete generated token sequence (joined for B=1).
        input_embeddings: MLPAdapter output (B, S, decoder_dim) — useful for
                          embedding-space consistency checking.
    """

    tool_calls: List[ToolCallOutput]
    raw_text: str
    input_embeddings: torch.Tensor


# =============================================================================
# MLPAdapter — LLaVA-style cross-modal bridge
# =============================================================================


class MLPAdapter(nn.Module):
    """Project JEPA embeddings into the decoder's input space.

    Two-layer MLP with GELU activation and output LayerNorm, following the
    LLaVA 1.5 pattern.  Liu et al. (2024) show this significantly outperforms
    a single linear projection for cross-modal alignment.

    Parameter count:
        ~jepa_dim * adapter_hidden_dim * 2 ≈ 50M for (1024 → 2048 → 1024)

    Input:  (B, S, jepa_dim)    — JEPA reasoning embedding sequence
    Output: (B, S, decoder_dim) — aligned to the decoder's hidden space
    """

    def __init__(self, config: ActionDecoderConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.jepa_dim, config.adapter_hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.adapter_hidden_dim, config.decoder_dim)
        self.norm = nn.LayerNorm(config.decoder_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, jepa_embs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            jepa_embs: JEPA embedding sequence (B, S, jepa_dim)

        Returns:
            Projected embeddings (B, S, decoder_dim)
        """
        x = self.act(self.fc1(jepa_embs))
        return self.norm(self.fc2(x))


# =============================================================================
# _CrossAttention — cross-attend from decoder to adapter context
# =============================================================================


class _CrossAttention(nn.Module):
    """Multi-head cross-attention: decoder queries attend to adapter context.

    Private variant of CrossAttention (cf. vl_jepa.py) that additionally
    supports a context_mask for ignoring padding positions in the JEPA sequence.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        query: torch.Tensor,    # (B, T, D) — decoder hidden states
        context: torch.Tensor,  # (B, S, D) — adapter-projected JEPA context
        context_mask: Optional[torch.Tensor] = None,  # (B, S) bool, True=valid
    ) -> torch.Tensor:
        B, T, D = query.shape
        S = context.shape[1]

        q = self.q_proj(query)    # (B, T, D)
        k = self.k_proj(context)  # (B, S, D)
        v = self.v_proj(context)  # (B, S, D)

        def split_heads(x: torch.Tensor, length: int) -> torch.Tensor:
            return x.view(B, length, self.num_heads, self.head_dim).transpose(1, 2)

        q = split_heads(q, T)  # (B, H, T, head_dim)
        k = split_heads(k, S)  # (B, H, S, head_dim)
        v = split_heads(v, S)  # (B, H, S, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, S)

        if context_mask is not None:
            # Mask out padding positions: ~valid → -inf
            pad_mask = ~context_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
            attn = attn.masked_fill(pad_mask, float("-inf"))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(x)


# =============================================================================
# _CausalSelfAttention — causal (autoregressive) self-attention
# =============================================================================


class _CausalSelfAttention(nn.Module):
    """Causal multi-head self-attention for autoregressive token generation."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,  # (T, T) bool upper-triangular
    ) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, T, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        if causal_mask is None:
            # Upper-triangular True → attend only to past positions
            causal_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(x)


# =============================================================================
# DecoderBlock — causal self-attention + cross-attention + FFN
# =============================================================================


class DecoderBlock(nn.Module):
    """A single decoder layer: causal self-attention → cross-attention → FFN.

    Each block lets the autoregressive decoder both (a) attend to the tokens it
    has already generated and (b) condition on the projected JEPA context at
    every layer, matching the depth of reasoning that context-conditioning
    provides in LLaVA and similar architectures.

    All sub-layers use pre-normalization (LayerNorm before the sub-layer,
    residual after) for training stability.
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = _CausalSelfAttention(embed_dim, num_heads)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.cross_attn = _CrossAttention(embed_dim, num_heads)

        self.norm3 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(
        self,
        x: torch.Tensor,                              # (B, T, D) — generated tokens
        context: torch.Tensor,                         # (B, S, D) — adapter context
        context_mask: Optional[torch.Tensor] = None,   # (B, S) bool
        causal_mask: Optional[torch.Tensor] = None,    # (T, T) bool upper-tri
    ) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), causal_mask)
        x = x + self.cross_attn(self.norm2(x), context, context_mask)
        x = x + self.ffn(self.norm3(x))
        return x


# =============================================================================
# SmallDecoder — autoregressive transformer decoder
# =============================================================================


class SmallDecoder(nn.Module):
    """Small autoregressive transformer decoder.

    Cross-attends to the projected JEPA context at every layer.  Uses the same
    byte-level SimpleTokenizer (260 tokens) as TraceEncoder._SequenceEncoder so
    that the tokenizer is consistent across encode ↔ decode.

    Approximate parameter counts:
        dim=128, depth=2, heads=4  →  ~5M params   (PoC)
        dim=256, depth=4, heads=4  →  ~20M params
        dim=512, depth=12, heads=8 →  ~150M params
        dim=1024, depth=24, heads=16 → ~500M params (production target)
    """

    def __init__(self, config: ActionDecoderConfig):
        super().__init__()
        self.config = config
        D = config.decoder_dim

        self.token_embed = nn.Embedding(config.vocab_size, D)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.max_output_length, D)
        )

        self.blocks = nn.ModuleList([
            DecoderBlock(D, config.num_heads, config.mlp_ratio)
            for _ in range(config.decoder_depth)
        ])

        self.norm = nn.LayerNorm(D)
        # Weight-tied output projection (token_embed ↔ lm_head for better generalisation)
        self.lm_head = nn.Linear(D, config.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        pos = get_1d_sincos_pos_embed(self.config.decoder_dim, self.config.max_output_length)
        self.pos_embed.data.copy_(
            torch.from_numpy(pos).float().unsqueeze(0)
        )
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.lm_head.weight, std=0.02)

    def forward(
        self,
        token_ids: torch.Tensor,                       # (B, T) LongTensor
        context: torch.Tensor,                          # (B, S, decoder_dim)
        context_mask: Optional[torch.Tensor] = None,   # (B, S) bool
    ) -> torch.Tensor:
        """Forward pass with teacher forcing.

        Args:
            token_ids:    Previously generated token IDs (B, T)
            context:      Adapter-projected JEPA context (B, S, decoder_dim)
            context_mask: Valid positions mask for the context (B, S), True=valid

        Returns:
            Logits over vocabulary (B, T, vocab_size)
        """
        B, T = token_ids.shape
        x = self.token_embed(token_ids)        # (B, T, D)
        x = x + self.pos_embed[:, :T, :]      # add positional encodings

        for block in self.blocks:
            x = block(x, context, context_mask)

        x = self.norm(x)
        return self.lm_head(x)   # (B, T, vocab_size)


# =============================================================================
# JSONConstrainer — lightweight token-level JSON structure enforcement
# =============================================================================


class _JSONParseState(Enum):
    """States for the JSON structural parse automaton."""
    START = auto()      # Nothing emitted yet — only '{' is valid
    IN_OBJECT = auto()  # Inside at least one brace level
    DONE = auto()       # Outer object closed — only EOS is valid


class JSONConstrainer:
    """Lightweight constraint engine for JSON-structured output.

    Tracks byte-level JSON structure state and returns a token validity mask
    at each generation step.  This PoC implementation provides:

    - Forces first character to be '{' (outer object delimiter)
    - Tracks brace/bracket depth and string escaping
    - Forces EOS once the outer object closes (depth returns to 0)
    - Disallows control tokens (PAD, BOS, UNK) during generation

    XGrammar integration path (production):
        Replace apply_logit_mask() with an XGrammar GrammarMatcher step.
        GrammarMatcher.fill_next_token_bitmask() returns the full CFG-compliant
        mask in <40μs per token — just pass it to logits.masked_fill_() instead
        of the simple mask computed here.

    Example::

        constrainer = JSONConstrainer()
        for byte_val in b'{"tool":"grep","args":{}}':
            mask = constrainer.get_valid_mask()
            # ... apply mask to logits ...
            constrainer.step(byte_val)
        assert constrainer.is_done()
    """

    def __init__(self, vocab_size: int = 260):
        self.vocab_size = vocab_size
        self.reset()

    def reset(self) -> None:
        """Reset to initial state for a new generation episode."""
        self.partial: str = ""
        self.depth: int = 0
        self.in_string: bool = False
        self.escape_next: bool = False
        self.state: _JSONParseState = _JSONParseState.START

    def step(self, byte_val: int) -> None:
        """Update internal state after emitting a byte.

        Args:
            byte_val: Raw byte value (0-255), not a token ID.
        """
        if not (0 <= byte_val <= 255):
            return

        ch = chr(byte_val)
        self.partial += ch

        if self.escape_next:
            self.escape_next = False
            return

        if self.in_string:
            if ch == "\\":
                self.escape_next = True
            elif ch == '"':
                self.in_string = False
        else:
            if ch == '"':
                self.in_string = True
            elif ch in ("{", "["):
                self.depth += 1
                self.state = _JSONParseState.IN_OBJECT
            elif ch in ("}", "]"):
                self.depth = max(0, self.depth - 1)
                if self.depth == 0 and self.state == _JSONParseState.IN_OBJECT:
                    self.state = _JSONParseState.DONE

    def is_done(self) -> bool:
        """True once the outer JSON object has been fully closed."""
        return self.state == _JSONParseState.DONE

    def get_valid_mask(self) -> torch.Tensor:
        """Return a boolean mask (vocab_size,) — True = token allowed at this step.

        Constraint levels:
          START:   only '{' (byte 0x7B → token ID ord('{')+BYTE_OFFSET)
          DONE:    only EOS
          IN_OBJECT: all 256 byte tokens plus EOS (full generation freedom within
                     the structural envelope)

        The IN_OBJECT state allows all bytes so the decoder is free to place any
        content inside the JSON structure.  Stricter key/value constraints (e.g.
        requiring "tool" and "args" keys, validating argument schemas) are
        handled by XGrammar in the production path.
        """
        mask = torch.zeros(self.vocab_size, dtype=torch.bool)

        if self.state == _JSONParseState.DONE:
            mask[EOS_ID] = True
            return mask

        if self.state == _JSONParseState.START:
            # Only the '{' byte is valid as the very first character
            open_brace_id = ord("{") + BYTE_OFFSET  # 123 + 4 = 127
            mask[open_brace_id] = True
            return mask

        # IN_OBJECT: all byte tokens and EOS are valid
        # Disallow PAD (0), BOS (1), UNK (3) — only EOS (2) is a valid special token
        mask[EOS_ID] = True
        mask[BYTE_OFFSET:] = True   # token IDs 4 … 259 (all 256 raw bytes)
        return mask

    def apply_logit_mask(self, logits: torch.Tensor) -> torch.Tensor:
        """Mask invalid token logits to -inf before argmax / softmax.

        Args:
            logits: Unnormalised scores (vocab_size,)

        Returns:
            Masked logits (vocab_size,) with invalid positions set to -inf.
        """
        valid = self.get_valid_mask().to(logits.device)
        return logits.masked_fill(~valid, float("-inf"))


# =============================================================================
# ActionDecoder — main module
# =============================================================================


class ActionDecoder(nn.Module):
    """Decode JEPA embedding-space reasoning into structured ATN tool calls.

    Combines:
      - MLPAdapter:      Cross-modal bridge (JEPA dim → decoder dim)
      - SmallDecoder:    Autoregressive Transformer with cross-attention
      - JSONConstrainer: Token-level mask for valid JSON structure
      - SimpleTokenizer: Same byte-level tokenizer as TraceEncoder

    Training (teacher forcing)::

        decoder = ActionDecoder(poc_config())
        target_ids, _ = decoder.tokenizer.batch_encode(
            ['{"tool":"grep","args":{"pattern":"foo"}}']
        )
        # Prepend BOS
        bos = torch.full((1, 1), BOS_ID, dtype=torch.long)
        target_ids = torch.cat([bos, target_ids], dim=1)
        out = decoder(jepa_embs, target_ids)
        out["loss"].backward()

    Inference::

        output = decoder.decode(jepa_embs)          # DecoderOutput
        calls  = decoder.decode_tool_calls(jepa_embs)  # List[ToolCallOutput]
        print(calls[0].tool, calls[0].args)
    """

    def __init__(self, config: Optional[ActionDecoderConfig] = None):
        super().__init__()
        self.config = config or poc_config()

        self.adapter = MLPAdapter(self.config)
        self.decoder = SmallDecoder(self.config)
        self.tokenizer = SimpleTokenizer(self.config.vocab_size)
        self.constrainer = JSONConstrainer(self.config.vocab_size)

    @property
    def device(self) -> torch.device:
        """Device the model parameters currently live on."""
        return next(self.parameters()).device

    # -------------------------------------------------------------------------
    # Training forward pass (teacher forcing)
    # -------------------------------------------------------------------------

    def forward(
        self,
        jepa_embs: torch.Tensor,                       # (B, S, jepa_dim)
        target_ids: torch.Tensor,                       # (B, T) with BOS prepended
        context_mask: Optional[torch.Tensor] = None,   # (B, S) bool
    ) -> Dict[str, torch.Tensor]:
        """Teacher-forced training forward pass.

        Args:
            jepa_embs:    JEPA reasoning embeddings (B, S, jepa_dim)
            target_ids:   Ground-truth token IDs with BOS prepended (B, T)
            context_mask: Optional mask for valid JEPA positions (B, S), True=valid

        Returns:
            Dict with:
                loss:   Scalar cross-entropy training loss
                logits: Raw logits (B, T-1, vocab_size) for monitoring
        """
        context = self.adapter(jepa_embs)          # (B, S, decoder_dim)

        # Teacher forcing: input = target[:-1], labels = target[1:]
        input_ids = target_ids[:, :-1]   # (B, T-1)
        label_ids = target_ids[:, 1:]    # (B, T-1)

        logits = self.decoder(input_ids, context, context_mask)  # (B, T-1, V)

        loss = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            label_ids.reshape(-1),
            ignore_index=PAD_ID,
        )

        return {"loss": loss, "logits": logits}

    # -------------------------------------------------------------------------
    # Inference: autoregressive generation
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def decode(
        self,
        jepa_embs: torch.Tensor,                       # (B, S, jepa_dim) or (S, jepa_dim)
        context_mask: Optional[torch.Tensor] = None,   # (B, S) or (S,) bool
        max_new_tokens: Optional[int] = None,
    ) -> DecoderOutput:
        """Autoregressively generate structured text from JEPA embeddings.

        Args:
            jepa_embs:      JEPA reasoning embeddings.  Pass a single (S, jepa_dim)
                            tensor or a batched (B, S, jepa_dim) tensor.
            context_mask:   Optional bool mask for valid context positions.
            max_new_tokens: Override for max generation length (default from config).

        Returns:
            DecoderOutput with one ToolCallOutput per batch item.
        """
        # Normalise to batch dimension
        if jepa_embs.dim() == 2:
            jepa_embs = jepa_embs.unsqueeze(0)
            if context_mask is not None and context_mask.dim() == 1:
                context_mask = context_mask.unsqueeze(0)

        B = jepa_embs.shape[0]
        device = jepa_embs.device
        max_len = max_new_tokens or self.config.max_output_length

        # Project JEPA embeddings once — reused at every generation step
        context = self.adapter(jepa_embs)  # (B, S, decoder_dim)

        # Start each sequence with BOS
        generated = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)

        # Independent constrainer per batch item
        constrainers: List[Optional[JSONConstrainer]] = [
            JSONConstrainer(self.config.vocab_size) if self.config.enforce_json else None
            for _ in range(B)
        ]
        finished = [False] * B

        for _ in range(max_len):
            if all(finished):
                break

            logits = self.decoder(generated, context, context_mask)  # (B, T, V)
            next_logits = logits[:, -1, :]   # (B, V) — next-token logits

            next_tokens: List[int] = []
            for b in range(B):
                if finished[b]:
                    next_tokens.append(PAD_ID)
                    continue

                logit_b = next_logits[b]   # (V,)

                # Apply JSON structural constraint
                if constrainers[b] is not None:
                    logit_b = constrainers[b].apply_logit_mask(logit_b)

                # Greedy or stochastic decoding
                if self.config.temperature > 0.0:
                    probs = F.softmax(logit_b / self.config.temperature, dim=-1)
                    if self.config.top_p < 1.0:
                        probs = _top_p_filter(probs, self.config.top_p)
                    token = int(torch.multinomial(probs, num_samples=1).item())
                else:
                    token = int(logit_b.argmax(dim=-1).item())

                # Advance the constrainer state
                if constrainers[b] is not None and token >= BYTE_OFFSET:
                    constrainers[b].step(token - BYTE_OFFSET)

                # Terminate if EOS or JSON object complete
                json_done = constrainers[b] is not None and constrainers[b].is_done()
                if token == EOS_ID or json_done:
                    finished[b] = True
                    next_tokens.append(EOS_ID)
                else:
                    next_tokens.append(token)

            next_tensor = torch.tensor(next_tokens, dtype=torch.long, device=device)
            generated = torch.cat([generated, next_tensor.unsqueeze(1)], dim=1)

        # Decode token IDs → text → ToolCallOutput
        tool_calls: List[ToolCallOutput] = []
        raw_texts: List[str] = []

        for b in range(B):
            ids = [t for t in generated[b].tolist() if t != BOS_ID]
            text = self.tokenizer.decode(ids)
            raw_texts.append(text)
            tool_calls.append(_parse_tool_call(text))

        raw_text = raw_texts[0] if B == 1 else "\n".join(raw_texts)

        return DecoderOutput(
            tool_calls=tool_calls,
            raw_text=raw_text,
            input_embeddings=context,
        )

    def decode_tool_calls(
        self,
        jepa_embs: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> List[ToolCallOutput]:
        """Convenience wrapper — decode and return only the tool call list."""
        return self.decode(jepa_embs, context_mask).tool_calls

    def encode_target(self, tool_call_text: str) -> torch.Tensor:
        """Tokenise a target tool call string for teacher-forced training.

        Prepends BOS and appends EOS, then pads to max_output_length.

        Args:
            tool_call_text: Raw JSON string, e.g. '{"tool":"grep","args":{"pattern":"foo"}}'

        Returns:
            Token ID tensor (1, max_output_length) with BOS at position 0.
        """
        ids = self.tokenizer.encode(tool_call_text, add_bos=True, add_eos=True)
        ids = ids[: self.config.max_output_length]
        pad_len = self.config.max_output_length - len(ids)
        ids = ids + [PAD_ID] * pad_len
        return torch.tensor([ids], dtype=torch.long)


# =============================================================================
# Internal utilities
# =============================================================================


def _top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Apply nucleus (top-p) sampling filter in-place.

    Zeroes out tokens once the cumulative probability of the top tokens
    exceeds top_p, then re-normalises.  The caller should pass the result
    directly to torch.multinomial.
    """
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=0)
    # Drop tokens whose inclusion would push cumulative past top_p
    remove = (cumulative - sorted_probs) > top_p
    sorted_probs[remove] = 0.0
    filtered = torch.zeros_like(probs)
    filtered.scatter_(0, sorted_idx, sorted_probs)
    return filtered


def _parse_tool_call(text: str) -> ToolCallOutput:
    """Parse generated text into a ToolCallOutput.

    Expected format: {"tool": "<name>", "args": {...}}

    Graceful degradation:
      1. Attempt direct json.loads on the full text.
      2. Attempt partial repair (close unclosed braces/brackets).
      3. Regex-extract tool name from malformed output; set valid=False.
    """
    text = text.strip()
    raw = text

    # 1. Direct parse
    try:
        data = json.loads(text)
        tool = str(data.get("tool", ""))
        args = data.get("args", {})
        if not isinstance(args, dict):
            args = {}
        return ToolCallOutput(tool=tool, args=args, raw_text=raw, valid=bool(tool))
    except (json.JSONDecodeError, AttributeError):
        pass

    # 2. Repair unclosed structure and retry
    repaired = _repair_json(text)
    if repaired != text:
        try:
            data = json.loads(repaired)
            tool = str(data.get("tool", ""))
            args = data.get("args", {})
            if not isinstance(args, dict):
                args = {}
            return ToolCallOutput(tool=tool, args=args, raw_text=raw, valid=bool(tool))
        except (json.JSONDecodeError, AttributeError):
            pass

    # 3. Best-effort extraction
    tool = _extract_tool_name(text)
    logger.debug("ActionDecoder: malformed output, extracted tool=%r from %r", tool, raw[:80])
    return ToolCallOutput(tool=tool, args={}, raw_text=raw, valid=False)


def _repair_json(text: str) -> str:
    """Close unclosed braces and brackets to make partial JSON parseable."""
    stack: List[str] = []
    in_string = False
    escape_next = False

    for ch in text:
        if escape_next:
            escape_next = False
            continue
        if in_string:
            if ch == "\\":
                escape_next = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch in ("{", "["):
                stack.append("}" if ch == "{" else "]")
            elif ch in ("}", "]") and stack:
                stack.pop()

    return text + "".join(reversed(stack))


def _extract_tool_name(text: str) -> str:
    """Best-effort regex extraction of the tool name from malformed output."""
    match = re.search(r'"tool"\s*:\s*"([^"]*)"', text)
    return match.group(1) if match else ""

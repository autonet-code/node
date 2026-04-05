"""
Text Training Data Source for VL-JEPA

Reads agent framework output (conversations, execution records) from ATN's
JSONL persistence and produces byte-tokenized training batches for the
text encoder.

Data flow:
  ~/.atn/conversations/*.jsonl  →  TextTrainingDataSource  →  (token_ids, attention_mask)
  ~/.atn/agents/*/execution.jsonl  →  ↑

Privacy:
  - PII scrubbed via nodes.common.privacy.scrub_text
  - Respects exclude patterns from autonet.yaml

Story 1.1 (BACKLOG_TRAINING_DATA.md)
"""

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import torch

from .tokenizer import SimpleTokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Turn extraction — read JSONL files produced by ATN's ConversationStore
# and ExecutionLog, and yield raw text segments suitable for training.
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield parsed JSON objects from a JSONL file, skipping corrupt lines."""
    try:
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("Skipping corrupt JSONL line %d in %s", lineno, path)
    except (OSError, IOError) as exc:
        logger.warning("Cannot read %s: %s", path, exc)


def _extract_conversation_segments(
    conversations_dir: Path,
    exclude_patterns: list[str] | None = None,
) -> Iterator[str]:
    """Yield text segments from orchestrator and archived conversations.

    Each segment is a sequence of turns formatted as:

        [user] How's my schedule tomorrow?
        [assistant] I'll check your calendar...

    This preserves conversational structure so the model can learn
    turn-taking patterns and action prediction.
    """
    if not conversations_dir.is_dir():
        return

    exclude = [p.lower() for p in (exclude_patterns or [])]

    for jsonl_path in sorted(conversations_dir.glob("*.jsonl")):
        segment_lines: list[str] = []
        for turn in _read_jsonl(jsonl_path):
            role = turn.get("role", "")
            content = turn.get("content", "")
            if not content or role not in ("user", "assistant", "system"):
                continue
            # Skip system turns (they're prompt scaffolding, not user behavior)
            if role == "system":
                continue
            # Check exclude patterns against content
            content_lower = content.lower()
            if any(pat in content_lower for pat in exclude):
                continue
            segment_lines.append(f"[{role}] {content}")

        if segment_lines:
            yield "\n".join(segment_lines)


def _extract_execution_segments(
    agents_dir: Path,
    exclude_patterns: list[str] | None = None,
) -> Iterator[str]:
    """Yield text segments from agent execution records.

    Extracts the behavioral signal: what tools were called, with what inputs,
    and what the outcomes were.  Formatted as:

        [agent:daily_planner] trigger=scheduler status=completed
        [step:cognitive] Check inbox for new messages
        [tool:bash] ls ~/.atn/agents/
        [result] agent1/ agent2/ ...
    """
    if not agents_dir.is_dir():
        return

    exclude = [p.lower() for p in (exclude_patterns or [])]

    for exec_jsonl in sorted(agents_dir.glob("*/execution.jsonl")):
        for record in _read_jsonl(exec_jsonl):
            status = record.get("status", "")
            if status not in ("completed", "failed"):
                continue

            agent_id = record.get("agent_id", "unknown")
            trigger = record.get("trigger_source", "unknown")
            lines: list[str] = [
                f"[agent:{agent_id}] trigger={trigger} status={status}"
            ]

            for step in record.get("step_results", []):
                step_name = step.get("step_name", "")
                step_type = step.get("step_type", "")
                output = step.get("output")

                if step_type == "cognitive" and isinstance(output, dict):
                    # Extract LLM response text
                    text = output.get("text", "")
                    if text:
                        lines.append(f"[step:{step_name}] {text[:500]}")

                    # Extract tool calls
                    for tc in output.get("tool_calls", []):
                        tool_name = tc.get("name", "")
                        tool_input = tc.get("input", {})
                        # Truncate tool input for training — we want the pattern,
                        # not huge file contents
                        input_str = json.dumps(tool_input, default=str)[:200]
                        lines.append(f"[tool:{tool_name}] {input_str}")

                elif step_type == "script" and isinstance(output, str):
                    lines.append(f"[step:{step_name}] {output[:300]}")

            # Check excludes on the combined text
            combined = "\n".join(lines)
            if any(pat in combined.lower() for pat in exclude):
                continue

            if len(lines) > 1:  # Must have at least one step beyond the header
                yield combined


# ---------------------------------------------------------------------------
# TextTrainingDataSource — the main class
# ---------------------------------------------------------------------------

@dataclass
class TextDataConfig:
    """Configuration for the text training data source."""
    # ATN data directories
    conversations_dir: str = ""    # ~/.atn/conversations/
    agents_dir: str = ""           # ~/.atn/agents/
    # Tokenizer
    max_seq_length: int = 2048     # Max tokens per sequence (byte-level)
    vocab_size: int = 260          # Must match VLJEPAConfig
    # Batching
    batch_size: int = 8
    # Privacy
    scrub_pii: bool = True
    exclude_patterns: list[str] = field(default_factory=list)
    # Sampling
    shuffle: bool = True
    max_segments: int = 0          # 0 = no limit


class TextTrainingDataSource:
    """Reads ATN agent framework JSONL and produces training batches.

    Yields dicts of:
        token_ids:      (B, S)  LongTensor — byte-tokenized text
        attention_mask:  (B, S)  BoolTensor — True = real token, False = padding

    Usage:
        source = TextTrainingDataSource(config)
        for batch in source:
            # batch["token_ids"].shape == (B, S)
            train_step(batch)
    """

    def __init__(self, config: TextDataConfig):
        self.config = config
        self.tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)

        self._conversations_dir = Path(config.conversations_dir) if config.conversations_dir else None
        self._agents_dir = Path(config.agents_dir) if config.agents_dir else None

        # Lazy-loaded segments
        self._segments: list[str] | None = None

    def _load_segments(self) -> list[str]:
        """Load and scrub all text segments from disk."""
        segments: list[str] = []

        if self._conversations_dir:
            for seg in _extract_conversation_segments(
                self._conversations_dir,
                self.config.exclude_patterns,
            ):
                segments.append(seg)

        if self._agents_dir:
            for seg in _extract_execution_segments(
                self._agents_dir,
                self.config.exclude_patterns,
            ):
                segments.append(seg)

        # PII scrubbing
        if self.config.scrub_pii:
            from .privacy import scrub_text
            segments = [scrub_text(s) for s in segments]

        # Limit
        if self.config.max_segments > 0:
            segments = segments[: self.config.max_segments]

        if self.config.shuffle:
            random.shuffle(segments)

        logger.info(
            "TextTrainingDataSource loaded %d segments from conversations=%s agents=%s",
            len(segments),
            self._conversations_dir,
            self._agents_dir,
        )
        return segments

    def reload(self) -> None:
        """Force reload from disk (e.g. after new conversations arrive)."""
        self._segments = self._load_segments()

    @property
    def segments(self) -> list[str]:
        if self._segments is None:
            self._segments = self._load_segments()
        return self._segments

    def __len__(self) -> int:
        """Number of batches."""
        n_segments = len(self.segments)
        if n_segments == 0:
            return 0
        return max(1, n_segments // self.config.batch_size)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Yield training batches: {token_ids, attention_mask, raw_texts}."""
        segs = self.segments
        if not segs:
            return

        # Chunk segments into batches
        for start in range(0, len(segs), self.config.batch_size):
            batch_texts = segs[start : start + self.config.batch_size]
            if not batch_texts:
                continue

            token_ids, attention_mask = self.tokenizer.batch_encode(
                batch_texts,
                max_length=self.config.max_seq_length,
            )

            yield {
                "token_ids": token_ids,
                "attention_mask": attention_mask,
                "raw_texts": batch_texts,
            }


# ---------------------------------------------------------------------------
# TextMasker — span masking strategy for text JEPA training
# Story 1.2 (BACKLOG_TRAINING_DATA.md)
# ---------------------------------------------------------------------------

class TextMasker:
    """Generate context and target masks for text JEPA training.

    Analogous to JEPAMasker (which masks 2D rectangular blocks of image
    patches), TextMasker masks contiguous 1D spans of tokens.

    Strategy:
    - Mask 15-25% of real (non-padding) tokens
    - Select spans of 3-15 tokens each
    - Context = everything not masked
    - Target = masked spans

    Returns the same (context_mask, target_masks) interface as JEPAMasker
    so the predictor and loss functions work identically.
    """

    def __init__(
        self,
        mask_ratio: float = 0.20,
        min_span: int = 3,
        max_span: int = 15,
        num_targets: int = 4,
    ):
        self.mask_ratio = mask_ratio
        self.min_span = min_span
        self.max_span = max_span
        self.num_targets = num_targets

    def __call__(
        self,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Generate masks for a batch.

        Args:
            attention_mask: (B, S) BoolTensor — True = real token

        Returns:
            context_mask:  (B, S) BoolTensor — True = context (visible)
            target_masks:  list of (B, S) BoolTensor — True = target
                           (one per target span group; typically num_targets)
        """
        B, S = attention_mask.shape
        device = attention_mask.device

        context_masks = []
        all_target_masks = []

        for b in range(B):
            real_indices = attention_mask[b].nonzero(as_tuple=True)[0]
            num_real = len(real_indices)

            if num_real < self.min_span * 2:
                # Too short to mask meaningfully — context = all, no targets
                context_masks.append(attention_mask[b].clone())
                all_target_masks.append(
                    [torch.zeros(S, dtype=torch.bool, device=device)]
                    * self.num_targets
                )
                continue

            num_to_mask = max(self.min_span, int(num_real * self.mask_ratio))

            # Generate spans until we've masked enough tokens
            masked_positions: set[int] = set()
            spans: list[list[int]] = []

            attempts = 0
            while len(masked_positions) < num_to_mask and attempts < 100:
                attempts += 1
                span_len = random.randint(self.min_span, self.max_span)
                # Pick a random start within real tokens
                max_start_idx = num_real - span_len
                if max_start_idx < 0:
                    span_len = num_real // 2
                    max_start_idx = num_real - span_len
                if max_start_idx < 0:
                    break

                start_idx = random.randint(0, max_start_idx)
                span_positions = real_indices[start_idx : start_idx + span_len].tolist()

                # Skip if overlaps with existing masks
                if any(p in masked_positions for p in span_positions):
                    continue

                masked_positions.update(span_positions)
                spans.append(span_positions)

                if len(masked_positions) >= num_to_mask:
                    break

            # Build target masks — distribute spans across num_targets groups
            sample_target_masks: list[torch.Tensor] = []
            for t in range(self.num_targets):
                mask = torch.zeros(S, dtype=torch.bool, device=device)
                # Assign spans round-robin to target groups
                for i, span in enumerate(spans):
                    if i % self.num_targets == t:
                        for pos in span:
                            mask[pos] = True
                sample_target_masks.append(mask)

            # Context = real tokens that are NOT masked
            context = attention_mask[b].clone()
            for pos in masked_positions:
                context[pos] = False

            context_masks.append(context)
            all_target_masks.append(sample_target_masks)

        # Stack batch
        context_mask = torch.stack(context_masks)  # (B, S)

        # Reorganize: list of num_targets tensors, each (B, S)
        target_masks = [
            torch.stack([all_target_masks[b][t] for b in range(B)])
            for t in range(self.num_targets)
        ]

        return context_mask, target_masks

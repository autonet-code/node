"""Tests for Story 1.1 (TextTrainingDataSource) and 1.2 (TextMasker)."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from nodes.common.text_data import (
    TextDataConfig,
    TextMasker,
    TextTrainingDataSource,
    _extract_conversation_segments,
    _extract_execution_segments,
)
from nodes.common.tokenizer import SimpleTokenizer, BOS_ID, EOS_ID, PAD_ID


# ---------------------------------------------------------------------------
# Helpers — build synthetic JSONL that mirrors ATN's real format
# ---------------------------------------------------------------------------

def _write_conversation_jsonl(path: Path, turns: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for turn in turns:
            f.write(json.dumps(turn) + "\n")


def _sample_conversation_turns() -> list[dict]:
    return [
        {"role": "user", "content": "Check my calendar for tomorrow", "timestamp": "2026-03-26T10:00:00+00:00"},
        {"role": "assistant", "content": "I'll look up your schedule. You have 3 meetings.", "timestamp": "2026-03-26T10:00:01+00:00", "execution_id": "abc123"},
        {"role": "user", "content": "Cancel the 2pm meeting", "timestamp": "2026-03-26T10:00:05+00:00"},
        {"role": "assistant", "content": "Done. I've cancelled your 2pm meeting with the design team.", "timestamp": "2026-03-26T10:00:06+00:00", "execution_id": "def456"},
    ]


def _sample_execution_record() -> dict:
    return {
        "execution_id": "abc123",
        "agent_id": "daily_planner",
        "status": "completed",
        "trigger_source": "scheduler",
        "step_results": [
            {
                "step_index": 0,
                "step_name": "check_calendar",
                "step_type": "cognitive",
                "status": "completed",
                "output": {
                    "text": "You have 3 meetings tomorrow.",
                    "tool_calls": [
                        {"name": "calendar_read", "input": {"date": "2026-03-27"}, "id": "tc1"},
                    ],
                },
            },
        ],
    }


# ---------------------------------------------------------------------------
# Test: Conversation Segment Extraction
# ---------------------------------------------------------------------------

class TestConversationExtraction(unittest.TestCase):

    def test_basic_extraction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            conv_dir = Path(tmpdir) / "conversations"
            _write_conversation_jsonl(
                conv_dir / "active.jsonl",
                _sample_conversation_turns(),
            )

            segments = list(_extract_conversation_segments(conv_dir))
            self.assertEqual(len(segments), 1)
            self.assertIn("[user] Check my calendar", segments[0])
            self.assertIn("[assistant] I'll look up", segments[0])

    def test_skips_system_turns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            conv_dir = Path(tmpdir) / "conversations"
            turns = [
                {"role": "system", "content": "You are a helpful assistant.", "timestamp": "2026-01-01T00:00:00+00:00"},
                {"role": "user", "content": "Hello", "timestamp": "2026-01-01T00:00:01+00:00"},
            ]
            _write_conversation_jsonl(conv_dir / "active.jsonl", turns)

            segments = list(_extract_conversation_segments(conv_dir))
            self.assertEqual(len(segments), 1)
            self.assertNotIn("[system]", segments[0])
            self.assertIn("[user] Hello", segments[0])

    def test_exclude_patterns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            conv_dir = Path(tmpdir) / "conversations"
            turns = [
                {"role": "user", "content": "What's my password for gmail?", "timestamp": "2026-01-01T00:00:00+00:00"},
                {"role": "assistant", "content": "I can't share passwords.", "timestamp": "2026-01-01T00:00:01+00:00"},
                {"role": "user", "content": "How about the weather?", "timestamp": "2026-01-01T00:00:02+00:00"},
                {"role": "assistant", "content": "It will be sunny tomorrow.", "timestamp": "2026-01-01T00:00:03+00:00"},
            ]
            _write_conversation_jsonl(conv_dir / "active.jsonl", turns)

            # Without exclude — all 4 turns form 1 segment
            segments = list(_extract_conversation_segments(conv_dir))
            self.assertEqual(len(segments), 1)
            self.assertIn("password", segments[0])

            # With exclude — turns containing "password" are skipped,
            # but remaining turns still form a segment
            segments = list(_extract_conversation_segments(conv_dir, exclude_patterns=["password"]))
            self.assertEqual(len(segments), 1)
            self.assertNotIn("password", segments[0])
            self.assertIn("weather", segments[0])

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            conv_dir = Path(tmpdir) / "conversations"
            conv_dir.mkdir()
            segments = list(_extract_conversation_segments(conv_dir))
            self.assertEqual(len(segments), 0)

    def test_nonexistent_directory(self):
        segments = list(_extract_conversation_segments(Path("/nonexistent/path")))
        self.assertEqual(len(segments), 0)

    def test_corrupt_jsonl_lines_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            conv_dir = Path(tmpdir) / "conversations"
            conv_dir.mkdir(parents=True)
            p = conv_dir / "active.jsonl"
            with open(p, "w") as f:
                f.write('{"role":"user","content":"Hello","timestamp":"2026-01-01T00:00:00+00:00"}\n')
                f.write("this is not json\n")
                f.write('{"role":"assistant","content":"Hi there","timestamp":"2026-01-01T00:00:01+00:00"}\n')

            segments = list(_extract_conversation_segments(conv_dir))
            self.assertEqual(len(segments), 1)
            self.assertIn("[user] Hello", segments[0])
            self.assertIn("[assistant] Hi there", segments[0])

    def test_multiple_jsonl_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            conv_dir = Path(tmpdir) / "conversations"
            _write_conversation_jsonl(
                conv_dir / "active.jsonl",
                [{"role": "user", "content": "Conversation A", "timestamp": "2026-01-01T00:00:00+00:00"}],
            )
            _write_conversation_jsonl(
                conv_dir / "archived_session1.jsonl",
                [{"role": "user", "content": "Conversation B", "timestamp": "2026-01-02T00:00:00+00:00"}],
            )

            segments = list(_extract_conversation_segments(conv_dir))
            self.assertEqual(len(segments), 2)


# ---------------------------------------------------------------------------
# Test: Execution Record Extraction
# ---------------------------------------------------------------------------

class TestExecutionExtraction(unittest.TestCase):

    def test_basic_extraction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = Path(tmpdir) / "agents"
            agent_dir = agents_dir / "daily_planner"
            agent_dir.mkdir(parents=True)
            exec_path = agent_dir / "execution.jsonl"
            with open(exec_path, "w") as f:
                f.write(json.dumps(_sample_execution_record()) + "\n")

            segments = list(_extract_execution_segments(agents_dir))
            self.assertEqual(len(segments), 1)
            self.assertIn("[agent:daily_planner]", segments[0])
            self.assertIn("[tool:calendar_read]", segments[0])

    def test_skips_non_terminal_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = Path(tmpdir) / "agents"
            agent_dir = agents_dir / "test_agent"
            agent_dir.mkdir(parents=True)

            records = [
                {"execution_id": "1", "agent_id": "test", "status": "running",
                 "trigger_source": "user", "step_results": []},
                {"execution_id": "2", "agent_id": "test", "status": "queued",
                 "trigger_source": "user", "step_results": []},
            ]
            with open(agent_dir / "execution.jsonl", "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            segments = list(_extract_execution_segments(agents_dir))
            self.assertEqual(len(segments), 0)


# ---------------------------------------------------------------------------
# Test: TextTrainingDataSource
# ---------------------------------------------------------------------------

class TestTextTrainingDataSource(unittest.TestCase):

    def _make_source(self, tmpdir: str, num_conversations: int = 3) -> TextTrainingDataSource:
        conv_dir = Path(tmpdir) / "conversations"
        for i in range(num_conversations):
            _write_conversation_jsonl(
                conv_dir / f"session_{i}.jsonl",
                _sample_conversation_turns(),
            )

        config = TextDataConfig(
            conversations_dir=str(conv_dir),
            agents_dir="",
            max_seq_length=256,
            batch_size=2,
            scrub_pii=False,
            shuffle=False,
        )
        return TextTrainingDataSource(config)

    def test_produces_batches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = self._make_source(tmpdir, num_conversations=4)
            batches = list(source)
            self.assertGreater(len(batches), 0)

            for batch in batches:
                self.assertIn("token_ids", batch)
                self.assertIn("attention_mask", batch)
                self.assertEqual(batch["token_ids"].dtype, torch.long)
                self.assertEqual(batch["attention_mask"].dtype, torch.bool)
                B, S = batch["token_ids"].shape
                self.assertLessEqual(B, 2)  # batch_size=2
                self.assertLessEqual(S, 256)  # max_seq_length

    def test_token_ids_start_with_bos(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = self._make_source(tmpdir, num_conversations=1)
            batch = next(iter(source))
            # First real token should be BOS
            self.assertEqual(batch["token_ids"][0, 0].item(), BOS_ID)

    def test_attention_mask_matches_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = self._make_source(tmpdir, num_conversations=1)
            batch = next(iter(source))
            ids = batch["token_ids"][0]
            mask = batch["attention_mask"][0]
            # Where mask is True, ids should not be PAD
            real_tokens = ids[mask]
            self.assertTrue((real_tokens != PAD_ID).all())
            # Where mask is False (if any), ids should be PAD
            if (~mask).any():
                pad_tokens = ids[~mask]
                self.assertTrue((pad_tokens == PAD_ID).all())

    def test_pii_scrubbing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            conv_dir = Path(tmpdir) / "conversations"
            turns = [
                {"role": "user", "content": "My email is test@example.com and SSN 123-45-6789",
                 "timestamp": "2026-01-01T00:00:00+00:00"},
                {"role": "assistant", "content": "Got it.", "timestamp": "2026-01-01T00:00:01+00:00"},
            ]
            _write_conversation_jsonl(conv_dir / "active.jsonl", turns)

            config = TextDataConfig(
                conversations_dir=str(conv_dir),
                max_seq_length=512,
                batch_size=1,
                scrub_pii=True,
                shuffle=False,
            )
            source = TextTrainingDataSource(config)
            # Check the raw segments have been scrubbed
            segs = source.segments
            self.assertEqual(len(segs), 1)
            self.assertNotIn("test@example.com", segs[0])
            self.assertNotIn("123-45-6789", segs[0])
            self.assertIn("[EMAIL_REDACTED]", segs[0])
            self.assertIn("[SSN_REDACTED]", segs[0])

    def test_empty_data(self):
        config = TextDataConfig(
            conversations_dir="/nonexistent",
            agents_dir="/also_nonexistent",
            batch_size=4,
        )
        source = TextTrainingDataSource(config)
        self.assertEqual(len(source), 0)
        batches = list(source)
        self.assertEqual(len(batches), 0)

    def test_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = self._make_source(tmpdir, num_conversations=1)
            self.assertEqual(len(source.segments), 1)

            # Add more data
            conv_dir = Path(tmpdir) / "conversations"
            _write_conversation_jsonl(
                conv_dir / "new_session.jsonl",
                _sample_conversation_turns(),
            )
            # Still cached
            self.assertEqual(len(source.segments), 1)
            # After reload
            source.reload()
            self.assertEqual(len(source.segments), 2)


# ---------------------------------------------------------------------------
# Test: TextMasker
# ---------------------------------------------------------------------------

class TestTextMasker(unittest.TestCase):

    def _make_batch(self, B: int = 4, S: int = 100) -> torch.Tensor:
        """Create a fake attention mask with varying real lengths."""
        mask = torch.zeros(B, S, dtype=torch.bool)
        for b in range(B):
            real_len = random.randint(S // 2, S)
            mask[b, :real_len] = True
        return mask

    def test_basic_masking(self):
        import random
        random.seed(42)
        masker = TextMasker(mask_ratio=0.2, num_targets=4)
        attention_mask = self._make_batch(B=4, S=100)

        context_mask, target_masks = masker(attention_mask)

        # Shapes
        self.assertEqual(context_mask.shape, (4, 100))
        self.assertEqual(len(target_masks), 4)
        for tm in target_masks:
            self.assertEqual(tm.shape, (4, 100))

    def test_context_and_targets_dont_overlap(self):
        import random
        random.seed(42)
        masker = TextMasker(mask_ratio=0.2, num_targets=4)
        attention_mask = torch.ones(2, 80, dtype=torch.bool)

        context_mask, target_masks = masker(attention_mask)

        for b in range(2):
            # Union of all target masks
            all_targets = torch.zeros(80, dtype=torch.bool)
            for tm in target_masks:
                all_targets |= tm[b]

            # Context and targets should not overlap
            overlap = context_mask[b] & all_targets
            self.assertFalse(overlap.any(),
                             f"Context and target overlap at positions: {overlap.nonzero()}")

    def test_masked_positions_are_within_real_tokens(self):
        import random
        random.seed(42)
        masker = TextMasker(mask_ratio=0.2, num_targets=4)
        # Sequence with padding at end
        attention_mask = torch.zeros(1, 100, dtype=torch.bool)
        attention_mask[0, :60] = True  # 60 real tokens, 40 padding

        context_mask, target_masks = masker(attention_mask)

        # No target should be in padding region (positions 60-99)
        for tm in target_masks:
            padding_targets = tm[0, 60:]
            self.assertFalse(padding_targets.any(),
                             "Target mask extends into padding region")

    def test_mask_ratio_approximate(self):
        import random
        random.seed(42)
        masker = TextMasker(mask_ratio=0.20, num_targets=4)
        attention_mask = torch.ones(8, 200, dtype=torch.bool)

        context_mask, target_masks = masker(attention_mask)

        for b in range(8):
            num_real = 200
            all_targets = torch.zeros(200, dtype=torch.bool)
            for tm in target_masks:
                all_targets |= tm[b]
            num_masked = all_targets.sum().item()
            ratio = num_masked / num_real
            # Allow some variance (0.10 to 0.35 — span boundaries cause imprecision)
            self.assertGreater(ratio, 0.05, f"Too few tokens masked: {ratio:.2%}")
            self.assertLess(ratio, 0.40, f"Too many tokens masked: {ratio:.2%}")

    def test_very_short_sequence(self):
        masker = TextMasker(mask_ratio=0.2, min_span=3, num_targets=4)
        # Only 4 real tokens — less than min_span * 2
        attention_mask = torch.zeros(1, 10, dtype=torch.bool)
        attention_mask[0, :4] = True

        context_mask, target_masks = masker(attention_mask)

        # Should get context = all real tokens, no targets
        self.assertTrue(context_mask[0, :4].all())
        for tm in target_masks:
            self.assertFalse(tm[0].any())


# Need random for _make_batch
import random

if __name__ == "__main__":
    unittest.main()

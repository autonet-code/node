"""Tests for Story 1.3 — TextJEPA end-to-end training.

Verifies that:
- TextJEPA model initializes and forward-passes correctly
- TextJEPATrainer.train_step() decreases loss over multiple steps
- Weight deltas are non-zero (actual learning happened)
- Evaluate returns verification-compatible metrics
- Works with real TextTrainingDataSource batches
"""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from nodes.common.text_jepa import TextJEPA, TextJEPAConfig, TextJEPATrainer
from nodes.common.text_data import (
    TextDataConfig,
    TextMasker,
    TextTrainingDataSource,
)
from nodes.common.tokenizer import SimpleTokenizer


class TestTextJEPAModel(unittest.TestCase):
    """Test the raw TextJEPA model."""

    def setUp(self):
        self.config = TextJEPAConfig(
            vocab_size=260,
            max_seq_length=64,
            embed_dim=64,
            num_heads=4,
            encoder_depth=2,
            predictor_depth=1,
            predictor_embed_dim=32,
        )

    def test_init(self):
        model = TextJEPA(self.config)
        # Context and target encoders should have same initial weights
        for p1, p2 in zip(model.context_encoder.parameters(),
                          model.target_encoder.parameters()):
            self.assertTrue(torch.equal(p1, p2))
        # Target encoder should be frozen
        for p in model.target_encoder.parameters():
            self.assertFalse(p.requires_grad)

    def test_forward(self):
        model = TextJEPA(self.config)
        B, S = 2, 32
        token_ids = torch.randint(4, 260, (B, S))
        attention_mask = torch.ones(B, S, dtype=torch.bool)

        masker = TextMasker(mask_ratio=0.2, num_targets=2)
        context_mask, target_masks = masker(attention_mask)

        outputs = model(token_ids, attention_mask, context_mask, target_masks)
        self.assertIn("loss", outputs)
        self.assertIn("predicted_embeddings", outputs)
        self.assertIn("target_embeddings", outputs)
        self.assertEqual(
            outputs["predicted_embeddings"].shape,
            outputs["target_embeddings"].shape,
        )

    def test_ema_update(self):
        model = TextJEPA(self.config)
        # Perturb context encoder
        with torch.no_grad():
            for p in model.context_encoder.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        # Before EMA: context != target
        diffs_before = []
        for p1, p2 in zip(model.context_encoder.parameters(),
                          model.target_encoder.parameters()):
            diffs_before.append((p1 - p2).abs().sum().item())
        self.assertGreater(sum(diffs_before), 0)

        # After EMA: target moves toward context
        model.update_target_encoder(momentum=0.5)
        diffs_after = []
        for p1, p2 in zip(model.context_encoder.parameters(),
                          model.target_encoder.parameters()):
            diffs_after.append((p1 - p2).abs().sum().item())
        self.assertLess(sum(diffs_after), sum(diffs_before))


class TestTextJEPATrainer(unittest.TestCase):
    """Test the trainer — loss decreases, metrics work."""

    def setUp(self):
        self.config = TextJEPAConfig(
            vocab_size=260,
            max_seq_length=64,
            embed_dim=64,
            num_heads=4,
            encoder_depth=2,
            predictor_depth=1,
            predictor_embed_dim=32,
        )

    def _make_batch(self, B=4, S=48) -> dict[str, torch.Tensor]:
        token_ids = torch.randint(4, 260, (B, S))
        attention_mask = torch.ones(B, S, dtype=torch.bool)
        return {"token_ids": token_ids, "attention_mask": attention_mask}

    def test_train_step_runs(self):
        trainer = TextJEPATrainer(self.config, device="cpu")
        optimizer = torch.optim.AdamW(
            [{"params": trainer.model.context_encoder.parameters()},
             {"params": trainer.model.predictor.parameters()}],
            lr=1e-3,
        )
        batch = self._make_batch()
        metrics = trainer.train_step(batch, optimizer)

        self.assertIn("loss", metrics)
        self.assertIn("cosine_similarity", metrics)
        self.assertGreater(metrics["loss"], 0)

    def test_loss_decreases(self):
        trainer = TextJEPATrainer(self.config, device="cpu")
        optimizer = torch.optim.AdamW(
            [{"params": trainer.model.context_encoder.parameters()},
             {"params": trainer.model.predictor.parameters()}],
            lr=1e-3,
        )

        # Use same batch repeatedly to test memorization
        batch = self._make_batch(B=8, S=48)
        losses = []
        for _ in range(30):
            m = trainer.train_step(batch, optimizer)
            losses.append(m["loss"])

        # Loss should decrease (first 5 avg > last 5 avg)
        early = sum(losses[:5]) / 5
        late = sum(losses[-5:]) / 5
        self.assertLess(late, early,
                        f"Loss did not decrease: early={early:.4f}, late={late:.4f}")

    def test_weight_deltas_nonzero(self):
        trainer = TextJEPATrainer(self.config, device="cpu")
        initial_weights = {k: v.clone() for k, v in trainer.model.state_dict().items()}

        optimizer = torch.optim.AdamW(
            [{"params": trainer.model.context_encoder.parameters()},
             {"params": trainer.model.predictor.parameters()}],
            lr=1e-3,
        )

        batch = self._make_batch()
        for _ in range(5):
            trainer.train_step(batch, optimizer)

        # Check that context encoder weights changed
        final_weights = trainer.model.state_dict()
        total_delta = 0.0
        for key in final_weights:
            if "target_encoder" not in key:
                delta = (final_weights[key] - initial_weights[key]).abs().sum().item()
                total_delta += delta

        self.assertGreater(total_delta, 0, "No weight change after training")

    def test_evaluate(self):
        trainer = TextJEPATrainer(self.config, device="cpu")
        batch = self._make_batch()
        metrics = trainer.evaluate(batch)

        self.assertIn("l2_distance", metrics)
        self.assertIn("cosine_similarity", metrics)
        self.assertIn("smooth_l1_loss", metrics)
        self.assertIn("embedding_energy", metrics)

    def test_get_and_load_weights(self):
        trainer = TextJEPATrainer(self.config, device="cpu")
        weights = trainer.get_weights()
        self.assertIsInstance(weights, dict)
        self.assertGreater(len(weights), 0)

        # Load into a new trainer
        trainer2 = TextJEPATrainer(self.config, device="cpu")
        trainer2.load_weights(weights)

        # Weights should match
        for k in weights:
            self.assertTrue(
                torch.equal(trainer.model.state_dict()[k],
                            trainer2.model.state_dict()[k]),
                f"Weights mismatch for {k}",
            )


class TestTextJEPAWithRealData(unittest.TestCase):
    """Test training with real TextTrainingDataSource output."""

    def test_end_to_end(self):
        """Full pipeline: JSONL → TextTrainingDataSource → TextJEPATrainer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create synthetic conversation data
            conv_dir = Path(tmpdir) / "conversations"
            conv_dir.mkdir()
            turns = [
                {"role": "user", "content": "Schedule a meeting with the design team for Thursday at 2pm",
                 "timestamp": "2026-03-26T10:00:00+00:00"},
                {"role": "assistant", "content": "I've scheduled a meeting with the design team for Thursday March 28 at 2pm. I've sent calendar invites to all team members.",
                 "timestamp": "2026-03-26T10:00:02+00:00"},
                {"role": "user", "content": "Also remind me to review the Q1 report before Friday",
                 "timestamp": "2026-03-26T10:00:05+00:00"},
                {"role": "assistant", "content": "Reminder set for Thursday evening to review the Q1 report.",
                 "timestamp": "2026-03-26T10:00:06+00:00"},
            ]
            for i in range(5):  # 5 conversations for enough data
                with open(conv_dir / f"session_{i}.jsonl", "w") as f:
                    for turn in turns:
                        f.write(json.dumps(turn) + "\n")

            # Create data source
            data_config = TextDataConfig(
                conversations_dir=str(conv_dir),
                max_seq_length=128,
                batch_size=2,
                scrub_pii=False,
                shuffle=False,
            )
            source = TextTrainingDataSource(data_config)
            self.assertGreater(len(source), 0)

            # Create trainer with small model
            model_config = TextJEPAConfig(
                vocab_size=260,
                max_seq_length=128,
                embed_dim=64,
                num_heads=4,
                encoder_depth=2,
                predictor_depth=1,
                predictor_embed_dim=32,
            )
            trainer = TextJEPATrainer(model_config, device="cpu")
            optimizer = torch.optim.AdamW(
                [{"params": trainer.model.context_encoder.parameters()},
                 {"params": trainer.model.predictor.parameters()}],
                lr=1e-3,
            )

            # Train for a few epochs
            all_losses = []
            for epoch in range(3):
                source.reload()  # Reshuffle
                for batch in source:
                    metrics = trainer.train_step(batch, optimizer)
                    all_losses.append(metrics["loss"])

            self.assertGreater(len(all_losses), 0, "No training steps executed")

            # Verify evaluation works
            batch = next(iter(source))
            eval_metrics = trainer.evaluate(batch)
            self.assertIn("cosine_similarity", eval_metrics)
            self.assertIn("embedding_energy", eval_metrics)


if __name__ == "__main__":
    unittest.main()

"""Tests for Story 5.1 — Behavioral Semantic Profile Accumulation.

Verifies that:
- Profile initializes with correct dimensions
- EMA update works correctly (first update initializes, subsequent decay)
- Profile persists across save/load cycles
- Profile hash is deterministic (same data → same hash)
- EMA decay behavior over simulated epochs matches expected math
- Similarity/distance computations work
- compute_training_embeddings extracts from a trained model
"""

import tempfile
import unittest
from pathlib import Path

import torch

from nodes.common.behavioral_profile import BehavioralProfile, compute_training_embeddings


class TestBehavioralProfileInit(unittest.TestCase):
    """Test profile initialization."""

    def test_default_init(self):
        profile = BehavioralProfile(K=32, D=384)
        self.assertEqual(profile.K, 32)
        self.assertEqual(profile.D, 384)
        self.assertEqual(profile.profile.shape, (32, 384))
        self.assertFalse(profile.initialized)
        self.assertEqual(profile.update_count, 0)

    def test_custom_dimensions(self):
        profile = BehavioralProfile(K=8, D=64)
        self.assertEqual(profile.profile.shape, (8, 64))

    def test_custom_decay(self):
        profile = BehavioralProfile(K=4, D=16, decay=0.99)
        self.assertEqual(profile.decay, 0.99)


class TestBehavioralProfileUpdate(unittest.TestCase):
    """Test EMA update mechanics."""

    def test_first_update_initializes(self):
        """First update should set profile directly (not decay zeros)."""
        profile = BehavioralProfile(K=4, D=8)
        embeddings = torch.randn(4, 8)

        profile.update(embeddings)

        self.assertTrue(profile.initialized)
        self.assertEqual(profile.update_count, 1)
        self.assertTrue(torch.allclose(profile.profile, embeddings))

    def test_ema_update(self):
        """Subsequent updates apply EMA decay."""
        K, D = 4, 8
        decay = 0.9
        profile = BehavioralProfile(K=K, D=D, decay=decay)

        first = torch.ones(K, D)
        second = torch.zeros(K, D)

        profile.update(first)
        profile.update(second)

        # Expected: 0.9 * ones + 0.1 * zeros = 0.9 * ones
        expected = decay * first + (1 - decay) * second
        self.assertTrue(torch.allclose(profile.profile, expected))

    def test_batch_update_mean_pools(self):
        """3D input (N, K, D) should be mean-pooled over N."""
        K, D = 4, 8
        profile = BehavioralProfile(K=K, D=D)

        # Two samples that average to 0.5
        batch = torch.stack([torch.zeros(K, D), torch.ones(K, D)])  # (2, K, D)
        profile.update(batch)

        expected = torch.full((K, D), 0.5)
        self.assertTrue(torch.allclose(profile.profile, expected))

    def test_wrong_shape_raises(self):
        profile = BehavioralProfile(K=4, D=8)
        with self.assertRaises(ValueError):
            profile.update(torch.randn(5, 8))  # Wrong K
        with self.assertRaises(ValueError):
            profile.update(torch.randn(4, 10))  # Wrong D

    def test_1d_tensor_raises(self):
        profile = BehavioralProfile(K=4, D=8)
        with self.assertRaises(ValueError):
            profile.update(torch.randn(32))


class TestBehavioralProfileDecay(unittest.TestCase):
    """Test EMA decay behavior over simulated epochs."""

    def test_decay_to_new_value(self):
        """After many updates with constant value, profile converges."""
        K, D = 4, 8
        decay = 0.9
        profile = BehavioralProfile(K=K, D=D, decay=decay)

        initial = torch.ones(K, D)
        target = torch.full((K, D), 5.0)

        profile.update(initial)

        # Apply many updates with constant target
        for _ in range(200):
            profile.update(target)

        # Should have converged very close to target
        self.assertTrue(
            torch.allclose(profile.profile, target, atol=0.01),
            f"Profile did not converge: max diff = {(profile.profile - target).abs().max():.4f}"
        )

    def test_decay_rate_math(self):
        """Verify EMA decay matches expected exponential decay formula.

        After n updates with zeros (after initial), the contribution of
        initial value is decay^n.
        """
        K, D = 2, 2
        decay = 0.998
        profile = BehavioralProfile(K=K, D=D, decay=decay)

        initial = torch.ones(K, D)
        profile.update(initial)

        # Apply 500 updates with zeros (simulating ~500 days of no activity)
        num_updates = 500
        zeros = torch.zeros(K, D)
        for _ in range(num_updates):
            profile.update(zeros)

        # Expected: initial * decay^500
        expected_value = decay ** num_updates
        actual_value = profile.profile[0, 0].item()

        self.assertAlmostEqual(actual_value, expected_value, places=4,
                               msg=f"EMA decay mismatch: expected {expected_value:.6f}, got {actual_value:.6f}")

    def test_500_days_decay_to_1_over_e(self):
        """With decay=0.998, ~500 days → 1/e decay. Verify approximately."""
        import math
        K, D = 2, 2
        decay = 0.998
        profile = BehavioralProfile(K=K, D=D, decay=decay)

        profile.update(torch.ones(K, D))

        # Apply 500 zero updates
        for _ in range(500):
            profile.update(torch.zeros(K, D))

        # decay^500 ≈ e^(-1) ≈ 0.3679
        expected = math.exp(-1)
        actual = profile.profile[0, 0].item()

        # Should be within 5% of 1/e
        self.assertAlmostEqual(actual, expected, delta=0.02,
                               msg=f"500-day decay: expected ~{expected:.4f}, got {actual:.4f}")


class TestBehavioralProfilePersistence(unittest.TestCase):
    """Test save/load cycle."""

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "profile.pt")

            # Create and update profile
            p1 = BehavioralProfile(K=4, D=8, decay=0.99, profile_path=path)
            embeddings = torch.randn(4, 8)
            p1.update(embeddings)
            p1.save()

            # Load in new instance
            p2 = BehavioralProfile(K=4, D=8, decay=0.99, profile_path=path)

            self.assertTrue(p2.initialized)
            self.assertEqual(p2.update_count, 1)
            self.assertTrue(torch.equal(p1.profile, p2.profile))

    def test_load_nonexistent_starts_fresh(self):
        profile = BehavioralProfile(K=4, D=8, profile_path="/nonexistent/path.pt")
        self.assertFalse(profile.initialized)
        self.assertEqual(profile.update_count, 0)

    def test_dimension_mismatch_starts_fresh(self):
        """Loading a profile with different dimensions should start fresh."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "profile.pt")

            # Save with K=4
            p1 = BehavioralProfile(K=4, D=8, profile_path=path)
            p1.update(torch.randn(4, 8))
            p1.save()

            # Load with K=8 — should discard and start fresh
            p2 = BehavioralProfile(K=8, D=8, profile_path=path)
            self.assertFalse(p2.initialized)


class TestBehavioralProfileHash(unittest.TestCase):
    """Test deterministic hashing for on-chain attestation."""

    def test_hash_deterministic(self):
        """Same profile → same hash."""
        p1 = BehavioralProfile(K=4, D=8)
        p2 = BehavioralProfile(K=4, D=8)

        embeddings = torch.randn(4, 8)
        p1.update(embeddings)
        p2.update(embeddings)

        self.assertEqual(p1.hash(), p2.hash())

    def test_hash_changes_with_update(self):
        """Different profiles → different hashes."""
        profile = BehavioralProfile(K=4, D=8)

        profile.update(torch.randn(4, 8))
        hash1 = profile.hash()

        profile.update(torch.randn(4, 8))
        hash2 = profile.hash()

        self.assertNotEqual(hash1, hash2)

    def test_hash_is_hex_string(self):
        profile = BehavioralProfile(K=4, D=8)
        profile.update(torch.randn(4, 8))
        h = profile.hash()

        self.assertIsInstance(h, str)
        self.assertEqual(len(h), 64)  # SHA-256 hex = 64 chars
        # Should be valid hex
        int(h, 16)


class TestBehavioralProfileSimilarity(unittest.TestCase):
    """Test similarity and distance computations."""

    def test_identical_profiles_similar(self):
        p1 = BehavioralProfile(K=4, D=8)
        p2 = BehavioralProfile(K=4, D=8)

        emb = torch.randn(4, 8)
        p1.update(emb)
        p2.update(emb)

        self.assertAlmostEqual(p1.similarity_to(p2), 1.0, places=5)

    def test_uninitialized_similarity_zero(self):
        p1 = BehavioralProfile(K=4, D=8)
        p2 = BehavioralProfile(K=4, D=8)
        self.assertEqual(p1.similarity_to(p2), 0.0)

    def test_distance_to_embedding(self):
        profile = BehavioralProfile(K=4, D=8)
        emb = torch.randn(4, 8)
        profile.update(emb)

        # Distance to itself should be 1.0
        sim = profile.distance_to_embedding(emb)
        self.assertAlmostEqual(sim, 1.0, places=5)

    def test_distance_to_3d_embedding(self):
        """Should handle (1, K, D) shaped embeddings."""
        profile = BehavioralProfile(K=4, D=8)
        emb = torch.randn(4, 8)
        profile.update(emb)

        sim = profile.distance_to_embedding(emb.unsqueeze(0))
        self.assertAlmostEqual(sim, 1.0, places=5)

    def test_orthogonal_profiles_low_similarity(self):
        """Profiles from very different data should have low similarity."""
        p1 = BehavioralProfile(K=4, D=8)
        p2 = BehavioralProfile(K=4, D=8)

        # Use deterministic opposite embeddings
        torch.manual_seed(42)
        emb1 = torch.randn(4, 8)
        emb2 = -emb1  # Negation = opposite direction

        p1.update(emb1)
        p2.update(emb2)

        self.assertLess(p1.similarity_to(p2), 0.0)


class TestBehavioralProfileToDict(unittest.TestCase):
    """Test metadata serialization."""

    def test_to_dict_uninitialized(self):
        profile = BehavioralProfile(K=4, D=8)
        d = profile.to_dict()

        self.assertEqual(d["K"], 4)
        self.assertEqual(d["D"], 8)
        self.assertFalse(d["initialized"])
        self.assertIsNone(d["hash"])

    def test_to_dict_initialized(self):
        profile = BehavioralProfile(K=4, D=8)
        profile.update(torch.randn(4, 8))
        d = profile.to_dict()

        self.assertTrue(d["initialized"])
        self.assertEqual(d["update_count"], 1)
        self.assertIsNotNone(d["hash"])
        self.assertGreater(d["profile_norm"], 0)


class TestComputeTrainingEmbeddings(unittest.TestCase):
    """Test embedding extraction from trained model."""

    def test_extracts_embeddings(self):
        from nodes.common.text_jepa import TextJEPAConfig, TextJEPATrainer

        config = TextJEPAConfig(
            vocab_size=260,
            max_seq_length=64,
            embed_dim=64,
            num_heads=4,
            encoder_depth=2,
            predictor_depth=1,
            predictor_embed_dim=32,
        )
        trainer = TextJEPATrainer(config, device="cpu")

        # Create fake batches
        batches = [
            {
                "token_ids": torch.randint(4, 260, (2, 48)),
                "attention_mask": torch.ones(2, 48, dtype=torch.bool),
            }
            for _ in range(3)
        ]

        embeddings = compute_training_embeddings(trainer, batches, max_batches=3)

        # Should have 6 samples (3 batches * 2 per batch), each D=64
        self.assertEqual(embeddings.shape, (6, 64))

    def test_max_batches_limit(self):
        from nodes.common.text_jepa import TextJEPAConfig, TextJEPATrainer

        config = TextJEPAConfig(
            vocab_size=260, max_seq_length=64, embed_dim=64,
            num_heads=4, encoder_depth=2, predictor_depth=1,
            predictor_embed_dim=32,
        )
        trainer = TextJEPATrainer(config, device="cpu")

        batches = [
            {
                "token_ids": torch.randint(4, 260, (2, 48)),
                "attention_mask": torch.ones(2, 48, dtype=torch.bool),
            }
            for _ in range(10)
        ]

        embeddings = compute_training_embeddings(trainer, batches, max_batches=2)

        # Should only process 2 batches = 4 samples
        self.assertEqual(embeddings.shape[0], 4)

    def test_empty_data_source(self):
        from nodes.common.text_jepa import TextJEPAConfig, TextJEPATrainer

        config = TextJEPAConfig(
            vocab_size=260, max_seq_length=64, embed_dim=64,
            num_heads=4, encoder_depth=2, predictor_depth=1,
            predictor_embed_dim=32,
        )
        trainer = TextJEPATrainer(config, device="cpu")

        embeddings = compute_training_embeddings(trainer, [], max_batches=10)
        self.assertEqual(embeddings.numel(), 0)


if __name__ == "__main__":
    unittest.main()

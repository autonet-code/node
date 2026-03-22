"""Tests for Story 5.3 — FramePreprocessor and patch extraction."""

import io
import unittest

import torch
from PIL import Image

from nodes.common.config import AutonetConfig, ModelConfig, TrainingConfig
from nodes.common.capture import Frame
from nodes.common.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    FramePreprocessor,
    TrainingBatch,
    extract_patches,
)


def _make_png_frame(width=64, height=64, color=(128, 64, 32)) -> Frame:
    """Create a Frame with valid PNG image_data."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Frame(
        image_data=buf.getvalue(),
        timestamp=1000.0,
        source_id="test-screen",
        modality="screen",
        width=width,
        height=height,
    )


class TestFramePreprocessor(unittest.TestCase):
    """Tests for FramePreprocessor."""

    def _make_config(self, image_size=32, batch_size=4):
        cfg = AutonetConfig()
        cfg.model = ModelConfig(image_size=image_size)
        cfg.training = TrainingConfig(batch_size=batch_size)
        return cfg

    def test_process_single_frame(self):
        """Single frame is buffered, not returned as batch."""
        cfg = self._make_config(image_size=32, batch_size=4)
        proc = FramePreprocessor(config=cfg)
        frame = _make_png_frame(64, 64)
        result = proc.process_frame(frame)
        self.assertIsNone(result)
        self.assertEqual(proc.buffer_size, 1)
        self.assertEqual(proc.stats.frames_received, 1)

    def test_batch_produced_when_full(self):
        """Batch is returned when buffer reaches batch_size."""
        cfg = self._make_config(image_size=32, batch_size=3)
        proc = FramePreprocessor(config=cfg)

        for i in range(2):
            result = proc.process_frame(_make_png_frame())
            self.assertIsNone(result)

        # Third frame triggers batch
        result = proc.process_frame(_make_png_frame())
        self.assertIsNotNone(result)
        self.assertIsInstance(result, TrainingBatch)
        self.assertEqual(result.batch_size, 3)

    def test_batch_tensor_shape(self):
        """Batch tensor has shape [B, C, H, W]."""
        cfg = self._make_config(image_size=32, batch_size=2)
        proc = FramePreprocessor(config=cfg)

        proc.process_frame(_make_png_frame())
        batch = proc.process_frame(_make_png_frame())

        self.assertIsNotNone(batch)
        tensor = batch.tensor
        self.assertEqual(tensor.shape, torch.Size([2, 3, 32, 32]))

    def test_resize_to_model_size(self):
        """Frames are resized to model.image_size."""
        cfg = self._make_config(image_size=16, batch_size=1)
        proc = FramePreprocessor(config=cfg)

        batch = proc.process_frame(_make_png_frame(128, 128))
        self.assertIsNotNone(batch)
        self.assertEqual(batch.tensor.shape, torch.Size([1, 3, 16, 16]))

    def test_normalization(self):
        """Pixel values are normalized with ImageNet stats."""
        cfg = self._make_config(image_size=8, batch_size=1)
        proc = FramePreprocessor(config=cfg)

        # Create a white image (all 255)
        batch = proc.process_frame(_make_png_frame(8, 8, color=(255, 255, 255)))
        tensor = batch.tensor  # [1, 3, 8, 8]

        # After normalization, white (1.0) should be (1.0 - mean) / std
        for c in range(3):
            expected = (1.0 - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
            actual = tensor[0, c, 0, 0].item()
            self.assertAlmostEqual(actual, expected, places=4)

    def test_flush_partial(self):
        """Flush returns partial batch."""
        cfg = self._make_config(image_size=32, batch_size=10)
        proc = FramePreprocessor(config=cfg)

        proc.process_frame(_make_png_frame())
        proc.process_frame(_make_png_frame())

        batch = proc.flush()
        self.assertIsNotNone(batch)
        self.assertEqual(batch.batch_size, 2)
        self.assertEqual(proc.buffer_size, 0)

    def test_flush_empty(self):
        """Flush returns None when buffer is empty."""
        cfg = self._make_config()
        proc = FramePreprocessor(config=cfg)
        self.assertIsNone(proc.flush())

    def test_stats_tracking(self):
        """Stats are tracked correctly."""
        cfg = self._make_config(image_size=32, batch_size=2)
        proc = FramePreprocessor(config=cfg)

        proc.process_frame(_make_png_frame())
        proc.process_frame(_make_png_frame())

        self.assertEqual(proc.stats.frames_received, 2)
        self.assertEqual(proc.stats.batches_produced, 1)

    def test_rgba_conversion(self):
        """RGBA images are converted to RGB."""
        img = Image.new("RGBA", (32, 32), color=(128, 64, 32, 200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        frame = Frame(
            image_data=buf.getvalue(),
            timestamp=1.0, source_id="test", modality="screen",
        )

        cfg = self._make_config(image_size=32, batch_size=1)
        proc = FramePreprocessor(config=cfg)
        batch = proc.process_frame(frame)
        self.assertIsNotNone(batch)
        self.assertEqual(batch.tensor.shape[1], 3)  # 3 channels, not 4

    def test_grayscale_conversion(self):
        """Grayscale images are converted to RGB."""
        img = Image.new("L", (32, 32), color=128)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        frame = Frame(
            image_data=buf.getvalue(),
            timestamp=1.0, source_id="test", modality="screen",
        )

        cfg = self._make_config(image_size=32, batch_size=1)
        proc = FramePreprocessor(config=cfg)
        batch = proc.process_frame(frame)
        self.assertIsNotNone(batch)
        self.assertEqual(batch.tensor.shape[1], 3)


class TestExtractPatches(unittest.TestCase):
    """Tests for the extract_patches utility."""

    def test_basic_extraction(self):
        """Extract patches from a simple tensor."""
        tensor = torch.randn(2, 3, 16, 16)
        patches = extract_patches(tensor, patch_size=4)
        # 16/4 = 4 patches per dim, 4*4 = 16 patches
        # Each patch: 3 * 4 * 4 = 48 dims
        self.assertEqual(patches.shape, torch.Size([2, 16, 48]))

    def test_single_patch(self):
        """When patch_size = image_size, get 1 patch."""
        tensor = torch.randn(1, 3, 8, 8)
        patches = extract_patches(tensor, patch_size=8)
        self.assertEqual(patches.shape, torch.Size([1, 1, 192]))

    def test_patch_count(self):
        """Verify number of patches for various sizes."""
        tensor = torch.randn(1, 3, 32, 32)
        patches = extract_patches(tensor, patch_size=4)
        # 32/4 = 8 per dim, 8*8 = 64 patches
        self.assertEqual(patches.shape[1], 64)

    def test_invalid_patch_size(self):
        """Non-divisible patch size raises assertion."""
        tensor = torch.randn(1, 3, 10, 10)
        with self.assertRaises(AssertionError):
            extract_patches(tensor, patch_size=3)


if __name__ == "__main__":
    unittest.main()

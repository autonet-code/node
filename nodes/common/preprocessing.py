"""
Local Data Preprocessing for Autonet Training Pipeline.

Converts captured frames into JEPA-ready training batches:
  1. Resize to model image_size
  2. Normalize pixel values
  3. Extract patches (optional, for patch-based models)
  4. Accumulate frames into batches of [B, C, H, W] tensors

Story 5.3: Frames -> JEPA training batches.
"""

import io
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .capture import Frame
from .config import AutonetConfig, load_config

logger = logging.getLogger(__name__)

# ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class TrainingBatch:
    """A batch of preprocessed frames ready for JEPA training."""
    tensor: object  # torch.Tensor [B, C, H, W], typed as object to avoid hard torch dep
    timestamps: List[float]
    source_ids: List[str]
    batch_size: int = 0


@dataclass
class PreprocessorStats:
    """Counters for the preprocessing pipeline."""
    frames_received: int = 0
    frames_dropped: int = 0
    batches_produced: int = 0


class FramePreprocessor:
    """
    Converts captured frames into JEPA training batches.

    Pipeline:
        Frame (PNG bytes) -> PIL Image -> Resize -> Tensor -> Normalize -> Batch

    Accumulates frames until batch_size is reached, then yields a
    TrainingBatch with shape [B, C, H, W].
    """

    def __init__(self, config: Optional[AutonetConfig] = None):
        self.config = config or load_config()
        self.image_size: int = self.config.model.image_size
        self.batch_size: int = self.config.training.batch_size
        self.stats = PreprocessorStats()

        # Accumulator buffer
        self._buffer: List[Tuple[object, float, str]] = []  # (tensor, timestamp, source_id)

    def process_frame(self, frame: Frame) -> Optional[TrainingBatch]:
        """
        Process a single captured frame.

        Returns a TrainingBatch when the buffer reaches batch_size,
        otherwise returns None (frame is buffered).
        """
        try:
            tensor = self._frame_to_tensor(frame)
            if tensor is None:
                self.stats.frames_dropped += 1
                return None

            self.stats.frames_received += 1
            self._buffer.append((tensor, frame.timestamp, frame.source_id))

            if len(self._buffer) >= self.batch_size:
                return self._flush_buffer()

            return None

        except Exception as e:
            logger.debug(f"Failed to process frame: {e}")
            self.stats.frames_dropped += 1
            return None

    def flush(self) -> Optional[TrainingBatch]:
        """
        Flush remaining buffered frames as a partial batch.

        Returns None if buffer is empty.
        """
        if self._buffer:
            return self._flush_buffer()
        return None

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    def _flush_buffer(self) -> TrainingBatch:
        """Build a TrainingBatch from the current buffer."""
        import torch

        tensors = [t for t, _, _ in self._buffer]
        timestamps = [ts for _, ts, _ in self._buffer]
        source_ids = [sid for _, _, sid in self._buffer]

        batch_tensor = torch.stack(tensors, dim=0)  # [B, C, H, W]
        batch = TrainingBatch(
            tensor=batch_tensor,
            timestamps=timestamps,
            source_ids=source_ids,
            batch_size=len(tensors),
        )

        self._buffer.clear()
        self.stats.batches_produced += 1
        logger.debug(f"Produced batch: {batch.batch_size} frames, shape {batch_tensor.shape}")
        return batch

    def _frame_to_tensor(self, frame: Frame) -> Optional[object]:
        """
        Convert a Frame's raw image bytes to a normalized [C, H, W] tensor.

        Supports PNG and JPEG encoded image_data.
        """
        try:
            import torch
            from PIL import Image

            # Decode image bytes
            img = Image.open(io.BytesIO(frame.image_data))

            # Convert to RGB if necessary
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Resize to model image_size
            if img.size != (self.image_size, self.image_size):
                img = img.resize(
                    (self.image_size, self.image_size),
                    Image.BILINEAR,
                )

            # Convert to tensor [C, H, W] in [0, 1]
            import numpy as np
            arr = np.array(img, dtype=np.float32) / 255.0  # [H, W, C]
            tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [C, H, W]

            # Normalize with ImageNet stats
            tensor = self._normalize(tensor)

            return tensor

        except ImportError as e:
            logger.warning(f"Missing dependency for preprocessing: {e}")
            return None
        except Exception as e:
            logger.debug(f"Frame decode error: {e}")
            return None

    def _normalize(self, tensor) -> object:
        """Apply ImageNet normalization to a [C, H, W] tensor."""
        import torch

        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return (tensor - mean) / std


def extract_patches(
    tensor,
    patch_size: int,
) -> object:
    """
    Extract non-overlapping patches from a [B, C, H, W] tensor.

    Returns a tensor of shape [B, num_patches, C * patch_size * patch_size].

    This is a utility for patch-based architectures (ViT, JEPA).
    The ViT encoder handles patching internally, but this is available
    for custom preprocessing pipelines.
    """
    import torch

    B, C, H, W = tensor.shape
    assert H % patch_size == 0, f"Height {H} not divisible by patch_size {patch_size}"
    assert W % patch_size == 0, f"Width {W} not divisible by patch_size {patch_size}"

    # Unfold into patches
    # [B, C, H/p, p, W/p, p] -> [B, H/p * W/p, C * p * p]
    patches = tensor.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
    patches = patches.permute(0, 2, 1, 3, 4)  # [B, num_patches, C, p, p]
    num_patches = patches.shape[1]
    patches = patches.reshape(B, num_patches, -1)  # [B, num_patches, C*p*p]

    return patches

"""
Simple Byte-Level Tokenizer for VL-JEPA

Zero external dependencies. Encodes text as byte sequences with a small
learned vocabulary. Sufficient for MVP; can be swapped for sentencepiece
or tiktoken later without changing the model architecture.

Special tokens:
  0: [PAD]
  1: [BOS] (beginning of sequence)
  2: [EOS] (end of sequence)
  3: [UNK] (unknown)
  4-259: raw bytes (0x00-0xFF)
"""

import logging
from typing import List, Tuple

import torch

logger = logging.getLogger(__name__)

# Special token IDs
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
BYTE_OFFSET = 4  # raw bytes start at ID 4

# Default vocab = 4 special + 256 bytes = 260
DEFAULT_VOCAB_SIZE = 260


class SimpleTokenizer:
    """Byte-level tokenizer for VL-JEPA text processing."""

    def __init__(self, vocab_size: int = DEFAULT_VOCAB_SIZE):
        self.vocab_size = max(vocab_size, DEFAULT_VOCAB_SIZE)
        self.pad_id = PAD_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID
        self.unk_id = UNK_ID

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """Encode text to token IDs (byte-level)."""
        token_ids = []
        if add_bos:
            token_ids.append(self.bos_id)

        for byte in text.encode("utf-8"):
            token_ids.append(byte + BYTE_OFFSET)

        if add_eos:
            token_ids.append(self.eos_id)

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        raw_bytes = []
        for tid in token_ids:
            if tid in (self.pad_id, self.bos_id, self.eos_id, self.unk_id):
                continue
            byte_val = tid - BYTE_OFFSET
            if 0 <= byte_val <= 255:
                raw_bytes.append(byte_val)
        try:
            return bytes(raw_bytes).decode("utf-8", errors="replace")
        except Exception:
            return ""

    def batch_encode(
        self,
        texts: List[str],
        max_length: int = 512,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a batch of texts with padding.

        Returns:
            token_ids: (B, max_length) LongTensor
            attention_mask: (B, max_length) BoolTensor — True = real token
        """
        batch_ids = []
        for text in texts:
            ids = self.encode(text, add_bos=add_bos, add_eos=add_eos)
            # Truncate
            ids = ids[:max_length]
            batch_ids.append(ids)

        # Pad to max length in batch
        actual_max = min(max(len(ids) for ids in batch_ids), max_length)
        token_ids = torch.full((len(texts), actual_max), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros(len(texts), actual_max, dtype=torch.bool)

        for i, ids in enumerate(batch_ids):
            length = len(ids)
            token_ids[i, :length] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :length] = True

        return token_ids, attention_mask

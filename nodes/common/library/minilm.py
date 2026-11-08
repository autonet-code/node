"""The local search model: all-MiniLM-L6-v2, run with onnxruntime + tokenizers.

Decision 2026-11-08 (docs/tool_substrate.md): tool search is local to each
daemon. Every daemon runs the same model, pinned to one revision of its
official Hugging Face repo; ``atn/search_model.py`` fetches the two files
below and verifies them against these hashes before anything loads them.

Embedding, as sentence-transformers runs this model: tokenize (at most
``MAX_TOKENS`` tokens), run the ONNX graph, mean-pool the last hidden state
over the attention mask, L2-normalize. Full ``MODEL_DIM`` dimensions, no
projection. No torch, no sentence-transformers.

Vectors from this module never enter consensus: they live in the daemon's
own search index.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_DIM = 384
# sentence-transformers runs this model with max_seq_length 256
# (sentence_bert_config.json at the pinned revision).
MAX_TOKENS = 256


@dataclass(frozen=True)
class ModelFile:
    """One pinned file of the model: its path in the repo, size, sha256."""
    path: str
    size: int
    sha256: str


MODEL_FILE = ModelFile(
    "onnx/model.onnx", 90405214,
    "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452")
TOKENIZER_FILE = ModelFile(
    "tokenizer.json", 466247,
    "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037")
MODEL_FILES: Tuple[ModelFile, ...] = (MODEL_FILE, TOKENIZER_FILE)


def model_key() -> str:
    """What each stored vector records, so a vector from another model or
    revision is recognized and embedded again."""
    return f"{MODEL_ID}@{MODEL_REVISION}"


class MiniLMEmbedder:
    """all-MiniLM-L6-v2 over onnxruntime.

    ``session`` and ``tokenizer`` can be injected (tests use a stub session
    and a tiny tokenizer); otherwise they load from the verified files.
    Thread-safe: an ONNX session runs concurrent calls, and the tokenizer is
    configured once at construction.
    """

    model_id = MODEL_ID
    revision = MODEL_REVISION
    dim = MODEL_DIM

    def __init__(
        self,
        model_path: Optional[Path] = None,
        tokenizer_path: Optional[Path] = None,
        *,
        session: Any = None,
        tokenizer: Any = None,
        threads: int = 2,
        max_tokens: int = MAX_TOKENS,
    ) -> None:
        if tokenizer is None:
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
        tokenizer.enable_truncation(max_length=max_tokens)
        # Pad to the longest text of a batch (the file pads to a fixed 128).
        pad_id = 0
        pad_token = "[PAD]"
        try:
            got = tokenizer.token_to_id(pad_token)
            if got is not None:
                pad_id = int(got)
        except Exception:  # noqa: BLE001 (a stub tokenizer may not know it)
            pass
        tokenizer.enable_padding(pad_id=pad_id, pad_token=pad_token)
        self._tokenizer = tokenizer
        if session is None:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = max(1, int(threads))
            opts.inter_op_num_threads = 1
            session = ort.InferenceSession(
                str(model_path), sess_options=opts,
                providers=["CPUExecutionProvider"])
        self._session = session
        try:
            self._inputs = {i.name for i in session.get_inputs()}
        except Exception:  # noqa: BLE001
            self._inputs = {"input_ids", "attention_mask", "token_type_ids"}
        self._lock = threading.Lock()

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        """``(len(texts), 384)`` float32 rows, each L2-normalized. An empty
        text embeds to the zero vector."""
        texts = [str(t or "") for t in texts]
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        live = [i for i, t in enumerate(texts) if t.strip()]
        if not live:
            return out
        encodings = self._tokenizer.encode_batch([texts[i] for i in live])
        ids = np.asarray([e.ids for e in encodings], dtype=np.int64)
        mask = np.asarray([e.attention_mask for e in encodings], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            types = [getattr(e, "type_ids", None) for e in encodings]
            feeds["token_type_ids"] = (
                np.asarray(types, dtype=np.int64)
                if all(t is not None for t in types) else np.zeros_like(ids))
        hidden = self._session.run(None, feeds)[0]          # (n, seq, dim)
        hidden = np.asarray(hidden, dtype=np.float32)
        weights = mask[:, :, None].astype(np.float32)
        summed = (hidden * weights).sum(axis=1)
        counts = np.clip(weights.sum(axis=1), 1e-9, None)
        pooled = summed / counts
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        pooled = pooled / np.clip(norms, 1e-12, None)
        out[live] = pooled
        return out

    def __call__(self, text: str) -> Tuple[float, ...]:
        return tuple(float(v) for v in self.embed_batch([text])[0])


# ---------------------------------------------------------------------------
# The daemon's one loaded model
# ---------------------------------------------------------------------------

_shared: Optional[MiniLMEmbedder] = None
_shared_lock = threading.Lock()


def set_shared(embedder: Optional[MiniLMEmbedder]) -> None:
    """Install the loaded model for the whole process (atn/search_model.py
    calls this once the files are verified). None unloads it (tests)."""
    global _shared
    with _shared_lock:
        _shared = embedder


def shared() -> Optional[MiniLMEmbedder]:
    """The loaded model, or None while it is still being fetched."""
    return _shared


def load(model_path: Path, tokenizer_path: Path) -> MiniLMEmbedder:
    """Load the embedder from verified files (blocking: about a second)."""
    return MiniLMEmbedder(model_path, tokenizer_path)


def embed_texts(texts: List[str], batch_size: int = 32) -> Optional[np.ndarray]:
    """Embed ``texts`` with the loaded model in batches; None when no model
    is loaded yet."""
    emb = shared()
    if emb is None:
        return None
    rows = []
    for start in range(0, len(texts), batch_size):
        rows.append(emb.embed_batch(texts[start:start + batch_size]))
    if not rows:
        return np.zeros((0, MODEL_DIM), dtype=np.float32)
    return np.vstack(rows)

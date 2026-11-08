"""The local search model's embedder (minilm.py): tokenize, run, mean-pool
over the attention mask, L2-normalize, 384 dims. A stub session and
tokenizer pin the arithmetic; one test runs the real pinned model when its
files are on disk (ATN_TEST_MODEL_DIR, default D:/atn_test_cache/minilm_dl)."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from nodes.common.world_model_substrate import minilm
from nodes.common.world_model_substrate.minilm import MiniLMEmbedder


class StubTokenizer:
    """Word-per-token tokenizer that pads to the longest text."""

    def __init__(self):
        self.truncation = None
        self.padding = None

    def enable_truncation(self, max_length):
        self.truncation = max_length

    def enable_padding(self, pad_id=0, pad_token="[PAD]"):
        self.padding = (pad_id, pad_token)

    def token_to_id(self, token):
        return 0 if token == "[PAD]" else None

    def encode_batch(self, texts):
        words = [t.split()[: self.truncation] for t in texts]
        width = max(len(w) for w in words)
        out = []
        for w in words:
            ids = [len(x) + 1 for x in w] + [0] * (width - len(w))
            mask = [1] * len(w) + [0] * (width - len(w))
            out.append(SimpleNamespace(ids=ids, attention_mask=mask,
                                       type_ids=[0] * width))
        return out


class StubSession:
    """Hidden state of token t at position p: a vector filled with the
    token id, plus a huge value on padded positions (which pooling must
    ignore)."""

    def __init__(self):
        self.feeds = None

    def get_inputs(self):
        return [SimpleNamespace(name=n)
                for n in ("input_ids", "attention_mask", "token_type_ids")]

    def run(self, _names, feeds):
        self.feeds = feeds
        ids = feeds["input_ids"].astype(np.float32)
        mask = feeds["attention_mask"].astype(np.float32)
        hidden = np.repeat(ids[:, :, None], minilm.MODEL_DIM, axis=2)
        hidden[:, :, 1] = 1.0
        hidden += (1.0 - mask)[:, :, None] * 1e6
        return [hidden]


def _stub():
    tok, sess = StubTokenizer(), StubSession()
    return MiniLMEmbedder(session=sess, tokenizer=tok), tok, sess


def test_mean_pool_over_mask_then_normalize():
    emb, tok, sess = _stub()
    assert tok.truncation == minilm.MAX_TOKENS
    assert tok.padding == (0, "[PAD]")
    out = emb.embed_batch(["ab cdef", "x"])           # ids [3,5] and [2]
    assert out.shape == (2, 384) and out.dtype == np.float32
    assert set(sess.feeds) == {"input_ids", "attention_mask", "token_type_ids"}
    # Row 0: mean of 3 and 5 = 4 everywhere but dim 1 (= 1); padding ignored.
    expect = np.full(384, 4.0, dtype=np.float32)
    expect[1] = 1.0
    expect /= np.linalg.norm(expect)
    assert np.allclose(out[0], expect, atol=1e-6)
    expect = np.full(384, 2.0, dtype=np.float32)
    expect[1] = 1.0
    expect /= np.linalg.norm(expect)
    assert np.allclose(out[1], expect, atol=1e-6)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)


def test_empty_text_is_the_zero_vector():
    emb, _tok, _sess = _stub()
    out = emb.embed_batch(["", "  ", "word"])
    assert not out[0].any() and not out[1].any()
    assert abs(float(np.linalg.norm(out[2])) - 1.0) < 1e-6
    assert len(emb("word")) == 384


def test_embed_texts_needs_the_loaded_model():
    minilm.set_shared(None)
    assert minilm.embed_texts(["a"]) is None
    emb, _tok, _sess = _stub()
    minilm.set_shared(emb)
    try:
        assert minilm.embed_texts(["a b", "c"] * 40, batch_size=16).shape == (80, 384)
    finally:
        minilm.set_shared(None)


def test_model_key_names_model_and_revision():
    assert minilm.model_key() == (
        "sentence-transformers/all-MiniLM-L6-v2@"
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41")


_REAL_DIR = Path(os.environ.get("ATN_TEST_MODEL_DIR", r"D:\atn_test_cache\minilm_dl"))


def _real_file(f):
    for name in (f.sha256, Path(f.path).name):
        p = _REAL_DIR / name
        if p.is_file() and p.stat().st_size == f.size:
            return p
    return None


@pytest.mark.skipif(
    _real_file(minilm.MODEL_FILE) is None or _real_file(minilm.TOKENIZER_FILE) is None,
    reason="pinned model files not on disk (ATN_TEST_MODEL_DIR)")
def test_real_model_matches_the_reference_output():
    from atn.search_model import verify_file
    model = _real_file(minilm.MODEL_FILE)
    tok = _real_file(minilm.TOKENIZER_FILE)
    assert verify_file(model, minilm.MODEL_FILE)
    assert verify_file(tok, minilm.TOKENIZER_FILE)
    emb = minilm.load(model, tok)
    v = emb.embed_batch(["This is an example sentence",
                         "parse a CSV file",
                         "read comma-separated values into rows",
                         "send an email to a customer"])
    # sentence-transformers' published output for this sentence starts so.
    assert np.allclose(v[0][:5], [0.0677, 0.0635, 0.0487, 0.0793, 0.0374], atol=5e-4)
    sims = v @ v.T
    assert sims[1, 2] > 0.4 > sims[1, 3]

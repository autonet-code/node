"""Tests for the model store (atn/model_specs.py)."""
from __future__ import annotations

import pytest

from atn.model_specs import (
    ModelSpec,
    context_window,
    list_by_class,
    list_by_family,
    list_models,
    max_output_tokens,
    model_class,
    relative_cost,
    resolve,
)


# ---------------------------------------------------------------------------
# resolve()
# ---------------------------------------------------------------------------

def test_resolve_exact_id():
    spec = resolve("claude-opus-4-7")
    assert spec.id == "claude-opus-4-7"
    assert spec.klass == "opus"
    assert spec.context_window == 1_000_000


def test_resolve_longest_prefix():
    """A versioned model ID falls through to the matching prefix."""
    spec = resolve("claude-opus-4-7-20260320")
    assert spec.id == "claude-opus-4-7"


def test_resolve_picks_longest_match():
    """gpt-5.5-preview should match gpt-5.5, not gpt-5."""
    spec = resolve("gpt-5.5-preview")
    assert spec.id == "gpt-5.5"


def test_resolve_unknown_returns_default():
    spec = resolve("totally-made-up-model")
    assert spec.id == "default"
    assert spec.klass == "other"


def test_resolve_empty_returns_default():
    spec = resolve("")
    assert spec.id == "default"


def test_resolve_case_insensitive():
    spec = resolve("CLAUDE-Opus-4-7")
    assert spec.id == "claude-opus-4-7"


# ---------------------------------------------------------------------------
# Class / family lookups
# ---------------------------------------------------------------------------

def test_model_class_bucketing():
    assert model_class("claude-haiku-4-5") == "haiku"
    assert model_class("claude-sonnet-4-6") == "sonnet"
    assert model_class("claude-opus-4-7") == "opus"
    assert model_class("gpt-5.5") == "other"


def test_list_by_family():
    claudes = list_by_family("claude")
    assert all(s.family == "claude" for s in claudes)
    assert any(s.id == "claude-opus-4-7" for s in claudes)


def test_list_by_class():
    sonnets = list_by_class("sonnet")
    assert all(s.klass == "sonnet" for s in sonnets)
    assert any(s.id == "claude-sonnet-4-7" for s in sonnets)


def test_list_models_returns_curated_set():
    models = list_models()
    assert len(models) > 10
    # Each entry must have minimum required fields populated.
    for s in models:
        assert s.id and s.family and s.klass
        assert s.context_window > 0
        assert s.max_output_tokens > 0


# ---------------------------------------------------------------------------
# Backward-compat shims
# ---------------------------------------------------------------------------

def test_context_window_shim_matches_spec():
    assert context_window("claude-opus-4-7") == 1_000_000
    assert context_window("claude-haiku-4-5") == 200_000
    assert context_window("unknown") == 128_000  # default


def test_max_output_tokens_shim_matches_spec():
    assert max_output_tokens("claude-opus-4-7") == 64_000
    assert max_output_tokens("claude-haiku-4-5") == 8_192


def test_relative_cost_for_classes():
    """Bootstrap multipliers used by the per-model estimator."""
    assert relative_cost("claude-haiku-4-5") < 1.0  # cheaper than sonnet
    assert relative_cost("claude-sonnet-4-7") == pytest.approx(1.0)
    assert relative_cost("claude-opus-4-7") > 1.0   # pricier than sonnet

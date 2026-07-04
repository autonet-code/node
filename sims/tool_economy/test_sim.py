"""Tests for the tool-economy simulator (sims/tool_economy/simulate.py).

Run:  python -m pytest sims/tool_economy/test_sim.py -q

Covers:
  1. Determinism — identical seed produces byte-identical sweep CSV.
  2. Baseline damper reproduces a HAND-COMPUTABLE tiny scenario driven
     through the REAL federated close: mint = standing * log1p(ok_count).
  3. Damper sanity — out_of_lineage zeroes a pure self-pumping wash tool;
     combo does the same and is no worse than out_of_lineage.
"""

from __future__ import annotations

import logging
import math
import os
import random
import sys

_HERE = os.path.dirname(__file__)
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

logging.disable(logging.INFO)  # silence the close's INFO chatter

from sims.tool_economy import simulate as sim  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Determinism
# ---------------------------------------------------------------------------

def test_sweep_deterministic_same_seed(tmp_path):
    rows_a = sim.sweep(42, [1, 3], [0.5, 1.0, 2.0])
    rows_b = sim.sweep(42, [1, 3], [0.5, 1.0, 2.0])
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    sim.write_csv(rows_a, str(a))
    sim.write_csv(rows_b, str(b))
    assert a.read_text(encoding="utf-8") == b.read_text(encoding="utf-8")


def test_sweep_differs_with_seed(tmp_path):
    a = sim.sweep(1, [1], [1.0])
    b = sim.sweep(2, [1], [1.0])
    # Different seeds should move at least one metric (population differs).
    assert a != b


def test_run_config_deterministic():
    cfg = sim.PopConfig(wash_intensity=1.0)
    m1 = sim.run_config(7, cfg, "combo", 3)
    m2 = sim.run_config(7, cfg, "combo", 3)
    # repr() is NaN-safe for equality (float('nan') != float('nan')).
    assert repr(m1) == repr(m2)


# ---------------------------------------------------------------------------
# 2. Baseline reproduces a hand-computable scenario through the real close
# ---------------------------------------------------------------------------

def _one_tool_epoch(n_supporters: int, ok_receipts: int, fail_receipts: int):
    """Drive the REAL close for ONE honest pinned tool with a known
    standing (1 author-post + n_supporters organic PRO posts) and a known
    OK receipt count. Returns (epoch, tools)."""
    rng = random.Random(123)
    author = "alice"
    digest = "".join(rng.choice("0123456789abcdef") for _ in range(64))
    tool = sim.Tool(digest, author, "honest", "tmnode_solo")
    tool.lineage = {author}
    tool.n_supporters = n_supporters
    for i in range(ok_receipts):
        tool.receipts.append((f"user_{i}", True))
    for i in range(fail_receipts):
        tool.receipts.append((f"user_{i}", False))
    epoch = sim.run_epoch([tool], rng)
    return epoch, [tool]


def test_baseline_matches_closed_form():
    n_supporters, ok, fail = 3, 5, 2
    epoch, tools = _one_tool_epoch(n_supporters, ok, fail)

    # The real close must report the tool with the expected standing and
    # ok_count (failures excluded by production, per tool_usage.py).
    tm = epoch["result"]["tool_mint"]
    digest = tools[0].digest
    assert digest in tm, "honest pinned tool should mint"
    entry = tm[digest]
    expected_standing = 1 + n_supporters          # author post + supporters
    assert entry["standing"] == expected_standing
    assert entry["ok_count"] == ok                # 5 OK, 2 failures dropped

    # Production mint == standing * log1p(ok_count).
    assert math.isclose(entry["mint"],
                        expected_standing * math.log1p(ok),
                        rel_tol=1e-9)

    # The sim's baseline damper reproduces production exactly.
    mint_map = sim.compute_damper_mint(epoch, tools, "baseline")
    got = mint_map[digest]
    assert got["standing"] == expected_standing
    assert got["ok_receipts"] == ok
    assert math.isclose(got["mint"], expected_standing * math.log1p(ok),
                        rel_tol=1e-12)
    # Hand number: standing 4 * ln(6) = 4 * 1.791759... = 7.167036...
    assert math.isclose(got["mint"], 4 * math.log(6), rel_tol=1e-12)


def test_failures_never_mint():
    # A tool with only failing receipts mints nothing under baseline.
    epoch, tools = _one_tool_epoch(n_supporters=2, ok_receipts=0,
                                   fail_receipts=5)
    mint_map = sim.compute_damper_mint(epoch, tools, "baseline")
    assert mint_map[tools[0].digest]["mint"] == 0.0


# ---------------------------------------------------------------------------
# 3. Damper sanity
# ---------------------------------------------------------------------------

def _wash_tool_epoch(n_receipts: int):
    """One pure self-pumping wash tool: standing 1, all receipts from its
    own in-lineage sybils."""
    rng = random.Random(55)
    author = "mallory"
    digest = "".join(rng.choice("0123456789abcdef") for _ in range(64))
    tool = sim.Tool(digest, author, "wash", "tmnode_wash")
    sybils = [f"sybil_{author}_{s}" for s in range(3)]
    tool.lineage = {author} | set(sybils)
    tool.n_supporters = 0
    for i in range(n_receipts):
        tool.receipts.append((sybils[i % len(sybils)], True))
    epoch = sim.run_epoch([tool], rng)
    return epoch, [tool]


def test_out_of_lineage_zeroes_self_pump():
    epoch, tools = _wash_tool_epoch(30)
    digest = tools[0].digest
    base = sim.compute_damper_mint(epoch, tools, "baseline")[digest]
    oob = sim.compute_damper_mint(epoch, tools, "out_of_lineage")[digest]
    assert base["mint"] > 0.0            # baseline pays the pump
    assert oob["mint"] == 0.0            # out-of-lineage kills it


def test_combo_no_worse_than_out_of_lineage_on_wash():
    epoch, tools = _wash_tool_epoch(30)
    digest = tools[0].digest
    oob = sim.compute_damper_mint(epoch, tools, "out_of_lineage")[digest]
    combo = sim.compute_damper_mint(epoch, tools, "combo")[digest]
    assert combo["mint"] <= oob["mint"] + 1e-12
    assert combo["mint"] == 0.0


def test_burn_charges_callers():
    epoch, tools = _wash_tool_epoch(20)
    digest = tools[0].digest
    burn = sim.compute_damper_mint(epoch, tools, "burn", burn=0.1)[digest]
    assert burn["ok_receipts"] == 20
    assert math.isclose(burn["cost"], 20 * 0.1, rel_tol=1e-12)
    assert math.isclose(burn["net"], burn["mint"] - burn["cost"], rel_tol=1e-12)

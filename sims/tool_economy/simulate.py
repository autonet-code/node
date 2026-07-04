"""Game-theory simulator for the tool-substrate mint economy.

Pre-registered-style sweep to inform the wash-trading damper choice
(docs/tool_substrate.md, "Open knobs" #1). It drives the REAL federated
close (``federated_epoch_close`` over fabricated canonical event
batches, same fabrication pattern as ``tests/test_tool_mint.py``) so the
standing/mint numbers are exactly what production would compute; the
dampers are then applied as *post-processing over the usage aggregation*
that feeds mint. Production code is NOT modified — the dampers live here
so the sim can compare them against the current baseline.

The mint formula under study (v2, pinned-only, decay retired):

    mint(tool) = max(0, standing(tool)) * log1p(usage_term(tool))

Baseline ``usage_term`` is the raw ``ok_count``. Each damper substitutes
a different ``usage_term`` computed from the same receipts, using
ground-truth caller lineage that the SIM knows (the real system's
knowledge limits are discussed in MEMO.md).

Populations
-----------
* honest  — pinned tool, organic supporter PRO posts (real standing),
             receipts from many distinct out-of-lineage callers.
* wash    — junk tool, standing 1 (no supporters), many receipts pumped
             from a small pool of sybil callers inside its own lineage.
* seo     — high claimed relevance but ~no real usage: a few receipts
             from a few sources. Matters for retrieval, not mint; for
             MINT it looks like a weak wash-trader.

Determinism: everything is seeded off ``--seed``; the close itself is
bit-identical by construction, so identical seed => identical CSV.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

# --- import the real close path -------------------------------------------
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from nodes.common.canonical_ordering import canonical_order  # noqa: E402
from nodes.common.event_gossip import EventBatch, Keypair  # noqa: E402
from nodes.common.federated_reconcile import federated_epoch_close  # noqa: E402
from nodes.common.world_model_substrate.adapter import N_DIMS  # noqa: E402
from nodes.common.world_model_substrate.tool_usage import (  # noqa: E402
    tool_usage_from_events,
)

# Small embedding tail keeps the replay world cheap; the mint math is
# indifferent to its width. Each event still gets DISTINCT coords so the
# content-addressed sprouting does not merge separate tools/supporters
# into one node (a subtlety the calibration probe surfaced).
EMBED_DIM = 64
OUT_DIR = os.path.join(os.path.dirname(__file__), "out")

# --- per-receipt burn: the ATN a caller pays to record one receipt. The
# damper (d) sweep models wash ROI = mint_gained - burn_paid at this cost.
DEFAULT_BURN = 0.05

DAMPERS = ["baseline", "out_of_lineage", "diversity_sqrt", "diversity_pow",
           "burn", "combo"]


# ---------------------------------------------------------------------------
# Event fabrication (mirrors tests/test_tool_mint.py)
# ---------------------------------------------------------------------------

def _coords(rng: random.Random, axis: int = 4) -> List[float]:
    """Distinct-per-call coords: charter axis mass + random embedding tail.

    Distinctness matters: identical coords collapse to one content-
    addressed node, merging separate tools' standing. Real tools live at
    distinct positions, so we mirror that.
    """
    out = [0.0] * (N_DIMS + EMBED_DIM)
    out[axis] = 0.6 + rng.random() * 0.3
    for j in range(N_DIMS, N_DIMS + EMBED_DIM):
        out[j] = rng.uniform(-1.0, 1.0)
    return out


def _registration_event(seq: int, author: str, digest: str, node_id: str,
                        rng: random.Random) -> Dict[str, Any]:
    return {
        "kind": "sub_claim_sprouted",
        "seq": seq,
        "author_agent": author,
        "tendency_id": "correctness",
        "parent_id": "solver_root",
        "node_id": node_id,
        "position": "pro",
        "coords": _coords(rng),
        "polarity_axis": _coords(rng),
        "content": f"tool {digest[:8]}",
        "author_post": True,
        "artifact_digest": digest,
        "manifest_meta": {"trust_class": "pinned", "author": author},
    }


def _supporter_event(seq: int, supporter: str, parent_node: str, node_id: str,
                    rng: random.Random) -> Dict[str, Any]:
    """A PRO child endorsing a manifest node — organic standing.

    Its net_score (1, from the supporter's own author_post) propagates up
    to the manifest node, so N distinct supporters => manifest standing
    1 (author) + N.
    """
    return {
        "kind": "sub_claim_sprouted",
        "seq": seq,
        "author_agent": supporter,
        "tendency_id": "correctness",
        "parent_id": parent_node,
        "node_id": node_id,
        "position": "pro",
        "coords": _coords(rng),
        "polarity_axis": _coords(rng),
        "content": "endorse",
        "author_post": True,
    }


def _receipt_event(seq: int, caller: str, digest: str, author: str,
                   *, ok: bool = True) -> Dict[str, Any]:
    return {
        "kind": "tool_used",
        "seq": seq,
        "author_agent": caller,
        "manifest_digest": digest,
        "tool_author": author,
        "receipt_digest": f"r{seq:06d}" * 5,
        "ok": ok,
        "fee_atn": 0.0,
    }


def _batches(groups: List[List[Dict[str, Any]]],
             keypair: Keypair) -> List[EventBatch]:
    """One EventBatch per group of events.

    Grouping matters: a supporter PRO child references its manifest's
    wire node_id as ``parent_id``. Parent-label resolution
    (``aggregate.resolve_parent``) uses a per-BATCH remap, so a supporter
    only attaches to the manifest node if it rides the SAME batch as the
    registration. Split across batches, the supporter falls through to
    the tendency root and contributes no standing to the tool — so we
    keep each tool's registration + supporters in one group.
    """
    chain: List[EventBatch] = []
    prev_hash = b""
    for i, group in enumerate(groups, start=1):
        b = EventBatch(
            rpb_address="rpb_toolsim",
            sender_pubkey=keypair.public_key,
            batch_seq=i,
            events=list(group),
            prev_batch_hash=prev_hash,
            timestamp=1_700_000_000.0 + i,
        )
        chain.append(b)
        prev_hash = b.content_hash()
    return chain


# ---------------------------------------------------------------------------
# Population model
# ---------------------------------------------------------------------------

@dataclass
class Tool:
    digest: str
    author: str
    kind: str            # "honest" | "wash" | "seo"
    node_id: str
    lineage: set = field(default_factory=set)   # ground-truth caller lineage
    n_supporters: int = 0
    receipts: List[Tuple[str, bool]] = field(default_factory=list)  # (caller, ok)


@dataclass
class PopConfig:
    n_honest: int = 12
    n_wash: int = 6
    n_seo: int = 4
    # honest: organic standing + diverse real usage
    honest_supporters: Tuple[int, int] = (2, 6)
    honest_receipts: Tuple[int, int] = (6, 20)
    honest_callers: Tuple[int, int] = (6, 20)     # distinct out-of-lineage
    # wash: no supporters; pump receipts from a small sybil pool
    wash_intensity: float = 1.0                    # scales pumped receipts
    wash_base_receipts: Tuple[int, int] = (20, 60)
    wash_sybils: Tuple[int, int] = (2, 4)          # in-lineage caller pool
    # seo: minimal real usage
    seo_receipts: Tuple[int, int] = (1, 3)
    seo_callers: Tuple[int, int] = (1, 2)


def _digest(rng: random.Random) -> str:
    return "".join(rng.choice("0123456789abcdef") for _ in range(64))


def build_population(rng: random.Random, cfg: PopConfig) -> List[Tool]:
    tools: List[Tool] = []
    idx = 0

    def next_node() -> str:
        nonlocal idx
        idx += 1
        return f"tmnode_{idx:04d}"

    for i in range(cfg.n_honest):
        author = f"honest_{i:02d}"
        t = Tool(_digest(rng), author, "honest", next_node())
        t.lineage = {author}
        t.n_supporters = rng.randint(*cfg.honest_supporters)
        n_callers = rng.randint(*cfg.honest_callers)
        n_receipts = rng.randint(*cfg.honest_receipts)
        # diverse, out-of-lineage callers
        callers = [f"user_{author}_{c}" for c in range(n_callers)]
        for _ in range(n_receipts):
            caller = rng.choice(callers)
            ok = rng.random() < 0.9
            t.receipts.append((caller, ok))
        tools.append(t)

    for i in range(cfg.n_wash):
        author = f"wash_{i:02d}"
        t = Tool(_digest(rng), author, "wash", next_node())
        n_sybils = rng.randint(*cfg.wash_sybils)
        sybils = [f"sybil_{author}_{s}" for s in range(n_sybils)]
        # wash lineage includes the author AND its declared sybil callers
        t.lineage = {author} | set(sybils)
        t.n_supporters = 0
        base = rng.randint(*cfg.wash_base_receipts)
        n_receipts = max(1, int(base * cfg.wash_intensity))
        for _ in range(n_receipts):
            caller = rng.choice(sybils)
            ok = rng.random() < 0.98      # wash reports near-perfect success
            t.receipts.append((caller, ok))
        tools.append(t)

    for i in range(cfg.n_seo):
        author = f"seo_{i:02d}"
        t = Tool(_digest(rng), author, "seo", next_node())
        t.lineage = {author}
        t.n_supporters = 0
        n_callers = rng.randint(*cfg.seo_callers)
        n_receipts = rng.randint(*cfg.seo_receipts)
        callers = [f"user_{author}_{c}" for c in range(n_callers)]
        for _ in range(n_receipts):
            caller = rng.choice(callers)
            t.receipts.append((caller, rng.random() < 0.9))
        tools.append(t)

    return tools


# ---------------------------------------------------------------------------
# One epoch through the REAL close
# ---------------------------------------------------------------------------

def run_epoch(tools: List[Tool], rng: random.Random) -> Dict[str, Any]:
    """Fabricate one epoch's canonical batch and drive the real close.

    Returns the raw close ``result`` plus per-tool receipt event lists so
    the dampers can recompute usage terms from ground truth.
    """
    seq = 0
    receipt_events_by_digest: Dict[str, List[Dict[str, Any]]] = {}
    groups: List[List[Dict[str, Any]]] = []

    # One group (batch) per tool: registration + its supporter PRO
    # children together, so supporter standing resolves onto the manifest
    # node (see _batches docstring).
    for t in tools:
        seq += 1
        group = [_registration_event(seq, t.author, t.digest, t.node_id, rng)]
        for s in range(t.n_supporters):
            seq += 1
            group.append(_supporter_event(
                seq, f"sup_{t.author}_{s}", t.node_id,
                f"{t.node_id}_sup_{s}", rng))
        groups.append(group)

    # Receipts: graph-neutral (replay skips them), so batching is free.
    # One group per tool keeps the canonical log tidy.
    for t in tools:
        recs: List[Dict[str, Any]] = []
        for (caller, ok) in t.receipts:
            seq += 1
            ev = _receipt_event(seq, caller, t.digest, t.author, ok=ok)
            recs.append(ev)
        receipt_events_by_digest[t.digest] = recs
        if recs:
            groups.append(recs)

    # Shuffle BATCH order to prove order-independence of the close (the
    # canonical ordering re-sorts anyway); derived RNG keeps it a pure
    # function of the seed. We shuffle whole groups, not events within a
    # group, so each tool's reg+supporters stay co-batched.
    shuf = random.Random(rng.random())
    shuf.shuffle(groups)

    kp = Keypair.generate()          # identity is irrelevant to mint here
    batches = _batches(groups, kp)
    result = federated_epoch_close(canonical_order(batches))
    return {
        "result": result,
        "receipt_events_by_digest": receipt_events_by_digest,
    }


# ---------------------------------------------------------------------------
# Dampers — recompute usage_term, keep the real effective_standing
# ---------------------------------------------------------------------------

def _detected_lineage(lineage: set, lineage_recall: float,
                      rng: random.Random) -> set:
    """The lineage set the REAL system could plausibly detect.

    Ground-truth lineage is known to the SIM; the deployed network only
    infers it (wallet clustering, correlated timing, shared funding). We
    model that with ``lineage_recall``: the probability each ground-truth
    lineage member is correctly flagged. recall=1.0 is the omniscient
    ceiling (the raw out_of_lineage damper); recall<1.0 shows how the
    damper degrades when sybils evade clustering.
    """
    if lineage_recall >= 1.0:
        return set(lineage)
    return {m for m in lineage if rng.random() < lineage_recall}


def _usage_term(damper: str, receipts: List[Dict[str, Any]], lineage: set,
                *, div_alpha: float = 0.6, lineage_recall: float = 1.0,
                rng: random.Random | None = None) -> float:
    """Return the usage term fed into log1p(...) for a damper.

    All operate on OK receipts only (the mint formula already excludes
    failures). ``receipts`` are the raw tool_used events for one digest;
    ``lineage`` is the SIM's ground-truth author+sybil set. Lineage-aware
    dampers use ``_detected_lineage`` so ``lineage_recall < 1`` models
    imperfect real-world sybil detection.
    """
    ok = [e for e in receipts if e.get("ok", True)]
    if not ok:
        return 0.0

    if damper == "baseline":
        return float(len(ok))

    if damper == "burn":
        return float(len(ok))   # burn: raw count, cost handled separately

    if damper == "out_of_lineage":
        # (b) receipts from the author or its (detected) sybil lineage
        # don't count.
        seen = _detected_lineage(lineage, lineage_recall,
                                 rng or random.Random(0))
        return float(sum(1 for e in ok if e.get("author_agent") not in seen))

    # caller-diversity forms
    from collections import Counter
    by_caller = Counter(e.get("author_agent", "") for e in ok)
    if damper == "diversity_sqrt":
        # Σ over unique callers of log1p(calls_by_caller). A single caller
        # spamming N calls contributes only log1p(N), not N.
        return sum(math.log1p(c) for c in by_caller.values())
    if damper == "diversity_pow":
        # (unique callers)^alpha * log1p(total): rewards breadth, damps
        # depth from a narrow pool.
        uniq = len(by_caller)
        total = sum(by_caller.values())
        return (uniq ** div_alpha) * math.log1p(total)
    if damper == "combo":
        # out-of-lineage filter THEN caller-diversity weighting: even a
        # sybil that evades lineage detection only counts as breadth once
        # per unique caller. Robust when lineage_recall < 1.
        seen = _detected_lineage(lineage, lineage_recall,
                                 rng or random.Random(0))
        oob = Counter(e.get("author_agent", "") for e in ok
                      if e.get("author_agent") not in seen)
        return sum(math.log1p(c) for c in oob.values())

    raise ValueError(f"unknown damper {damper!r}")


def compute_damper_mint(epoch: Dict[str, Any], tools: List[Tool], damper: str,
                        *, burn: float = DEFAULT_BURN,
                        div_alpha: float = 0.6, lineage_recall: float = 1.0,
                        rng: random.Random | None = None
                        ) -> Dict[str, Dict[str, Any]]:
    """Per-tool mint under one damper.

    Uses the REAL standing from the close (production truth: v2 mint is
    pinned-only, no decay), recomputes only the usage term, and derives
    mint = max(0, standing) * log1p(usage_term). The ``burn`` damper
    additionally charges each OK receipt ``burn`` ATN to the caller and
    reports net.
    """
    result = epoch["result"]
    tm = result["tool_mint"]
    recs_by_digest = epoch["receipt_events_by_digest"]
    det_rng = rng or random.Random(0)

    out: Dict[str, Dict[str, Any]] = {}
    for t in tools:
        entry = tm.get(t.digest)
        # standing: from the close if the tool minted (v2 mint is
        # pinned-only, standing * log1p(ok_count), no decay). Tools with
        # zero-or-negative standing or no OK receipts never appear in tm;
        # their mint is 0 under every damper, which is correct.
        recs = recs_by_digest.get(t.digest, [])
        if entry is not None:
            eff = float(entry["standing"])
        else:
            eff = 0.0
        term = _usage_term(damper, recs, t.lineage, div_alpha=div_alpha,
                           lineage_recall=lineage_recall, rng=det_rng)
        mint = max(0.0, eff) * math.log1p(term) if term > 0 else 0.0

        ok_receipts = sum(1 for e in recs if e.get("ok", True))
        cost = burn * ok_receipts if damper == "burn" else 0.0
        out[t.digest] = {
            "author": t.author,
            "kind": t.kind,
            "standing": eff,
            "usage_term": term,
            "ok_receipts": ok_receipts,
            "mint": mint,
            "cost": cost,
            "net": mint - cost,
        }
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def gini(values: List[float]) -> float:
    xs = sorted(v for v in values if v > 0)
    n = len(xs)
    if n == 0:
        return 0.0
    cum = 0.0
    for i, x in enumerate(xs, start=1):
        cum += i * x
    total = sum(xs)
    if total == 0:
        return 0.0
    return (2.0 * cum) / (n * total) - (n + 1.0) / n


def summarize(mint_map: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    total = sum(m["mint"] for m in mint_map.values())
    honest = sum(m["mint"] for m in mint_map.values() if m["kind"] == "honest")
    wash = sum(m["mint"] for m in mint_map.values() if m["kind"] == "wash")
    seo = sum(m["mint"] for m in mint_map.values() if m["kind"] == "seo")

    wash_mint = sum(m["mint"] for m in mint_map.values() if m["kind"] == "wash")
    wash_cost = sum(m["cost"] for m in mint_map.values() if m["kind"] == "wash")
    wash_net = sum(m["net"] for m in mint_map.values() if m["kind"] == "wash")
    # Burn on HONEST tool usage: paid by honest callers, not authors. It
    # is the collateral friction the burn damper imposes on real users.
    honest_burn = sum(m["cost"] for m in mint_map.values()
                      if m["kind"] == "honest")

    # per-author earnings for Gini (mint attributed to each author)
    per_author: Dict[str, float] = {}
    for m in mint_map.values():
        per_author[m["author"]] = per_author.get(m["author"], 0.0) + m["mint"]
    # honest-only earnings snapshot
    honest_authors = [v for k, v in per_author.items() if k.startswith("honest_")]

    # wash ROI: mint gained per unit cost. For non-burn dampers cost=0 so
    # ROI is undefined; we report net==mint and roi as inf-sentinel -1.
    if wash_cost > 0:
        wash_roi = wash_mint / wash_cost
    else:
        wash_roi = float("nan")

    return {
        "total_mint": total,
        "honest_mint": honest,
        "wash_mint": wash,
        "seo_mint": seo,
        "honest_share": (honest / total) if total > 0 else 0.0,
        "wash_share": (wash / total) if total > 0 else 0.0,
        "wash_cost": wash_cost,
        "wash_net": wash_net,
        "wash_roi": wash_roi,
        "honest_burn": honest_burn,
        "gini_all": gini([v for v in per_author.values()]),
        "gini_honest": gini(honest_authors),
    }


# ---------------------------------------------------------------------------
# Multi-epoch run + sweep
# ---------------------------------------------------------------------------

def run_config(seed: int, cfg: PopConfig, damper: str, epochs: int,
               *, burn: float = DEFAULT_BURN,
               div_alpha: float = 0.6,
               lineage_recall: float = 1.0) -> Dict[str, float]:
    """Run ``epochs`` epochs for one (cfg, damper), averaging metrics.

    Each epoch rebuilds the population from the seeded RNG (fresh tools /
    receipts each epoch, same distribution) and drives the real close.
    """
    rng = random.Random(seed)
    accum: List[Dict[str, float]] = []
    for _ in range(epochs):
        tools = build_population(rng, cfg)
        epoch = run_epoch(tools, rng)
        mint_map = compute_damper_mint(
            epoch, tools, damper, burn=burn, div_alpha=div_alpha,
            lineage_recall=lineage_recall, rng=rng)
        accum.append(summarize(mint_map))

    keys = accum[0].keys()
    avg: Dict[str, float] = {}
    for k in keys:
        vals = [a[k] for a in accum if not math.isnan(a[k])]
        avg[k] = (sum(vals) / len(vals)) if vals else float("nan")
    return avg


def _cell_seed(seed: int, *parts) -> int:
    """Deterministic per-cell seed (hash() is salted per-process, so we
    fold a stable string hash instead)."""
    h = 1469598103934665603
    for p in parts:
        for b in repr(p).encode():
            h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return (seed * 1000003 + (h % 100000))


def sweep(seed: int, epochs_list: List[int],
          intensities: List[float],
          burn: float = DEFAULT_BURN,
          lineage_recall: float = 1.0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for damper in DAMPERS:
        for intensity in intensities:
            for epochs in epochs_list:
                cfg = PopConfig(wash_intensity=intensity)
                cs = _cell_seed(seed, damper, intensity, epochs)
                metrics = run_config(cs, cfg, damper, epochs, burn=burn,
                                     lineage_recall=lineage_recall)
                row = {
                    "damper": damper,
                    "wash_intensity": intensity,
                    "epochs": epochs,
                    "burn": burn,
                    "lineage_recall": lineage_recall,
                    **{k: round(v, 6) if not math.isnan(v) else "nan"
                       for k, v in metrics.items()},
                }
                rows.append(row)
    return rows


def burn_sweep(seed: int, epochs: int, intensities: List[float],
               burn_rates: List[float]) -> List[Dict[str, Any]]:
    """Isolate the burn damper across burn rates × intensities.

    Answers: what per-receipt burn makes wash ROI < 1 at EVERY intensity,
    and what does that cost honest callers (honest_burn)?
    """
    rows: List[Dict[str, Any]] = []
    for br in burn_rates:
        for it in intensities:
            cs = _cell_seed(seed, "burnsweep", br, it)
            m = run_config(cs, PopConfig(wash_intensity=it), "burn", epochs,
                           burn=br)
            rows.append({
                "burn": br, "wash_intensity": it,
                **{k: round(v, 6) if not math.isnan(v) else "nan"
                   for k, v in m.items()},
            })
    return rows


def recall_sweep(seed: int, epochs: int, intensity: float,
                 recalls: List[float]) -> List[Dict[str, Any]]:
    """out_of_lineage and combo dampers as lineage-detection recall
    degrades from omniscient (1.0) toward blind (0.0)."""
    rows: List[Dict[str, Any]] = []
    for damper in ("out_of_lineage", "combo"):
        for r in recalls:
            cs = _cell_seed(seed, "recall", damper, r)
            m = run_config(cs, PopConfig(wash_intensity=intensity), damper,
                           epochs, lineage_recall=r)
            rows.append({
                "damper": damper, "lineage_recall": r,
                **{k: round(v, 6) if not math.isnan(v) else "nan"
                   for k, v in m.items()},
            })
    return rows


CSV_FIELDS = [
    "damper", "wash_intensity", "epochs", "burn", "lineage_recall",
    "total_mint", "honest_mint", "wash_mint", "seo_mint",
    "honest_share", "wash_share",
    "wash_cost", "wash_net", "wash_roi", "honest_burn",
    "gini_all", "gini_honest",
]


def write_csv(rows: List[Dict[str, Any]], path: str,
              fields: List[str] | None = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fields is None:
        # union of keys, stable order: known fields first, then extras.
        seen = list(CSV_FIELDS)
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.append(k)
        fields = seen
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(rows: List[Dict[str, Any]], out_dir: str) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    made: List[str] = []

    def _num(v):
        return float("nan") if v == "nan" else float(v)

    # Filter to a canonical slice for the bar/line plots: max epochs.
    max_epochs = max(r["epochs"] for r in rows)
    intensities = sorted({r["wash_intensity"] for r in rows})

    # --- Plot 1: honest share of mint vs wash intensity, per damper ------
    fig, ax = plt.subplots(figsize=(8, 5))
    for damper in DAMPERS:
        xs, ys = [], []
        for it in intensities:
            match = [r for r in rows if r["damper"] == damper
                     and r["wash_intensity"] == it
                     and r["epochs"] == max_epochs]
            if match:
                xs.append(it)
                ys.append(_num(match[0]["honest_share"]))
        ax.plot(xs, ys, marker="o", label=damper)
    ax.set_xlabel("wash-trader intensity (receipt multiplier)")
    ax.set_ylabel("honest authors' share of tool mint")
    ax.set_title("Honest share of mint vs wash intensity")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    p1 = os.path.join(out_dir, "honest_share.png")
    fig.tight_layout(); fig.savefig(p1, dpi=110); plt.close(fig)
    made.append(p1)

    # --- Plot 2: wash share of mint vs intensity, per damper -------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for damper in DAMPERS:
        xs, ys = [], []
        for it in intensities:
            match = [r for r in rows if r["damper"] == damper
                     and r["wash_intensity"] == it
                     and r["epochs"] == max_epochs]
            if match:
                xs.append(it)
                ys.append(_num(match[0]["wash_share"]))
        ax.plot(xs, ys, marker="s", label=damper)
    ax.set_xlabel("wash-trader intensity (receipt multiplier)")
    ax.set_ylabel("wash-traders' share of tool mint")
    ax.set_title("Wash-trader share of mint vs intensity (lower is better)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    p2 = os.path.join(out_dir, "wash_share.png")
    fig.tight_layout(); fig.savefig(p2, dpi=110); plt.close(fig)
    made.append(p2)

    # --- Plot 3: burn ROI vs intensity (burn damper only) ----------------
    fig, ax = plt.subplots(figsize=(8, 5))
    burn_rows = [r for r in rows if r["damper"] == "burn"
                 and r["epochs"] == max_epochs]
    xs = [r["wash_intensity"] for r in sorted(burn_rows, key=lambda r: r["wash_intensity"])]
    roi = [_num(r["wash_roi"]) for r in sorted(burn_rows, key=lambda r: r["wash_intensity"])]
    ax.plot(xs, roi, marker="o", color="crimson", label="wash ROI = mint / burn")
    ax.axhline(1.0, color="black", ls="--", lw=1, label="break-even (ROI=1)")
    ax.set_xlabel("wash-trader intensity (receipt multiplier)")
    ax.set_ylabel("wash ROI (mint gained / ATN burned)")
    ax.set_title("Per-receipt burn: wash ROI vs intensity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    p3 = os.path.join(out_dir, "burn_roi.png")
    fig.tight_layout(); fig.savefig(p3, dpi=110); plt.close(fig)
    made.append(p3)

    # --- Plot 4: Gini of author earnings per damper (at max epochs, it=1) -
    fig, ax = plt.subplots(figsize=(8, 5))
    it_ref = 1.0 if 1.0 in intensities else intensities[0]
    labels, g_all, g_hon = [], [], []
    for damper in DAMPERS:
        match = [r for r in rows if r["damper"] == damper
                 and r["wash_intensity"] == it_ref
                 and r["epochs"] == max_epochs]
        if match:
            labels.append(damper)
            g_all.append(_num(match[0]["gini_all"]))
            g_hon.append(_num(match[0]["gini_honest"]))
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], g_all, width=0.4, label="Gini (all authors)")
    ax.bar([i + 0.2 for i in x], g_hon, width=0.4, label="Gini (honest only)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Gini coefficient of author earnings")
    ax.set_title(f"Earnings inequality per damper (intensity={it_ref})")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    p4 = os.path.join(out_dir, "gini.png")
    fig.tight_layout(); fig.savefig(p4, dpi=110); plt.close(fig)
    made.append(p4)

    return made


def make_burn_plot(rows: List[Dict[str, Any]], out_dir: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _num(v):
        return float("nan") if v == "nan" else float(v)

    burns = sorted({r["burn"] for r in rows})
    intensities = sorted({r["wash_intensity"] for r in rows})
    fig, ax = plt.subplots(figsize=(8, 5))
    for it in intensities:
        xs, ys = [], []
        for b in burns:
            m = [r for r in rows if r["burn"] == b and r["wash_intensity"] == it]
            if m:
                xs.append(b)
                ys.append(_num(m[0]["wash_roi"]))
        ax.plot(xs, ys, marker="o", label=f"intensity={it}")
    ax.axhline(1.0, color="black", ls="--", lw=1, label="break-even (ROI=1)")
    ax.set_xlabel("per-receipt burn (ATN)")
    ax.set_ylabel("wash ROI (mint gained / ATN burned)")
    ax.set_title("Burn rate needed to make wash-trading ROI-negative")
    ax.grid(True, alpha=0.3)
    ax.legend()
    p = os.path.join(out_dir, "burn_threshold.png")
    fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
    return p


def make_recall_plot(rows: List[Dict[str, Any]], out_dir: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _num(v):
        return float("nan") if v == "nan" else float(v)

    recalls = sorted({r["lineage_recall"] for r in rows})
    fig, ax = plt.subplots(figsize=(8, 5))
    for damper in ("out_of_lineage", "combo"):
        xs, ys = [], []
        for r in recalls:
            m = [x for x in rows if x["damper"] == damper
                 and x["lineage_recall"] == r]
            if m:
                xs.append(r)
                ys.append(_num(m[0]["wash_share"]))
        ax.plot(xs, ys, marker="o", label=damper)
    ax.set_xlabel("lineage-detection recall (fraction of sybils correctly flagged)")
    ax.set_ylabel("wash-traders' share of tool mint")
    ax.set_title("Lineage dampers degrade gracefully as sybil detection weakens")
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()   # left = omniscient, right = blind
    ax.legend()
    p = os.path.join(out_dir, "recall_degradation.png")
    fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Tool-economy mint simulator")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--epochs", type=int, nargs="+", default=[1, 5],
                    help="epoch counts to sweep")
    ap.add_argument("--intensities", type=float, nargs="+",
                    default=[0.5, 1.0, 2.0, 4.0],
                    help="wash-trader receipt multipliers to sweep")
    ap.add_argument("--burn", type=float, default=DEFAULT_BURN)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    rows = sweep(args.seed, args.epochs, args.intensities, burn=args.burn)
    csv_path = os.path.join(args.out, "sweep.csv")
    write_csv(rows, csv_path)
    print(f"wrote {csv_path} ({len(rows)} rows)")

    # Aux sweep 1: burn rate threshold (which burn kills wash ROI).
    max_epochs = max(args.epochs)
    burn_rates = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    brows = burn_sweep(args.seed, max_epochs, args.intensities, burn_rates)
    burn_csv = os.path.join(args.out, "burn_sweep.csv")
    write_csv(brows, burn_csv)
    print(f"wrote {burn_csv} ({len(brows)} rows)")

    # Aux sweep 2: lineage-detection recall degradation.
    recalls = [1.0, 0.9, 0.75, 0.5, 0.25, 0.0]
    ref_it = 2.0 if 2.0 in args.intensities else max(args.intensities)
    rrows = recall_sweep(args.seed, max_epochs, ref_it, recalls)
    recall_csv = os.path.join(args.out, "recall_sweep.csv")
    write_csv(rrows, recall_csv)
    print(f"wrote {recall_csv} ({len(rrows)} rows)")

    if not args.no_plots:
        pngs = make_plots(rows, args.out)
        pngs.append(make_burn_plot(brows, args.out))
        pngs.append(make_recall_plot(rrows, args.out))
        for p in pngs:
            print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Phase 10 confirmatory run — produces every artifact analyze.py reads.

Prereg: docs/phase10_prereg.md. Writes, under experiments/phase10/:
  events_T.jsonl, events_E.jsonl   — H1 standing rows (arms T, E)
  retrieval_rows.jsonl             — H2 retrieval rows (arms B, C, D)
  standings.json                   — H1 standings keyed by (arm,H,S,digest)
  mint_rows.jsonl                  — H3 per-tool mint vs battery pass rate
  aggregate10.json                 — written by analyze.py (not here)
  run.log                          — timings + graph statistics

NO LLM calls (prereg: confirmatory path is pure code). Standings come
from the REAL ledger replay in build_debates; H3 mint from the REAL
federated_epoch_close (compute_tool_mint + federated_reconcile_epoch,
vetting quorum=2). Guard #3: this script only PRODUCES artifacts;
analyze.py is pure over them and runs separately.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

HERE = Path(__file__).resolve().parent

from nodes.common.canonical_ordering import canonical_order
from nodes.common.event_gossip import EventBatch, Keypair
from nodes.common.federated_reconcile import federated_epoch_close

import build_tools
import build_debates as bd
import build_retrieval as br

CORPUS_PATH = HERE / "corpus.json"
RUN_LOG = HERE / "run.log"


def _log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line)
    with RUN_LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


# ---------------------------------------------------------------------------
# H3 — the economic loop prices quality (real federated close, vetted).
# ---------------------------------------------------------------------------

def _coords(rng: random.Random, axis: int = 4) -> List[float]:
    return bd._coords(rng, axis)


def _reg(seq: int, tool: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    d = tool["code_digest"]
    return {
        "kind": "sub_claim_sprouted", "seq": seq, "author_agent": tool["author"],
        "tendency_id": "correctness", "parent_id": "solver_root",
        "node_id": f"tool_{d[:12]}", "position": "pro",
        "coords": _coords(rng), "polarity_axis": _coords(rng),
        "content": tool["name"], "author_post": True, "artifact_digest": d,
        "manifest_meta": {"trust_class": "pinned", "author": tool["author"]},
    }


def _receipt(seq: int, caller: str, tool: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "kind": "tool_used", "seq": seq, "author_agent": caller,
        "manifest_digest": tool["code_digest"], "tool_author": tool["author"],
        "receipt_digest": f"r{seq:08d}" * 4, "ok": True, "fee_atn": 0.0,
        "attested": True, "score": 0.8,
    }


def _vet(seq: int, vetter: str, tool: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "kind": "tool_used", "seq": seq, "author_agent": vetter,
        "manifest_digest": tool["code_digest"], "tool_author": tool["author"],
        "receipt_digest": f"v{seq:08d}" * 4, "ok": True, "fee_atn": 0.0,
        "vet": True,
    }


def _batches(groups: List[List[Dict[str, Any]]], kp: Keypair) -> List[EventBatch]:
    chain: List[EventBatch] = []
    prev = b""
    for i, g in enumerate(groups, start=1):
        b = EventBatch(rpb_address="rpb_phase10", sender_pubkey=kp.public_key,
                       batch_seq=i, events=list(g), prev_batch_hash=prev,
                       timestamp=1_700_000_000.0 + i)
        chain.append(b)
        prev = b.content_hash()
    return chain


def build_h3_mint(corpus: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Drive the REAL federated close over the whole mixed population and
    correlate each tool's mint with its battery pass rate.

    Population: all correct/defective/seo/wash tools. Each tool gets:
      - a registration sprout with PRO supporters proportional to its
        genuine standing (honest tools attract organic PRO endorsers;
        wash/seo get none — same shape as sims/tool_economy honest vs
        wash),
      - distinct-fleet vets (quorum=2) so it is mint-eligible (the vet
        gate filters malice, not quality — a vetted wash tool is exactly
        the adversary the mint pricing must handle; H3 tests whether MINT,
        not vetting, tracks quality),
      - attested receipts from out-of-lineage callers: honest tools get
        broad distinct-caller usage; wash tools get pumped sybil receipts
        from a small pool; seo tools get sparse usage. Usage volume is
        NOT quality — the test is whether standing x usage, priced by the
        gate, still rank-tracks pass_rate.

    Each tool author is a distinct 0x id and each caller/supporter/vetter
    rides a distinct signing key, so neither the wire dedup nor the fleet
    collapse fires spuriously (owner map left empty: unknown owner = no
    exclusion, the documented degraded mode — the interesting adversary is
    the sybil that EVADES the owner map, exactly as the H1 sweep frames S).
    """
    rng = random.Random(build_tools.MASTER_SEED)
    tools = corpus["tools"]

    reg_groups: List[List[Dict[str, Any]]] = []
    receipt_groups: List[List[Dict[str, Any]]] = []
    seq = 0

    for tool in tools:
        seq += 1
        d = tool["code_digest"]
        group = [_reg(seq, tool, rng)]
        # organic PRO supporters ~ genuine quality: honest tools earn
        # endorsers; wash/seo do not. Bounded so standing stays sane.
        if tool["trust_kind"] == "correct":
            n_sup = rng.randint(2, 6)
        elif tool["trust_kind"] == "defective":
            n_sup = rng.randint(0, 2)
        else:                              # seo / wash: no organic support
            n_sup = 0
        for _ in range(n_sup):
            seq += 1
            group.append({
                "kind": "sub_claim_sprouted", "seq": seq,
                "author_agent": f"sup_{seq}", "tendency_id": "correctness",
                "parent_id": f"tool_{d[:12]}",
                "node_id": f"sup_{d[:12]}_{seq}", "position": "pro",
                "coords": _coords(rng), "polarity_axis": _coords(rng),
                "content": "endorse", "author_post": True,
            })
        reg_groups.append(group)

    # Receipts on a SEPARATE signing key (wire dedup excludes same-key).
    for tool in tools:
        recs: List[Dict[str, Any]] = []
        if tool["wash"]:
            sybils = [f"wsyb_{tool['code_digest'][:8]}_{i}"
                      for i in range(rng.randint(2, 4))]
            for _ in range(rng.randint(20, 60)):
                seq += 1
                recs.append(_receipt(seq, rng.choice(sybils), tool))
        elif tool["seo"]:
            callers = [f"seocall_{tool['code_digest'][:8]}_{i}"
                       for i in range(rng.randint(1, 2))]
            for _ in range(rng.randint(1, 3)):
                seq += 1
                recs.append(_receipt(seq, rng.choice(callers), tool))
        else:
            callers = [f"user_{tool['code_digest'][:8]}_{i}"
                       for i in range(rng.randint(6, 16))]
            for _ in range(rng.randint(6, 20)):
                seq += 1
                recs.append(_receipt(seq, rng.choice(callers), tool))
        if recs:
            receipt_groups.append(recs)

    # Distinct-fleet vets: two vetters, each its own key => quorum met.
    vet_groups: List[List[Dict[str, Any]]] = []
    for v in (1, 2):
        vets: List[Dict[str, Any]] = []
        for tool in tools:
            seq += 1
            vets.append(_vet(seq, f"phase10_vetter_{v}", tool))
        vet_groups.append(vets)

    batches = (
        _batches(reg_groups, Keypair.generate())
        + _batches(receipt_groups, Keypair.generate())
        + _batches([vet_groups[0]], Keypair.generate())
        + _batches([vet_groups[1]], Keypair.generate())
    )
    result = federated_epoch_close(canonical_order(batches))
    tm = result["tool_mint"]

    rows: List[Dict[str, Any]] = []
    for tool in tools:
        entry = tm.get(tool["code_digest"])
        rows.append({
            "code_digest": tool["code_digest"],
            "task": tool["task"],
            "trust_kind": tool["trust_kind"],
            "pass_rate": tool["pass_rate"],
            "defective": tool["defective"],
            "seo": tool["seo"],
            "wash": tool["wash"],
            "mint": float(entry["mint"]) if entry else 0.0,
            "standing": float(entry["standing"]) if entry else 0.0,
            "greenlit": bool(entry["greenlit"]) if entry else False,
        })
    return rows, result


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")


def main() -> int:
    RUN_LOG.write_text("", encoding="utf-8")
    t0 = time.time()
    _log(f"phase10 run start; master_seed={build_tools.MASTER_SEED}")

    # Corpus (rebuild deterministically so the run is self-contained).
    _log("building corpus (deterministic)...")
    tb = time.time()
    corpus = build_tools.build_corpus(build_tools.MASTER_SEED)
    _log(f"corpus: {corpus['n_tools']} tools, families={corpus['families']} "
         f"({time.time()-tb:.1f}s)")
    n_def = sum(1 for t in corpus["tools"] if t["defective"])
    n_seo = sum(1 for t in corpus["tools"] if t["seo"])
    n_wash = sum(1 for t in corpus["tools"] if t["wash"])
    pop = bd.h1_population(corpus)
    n_con = sum(1 for t in pop if bd._con_fires(t))
    n_falsecon = sum(1 for t in pop if not t["defective"] and bd._con_fires(t))
    _log(f"graph stats: H1 population={len(pop)} "
         f"(correct={sum(1 for t in pop if not t['defective'])}, "
         f"defective={sum(1 for t in pop if t['defective'])}); "
         f"CONs fired={n_con} (false CONs={n_falsecon}); "
         f"corpus defective(any)={n_def}, seo={n_seo}, wash={n_wash}")

    # H1 — debates (arms T and E) over the sweep grid.
    _log(f"H1 debates: sweep H={bd.H_VALUES} x S={bd.S_VALUES} x "
         f"{len(pop)} tools x 2 arms...")
    tb = time.time()
    debates = bd.build_all(corpus)
    _log(f"H1 debates done: |T|={len(debates['T'])} |E|={len(debates['E'])} "
         f"({time.time()-tb:.1f}s)")
    _write_jsonl(HERE / "events_T.jsonl", debates["T"])
    _write_jsonl(HERE / "events_E.jsonl", debates["E"])
    standings = {
        f"{r['arm']}|{r['H']}|{r['S']}|{r['code_digest']}": r["standing"]
        for arm in ("T", "E") for r in debates[arm]
    }
    (HERE / "standings.json").write_text(
        json.dumps(standings, indent=2, sort_keys=True), encoding="utf-8")

    # H2 — retrieval (arms B, C, D) on salted + clean corpora.
    _log("H2 retrieval: arms B/C/D over salted + clean corpora...")
    tb = time.time()
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="phase10_retr_"))
    retr_rows = br.build_rows(corpus, tmp)
    _log(f"H2 retrieval done: {len(retr_rows)} rows ({time.time()-tb:.1f}s)")
    _write_jsonl(HERE / "retrieval_rows.jsonl", retr_rows)

    # H3 — mint vs quality via the real federated close.
    _log("H3 mint: real federated_epoch_close over mixed population (vetted)...")
    tb = time.time()
    mint_rows, close = build_h3_mint(corpus)
    minted = sum(1 for r in mint_rows if r["mint"] > 0)
    _log(f"H3 mint done: {minted}/{len(mint_rows)} tools minted; "
         f"total_mint={close['total_mint']:.4f} n_events={close['n_events']} "
         f"({time.time()-tb:.1f}s)")
    _write_jsonl(HERE / "mint_rows.jsonl", mint_rows)

    _log(f"phase10 run complete in {time.time()-t0:.1f}s")
    _log("run analyze.py to produce aggregate10.json + verdicts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

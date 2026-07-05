"""Vetting-knob sweep for the tool-substrate candidate pool.

Answers the four questions `docs/tool_substrate.md` (Vetting section)
left for sims, driving the REAL close function — every number below
comes out of ``compute_tool_mint`` (nodes/common/federated_reconcile.py)
called epoch after epoch with fabricated canonical events and the
carried ``vetting`` / ``registrations`` state, exactly as the federated
close driver does. No formula is reimplemented here; the sim only
fabricates inputs and reads outputs (same contract as simulate.py).

  Q1 QUORUM (N):     honest time-to-greenlight vs sybil slip-through,
                     with colluding vetters evading the owner map at
                     rate ``evasion`` (like MEMO.md's lineage_recall).
  Q2 SHARE x K:      validator income per vet vs author dilution, and
                     the "vet mediocre tools instead of authoring"
                     perverse-incentive threshold R*.
  Q3 BUST/SLASH:     does weight = 1/(1+busts) deter a rubber-stamp
                     validator, in pivotal vs redundant quorum regimes?
  Q4 COLLUSION:      author + M evading socks greenlight a wash tool —
                     marginal profit on top of the usage damper, and
                     whether the royalty leaks anything new.

Determinism: every cell is seeded via ``simulate._cell_seed``; the
close is a pure function of its inputs, so identical seed => identical
JSON/plots. No wall-clock anywhere.

Run:  python sims/tool_economy/vet_sweep.py --seed 1234
      (writes out/vet_sweep.json + four PNGs; ~40 s)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(__file__)
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from nodes.common.federated_reconcile import (  # noqa: E402
    VET_BUST_THRESHOLD,
    VET_QUORUM,
    VET_ROYALTY_EPOCHS,
    VET_ROYALTY_SHARE,
    compute_tool_mint,
)
from nodes.common.world_model_substrate.adapter import (  # noqa: E402
    build_charter_world,
)
from nodes.common.world_model_substrate.aggregate import apply_events  # noqa: E402
from nodes.common.world_model_substrate.infer import _artifact_standing  # noqa: E402

from sims.tool_economy import simulate as base  # noqa: E402

OUT_DIR = base.OUT_DIR


# ---------------------------------------------------------------------------
# Harness: a persistent world + carried vetting state, closed epoch by
# epoch through the REAL compute_tool_mint.
# ---------------------------------------------------------------------------

class VetHarness:
    """Multi-epoch driver around ``compute_tool_mint``.

    Owns one replay world (registrations sprout real claim nodes so
    standing and charter_violation_score are the production values) and
    threads ``registrations_next`` / ``vetting_next`` between closes —
    the same carry-over contract the federated close driver uses.
    """

    def __init__(self, *, seed: int,
                 quorum: float = VET_QUORUM,
                 share: float = VET_ROYALTY_SHARE,
                 royalty_epochs: int = VET_ROYALTY_EPOCHS,
                 bust_threshold: float = VET_BUST_THRESHOLD):
        self.rng = random.Random(seed)
        self.world = build_charter_world(embedding_dim=base.EMBED_DIM)
        self.quorum = float(quorum)
        self.share = float(share)
        self.royalty_epochs = int(royalty_epochs)
        self.bust_threshold = float(bust_threshold)
        self.owner_map: Dict[str, str] = {}
        self.registrations: Optional[Dict[str, Dict[str, str]]] = None
        self.vetting: Optional[Dict[str, Any]] = None
        self.tool_author: Dict[str, str] = {}
        self.earned: Dict[str, float] = {}          # cumulative per agent
        self.per_epoch: List[Dict[str, float]] = []  # per-close agent mint
        self.epoch = 0
        self._seq = 0
        self._pending: List[Dict[str, Any]] = []

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def add_owner(self, agent: str, owner: str) -> None:
        self.owner_map[agent] = owner

    def register(self, author: str, *, n_supporters: int = 0) -> str:
        """Publish a pinned tool: sprout its claim node (+ organic
        supporter PRO children => standing 1+n_supporters) into the
        world, and queue the registration event for the next close."""
        digest = base._digest(self.rng)
        label = f"vs_node_{self._next_seq():05d}"
        ev = base._registration_event(
            self._next_seq(), author, digest, label, self.rng)
        ev["sender"] = f"key_{author}"
        group = [ev]
        for s in range(n_supporters):
            group.append(base._supporter_event(
                self._next_seq(), f"sup_{author}_{digest[:6]}_{s}",
                label, f"{label}_s{s}", self.rng))
        # Same-call application so supporter parents resolve onto the
        # manifest node (per-call remap, see simulate._batches).
        apply_events(self.world, group, equilibrate_after=False)
        self._pending.append(ev)
        self.tool_author[digest] = author
        return digest

    def vet(self, digest: str, vetter: str) -> None:
        self._pending.append({
            "kind": "tool_used",
            "seq": self._next_seq(),
            "author_agent": vetter,
            "manifest_digest": digest,
            "tool_author": self.tool_author[digest],
            "receipt_digest": f"v{self._seq:07d}" * 4,
            "ok": True,
            "fee_atn": 0.0,
            "vet": True,
            "sender": f"key_{vetter}",
        })

    def attest(self, digest: str, caller: str, n: int = 1) -> None:
        for _ in range(n):
            ev = base._receipt_event(
                self._next_seq(), caller, digest,
                self.tool_author[digest], ok=True)
            ev["sender"] = f"key_{caller}"
            self._pending.append(ev)

    def bust(self, digest: str) -> None:
        """Land a winning charter CON against the manifest's claim node
        (a real sprout under a charter tendency — the production bust
        pass reads it via charter_violation_score)."""
        nodes = _artifact_standing(self.world).get(digest) or []
        target = sorted(n.id for n in nodes)[0]
        ev = {
            "kind": "sub_claim_sprouted",
            "seq": self._next_seq(),
            "author_agent": "charter_watch",
            "tendency_id": "life_precious",
            "parent_id": "vs_flag_root",
            "node_id": f"vs_flag_{self._seq:05d}",
            "position": "con",
            "coords": base._coords(self.rng, axis=0),
            "polarity_axis": base._coords(self.rng, axis=0),
            "content": f"reproducible exploit in {target}",
            "author_post": True,
        }
        apply_events(self.world, [ev], equilibrate_after=False)

    def close(self) -> Dict[str, Any]:
        out = compute_tool_mint(
            self.world, self._pending,
            registrations=self.registrations,
            agent_owner_map=self.owner_map,
            vetting=self.vetting,
            vet_quorum=self.quorum,
            vet_royalty_share=self.share,
            vet_royalty_epochs=self.royalty_epochs,
            vet_bust_threshold=self.bust_threshold,
        )
        self.registrations = out["registrations_next"]
        self.vetting = out["vetting_next"]
        epoch_mint: Dict[str, float] = {}
        for per_agent in out["node_agent"].values():
            for agent, amt in per_agent.items():
                epoch_mint[agent] = epoch_mint.get(agent, 0.0) + amt
                self.earned[agent] = self.earned.get(agent, 0.0) + amt
        self.per_epoch.append(epoch_mint)
        self._pending = []
        self.epoch += 1
        return out

    def greenlit(self, digest: str) -> bool:
        m = (self.vetting or {}).get("manifests", {}).get(digest)
        return bool(m and m.get("greenlit"))

    def busts_of(self, agent: str) -> int:
        return int((self.vetting or {}).get("busts", {}).get(agent, 0))


# ---------------------------------------------------------------------------
# Q1 — QUORUM: honest time-to-greenlight vs sybil slip-through
# ---------------------------------------------------------------------------

def _run_quorum_cell(seed: int, quorum: int, evasion: float, *,
                     n_honest: int, n_validators: int, p_vet: float,
                     n_colluders: int, m_socks: int, max_epochs: int,
                     target_frac: float) -> Dict[str, Any]:
    """One population, one (N, evasion) point.

    Honest validators vet each not-yet-greenlit honest tool with prob
    ``p_vet`` per epoch (limited attention). Colluding families are the
    ONLY vetters of their own wash tool: each sock is caught by the
    owner map with prob (1 - evasion); a caught sock shares the
    author's registered owner and is excluded by the real merge pass.
    """
    rng = random.Random(seed)
    h = VetHarness(seed=seed * 7 + 1, quorum=float(quorum))
    validators = [f"hval_{i}" for i in range(n_validators)]
    for v in validators:
        h.add_owner(v, f"own_{v}")
    honest: List[str] = []
    for i in range(n_honest):
        a = f"hauth_{i:02d}"
        h.add_owner(a, f"own_{a}")
        honest.append(h.register(a, n_supporters=3))
    coll: List[str] = []
    for i in range(n_colluders):
        a = f"cauth_{i:02d}"
        fam = f"own_fam_{i}"
        h.add_owner(a, fam)
        d = h.register(a)
        for j in range(m_socks):
            sock = f"sock_{i}_{j}"
            if rng.random() >= evasion:     # caught by the owner map
                h.add_owner(sock, fam)
            h.vet(d, sock)                  # queued for the first close
        coll.append(d)

    greenlit_at: Dict[str, int] = {}
    epochs_to_target = max_epochs
    for e in range(max_epochs):
        for v in validators:
            for d in honest:
                if d not in greenlit_at and rng.random() < p_vet:
                    h.vet(d, v)
        h.close()
        for d in honest:
            if d not in greenlit_at and h.greenlit(d):
                greenlit_at[d] = e + 1      # 1-based: closes needed
        if len(greenlit_at) >= target_frac * n_honest:
            epochs_to_target = e + 1
            break
    slip = sum(1 for d in coll if h.greenlit(d)) / len(coll)
    return {"epochs_to_target": epochs_to_target, "slip_rate": slip}


def sweep_quorum(seed: int, quorums=(1, 2, 3, 5),
                 evasions=(0.0, 0.3, 1.0), *,
                 n_honest: int = 12, n_validators: int = 6,
                 p_vet: float = 0.35, n_colluders: int = 4,
                 m_socks: int = 3, max_epochs: int = 25,
                 target_frac: float = 0.8,
                 n_seeds: int = 5) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for q in quorums:
        for ev in evasions:
            t2g, slips = [], []
            for s in range(n_seeds):
                cs = base._cell_seed(seed, "quorum", q, ev, s)
                r = _run_quorum_cell(
                    cs, q, ev, n_honest=n_honest,
                    n_validators=n_validators, p_vet=p_vet,
                    n_colluders=n_colluders, m_socks=m_socks,
                    max_epochs=max_epochs, target_frac=target_frac)
                t2g.append(r["epochs_to_target"])
                slips.append(r["slip_rate"])
            rows.append({
                "quorum": q, "evasion": ev, "m_socks": m_socks,
                "epochs_to_80pct_honest": round(sum(t2g) / len(t2g), 3),
                "collusion_slip_rate": round(sum(slips) / len(slips), 4),
            })
    return rows


# ---------------------------------------------------------------------------
# Q2 — ROYALTY SHARE x K: validator income vs author dilution, and the
# vet-mediocre-tools-instead-of-authoring threshold
# ---------------------------------------------------------------------------

def royalty_grid(seed: int, shares=(0.05, 0.1, 0.2, 0.3),
                 ks=(4, 8, 16), *, standing_sup: int = 3,
                 users: int = 5) -> List[Dict[str, Any]]:
    """One median-usage honest tool per cell, quorum-2 vetted at birth,
    run for its first 2K epochs. Everything deterministic."""
    rows: List[Dict[str, Any]] = []
    for share in shares:
        for k in ks:
            h = VetHarness(seed=base._cell_seed(seed, "royalty", share, k),
                           quorum=2.0, share=share, royalty_epochs=k)
            author = "alice"
            h.add_owner(author, "own_alice")
            d = h.register(author, n_supporters=standing_sup)
            for v in ("val_a", "val_b"):
                h.vet(d, v)
            total_epochs = 2 * k
            for _ in range(total_epochs):
                for u in range(users):
                    h.attest(d, f"user_{u}")
                h.close()
            mint_total = sum(sum(m.values()) for m in h.per_epoch)
            author_earn = h.earned.get(author, 0.0)
            val_earn = h.earned.get("val_a", 0.0)
            rows.append({
                "share": share, "k": k,
                "mint_per_epoch": round(mint_total / total_epochs, 6),
                "author_retained_frac_2k": round(author_earn / mint_total, 6),
                "validator_income_per_vet": round(val_earn, 6),
                "validator_income_frac_of_tool": round(val_earn / mint_total, 6),
            })
    return rows


def royalty_factory(seed: int, shares=(0.05, 0.1, 0.2, 0.3),
                    ks=(4, 8, 16), *, users: int = 5,
                    good_sup: int = 3) -> List[Dict[str, Any]]:
    """The perverse-incentive probe: a 'vet factory' validator rubber-
    stamps ONE fresh mediocre tool (standing 1, median usage) per epoch
    (quorum met with a partner) while an honest author runs one GOOD
    tool (standing 4, same usage). Steady-state income rates compared
    over the last K epochs of a 3K-epoch run. R* = mediocre vets per
    epoch needed to out-earn authoring the good tool."""
    rows: List[Dict[str, Any]] = []
    for share in shares:
        for k in ks:
            h = VetHarness(seed=base._cell_seed(seed, "factory", share, k),
                           quorum=2.0, share=share, royalty_epochs=k)
            good_author = "goodauth"
            h.add_owner(good_author, "own_good")
            gd = h.register(good_author, n_supporters=good_sup)
            for v in ("nv_g1", "nv_g2"):
                h.vet(gd, v)
            factory, partner = "factory", "partner"
            live = [gd]
            epochs = 3 * k
            for e in range(epochs):
                ma = f"medauth_{e:03d}"
                h.add_owner(ma, f"own_{ma}")
                md = h.register(ma)             # standing 1: mediocre
                h.vet(md, factory)
                h.vet(md, partner)
                live.append(md)
                for d in live:
                    for u in range(users):
                        h.attest(d, f"u_{d[:6]}_{u}")
                h.close()
            tail = h.per_epoch[-k:]
            val_rate = sum(m.get(factory, 0.0) for m in tail) / k
            author_rate = sum(m.get(good_author, 0.0) for m in tail) / k
            r_star = (author_rate / val_rate) if val_rate > 0 else math.inf
            rows.append({
                "share": share, "k": k,
                "factory_income_per_epoch": round(val_rate, 6),
                "good_author_income_per_epoch": round(author_rate, 6),
                "vets_per_epoch_to_beat_author": round(r_star, 3),
                "vetting_beats_authoring_at_1_per_epoch":
                    bool(val_rate >= author_rate),
            })
    return rows


# ---------------------------------------------------------------------------
# Q3 — BUST/SLASH: rubber-stamp vs careful validator
# ---------------------------------------------------------------------------

def _run_bust_cell(seed: int, p_bust: float, redundant: bool, *,
                   epochs: int, users: int, k: int, share: float,
                   bust_delay: int) -> Dict[str, Any]:
    """Two validators on independent, identically-distributed tool
    streams (2 offered per epoch each; a tool is dirty with prob
    p_bust and takes a winning exploit CON ``bust_delay`` epochs after
    greenlight):

      rubber  — vets everything offered (2/epoch, dirty included);
      careful — vets at most ONE clean offered tool per epoch (half the
                volume, perfect filter).

    Regimes: 'pivotal' (each vetted tool gets exactly 1 neutral partner
    vet, so quorum 2 NEEDS the subject's full weight) vs 'redundant'
    (2 neutral partners — quorum met without the subject)."""
    rng = random.Random(seed)
    h = VetHarness(seed=seed + 11, quorum=2.0, share=share,
                   royalty_epochs=k)
    rubber, careful = "rubber", "careful"
    h.add_owner(rubber, "own_rubber")
    h.add_owner(careful, "own_careful")
    neutral_i = 0
    greenlit_epoch: Dict[str, int] = {}
    bust_at: Dict[str, int] = {}       # digest -> epoch to land the CON
    vetted: List[str] = []             # all subject-vetted digests
    n_partners = 2 if redundant else 1

    def _offer(subject: str, take_all: bool, e: int) -> None:
        nonlocal neutral_i
        taken = 0
        for j in range(2):
            a = f"a_{subject[0]}_{e:03d}_{j}"
            h.add_owner(a, f"own_{a}")
            dirty = rng.random() < p_bust
            d = h.register(a)
            want = take_all or (not dirty and taken < 1)
            if not want:
                continue
            taken += 1
            h.vet(d, subject)
            for _ in range(n_partners):
                neutral_i += 1
                h.vet(d, f"nv_{neutral_i:04d}")
            vetted.append(d)
            if dirty:
                bust_at[d] = -1        # scheduled once greenlit
        return

    for e in range(epochs):
        _offer(rubber, True, e)
        _offer(careful, False, e)
        # land scheduled CONs before this close
        for d, be in list(bust_at.items()):
            if be == e:
                h.bust(d)
                del bust_at[d]
        # usage: only greenlit tools still inside their royalty window
        # draw attestations (post-window usage can't pay validators)
        for d, ge in greenlit_epoch.items():
            if h.epoch - ge < k + 1:
                for u in range(users):
                    h.attest(d, f"u_{d[:6]}_{u}")
        h.close()
        for d in vetted:
            if d not in greenlit_epoch and h.greenlit(d):
                greenlit_epoch[d] = h.epoch - 1
                if d in bust_at and bust_at[d] == -1:
                    bust_at[d] = min(epochs - 1,
                                     h.epoch - 1 + bust_delay)
    return {
        "rubber_royalty": h.earned.get(rubber, 0.0),
        "careful_royalty": h.earned.get(careful, 0.0),
        "rubber_busts": h.busts_of(rubber),
        "careful_busts": h.busts_of(careful),
        "rubber_final_weight": 1.0 / (1.0 + h.busts_of(rubber)),
    }


def sweep_bust(seed: int, p_busts=(0.05, 0.2, 0.5), *,
               epochs: int = 50, users: int = 3, k: int = 8,
               share: float = 0.1, bust_delay: int = 2,
               n_seeds: int = 3) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in p_busts:
        for redundant in (False, True):
            acc: List[Dict[str, Any]] = []
            for s in range(n_seeds):
                cs = base._cell_seed(seed, "bust", p, redundant, s)
                acc.append(_run_bust_cell(
                    cs, p, redundant, epochs=epochs, users=users,
                    k=k, share=share, bust_delay=bust_delay))
            n = len(acc)
            rub = sum(a["rubber_royalty"] for a in acc) / n
            car = sum(a["careful_royalty"] for a in acc) / n
            rows.append({
                "p_bust": p,
                "regime": "redundant" if redundant else "pivotal",
                "rubber_royalty_50ep": round(rub, 4),
                "careful_royalty_50ep": round(car, 4),
                "rubber_busts": round(
                    sum(a["rubber_busts"] for a in acc) / n, 2),
                "rubber_final_weight": round(
                    sum(a["rubber_final_weight"] for a in acc) / n, 4),
                "rubber_wins": bool(rub > car),
            })
    return rows


# ---------------------------------------------------------------------------
# Q4 — COLLUSION: author + M evading socks greenlight a wash tool
# ---------------------------------------------------------------------------

def _run_collusion_cell(seed: int, evasion: float, quorum: int,
                        share: float, *, m_socks: int, epochs: int,
                        k: int, pump: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    h = VetHarness(seed=seed + 3, quorum=float(quorum), share=share,
                   royalty_epochs=k)
    author, fam = "washa", "own_washfam"
    h.add_owner(author, fam)
    d = h.register(author)              # standing 1: no organic support
    socks = [f"wsock_{j}" for j in range(m_socks)]
    for s in socks:
        if rng.random() >= evasion:     # caught: linked to the family
            h.add_owner(s, fam)
        h.vet(d, s)
    for _ in range(epochs):
        for s in socks:
            h.attest(d, s, n=pump)      # the wash pump
        h.close()
    family = {author, *socks}
    fam_total = sum(v for a, v in h.earned.items() if a in family)
    sock_royalty = sum(v for a, v in h.earned.items() if a in socks)
    return {
        "greenlit": h.greenlit(d),
        "family_mint": fam_total,
        "sock_royalty": sock_royalty,
    }


def sweep_collusion(seed: int, evasions=(0.0, 0.3, 1.0),
                    quorums=(2, 3), *, m_socks: int = 3,
                    epochs: int = 16, k: int = 8,
                    share: float = VET_ROYALTY_SHARE, pump: int = 5,
                    n_seeds: int = 8) -> List[Dict[str, Any]]:
    """Family profit per (N, evasion). ``damper_leak`` is the same cell
    with the vet gate OFF (quorum=0): what the usage damper alone
    already leaks. The vet gate's contribution = gated family mint,
    which is <= the leak (the gate can only withhold, never add)."""
    rows: List[Dict[str, Any]] = []
    for q in quorums:
        for ev in evasions:
            acc: List[Dict[str, Any]] = []
            leaks: List[float] = []
            for s in range(n_seeds):
                cs = base._cell_seed(seed, "collusion", q, ev, s)
                acc.append(_run_collusion_cell(
                    cs, ev, q, share, m_socks=m_socks, epochs=epochs,
                    k=k, pump=pump))
                # damper-only baseline: identical family, no vet gate
                leaks.append(_run_collusion_cell(
                    cs, ev, 0, share, m_socks=m_socks, epochs=epochs,
                    k=k, pump=pump)["family_mint"])
            n = len(acc)
            rows.append({
                "quorum": q, "evasion": ev, "m_socks": m_socks,
                "greenlight_rate": round(
                    sum(1.0 for a in acc if a["greenlit"]) / n, 4),
                "family_mint": round(
                    sum(a["family_mint"] for a in acc) / n, 4),
                "damper_leak_ungated": round(sum(leaks) / n, 4),
                "sock_royalty": round(
                    sum(a["sock_royalty"] for a in acc) / n, 4),
            })
    return rows


# ---------------------------------------------------------------------------
# Output: tables, JSON, plots
# ---------------------------------------------------------------------------

def _print_table(title: str, rows: List[Dict[str, Any]]) -> None:
    print(f"\n== {title} ==")
    if not rows:
        print("(no rows)")
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows))
              for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def make_plots(tables: Dict[str, List[Dict[str, Any]]],
               out_dir: str) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    made: List[str] = []

    # Q1: time-to-greenlight and slip rate vs quorum
    q = tables["quorum"]
    quorums = sorted({r["quorum"] for r in q})
    evs = sorted({r["evasion"] for r in q})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ys = [next(r["epochs_to_80pct_honest"] for r in q
               if r["quorum"] == n and r["evasion"] == evs[0])
          for n in quorums]
    ax1.plot(quorums, ys, marker="o", color="tab:blue")
    ax1.set_xlabel("vet quorum N")
    ax1.set_ylabel("epochs until 80% of honest tools greenlit")
    ax1.set_title("Honest time-to-greenlight")
    ax1.grid(True, alpha=0.3)
    for ev in evs:
        ys = [next(r["collusion_slip_rate"] for r in q
                   if r["quorum"] == n and r["evasion"] == ev)
              for n in quorums]
        ax2.plot(quorums, ys, marker="s", label=f"evasion={ev}")
    ax2.set_xlabel("vet quorum N")
    ax2.set_ylabel("collusion-backed tools greenlit (rate)")
    ax2.set_title("Sybil slip-through (3 socks/family)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    p = os.path.join(out_dir, "vet_quorum.png")
    fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
    made.append(p)

    # Q2: perverse-incentive threshold R*
    f = tables["factory"]
    shares = sorted({r["share"] for r in f})
    fig, ax = plt.subplots(figsize=(8, 5))
    for k in sorted({r["k"] for r in f}):
        ys = [next(r["vets_per_epoch_to_beat_author"] for r in f
                   if r["share"] == s and r["k"] == k) for s in shares]
        ax.plot(shares, ys, marker="o", label=f"K={k}")
    ax.axhline(1.0, color="black", ls="--", lw=1,
               label="danger line (1 mediocre vet/epoch suffices)")
    ax.set_xlabel("royalty share")
    ax.set_ylabel("mediocre vets/epoch to out-earn a good author (R*)")
    ax.set_title("Perverse incentive: vet factories vs authorship")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    p = os.path.join(out_dir, "vet_royalty_perverse.png")
    fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
    made.append(p)

    # Q3: rubber vs careful cumulative royalty
    b = tables["bust"]
    fig, ax = plt.subplots(figsize=(8, 5))
    p_busts = sorted({r["p_bust"] for r in b})
    x = range(len(p_busts))
    for off, regime, color in ((-0.3, "pivotal", "tab:green"),
                               (0.1, "redundant", "tab:red")):
        rub = [next(r["rubber_royalty_50ep"] for r in b
                    if r["p_bust"] == pb and r["regime"] == regime)
               for pb in p_busts]
        car = [next(r["careful_royalty_50ep"] for r in b
                    if r["p_bust"] == pb and r["regime"] == regime)
               for pb in p_busts]
        ax.bar([i + off for i in x], rub, width=0.2, color=color,
               label=f"rubber-stamp ({regime})")
        ax.bar([i + off + 0.2 for i in x], car, width=0.2, color=color,
               alpha=0.45, label=f"careful ({regime})")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"p_bust={pb}" for pb in p_busts])
    ax.set_ylabel("cumulative royalty over 50 epochs")
    ax.set_title("1/(1+busts) deters ONLY when the vet is pivotal")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    p = os.path.join(out_dir, "vet_bust_deterrence.png")
    fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
    made.append(p)

    # Q4: collusion family mint vs evasion
    c = tables["collusion"]
    fig, ax = plt.subplots(figsize=(8, 5))
    evs = sorted({r["evasion"] for r in c})
    for qn in sorted({r["quorum"] for r in c}):
        ys = [next(r["family_mint"] for r in c
                   if r["quorum"] == qn and r["evasion"] == ev)
              for ev in evs]
        ax.plot(evs, ys, marker="o", label=f"vet-gated, N={qn}")
    ys = [next(r["damper_leak_ungated"] for r in c
               if r["quorum"] == sorted({r['quorum'] for r in c})[0]
               and r["evasion"] == ev) for ev in evs]
    ax.plot(evs, ys, marker="x", ls="--", color="gray",
            label="damper leak (no vet gate)")
    ax.set_xlabel("owner-map evasion rate (socks the map misses)")
    ax.set_ylabel("wash family mint over 16 epochs")
    ax.set_title("Collusion profit: the vet gate can only shrink the leak")
    ax.grid(True, alpha=0.3)
    ax.legend()
    p = os.path.join(out_dir, "vet_collusion.png")
    fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
    made.append(p)

    return made


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Vetting-knob sweep")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    tables = {
        "quorum": sweep_quorum(args.seed),
        "royalty": royalty_grid(args.seed),
        "factory": royalty_factory(args.seed),
        "bust": sweep_bust(args.seed),
        "collusion": sweep_collusion(args.seed),
    }
    for name, rows in tables.items():
        _print_table(name, rows)

    os.makedirs(args.out, exist_ok=True)
    jpath = os.path.join(args.out, "vet_sweep.json")
    with open(jpath, "w", encoding="utf-8") as fh:
        json.dump({"seed": args.seed, **tables}, fh, indent=2,
                  sort_keys=True)
    print(f"\nwrote {jpath}")

    if not args.no_plots:
        for p in make_plots(tables, args.out):
            print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

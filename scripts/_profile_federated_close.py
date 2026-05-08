"""Profile federated_epoch_close to find the bottleneck.

Builds a realistic 2-sender canonical order and times each phase of
federated_epoch_close at increasing embedding_dim values.
"""

from __future__ import annotations

import cProfile
import pstats
import sys
import time
from io import StringIO
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if (_REPO_ROOT / "nodes" / "__init__.py").exists():
    sys.path.insert(0, str(_REPO_ROOT))

from nodes.common.canonical_ordering import canonical_order
from nodes.common.event_gossip import EventBatch, Keypair
from nodes.common.federated_reconcile import federated_epoch_close
from nodes.common.world_model_substrate.adapter import N_DIMS


def _make_chain(rpb: str, kp: Keypair, agent_id: str, charter_axis: int,
                n: int, embedding_dim: int) -> list:
    chain = []
    prev = b""
    for i in range(1, n + 1):
        coords = [0.0] * (N_DIMS + embedding_dim)
        coords[charter_axis] = 0.5
        # Spread across the embedding tail.
        for j in range(min(8, embedding_dim)):
            coords[N_DIMS + j] = 0.05 * ((i + j) % 5)
        ev = {
            "kind": "observation_added",
            "seq": 1,
            "author_agent": agent_id,
            "obs_id": f"obs_{agent_id}_{i}",
            "coords": coords,
            "label": f"{agent_id}_{i}",
        }
        b = EventBatch(
            rpb_address=rpb,
            sender_pubkey=kp.public_key,
            batch_seq=i,
            events=[ev],
            prev_batch_hash=prev,
            timestamp=1.0 + i,
        )
        chain.append(b)
        prev = b.content_hash()
    return chain


def run(embedding_dim: int, n_per_sender: int = 5):
    rpb = "rpb_profile"
    kp_a = Keypair.generate()
    kp_b = Keypair.generate()
    chain_a = _make_chain(rpb, kp_a, "0xAAA", 0, n_per_sender, embedding_dim)
    chain_b = _make_chain(rpb, kp_b, "0xBBB", 2, n_per_sender, embedding_dim)
    canonical = canonical_order(chain_a + chain_b)
    print(
        f"\n--- embedding_dim={embedding_dim}, batches={len(canonical.ordered_batches)} ---",
        flush=True,
    )

    t0 = time.time()
    result = federated_epoch_close(canonical, embedding_dim=embedding_dim)
    t1 = time.time()
    print(f"federated_epoch_close: {t1-t0:.2f}s", flush=True)
    print(f"  agent_mint: {result['authoritative_payload']['agent_mint']}", flush=True)


def profile(embedding_dim: int, n_per_sender: int = 5):
    pr = cProfile.Profile()
    pr.enable()
    run(embedding_dim, n_per_sender)
    pr.disable()
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(25)
    print(s.getvalue(), flush=True)


if __name__ == "__main__":
    # Ladder of dims to find scaling behavior
    for dim in (16, 64, 256, 1024):
        try:
            run(dim, n_per_sender=5)
        except Exception as e:
            print(f"ERROR at dim={dim}: {e}", flush=True)
            break

    # Profile the worst case
    print("\n=== profiling at embedding_dim=256, 5 batches/sender ===", flush=True)
    profile(256, n_per_sender=5)

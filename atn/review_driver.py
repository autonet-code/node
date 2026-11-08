"""Staked review driver (docs/multi_jurisdiction.md decision 10).

Walks every hosted agent's review batches through ToolBond's epoch rounds
(trustless-contracts ToolBond 448023a; batches and chain calls live in
atn/runtime/review_batch.py). Batches are keyed by the reviewer's address,
and every external agent reviews as the operator (Decision 2026-11-08), so
the household's external sessions share one batch per epoch, committed,
revealed and claimed with the operator key:

1. COMMIT. When the epoch's federated close fires on this daemon
   (``on_local_close``, hooked from the autonet bridge's federated-close
   subscriber), each agent's pending batch for the open epoch
   (``ToolBond.currentEpoch()``) is committed: one commitment plus the
   total ATN stake, once per agent per epoch. The batch (salt included) is
   written to disk BEFORE the commit transaction goes out, so a crash can
   never lose the salt of a landed commit. A pending batch whose epoch was
   anchored before it was committed can no longer be committed; its reviews
   roll into the open epoch's batch.
2. REVEAL. After the epoch's anchor (``revealEnds(epoch) > 0``) every
   committed, unrevealed batch is revealed while the window is open. A
   reveal is relayable: a sealed copy sits in the batch file, which the WS
   ``review_batches`` read lists for the owner to back up.
3. FINALIZE and CLAIM. Once the reveal window has passed, the driver
   finalizes the rounds its agents reviewed (permissionless, in batches) and
   claims each agent's settlement (refund, winnings, REP). A round with a
   settled harm waits out the appeal window first; appealing is a DAO
   proposal (``daoOverrule`` is timelock-only), not something a daemon
   sends.

The driver never forfeits anyone: enforcing another agent's missed reveal is
that agent's keepers' business (anyone may, nothing is gained by it here).

Every step is idempotent: a ``tick`` re-reads the chain, so another daemon
finalizing first, or a restart mid-round, just moves the batch along.
Never raises out of ``tick``.
"""
from __future__ import annotations

import asyncio
import logging
import os
import secrets
from typing import Any, Optional

from .runtime.review_batch import (
    STAGE_COOLING, STAGE_DONE, STAGE_FINALIZING, STAGE_NONE, STAGE_OPEN, STAGES,
    EpochBatch, client_for, commitment_of, store_for,
)

log = logging.getLogger(__name__)

POLL_INTERVAL_S = 60.0
FINALIZE_BATCH = 50
FINALIZE_MAX_CALLS = 20

# ToolBond custom errors a racing keeper can trip. The ABI carries no error
# entries, so a revert surfaces as the raw 4-byte selector; match both.
_BENIGN = ("WrongStage", "TooEarly", "NothingToClaim")


def _selector(sig: str) -> str:
    try:
        from eth_utils import keccak
        return "0x" + keccak(text=sig).hex()[:8]
    except Exception:  # noqa: BLE001
        return "\x00"


def reverted_with(result: dict[str, Any], *names: str) -> bool:
    if result.get("success"):
        return False
    err = str(result.get("error") or "")
    return any(n in err or _selector(f"{n}()") in err for n in names)


def driver_disabled() -> bool:
    return (os.environ.get("ATN_DISABLE_PROVIDER_AUTODETECT", "").strip() == "1"
            or os.environ.get("ATN_DISABLE_REVIEW_DRIVER", "").strip() == "1")


class ReviewDriver:
    """One per daemon. ``tick`` is idempotent; ``run`` loops it until
    cancelled. ``client`` / ``store`` default to the runtime's."""

    def __init__(self, runtime: Any, *, client: Any = None, store: Any = None,
                 interval_s: float = POLL_INTERVAL_S) -> None:
        self._runtime = runtime
        self._client = client
        self._store = store
        self._interval = interval_s
        # Open epochs whose federated close fired here: their pending
        # batches are committed (retried each tick until they land).
        self._close_epochs: set[int] = set()
        self._close_pending = False
        self._wake: Optional[asyncio.Event] = None

    @property
    def client(self) -> Any:
        return self._client if self._client is not None else client_for(self._runtime)

    @property
    def store(self) -> Any:
        return self._store if self._store is not None else store_for(self._runtime)

    # -- triggers ----------------------------------------------------------
    def on_local_close(self) -> None:
        """The epoch's federated close fired here: commit this epoch's
        batches on the next tick. Thread-safe enough for a close thread to
        call (sets a flag; the loop is woken when it runs on one)."""
        self._close_pending = True
        wake = self._wake
        if wake is not None:
            try:
                loop = getattr(self, "_loop", None)
                if loop is not None:
                    loop.call_soon_threadsafe(wake.set)
            except Exception:  # noqa: BLE001
                pass

    async def run(self) -> None:
        self._wake = asyncio.Event()
        self._loop = asyncio.get_running_loop()
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("review driver tick failed")
            try:
                await asyncio.wait_for(self._wake.wait(), self._interval)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()

    # -- agents ------------------------------------------------------------
    def _agent_keys(self) -> dict[str, str]:
        """Registered hosted agents: lowercased 0x -> private key."""
        out: dict[str, str] = {}
        try:
            for defn, _status in self._runtime.list_agents():
                ident = getattr(defn, "identity", None)
                if ident is None or not getattr(ident, "registered_on_chain", False):
                    continue
                addr = str(getattr(ident, "address", "") or "")
                key = self._runtime.registry.get_agent_key(defn.id)
                if addr and key:
                    out[addr.lower()] = key
        except Exception:  # noqa: BLE001
            log.debug("review driver: agent listing failed", exc_info=True)
        return out

    def _any_key(self, keys: dict[str, str]) -> Optional[str]:
        return keys[sorted(keys)[0]] if keys else None

    # -- tick --------------------------------------------------------------
    async def tick(self) -> dict[str, int]:
        """One pass. Returns counts of what it did (for tests and logs)."""
        done = {"committed": 0, "revealed": 0, "finalized": 0, "claimed": 0,
                "rolled": 0, "forfeited": 0}
        client, store = self.client, self.store
        if not client.available():
            return done
        epoch = await client.current_epoch()
        if epoch is None:
            return done
        try:
            # External agents act as the operator: once the owner registers
            # it (from any surface), their held reviews can go out.
            refresh = getattr(getattr(self._runtime, "registry", None),
                              "refresh_operator_registration", None)
            if callable(refresh):
                await refresh()
        except Exception as exc:  # noqa: BLE001
            log.debug("review driver: operator registration not read: %s", exc)
        try:
            from .runtime.review_batch import flush_held_reviews
            held = await flush_held_reviews(self._runtime)
            if held:
                done["held_submitted"] = held
        except Exception as exc:  # noqa: BLE001
            log.debug("review driver: held reviews not flushed: %s", exc)
        now = await client.chain_time()
        keys = self._agent_keys()
        if self._close_pending:
            self._close_pending = False
            self._close_epochs.add(int(epoch))
        self._close_epochs = {e for e in self._close_epochs if e >= epoch}

        for batch in store.all():
            agent = batch.agent.lower()
            try:
                if batch.status in ("settled", "rolled", "forfeited"):
                    continue
                if not batch.commitment:
                    if batch.epoch < epoch and batch.reviews:
                        # Anchored before a commit: nothing can be committed
                        # for that epoch any more, the reviews roll forward.
                        self._roll(batch, epoch)
                        done["rolled"] += 1
                        continue
                    if (batch.epoch == epoch and batch.reviews
                            and epoch in self._close_epochs and agent in keys):
                        if await self._commit(batch, keys[agent]):
                            done["committed"] += 1
                    continue
                if batch.status == "committing" or not batch.committed_tx:
                    if not await self._confirm_commit(batch, epoch):
                        continue
                if not batch.revealed_tx:
                    r = await self._reveal(batch, keys, now)
                    if r == "revealed":
                        done["revealed"] += 1
                    elif r == "forfeited":
                        done["forfeited"] += 1
                    continue
                f, c = await self._settle(batch, keys, now)
                done["finalized"] += f
                done["claimed"] += c
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("review driver: batch %s/%s failed: %s",
                            agent[:10], batch.epoch, exc)
                batch.last_error = str(exc)
                store.save(batch)
        return done

    # -- steps -------------------------------------------------------------
    def _roll(self, batch: EpochBatch, epoch: int) -> None:
        store = self.store
        for entry in batch.reviews:
            store.add(batch.agent, epoch, batch.axis_ids, entry)
        batch.status = "rolled"
        batch.last_error = f"epoch anchored before a commit; reviews moved to epoch {epoch}"
        store.save(batch)

    async def _commit(self, batch: EpochBatch, key: str) -> bool:
        store = self.store
        if not batch.salt:
            batch.salt = secrets.token_hex(32)
        batch.commitment = commitment_of(
            batch.epoch, batch.agent, batch.inputs(), bytes.fromhex(batch.salt)).hex()
        batch.status = "committing"
        store.save(batch)  # the salt is on disk before the tx can land
        res = await self.client.commit(key, batch)
        if res.get("success"):
            batch.status = "committed"
            batch.last_error = ""
            store.save(batch)
            return True
        if reverted_with(res, "AlreadyCommitted"):
            # A commit of ours landed earlier (crash after send): confirm.
            store.save(batch)
            return False
        batch.commitment = ""
        batch.status = ""
        batch.last_error = f"commit: {res.get('error') or 'failed'}"
        if res.get("step"):
            batch.last_error += f" (at {res['step']})"
        store.save(batch)
        return False

    async def _confirm_commit(self, batch: EpochBatch, epoch: int) -> bool:
        """A batch left "committing" (crash, or a commit that may have
        failed): settle what the chain says."""
        on_chain = await self.client.commit_of(batch.epoch, batch.agent)
        if on_chain is None:
            return False
        ours = str(batch.commitment).lower().removeprefix("0x")
        if str(on_chain.get("commitment", "")).lower().removeprefix("0x") == ours:
            batch.committed_tx = batch.committed_tx or "confirmed-on-chain"
            batch.status = "committed"
            self.store.save(batch)
            return True
        if batch.epoch < epoch or int(on_chain.get("commitment", "0") or "0", 16) != 0:
            # The commit never landed and can't any more (or another
            # commitment holds the slot): roll the reviews forward.
            batch.commitment = ""
            self._roll(batch, epoch if batch.epoch < epoch else batch.epoch + 1)
            return False
        batch.commitment = ""
        batch.status = ""
        self.store.save(batch)
        return False

    async def _reveal(self, batch: EpochBatch, keys: dict[str, str],
                      now: Optional[int]) -> str:
        client, store = self.client, self.store
        ends = await client.reveal_ends(batch.epoch)
        if not ends:
            return "waiting"  # epoch not anchored yet
        if now is not None and now > ends:
            on_chain = await client.commit_of(batch.epoch, batch.agent)
            if on_chain is not None and on_chain.get("revealed"):
                batch.revealed_tx = batch.revealed_tx or "relayed"
                batch.status = "revealed"
                store.save(batch)
                return "revealed"
            batch.status = "forfeited"
            batch.last_error = "the reveal window closed before this daemon revealed"
            store.save(batch)
            return "forfeited"
        key = keys.get(batch.agent.lower()) or self._any_key(keys)
        if not key:
            batch.last_error = "no agent key on this daemon to send the reveal"
            store.save(batch)
            return "waiting"
        res = await client.reveal(key, batch)
        if res.get("success"):
            batch.status = "revealed"
            batch.last_error = ""
            store.save(batch)
            return "revealed"
        on_chain = await client.commit_of(batch.epoch, batch.agent)
        if on_chain is not None and on_chain.get("revealed"):
            batch.revealed_tx = "relayed"
            batch.status = "revealed"
            store.save(batch)
            return "revealed"
        batch.last_error = f"reveal: {res.get('error') or 'failed'}"
        store.save(batch)
        return "waiting"

    async def _settle(self, batch: EpochBatch, keys: dict[str, str],
                      now: Optional[int]) -> tuple[int, int]:
        """Finalize every round this batch reviewed once the reveal window
        has passed, then claim this agent's settlement in each."""
        client, store = self.client, self.store
        ends = await client.reveal_ends(batch.epoch)
        if not ends or now is None or now <= ends:
            return 0, 0
        gas_key = keys.get(batch.agent.lower()) or self._any_key(keys)
        own_key = keys.get(batch.agent.lower())
        finalized = claimed = 0
        open_rounds = 0
        for entry in batch.reviews:
            digest = entry.digest
            if batch.rounds.get(digest) in ("claimed", "none"):
                continue
            rnd = await client.round_of(digest, batch.epoch)
            if rnd is None:
                open_rounds += 1
                continue
            stage = int(rnd["stage"])
            if stage == STAGE_NONE:
                # The review did not count (unbonded tool, the author's own,
                # malformed): its stake was refunded at reveal.
                batch.rounds[digest] = "none"
                continue
            for _ in range(FINALIZE_MAX_CALLS):
                if stage == STAGE_DONE or not gas_key:
                    break
                if stage == STAGE_COOLING and not rnd["overruled"] and now <= rnd["appeal_ends"]:
                    break  # inside the appeal window
                if stage not in (STAGE_OPEN, STAGE_COOLING, STAGE_FINALIZING):
                    break
                res = await client.finalize(gas_key, digest, batch.epoch, FINALIZE_BATCH)
                if res.get("success"):
                    finalized += 1
                elif not reverted_with(res, *_BENIGN):
                    batch.last_error = f"finalize {digest[:12]}: {res.get('error')}"
                    break
                nxt = await client.round_of(digest, batch.epoch)
                if nxt is None or int(nxt["stage"]) == stage and not res.get("success"):
                    break
                rnd, stage = nxt, int(nxt["stage"])
            batch.rounds[digest] = STAGES[stage] if stage < len(STAGES) else str(stage)
            if stage != STAGE_DONE:
                open_rounds += 1
                continue
            review = await client.review_of(digest, batch.epoch, batch.agent)
            if review is not None and review.get("claimed"):
                batch.rounds[digest] = "claimed"
                continue
            key = own_key or gas_key
            if not key:
                open_rounds += 1
                continue
            res = await client.claim(key, digest, batch.epoch, batch.agent)
            if res.get("success") or reverted_with(res, "NothingToClaim"):
                batch.rounds[digest] = "claimed"
                claimed += 1 if res.get("success") else 0
            else:
                batch.last_error = f"claim {digest[:12]}: {res.get('error')}"
                open_rounds += 1
        if open_rounds == 0:
            batch.status = "settled"
        store.save(batch)
        return finalized, claimed


def batch_rows(runtime: Any) -> list[dict[str, Any]]:
    """Every stored batch with its status and sealed reveal, newest epoch
    first: what the owner backs up (WS ``review_batches``)."""
    rows = []
    for b in reversed(store_for(runtime).all()):
        rows.append({
            "epoch": b.epoch, "agent": b.agent, "status": b.status or (
                "pending" if not b.commitment else "committed"),
            "reviews": len(b.reviews), "total_stake": str(b.total_stake),
            "commitment": b.commitment, "committed_tx": b.committed_tx,
            "revealed_tx": b.revealed_tx, "rounds": dict(b.rounds),
            "last_error": b.last_error,
            "sealed_reveal": b.sealed_reveal() if b.commitment else None,
        })
    return rows


def start_driver(runtime: Any) -> Optional[tuple["ReviewDriver", asyncio.Task]]:
    """Started with the autonet bridge; None under the test gate."""
    if driver_disabled():
        return None
    driver = ReviewDriver(runtime)
    try:
        setattr(runtime, "review_driver", driver)
    except Exception:  # noqa: BLE001
        pass
    return driver, asyncio.create_task(driver.run())


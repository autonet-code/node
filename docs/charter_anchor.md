# Charter governance (the CharterAnchor)

**Status: live (beta on testnet).** Contract + daemon-side hash + drift
detection are in the tree, and `CharterAnchor` is **deployed on the Autonet
jurisdiction (Etherlink Shadownet) with charter v1 anchored on-chain**
(anchored 2026-07-10; the anchored hash matches the local charter). Its
governor is the **DAO timelock**, so a new charter version now takes a
governance proposal. The deployed address is in `registry.json`
(`charter_anchor`), fetched by daemons from GitHub raw master — not
hardcoded here. Follow-the-anchor migration machinery is deliberately
deferred (detection first).

## What the charter is

The charter is the 6-root alignment vocabulary every substrate debate runs
around: four alignment axes (`life_precious`, `self_preservation`,
`promotion_of_intelligence`, `evolution`) plus two usefulness axes
(`correctness`, `simplicity`). It lives off-chain as a hardcoded Python constant,
`CHARTER` in `nodes/common/world_model_substrate/adapter.py`, materialized into
every daemon's world by `build_charter_world()`. It is the coordinate basis every
historical claim is embedded in.

Until now there was **no on-chain anchor** for it — the charter was changed only
by editing the file and having every daemon re-pull the code. The whole point of
a jurisdiction is that governance can change these core values, so the charter
needs a governed anchor.

## Where the jurisdiction's governor changes core values

`contracts/core/CharterAnchor.sol` is the anchor. It does **not** store the
charter (the values stay off-chain); it anchors **which version is canonical** by
committing the sha256 of the canonical charter blob plus a monotonic version
counter.

It is the **deliberate opposite of Substrate.sol**. Substrate.sol is
intentionally ungoverned — no admin keys, no owner, no privileged setter — because
its purity (nobody can rewrite training history) is a feature. CharterAnchor has
exactly one privileged actor, the **governor**, and one privileged act,
`anchorCharter(charterHash, uri)`. The governor is set at construction and is
immutable (governance handoff is by naming a timelock at deploy). On the live
Autonet jurisdiction the governor is the **DAO timelock**, so anchoring a new
charter version runs through a governance proposal.

Surface:

- `anchorCharter(bytes32 charterHash, string uri)` — governor-only, append-only.
  Version `n+1` chains to version `n` via `prevHash` (version 1 chains to
  `bytes32(0)`), mirroring the epoch hash-chain on Substrate.sol. Emits
  `CharterAnchored(version, hash, uri, prevHash)`.
- `currentCharter()` — latest `(version, hash, uri, prevHash, timestamp)`; reverts
  `NoCharter` before anything is anchored.
- `charterAt(version)` — a specific version; reverts `BadVersion` out of range.
- `versionCount()` — number of versions anchored (== current version).

## Forward-only

A charter change is a **forward-only fork boundary**. The charter is the
coordinate basis every past claim was embedded in — change the roots and every
historical node's alignment coordinates change meaning. You cannot re-equilibrate
history against a new basis and keep bit-identical replay (replay / float-order is
consensus-relevant). So charter history is append-only: version `N+1` takes effect
at an epoch boundary, old epochs stay anchored under version `N`, the counter only
increases, and there is no re-scoring of the past. This matches the project's "no
rollback, move forward through governance" principle.

## Daemon side: detection first

The daemon computes its local charter hash from the **same** constant
`build_charter_world` uses (it imports `adapter.CHARTER`, never duplicates it):

- `nodes/common/world_model_substrate/charter_version.py`:
  - `charter_payload()` — the charter roots as a schema-tagged, axis-ordered dict.
  - `charter_bytes()` — canonical JSON (`sort_keys=True`, fixed separators), so
    the bytes are a pure function of the values (dict key order irrelevant).
  - `charter_hash()` — sha256 hex of those bytes.
- `build_charter_world()` logs the active `charter_hash` when it builds a world.
- `federated_epoch_close`'s authoritative payload carries `charter_hash` — the
  close is stamped with the charter it ran against.
- `atn/on_chain.py: verify_charter_against_anchor(w3, anchor_address)` reads
  `currentCharter()` and diffs its hash against the local `charter_hash()`. On
  mismatch the daemon logs a **LOUD** warning (`CHARTER DIVERGENCE …`): a divergent
  daemon runs a different charter than the governor anchored and its closes will
  not be bit-identical to the canonical charter.

Config: `autonet.charter_anchor_address` (optional). When set, the daemon can
verify against the deployed anchor.

**Follow-the-anchor migration is future work.** Today the daemon *detects*
divergence; automatically pulling the anchored charter version and re-basing at the
epoch boundary is deliberately deferred. Detection first.

## Current charter hash

The hash of the current (v-in-code) 6-root charter is:

```
5756ed3aa1831533c6ae7a1728cd6af73241c787049080ac3469d5b40f841cd5
```

This is the hash anchored on-chain as **version 1** on the live Autonet
jurisdiction; the daemon's `verify_charter_against_anchor` matches its local
`charter_hash()` against it.

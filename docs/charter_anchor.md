# Charter version (DAO Registry keys)

> **Superseded by `multi_jurisdiction.md` decision 5 (2026-10-17).** The
> Registry's `jurisdiction.<id>.charter` now holds the charter itself (JSON
> axes), read pinned to the previous anchor (`voice_state.charter_text_from_logs`)
> and installed at join; a legacy `0x` hash value is ignored, and
> `charter_uri` is legacy. Below is the 2026-10-16 hash-and-URI design, kept
> as history.

**Status (historical): decided 2026-10-16 with the DAO side.** The standalone
`CharterAnchor.sol` is retired. The jurisdiction's charter version now lives in
two keys of the jurisdiction's DAO Registry:

- `jurisdiction.<id>.charter`: `0x` + 64 lowercase hex, the sha256 of the
  canonical charter blob.
- `jurisdiction.<id>.charter_uri`: where the charter blob can be fetched.

Both are written at install and changed only by DAO proposal (the Registry
setter is governed), so a new charter version still takes a governance vote.
Daemons seed them from the registry contracts keys `charter` / `charter_uri`
into config fields `charter_hash` / `charter_uri`.

## What the charter is

The 6-root alignment vocabulary: four alignment axes (`life_precious`,
`self_preservation`, `promotion_of_intelligence`, `evolution`) plus two
usefulness axes (`correctness`, `simplicity`). It lives off-chain as the
`CHARTER` constant in `nodes/common/world_model_substrate/adapter.py`,
materialized into every daemon's world by `build_charter_world()`. It is the
coordinate basis every historical claim is embedded in. The registry keys
anchor only WHICH version is canonical, not its content.

## Forward-only

A charter change is a forward-only fork boundary: changing the roots changes
the meaning of every historical alignment coordinate, and bit-identical replay
cannot survive re-basing history. A new version takes effect at an epoch
boundary, old epochs stay under the old version, and there is no rollback.

## Daemon side: detection first

- `nodes/common/world_model_substrate/charter_version.py`: `charter_payload()`,
  `charter_bytes()` (canonical JSON), `charter_hash()` (sha256 hex).
- `federated_epoch_close`'s payload carries `charter_hash`.
- `atn/on_chain.py: verify_charter(expected_hash, uri="")` is pure (no chain
  read): it compares the registry value to the local `charter_hash()` and
  returns `{match, local_hash, chain_hash, uri}` (`match` is None when no
  charter is published). On mismatch it logs a loud `CHARTER DIVERGENCE`
  warning.

Following the governed charter automatically (pulling the new version and
re-basing at the epoch boundary) is future work. Detection first.

## Current charter hash

```
5756ed3aa1831533c6ae7a1728cd6af73241c787049080ac3469d5b40f841cd5
```

The registry value is this with a `0x` prefix.

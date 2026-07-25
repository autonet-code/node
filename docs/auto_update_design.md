# Daemon Auto-Update: Design

Status: implementing 2026-06-17. Mode chosen by user: **stage on poll, apply on next boot**
(the running daemon never self-restarts).

## Why this exists / trust model

The daemon ships via PyPI (`autonet-computer`); the user pushes frequent bugfix
releases and there's no way to "force a deployment" the way a web app can:
every node runs its own process. Auto-update closes that gap.

This is justified, not sneaky, *only* because control of the official codebase
is meant to be decentralized via reputation/governance. Today that governance is
**aspirational, not wired** (reputation is on-chain but gates nothing; the
`node.code.hash.<version>` Registry key is read by `atn/_cache.py` but no contract
writes it yet). So the V1 trust anchor is:

1. **Version pointer**: PyPI (the release authority today).
2. **Integrity**: the published wheel SHA-256, plus an *advisory* check against the
   on-chain core-hash (`atn/_cache.py:validate`) when the Registry key is present.

The code is structured so the **version pointer can later move** from PyPI to a
governance-approved on-chain release pointer without touching the stage/apply
machinery: that's the seam where decentralization lands.

## Flow

```
background poll (daily, gated by auto_update.enabled)
  → query PyPI JSON API for latest autonet-computer version
  → newer than running? (PEP 440 compare)
  → download the wheel + its sha256 (from PyPI release metadata)
  → verify wheel sha256
  → advisory: validate() core-hash against on-chain Registry (log, don't block on absent)
  → stage to ~/.atn/staged_update/<version>/<wheel> + write pending.json marker
  → emit UPDATE_STATUS event (staged) ; never touches running process

next daemon boot (atn/cli.py:main(), BEFORE asyncio.run / heavy imports)
  → read ~/.atn/staged_update/pending.json
  → staged version newer than running AND not already attempted this boot?
  → pip install the staged wheel into the current environment
  → on success: clear marker, set ATN_UPDATE_APPLIED=1, os.execv re-exec once
  → on failure: log, leave running version, clear marker (no loop)
```

## Components

- `nodes/common/updater.py` (EXTEND existing `AutonetUpdater`):
  - add `UpdateSource.PYPI = "pypi"` + `_check_pypi()` (PyPI JSON API).
  - add `stage_update(info)`: download wheel → verify sha256 → advisory on-chain
    verify → write to staged dir + `pending.json`. Replaces the `.update_package`
    stub for the stage path.
  - add `get_status()` → dict {state, current_version, available_version, staged_version, last_check, last_error}.
  - PEP 440 version compare (use `packaging.version` if available, else tuple split fallback).
  - staged dir = `~/.atn/staged_update/`. `pending.json` = {version, wheel_path, sha256, staged_at, source}.
- `nodes/common/config.py` `UpdateConfig`: keep existing fields; the daemon-side
  `atn/config.py` gets a new `AutoUpdateConfig` that the daemon actually reads
  (enabled flag, check_interval_secs, source, pypi index url override). Bridge the
  two: the AutonetUpdater is constructed from the daemon's AutoUpdateConfig values.
- `atn/config.py`: new `AutoUpdateConfig` dataclass + field on `ATNConfig` + parse
  in `load_config()`. Fields:
    enabled: bool = False
    check_interval_secs: int = 86400
    source: str = "pypi"          # pypi | git | http | blob_store
    pypi_index_url: str = ""      # "" → pypi.org default
    package_name: str = "autonet-computer"
- `atn/update_boot.py` (NEW, deliberately import-light: only os/sys/json/subprocess/pathlib):
  `apply_staged_update_if_any()`, the boot-time pre-init step. No atn.* imports.
  Returns None; re-execs on success.
- `atn/cli.py`:
  - `main()`: call `apply_staged_update_if_any()` as the first line.
  - `run_cli()`: after `runtime.start()`, if `config.auto_update.enabled`, spawn a
    background poll task; cancel it in the finally block.
- `atn/events.py`: add `UPDATE_STATUS = "update.status"`.
- `atn/ws_server.py`: add `update_status` request handler.
- `atn/runtime/snapshot.py`: add `"update"` section (status dict) to the snapshot.

## Re-exec loop guard

`ATN_UPDATE_APPLIED=1` env var, set right before the boot-time re-exec. If already
set on entry to `apply_staged_update_if_any()`, skip (we've applied once this
process lineage). Also: the boot step clears `pending.json` whether install
succeeds OR fails: a wheel that won't install must not be retried every boot.

## Verification (advisory today, blocking-ready)

`stage_update` calls `atn/_cache.py:validate(rpc_url, registry_addr, version)`:
- returns True (proceed) when the key is absent / chain unreachable, matching
  existing fail-open semantics, BUT we log it at WARNING so "unverified" is visible.
- returns False only on a confirmed mismatch → refuse to stage (do not download-and-run
  code that fails its on-chain fingerprint).
rpc_url + registry_address come from the autonet config / jurisdiction discovery.

## Safety invariants

- Default OFF (`enabled: False`). Opt-in only.
- Never interrupts a running process (stage-only at runtime).
- Boot apply is idempotent and loop-guarded.
- A corrupt/failed wheel clears the marker and the daemon runs the version it has.
- Wheel sha256 mismatch → refuse to stage.
- On-chain core-hash mismatch (when published) → refuse to stage.

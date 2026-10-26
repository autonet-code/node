# Vendored: world-model

This directory is a vendored copy of
[autonet-code/world-model](https://github.com/autonet-code/world-model)
(branch: master).

## Why vendored

PyPI rejects direct git URL dependencies in `pyproject.toml`. Vendoring
the source under `world_model/` lets `pip install autonet-computer`
work without external git access.

## Updating

Sync from the upstream repo when the substrate engine changes there:

```bash
cd C:/code/autonet
rm -rf world_model
cp -r C:/code/world-model/world_model world_model
find world_model -name __pycache__ -type d -exec rm -rf {} +
git add world_model
git commit -m "vendor: sync world-model from upstream"
```

Long-term plan: publish `world-model` to PyPI and reference it as a
normal dependency, then remove this vendored copy.

## Local patches (not yet upstream)

- `generalized/detmath.py` (2026-10-26): platform-independent `exp`,
  `log`, `log1p` built from IEEE basic operations only. The kernels in
  `coordinate_frame.py`, `equilibrate.py` and `lindblad.py` call it
  instead of `math.exp` / `math.log`, so the canonical world checkpoint
  (`world_cid`) hashes the same on every C library. Port this upstream
  before the next sync, or the sync reverts it.

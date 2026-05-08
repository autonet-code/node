"""
Analysis tools for the world model.

These modules inspect equilibria produced by the arena and quantify
properties relevant to the engine's architectural claims (sparsity,
fractality, compressibility, etc.). They are read-only over the
core data structures.

Note: this package intentionally does not re-export from submodules,
so importing ``world_model.analysis`` does not trigger imports of
sibling subpackages that may have unrelated breakages. Import
submodules directly:

    from world_model.analysis.sparsity import compute_sparsity_metrics
"""

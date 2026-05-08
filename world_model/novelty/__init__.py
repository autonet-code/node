"""
Novelty - Measuring deviation from reference frames.

Novelty is not a property of concepts in isolation. It is the result of
a loop that measures how a concept relates to an agent's existing beliefs.
The termination reason IS the novelty measurement.

Key exports for integration with attention:

    from world_model.novelty import (
        # Core types
        NoveltyResult,
        Termination,
        Stance,
        ReferenceFrame,

        # Probes (the measurement loop)
        HybridProbe,
        NeuralProbe,

        # Frames (what you measure against)
        HybridFrame,
        NeuralFrame,

        # Convenience functions
        measure_against_claims,
        measure_hybrid_novelty,
    )

Quick usage:

    from world_model.novelty import measure_against_claims

    result = measure_against_claims(
        concept="Bitcoin",
        claim_texts=[
            "Traditional banking provides security",
            "Trust in institutions is necessary",
        ]
    )

    print(result.termination)  # Termination.CONTRADICTS_ROOT
    print(result.composite)    # 0.052 (novelty score 0-1)
"""

# Core types
from .core import (
    NoveltyProbe,
    ReferenceFrame,
    Focus,
    ParseResult,
    Termination,
    Stance,
    Claim,
    NoveltyResult,
)

# Hybrid probe (Wikidata graph + Neural NLI)
from .hybrid_probe import (
    HybridProbe,
    HybridFrame,
    HybridClaim,
    HybridFetchResult,
    measure_hybrid_novelty,
    measure_against_claims,
)

# Neural probe (pure NLI, no Wikidata)
from .neural_probe import (
    NeuralProbe,
    NeuralFrame,
    NeuralClaim,
    measure_neural_novelty,
)

# Wikidata probe (graph structure only)
from .wikidata_probe import (
    WikidataProbe,
    WikidataFrame,
    WikidataClaim,
    measure_novelty as measure_wikidata_novelty,
)

# Embeddings utilities (for direct use)
from .embeddings import (
    semantic_similarity,
    cached_similarity,
    nli_inference,
    NLIResult,
)

__all__ = [
    # Core
    "NoveltyProbe",
    "ReferenceFrame",
    "Focus",
    "ParseResult",
    "Termination",
    "Stance",
    "Claim",
    "NoveltyResult",

    # Hybrid (recommended)
    "HybridProbe",
    "HybridFrame",
    "HybridClaim",
    "measure_hybrid_novelty",
    "measure_against_claims",

    # Neural
    "NeuralProbe",
    "NeuralFrame",
    "NeuralClaim",
    "measure_neural_novelty",

    # Wikidata
    "WikidataProbe",
    "WikidataFrame",
    "WikidataClaim",
    "measure_wikidata_novelty",

    # Utilities
    "semantic_similarity",
    "cached_similarity",
    "nli_inference",
    "NLIResult",
]

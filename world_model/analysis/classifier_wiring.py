"""
Shared wiring helpers for engine-as-classifier demos.

The same wiring strategy is used across iris, digits, letters, and
future classification demos. Centralizing it here means an architectural
finding (e.g., "contrast wiring beats absolute wiring") changes one
file and benefits every demo, rather than getting tuned per dataset.

Strategy: contrast wiring via z-scores

For each feature, the edge weight between a class tendency and its
``_high``/``_low`` evidence tendencies depends on how far that class's
centroid sits *above or below the population mean*, not its absolute
position in the value range. A class whose feature 5 is one standard
deviation above the population mean gets a strong feature_5_high
edge regardless of whether the absolute value is 1.5 or 1500.

This handles the case where many classes occupy similar absolute
regions of feature space but differ in relative position. Letters
in particular has 26 classes packed into a 16-dim integer space,
where absolute-value wiring collapses to whichever class has the
"lowest everything" centroid.

For datasets that already separate cleanly in absolute terms
(iris setosa is far from the others on multiple features), contrast
wiring still works -- it just produces near-equivalent edge weights
to absolute wiring in those cases.

Public API:

  fit_classifier_profile(X_train, y_train, classes) -> ClassifierProfile
  build_classifier_state(profile, class_prefix, feature_prefix) -> PresentState
  classify_case(case_features, profile, base_state, ...) -> winning_class

The class_prefix / feature_prefix arguments let domains namespace
their tendencies: iris uses species_setosa / petal_length_high,
digits uses digit_3 / px05_high, letters uses letter_K / f12_high.
The wiring math is identical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ..models.tendency import Tendency, TendencySet
from ..models.factory import DefaultTendencyFactory, TendencySpec
from ..models.lineage import Lineage, StakeWeightGraph
from ..dynamics.reseed import PresentState, Substitution, reseed_and_equilibrate


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------


@dataclass
class ClassifierProfile:
    """Per-class statistics in contrast (z-score) form."""

    classes: list                                       # class labels (any hashable)
    n_features: int
    class_centroids: dict[object, list[float]]
    class_zscores: dict[object, list[float]]
    feature_pop_mean: list[float]
    feature_pop_std: list[float]


def fit_classifier_profile(X_train, y_train, classes) -> ClassifierProfile:
    """Compute centroids, population stats, and per-class z-scores.

    X_train is array-like of shape (n_cases, n_features), numeric.
    y_train is a 1-D array-like of class labels (any hashable type).
    classes is the ordered list of class labels to model.
    """
    n_features = X_train.shape[1]
    n_cases = X_train.shape[0]

    # Class centroids
    centroids: dict[object, list[float]] = {}
    for c in classes:
        rows = [X_train[i] for i in range(len(y_train)) if y_train[i] == c]
        if not rows:
            centroids[c] = [0.0] * n_features
            continue
        centroids[c] = [
            sum(float(r[f]) for r in rows) / len(rows)
            for f in range(n_features)
        ]

    # Population mean and std per feature (across cases, not class means)
    feature_pop_mean: list[float] = []
    feature_pop_std: list[float] = []
    for f in range(n_features):
        vals = [float(X_train[i, f]) for i in range(n_cases)]
        m = sum(vals) / n_cases
        var = sum((v - m) ** 2 for v in vals) / n_cases
        feature_pop_mean.append(m)
        feature_pop_std.append(max(var ** 0.5, 1e-9))

    # Per-class z-score per feature
    class_zscores: dict[object, list[float]] = {}
    for c in classes:
        class_zscores[c] = [
            (centroids[c][f] - feature_pop_mean[f]) / feature_pop_std[f]
            for f in range(n_features)
        ]

    return ClassifierProfile(
        classes=list(classes),
        n_features=n_features,
        class_centroids=centroids,
        class_zscores=class_zscores,
        feature_pop_mean=feature_pop_mean,
        feature_pop_std=feature_pop_std,
    )


# ---------------------------------------------------------------------------
# Wiring math (the architectural choice)
# ---------------------------------------------------------------------------


def contrast_to_edge_weight(z: float) -> tuple[float, float]:
    """Map a class's z-score on a feature to (low_edge, high_edge) weights.

    z-score is squashed via tanh(z/2) so values in roughly [-3, 3] map
    to roughly [-1, 1]. Edge weights stay in [0.05, 0.95] to keep the
    graph propagation well-conditioned (no edge is 0 or 1 exactly).
    """
    signed = math.tanh(z / 2.0)
    high = 0.5 + 0.5 * signed
    low = 0.5 - 0.5 * signed
    return 0.05 + 0.9 * low, 0.05 + 0.9 * high


def case_feature_strength(value: float, pop_mean: float, pop_std: float) -> tuple[float, float]:
    """Map a case's value on a feature to (low_strength, high_strength).

    Same z-score math as the wiring. A case value above the population
    mean produces high evidence strength, below produces low strength,
    near the mean produces neutral. This pairs naturally with the
    edges -- a case whose feature is unusually high pulls strongly on
    classes whose centroids are also unusually high on that feature.
    """
    z = (value - pop_mean) / max(pop_std, 1e-9)
    signed = math.tanh(z / 2.0)
    high = 0.5 + 0.5 * signed
    low = 0.5 - 0.5 * signed
    return low, high


# ---------------------------------------------------------------------------
# State construction and classification
# ---------------------------------------------------------------------------


def build_classifier_state(
    profile: ClassifierProfile,
    class_prefix: str = "class_",
    feature_prefix: str = "f",
) -> PresentState:
    """Construct a starting state with class + feature-evidence tendencies."""
    factory = DefaultTendencyFactory()
    n_features = profile.n_features
    n_classes = len(profile.classes)

    class_alloc = 0.5 / n_classes
    feature_alloc = 0.5 / (2 * n_features)

    specs = []
    for c in profile.classes:
        specs.append(TendencySpec(
            id=f"{class_prefix}{c}",
            initial_allocation=class_alloc,
        ))
    for f in range(n_features):
        specs.append(TendencySpec(
            id=f"{feature_prefix}{f:02d}_low",
            initial_allocation=feature_alloc,
        ))
        specs.append(TendencySpec(
            id=f"{feature_prefix}{f:02d}_high",
            initial_allocation=feature_alloc,
        ))

    tendencies = factory.build_set(specs)

    graph = StakeWeightGraph()
    for c in profile.classes:
        z_per_feature = profile.class_zscores[c]
        for f in range(n_features):
            low_edge, high_edge = contrast_to_edge_weight(z_per_feature[f])
            graph.add_edge(f"{class_prefix}{c}", f"{feature_prefix}{f:02d}_high", high_edge)
            graph.add_edge(f"{class_prefix}{c}", f"{feature_prefix}{f:02d}_low",  low_edge)

    lineages = {tid: Lineage() for tid in tendencies.ids()}
    return PresentState(tendencies=tendencies, lineages=lineages, graph=graph)


def classify_case(
    case_features,
    profile: ClassifierProfile,
    base_state: PresentState,
    class_prefix: str = "class_",
    feature_prefix: str = "f",
    learning_rate: float = 0.3,
    tolerance: float = 1e-5,
    max_iterations: int = 300,
    sharpen_strength: float = 0.0,
    surprise_weighting: bool = True,
):
    """Substitute feature-evidence tendencies, equilibrate, return winning class.

    surprise_weighting: when True, each feature's substitution budget
    is scaled by |z| -- how anomalous the case's value is compared to
    the population. A typical feature (z=0) contributes very little
    to the substitution; an unusual feature (|z|=2+) dominates.

    This is the engine's original novelty/surprise primitive applied
    to classification: surprising signals capture attention. Classes
    whose centroids share the case's anomalies win because the
    anomalous feature-evidence tendencies pull on them most strongly.

    sharpen_strength: optional softmax sharpening among class tendencies
    after each calibration iteration. Adds competitive dynamics on top
    of the smoothing relaxation.
    """
    import math
    n_features = profile.n_features

    # Per-feature surprise (|z|) and the resulting budget weighting
    if surprise_weighting:
        feature_surprises = []
        for f in range(n_features):
            z = (float(case_features[f]) - profile.feature_pop_mean[f]) / max(
                profile.feature_pop_std[f], 1e-9
            )
            feature_surprises.append(abs(z))
        # Add a small floor so every feature contributes nonzero, but
        # surprising ones dominate.
        floor = 0.1
        weights = [floor + s for s in feature_surprises]
        total = sum(weights)
        if total > 0:
            feature_budgets = [0.5 * (w / total) for w in weights]
        else:
            feature_budgets = [0.5 / n_features] * n_features
    else:
        feature_budgets = [0.5 / n_features] * n_features

    substitutions = []
    for f in range(n_features):
        low_strength, high_strength = case_feature_strength(
            float(case_features[f]),
            profile.feature_pop_mean[f],
            profile.feature_pop_std[f],
        )
        budget_f = feature_budgets[f]
        # The remaining 0.5 of the total budget would normally go to
        # *both* low and high evenly, but since strengths are
        # complementary (low + high ≈ 1 when |z| is large, both ≈ 0.5
        # when z ≈ 0), we just assign each its strength share of the
        # feature's budget. Total mass for each feature pair is budget_f.
        substitutions.append(Substitution(
            id=f"{feature_prefix}{f:02d}_high",
            new_tendency=Tendency(
                id=f"{feature_prefix}{f:02d}_high",
                allocation=budget_f * high_strength,
            ),
        ))
        substitutions.append(Substitution(
            id=f"{feature_prefix}{f:02d}_low",
            new_tendency=Tendency(
                id=f"{feature_prefix}{f:02d}_low",
                allocation=budget_f * low_strength,
            ),
        ))

    sharpen_ids = (
        {f"{class_prefix}{c}" for c in profile.classes}
        if sharpen_strength > 0 else None
    )

    result = reseed_and_equilibrate(
        base_state,
        substitutions=substitutions,
        propagate_via_graph=True,
        learning_rate=learning_rate,
        tolerance=tolerance,
        max_iterations=max_iterations,
        sharpen_among_ids=sharpen_ids,
        sharpen_strength=sharpen_strength,
    )

    class_allocs = {
        c: result.state.tendencies.get(f"{class_prefix}{c}").allocation
        for c in profile.classes
    }
    return max(class_allocs, key=class_allocs.get)

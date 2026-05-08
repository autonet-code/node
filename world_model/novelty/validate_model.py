"""
Validation: Can our novelty model predict Wikidata graph properties?

The Test:
---------
If our theoretical framework captures real structure in knowledge graphs,
then our novelty components should PREDICT observable Wikidata properties.

Hypotheses:
-----------
1. integration_resistance should predict:
   - FEWER incoming references (hard to link to)
   - LOWER sitelinks (less global integration)
   - r < 0 (negative correlation)

2. coverage_gap should predict:
   - LOWER sitelinks (less global coverage)
   - FEWER properties (less semantic richness)
   - r < 0 (negative correlation)

3. disruption_potential should predict:
   - LOWER centrality ratio (not yet established)
   - LOWER incoming/outgoing ratio
   - r < 0 (negative correlation)

4. depth_factor should predict:
   - HIGHER actual hierarchy depth
   - MORE ancestors in P279 chain
   - r > 0 (positive correlation)

If these correlations hold, our model captures something real.
If not, our theory needs revision.
"""

import math
import statistics
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import json

from wikidata import (
    WikidataNoveltyInputs,
    best_match,
    GraphMetrics,
)


@dataclass
class ValidationResult:
    """Result of validating model against Wikidata."""
    concept: str
    qid: str

    # Our model's predictions (novelty components)
    integration_resistance: float
    depth_factor: float
    coverage_gap: float
    disruption_potential: float

    # Wikidata ground truth (what we're predicting)
    incoming_refs: int
    sitelinks: int
    properties: int
    centrality_ratio: float
    hierarchy_depth: int
    subclass_count: int


def gather_validation_data(concepts: List[str]) -> Tuple[List[ValidationResult], List[str]]:
    """
    Gather validation data for a list of concepts.

    Returns:
        (results, errors) - list of ValidationResults and list of error messages
    """
    results = []
    errors = []

    for i, concept in enumerate(concepts):
        print(f"[{i+1}/{len(concepts)}] {concept}...", end=" ", flush=True)

        try:
            inputs = WikidataNoveltyInputs.from_text(concept)
            if inputs is None:
                print("NOT FOUND")
                errors.append(f"{concept}: Not found in Wikidata")
                continue

            result = ValidationResult(
                concept=concept,
                qid=inputs.qid,
                integration_resistance=inputs.integration_resistance,
                depth_factor=inputs.depth_factor,
                coverage_gap=inputs.coverage_gap,
                disruption_potential=inputs.disruption_potential,
                incoming_refs=inputs.metrics.incoming_refs,
                sitelinks=inputs.metrics.sitelinks,
                properties=inputs.metrics.properties,
                centrality_ratio=inputs.metrics.centrality_ratio,
                hierarchy_depth=inputs.metrics.depth,
                subclass_count=inputs.metrics.subclass_count,
            )
            results.append(result)
            print(f"OK ({inputs.qid})")

        except Exception as e:
            print(f"ERROR: {e}")
            errors.append(f"{concept}: {e}")

    return results, errors


def pearson_correlation(xs: List[float], ys: List[float]) -> float:
    """Compute Pearson correlation coefficient."""
    n = len(xs)
    if n < 2:
        return 0.0

    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)

    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)

    denominator = math.sqrt(var_x * var_y)
    return numerator / denominator if denominator > 0 else 0.0


def analyze_predictions(results: List[ValidationResult]) -> Dict[str, Dict]:
    """
    Analyze how well our model predicts Wikidata properties.

    Returns dict of prediction analyses.
    """
    if len(results) < 3:
        return {'error': 'Not enough data points'}

    # Extract arrays
    ir = [r.integration_resistance for r in results]
    df = [r.depth_factor for r in results]
    cg = [r.coverage_gap for r in results]
    dp = [r.disruption_potential for r in results]

    # Log-transform counts for better correlation (avoid zero issues)
    incoming_log = [math.log(r.incoming_refs + 1) for r in results]
    sitelinks_log = [math.log(r.sitelinks + 1) for r in results]
    properties_log = [math.log(r.properties + 1) for r in results]
    centrality_log = [math.log(r.centrality_ratio + 0.01) for r in results]
    depth = [float(r.hierarchy_depth) for r in results]
    subclasses = [float(r.subclass_count) for r in results]

    # Compute correlations for each hypothesis
    analyses = {
        'integration_resistance': {
            'hypothesis': 'Predicts FEWER incoming refs (r < 0)',
            'correlations': {
                'incoming_refs': pearson_correlation(ir, incoming_log),
                'sitelinks': pearson_correlation(ir, sitelinks_log),
            },
            'expected_sign': 'negative',
        },
        'coverage_gap': {
            'hypothesis': 'Predicts LOWER coverage (r < 0)',
            'correlations': {
                'sitelinks': pearson_correlation(cg, sitelinks_log),
                'properties': pearson_correlation(cg, properties_log),
            },
            'expected_sign': 'negative',
        },
        'disruption_potential': {
            'hypothesis': 'Predicts LOWER establishment (r < 0)',
            'correlations': {
                'centrality_ratio': pearson_correlation(dp, centrality_log),
                'incoming_refs': pearson_correlation(dp, incoming_log),
            },
            'expected_sign': 'negative',
        },
        'depth_factor': {
            'hypothesis': 'Predicts HIGHER actual depth (r > 0)',
            'correlations': {
                'hierarchy_depth': pearson_correlation(df, depth),
                'subclass_count': pearson_correlation(df, subclasses),
            },
            'expected_sign': 'positive',
        },
    }

    # Evaluate each hypothesis
    for name, analysis in analyses.items():
        expected = analysis['expected_sign']
        correct = 0
        total = 0

        for metric, r in analysis['correlations'].items():
            total += 1
            if expected == 'negative' and r < 0:
                correct += 1
            elif expected == 'positive' and r > 0:
                correct += 1

        analysis['hypotheses_supported'] = correct
        analysis['hypotheses_total'] = total
        analysis['validation'] = 'SUPPORTED' if correct == total else 'PARTIAL' if correct > 0 else 'REJECTED'

    return analyses


def print_validation_report(results: List[ValidationResult], analyses: Dict, errors: List[str]):
    """Print a formatted validation report."""
    print("\n" + "=" * 70)
    print("MODEL VALIDATION REPORT: Can Novelty Predict Wikidata?")
    print("=" * 70)

    print(f"\nData: {len(results)} concepts successfully analyzed")
    print(f"Errors: {len(errors)}")

    print("\n" + "-" * 70)
    print("HYPOTHESIS TESTING")
    print("-" * 70)

    supported = 0
    total = 0

    for component, analysis in analyses.items():
        print(f"\n{component.upper()}")
        print(f"  Hypothesis: {analysis['hypothesis']}")
        print(f"  Correlations:")

        for metric, r in analysis['correlations'].items():
            direction = "+" if r > 0 else "-" if r < 0 else "0"
            strength = "strong" if abs(r) > 0.5 else "moderate" if abs(r) > 0.3 else "weak"
            print(f"    vs {metric}: r = {r:+.3f} ({direction}, {strength})")

        status = analysis['validation']
        if status == 'SUPPORTED':
            supported += 1
            print(f"  Result: [PASS] {status}")
        elif status == 'PARTIAL':
            supported += 0.5
            print(f"  Result: [PARTIAL] {status}")
        else:
            print(f"  Result: [FAIL] {status}")
        total += 1

    print("\n" + "-" * 70)
    print("OVERALL VALIDATION")
    print("-" * 70)

    score = supported / total if total > 0 else 0
    print(f"\nHypotheses supported: {supported}/{total} ({score:.0%})")

    if score >= 0.75:
        print("\n--> MODEL VALIDATED: Strong predictive power over Wikidata structure")
    elif score >= 0.5:
        print("\n--> MODEL PARTIALLY VALIDATED: Some predictive power, needs refinement")
    else:
        print("\n--> MODEL NEEDS REVISION: Poor predictive power")

    # Show a few example predictions
    print("\n" + "-" * 70)
    print("SAMPLE DATA (showing model vs ground truth)")
    print("-" * 70)

    # Sort by composite novelty for interesting spread
    sorted_results = sorted(results, key=lambda r: (
        r.integration_resistance + r.coverage_gap + r.disruption_potential
    ) / 3)

    # Show low, medium, high novelty examples
    samples = []
    if len(sorted_results) >= 3:
        samples = [
            sorted_results[0],  # lowest novelty
            sorted_results[len(sorted_results) // 2],  # median
            sorted_results[-1],  # highest novelty
        ]
    else:
        samples = sorted_results

    for r in samples:
        composite = (r.integration_resistance + r.coverage_gap + r.disruption_potential + r.depth_factor) / 4
        print(f"\n{r.concept} ({r.qid}) - Composite novelty: {composite:.2f}")
        print(f"  Model predictions:")
        print(f"    integration_resistance: {r.integration_resistance:.2f}")
        print(f"    coverage_gap: {r.coverage_gap:.2f}")
        print(f"    disruption_potential: {r.disruption_potential:.2f}")
        print(f"    depth_factor: {r.depth_factor:.2f}")
        print(f"  Wikidata ground truth:")
        print(f"    incoming_refs: {r.incoming_refs:,}")
        print(f"    sitelinks: {r.sitelinks}")
        print(f"    centrality_ratio: {r.centrality_ratio:.2f}")
        print(f"    hierarchy_depth: {r.hierarchy_depth}")


# Test concepts covering a range of novelty levels
TEST_CONCEPTS = [
    # Very established/foundational (should have LOW novelty scores)
    "water",
    "time",
    "matter",
    "energy",
    "life",
    "death",
    "human",
    "animal",
    "plant",

    # Moderately established
    "democracy",
    "capitalism",
    "science",
    "mathematics",
    "philosophy",
    "religion",
    "technology",
    "agriculture",

    # Scientific concepts (mixed novelty)
    "photosynthesis",
    "gravity",
    "evolution",
    "relativity",
    "quantum mechanics",
    "genetics",
    "DNA",

    # Historical innovations (were once novel)
    "printing press",
    "steam engine",
    "electricity",
    "internet",
    "computer",
    "telephone",

    # More recent/emerging (should have HIGHER novelty scores)
    "blockchain",
    "machine learning",
    "CRISPR",
    "cryptocurrency",
    "neural network",
    "virtual reality",
]


def main():
    print("Gathering validation data from Wikidata...")
    print("=" * 70)

    results, errors = gather_validation_data(TEST_CONCEPTS)

    if len(results) < 5:
        print(f"\nNot enough data ({len(results)} results). Cannot validate.")
        return

    analyses = analyze_predictions(results)
    print_validation_report(results, analyses, errors)

    return results, analyses


if __name__ == "__main__":
    main()

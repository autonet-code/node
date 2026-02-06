# Trimmed Mean Aggregation - Usage Guide

## Overview

Trimmed Mean is a Byzantine-resistant aggregation method added to the Autonet Aggregator node. It protects against malicious nodes by trimming extreme values before averaging model updates.

## How It Works

For each model parameter:
1. Collect values from all solver updates
2. Sort values
3. Trim top and bottom X% (default: 20%)
4. Average the remaining values

This ensures that up to X% of malicious nodes cannot influence the aggregated result.

## Usage

### Basic Usage (FedAvg - Default)

```python
from nodes.aggregator.main import AggregatorNode
from nodes.common.contracts import ContractRegistry
from nodes.common.ipfs import IPFSClient

# Create aggregator with default FedAvg
node = AggregatorNode(
    registry=registry,
    ipfs=ipfs,
    node_id="aggregator-1",
    project_id=1,
    # aggregation_method="fedavg"  # This is the default
)

node.run()
```

### Using Trimmed Mean

```python
# Create aggregator with Trimmed Mean
node = AggregatorNode(
    registry=registry,
    ipfs=ipfs,
    node_id="aggregator-1",
    project_id=1,
    aggregation_method="trimmed_mean",  # Use Trimmed Mean
    trim_ratio=0.2  # Trim 20% from top and bottom (default)
)

node.run()
```

### Custom Trim Ratio

```python
# More aggressive trimming (30%)
node = AggregatorNode(
    registry=registry,
    ipfs=ipfs,
    node_id="aggregator-1",
    project_id=1,
    aggregation_method="trimmed_mean",
    trim_ratio=0.3  # Trim 30% from top and bottom
)

node.run()
```

## Configuration Options

### `aggregation_method` (str)
- `"fedavg"` (default): Standard Federated Averaging
- `"trimmed_mean"`: Byzantine-resistant Trimmed Mean

### `trim_ratio` (float)
- Default: `0.2` (20%)
- Range: `0.0` to `0.5`
- Percentage to trim from top and bottom
- Example: `0.2` means trim 20% highest and 20% lowest values

## When to Use Trimmed Mean

### Use Trimmed Mean when:
- You have concerns about malicious nodes
- The network has untrusted participants
- You want robustness against outliers
- Byzantine resistance is required

### Use FedAvg when:
- All nodes are trusted
- Maximum performance is needed (slight overhead with trimming)
- You have very few solver nodes (< 5)

## Performance Considerations

### Minimum Updates Required
- With `trim_ratio=0.2` (20%), you need at least 6 updates for effective trimming
- If fewer updates are available, the algorithm falls back to regular mean
- Formula: `min_updates = ceil(1 / trim_ratio) * 2 + 1`

### Examples:
- `trim_ratio=0.2` (20%): Need 6+ updates
- `trim_ratio=0.3` (30%): Need 4+ updates
- `trim_ratio=0.1` (10%): Need 11+ updates

## Byzantine Resistance Guarantees

The Trimmed Mean method provides the following guarantees:

- **Up to `trim_ratio` malicious nodes** can be tolerated
- With `trim_ratio=0.2` (20%), up to 20% of nodes can be Byzantine
- Malicious nodes cannot influence the aggregated result
- Even if malicious nodes submit extreme values (e.g., 1000x normal), they are trimmed out

### Example Scenarios:

#### Scenario 1: 10 Nodes, 2 Malicious (20%)
```
Honest nodes: [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
Malicious:    [1000.0, -1000.0]

With trim_ratio=0.2:
- Trim 2 from top (1000.0, 1.7)
- Trim 2 from bottom (-1000.0, 1.0)
- Aggregate: mean([1.1, 1.2, 1.3, 1.4, 1.5, 1.6]) = 1.35
```

#### Scenario 2: 5 Nodes, 1 Malicious (20%)
```
Honest nodes: [0.9, 1.0, 1.1, 1.2]
Malicious:    [999.0]

With trim_ratio=0.2:
- Trim 1 from top (999.0)
- Trim 1 from bottom (0.9)
- Aggregate: mean([1.0, 1.1, 1.2]) = 1.1
```

## Testing

A comprehensive test suite is available in `test_trimmed_mean.py`:

```bash
python test_trimmed_mean.py
```

Tests verify:
- Basic trimming functionality
- Edge cases (too few updates)
- Metric aggregation (loss/accuracy)
- Byzantine resistance

## Output Metadata

When using Trimmed Mean, the aggregated model includes additional metadata:

```json
{
  "aggregation_method": "trimmed_mean",
  "trim_ratio": 0.2,
  "num_updates": 10,
  "num_trimmed_per_end": 2,
  "num_used_for_mean": 6,
  "real_training": true,
  "aggregated_metrics": {
    "avg_loss": 0.15,
    "avg_accuracy": 0.92,
    "total_samples": 5000
  }
}
```

## Comparison: FedAvg vs Trimmed Mean

| Feature | FedAvg | Trimmed Mean |
|---------|--------|--------------|
| Byzantine Resistance | No | Yes (up to trim_ratio) |
| Performance | Fastest | Slightly slower |
| Memory Usage | Low | Moderate (sorting) |
| Min Updates Required | 2 | ~6 (for trim_ratio=0.2) |
| Best For | Trusted networks | Untrusted networks |

## Implementation Details

The implementation includes:

1. **`_trimmed_mean_aggregate()`**: Main entry point, dispatches to real or mock
2. **`_trimmed_mean_real_weights()`**: Aggregates PyTorch weight deltas
3. **`_trimmed_mean_mock()`**: Fallback for mock training results

### Algorithm Complexity
- Time: O(n * k * log(n)) where n = num_updates, k = num_parameters
- Space: O(n * k)
- The sorting step dominates the complexity

## References

- **Paper**: "Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates" (Yin et al., 2018)
- **Related**: Krum, Median, Geometric Median (other Byzantine-robust aggregation methods)
- **Autonet Docs**: See `CLAUDE.md` for full architecture overview

## Future Enhancements

Possible improvements:
- Add other Byzantine-robust methods (Krum, Median, etc.)
- Adaptive trim_ratio based on detected anomalies
- Per-layer trimming with different ratios
- Coordinate median instead of trimmed mean

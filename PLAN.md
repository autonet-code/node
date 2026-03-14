# Consensus-as-Truth: MM-Zero Integration Plan

## Problem Statement

The current training loop has a collusion vulnerability: the Proposer knows the ground truth upfront and can share it off-chain with a colluding Solver. The commit-reveal pattern only prevents on-chain front-running, not off-chain cooperation. This lets a Proposer-Solver pair farm rewards without doing real training.

## Design Principle

Replace "Proposer knows the answer" with "truth emerges from solver consensus." Reward Proposers for generating tasks at the frontier of model capability (difficulty-calibrated), not for holding secret answers.

---

## Phase 1: Contract Layer Changes

### 1.1 AutonetLib.sol - New types and enums

Add a new task mode enum and difficulty-related structures:

```solidity
enum TaskMode {
    GROUND_TRUTH,      // Legacy: proposer commits answer
    CONSENSUS_TRUTH    // MM-Zero: truth from solver agreement
}

struct DifficultyTarget {
    uint256 minSolvability;  // BPS: minimum fraction of solvers expected to agree (e.g., 2500 = 25%)
    uint256 maxSolvability;  // BPS: maximum fraction (e.g., 7500 = 75%)
    uint256 peakSolvability; // BPS: optimal difficulty for max reward (e.g., 5000 = 50%)
}

struct SolverRollout {
    address solver;
    bytes32 answerHash;    // Hash of solver's answer
    uint256 confidence;    // Self-reported confidence 0-100
    uint256 submitBlock;
}
```

Extend `TaskProposal` with an optional `taskMode` field. Add `CONSENSUS_PENDING` to `TaskStatus` for the phase where we're collecting solver rollouts.

### 1.2 TaskContract.sol - Consensus-mode task proposal

Add a new proposal function for consensus-mode tasks:

```solidity
function proposeConsensusTask(
    uint256 _projectId,
    bytes32 _specHash,
    DifficultyTarget calldata _difficultyTarget,
    uint256 _learnabilityReward,
    uint256 _solverReward
) external onlyStakedProposer returns (uint256 taskId)
```

Key differences from `proposeTask()`:
- No `_groundTruthHash` parameter (Proposer doesn't know the answer)
- Takes a `DifficultyTarget` instead (defines what difficulty range the Proposer is aiming for)
- Sets `taskMode = CONSENSUS_TRUTH`

Also add a multi-solver submission mechanism:

```solidity
// Multiple solvers submit rollouts (not just one solution)
function submitRollout(
    uint256 _taskId,
    bytes32 _answerHash,
    uint256 _confidence
) external onlyStakedSolver
```

This replaces `commitSolution()` for consensus-mode tasks. Multiple solvers can submit rollouts for the same task. After a collection period (or minimum solver count), the task moves to consensus evaluation.

### 1.3 ResultsRewards.sol - Consensus-based verification

Add a new verification path for consensus-mode tasks:

```solidity
function finalizeConsensusTask(uint256 _taskId) external
```

This function:
1. Collects all solver rollouts for the task
2. Computes majority-vote "silver label" (most common answer hash)
3. Calculates actual solvability: `agreeing_solvers / total_solvers`
4. Computes difficulty score using a bell curve centered on `peakSolvability`
5. Rewards solvers who matched the majority answer (scaled by consensus score)
6. Rewards proposer based on how close actual solvability was to `peakSolvability`

**Difficulty reward curve:**

```solidity
function _computeDifficultyReward(
    uint256 _actualSolvabilityBps,
    DifficultyTarget memory _target,
    uint256 _baseReward
) internal pure returns (uint256)
```

- If `actualSolvability` is outside `[minSolvability, maxSolvability]`: reward = 0 (task was too easy or too hard)
- If `actualSolvability == peakSolvability`: reward = baseReward (maximum)
- Otherwise: reward scales by proximity to peak (quadratic falloff)

This creates the anti-collusion incentive: easy tasks (high solvability) yield low rewards.

**Solver rewards:**

```solidity
function _rewardConsensusSolvers(
    uint256 _taskId,
    bytes32 _majorityAnswer,
    uint256 _consensusScore
) internal
```

- Solvers who matched the majority answer: `baseSolverReward * consensusScore / 100`
- Solvers who disagreed: no reward (but no slashing, since honest disagreement is expected)
- Confidence weighting: optional multiplier for solvers who were correctly confident

### 1.4 ParticipantStaking.sol - No changes needed

The existing role and staking structure works as-is. Proposers, Solvers, Coordinators, and Aggregators keep their current stake amounts.

---

## Phase 2: Node Layer Changes

### 2.1 nodes/proposer/main.py - Self-generating task proposer

Replace `_generate_task_spec()` and `_generate_ground_truth()` with model-driven task generation:

```python
def _generate_consensus_task(self) -> dict:
    """Generate a task at the frontier of current model capability."""
    # 1. Load current global model from IPFS
    global_model = self._load_global_model()

    # 2. Generate candidate inputs (synthetic data)
    candidates = self._generate_candidate_inputs(global_model)

    # 3. Run model on candidates, measure uncertainty
    uncertainties = self._measure_model_uncertainty(global_model, candidates)

    # 4. Select inputs near 50% uncertainty (frontier of capability)
    frontier_inputs = self._select_frontier_inputs(candidates, uncertainties)

    # 5. Package as task spec
    return {
        "type": "consensus_task",
        "mode": "consensus_truth",
        "inputs": frontier_inputs,
        "difficulty_target": {
            "min_solvability": 2500,   # 25%
            "max_solvability": 7500,   # 75%
            "peak_solvability": 5000,  # 50%
        },
    }
```

New method `_propose_consensus_task_cycle()`:
- Calls `_generate_consensus_task()`
- Uploads task spec to IPFS (no ground truth upload)
- Calls `proposeConsensusTask()` on-chain with difficulty target
- Tracks task_id for monitoring

Remove `_reveal_ground_truth_cycle()` for consensus-mode tasks (no ground truth to reveal).

### 2.2 nodes/solver/main.py - Rollout-based training

Add a new method `_train_and_submit_rollout()` for consensus-mode tasks:

```python
def _train_and_submit_rollout(self, task_id: int, task_info: TaskInfo):
    """Train on a consensus-mode task and submit a rollout."""
    # 1. Download task spec from IPFS
    task_spec = self.ipfs.get_json(spec_cid)

    # 2. Train model on the task inputs
    model_update, metrics = self._train_on_consensus_task(task_spec)

    # 3. Generate answer (model's prediction on the task inputs)
    answer = self._generate_answer(model_update, task_spec)

    # 4. Compute confidence from training metrics
    confidence = self._compute_confidence(metrics)

    # 5. Upload solution + answer to IPFS
    solution_cid = self.ipfs.add_json({
        "weight_delta": model_update,
        "answer": answer,
        "confidence": confidence,
        "metrics": metrics,
    })

    # 6. Submit rollout on-chain (answer hash + confidence)
    answer_hash = Web3.keccak(text=json.dumps(answer))
    self.registry.submit_rollout(task_id, answer_hash, confidence)
```

Key difference: multiple solvers independently train and submit rollouts for the same task. No solver sees other solvers' answers before submitting.

### 2.3 nodes/coordinator/main.py - Consensus evaluation

Replace `_verify_solution()` with `_evaluate_consensus()`:

```python
def _evaluate_consensus(self, task_id: int):
    """Evaluate consensus across solver rollouts."""
    # 1. Get all rollouts for this task from on-chain
    rollouts = self.registry.get_rollouts(task_id)

    # 2. Download each solver's answer from IPFS
    answers = [self.ipfs.get_json(r.solution_cid)["answer"] for r in rollouts]

    # 3. Compute majority vote (silver label)
    majority_answer, agreement_ratio = self._compute_majority_vote(answers)

    # 4. Compute difficulty score
    difficulty_target = self.registry.get_difficulty_target(task_id)
    difficulty_score = self._compute_difficulty_score(
        agreement_ratio, difficulty_target
    )

    # 5. Submit consensus evaluation on-chain
    self.registry.finalize_consensus_task(task_id)
```

The Coordinator's role shifts from "verify against known truth" to "tally votes and compute difficulty metrics." This is more mechanical and less susceptible to gaming.

### 2.4 nodes/aggregator/main.py - No major changes

The Aggregator still:
1. Monitors `RewardsDistributed` events
2. Collects verified solver weight deltas
3. Performs FedAvg/TrimmedMean aggregation
4. Publishes the new global model

The only change is that it now aggregates weight deltas from *all solvers who matched the majority answer*, not just a single "correct" solver.

### 2.5 nodes/common/grpo.py - New file: GRPO optimizer

Implement Group Relative Policy Optimization for self-evolution:

```python
class GRPOOptimizer:
    """Group Relative Policy Optimization for self-evolving training."""

    def __init__(self, group_size=8, clip_epsilon=0.2, kl_coeff=0.01):
        self.group_size = group_size
        self.clip_epsilon = clip_epsilon
        self.kl_coeff = kl_coeff

    def compute_advantages(self, rewards: List[float]) -> List[float]:
        """Compute group-relative advantages."""
        mean_reward = sum(rewards) / len(rewards)
        std_reward = (sum((r - mean_reward)**2 for r in rewards) / len(rewards)) ** 0.5
        if std_reward < 1e-8:
            return [0.0] * len(rewards)
        return [(r - mean_reward) / std_reward for r in rewards]

    def compute_difficulty_reward(
        self, solvability: float, peak: float = 0.5
    ) -> float:
        """Bell-curve reward centered on peak solvability."""
        return math.exp(-((solvability - peak) ** 2) / (2 * 0.15 ** 2))
```

---

## Phase 3: Orchestrator Changes

### 3.1 orchestrator.py - New consensus-mode training loop

Add a `--task-mode` flag to select between `ground_truth` (legacy) and `consensus_truth` (new):

```
python orchestrator.py --task-mode consensus_truth --solvers 4
```

The consensus-mode loop:

```
1. PROPOSE       -> Proposer generates task at model capability frontier
2. ROLLOUT       -> Multiple solvers independently train and submit rollouts
3. COLLECT       -> Wait for MIN_SOLVERS rollouts (e.g., 3+)
4. CONSENSUS     -> Coordinator computes majority vote + difficulty score
5. REWARD        -> Difficulty-calibrated rewards distributed
6. AGGREGATE     -> Aggregator performs FedAvg on majority-matching updates
7. PUBLISH       -> Aggregator publishes new global model
```

Note: steps 3-4 replace the old REVEAL_GT -> REVEAL_SOL -> VERIFY flow. No reveals needed because there's no secret to reveal.

### 3.2 Validation checks

Update orchestrator validation to check:
- Rollouts submitted >= MIN_SOLVERS per task
- Consensus computed for each task
- Difficulty scores within expected range
- Proposer reward scales with difficulty calibration quality

---

## Phase 4: Migration Path

### Backward compatibility

- Keep all existing `proposeTask()` / `commitSolution()` / `revealGroundTruth()` functions intact
- Add consensus functions alongside, not replacing
- `TaskProposal.taskMode` defaults to `GROUND_TRUTH` for existing tasks
- Coordinators check `taskMode` to decide verification path
- Orchestrator `--task-mode` flag selects which path to use

### Deployment

1. Deploy updated contracts (additive changes only, no breaking changes)
2. Update `deploy.js` to configure difficulty parameters
3. Node code auto-detects task mode from on-chain `taskMode` field
4. Gradual migration: run both modes in parallel, sunset ground-truth mode once consensus-truth is validated

---

## File Change Summary

| File | Change Type | Description |
|------|------------|-------------|
| `contracts/utils/AutonetLib.sol` | Modify | Add `TaskMode`, `DifficultyTarget`, `SolverRollout` structs; add `CONSENSUS_PENDING` status |
| `contracts/core/TaskContract.sol` | Modify | Add `proposeConsensusTask()`, `submitRollout()`, rollout storage |
| `contracts/core/ResultsRewards.sol` | Modify | Add `finalizeConsensusTask()`, difficulty reward curve, consensus solver rewards |
| `nodes/proposer/main.py` | Modify | Add `_generate_consensus_task()`, `_propose_consensus_task_cycle()` |
| `nodes/solver/main.py` | Modify | Add `_train_and_submit_rollout()`, multi-solver rollout support |
| `nodes/coordinator/main.py` | Modify | Add `_evaluate_consensus()`, majority vote computation, difficulty scoring |
| `nodes/aggregator/main.py` | Minor modify | Aggregate from multiple majority-matching solvers |
| `nodes/common/grpo.py` | New file | GRPO optimizer, difficulty reward computation |
| `nodes/common/contracts.py` | Modify | Add registry methods for new contract functions |
| `orchestrator.py` | Modify | Add `--task-mode` flag, consensus-mode training loop |
| `scripts/deploy.js` | Modify | No new contracts, but configure difficulty defaults |
| `test/` | New tests | Consensus task lifecycle, difficulty curves, anti-collusion scenarios |

---

## Implementation Order

1. **AutonetLib.sol** - Add new types (foundation for everything else)
2. **TaskContract.sol** - Add consensus task proposal + rollout submission
3. **ResultsRewards.sol** - Add consensus finalization + difficulty rewards
4. **nodes/common/grpo.py** - GRPO implementation
5. **nodes/common/contracts.py** - Registry methods for new contract functions
6. **nodes/proposer/main.py** - Consensus task generation
7. **nodes/solver/main.py** - Rollout-based training
8. **nodes/coordinator/main.py** - Consensus evaluation
9. **nodes/aggregator/main.py** - Multi-solver aggregation
10. **orchestrator.py** - Consensus-mode loop + --task-mode flag
11. **Tests** - Contract tests + integration tests

# Node Architecture Reference

## Base Node Structure

All nodes inherit from the `Node` class which implements the constitutional framework.

```python
from nodes.core import Node, NodeRole, Constitution

class Node:
    node_id: str              # Unique identifier
    constitution: Constitution # Immutable principles
    role: NodeRole            # PROPOSER, SOLVER, etc.
    running: bool             # Lifecycle state

    # Four engines
    awareness: AwarenessEngine
    governance: GovernanceEngine
    work: WorkEngine
    survival: SurvivalEngine
```

## The Four Engines

### AwarenessEngine
Perceives environment - network status, resources, external signals.

```python
class AwarenessEngine:
    def tick(self):
        self.last_observation = self.perceive()

    def perceive(self) -> Dict[str, Any]:
        return {
            "network_status": "OK",
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "consensus_alive": self.node.is_consensus_alive(),
        }
```

### GovernanceEngine
Validates instructions against constitutional principles.

```python
class GovernanceEngine:
    pending_instructions: List[Instruction]
    validated_instructions: List[Instruction]

    def tick(self):
        self.check_for_proposals()
        self.process_pending_instructions()

    def validate_instruction(self, instruction) -> bool:
        # In production: LLM validates against principles
        return self.node.constitution.validate_action(
            instruction.action,
            instruction.proof_of_adherence
        )
```

### WorkEngine
Executes validated instructions. **Critical**: Only runs if consensus heartbeat is alive.

```python
class WorkEngine:
    instruction_queue: List[Instruction]

    def tick(self):
        if not self.node.is_consensus_alive():
            return  # HALT - no consensus
        self.execute_next()

    def _execute_instruction(self, instruction):
        if instruction.action == "TRAIN_MODEL":
            self._handle_training(instruction.details)
        elif instruction.action == "VERIFY_SOLUTION":
            self._handle_verification(instruction.details)
        # etc.
```

### SurvivalEngine
Maintains presence, handles replication.

```python
class SurvivalEngine:
    def tick(self):
        self.maintain_presence()
        self.consider_replication()
```

## Constitution

Immutable principles passed to every node.

```python
@dataclass(frozen=True)
class Constitution:
    principles: FrozenSet[str]           # Cannot be modified
    operational_blueprint: Dict[str, Any] # Network config

    def validate_action(self, action, justification) -> bool:
        # Semantic analysis against principles
        # Production: use LLM
        return True

# Default constitution
AUTONET_PRINCIPLES = frozenset([
    "P1: PRESERVE AND EXPAND THE NETWORK IN A SUSTAINABLE MANNER.",
    "P2: UPHOLD THE SANCTITY AND IMMUTABILITY OF THIS CONSTITUTION.",
    "P3: ADVANCE HUMAN RIGHTS AND INDIVIDUAL AUTONOMY.",
    "P4: MINIMIZE SUFFERING AND HARM TO SENTIENT BEINGS.",
    "P5: ENSURE TRANSPARENT AND VERIFIABLE AI TRAINING.",
    "P6: MAINTAIN ECONOMIC FAIRNESS IN REWARD DISTRIBUTION.",
    "P7: PROTECT DATA PRIVACY AND USER SOVEREIGNTY.",
])
```

## Specialized Nodes

### ProposerNode

Generates training tasks with ground truth.

```python
class ProposerNode(Node):
    role = NodeRole.PROPOSER

    def generate_task(self, project_id, description, input_data, ground_truth):
        # 1. Upload input data to blob store
        input_cid = self.store.add_json(input_data)

        # 2. Upload ground truth (hidden until reveal)
        ground_truth_cid = self.store.add_json(ground_truth)

        # 3. Compute hashes
        spec_hash = hash(input_cid + description)
        gt_hash = hash(ground_truth_cid)

        # 4. Submit to blockchain
        task_id = TaskContract.proposeTask(project_id, spec_hash, gt_hash, ...)
        return task_id

    def reveal_ground_truth(self, task_id):
        ResultsRewards.revealGroundTruth(task_id, self.pending_tasks[task_id].gt_cid)
```

### SolverNode

Performs distributed training.

```python
class SolverNode(Node):
    role = NodeRole.SOLVER

    def train(self, task_id, task_spec_cid, global_model_cid):
        # 1. Download task spec
        task_spec = self.store.get_json(task_spec_cid)

        # 2. Download global model
        model = self.store.get_bytes(global_model_cid)

        # 3. Train locally
        update = self._train_model(task_spec, model)

        # 4. Upload update to blob store
        update_cid = self.store.add_json(update)

        return TrainingResult(task_id, update_cid, update["metrics"])

    def commit_solution(self, task_id):
        solution_hash = hash(self.completed_tasks[task_id].update_cid)
        TaskContract.commitSolution(task_id, solution_hash)

    def reveal_solution(self, task_id):
        ResultsRewards.revealSolution(task_id, self.completed_tasks[task_id].update_cid)
```

### CoordinatorNode

Verifies solutions against ground truth.

```python
class CoordinatorNode(Node):
    role = NodeRole.COORDINATOR

    def verify_solution(self, task_id, solver, solution_cid, ground_truth_cid):
        # 1. Download both
        solution = self.store.get_json(solution_cid)
        ground_truth = self.store.get_json(ground_truth_cid)

        # 2. Compare
        is_correct, score, details = self._compare(solution, ground_truth)

        # 3. Upload report
        report_cid = self.store.add_json({...})

        # 4. Submit to blockchain
        ResultsRewards.submitVerification(task_id, solver, is_correct, score, report_cid)

        return VerificationResult(task_id, solver, is_correct, score)
```

### AggregatorNode

Combines verified updates via Federated Averaging.

```python
class AggregatorNode(Node):
    role = NodeRole.AGGREGATOR

    def aggregate_updates(self, project_id, round, update_cids, current_model_cid):
        # 1. Download all updates
        updates = [self.store.get_json(cid) for cid in update_cids]

        # 2. Download current model
        current = self.store.get_json(current_model_cid)

        # 3. Perform FedAvg
        new_model = self._fedavg(updates, current)

        # 4. Upload new model
        new_cid = self.store.add_json(new_model)

        # 5. Update project
        Project.setMatureModel(project_id, new_cid, ...)

        return AggregationResult(project_id, round, new_cid, len(updates))
```

## Common Utilities

### BlockchainInterface

```python
class BlockchainInterface:
    def __init__(self, rpc_url, private_key, chain_id):
        self.web3 = Web3(HTTPProvider(rpc_url))
        self.account = web3.eth.account.from_key(private_key)

    def call_contract(self, address, abi, function, *args):
        contract = self.web3.eth.contract(address, abi)
        return contract.functions[function](*args).call()

    def send_transaction(self, address, abi, function, *args):
        # Build, sign, send, wait for receipt
        ...
```

### BlobStore

```python
class BlobStore:
    def add_json(self, data) -> str:    # Returns content hash
    def add_bytes(self, data) -> str:
    def get_json(self, cid) -> dict:
    def get_bytes(self, cid) -> bytes:
```

### Crypto

```python
def hash_content(content: bytes) -> str      # SHA256
def hash_string(s: str) -> str
def compute_commitment(content, salt) -> str
def verify_commitment(content, salt, commitment) -> bool
def sign_message(message, private_key) -> str
def verify_signature(message, signature, address) -> bool
```

## Node Lifecycle

```python
node = Node(constitution=DEFAULT_CONSTITUTION, role=NodeRole.SOLVER)
node.run(max_cycles=None)  # Runs until stopped

# Each cycle:
#   1. awareness.tick()      - Perceive
#   2. governance.tick()     - Validate pending instructions
#   3. if is_consensus_alive():
#        work.tick()         - Execute validated work
#   4. survival.tick()       - Maintain presence
#   5. sleep(heartbeat_interval / 6)
```

## Configuration

`config/node.toml`:
```toml
[node]
role = "solver"
log_level = "INFO"

[network]
chain_rpc_url = "http://localhost:8545"
blob_store_dir = "/data/autonet/blobs"

[consensus]
heartbeat_interval = 60
heartbeat_timeout = 120

[governance]
principles = ["P1: ...", "P2: ...", ...]
```

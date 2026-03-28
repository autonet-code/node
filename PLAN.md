# RPB Deployment Plan -- Jurisdiction + Training

## Goal
Deploy the Autonet jurisdiction (DAO on werule + RPB contract) and wire
the agent framework so users can flip a switch to start training and
earning rewards. Each step is independently observable.

## Naming
- **RPB** = Recursive Principled Body (the system/pattern)
- **Autonet** = the first jurisdiction deploying an RPB
- The contract stays named `Autonet.sol` for now (rename is cosmetic, do later)
- In code comments and docs, refer to it as "the RPB contract"

---

## Phase 1: Deploy Autonet jurisdiction on local hardhat

**Codebase:** `c:\code\dao\trustless-contracts`

### 1.1 Create unified deployment script

Write `scripts/deploy-autonet-jurisdiction.js` that does the full 3-step
TrustlessFactory flow + RPB deployment in one shot:

**Step 1 -- Infrastructure:**
`trustlessFactory.deployInfrastructure(timelockDelay=1, arbitrationFee=250)`
Deploys: Economy, Timelock, Registry

**Step 2 -- DAO Token:**
`trustlessFactory.deployDAOToken(registry, timelock, tokenParams, govParams)`
Deploys: RepToken ("Autonet Rep", "AREP"), HomebaseDAO ("Autonet DAO")
Initial members: [deployer] with 1000 AREP

**Step 3 -- Configure & Finalize:**
`trustlessFactory.configureAndFinalize(addresses, economyParams, registryKeys, registryValues)`
Links all contracts, transfers control to Timelock.
Include in registryKeys: `"jurisdiction.rpb"` (will be populated after RPB deploy)

**Step 4 -- Deploy RPB (Autonet.sol):**
Deploy AutonetFactory with fresh Autonet implementation.
Call `factory.createAutonet(economy, registry, repToken, timelock, "Autonet Credits", "ATN")`
Note: deployer must hold RepToken (satisfied by Step 2 initial distribution).

**Step 5 -- Configure RPB:**
- `autonet.configureEmission(1000e18, 9900, 300)` (1000 ATN/epoch, 1% decay, 5min epochs for testing)
- Register training service: `autonet.registerService(serviceId, economyAddress, gitHash)`
- Activate service: `autonet.activateService(serviceId)`

**Step 6 -- Store RPB address in Registry:**
Since Registry is now owned by Timelock, either:
(a) Include RPB address registration in Step 3's registryKeys (before finalize), or
(b) Use Timelock to execute `registry.editRegistry("jurisdiction.rpb", rpbAddress)` post-deploy.
Option (a) requires deploying RPB before finalize. Reorder: Steps 1,2,4,3,5.
Option (b) requires a governance proposal. For local testing, deployer IS timelock, so direct call works.

**Output:** Save all addresses to `deployments/autonet-jurisdiction-{network}.json`
and copy relevant addresses to `c:\code\autonet\deployment-addresses.json`.

**Observable:** DAO deployed, RPB deployed, epoch running, service active.
Can verify via hardhat console: `autonet.currentEpoch()` returns 1.

---

## Phase 2: Wire GovernanceBridge into training pipeline

**Codebase:** `c:\code\autonet`

### 2.1 ABI availability

The Python `ContractRegistry` in `nodes/common/contracts.py` loads ABIs from
`artifacts/contracts/governance/AutonetEconomy.sol/Autonet.json`.

Create the directory structure and copy the compiled ABI from
`c:\code\dao\trustless-contracts\artifacts\contracts\Autonet.sol\Autonet.json`.

Add a helper script `scripts/copy-abis.sh` (or .ps1) that does this copy.

### 2.2 Wire GovernanceBridge in AutonetService

**File:** `c:\code\autonet\nodes\service.py`

Add `_init_governance()` method that:
1. Reads blockchain config (rpc_url, private_key, chain_id)
2. Creates `BlockchainInterface`
3. Creates `ContractRegistry` with deployment addresses
4. Creates `GovernanceBridge`
5. Returns it (or None if config missing)

Change `_init_training_feed()` to pass governance:
```python
governance = self._init_governance()
self._training_feed = TrainingDataFeed(feed_config, governance=governance)
```

If governance is None, training still works (offline mode -- already supported).

### 2.3 Update deployment-addresses.json

After Phase 1 deployment, the addresses file needs:
```json
{
  "AutonetEconomy": "0x...",   // RPB contract address
  "AutonetDAO": "0x...",       // HomebaseDAO address
  ...
}
```

The Phase 1 script handles this.

### 2.4 Blockchain config in autonet.yaml

Ensure the config dataclass supports:
```yaml
blockchain:
  rpc_url: "http://127.0.0.1:8545"
  private_key: "0xac0974..."
  chain_id: 31337
```

Check existing `nodes/common/config.py` for the config structure.

**Observable:** Start AutonetService with blockchain config. Training runs.
Hardhat console shows `attestUsage` transactions. Epoch usage increments.

---

## Phase 3: Agent framework defaults

**Codebase:** `c:\code\autonet\atn\`

### 3.1 Pass private_key through AutonetBridge

**File:** `atn/autonet_service.py` line ~169

Add alongside existing rpc_url/chain_id passthrough:
```python
if self.config.private_key:
    self._autonet_config.blockchain.private_key = self.config.private_key
```

### 3.2 Hardcode Autonet jurisdiction defaults

In ATN config, set default rpc_url and chain_id for Shadownet (or local).
Users who flip the training switch don't need to configure anything.
The private_key comes from their wallet connection or env var.

**Observable:** User starts ATN, flips training switch. Training begins.
On-chain attestations flow automatically. No manual config needed.

---

## Phase 4: End-to-end verification

### 4.1 Integration test

Create `tests/test_governance_integration.py`:
- Mock BlockchainInterface (don't need real chain for unit tests)
- Verify GovernanceBridge created when config has rpc_url + private_key
- Verify TrainingDataFeed receives governance bridge
- Verify `attest_task_completion()` calls `registry.attest_usage()`

### 4.2 Manual smoke test

1. Start hardhat node
2. Run Phase 1 deploy script
3. Start AutonetService with blockchain config
4. Generate synthetic agent data (JSONL files)
5. Trigger training cycle
6. Check hardhat console for attestUsage transaction
7. Fast-forward time: `await network.provider.send("evm_increaseTime", [300])`
8. Trigger another attestation (auto-advances epoch)
9. Call `claimParticipantReward()` from hardhat console
10. Verify ATN balance > 0

---

## Implementation order

1. Phase 1: `scripts/deploy-autonet-jurisdiction.js` in trustless-contracts
2. Phase 2.1: Copy ABI artifacts
3. Phase 2.2: Wire GovernanceBridge in `nodes/service.py`
4. Phase 2.4: Verify/update config structure
5. Phase 3.1: Pass private_key in `atn/autonet_service.py`
6. Phase 4.1: Integration test
7. Phase 4.2: Manual smoke test

## Deferred (not in this plan)

- Renaming Autonet.sol to RPB.sol (cosmetic)
- EvolutionProposal deployment (not needed until architecture changes)
- Alignment-weighted rewards (add via CapabilityScorecard later)
- Multi-jurisdiction support (single jurisdiction first)
- Earn-only token constraint (governance decision, token contract change)
- Deploy to Shadownet (do after local hardhat verification)

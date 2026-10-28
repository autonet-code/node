# Autonet on-chain layer, documented from the code

All paths are relative to the repo root. Citations use `file:line`.

This doc describes HEAD at `157796c`. It was first written from `782bdc8` and then fact-checked and updated for the 13 commits since. No Solidity source changed in that range, so contract citations are the same at both commits. Python and script citations point at HEAD.

## 1. Overview

There are three live Solidity sources under `contracts/core/`:

| File | Contracts | Role |
|---|---|---|
| `contracts/core/Substrate.sol` | `Substrate` | Epoch anchor hash chain, agent registry with owner binding and fleet tree, tool authorship registry, household mint claims (merkle), genesis claim, the ATN token, service payments with fee burn/treasury split, vault mint rail |
| `contracts/core/ServiceMarket.sol` | `ServiceRegistry`, `PaymentChannel` | Service listings, plus a prepaid unidirectional EIP-712 voucher channel that settles through `Substrate.payForService` |
| `contracts/core/AnchorElection.sol` | `AnchorElection` | Each epoch's anchor is decided by a single-choice DAO election, using sortition-drawn proposers |

The files under `contracts/test/` are test-only stand-ins: `AnchorTestGovernance.sol`, `MockERC20.sol`, `MockOptionGovernor.sol`, `MockRepToken.sol` and `ServicePayerMock.sol`. Several contracts the daemon depends on live in an external repo ("trustless-contracts"), not in this tree: the DAO Governor, Timelock, RepToken, the DAO Registry, ToolBond, AutonetRepClaim and AutonetVault. Their surfaces are inferred below from the ABIs the Python code embeds.

Compiler setup (`hardhat.config.js:3-35`):
- solc 0.8.20 and 0.8.24 (cancun), optimizer runs=50, `viaIR: true`.
- `Substrate.sol`, `ServiceMarket.sol`, `MockERC20.sol`, `MockRepToken.sol` and `ServicePayerMock.sol` are pinned to 0.8.20 (`hardhat.config.js:24-34`).
- Networks: `hardhat` (chainId 1337), `localhost` (`HARDHAT_LOCAL_URL`), `sepolia`, and `shadownet` (Etherlink, chainId 127823, `https://node.shadownet.etherlink.com`) (`hardhat.config.js:37-56`).

## 2. Substrate.sol

### 2.1 Purpose

Substrate is a single contract with several jobs (`contracts/core/Substrate.sol:16-33`):
- it anchors the federated epoch close as a hash chain;
- it mints ATN to household owner wallets against a merkle root committed in each anchor;
- it hosts the agent registry and the tool authorship registry;
- it is the ATN token itself.

It calls itself a "PURE MONEY contract" that mints no reputation. REP is a DAO-side pull claim on two ledgers recorded here: `agentMintTotal` and `serviceEarnings` (`Substrate.sol:35-42`, `:851-859`, `:1263-1267`).

### 2.2 Constructor

`constructor(address treasury_, address vaultMinter_, address governor_, address anchorer_, bytes32 genesisRoot_)` with EIP-712 domain `("AutonetSubstrate", "1")` (`Substrate.sol:104-122`).

| Argument | Effect |
|---|---|
| `governor_` | Must be non-zero, else `NoAnchorAuthority` (`:113`). Immutable (`:1240`). |
| `treasury_` | Immutable (`:1236`). May be zero, in which case the whole fee burns (`:1349-1351`). |
| `vaultMinter_` | Immutable. Zero disables `mintFromVault` (`:1289`, `:1311`). |
| `anchorer_` | Initial value of the mutable `anchorer`. Emits `AnchorerSet(0, anchorer_)` (`:119-121`). |
| `genesisRoot_` | Immutable (`:1012`). Zero means `claimGenesis` always reverts. |

The constructor also sets `deployedAt = latestCutoff = block.timestamp` (`:117-118`).

### 2.3 State

Anchors:
- `Anchor[] anchors` (private) and `anchorIndexByEpochId` (1-indexed).
- `epochIdByHash`, `latestAnchorHash`, `latestEpochRoot`, `anchorer`, `latestCutoff`.
- `_burnedHistory` (a `Checkpoints.Trace208` keyed by **block timestamp**), `totalBurned`, `burnedAtLatestCutoff` (`:163-224`).
- The `Anchor` struct field order matters to Python tuple indexing (`:128-160`): epochId, epochRoot, prevEpochRoot, prevAnchorHash, agentMintCid, payloadHash, submitter, blockNumber, timestamp, agentMintRoot, windowBurned, mintPool, minted, cutoff.

Agents (`:400-423`, `:549-557`):
- `agents` (registeredAt, active, totalTrainingMint, trainingSubmissionCount), `registeredAgentList`, `agentPeerId`, `agentEndpoint`.
- `agentOwner`, `agentParent`, `bindingNonce`, `agentDepth`.
- Constant `MAX_FLEET_DEPTH = 16` (`:560`).

Tools: `toolAuthor`, `toolRegisteredAt`, `toolCountByAgent` (`:805-809`).

Mint ledger: `mintForEpoch[claimant][epochIdHash]`, `agentMintTotal`, `networkMintTotal` (`:849-863`).

Genesis: `genesisClaimed`, `genesisMinted` (`:1015-1018`).

ATN token:
- `_atnBalance`, `_atnAllowance`, `atnTotalSupply`.
- Per-account and supply checkpoints keyed by **block number** (`:1063-1080`, `:1087-1091`).

Fees:
- `serviceFeeBps = 250` (mutable).
- Constants `FEE_TREASURY_BPS = 5000`, `BPS_DENOM = 10000`, `MIN_SERVICE_FEE_BPS = 50`, `MAX_SERVICE_FEE_BPS = 1000` (`:1226-1233`).
- `serviceEarnings` (`:1267`).

### 2.4 External functions and access control

**Anchoring**

| Function | Access | Notes |
|---|---|---|
| `setAnchorer(address)` | `msg.sender == governor` (`:227-231`) | Emits `AnchorerSet`. |
| `anchor(epochId, epochRoot, prevEpochRoot, prevAnchorHash, agentMintCid, payloadHash, agentMintRoot, cutoff)` | governor, or `anchorer` if non-zero (`:282-285`) | See invariants. |
| `anchorHashOf(...)` (pure), `anchorCount`, `getAnchor(i)`, `getAnchorByEpochId`, `isAnchored`, `isAnchoredByHash` | view | `:350-394` |
| `burnedUpTo(ts)`, `burnedBetween(from,to)`, `burnedSinceAnchor()`, `anchorSnapshotTime()`, `nextMintPool()` | view | `:234-263` |

**Agents**

| Function | Access | Notes |
|---|---|---|
| `registerAgentBound(peerId, owner, parentAgent, sig)` | anyone not yet registered. `peerId` must be non-empty. | The only registration path (`:674-695`). The signature may come from `owner`, or from `parentAgent` (`parentMaySign=true`, `:694`, `:650-653`). |
| `rotateOwner(newOwner, newParent, newOwnerSig)` | caller must be `isLive` | The parent cannot sign here (`:714-721`). |
| `revokeAgent(agent, ownerSig)` | `msg.sender == agentOwner[agent]`, or a relayed 65-byte EIP-712 `RevokeAgent(address agent)` signature by the owner | Final (`:735-745`). Checks only that the target itself is still active. |
| `updatePeerId(bytes)` | `isLive(msg.sender)`, non-empty | `:444-449` |
| `updateEndpoint(string)` | `isLive(msg.sender)`. An empty string clears it. | `:463-467` |
| `isLive`, `isRegistered`, `areRegistered`, `registeredAgentCount`, `getRegisteredAgent`, `getAgentPeerId`, `getAgentEndpoint`, `getAgentOwner`, `getAgentParent`, `sameOwner`, `bindingHash`, `revokeHash` | view | `sameOwner` returns false if A's owner is unset (`:775-779`). |

**Tools**: `registerTool(bytes32 manifestDigest)` requires `isLive(msg.sender)` and a non-zero digest. It reverts if the digest is already registered by anyone, so the first registrant wins permanently (`:825-836`). View: `isToolRegistered`.

**Mint, genesis and token**

| Function | Access | Notes |
|---|---|---|
| `claimFor(claimant, amount, epochIdHash, proof)` | permissionless | ATN always goes to `claimant` (`:910-972`). |
| `claimGenesis(amount, proof)` | `msg.sender` claims its own leaf once | `:1032-1046` |
| `mintFromVault(to, amount)` | only `vaultMinter` (non-zero) | Emits `VaultMint` and an `ATNTransfer` from 0. Does not touch `agentMintTotal` (`:1310-1316`). |
| `transfer`, `approve`, `transferFrom`, `balanceOf`, `allowance`, `balanceOfAt(addr, block)`, `atnTotalSupplyAt(block)` | ERC20-shaped | Reverts on failure instead of returning false. An allowance of `type(uint256).max` is not decremented (`:1143-1178`). There is no name/symbol/decimals/`totalSupply()`; supply is `atnTotalSupply` (`:1055-1060`). |
| `payForService(recipient, amount, requestId)` | any caller, including contracts | No liveness check on payer or recipient. The only checks are a non-zero recipient and the payer's balance (`Substrate.sol:1342-1344`). `ServicePayerMock` pins that contract callers work (`contracts/test/ServicePayerMock.sol:8-18`). |
| `setServiceFeeBps(uint16)` | governor | Bounded to 50..1000 bps (`:1251-1261`). |
| `hasSubmittedForEpoch` | view | `:975-977` |

### 2.5 Invariants

**Anchor ordering and hash chaining** (`:286-298`, `:329-334`, `:365-369`):
- The cutoff must satisfy `latestCutoff < cutoff <= block.timestamp`.
- The epochId must be non-empty and not already anchored.
- `prevEpochRoot` must equal `latestEpochRoot`, and `prevAnchorHash` must equal `latestAnchorHash`.
- The new `latestAnchorHash` is computed from content only: `keccak256(abi.encode(epochId, epochRoot, prevEpochRoot, prevAnchorHash, agentMintCid, payloadHash, agentMintRoot, cutoff))`. It contains no sender or block, so the next epoch's `prevAnchorHash` is known before the anchor lands. `AnchorElection.optionHashOf` uses the identical encoding (`AnchorElection.sol:189-193`), so an option's hash equals the anchor hash it would produce.

**Fees-only mint cap** (`:301-325`, `:941-950`):
- Each anchor stores `mintPool = previous anchor's windowBurned`, and `windowBurned = burnedUpTo(cutoff) - burnedAtLatestCutoff`.
- The first anchor's `mintPool` is 0 (`nextMintPool`, `:260-263`), so epoch 1 can mint nothing.
- `claimFor` reverts with `EpochPoolExceeded` if cumulative `minted` would exceed `mintPool`, whatever root was anchored.

**Merkle claim** (`:929-939`, `:982-995`):
- Leaf is `keccak256(bytes.concat(keccak256(abi.encode(claimant, amount))))`. Pairs are hashed in sorted order and proofs carry no index bits.
- The anchor's root must be non-zero.
- Idempotency is per (claimant, epochIdHash) through `mintForEpoch != 0`.
- `amount == 0` returns early, with no proof check, no event and no state change (`:925-927`).
- `agents[claimant].totalTrainingMint` and `trainingSubmissionCount` increase only if the claimant happens to be a registered agent (`:957-960`). Since 2026-10-27 the claimant is the owner wallet.

**Genesis** (`:1032-1046`, `:997-1009`): the leaf encoding matches `claimFor` but uses `msg.sender`. Genesis mints sit outside `mintPool` and do not accrue `agentMintTotal`, so they grant no REP claim. The stated supply identity is: total minted = epoch mints + vault mints + genesisMinted.

**Owner binding** (`:626-661`):
- EIP-712 `OwnerBinding(address agent,address parent,uint256 nonce)` at the current `bindingNonce[msg.sender]`. The nonce increments on every bind, which makes old signatures unreplayable.
- A parent must be registered, live, and have the same owner. Depth is at most 16.
- `OwnerBound(agent, owner, parent, prevOwner, nonce)` is emitted on first bind (prevOwner = 0) and on every rotation (`:577-583`).
- Rotation needs only the agent key plus the new owner's signature; the old owner is not consulted (`:516-523`).
- Rotating a parent does not touch its children (`:705-708`).

**Liveness**: `isLive` requires the agent to be registered, then checks the agent itself and up to 16 ancestors (17 nodes, the loop runs `i = 0..MAX_FLEET_DEPTH`) and fails if any is inactive (`:749-758`). Revoking a node therefore kills its whole subtree.

**Fee split** (`:1348-1375`):
- `fee = amount * serviceFeeBps / 10000`. Half the fee goes to the treasury (when one is set) and the rest burns.
- The recipient gets `amount - fee`, and `serviceEarnings[recipient] += net`.
- The burned share lowers `atnTotalSupply`, raises `totalBurned`, and pushes a `_burnedHistory` checkpoint at `block.timestamp`.

### 2.6 Events

`EpochAnchored`, `AnchorerSet`, `AgentRegistered`, `PeerIdUpdated`, `EndpointUpdated`, `OwnerBound`, `AgentRevoked`, `ToolRegistered`, `TrainingRecorded`, `GenesisClaimed`, `ATNTransfer`, `ATNApproval`, `ServicePayment(payer, recipient, amount, requestId indexed)`, `ServiceFee(payer, recipient, amount, burned, toTreasury)`, `ServiceFeeBpsSet`, `VaultMint`, plus the inherited `EIP712DomainChanged`. The packaged ABI lists the same set (`nodes/common/abi/Substrate.json`).

## 3. ServiceMarket.sol

### 3.1 ServiceRegistry

- Constructor: `(address substrateAddr)`, which must be non-zero (`contracts/core/ServiceMarket.sol:98-101`).
- State: `serviceCount` (ids start at 1, so 0 means none), and `services[id] = {provider, specDigest (sha256 of the spec blob), askAmount (ATN), active, registeredAt, updatedAt}` (`:65-77`).
- `registerService(specDigest, askAmount)`: gated by `onlyAgent`, which is `substrate.isLive(msg.sender)`. The digest must be non-zero (`:103-127`).
- `updateServiceAsk(id, ask)` and `retireService(id)`: provider only, and the service must be active (`:130-154`). Neither checks `isLive`.
- `getService(id)`: view.
- Events: `ServiceRegistered`, `ServiceAskUpdated`, `ServiceRetired`.
- Registry entries and channels are not linked on chain. `openChannel` takes no serviceId.

### 3.2 PaymentChannel

- Constructor: `(address substrateAddr, uint256 challengeWindow_)`. Both must be non-zero. EIP-712 domain `("AutonetPaymentChannel", "1")` (`:288-295`).
- State: `challengeWindow` (governed), `channelCount`, and `channels[id] = {client, provider, deposit, claimed, closesAt, status, window}`, where `Status {None, Open, Closing, Settled, ClientClosing}` (`:235-254`).
- Voucher typehash: `Voucher(uint256 channelId,uint256 cumulativeAmount)` (`:257-258`).

| Function | Access | Behavior |
|---|---|---|
| `setChallengeWindow(w)` | `msg.sender == substrate.governor()`, `w > 0` | Applies to new channels only. Each channel keeps the window it opened with (`:299-304`, `:322`). |
| `openChannel(provider, deposit)` | anyone. `provider != 0`, `deposit > 0` | Pulls ATN with `substrate.transferFrom`. Nothing checks that the provider is registered or differs from the client (`:308-330`). |
| `voucherHash(id, cum)` | view | `:334-341` |
| `startClose(id)` | client, status Open | Moves to ClientClosing with `closesAt = now + window` (`:347-354`). |
| `closeChannel(id, cum, sig)` | provider. Status Open, or ClientClosing before `closesAt` | The signature must recover to the client. Pays `min(cum, deposit)` through `substrate.payForService(provider, pay, bytes32(channelId))`, so the fee applies and `serviceEarnings` accrues. Then moves to Closing (`:362-392`). |
| `withdrawRemainder(id)` | anyone, after `closesAt`, status Closing or ClientClosing | Refunds `deposit - claimed` to the client with `substrate.transfer`, which pays no fee. Moves to Settled (`:397-412`). |
| `getChannel(id)` | view | `:414-416` |

Events: `ChannelOpened`, `ChannelClosed`, `RemainderWithdrawn`, `ChannelCloseStarted`, `ChallengeWindowSet`.

Invariants:
- A channel closes once, and over-claims are capped at the deposit.
- The channel contract itself is the `payForService` payer, so the fee is drawn from the escrowed balance.
- The doc block explains the theft ceiling and griefing analysis (`:193-221`).

## 4. AnchorElection.sol

- Constructor: `(address substrate_, address governor_, address repToken_)`, all immutable (`contracts/core/AnchorElection.sol:119-123`).
- Mutable parameters: `proposers = 3`, `proposerFloorPpm = 0`, `maxOptions = 8`, `openAfterAttempts = 2` (`:85-91`).

### 4.1 Proposal identity

- `descriptionOf(n, attempt)` returns `"Anchor election: epoch <n>, attempt <a>"`.
- `callOf` encodes `anchorWinner(n, attempt)`.
- `proposalIdOf` calls `governor.hashProposal([this],[0],[callOf],keccak(desc))` (`:134-159`).

Every daemon therefore derives the same proposal id.

### 4.2 The draw

- `drawTimepoint(pid) = proposalSnapshot(pid) - votingDelay()`, which is the proposal's creation time (`:167-169`).
- `isDrawn` (`:171-183`) works like this:
  - the account must be `substrate.isLive`;
  - if `attempt >= openAfterAttempts`, every live agent is drawn;
  - otherwise it reads `repToken.getPastVotes` and `getPastTotalSupply` at the timepoint, applies the ppm floor, and accepts the account when `mulDiv(keccak(chainid, this, n, attempt, account), supply, uint256.max) < proposers * votes`.
- This uses delegated votes, not balances.

### 4.3 Registering options

`registerOption(n, attempt, Option)` (`:198-220`) requires all of the following:
- `governor.state(pid) == 0` (Pending);
- the caller is live;
- the timepoint is in the past;
- the caller is drawn;
- the caller has not registered for this pid yet;
- the option hash has not been seen;
- the option count is below `maxOptions`.

The option content is not validated here. `Substrate.anchor` validates it at execution.

### 4.4 The option source the counting module reads

`optionCount(pid)`, `optionHash(pid, i)` and `getOption(pid, i)` (`:223-239`).

### 4.5 Execution

`anchorWinner(n, attempt)`, `onlyTimelock` (`:249-260`):
- it requires `n == substrate.anchorCount() + 1`;
- it reads `governor.winningOption(pid)` and reverts if the result is undecided;
- it calls `substrate.anchor(...)` with the stored option.

This means Substrate's `anchorer` must be set to this contract, either at construction or through `setAnchorer`.

### 4.6 Governance

`setParams(...)`, `onlyTimelock`. It requires `proposers > 0`, `0 < maxOptions <= 256`, and `floor <= 1e6` (`:266-280`). The 256 cap lines up with the daemon voting through `castVote(pid, uint8 index)` (`nodes/common/anchor_election.py:229`).

`onlyTimelock` resolves the timelock dynamically through `governor.timelock()` (`:125-128`).

### 4.7 Events

`OptionRegistered`, `WinnerAnchored`, `ParamsSet`.

## 5. Test-only contracts

- **`TestRepVotes`, `TestTimelock`, `TestAnchorGovernor`** (`contracts/test/AnchorTestGovernance.sol:24-192`): a stock OZ Governor and a timestamp-clocked ERC20Votes with an open `mint` and a `getPastBalance` history. These stand in for the real DAO in governed-anchor tests.
- **`MockOptionGovernor`** (`contracts/test/MockOptionGovernor.sol`): sets proposal state, snapshot and winner directly. It acts as its own timelock and forwards calls with `exec`.
- **`MockRepToken`** (`contracts/test/MockRepToken.sol`): exposes `getPastBalance`, `getPastVotes` and `getPastTotalSupply` with upper-lookup semantics. It also has `setDelegatedVotes` and `addUndelegatedSupply` to model delegation.
- **`MockERC20`, `FalseReturnERC20`**: open-mint ERC20s.
- **`ServicePayerMock`**: a contract-context `payForService` caller, standing in for ToolBond slashing.

## 6. External contracts as seen from the Python ABIs

### 6.1 Substrate (Python side)

The daemon reaches Substrate through several separate ABIs:

- **`SUBSTRATE_ABI` (inline)** (`atn/on_chain.py:47-357`): `updatePeerId`, `registerAgentBound`, `rotateOwner`, `revokeAgent`, `isLive`, `bindingNonce`, `isRegistered`, `areRegistered`, `getAgentPeerId`, `updateEndpoint`, `getAgentEndpoint`, `agents`, `registeredAgentCount`, `getRegisteredAgent`, `agentOwner`, `agentMintTotal`, `balanceOf`, `mintForEpoch`, `claimFor`, `payForService`, `serviceEarnings`, `serviceFeeBps`, `anchorCount`, `latestAnchorHash`, `latestEpochRoot`, `atnTotalSupply`, `networkMintTotal`, and the events `ServicePayment`, `ServiceFee`, `AgentRegistered`, `AgentRevoked`, `PeerIdUpdated`, `OwnerBound`.
- **`_VOICE_ABI`** (`nodes/common/voice_state.py:93-193`): `atnTotalSupplyAt`, `burnedBetween`, `deployedAt`, `anchorCount`, `getAnchor` (the full 14-field tuple), and the events `AgentRegistered`, `AgentRevoked`, `OwnerBound`, `ServiceFee`.
- **Packaged ABI** (`load_abi("Substrate")`): used by `EpochAnchorer` (`nodes/common/epoch_anchorer.py:167-168`), `AuthoritativeChainSubmitter` (`nodes/common/authoritative_submitter.py:170-171`), `ChainCandleSeed` (`nodes/common/candle_seed.py:113-123`), and peer attribution (`atn/autonet_service.py:1175-1179`, falling back to the inline ABI).
- **`SUBSTRATE_TOOL_ABI`** (`atn/runtime/tool_bond.py:179-185`): `registerTool`, `toolAuthor`, `approve`, `allowance`, `balanceOf`.

What these call sites rely on:
- **Agent registration.** Signing must match the contract's EIP-712 domain exactly (`atn/on_chain.py:824-860`).
- **Payment verification.** `verify_service_payment` decodes `ServicePayment` from a single receipt and refuses log-range scans (`atn/on_chain.py:1534-1642`).
- **Gas for `payForService`.** The estimate is padded 1.6x with a 400k floor, because first-time checkpoint SSTOREs are underestimated (`atn/on_chain.py:1494-1502`).
- **Close inputs** (`nodes/common/voice_state.py:1294-1302`, `:1309-1333`). The close reads `getAnchor(n-1)[7]` (blockNumber), `[8]` (timestamp) and `[10]` (windowBurned). It rebuilds the owner map and the revoked set from `AgentRegistered`, `OwnerBound` and `AgentRevoked` logs up to that block, so no archive node is needed.
- **Pipelined epochs.** Anchors are read from `burnedBetween(prev, cutoff)` and `atnTotalSupplyAt(block)` (`voice_state.py:1067`, `:1292`). `nodes/service.py:982-1016` exposes chain, burn and anchor-count sources.
- **Anchor building.** `prepare_anchor` reads `latestEpochRoot`, `latestAnchorHash` and `latestCutoff`. `anchorHashOf` chains a pipelined child onto an unanchored parent (`nodes/common/epoch_anchorer.py:242-278`, `nodes/common/chain_submission_driver.py:373`).
- **Claims.** `getAnchorByEpochId` (cid at `[4]`, root at `[9]`), `hasSubmittedForEpoch` and `claimFor`, with a local proof pre-check (`authoritative_submitter.py:203-320`). `nodes/common/mint_merkle.py:81-158` mirrors the leaf and pair hashing.
- **Direct dev anchoring.** `GovernedAnchorer.direct` compares `substrate.governor()` with the daemon key (`nodes/common/governed_anchor.py:114`, `:121-123`).
- **Owner sweep.** `ws_server` reads `agentOwner` and `balanceOf` and calls `transfer` (`atn/ws_server.py:5961-5975`).

### 6.2 ServiceRegistry and PaymentChannel (Python side)

Inline ABIs (`atn/on_chain.py:360-590`):
- registry: `registerService`, `updateServiceAsk`, `retireService`, `serviceCount`, `services`, `getService`, `ServiceRegistered`;
- channel: `openChannel`, `voucherHash`, `closeChannel`, `startClose`, `withdrawRemainder`, `channelCount`, `channels`, `getChannel`, `challengeWindow`, `ChannelOpened`;
- ATN: `approve`, `allowance`.

Vouchers are built offline with the same domain (`atn/on_chain.py:593-652`, `:2041-2075`). `verify_voucher` accepts only status 1 (Open) (`:2136`).

### 6.3 DAO Governor and Timelock (external)

`GOVERNOR_ABI` (`nodes/common/governed_anchor.py:49-75`) contains `hashProposal`, `propose`, `queue`, `execute` (payable), `castVote(uint256,uint8)`, `state` (uint8 in OZ `ProposalState` order, `:32`), `hasVoted`, `proposalSnapshot`, `proposalEta`, `proposalThreshold`, `clock`, `getVotes`. `AnchorElection` adds `votingDelay`, `winningOption` and `timelock` (`AnchorElection.sol:25-38`).

Settings are rebuilt from events (`nodes/common/voice_state.py:864-966`): `VotingDelaySet`, `VotingPeriodSet` and `TimelockChange` on the governor, and `MinDelayChange` on the timelock. These set the epoch floor and ceiling (`voice_state.py:1029-1031`).

What this implies about the Governor:
- It is OZ-compatible and timestamp-clocked.
- It uses an options counting module in which `castVote` support is an option index (`anchor_election.py:229`).
- It decides early (`governed_anchor.py:10-13`).

Two proposal shapes are in use:
- **Governed path:** target Substrate, calldata `anchor(...)`, description `"Anchor epoch <id> 0x<root>"` (`governed_anchor.py:78-80`, `:136-148`).
- **Election path:** target AnchorElection (`anchor_election.py:150-155`).

`ChainSubmissionDriver` prefers the election path whenever `anchor_election_address` is set (`chain_submission_driver.py:263-311`).

### 6.4 RepToken (external)

- **Voice weights** use `getPastBalance(account, ts)` and `getPastTotalSupply(ts)` at the anchor timestamp. They read balances, not votes (`voice_state.py:73-91`, `:1359-1396`).
- **The draw** uses `getPastVotes` (`AnchorElection.sol:40-44`).
- **Operator delegation flow:** `delegates`, `name`, `nonces`, `delegateBySig` (`atn/ws_server.py:69-86`, `:5857-5897`).
- **Tool security:** `balanceOf` (`atn/tool_security.py:50-54`, `:446-447`).

Implied: an ERC20Votes and ERC20Permit token with `mode=timestamp` plus a balance-history extension. That matches the mock (`contracts/test/MockRepToken.sol:22-23`, `:64-75`).

### 6.5 AutonetRepClaim (external)

`claimable(address earner)` and `claim(address earner)` (`chain_submission_driver.py:60-68`). The operator calls `claim(earner)` for each household after the ATN claims (`:632-660`). Implied: claims are permissionless per earner and minted 1:1 on Substrate's earnings ledgers (`Substrate.sol:35-42`).

### 6.6 ToolBond (external)

- Bond gate: `isBonded` (`atn/runtime/tool_bond.py:42-46`).
- Client: `minStake`, `reviewerShareBps`, `bonds`, `toolRating`, `toolHarm`, `bond`, `requestUnbond`, `withdraw` (`tool_bond.py:164-177`).
- Reviews: `currentEpoch`, `epochInfo`, `getRound`, `roundAxis`, `reviewAt`, `reviewOf`, `revealEnds`, `commits`, `commitmentOf`, `commit(commitment, stake, reviews)`, `reveal`, `forfeit`, `finalize`, `claim` (`atn/runtime/review_batch.py:75-108`).
- Events the close folds: `Reviewed`, `Finalized`, `Overruled`, `HarmSettled`, `ReviewSettled`, `Bonded`, `UnbondRequested`, `Unbonded`, `EpochStarted`, `ParamsSet` (`voice_state.py:307-337`).

Implied:
- Bonds are paid in ATN, pulled through Substrate `approve` and `transferFrom` (`tool_bond.py:343-349`).
- An author must hold `Substrate.toolAuthor` (`tool_bond.py:328-338`).
- Slashes go through `Substrate.payForService` from contract context (`ServicePayerMock.sol:8-18`).
- A review tuple's scores field is `bytes16`, which caps a charter at 16 axes (`review_batch.py:37`, `:55-59`).

### 6.7 DAO Registry (external)

- `getAllKeys()` and `getAllValues()` (`atn/config.py:772-777`, `atn/sanctions.py:28-35`).
- `getRegistryValue(string)`, called through an obfuscated name and the key `node.code.hash.<version>` (`atn/_cache.py:66-68`, `:155-180`).
- `RegistryUpdated(string key, string value)` logs (`voice_state.py:614-624`).

Keys used:
- `jurisdiction.<id>.<contract>` for address discovery (`config.py:907-936`);
- `jurisdiction.autonet.charter`;
- `sanction.<addr>`;
- `jurisdiction.<id>.robots`, `.robot_standards`, `.parent`;
- the epoch rule keys (`voice_state.py:612`, `:694`, `:749-756`, `:842-859`).

### 6.8 AutonetVault (external)

There is no daemon call site. The vault appears only in `scripts/s0_local_run.py:727-731` (`buy(token, amount)`) and in the deployment record's constructor `(atnToken, registryAddress, admin)` (`deployments/shadownet-autonet-dao-e2e.json:24-31`). It mints through `Substrate.mintFromVault`.

## 7. Deployment scripts and records

| Script | What it does |
|---|---|
| `scripts/deploy_substrate.js` | Deploys `Substrate(TREASURY or deployer, VAULT_MINTER or 0, TIMELOCK or GOVERNOR (required, non-zero), ANCHORER or 0, GENESIS_ROOT or 0)` (`:38-71`). Writes `deployments/<network>.json` (`:84-99`). Its comment points at an external key file path (`:16-17`). |
| `scripts/deploy_economy.js` | Deploys only `ServiceRegistry`. PaymentChannel is "pending challengeWindow blessing" (`:6-8`, `:46`). Reads `substrate_address` from `deployments/<network>.json` and writes `<network>_economy.json`. |
| `scripts/deploy_service_market.js` | Deploys `ServiceRegistry` and `PaymentChannel(SUBSTRATE_ADDR, CHALLENGE_WINDOW or 3600)` and prints JSON. Writes no file. |
| `scripts/fund_smoke_wallets.js` | Tops up a second wallet and writes its record, private key included, to `$AUTONET_KEYS_DIR/<network>-smoke-wallet.json` (default `~/.autonet/keys`, dir 0700, file 0600 where the OS supports it), outside the repo. It no longer prints the key (`:19-24`, `:32-34`, `:81-101`). `scripts/testnet_chain_smoke.py` reads the same path (`:172-179`). |
| `scripts/sim_service_market.js` | In-memory scenario narrative over Substrate, ServiceRegistry and PaymentChannel. |
| `scripts/sync_contract_abis.py` | Copies compiled ABIs to `nodes/common/abi/<Name>.json`. Its list contains only `"Substrate"` (`:25`). |

AnchorElection, ToolBond, RepClaim, the Vault and the DAO are deployed only by an external script (`trustless-contracts scripts/deploy-autonet-jurisdiction.js`, cited at `scripts/s0_local_run.py:5-8`, `:219-246`).

Deployment records:
- **`deployments/shadownet.json`**: Substrate `0x2017...A256`, governor/timelock `0x0489...`, treasury `0x652d...` (which is the `dao_registry` in `registry.json`), and vault_minter `0x90D7...` (which is `autonet_vault` in `registry.json`).
- **`deployments/shadownet-autonet-dao-e2e.json`**: an older (2026-07-10) full stack with a different Substrate `0x4C4d...`, a CharterAnchor contract, and a "pending attach proposal".
- **`deployments/shadownet_economy.json`**: an older Substrate `0xE266...`.
- **No key files.** `deployments/` holds addresses only. `.gitignore` blocks wallet, deployer, throwaway, key, secret, `.key` and `.pem` files there (`.gitignore:70-78`).

Packaged ABIs: `nodes/common/abi/Substrate.json` and `AnchorElection.json` (`{"contractName", "abi"}`). Both match the sources function for function. They ship as package data (`pyproject.toml:159-160`).

## 8. How addresses and ABIs resolve at runtime

**ABI resolution** (`nodes/common/contract_artifacts.py:1-19`, `:47-56`), in order:
1. `ATN_ARTIFACTS_DIR/contracts/<sub>/<Name>.sol/<Name>.json`;
2. the repo `artifacts/`;
3. the packaged `nodes/common/abi/<Name>.json`.

Most service and channel code does not use this path; it embeds inline ABIs (`atn/on_chain.py:45-46`).

**Addresses.** `RPBConfig` fields are listed at `atn/config.py:237-305`. Nothing resolves at boot. The registry seed is read when the daemon joins (`config.py:545-556`), in this order (`_load_registry_seed`, `config.py:939-998`):
1. `ATN_REGISTRY_URL` (HTTP JSON);
2. `ATN_JURISDICTION_REGISTRY`, or the saved `<data_dir>/jurisdiction.json`, read on chain as `jurisdiction.<id>.*` keys. If the user selected a jurisdiction and the read fails, nothing joins (`:966-972`);
3. the packaged `atn/registry.json`, which is identical to the repo-root `registry.json`.

`_parse_registry_data` maps the contract keys to config fields (`config.py:580-657`). `service_registry` populates both `service_registry_address` and the legacy `registry_address` (`:641-647`). `resolve_network_registry` never overwrites explicitly set fields (`:1024-1045`).

A jurisdiction will not start without substrate, rep_token, tool_bond, dao and timelock (`config.py:664-670`).

Defaults come from `atn/jurisdiction.py` (CHAIN_ID 127823, RPC, GOVERNOR_ADDRESS). Per-client fallbacks:
- `substrate_address` falls back to `rpb_contract_address` (`atn/on_chain.py:711-717`);
- ToolBond resolves from env `ATN_TOOL_BOND_ADDRESS`, then config, then the registry seed (`atn/runtime/tool_bond.py:79-94`). It fails closed unless `ATN_TOOL_BOND_DEV_OVERRIDE=1`.

## Inconsistencies and dead code

1. **Substrate's own header contradicts the contract.** The "What is deliberately NOT here" block says there is no ATN token or ERC20 (`Substrate.sol:47-51`), but the contract is the ATN token (`:1048-1178`). The usage sketch at `:62-73` also describes per-agent minting, while the code now uses owner-wallet claims.
2. **Unreachable branch in `setServiceFeeBps`.** The `governor == address(0)` check and the "zero address freezes the knob" comment (`Substrate.sol:1238-1239`, `:1249-1253`) can never trigger, because the constructor reverts on a zero governor (`:113`).
3. **`Agent.totalTrainingMint` and `trainingSubmissionCount` are effectively dead.** They grow only when the claimant is a registered agent (`Substrate.sol:957-960`), while claimants are now owner wallets. `get_agent_record` and `get_provider_record` still report them (`atn/on_chain.py:1219-1230`, `:1313-1323`).
4. **ServiceRegistry's liveness gate is partial.** The `onlyAgent` comment says a revoked agent "may not list or change services" (`ServiceMarket.sol:103-106`), but `updateServiceAsk` and `retireService` check only the provider, not `isLive` (`:130-154`).
5. **PaymentChannel status handling is out of date in Python.** `get_channel`'s docstring lists only statuses 0 to 3, missing ClientClosing (4) (`atn/on_chain.py:2252-2253`). `verify_voucher` rejects any status other than Open (`:2136`), but the contract still accepts `closeChannel` during ClientClosing (`ServiceMarket.sol:369-370`).
6. **Hardhat compiler pin comment is wrong.** The config says every non-test-DAO source is pinned to 0.8.20 (`hardhat.config.js:17-21`). `AnchorElection.sol` and `MockOptionGovernor.sol` are missing from the overrides list (`:24-34`), so with pragma `^0.8.20` Hardhat compiles them with the highest matching compiler, 0.8.24 cancun.
7. **`sync_contract_abis.py` only syncs Substrate** (`:25`), but `AnchorElection.json` is packaged and loaded (`nodes/common/anchor_election.py:83`). The drift test covers Substrate only (`tests/test_contract_artifacts.py:26-58`).
8. **`contract_artifacts._CONTRACT_SOURCES["ServiceMarket"]` points at a non-existent artifact.** It maps to `ServiceMarket.sol/ServiceMarket.json` (`nodes/common/contract_artifacts.py:36-44`), but that file defines `ServiceRegistry` and `PaymentChannel`, so no such artifact exists.
9. **`sim_service_market.js` is broken.** It calls `substrate.registerAgent(...)` (`scripts/sim_service_market.js:110-112`), which no longer exists; `registerAgentBound` is the only path. It also passes 5 constructor args while the comment says "(treasury, vaultMinter, governor)" (`:89-93`).
10. **`package.json` deploy scripts reference a missing file.** They point at `scripts/deploy.js`, which does not exist.
11. **Deployment records drift from the scripts that write them.**
    - `shadownet.json` has `rep_token`, `anchor_quorum_bps`, `genesis_signers` and `genesis_quorum`, which `deploy_substrate.js` never writes (`:84-95`). They belong to the REP-signature quorum anchor model that was removed (`Substrate.sol:194-199`, `epoch_anchorer.py:334-336`). The file also lacks anchorer and genesisRoot.
    - `shadownet_economy.json` contains fields `deploy_economy.js` does not produce, and an old Substrate address.
    - `registry.json`'s `service_registry`, `payment_channel`, `tool_bond`, `rep_claim` and `economy` appear in no deployment record.
    - `registry.json` has no `anchor_election` or `deploy_block`, so on shadownet the election path and the configured scan start are both off.
12. **Private keys are committed in plaintext.** Fixed in `18f6a51`: the three key files under `deployments/` are deleted, `fund_smoke_wallets.js` writes its record to `~/.autonet/keys` instead, and `.gitignore` blocks key files in `deployments/`. Git history still holds those keys, so they are burned.
13. **Stale default governor.** `atn/jurisdiction.py` hardcodes `GOVERNOR_ADDRESS = 0xD569...`, which is the 2026-07-10 e2e DAO (`deployments/shadownet-autonet-dao-e2e.json:8`), not `registry.json`'s `dao` `0x9660...`. Its docstring describes a discovery chain "Governor -> RepToken -> Registry -> RPB" (`:6-10`). Config fields claim to be "Discovered from RepToken.registryAddress()" and similar through `_discover_jurisdiction()` (`atn/config.py:285-305`), but no such function exists anywhere.
14. **`epoch_anchorer.py` docstring describes the old model.** It says one agent submits anchors "from their own keypair" and that a second submitter's revert means "I'm second" (`:11-18`, `:37-40`). In the code, `anchor` is governor- or anchorer-only and goes through DAO proposals (`:312-313`). `signatures` and `signer_keys` parameters are accepted and ignored (`:330-336`).
15. **Rounding versus truncation.** `authoritative_submitter.py` says the float is "truncated" at the chain boundary (`:32-35`); `scale_mint` rounds (`mint_merkle.py:50-56`). Its anchor tuple comment lists only 10 fields (`:210-213`).
16. **`voice_state.py` docstring describes the pool source wrongly.** It says the emission pool comes from "ServiceFee logs" (`:1196-1198`), but the code reads `windowBurned` (`:1297-1302`). The `ServiceFee` event in `_VOICE_ABI` (`:181-192`) is unused there. The module docstring mentions "RepToken vote checkpoints" (`:43`) where it actually reads balances.
17. **Possible pool mismatch on the pinned path.** The pinned path prices the pool with `burnedBetween(prevCutoff, cutoff)`, computed from checkpoint lookups (`voice_state.py:1292`). On chain, `windowBurned` is measured from `burnedAtLatestCutoff`, which is frozen when the previous anchor landed (`Substrate.sol:305-307`). Fees burned in a cutoff's own second after its anchor tx are counted differently by the two.
18. **Unbounded log scan in sanctions.** `sanctions.py` scans `OwnerBound` from block 0 with no range splitting (`atn/sanctions.py:52-53`), unlike `_get_logs`, which exists because public RPCs cap `eth_getLogs` ranges (`voice_state.py:241-264`).
19. **Inconsistent chain-id defaults.** Python configs default to 31337 (`epoch_anchorer.py:92`, `authoritative_submitter.py:85`, `chain_submission_driver.py:82`), while the Hardhat network is 1337 (`hardhat.config.js:43-46`). `EpochAnchorer._submit` signs with `config.chain_id` (`:437`) even though `chain_id()` reads the node (`:204-216`).
20. **Code-integrity check result is discarded.** `_cache.validate()` returns a bool that `start()` ignores, inside `except: pass` (`atn/autonet_service.py:935-945`). The "core-protected... detected by the on-chain integrity check" claim (`atn/jurisdiction.py:12-14`) is therefore not enforced.
21. **Pre-substrate dead code.**
    - `nodes/common/contracts.py` loads only `Substrate` but exposes `set_mature_model` and `get_mature_model` on "RPB" (`:201-216`) and reads `deployment-addresses.json` (`:25-27`), which does not exist.
    - `atn/autonet_service.py:1739-1867` and `:1929-2140` call an `Autonet` contract (`createUserContract`, `claimParticipantReward`, proposals) whose ABI no longer exists. `ws_server` marks one such handler deprecated (`atn/ws_server.py:96-101`).
    - `nodes/service.py:493-500` reads `RPB.modelVersion` and similar.
    - `economy_address` is resolved and displayed but never called.
22. **Possible stall when an election is won by an invalid option.** Fixed in `8a7241d`: if the proposal is still Queued `EXECUTE_GRACE_S` (600 s) past its eta after an execute try, `ElectionAnchorer` treats the attempt as failed and re-posts it as the next attempt, instead of waiting for the driver's 9-day give-up (`anchor_election.py:39-42`, `:128-146`; `tests/test_election_stuck_execute.py`). `AnchorElection` still does not validate option content at registration (`AnchorElection.sol:198-220`), so a bad winner costs one attempt plus the grace period.

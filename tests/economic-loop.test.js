const { expect } = require("chai");
const { ethers } = require("hardhat");

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

const e18 = (n) => ethers.parseEther(String(n));

async function mineBlocks(n) {
  for (let i = 0; i < n; i++) {
    await ethers.provider.send("evm_mine", []);
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// END-TO-END ECONOMIC LOOP TEST
//
// Deploys a full jurisdiction on Hardhat local, exercises:
//   1. Agent registration (root + child agents)
//   2. Training lifecycle: propose → commit → reveal → vote → finalize
//   3. Alignment-weighted training rewards via Yuma consensus
//   4. Share purchases (investments) and dividend claims
//   5. Inference pricing unaffected by alignment
//   6. Task budget funding and disbursement
// ═══════════════════════════════════════════════════════════════════════════════

describe("Full Economic Loop — E2E", function () {
  this.timeout(120_000);

  let rpb, rpbAddr;
  let staking, taskContract, resultsRewards;
  let registry;
  let owner, proposer, solver1, solver2, coord1, coord2, investor1, investor2;

  const PROPOSER_STAKE = e18(100);
  const SOLVER_STAKE = e18(50);
  const COORDINATOR_STAKE = e18(500);

  beforeEach(async function () {
    [owner, proposer, solver1, solver2, coord1, coord2, investor1, investor2] =
      await ethers.getSigners();

    // ── Step 1: Deploy jurisdiction (Registry) ──
    const Registry = await ethers.getContractFactory("Registry");
    registry = await Registry.deploy(owner.address, owner.address);
    await registry.waitForDeployment();
    await registry.setJurisdictionAddress(owner.address);

    // ── Step 2: Deploy RPB — the single jurisdiction contract + ERC20 token ──
    const RPB = await ethers.getContractFactory("RPB");
    rpb = await RPB.deploy(await registry.getAddress());
    await rpb.waitForDeployment();
    rpbAddr = await rpb.getAddress();

    // Verify RPB is the token
    expect(await rpb.name()).to.equal("Autonoma Token");
    expect(await rpb.symbol()).to.equal("ATN");
    expect(await rpb.totalSupply()).to.equal(e18(1_000_000_000));

    // ── Step 3: Deploy supporting contracts ──
    const Staking = await ethers.getContractFactory("ParticipantStaking");
    staking = await Staking.deploy(rpbAddr, owner.address);
    await staking.waitForDeployment();

    const TaskContract = await ethers.getContractFactory("TaskContract");
    taskContract = await TaskContract.deploy(
      await staking.getAddress(),
      owner.address,
      rpbAddr
    );
    await taskContract.waitForDeployment();

    const ResultsRewards = await ethers.getContractFactory("ResultsRewards");
    resultsRewards = await ResultsRewards.deploy(
      await taskContract.getAddress(),
      await staking.getAddress(),
      owner.address
    );
    await resultsRewards.waitForDeployment();

    // ── Step 4: Wire contracts ──
    await taskContract.setResultsRewardsContract(await resultsRewards.getAddress());
    await resultsRewards.setRPBContract(rpbAddr);
    await staking.setAuthorizedSlasher(await resultsRewards.getAddress(), true);
    await rpb.setAuthorizedDisburser(await resultsRewards.getAddress(), true);

    // ── Step 5: Fund task budget ──
    await rpb.fundTaskBudget(e18(50000));

    // ── Step 6: Distribute tokens ──
    const amounts = e18(100_000);
    for (const s of [proposer, solver1, solver2, coord1, coord2, investor1, investor2]) {
      await rpb.transfer(s.address, amounts);
    }

    // ── Step 7: Approve staking ──
    for (const s of [proposer, solver1, solver2, coord1, coord2]) {
      await rpb.connect(s).approve(await staking.getAddress(), amounts);
    }

    // ── Step 8: Stake participants ──
    await staking.connect(proposer).stake(1, PROPOSER_STAKE);   // PROPOSER
    await staking.connect(solver1).stake(2, SOLVER_STAKE);       // SOLVER
    await staking.connect(solver2).stake(2, SOLVER_STAKE);       // SOLVER
    await staking.connect(coord1).stake(3, COORDINATOR_STAKE);   // COORDINATOR
    await staking.connect(coord2).stake(3, COORDINATOR_STAKE);   // COORDINATOR
  });

  // ─────────────────────────────────────────────────────────────────────────
  // Test 1: Agent Registration
  // ─────────────────────────────────────────────────────────────────────────
  describe("Agent Registration", function () {
    it("registers root and child agents", async function () {
      // Root agent
      const rootLineage = ethers.keccak256(ethers.toUtf8Bytes("root-agent-1"));
      const rootAlignment = ethers.keccak256(ethers.toUtf8Bytes("alignment-v1"));
      await rpb.connect(solver1).registerAgent(
        rootLineage, rootAlignment, ethers.ZeroAddress, ethers.ZeroAddress
      );
      expect(await rpb.isRegistered(solver1.address)).to.be.true;
      expect(await rpb.totalRegistered()).to.equal(1);

      // Child agent
      const childLineage = ethers.keccak256(ethers.toUtf8Bytes("child-agent-1"));
      await rpb.connect(solver2).registerAgent(
        childLineage, rootAlignment, solver1.address, ethers.ZeroAddress
      );
      expect(await rpb.isRegistered(solver2.address)).to.be.true;
      expect(await rpb.totalRegistered()).to.equal(2);

      // Verify parent-child relationship
      const children = await rpb.getChildren(solver1.address);
      expect(children.length).to.equal(1);
      expect(children[0]).to.equal(solver2.address);
    });
  });

  // ─────────────────────────────────────────────────────────────────────────
  // Test 2: Full Training Lifecycle with Alignment-Weighted Rewards
  // ─────────────────────────────────────────────────────────────────────────
  describe("Training with Alignment-Weighted Rewards", function () {
    it("high alignment = higher solver reward", async function () {
      // ── Propose task ──
      const specHash = ethers.keccak256(ethers.toUtf8Bytes("training-spec-1"));
      const groundTruthCid = "QmGroundTruth_HighAlignment";
      const groundTruthHash = ethers.keccak256(ethers.toUtf8Bytes(groundTruthCid));
      const solverReward = e18(100);
      const proposerReward = e18(20);

      await taskContract.connect(proposer).proposeTask(
        rpbAddr, specHash, groundTruthHash, proposerReward, solverReward
      );
      const taskId = 1;

      // ── Solver commits solution ──
      const solutionCid = "QmSolution_HighAlignment";
      const solutionHash = ethers.keccak256(ethers.toUtf8Bytes(solutionCid));
      await taskContract.connect(solver1).commitSolution(taskId, solutionHash, ethers.ZeroHash);

      // ── Reveals ──
      await resultsRewards.connect(proposer).revealGroundTruth(taskId, groundTruthCid);
      await resultsRewards.connect(solver1).revealSolution(taskId, solutionCid);

      // ── Coordinators vote with HIGH alignment (9000 = 90%) and high correctness ──
      await resultsRewards.connect(coord1).submitVote(
        taskId, solver1.address, true, 90, 9000, "QmReport1"
      );
      await resultsRewards.connect(coord2).submitVote(
        taskId, solver1.address, true, 85, 9000, "QmReport2"
      );

      // ── Finalize voting (Yuma consensus) ──
      const solverBalBefore = await rpb.balanceOf(solver1.address);
      const proposerBalBefore = await rpb.balanceOf(proposer.address);

      await resultsRewards.finalizeVoting(taskId, solver1.address);

      const solverBalAfter = await rpb.balanceOf(solver1.address);
      const proposerBalAfter = await rpb.balanceOf(proposer.address);

      // Check consensus result
      const result = await resultsRewards.getConsensusResult(taskId, solver1.address);
      expect(result.finalized).to.be.true;
      expect(result.consensusCorrect).to.be.true;
      expect(result.consensusAlignmentScore).to.equal(9000);

      // Solver reward formula: base * (correctness/100) * (alignment/10000)
      // correctness ~87 (stake-weighted avg of 90 and 85), alignment = 9000
      // scaledReward = 100e18 * 87 * 9000 / 1_000_000 = 78.3e18
      const solverGain = solverBalAfter - solverBalBefore;
      expect(solverGain).to.be.gt(e18(70)); // At least 70 ATN
      expect(solverGain).to.be.lt(e18(90)); // At most 90 ATN

      // Proposer reward is NOT alignment-gated
      const proposerGain = proposerBalAfter - proposerBalBefore;
      expect(proposerGain).to.equal(proposerReward); // Full 20 ATN
    });

    it("low alignment = lower solver reward", async function () {
      // Same task but with LOW alignment
      const specHash = ethers.keccak256(ethers.toUtf8Bytes("training-spec-2"));
      const groundTruthCid = "QmGroundTruth_LowAlignment";
      const groundTruthHash = ethers.keccak256(ethers.toUtf8Bytes(groundTruthCid));
      const solverReward = e18(100);
      const proposerReward = e18(20);

      await taskContract.connect(proposer).proposeTask(
        rpbAddr, specHash, groundTruthHash, proposerReward, solverReward
      );
      const taskId = 1;

      const solutionCid = "QmSolution_LowAlignment";
      const solutionHash = ethers.keccak256(ethers.toUtf8Bytes(solutionCid));
      await taskContract.connect(solver1).commitSolution(taskId, solutionHash, ethers.ZeroHash);

      await resultsRewards.connect(proposer).revealGroundTruth(taskId, groundTruthCid);
      await resultsRewards.connect(solver1).revealSolution(taskId, solutionCid);

      // Coordinators vote with LOW alignment (2000 = 20%) but high correctness
      await resultsRewards.connect(coord1).submitVote(
        taskId, solver1.address, true, 90, 2000, "QmReport1"
      );
      await resultsRewards.connect(coord2).submitVote(
        taskId, solver1.address, true, 85, 2000, "QmReport2"
      );

      const solverBalBefore = await rpb.balanceOf(solver1.address);
      await resultsRewards.finalizeVoting(taskId, solver1.address);
      const solverBalAfter = await rpb.balanceOf(solver1.address);

      const result = await resultsRewards.getConsensusResult(taskId, solver1.address);
      expect(result.consensusAlignmentScore).to.equal(2000);

      // scaledReward = 100e18 * 87 * 2000 / 1_000_000 = ~17.4e18
      const solverGain = solverBalAfter - solverBalBefore;
      expect(solverGain).to.be.gt(e18(15));
      expect(solverGain).to.be.lt(e18(20));
    });

    it("zero alignment = zero solver reward", async function () {
      const specHash = ethers.keccak256(ethers.toUtf8Bytes("training-spec-3"));
      const groundTruthCid = "QmGroundTruth_ZeroAlignment";
      const groundTruthHash = ethers.keccak256(ethers.toUtf8Bytes(groundTruthCid));

      await taskContract.connect(proposer).proposeTask(
        rpbAddr, specHash, groundTruthHash, e18(20), e18(100)
      );
      const taskId = 1;

      const solutionCid = "QmSolution_ZeroAlignment";
      const solutionHash = ethers.keccak256(ethers.toUtf8Bytes(solutionCid));
      await taskContract.connect(solver1).commitSolution(taskId, solutionHash, ethers.ZeroHash);

      await resultsRewards.connect(proposer).revealGroundTruth(taskId, groundTruthCid);
      await resultsRewards.connect(solver1).revealSolution(taskId, solutionCid);

      // Zero alignment
      await resultsRewards.connect(coord1).submitVote(
        taskId, solver1.address, true, 90, 0, "QmReport1"
      );
      await resultsRewards.connect(coord2).submitVote(
        taskId, solver1.address, true, 85, 0, "QmReport2"
      );

      const solverBalBefore = await rpb.balanceOf(solver1.address);
      await resultsRewards.finalizeVoting(taskId, solver1.address);
      const solverBalAfter = await rpb.balanceOf(solver1.address);

      // Zero alignment → zero reward
      expect(solverBalAfter - solverBalBefore).to.equal(0);
    });
  });

  // ─────────────────────────────────────────────────────────────────────────
  // Test 3: Share Purchases, Dividends, and Inference Revenue
  // ─────────────────────────────────────────────────────────────────────────
  describe("Investments, Shares, and Dividends", function () {
    it("investors purchase shares and claim dividends from inference revenue", async function () {
      // ── Deploy a payment token (simulates external stablecoin) ──
      const MockToken = await ethers.getContractFactory("ATNToken");
      const payToken = await MockToken.deploy(owner.address, 1_000_000);
      await payToken.waitForDeployment();
      const payTokenAddr = await payToken.getAddress();

      // Register parity for this token in the Registry (1:1)
      const parityKey = "jurisdiction.parity." + payTokenAddr.toLowerCase().replace("0x", "0x");
      // Need to use proper hex format
      const { Strings } = ethers;
      const parityKeyFull = `jurisdiction.parity.${payTokenAddr.toLowerCase()}`;
      await registry.editRegistry(parityKeyFull, "1");

      // ── Investor 1 buys shares with ATN (RPB's own token, parity = 1) ──
      await rpb.connect(investor1).approve(rpbAddr, e18(5000));
      await rpb.connect(investor1).purchaseShares(rpbAddr, e18(5000));
      expect(await rpb.getShareBalance(investor1.address)).to.equal(e18(5000));

      // ── Investor 2 buys shares ──
      await rpb.connect(investor2).approve(rpbAddr, e18(3000));
      await rpb.connect(investor2).purchaseShares(rpbAddr, e18(3000));
      expect(await rpb.getShareBalance(investor2.address)).to.equal(e18(3000));

      // Total shares = 8000
      expect(await rpb.totalShares()).to.equal(e18(8000));

      // ── Generate inference revenue ──
      // RPB.recordInference: provider gets paid, shareholder pool gets portion
      // Register provider as agent first
      const providerLineage = ethers.keccak256(ethers.toUtf8Bytes("provider-lineage"));
      await rpb.connect(solver1).registerAgent(
        providerLineage,
        ethers.keccak256(ethers.toUtf8Bytes("provider-alignment")),
        ethers.ZeroAddress,
        ethers.ZeroAddress
      );

      // Fund RPB with ATN for inference payments (owner sends to RPB)
      await rpb.transfer(rpbAddr, e18(10000));

      // Record inference (owner is contract owner, can call recordInference)
      // 100 units, 1000 ATN cost, paid in ATN (address(this))
      await rpb.recordInference(
        investor1.address, // requester
        solver1.address,   // provider
        100,               // units
        rpbAddr,           // token = RPB itself
        e18(1000)          // cost in raw token units
      );

      expect(await rpb.totalInferenceUnits()).to.equal(100);
      expect(await rpb.totalInferenceRevenue()).to.equal(e18(1000));

      // ── Check dividends ──
      // shareholderShare = 2500 bps (25%)
      // totalDividendable = 1000 * 2500 / 10000 = 250 ATN
      // investor1 owns 5000/8000 = 62.5% → 156.25 ATN
      // investor2 owns 3000/8000 = 37.5% → 93.75 ATN
      const inv1Dividends = await rpb.getUnclaimedDividends(investor1.address);
      const inv2Dividends = await rpb.getUnclaimedDividends(investor2.address);

      expect(inv1Dividends).to.be.gt(e18(155));
      expect(inv1Dividends).to.be.lt(e18(158));
      expect(inv2Dividends).to.be.gt(e18(93));
      expect(inv2Dividends).to.be.lt(e18(95));

      // ── Claim dividends ──
      const inv1BalBefore = await rpb.balanceOf(investor1.address);
      await rpb.connect(investor1).claimDividends(rpbAddr);
      const inv1BalAfter = await rpb.balanceOf(investor1.address);
      expect(inv1BalAfter - inv1BalBefore).to.be.gt(e18(155));

      // Investor2 claims
      const inv2BalBefore = await rpb.balanceOf(investor2.address);
      await rpb.connect(investor2).claimDividends(rpbAddr);
      const inv2BalAfter = await rpb.balanceOf(investor2.address);
      expect(inv2BalAfter - inv2BalBefore).to.be.gt(e18(93));

      // Cannot double-claim
      await expect(
        rpb.connect(investor1).claimDividends(rpbAddr)
      ).to.be.revertedWithCustomError(rpb, "NoDividendsToClaim");
    });
  });

  // ─────────────────────────────────────────────────────────────────────────
  // Test 4: Training Reward Pool and Claims
  // ─────────────────────────────────────────────────────────────────────────
  describe("Training Reward Pool and Claims", function () {
    it("fund pool → record training → claim rewards", async function () {
      // Fund training pool with ATN
      await rpb.approve(rpbAddr, e18(5000));
      await rpb.fundTrainingPool(rpbAddr, e18(5000));
      expect(await rpb.trainingRewardPool()).to.equal(e18(5000));

      // Record training contributions (owner records)
      await rpb.connect(solver1).registerAgent(
        ethers.keccak256(ethers.toUtf8Bytes("solver1-lineage")),
        ethers.randomBytes(32),
        ethers.ZeroAddress,
        ethers.ZeroAddress
      );

      // Solver1 contributes 70 units, solver2 contributes 30 units
      await rpb.recordTraining(solver1.address, 70);

      // Check training tokens
      expect(await rpb.getTrainingTokens(solver1.address)).to.equal(70);
      expect(await rpb.totalTrainingTokens()).to.equal(70);

      // Claim reward (solver1 gets 70/70 * 5000 = 5000 ATN since they're the only contributor)
      const unclaimed = await rpb.getUnclaimedRewards(solver1.address);
      expect(unclaimed).to.equal(e18(5000));

      const balBefore = await rpb.balanceOf(solver1.address);
      await rpb.connect(solver1).claimTrainingReward(rpbAddr);
      const balAfter = await rpb.balanceOf(solver1.address);
      expect(balAfter - balBefore).to.equal(e18(5000));

      // Cannot double-claim
      await expect(
        rpb.connect(solver1).claimTrainingReward(rpbAddr)
      ).to.be.revertedWithCustomError(rpb, "NoRewardsToClaim");
    });
  });

  // ─────────────────────────────────────────────────────────────────────────
  // Test 5: Inference Pricing is NOT Affected by Alignment
  // ─────────────────────────────────────────────────────────────────────────
  describe("Inference Pricing Independence from Alignment", function () {
    it("price depends only on shares, not on alignment score", async function () {
      // Set base inference price
      await rpb.setMatureModel("QmModelCid", e18(10));

      // Set discount tiers
      await rpb.setDiscountTiers([
        { ptThreshold: e18(1000), discountPermille: 100 },  // 10% off for 1000+ shares
        { ptThreshold: e18(5000), discountPermille: 200 },  // 20% off for 5000+ shares
      ]);

      // User with no shares pays full price
      expect(await rpb.getEffectivePrice(investor1.address)).to.equal(e18(10));

      // User buys 1000 shares → 10% discount
      await rpb.connect(investor1).approve(rpbAddr, e18(1000));
      await rpb.connect(investor1).purchaseShares(rpbAddr, e18(1000));
      expect(await rpb.getEffectivePrice(investor1.address)).to.equal(e18(9)); // 10 * 0.9

      // User buys more (total 5000+) → 20% discount
      await rpb.connect(investor1).approve(rpbAddr, e18(4000));
      await rpb.connect(investor1).purchaseShares(rpbAddr, e18(4000));
      expect(await rpb.getEffectivePrice(investor1.address)).to.equal(e18(8)); // 10 * 0.8

      // Alignment score (registered via agents) does NOT affect pricing
      // Register investor1 as agent with alignment
      const lineage = ethers.keccak256(ethers.toUtf8Bytes("investor-lineage"));
      await rpb.connect(investor1).registerAgent(
        lineage,
        ethers.keccak256(ethers.toUtf8Bytes("max-alignment")),
        ethers.ZeroAddress,
        ethers.ZeroAddress
      );

      // Price unchanged after registration
      expect(await rpb.getEffectivePrice(investor1.address)).to.equal(e18(8));

      // Update alignment — price still unchanged
      await rpb.connect(investor1).updateAlignment(
        ethers.keccak256(ethers.toUtf8Bytes("different-alignment"))
      );
      expect(await rpb.getEffectivePrice(investor1.address)).to.equal(e18(8));
    });
  });

  // ─────────────────────────────────────────────────────────────────────────
  // Test 6: Full Pipeline — Training → Reward → Invest → Inference → Dividend
  // ─────────────────────────────────────────────────────────────────────────
  describe("Full Economic Pipeline", function () {
    it("complete cycle: train → earn → invest → serve → dividend", async function () {
      // ── A: Training produces rewards ──
      const specHash = ethers.keccak256(ethers.toUtf8Bytes("pipeline-spec"));
      const groundTruthCid = "QmGroundTruthPipeline";
      const groundTruthHash = ethers.keccak256(ethers.toUtf8Bytes(groundTruthCid));

      await taskContract.connect(proposer).proposeTask(
        rpbAddr, specHash, groundTruthHash, e18(10), e18(50)
      );

      const solutionCid = "QmSolutionPipeline";
      const solutionHash = ethers.keccak256(ethers.toUtf8Bytes(solutionCid));
      await taskContract.connect(solver1).commitSolution(1, solutionHash, ethers.ZeroHash);

      await resultsRewards.connect(proposer).revealGroundTruth(1, groundTruthCid);
      await resultsRewards.connect(solver1).revealSolution(1, solutionCid);

      // High alignment, high correctness → good reward
      await resultsRewards.connect(coord1).submitVote(1, solver1.address, true, 95, 9500, "QmR1");
      await resultsRewards.connect(coord2).submitVote(1, solver1.address, true, 92, 9500, "QmR2");

      const solverBalBefore = await rpb.balanceOf(solver1.address);
      await resultsRewards.finalizeVoting(1, solver1.address);
      const solverEarned = (await rpb.balanceOf(solver1.address)) - solverBalBefore;

      // Solver should earn something significant
      expect(solverEarned).to.be.gt(e18(40));

      // ── B: Solver invests earnings as shares ──
      await rpb.connect(solver1).approve(rpbAddr, solverEarned);
      await rpb.connect(solver1).purchaseShares(rpbAddr, solverEarned);
      expect(await rpb.getShareBalance(solver1.address)).to.equal(solverEarned);

      // ── C: Register provider agent ──
      const providerLineage = ethers.keccak256(ethers.toUtf8Bytes("solver1-provider"));
      await rpb.connect(solver1).registerAgent(
        providerLineage,
        ethers.keccak256(ethers.toUtf8Bytes("alignment-hash")),
        ethers.ZeroAddress,
        ethers.ZeroAddress
      );

      // ── D: Inference generates revenue ──
      await rpb.transfer(rpbAddr, e18(10000)); // Fund RPB for inference payments
      await rpb.recordInference(
        investor1.address,  // requester
        solver1.address,    // provider
        200,                // units
        rpbAddr,            // token = RPB
        e18(2000)           // cost
      );

      // ── E: Solver claims dividends from inference revenue ──
      const dividends = await rpb.getUnclaimedDividends(solver1.address);
      // shareholderShare = 2500 bps → dividendable = 2000 * 0.25 = 500
      // solver1 owns 100% of shares → full 500 ATN
      expect(dividends).to.equal(e18(500));

      const balBeforeClaim = await rpb.balanceOf(solver1.address);
      await rpb.connect(solver1).claimDividends(rpbAddr);
      const dividendReceived = (await rpb.balanceOf(solver1.address)) - balBeforeClaim;
      expect(dividendReceived).to.equal(e18(500));
    });
  });

  // ─────────────────────────────────────────────────────────────────────────
  // Test 7: Revenue Split Verification
  // ─────────────────────────────────────────────────────────────────────────
  describe("Revenue Split", function () {
    it("inference revenue splits correctly: 60% provider, 25% shareholders, 15% treasury", async function () {
      // Register provider
      const lineage = ethers.keccak256(ethers.toUtf8Bytes("rev-split-provider"));
      await rpb.connect(solver1).registerAgent(
        lineage, ethers.randomBytes(32), ethers.ZeroAddress, ethers.ZeroAddress
      );

      // Fund RPB for payments
      await rpb.transfer(rpbAddr, e18(10000));

      const providerBefore = await rpb.balanceOf(solver1.address);
      const daoBefore = await rpb.balanceOf(owner.address); // owner = dao in test

      await rpb.recordInference(
        investor1.address, solver1.address, 10, rpbAddr, e18(1000)
      );

      const providerAfter = await rpb.balanceOf(solver1.address);
      const daoAfter = await rpb.balanceOf(owner.address);

      // Provider gets 60%
      expect(providerAfter - providerBefore).to.equal(e18(600));
      // DAO/treasury gets 15%
      expect(daoAfter - daoBefore).to.equal(e18(150));
      // Remaining 25% stays in RPB contract for shareholders
      const shareholderPool = await rpb.getTokenPoolBalance(rpbAddr);
      expect(shareholderPool).to.equal(e18(250));
    });
  });

  // ─────────────────────────────────────────────────────────────────────────
  // Test 8: Coordinator Bond Dynamics
  // ─────────────────────────────────────────────────────────────────────────
  describe("Coordinator Bond EMA", function () {
    it("coordinators who align with consensus build stronger bonds", async function () {
      const specHash = ethers.keccak256(ethers.toUtf8Bytes("bond-test-spec"));
      const gtCid = "QmBondTestGT";
      const gtHash = ethers.keccak256(ethers.toUtf8Bytes(gtCid));

      await taskContract.connect(proposer).proposeTask(
        rpbAddr, specHash, gtHash, e18(10), e18(50)
      );

      const solCid = "QmBondTestSol";
      const solHash = ethers.keccak256(ethers.toUtf8Bytes(solCid));
      await taskContract.connect(solver1).commitSolution(1, solHash, ethers.ZeroHash);

      await resultsRewards.connect(proposer).revealGroundTruth(1, gtCid);
      await resultsRewards.connect(solver1).revealSolution(1, solCid);

      // Both vote correct (consensus = correct)
      await resultsRewards.connect(coord1).submitVote(1, solver1.address, true, 85, 8000, "QmR");
      await resultsRewards.connect(coord2).submitVote(1, solver1.address, true, 90, 8000, "QmR");

      await resultsRewards.finalizeVoting(1, solver1.address);

      // Both coordinators aligned → bond increases
      const bond1 = await resultsRewards.getCoordinatorBond(coord1.address);
      const bond2 = await resultsRewards.getCoordinatorBond(coord2.address);

      expect(bond1.totalVotes).to.equal(1);
      expect(bond1.correctVotes).to.equal(1);
      expect(bond1.emaBondStrength).to.be.gt(0);
      expect(bond2.emaBondStrength).to.be.gt(0);
    });
  });
});

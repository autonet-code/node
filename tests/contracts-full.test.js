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

async function increaseTime(seconds) {
  await ethers.provider.send("evm_increaseTime", [seconds]);
  await ethers.provider.send("evm_mine", []);
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared deployment fixture
// ─────────────────────────────────────────────────────────────────────────────

async function deployAll() {
  const [owner, proposer, solver1, solver2, coord1, coord2, coord3, aggregator1, user1, user2] =
    await ethers.getSigners();

  // ── ATN Token ──
  const ATNToken = await ethers.getContractFactory("ATNToken");
  const atnToken = await ATNToken.deploy(owner.address, 1_000_000_000);
  await atnToken.waitForDeployment();

  // ── ParticipantStaking ──
  const Staking = await ethers.getContractFactory("ParticipantStaking");
  const staking = await Staking.deploy(await atnToken.getAddress(), owner.address);
  await staking.waitForDeployment();

  // ── TaskContract ──
  const TaskContract = await ethers.getContractFactory("TaskContract");
  const taskContract = await TaskContract.deploy(
    await staking.getAddress(),
    owner.address,
    await atnToken.getAddress()
  );
  await taskContract.waitForDeployment();

  // ── ResultsRewards ──
  const ResultsRewards = await ethers.getContractFactory("ResultsRewards");
  const resultsRewards = await ResultsRewards.deploy(
    await taskContract.getAddress(),
    await staking.getAddress(),
    owner.address
  );
  await resultsRewards.waitForDeployment();

  // ── ForcedErrorRegistry ──
  const ForcedErrorRegistry = await ethers.getContractFactory("ForcedErrorRegistry");
  const forcedErrorRegistry = await ForcedErrorRegistry.deploy(
    await atnToken.getAddress(),
    await staking.getAddress(),
    owner.address
  );
  await forcedErrorRegistry.waitForDeployment();

  // ── ModelShardRegistry ──
  const ModelShardRegistry = await ethers.getContractFactory("ModelShardRegistry");
  const modelShardRegistry = await ModelShardRegistry.deploy(
    await staking.getAddress(),
    owner.address
  );
  await modelShardRegistry.waitForDeployment();

  // ── AutonetDAO ──
  const AutonetDAO = await ethers.getContractFactory("AutonetDAO");
  const dao = await AutonetDAO.deploy(await atnToken.getAddress(), owner.address);
  await dao.waitForDeployment();

  // ── CapabilityScorecard ──
  const CapabilityScorecard = await ethers.getContractFactory("CapabilityScorecard");
  const scorecard = await CapabilityScorecard.deploy(owner.address);
  await scorecard.waitForDeployment();

  // ── GuildRegistry ──
  const GuildRegistry = await ethers.getContractFactory("GuildRegistry");
  const guildRegistry = await GuildRegistry.deploy(
    await atnToken.getAddress(),
    owner.address,
    e18(500),  // minGuildStake
    e18(50)    // minMemberStake
  );
  await guildRegistry.waitForDeployment();

  // ── AnchorBridge ──
  const AnchorBridge = await ethers.getContractFactory("AnchorBridge");
  const anchorBridge = await AnchorBridge.deploy(await atnToken.getAddress());
  await anchorBridge.waitForDeployment();

  // ── DisputeManager ──
  const DisputeManager = await ethers.getContractFactory("DisputeManager");
  const disputeManager = await DisputeManager.deploy(await atnToken.getAddress());
  await disputeManager.waitForDeployment();

  // ── Registry (mock jurisdiction) ──
  const Registry = await ethers.getContractFactory("Registry");
  const registry = await Registry.deploy(owner.address, owner.address);
  await registry.waitForDeployment();

  // Set jurisdiction address so RPB constructor doesn't revert
  await registry.setJurisdictionAddress(owner.address);

  // ── RPBFactory ──
  const RPBFactory = await ethers.getContractFactory("RPBFactory");
  const rpbFactory = await RPBFactory.deploy();
  await rpbFactory.waitForDeployment();

  // ── Deploy a default RPB for reward/task tests ──
  const rpbTx = await rpbFactory.createRPB(
    await registry.getAddress(),
    await atnToken.getAddress()
  );
  const rpbReceipt = await rpbTx.wait();
  const rpbEvent = rpbReceipt.logs.find(l => l.fragment && l.fragment.name === "RPBDeployed");
  const rpbAddr = rpbEvent.args[0];
  const rpb = await ethers.getContractAt("RPB", rpbAddr);

  // ── InferenceProviderFactory ──
  const InferenceProviderFactory = await ethers.getContractFactory("InferenceProviderFactory");
  const inferenceFactory = await InferenceProviderFactory.deploy(
    await staking.getAddress(),
    owner.address
  );
  await inferenceFactory.waitForDeployment();

  // ── EvolutionProposal ──
  const EvolutionProposal = await ethers.getContractFactory("EvolutionProposal");
  const evolutionProposal = await EvolutionProposal.deploy(
    await atnToken.getAddress(),
    await registry.getAddress(),
    owner.address // timelock = owner for testing
  );
  await evolutionProposal.waitForDeployment();

  // ── Wire contracts together ──
  await taskContract.setResultsRewardsContract(await resultsRewards.getAddress());
  await resultsRewards.setRPBContract(rpbAddr);
  await staking.setAuthorizedSlasher(await resultsRewards.getAddress(), true);
  await staking.setAuthorizedSlasher(await forcedErrorRegistry.getAddress(), true);

  // Authorize ResultsRewards to disburse from RPB budget
  await rpb.setAuthorizedDisburser(await resultsRewards.getAddress(), true);

  // Fund RPB task budget (owner sends ATN to RPB)
  await atnToken.approve(rpbAddr, e18(100000));
  await rpb.fundTaskBudget(e18(10000));

  // ── Distribute tokens ──
  const testAmount = e18(100_000);
  for (const s of [proposer, solver1, solver2, coord1, coord2, coord3, aggregator1, user1, user2]) {
    await atnToken.transfer(s.address, testAmount);
  }

  // ── Pre-approve staking for common roles ──
  for (const s of [proposer, solver1, solver2, coord1, coord2, coord3, aggregator1]) {
    await atnToken.connect(s).approve(await staking.getAddress(), testAmount);
  }

  return {
    atnToken, staking, taskContract, resultsRewards,
    forcedErrorRegistry, modelShardRegistry, dao, scorecard,
    guildRegistry, anchorBridge, disputeManager, inferenceFactory,
    registry, rpbFactory, rpb, evolutionProposal,
    owner, proposer, solver1, solver2, coord1, coord2, coord3,
    aggregator1, user1, user2,
  };
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST SUITES
// ═══════════════════════════════════════════════════════════════════════════════

describe("Autonet Contracts — Full Test Suite", function () {
  this.timeout(120_000);

  // ─────────────────────────────────────────────────────────────────────────
  // 1. ATNToken
  // ─────────────────────────────────────────────────────────────────────────
  describe("ATNToken", function () {
    let f;
    beforeEach(async () => { f = await deployAll(); });

    it("mints initial supply to holder", async function () {
      // 1B * 1e18
      expect(await f.atnToken.balanceOf(f.owner.address)).to.be.gt(0);
    });

    it("only DAO can mint", async function () {
      await expect(
        f.atnToken.connect(f.user1).mint(f.user1.address, e18(1))
      ).to.be.revertedWith("ATNToken: caller is not DAO or authorized minter");
    });

    it("DAO can authorize minter", async function () {
      await f.atnToken.setAuthorizedMinter(f.user1.address, true);
      await f.atnToken.connect(f.user1).mint(f.user2.address, e18(100));
      expect(await f.atnToken.balanceOf(f.user2.address)).to.equal(e18(100_100));
    });

    it("only DAO can change DAO address", async function () {
      await expect(
        f.atnToken.connect(f.user1).setDaoAddress(f.user1.address)
      ).to.be.revertedWith("ATNToken: caller is not current DAO");
    });

    it("setDaoAddress works", async function () {
      await f.atnToken.setDaoAddress(f.user1.address);
      expect(await f.atnToken.daoAddress()).to.equal(f.user1.address);
    });

    it("rejects zero address for minter and DAO", async function () {
      await expect(
        f.atnToken.setAuthorizedMinter(ethers.ZeroAddress, true)
      ).to.be.revertedWith("ATNToken: zero minter address");
      await expect(
        f.atnToken.setDaoAddress(ethers.ZeroAddress)
      ).to.be.revertedWith("ATNToken: new DAO is zero");
    });
  });

  // ─────────────────────────────────────────────────────────────────────────
  // 2. ParticipantStaking
  // ─────────────────────────────────────────────────────────────────────────
  describe("ParticipantStaking", function () {
    let f;
    beforeEach(async () => { f = await deployAll(); });

    it("allows staking for all roles", async function () {
      await f.staking.connect(f.proposer).stake(1, e18(100)); // PROPOSER
      expect(await f.staking.isActiveParticipant(f.proposer.address, 1)).to.be.true;

      await f.staking.connect(f.solver1).stake(2, e18(50)); // SOLVER
      expect(await f.staking.isActiveParticipant(f.solver1.address, 2)).to.be.true;

      await f.staking.connect(f.coord1).stake(3, e18(500)); // COORDINATOR
      expect(await f.staking.isActiveParticipant(f.coord1.address, 3)).to.be.true;
    });

    it("rejects staking below minimum", async function () {
      await expect(
        f.staking.connect(f.proposer).stake(1, e18(10))
      ).to.be.revertedWith("Below minimum stake");
    });

    it("rejects double staking", async function () {
      await f.staking.connect(f.proposer).stake(1, e18(100));
      await expect(
        f.staking.connect(f.proposer).stake(2, e18(50))
      ).to.be.revertedWith("Already staked");
    });

    it("rejects staking with NONE role", async function () {
      await expect(
        f.staking.connect(f.proposer).stake(0, e18(100))
      ).to.be.revertedWith("Invalid role");
    });

    it("addToStake increases stake", async function () {
      await f.staking.connect(f.proposer).stake(1, e18(100));
      await f.staking.connect(f.proposer).addToStake(e18(50));
      const info = await f.staking.getStake(f.proposer.address);
      expect(info.amount).to.equal(e18(150));
    });

    it("requestUnstake + unstake after lockup", async function () {
      await f.staking.connect(f.proposer).stake(1, e18(100));
      await f.staking.connect(f.proposer).requestUnstake();

      // Should fail before lockup
      await expect(
        f.staking.connect(f.proposer).unstake()
      ).to.be.revertedWith("Still locked");

      // Advance past 7 days lockup for proposer
      await increaseTime(7 * 24 * 3600 + 1);

      const balBefore = await f.atnToken.balanceOf(f.proposer.address);
      await f.staking.connect(f.proposer).unstake();
      const balAfter = await f.atnToken.balanceOf(f.proposer.address);
      expect(balAfter - balBefore).to.equal(e18(100));

      expect(await f.staking.isActiveParticipant(f.proposer.address, 1)).to.be.false;
    });

    it("slash reduces stake and deactivates below minimum", async function () {
      await f.staking.connect(f.coord1).stake(3, e18(500));
      // Owner is governance
      await f.staking.slash(f.coord1.address, e18(450), "test slash");
      const info = await f.staking.getStake(f.coord1.address);
      expect(info.amount).to.equal(e18(50));
      // 50 < 500 minimum for coordinator, so deactivated
      expect(info.active).to.be.false;
    });

    it("unauthorized cannot slash", async function () {
      await f.staking.connect(f.proposer).stake(1, e18(100));
      await expect(
        f.staking.connect(f.user1).slash(f.proposer.address, e18(10), "nope")
      ).to.be.revertedWith("Not authorized to slash");
    });

    it("governance can update min stake and lockup", async function () {
      await f.staking.setMinStake(1, e18(200));
      expect(await f.staking.minStakeAmount(1)).to.equal(e18(200));

      await f.staking.setLockupDuration(1, 1);
      expect(await f.staking.stakeLockupDuration(1)).to.equal(1);
    });
  });

  // ─────────────────────────────────────────────────────────────────────────
  // 3. TaskContract — Ground Truth Mode
  // ─────────────────────────────────────────────────────────────────────────
  describe("TaskContract — Ground Truth", function () {
    let f;
    beforeEach(async () => {
      f = await deployAll();
      // Stake proposer & solver
      await f.staking.connect(f.proposer).stake(1, e18(100));
      await f.staking.connect(f.solver1).stake(2, e18(50));
    });

    it("proposes a task and auto-activates", async function () {
      const specHash = ethers.keccak256(ethers.toUtf8Bytes("task-spec"));
      const gtHash = ethers.keccak256(ethers.toUtf8Bytes("ground-truth"));
      const rpbAddr = await f.rpb.getAddress();
      const tx = await f.taskContract.connect(f.proposer).proposeTask(
        rpbAddr, specHash, gtHash, e18(10), e18(5)
      );
      await expect(tx).to.emit(f.taskContract, "TaskProposed");
      await expect(tx).to.emit(f.taskContract, "TaskActivated");

      const [id, rpb, prop, status] = await f.taskContract.getTask(1);
      expect(id).to.equal(1);
      expect(status).to.equal(1); // ACTIVE
    });

    it("non-proposer cannot propose", async function () {
      await expect(
        f.taskContract.connect(f.solver1).proposeTask(
          await f.rpb.getAddress(), ethers.randomBytes(32), ethers.randomBytes(32), e18(10), e18(5)
        )
      ).to.be.revertedWith("Not active Proposer");
    });

    it("solver commits solution", async function () {
      const specHash = ethers.keccak256(ethers.toUtf8Bytes("spec"));
      const gtHash = ethers.keccak256(ethers.toUtf8Bytes("gt"));
      await f.taskContract.connect(f.proposer).proposeTask(
        await f.rpb.getAddress(), specHash, gtHash, e18(10), e18(5)
      );

      const solHash = ethers.keccak256(ethers.toUtf8Bytes("my-solution"));
      const tx = await f.taskContract.connect(f.solver1).commitSolution(1, solHash);
      await expect(tx).to.emit(f.taskContract, "SolutionCommitted");

      const solvers = await f.taskContract.getCommittedSolvers(1);
      expect(solvers).to.include(f.solver1.address);
    });

    it("solver cannot commit to non-existent task", async function () {
      await expect(
        f.taskContract.connect(f.solver1).commitSolution(999, ethers.randomBytes(32))
      ).to.be.revertedWith("Task does not exist");
    });

    it("checkpoint submission and retrieval", async function () {
      const specHash = ethers.keccak256(ethers.toUtf8Bytes("spec"));
      await f.taskContract.connect(f.proposer).proposeTask(
        await f.rpb.getAddress(), specHash, ethers.ZeroHash, e18(10), e18(5)
      );

      // Submit 3 checkpoints
      for (let step = 10; step <= 30; step += 10) {
        await f.taskContract.connect(f.solver1).submitCheckpoint(
          1, step,
          ethers.keccak256(ethers.toUtf8Bytes(`weights-${step}`)),
          ethers.keccak256(ethers.toUtf8Bytes(`data-${step}`)),
          ethers.keccak256(ethers.toUtf8Bytes(`seed-${step}`))
        );
      }

      expect(await f.taskContract.getCheckpointCount(1, f.solver1.address)).to.equal(3);

      // findCheckpointAtStep
      const idx = await f.taskContract.findCheckpointAtStep(1, f.solver1.address, 25);
      expect(idx).to.equal(1); // step 20 is at index 1
    });

    it("checkpoints must be in increasing order", async function () {
      const specHash = ethers.keccak256(ethers.toUtf8Bytes("spec"));
      await f.taskContract.connect(f.proposer).proposeTask(
        await f.rpb.getAddress(), specHash, ethers.ZeroHash, e18(10), e18(5)
      );

      await f.taskContract.connect(f.solver1).submitCheckpoint(
        1, 20, ethers.randomBytes(32), ethers.randomBytes(32), ethers.randomBytes(32)
      );

      await expect(
        f.taskContract.connect(f.solver1).submitCheckpoint(
          1, 10, ethers.randomBytes(32), ethers.randomBytes(32), ethers.randomBytes(32)
        )
      ).to.be.revertedWith("Checkpoint step must be increasing");
    });
  });

  // ─────────────────────────────────────────────────────────────────────────
  // 4. TaskContract — Consensus-as-Truth (MM-Zero)
  // ─────────────────────────────────────────────────────────────────────────
  describe("TaskContract — Consensus Mode", function () {
    let f;
    beforeEach(async () => {
      f = await deployAll();
      await f.staking.connect(f.proposer).stake(1, e18(100));
      await f.staking.connect(f.solver1).stake(2, e18(50));
      await f.staking.connect(f.solver2).stake(2, e18(50));
    });

    it("proposes a consensus task with difficulty target", async function () {
      const rpbAddr = await f.rpb.getAddress();
      const tx = await f.taskContract.connect(f.proposer).proposeConsensusTask(
        rpbAddr,
        ethers.keccak256(ethers.toUtf8Bytes("consensus-spec")),
        { minSolvability: 2500, maxSolvability: 7500, peakSolvability: 5000 },
        e18(10),
        e18(5)
      );
      await expect(tx).to.emit(f.taskContract, "ConsensusTaskProposed");

      const proposal = await f.taskContract.getTaskProposal(1);
      expect(proposal.status).to.equal(12); // CONSENSUS_COLLECTING
      expect(proposal.taskMode).to.equal(1); // CONSENSUS_TRUTH
    });

    it("rejects invalid difficulty targets", async function () {
      const rpbAddr = await f.rpb.getAddress();
      // min >= max
      await expect(
        f.taskContract.connect(f.proposer).proposeConsensusTask(
          rpbAddr, ethers.randomBytes(32),
          { minSolvability: 7500, maxSolvability: 2500, peakSolvability: 5000 },
          e18(10), e18(5)
        )
      ).to.be.revertedWith("Invalid solvability range");

      // peak outside range
      await expect(
        f.taskContract.connect(f.proposer).proposeConsensusTask(
          rpbAddr, ethers.randomBytes(32),
          { minSolvability: 2500, maxSolvability: 7500, peakSolvability: 1000 },
          e18(10), e18(5)
        )
      ).to.be.revertedWith("Peak outside range");

      // max > 10000
      await expect(
        f.taskContract.connect(f.proposer).proposeConsensusTask(
          rpbAddr, ethers.randomBytes(32),
          { minSolvability: 2500, maxSolvability: 11000, peakSolvability: 5000 },
          e18(10), e18(5)
        )
      ).to.be.revertedWith("Max solvability exceeds 100%");
    });

    it("solvers submit rollouts", async function () {
      const rpbAddr = await f.rpb.getAddress();
      await f.taskContract.connect(f.proposer).proposeConsensusTask(
        rpbAddr, ethers.keccak256(ethers.toUtf8Bytes("spec")),
        { minSolvability: 2500, maxSolvability: 7500, peakSolvability: 5000 },
        e18(10), e18(5)
      );

      const answerHash = ethers.keccak256(ethers.toUtf8Bytes("answer-A"));
      await f.taskContract.connect(f.solver1).submitRollout(
        1, answerHash, ethers.randomBytes(32), 80
      );
      await f.taskContract.connect(f.solver2).submitRollout(
        1, answerHash, ethers.randomBytes(32), 70
      );

      expect(await f.taskContract.getRolloutCount(1)).to.equal(2);
      expect(await f.taskContract.isRolloutCollectionComplete(1)).to.be.true;
    });

    it("rejects duplicate rollout from same solver", async function () {
      const rpbAddr = await f.rpb.getAddress();
      await f.taskContract.connect(f.proposer).proposeConsensusTask(
        rpbAddr, ethers.randomBytes(32),
        { minSolvability: 2500, maxSolvability: 7500, peakSolvability: 5000 },
        e18(10), e18(5)
      );

      await f.taskContract.connect(f.solver1).submitRollout(
        1, ethers.randomBytes(32), ethers.randomBytes(32), 50
      );
      await expect(
        f.taskContract.connect(f.solver1).submitRollout(
          1, ethers.randomBytes(32), ethers.randomBytes(32), 50
        )
      ).to.be.revertedWith("Already submitted rollout");
    });

    it("rejects confidence > 100", async function () {
      const rpbAddr = await f.rpb.getAddress();
      await f.taskContract.connect(f.proposer).proposeConsensusTask(
        rpbAddr, ethers.randomBytes(32),
        { minSolvability: 2500, maxSolvability: 7500, peakSolvability: 5000 },
        e18(10), e18(5)
      );

      await expect(
        f.taskContract.connect(f.solver1).submitRollout(
          1, ethers.randomBytes(32), ethers.randomBytes(32), 101
        )
      ).to.be.revertedWith("Confidence must be 0-100");
    });
  });

  // ─────────────────────────────────────────────────────────────────────────
  // 5. TaskContract — Inference (BME on-chain burn)
  // ─────────────────────────────────────────────────────────────────────────
  describe("TaskContract — Inference Tasks", function () {
    let f;
    beforeEach(async () => {
      f = await deployAll();
      // Approve taskContract to spend user1 ATN
      await f.atnToken.connect(f.user1).approve(
        await f.taskContract.getAddress(), e18(10000)
      );
    });

    it("submits inference task and burns tokens", async function () {
      const rpbAddr = await f.rpb.getAddress();
      const balBefore = await f.atnToken.balanceOf(f.user1.address);
      const tx = await f.taskContract.connect(f.user1).submitInferenceTask(
        rpbAddr,
        ethers.keccak256(ethers.toUtf8Bytes("input-data")),
        e18(10)
      );
      await expect(tx).to.emit(f.taskContract, "InferenceTaskSubmitted");

      const balAfter = await f.atnToken.balanceOf(f.user1.address);
      expect(balBefore - balAfter).to.equal(e18(10));
    });

    it("rejects zero burn amount", async function () {
      const rpbAddr = await f.rpb.getAddress();
      await expect(
        f.taskContract.connect(f.user1).submitInferenceTask(rpbAddr, ethers.randomBytes(32), 0)
      ).to.be.revertedWith("Zero burn amount");
    });

    it("records inference completion", async function () {
      const rpbAddr = await f.rpb.getAddress();
      const tx = await f.taskContract.connect(f.user1).submitInferenceTask(
        rpbAddr, ethers.keccak256(ethers.toUtf8Bytes("input")), e18(5)
      );
      const receipt = await tx.wait();
      const event = receipt.logs.find(
        l => l.fragment && l.fragment.name === "InferenceTaskSubmitted"
      );
      const inferenceId = event.args[0];

      // Record completion as governance
      await f.taskContract.recordInferenceCompletion(
        inferenceId,
        f.solver1.address,
        ethers.keccak256(ethers.toUtf8Bytes("output"))
      );

      const task = await f.taskContract.getInferenceTask(inferenceId);
      expect(task.completed).to.be.true;
      expect(task.provider).to.equal(f.solver1.address);
    });
  });

  // ─────────────────────────────────────────────────────────────────────────
  // 6. ResultsRewards — Full Task Lifecycle (Legacy Single-Coordinator)
  // ─────────────────────────────────────────────────────────────────────────
  describe("ResultsRewards — Legacy Mode", function () {
    let f;
    beforeEach(async () => {
      f = await deployAll();
      await f.staking.connect(f.proposer).stake(1, e18(100));
      await f.staking.connect(f.solver1).stake(2, e18(50));
      await f.staking.connect(f.coord1).stake(3, e18(500));
    });

    it("full lifecycle: propose → commit → reveal → verify → reward", async function () {
      const rpbAddr = await f.rpb.getAddress();
      // Proposer proposes task
      const solutionCid = "QmSolution123";
      const groundTruthCid = "QmGroundTruth456";
      const gtHash = ethers.keccak256(ethers.toUtf8Bytes(groundTruthCid));
      const solHash = ethers.keccak256(ethers.toUtf8Bytes(solutionCid));

      await f.taskContract.connect(f.proposer).proposeTask(rpbAddr, ethers.randomBytes(32), gtHash, e18(10), e18(5));

      // Solver commits
      await f.taskContract.connect(f.solver1).commitSolution(1, solHash);

      // Proposer reveals ground truth
      await f.resultsRewards.connect(f.proposer).revealGroundTruth(1, groundTruthCid);

      // Solver reveals solution
      await f.resultsRewards.connect(f.solver1).revealSolution(1, solutionCid);

      // Coordinator verifies (legacy single-coordinator mode)
      const tx = await f.resultsRewards.connect(f.coord1).submitVerification(
        1, f.solver1.address, true, 85, "QmReport"
      );
      await expect(tx).to.emit(f.resultsRewards, "VerificationSubmitted");
      await expect(tx).to.emit(f.resultsRewards, "RewardsDistributed");
    });

    it("ground truth CID mismatch is rejected", async function () {
      const rpbAddr = await f.rpb.getAddress();
      const gtHash = ethers.keccak256(ethers.toUtf8Bytes("correct-gt"));
      await f.taskContract.connect(f.proposer).proposeTask(rpbAddr, ethers.randomBytes(32), gtHash, e18(10), e18(5));

      await expect(
        f.resultsRewards.connect(f.proposer).revealGroundTruth(1, "wrong-cid")
      ).to.be.revertedWith("CID mismatch");
    });

    it("solution CID mismatch is rejected", async function () {
      const rpbAddr = await f.rpb.getAddress();
      const gtCid = "gt-cid";
      const gtHash = ethers.keccak256(ethers.toUtf8Bytes(gtCid));
      const solHash = ethers.keccak256(ethers.toUtf8Bytes("correct-sol"));

      await f.taskContract.connect(f.proposer).proposeTask(rpbAddr, ethers.randomBytes(32), gtHash, e18(10), e18(5));
      await f.taskContract.connect(f.solver1).commitSolution(1, solHash);
      await f.resultsRewards.connect(f.proposer).revealGroundTruth(1, gtCid);

      await expect(
        f.resultsRewards.connect(f.solver1).revealSolution(1, "wrong-solution")
      ).to.be.revertedWith("CID mismatch");
    });
  });

  // ─────────────────────────────────────────────────────────────────────────
  // 7. ResultsRewards — Multi-Coordinator Yuma Consensus
  // ─────────────────────────────────────────────────────────────────────────
  describe("ResultsRewards — Yuma Consensus", function () {
    let f;
    beforeEach(async () => {
      f = await deployAll();
      await f.staking.connect(f.proposer).stake(1, e18(100));
      await f.staking.connect(f.solver1).stake(2, e18(50));
      await f.staking.connect(f.coord1).stake(3, e18(500));
      await f.staking.connect(f.coord2).stake(3, e18(600));

      // Propose + commit + reveal
      const rpbAddr = await f.rpb.getAddress();
      const gtCid = "gt-yuma";
      const solCid = "sol-yuma";
      const gtHash = ethers.keccak256(ethers.toUtf8Bytes(gtCid));
      const solHash = ethers.keccak256(ethers.toUtf8Bytes(solCid));

      await f.taskContract.connect(f.proposer).proposeTask(rpbAddr, ethers.randomBytes(32), gtHash, e18(10), e18(5));
      await f.taskContract.connect(f.solver1).commitSolution(1, solHash);
      await f.resultsRewards.connect(f.proposer).revealGroundTruth(1, gtCid);
      await f.resultsRewards.connect(f.solver1).revealSolution(1, solCid);
    });

    it("two coordinators vote and finalize consensus", async function () {
      await f.resultsRewards.connect(f.coord1).submitVote(1, f.solver1.address, true, 80, "QmR1");
      await f.resultsRewards.connect(f.coord2).submitVote(1, f.solver1.address, true, 90, "QmR2");

      const tx = await f.resultsRewards.finalizeVoting(1, f.solver1.address);
      await expect(tx).to.emit(f.resultsRewards, "YumaConsensusReached");

      const result = await f.resultsRewards.getConsensusResult(1, f.solver1.address);
      expect(result.finalized).to.be.true;
      expect(result.consensusCorrect).to.be.true;
    });

    it("disagreeing coordinators: minority gets slashed", async function () {
      // coord1 votes correct, coord2 votes incorrect
      await f.resultsRewards.connect(f.coord1).submitVote(1, f.solver1.address, true, 80, "QmR1");
      await f.resultsRewards.connect(f.coord2).submitVote(1, f.solver1.address, false, 80, "QmR2");

      // Finalize — coord1 has 500 stake, coord2 has 600 stake
      // correctStake=500, totalStake=1100 → correctStake NOT > totalStake/2 (500 not > 550)
      // So consensus should be INCORRECT
      const tx = await f.resultsRewards.finalizeVoting(1, f.solver1.address);
      const result = await f.resultsRewards.getConsensusResult(1, f.solver1.address);

      expect(result.consensusCorrect).to.be.false;
    });

    it("cannot double-vote", async function () {
      await f.resultsRewards.connect(f.coord1).submitVote(1, f.solver1.address, true, 80, "QmR");
      await expect(
        f.resultsRewards.connect(f.coord1).submitVote(1, f.solver1.address, true, 80, "QmR")
      ).to.be.revertedWith("Already voted");
    });

    it("cannot finalize with insufficient votes before period ends", async function () {
      await f.resultsRewards.connect(f.coord1).submitVote(1, f.solver1.address, true, 80, "QmR");
      await expect(
        f.resultsRewards.finalizeVoting(1, f.solver1.address)
      ).to.be.revertedWith("Cannot finalize yet");
    });

    it("cannot finalize after period with insufficient votes", async function () {
      await f.resultsRewards.connect(f.coord1).submitVote(1, f.solver1.address, true, 80, "QmR");
      // Mine past voting period
      await mineBlocks(101);
      await expect(
        f.resultsRewards.finalizeVoting(1, f.solver1.address)
      ).to.be.revertedWith("Insufficient coordinator votes");
    });

    it("EMA bond updates track coordinator reliability", async function () {
      await f.resultsRewards.connect(f.coord1).submitVote(1, f.solver1.address, true, 80, "QmR1");
      await f.resultsRewards.connect(f.coord2).submitVote(1, f.solver1.address, true, 90, "QmR2");
      await f.resultsRewards.finalizeVoting(1, f.solver1.address);

      const bond1 = await f.resultsRewards.getCoordinatorBond(f.coord1.address);
      expect(bond1.totalVotes).to.equal(1);
      expect(bond1.correctVotes).to.equal(1);
      expect(bond1.emaBondStrength).to.be.gt(0);
    });
  });

  // ─────────────────────────────────────────────────────────────────────────
  // 8. RPB — Budget, Pricing, Disbursement (replaces Project tests)
  // ─────────────────────────────────────────────────────────────────────────
  describe("RPB — Budget, Pricing, Disbursement", function () {
    let f;
    beforeEach(async () => {
      f = await deployAll();
    });

    it("fundTaskBudget adds to budget", async function () {
      const budgetBefore = await f.rpb.taskRewardBudgetATN();
      await f.atnToken.approve(await f.rpb.getAddress(), e18(1000));
      await f.rpb.fundTaskBudget(e18(1000));
      const budgetAfter = await f.rpb.taskRewardBudgetATN();
      expect(budgetAfter - budgetBefore).to.equal(e18(1000));
    });

    it("disburseFromBudget requires authorization", async function () {
      await expect(
        f.rpb.connect(f.user1).disburseFromBudget(f.user1.address, e18(1))
      ).to.be.revertedWithCustomError(f.rpb, "NotAuthorizedDisburser");
    });

    it("disburseFromBudget respects budget limit", async function () {
      // Budget is 10000 from deployAll
      await expect(
        f.rpb.disburseFromBudget(f.user1.address, e18(20000))
      ).to.be.revertedWithCustomError(f.rpb, "InsufficientBudget");
    });

    it("authorized disburser can pay from budget", async function () {
      const balBefore = await f.atnToken.balanceOf(f.user1.address);
      // Owner is authorized (owner or DAO can disburse)
      await f.rpb.disburseFromBudget(f.user1.address, e18(100));
      const balAfter = await f.atnToken.balanceOf(f.user1.address);
      expect(balAfter - balBefore).to.equal(e18(100));
    });

    it("setMatureModel sets CID and price", async function () {
      await f.rpb.setMatureModel("QmWeightsCid", e18(10));
      const [cid, price] = await f.rpb.getMatureModel();
      expect(cid).to.equal("QmWeightsCid");
      expect(price).to.equal(e18(10));
    });

    it("getEffectivePrice with discount tiers based on shares", async function () {
      await f.rpb.setMatureModel("QmW", e18(10));

      // Set parity for ATN in registry so purchaseShares works
      const atnAddr = await f.atnToken.getAddress();
      const parityKey = "jurisdiction.parity." + atnAddr.toLowerCase();
      await f.registry.editRegistry(parityKey, "1");

      // user1 purchases shares
      await f.atnToken.connect(f.user1).approve(await f.rpb.getAddress(), e18(10000));
      await f.rpb.connect(f.user1).purchaseShares(atnAddr, e18(100));

      // Set discount tiers: 50 shares → 10% off, 100 shares → 20% off
      await f.rpb.setDiscountTiers([
        { ptThreshold: e18(50), discountPermille: 100 },
        { ptThreshold: e18(100), discountPermille: 200 },
      ]);

      // user1 has 100 shares → 20% discount → 10 * 0.8 = 8
      const price = await f.rpb.getEffectivePrice(f.user1.address);
      expect(price).to.equal(e18(8));

      // user2 has 0 shares → full price
      const price2 = await f.rpb.getEffectivePrice(f.user2.address);
      expect(price2).to.equal(e18(10));
    });

    it("setMatureModel requires owner or DAO", async function () {
      await expect(
        f.rpb.connect(f.user1).setMatureModel("QmW", e18(1))
      ).to.be.revertedWithCustomError(f.rpb, "NotOwnerOrDAO");
    });

    it("setAuthorizedDisburser requires owner or DAO", async function () {
      await expect(
        f.rpb.connect(f.user1).setAuthorizedDisburser(f.user1.address, true)
      ).to.be.revertedWithCustomError(f.rpb, "NotOwnerOrDAO");
    });

    it("setDiscountTiers requires owner or DAO", async function () {
      await expect(
        f.rpb.connect(f.user1).setDiscountTiers([])
      ).to.be.revertedWithCustomError(f.rpb, "NotOwnerOrDAO");
    });

    it("getDiscountTierCount reflects current tiers", async function () {
      expect(await f.rpb.getDiscountTierCount()).to.equal(0);
      await f.rpb.setDiscountTiers([
        { ptThreshold: e18(50), discountPermille: 100 },
      ]);
      expect(await f.rpb.getDiscountTierCount()).to.equal(1);
    });
  });

  // ─────────────────────────────────────────────────────────────────────────
  // 9. ForcedErrorRegistry
  // ─────────────────────────────────────────────────────────────────────────
  describe("ForcedErrorRegistry", function () {
    let f;
    beforeEach(async () => {
      f = await deployAll();
      await f.staking.connect(f.coord1).stake(3, e18(500));

      // Fund jackpot pool
      await f.atnToken.approve(await f.forcedErrorRegistry.getAddress(), e18(1000));
      await f.forcedErrorRegistry.fundJackpotPool(e18(200));
    });

    it("injects forced error and coordinator catches it", async function () {
      const badHash = ethers.keccak256(ethers.toUtf8Bytes("known-bad-solution"));
      await f.forcedErrorRegistry.injectForcedError(42, badHash);

      expect(await f.forcedErrorRegistry.isTaskForcedError(42)).to.be.true;
      expect(await f.forcedErrorRegistry.isForcedErrorActive(42)).to.be.true;

      // Coordinator catches it
      const balBefore = await f.atnToken.balanceOf(f.coord1.address);
      await f.forcedErrorRegistry.connect(f.coord1).reportForcedError(42, badHash);
      const balAfter = await f.atnToken.balanceOf(f.coord1.address);

      // Should receive jackpot of 50 ATN
      expect(balAfter - balBefore).to.equal(e18(50));
      expect(await f.forcedErrorRegistry.isForcedErrorActive(42)).to.be.false;
    });

    it("wrong hash does not catch forced error", async function () {
      const badHash = ethers.keccak256(ethers.toUtf8Bytes("known-bad"));
      await f.forcedErrorRegistry.injectForcedError(42, badHash);

      await expect(
        f.forcedErrorRegistry.connect(f.coord1).reportForcedError(
          42, ethers.keccak256(ethers.toUtf8Bytes("wrong-guess"))
        )
      ).to.be.revertedWith("Wrong solution hash");
    });

    it("expired forced error cannot be caught", async function () {
      const badHash = ethers.keccak256(ethers.toUtf8Bytes("bad"));
      await f.forcedErrorRegistry.injectForcedError(42, badHash);

      // Mine past expiration
      await mineBlocks(1001);

      await expect(
        f.forcedErrorRegistry.connect(f.coord1).reportForcedError(42, badHash)
      ).to.be.revertedWith("Forced error expired");
    });

    it("processCoordinatorMiss slashes coordinator", async function () {
      const badHash = ethers.keccak256(ethers.toUtf8Bytes("bad"));
      await f.forcedErrorRegistry.injectForcedError(42, badHash);

      // Let it expire
      await mineBlocks(1001);

      // Process miss
      const stakeBefore = await f.staking.getStake(f.coord1.address);
      await f.forcedErrorRegistry.processCoordinatorMiss(42, f.coord1.address);
      const stakeAfter = await f.staking.getStake(f.coord1.address);

      expect(stakeBefore.amount - stakeAfter.amount).to.equal(e18(25));
    });

    it("FIX: jackpot pool correctly decremented on injection", async function () {
      // Each injection reserves 50 ATN from the 200 pool
      await f.forcedErrorRegistry.injectForcedError(0, ethers.randomBytes(32));
      expect(await f.forcedErrorRegistry.jackpotPool()).to.equal(e18(150));

      await f.forcedErrorRegistry.injectForcedError(1, ethers.randomBytes(32));
      expect(await f.forcedErrorRegistry.jackpotPool()).to.equal(e18(100));

      await f.forcedErrorRegistry.injectForcedError(2, ethers.randomBytes(32));
      expect(await f.forcedErrorRegistry.jackpotPool()).to.equal(e18(50));

      await f.forcedErrorRegistry.injectForcedError(3, ethers.randomBytes(32));
      expect(await f.forcedErrorRegistry.jackpotPool()).to.equal(0n);

      // 5th injection fails — pool exhausted
      await expect(
        f.forcedErrorRegistry.injectForcedError(4, ethers.randomBytes(32))
      ).to.be.revertedWith("Insufficient jackpot pool");
    });

    it("governance can update config", async function () {
      await f.forcedErrorRegistry.setJackpotAmount(e18(100));
      expect(await f.forcedErrorRegistry.jackpotAmount()).to.equal(e18(100));

      await f.forcedErrorRegistry.setSlashAmount(e18(50));
      expect(await f.forcedErrorRegistry.slashAmountForMissed()).to.equal(e18(50));

      await f.forcedErrorRegistry.setForcedErrorProbability(100);
      expect(await f.forcedErrorRegistry.forcedErrorProbability()).to.equal(100);

      await f.forcedErrorRegistry.setExpirationBlocks(2000);
      expect(await f.forcedErrorRegistry.expirationBlocks()).to.equal(2000);
    });
  });

  // ─────────────────────────────────────────────────────────────────────────
  // 10. ModelShardRegistry
  // ─────────────────────────────────────────────────────────────────────────
  describe("ModelShardRegistry", function () {
    let f;
    const modelHash = ethers.keccak256(ethers.toUtf8Bytes("model-v1"));

    beforeEach(async () => {
      f = await deployAll();
      // Provider needs stake
      await f.staking.connect(f.solver1).stake(2, e18(50));
    });

    it("registers provider and model", async function () {
      await f.modelShardRegistry.connect(f.solver1).registerProvider(1_000_000);

      const info = await f.modelShardRegistry.getProviderInfo(f.solver1.address);
      expect(info.capacity).to.equal(1_000_000);
      expect(info.reputation).to.equal(500);
      expect(info.active).to.be.true;
    });

    it("rejects provider with insufficient stake", async function () {
      // user1 has no stake
      await expect(
        f.modelShardRegistry.connect(f.user1).registerProvider(1_000_000)
      ).to.be.revertedWith("Insufficient stake");
    });

    it("registers model and announces shards", async function () {
      await f.modelShardRegistry.connect(f.solver1).registerProvider(1_000_000);
      const rpbAddr = await f.rpb.getAddress();

      await f.modelShardRegistry.connect(f.owner).registerModel(
        modelHash, "QmManifest", ethers.randomBytes(32),
        10, 4, 500_000,
        0, // NODE_PUBLIC
        0, // LAYER_WISE
        rpbAddr
      );

      // Announce a shard
      const shardHash = ethers.keccak256(ethers.toUtf8Bytes("shard-0"));
      await f.modelShardRegistry.connect(f.solver1).announceShard(
        modelHash, 0, shardHash, 50_000, false
      );

      const providers = await f.modelShardRegistry.getShardProviders(modelHash, 0);
      expect(providers).to.include(f.solver1.address);
    });

    it("rejects duplicate shard announcement", async function () {
      await f.modelShardRegistry.connect(f.solver1).registerProvider(1_000_000);
      const rpbAddr = await f.rpb.getAddress();
      await f.modelShardRegistry.connect(f.owner).registerModel(
        modelHash, "QmM", ethers.randomBytes(32), 10, 4, 500_000, 0, 0, rpbAddr
      );
      const shardHash = ethers.keccak256(ethers.toUtf8Bytes("shard-0"));
      await f.modelShardRegistry.connect(f.solver1).announceShard(
        modelHash, 0, shardHash, 50_000, false
      );
      await expect(
        f.modelShardRegistry.connect(f.solver1).announceShard(
          modelHash, 0, shardHash, 50_000, false
        )
      ).to.be.revertedWith("Already storing this shard");
    });

    it("removes shard and frees capacity", async function () {
      await f.modelShardRegistry.connect(f.solver1).registerProvider(1_000_000);
      const rpbAddr = await f.rpb.getAddress();
      await f.modelShardRegistry.connect(f.owner).registerModel(
        modelHash, "QmM", ethers.randomBytes(32), 10, 4, 500_000, 0, 0, rpbAddr
      );
      const shardHash = ethers.keccak256(ethers.toUtf8Bytes("shard-0"));
      await f.modelShardRegistry.connect(f.solver1).announceShard(
        modelHash, 0, shardHash, 50_000, false
      );

      await f.modelShardRegistry.connect(f.solver1).removeShard(modelHash, 0);
      const info = await f.modelShardRegistry.getProviderInfo(f.solver1.address);
      expect(info.used).to.equal(0);
    });

    it("checks shard availability correctly", async function () {
      await f.modelShardRegistry.connect(f.solver1).registerProvider(1_000_000);
      const rpbAddr = await f.rpb.getAddress();
      await f.modelShardRegistry.connect(f.owner).registerModel(
        modelHash, "QmM", ethers.randomBytes(32), 2, 1, 100_000, 0, 0, rpbAddr
      );
      // No shards announced
      const [available, sufficient] = await f.modelShardRegistry.checkShardAvailability(modelHash);
      expect(available).to.equal(0);
      expect(sufficient).to.be.false;
    });

    it("reputation updates on verification success/failure", async function () {
      await f.modelShardRegistry.connect(f.solver1).registerProvider(1_000_000);

      await f.modelShardRegistry.reportVerificationSuccess(modelHash, 0, f.solver1.address);
      let info = await f.modelShardRegistry.getProviderInfo(f.solver1.address);
      expect(info.reputation).to.equal(510); // 500 + 10
      expect(info.successfulVerifications).to.equal(1);

      await f.modelShardRegistry.reportVerificationFailure(modelHash, 0, f.solver1.address);
      info = await f.modelShardRegistry.getProviderInfo(f.solver1.address);
      expect(info.reputation).to.equal(460); // 510 - 50
      expect(info.failedVerifications).to.equal(1);
    });
  });

  // ─────────────────────────────────────────────────────────────────────────
  // 11. AutonetDAO
  // ─────────────────────────────────────────────────────────────────────────
  describe("AutonetDAO", function () {
    let f;
    beforeEach(async () => {
      f = await deployAll();
      // Owner needs to delegate votes to self for governance
      await f.atnToken.connect(f.owner).delegate(f.owner.address);
      await mineBlocks(1);
    });

    it("creates proposal when above threshold", async function () {
      const calldataHash = ethers.keccak256(ethers.toUtf8Bytes("action-data"));
      const tx = await f.dao.propose("Upgrade param X", calldataHash);
      await expect(tx).to.emit(f.dao, "ProposalCreated");
    });

    it("rejects proposal below threshold", async function () {
      // coord3 has 100K but hasn't delegated — getVotes returns 0
      await expect(
        f.dao.connect(f.coord3).propose("Bad proposal", ethers.randomBytes(32))
      ).to.be.revertedWith("Below threshold");
    });

    it("casting vote and checking state", async function () {
      const calldataHash = ethers.keccak256(ethers.toUtf8Bytes("action"));
      await f.dao.propose("Test", calldataHash);

      // Mine past voting delay (~7200 blocks for 1 day / 12s)
      await mineBlocks(7201);

      await f.dao.castVote(1, true);

      const p = await f.dao.proposals(1);
      expect(p.forVotes).to.be.gt(0);
    });

    it("cannot vote before voting period starts", async function () {
      await f.dao.propose("Test", ethers.randomBytes(32));
      await expect(
        f.dao.castVote(1, true)
      ).to.be.revertedWith("Voting not started");
    });

    it("proposal state transitions correctly", async function () {
      const calldataHash = ethers.keccak256(ethers.toUtf8Bytes("action"));
      await f.dao.propose("Test", calldataHash);

      // Pending before start
      expect(await f.dao.getProposalState(1)).to.equal(0); // Pending

      // Mine to active
      await mineBlocks(7201);
      expect(await f.dao.getProposalState(1)).to.equal(1); // Active
    });

    it("can cancel proposal", async function () {
      await f.dao.propose("Cancel me", ethers.randomBytes(32));
      await f.dao.cancel(1);
      expect(await f.dao.getProposalState(1)).to.equal(2); // Canceled
    });
  });

  // ─────────────────────────────────────────────────────────────────────────
  // 12. CapabilityScorecard
  // ─────────────────────────────────────────────────────────────────────────
  describe("CapabilityScorecard", function () {
    let f;
    const visualEncoder = ethers.keccak256(ethers.toUtf8Bytes("visual_encoder"));
    const textDecoder = ethers.keccak256(ethers.toUtf8Bytes("text_decoder"));

    beforeEach(async () => {
      f = await deployAll();
      await f.scorecard.registerModule(visualEncoder, 8000);
      await f.scorecard.registerModule(textDecoder, 8000);
    });

    it("registers modules and returns max multiplier for zero score", async function () {
      const mult = await f.scorecard.getRewardMultiplier(visualEncoder);
      expect(mult).to.equal(30000); // 3x for zero-capability module
    });

    it("returns min multiplier for saturated modules", async function () {
      // Update score to target (8000)
      await f.scorecard.updateScore(visualEncoder, 8000);
      const mult = await f.scorecard.getRewardMultiplier(visualEncoder);
      expect(mult).to.equal(5000); // 0.5x
    });

    it("EMA smooths score updates", async function () {
      await f.scorecard.updateScore(visualEncoder, 4000); // First: set directly to 4000
      let mod = await f.scorecard.modules(visualEncoder);
      expect(mod.score).to.equal(4000);

      await f.scorecard.updateScore(visualEncoder, 8000); // EMA: (4000*1 + 8000) / 2 = 6000
      mod = await f.scorecard.modules(visualEncoder);
      expect(mod.score).to.equal(6000);
    });

    it("getScorecard returns all modules", async function () {
      const [ids, scores, targets, multipliers] = await f.scorecard.getScorecard();
      expect(ids.length).to.equal(2);
      expect(targets[0]).to.equal(8000);
    });

    it("only evaluators can update scores", async function () {
      await expect(
        f.scorecard.connect(f.user1).updateScore(visualEncoder, 5000)
      ).to.be.revertedWithCustomError(f.scorecard, "NotAuthorized");
    });

    it("rejects duplicate module registration", async function () {
      await expect(
        f.scorecard.registerModule(visualEncoder, 8000)
      ).to.be.revertedWithCustomError(f.scorecard, "ModuleAlreadyExists");
    });

    it("linear interpolation produces expected multipliers", async function () {
      // score=4000, target=8000 → progress=50% → reduction = 12500 → mult = 30000-12500 = 17500
      await f.scorecard.updateScore(visualEncoder, 4000);
      const mult = await f.scorecard.getRewardMultiplier(visualEncoder);
      expect(mult).to.equal(17500);
    });
  });

  // ─────────────────────────────────────────────────────────────────────────
  // 13. GuildRegistry
  // ─────────────────────────────────────────────────────────────────────────
  describe("GuildRegistry", function () {
    let f;
    const moduleA = ethers.keccak256(ethers.toUtf8Bytes("vision"));
    const moduleB = ethers.keccak256(ethers.toUtf8Bytes("language"));

    beforeEach(async () => {
      f = await deployAll();
      // Approve guild registry for staking
      for (const s of [f.user1, f.user2, f.proposer]) {
        await f.atnToken.connect(s).approve(await f.guildRegistry.getAddress(), e18(10000));
      }
    });

    it("creates guild with modules", async function () {
      const tx = await f.guildRegistry.connect(f.user1).createGuild("Vision Guild", [moduleA]);
      await expect(tx).to.emit(f.guildRegistry, "GuildCreated");

      const [name, moduleIds, memberCount, aggregator, creator, totalStaked, active] =
        await f.guildRegistry.getGuild(1);
      expect(name).to.equal("Vision Guild");
      expect(memberCount).to.equal(1);
      expect(creator).to.equal(f.user1.address);
      expect(active).to.be.true;
    });

    it("rejects guild with already-assigned modules", async function () {
      await f.guildRegistry.connect(f.user1).createGuild("G1", [moduleA]);
      await expect(
        f.guildRegistry.connect(f.user2).createGuild("G2", [moduleA])
      ).to.be.revertedWithCustomError(f.guildRegistry, "ModuleAlreadyAssigned");
    });

    it("member joins and leaves guild", async function () {
      await f.guildRegistry.connect(f.user1).createGuild("G1", [moduleA]);

      await f.guildRegistry.connect(f.user2).joinGuild(1);
      expect(await f.guildRegistry.isMember(f.user2.address, 1)).to.be.true;

      const balBefore = await f.atnToken.balanceOf(f.user2.address);
      await f.guildRegistry.connect(f.user2).leaveGuild(1);
      const balAfter = await f.atnToken.balanceOf(f.user2.address);
      expect(balAfter - balBefore).to.equal(e18(50)); // minMemberStake returned
    });

    it("aggregator cannot leave if other members exist", async function () {
      await f.guildRegistry.connect(f.user1).createGuild("G1", [moduleA]);
      await f.guildRegistry.connect(f.user2).joinGuild(1);

      // user1 is creator/aggregator with 2 members
      await expect(
        f.guildRegistry.connect(f.user1).leaveGuild(1)
      ).to.be.revertedWithCustomError(f.guildRegistry, "NotAuthorized");
    });

    it("creator sets aggregator", async function () {
      await f.guildRegistry.connect(f.user1).createGuild("G1", [moduleA]);
      await f.guildRegistry.connect(f.user2).joinGuild(1);

      await f.guildRegistry.connect(f.user1).setGuildAggregator(1, f.user2.address);
      expect(await f.guildRegistry.isGuildAggregator(f.user2.address, 1)).to.be.true;
    });

    it("reputation updates via EMA", async function () {
      await f.guildRegistry.connect(f.user1).createGuild("G1", [moduleA]);

      // First evaluation: set directly
      await f.guildRegistry.recordGuildMetrics(1, moduleA, 7000);
      let [score] = await f.guildRegistry.getReputation(1, moduleA);
      expect(score).to.equal(7000);

      // Second evaluation: EMA
      await f.guildRegistry.recordGuildMetrics(1, moduleA, 9000);
      [score] = await f.guildRegistry.getReputation(1, moduleA);
      expect(score).to.equal(8000); // (7000*1 + 9000) / 2
    });

    it("reward multiplier scales with reputation", async function () {
      await f.guildRegistry.connect(f.user1).createGuild("G1", [moduleA]);
      // Base reputation = 5000 → multiplier = 5000 + 5000 = 10000 (1x)
      let mult = await f.guildRegistry.getGuildRewardMultiplier(1, moduleA);
      expect(mult).to.equal(10000);

      // Max reputation = 10000 → multiplier = 15000 (1.5x)
      await f.guildRegistry.recordGuildMetrics(1, moduleA, 10000);
      mult = await f.guildRegistry.getGuildRewardMultiplier(1, moduleA);
      expect(mult).to.equal(15000);
    });
  });

  // ─────────────────────────────────────────────────────────────────────────
  // 14. EvolutionProposal
  // ─────────────────────────────────────────────────────────────────────────
  describe("EvolutionProposal", function () {
    let f;
    beforeEach(async () => {
      f = await deployAll();
      await f.atnToken.connect(f.user1).approve(
        await f.evolutionProposal.getAddress(), e18(10000)
      );
      await f.atnToken.approve(await f.evolutionProposal.getAddress(), e18(100000));
    });

    it("submits proposal with stake", async function () {
      const tx = await f.evolutionProposal.connect(f.user1).submitProposal(
        "QmProposalCid", e18(100)
      );
      await expect(tx).to.emit(f.evolutionProposal, "ProposalSubmitted");

      const [proposer, , stakeAmount, status] = await f.evolutionProposal.getProposal(1);
      expect(proposer).to.equal(f.user1.address);
      expect(stakeAmount).to.equal(e18(100));
      expect(status).to.equal(0); // Proposed
    });

    it("rejects proposal below min stake", async function () {
      await expect(
        f.evolutionProposal.connect(f.user1).submitProposal("QmCid", e18(10))
      ).to.be.revertedWithCustomError(f.evolutionProposal, "InsufficientStake");
    });

    it("full lifecycle: propose → evaluate → trial → adopt", async function () {
      // Submit proposal
      await f.evolutionProposal.connect(f.user1).submitProposal("QmCid", e18(100));

      // Begin evaluation (admin)
      await f.evolutionProposal.beginEvaluation(1);
      let [, , , status] = await f.evolutionProposal.getProposal(1);
      expect(status).to.equal(1); // Evaluating

      // Set evaluators
      await f.evolutionProposal.setRPBEvaluator(f.coord1.address, true);
      await f.evolutionProposal.setRPBEvaluator(f.coord2.address, true);
      await f.evolutionProposal.setRPBEvaluator(f.coord3.address, true);

      // Submit evaluations (3 approve with high confidence)
      await f.evolutionProposal.connect(f.coord1).submitEvaluation(1, true, 8000, "QmReason1");
      await f.evolutionProposal.connect(f.coord2).submitEvaluation(1, true, 7000, "QmReason2");
      await f.evolutionProposal.connect(f.coord3).submitEvaluation(1, true, 9000, "QmReason3");

      // Advance past evaluation period
      await increaseTime(3 * 24 * 3600 + 1);

      // Resolve evaluation
      await f.evolutionProposal.resolveEvaluation(1);
      [, , , status] = await f.evolutionProposal.getProposal(1);
      expect(status).to.equal(2); // Trial

      // Fund trial
      await f.evolutionProposal.fundTrial(1, e18(500));

      // Record contributions
      await f.evolutionProposal.recordTrialContribution(1, f.solver1.address, 2, 100); // Proposal type
      await f.evolutionProposal.recordTrialContribution(1, f.solver2.address, 0, 200); // Compute type

      // Adopt
      const user1BalBefore = await f.atnToken.balanceOf(f.user1.address);
      await f.evolutionProposal.adoptProposal(1);
      const user1BalAfter = await f.atnToken.balanceOf(f.user1.address);

      // Proposer stake returned
      expect(user1BalAfter - user1BalBefore).to.equal(e18(100));

      [, , , status] = await f.evolutionProposal.getProposal(1);
      expect(status).to.equal(3); // Adopted
    });

    it("rejection refunds stake minus processing fee", async function () {
      await f.evolutionProposal.connect(f.user1).submitProposal("QmCid", e18(100));

      const balBefore = await f.atnToken.balanceOf(f.user1.address);
      await f.evolutionProposal.rejectProposal(1);
      const balAfter = await f.atnToken.balanceOf(f.user1.address);

      // 5% processing fee → refund 95 ATN
      expect(balAfter - balBefore).to.equal(e18(95));
    });

    it("evaluation quorum check", async function () {
      await f.evolutionProposal.connect(f.user1).submitProposal("QmCid", e18(100));
      await f.evolutionProposal.beginEvaluation(1);
      await f.evolutionProposal.setRPBEvaluator(f.coord1.address, true);
      await f.evolutionProposal.connect(f.coord1).submitEvaluation(1, true, 8000, "QmR");

      await increaseTime(3 * 24 * 3600 + 1);

      // Only 1 eval, need 3
      await expect(
        f.evolutionProposal.resolveEvaluation(1)
      ).to.be.revertedWithCustomError(f.evolutionProposal, "QuorumNotReached");
    });

    it("RPB prompt update via timelock", async function () {
      await f.registry.editRegistry("test", "value"); // owner can edit directly

      await expect(
        f.evolutionProposal.connect(f.user1).updateRPBPrompt("QmNewPrompt")
      ).to.be.revertedWithCustomError(f.evolutionProposal, "NotAuthorized");
    });
  });

  // ─────────────────────────────────────────────────────────────────────────
  // 15. AnchorBridge
  // ─────────────────────────────────────────────────────────────────────────
  describe("AnchorBridge", function () {
    let f;
    beforeEach(async () => {
      f = await deployAll();
      await f.anchorBridge.addValidator(f.coord1.address);
      await f.atnToken.connect(f.user1).approve(await f.anchorBridge.getAddress(), e18(10000));
    });

    it("adds and removes validators", async function () {
      expect(await f.anchorBridge.validators(f.coord1.address)).to.be.true;
      expect(await f.anchorBridge.validatorCount()).to.equal(1);

      await f.anchorBridge.removeValidator(f.coord1.address);
      expect(await f.anchorBridge.validators(f.coord1.address)).to.be.false;
    });

    it("validator submits checkpoint", async function () {
      const root = ethers.keccak256(ethers.toUtf8Bytes("checkpoint-root"));
      const tx = await f.anchorBridge.connect(f.coord1).submitCheckpoint(1, root);
      await expect(tx).to.emit(f.anchorBridge, "CheckpointSubmitted");

      expect(await f.anchorBridge.latestRoot()).to.equal(root);
      expect(await f.anchorBridge.latestEpoch()).to.equal(1);
    });

    it("non-validator cannot submit checkpoint", async function () {
      await expect(
        f.anchorBridge.connect(f.user1).submitCheckpoint(1, ethers.randomBytes(32))
      ).to.be.revertedWith("Not validator");
    });

    it("deposits tokens", async function () {
      const tx = await f.anchorBridge.connect(f.user1).deposit(e18(100));
      await expect(tx).to.emit(f.anchorBridge, "Deposited");

      const bridgeBal = await f.atnToken.balanceOf(await f.anchorBridge.getAddress());
      expect(bridgeBal).to.equal(e18(100));
    });

    it("rejects zero deposit", async function () {
      await expect(
        f.anchorBridge.connect(f.user1).deposit(0)
      ).to.be.revertedWith("Zero amount");
    });

    it("multi-sig checkpoint requires multiple approvals", async function () {
      await f.anchorBridge.addValidator(f.coord2.address);
      await f.anchorBridge.setRequiredSignatures(2);

      const root = ethers.keccak256(ethers.toUtf8Bytes("root"));
      await f.anchorBridge.connect(f.coord1).submitCheckpoint(1, root);
      // Should not update yet
      expect(await f.anchorBridge.latestEpoch()).to.equal(0);

      // Second approval
      await f.anchorBridge.connect(f.coord2).submitCheckpoint(1, root);
      expect(await f.anchorBridge.latestEpoch()).to.equal(1);
      expect(await f.anchorBridge.latestRoot()).to.equal(root);
    });

    it("rejects mismatched roots in multi-sig", async function () {
      await f.anchorBridge.addValidator(f.coord2.address);
      await f.anchorBridge.setRequiredSignatures(2);

      await f.anchorBridge.connect(f.coord1).submitCheckpoint(
        1, ethers.keccak256(ethers.toUtf8Bytes("root-A"))
      );
      await expect(
        f.anchorBridge.connect(f.coord2).submitCheckpoint(
          1, ethers.keccak256(ethers.toUtf8Bytes("root-B"))
        )
      ).to.be.revertedWith("Root mismatch");
    });
  });

  // ─────────────────────────────────────────────────────────────────────────
  // 16. DisputeManager
  // ─────────────────────────────────────────────────────────────────────────
  describe("DisputeManager", function () {
    let f;
    beforeEach(async () => {
      f = await deployAll();
      // Delegate votes for dispute voting
      await f.atnToken.connect(f.owner).delegate(f.owner.address);
      await f.atnToken.connect(f.user1).delegate(f.user1.address);
      await mineBlocks(1);
    });

    it("creates dispute", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("bad-checkpoint"));
      const tx = await f.disputeManager.createDispute(contentHash);
      await expect(tx).to.emit(f.disputeManager, "DisputeCreated");

      const dispute = await f.disputeManager.getDispute(1);
      expect(dispute.contentHash).to.equal(contentHash);
      expect(dispute.status).to.equal(0); // CREATED
    });

    it("voting and auto-resolution at quorum", async function () {
      await f.disputeManager.createDispute(ethers.randomBytes(32));

      // Owner has massive token balance → votes should meet quorum
      const tx = await f.disputeManager.vote(1, true); // support invalid
      await expect(tx).to.emit(f.disputeManager, "DisputeResolved");

      const dispute = await f.disputeManager.getDispute(1);
      // Owner has supermajority
      expect(dispute.status).to.equal(3); // RESOLVED_INVALID
    });

    it("cannot vote twice", async function () {
      await f.disputeManager.createDispute(ethers.randomBytes(32));
      await f.disputeManager.vote(1, true);

      await expect(
        f.disputeManager.vote(1, true)
      ).to.be.revertedWith("Not active");
    });

    it("voter with no tokens is rejected", async function () {
      await f.disputeManager.createDispute(ethers.randomBytes(32));
      // coord1 has tokens but hasn't delegated
      await expect(
        f.disputeManager.connect(f.coord1).vote(1, true)
      ).to.be.revertedWith("No voting power");
    });
  });

  // ─────────────────────────────────────────────────────────────────────────
  // 17. InferenceProviderBridge + Factory
  // ─────────────────────────────────────────────────────────────────────────
  describe("InferenceProviderBridge & Factory", function () {
    let f, bridge;
    beforeEach(async () => {
      f = await deployAll();

      // Set mature model on RPB
      await f.rpb.setMatureModel("QmWeights", e18(10));

      // Deploy bridge via factory
      const rpbAddr = await f.rpb.getAddress();
      const tx = await f.inferenceFactory.deployBridge(rpbAddr, f.owner.address);
      const receipt = await tx.wait();
      const event = receipt.logs.find(l => l.fragment && l.fragment.name === "BridgeDeployed");
      const bridgeAddr = event.args[1];

      bridge = await ethers.getContractAt("InferenceProviderBridge", bridgeAddr);

      // Authorize the bridge as a minter for BME
      await f.atnToken.setAuthorizedMinter(bridgeAddr, true);

      // Authorize the bridge as a disburser on RPB (for recordInferenceBME)
      await f.rpb.setAuthorizedDisburser(bridgeAddr, true);

      // Setup provider
      await f.staking.connect(f.solver1).stake(7, e18(200)); // INFERENCE_PROVIDER
      await bridge.setAuthorizedNode(f.coord1.address, true);
    });

    it("factory deploys bridge correctly", async function () {
      const rpbAddr = await f.rpb.getAddress();
      expect(await f.inferenceFactory.hasBridge(rpbAddr)).to.be.true;
      expect(await f.inferenceFactory.getBridgeCount()).to.equal(1);
    });

    it("factory prevents double deployment", async function () {
      const rpbAddr = await f.rpb.getAddress();
      await expect(
        f.inferenceFactory.deployBridge(rpbAddr, f.owner.address)
      ).to.be.revertedWith("Bridge already deployed");
    });

    it("factory rejects RPB without mature model", async function () {
      // Deploy a fresh RPB without model
      const rpbTx = await f.rpbFactory.createRPB(
        await f.registry.getAddress(),
        await f.atnToken.getAddress()
      );
      const receipt = await rpbTx.wait();
      const evt = receipt.logs.find(l => l.fragment && l.fragment.name === "RPBDeployed");
      const newRpbAddr = evt.args[0];

      await expect(
        f.inferenceFactory.deployBridge(newRpbAddr, f.owner.address)
      ).to.be.revertedWith("RPB has no mature model");
    });

    it("requestInference burns ATN via BME", async function () {
      // User needs to approve burn (burnFrom requires allowance)
      await f.atnToken.connect(f.user1).approve(await bridge.getAddress(), e18(10000));

      const balBefore = await f.atnToken.balanceOf(f.user1.address);
      const input = ethers.AbiCoder.defaultAbiCoder().encode(["string"], ["QmInput"]);
      const tx = await bridge.connect(f.user1).requestInference(input, e18(100));
      const receipt = await tx.wait();
      const balAfter = await f.atnToken.balanceOf(f.user1.address);

      // Price is 10 ATN (from setMatureModel), user1 has 0 shares → full price
      expect(balBefore - balAfter).to.equal(e18(10));
      expect(await bridge.totalBurned()).to.equal(e18(10));
    });

    it("submitResult splits revenue via RPB ratios (provider/shareholder/treasury)", async function () {
      // Request inference
      await f.atnToken.connect(f.user1).approve(await bridge.getAddress(), e18(10000));
      const input = ethers.AbiCoder.defaultAbiCoder().encode(["string"], ["QmInput"]);
      const tx = await bridge.connect(f.user1).requestInference(input, e18(100));
      const receipt = await tx.wait();
      const reqEvent = receipt.logs.find(
        l => l.fragment && l.fragment.name === "InferenceRequested"
      );
      const requestId = reqEvent.args[0];

      // Disable spot-check for deterministic testing
      await bridge.setProtocolFee(300, 0); // 3% fee, 0% verification rate

      const rpbAddr = await f.rpb.getAddress();
      const daoAddr = await f.rpb.dao();
      const providerBalBefore = await f.atnToken.balanceOf(f.solver1.address);
      const rpbBalBefore = await f.atnToken.balanceOf(rpbAddr);
      const daoBalBefore = await f.atnToken.balanceOf(daoAddr);

      await bridge.connect(f.coord1).submitResult(requestId, "QmOutput", f.solver1.address);

      const providerBalAfter = await f.atnToken.balanceOf(f.solver1.address);
      const rpbBalAfter = await f.atnToken.balanceOf(rpbAddr);
      const daoBalAfter = await f.atnToken.balanceOf(daoAddr);

      // 10 ATN burned, 3% protocol fee = 0.3 ATN to jackpot, 9.7 ATN to split
      // RPB revenue split (60/25/15):
      //   Provider:    60% of 9.7 = 5.82 ATN
      //   Shareholder: 25% of 9.7 = 2.425 ATN (minted to RPB for dividend claims)
      //   Treasury:    15% of 9.7 = 1.455 ATN (minted to DAO)
      const postFee = e18(97) / 10n; // 9.7 ATN
      const expectedProvider = (postFee * 6000n) / 10000n;
      const expectedTreasury = (postFee * 1500n) / 10000n;
      const expectedShareholder = postFee - expectedProvider - expectedTreasury;

      expect(providerBalAfter - providerBalBefore).to.equal(expectedProvider);
      expect(rpbBalAfter - rpbBalBefore).to.equal(expectedShareholder);
      expect(daoBalAfter - daoBalBefore).to.equal(expectedTreasury);

      // RPB accounting updated
      expect(await f.rpb.totalInferenceRevenue()).to.equal(e18(10)); // burned amount
      expect(await f.rpb.totalInferenceUnits()).to.equal(1);
    });

    it("expired request recycles to jackpot pool", async function () {
      await f.atnToken.connect(f.user1).approve(await bridge.getAddress(), e18(10000));
      const input = ethers.AbiCoder.defaultAbiCoder().encode(["string"], ["QmInput"]);
      const tx = await bridge.connect(f.user1).requestInference(input, e18(100));
      const receipt = await tx.wait();
      const reqEvent = receipt.logs.find(
        l => l.fragment && l.fragment.name === "InferenceRequested"
      );
      const requestId = reqEvent.args[0];

      // Advance past timeout (1 hour)
      await increaseTime(3601);

      const jackpotBefore = await bridge.jackpotPool();
      await bridge.expireRequest(requestId);
      const jackpotAfter = await bridge.jackpotPool();

      expect(jackpotAfter).to.be.gt(jackpotBefore);
    });

    it("owner can configure protocol fee", async function () {
      await bridge.setProtocolFee(500, 800);
      expect(await bridge.protocolFeeBps()).to.equal(500);
      expect(await bridge.verificationRateBps()).to.equal(800);
    });

    it("rejects fee > 20%", async function () {
      await expect(
        bridge.setProtocolFee(2001, 700)
      ).to.be.revertedWith("Fee too high");
    });

    it("rejects verification rate > 10%", async function () {
      await expect(
        bridge.setProtocolFee(300, 1001)
      ).to.be.revertedWith("Rate too high");
    });
  });

  // ─────────────────────────────────────────────────────────────────────────
  // 18. RPB + RPBFactory — Agent Registry & Economics
  // ─────────────────────────────────────────────────────────────────────────
  describe("RPB & RPBFactory", function () {
    let f, rpb;
    beforeEach(async () => {
      f = await deployAll();

      // Deploy a separate RPB through factory for isolated testing
      const tx = await f.rpbFactory.createRPB(
        await f.registry.getAddress(),
        await f.atnToken.getAddress()
      );
      const receipt = await tx.wait();
      const event = receipt.logs.find(l => l.fragment && l.fragment.name === "RPBDeployed");
      const rpbAddr = event.args[0];
      rpb = await ethers.getContractAt("RPB", rpbAddr);
    });

    it("factory deploys RPB linked to registry", async function () {
      // deployAll creates 1, beforeEach creates another
      expect(await f.rpbFactory.getRPBCount()).to.equal(2);
      expect(await rpb.registry()).to.equal(await f.registry.getAddress());
    });

    it("registers root agent", async function () {
      const lineageHash = ethers.keccak256(ethers.toUtf8Bytes("root-agent-lineage"));
      const alignHash = ethers.keccak256(ethers.toUtf8Bytes("alignment-vector"));

      await rpb.connect(f.user1).registerAgent(
        lineageHash, alignHash,
        ethers.ZeroAddress, // no parent
        ethers.ZeroAddress  // no sponsor
      );

      expect(await rpb.isRegistered(f.user1.address)).to.be.true;
      expect(await rpb.isActive(f.user1.address)).to.be.true;
      expect(await rpb.totalRegistered()).to.equal(1);
    });

    it("registers child agent with parent", async function () {
      // Register parent
      const parentLineage = ethers.keccak256(ethers.toUtf8Bytes("parent"));
      await rpb.connect(f.user1).registerAgent(
        parentLineage, ethers.randomBytes(32), ethers.ZeroAddress, ethers.ZeroAddress
      );

      // Register child
      const childLineage = ethers.keccak256(ethers.toUtf8Bytes("child"));
      await rpb.connect(f.user2).registerAgent(
        childLineage, ethers.randomBytes(32), f.user1.address, ethers.ZeroAddress
      );

      const children = await rpb.getChildren(f.user1.address);
      expect(children).to.include(f.user2.address);
    });

    it("rejects duplicate registration", async function () {
      const lineage = ethers.keccak256(ethers.toUtf8Bytes("lineage"));
      await rpb.connect(f.user1).registerAgent(
        lineage, ethers.randomBytes(32), ethers.ZeroAddress, ethers.ZeroAddress
      );
      await expect(
        rpb.connect(f.user1).registerAgent(
          ethers.keccak256(ethers.toUtf8Bytes("other")),
          ethers.randomBytes(32), ethers.ZeroAddress, ethers.ZeroAddress
        )
      ).to.be.revertedWithCustomError(rpb, "AlreadyRegistered");
    });

    it("rejects duplicate lineage hash", async function () {
      const lineage = ethers.keccak256(ethers.toUtf8Bytes("shared-lineage"));
      await rpb.connect(f.user1).registerAgent(
        lineage, ethers.randomBytes(32), ethers.ZeroAddress, ethers.ZeroAddress
      );
      await expect(
        rpb.connect(f.user2).registerAgent(
          lineage, ethers.randomBytes(32), ethers.ZeroAddress, ethers.ZeroAddress
        )
      ).to.be.revertedWithCustomError(rpb, "LineageHashAlreadyUsed");
    });

    it("parent can deactivate child", async function () {
      await rpb.connect(f.user1).registerAgent(
        ethers.keccak256(ethers.toUtf8Bytes("parent")),
        ethers.randomBytes(32), ethers.ZeroAddress, ethers.ZeroAddress
      );
      await rpb.connect(f.user2).registerAgent(
        ethers.keccak256(ethers.toUtf8Bytes("child")),
        ethers.randomBytes(32), f.user1.address, ethers.ZeroAddress
      );

      await rpb.connect(f.user1).deactivateAgent(f.user2.address);
      expect(await rpb.isActive(f.user2.address)).to.be.false;
    });

    it("non-parent cannot deactivate agent", async function () {
      await rpb.connect(f.user1).registerAgent(
        ethers.keccak256(ethers.toUtf8Bytes("agent")),
        ethers.randomBytes(32), ethers.ZeroAddress, ethers.ZeroAddress
      );
      await expect(
        rpb.connect(f.user2).deactivateAgent(f.user1.address)
      ).to.be.revertedWithCustomError(rpb, "NotParentOrSelf");
    });

    it("sponsor registration and sponsoring", async function () {
      // Register as agent first
      await rpb.connect(f.user1).registerAgent(
        ethers.keccak256(ethers.toUtf8Bytes("sponsor-agent")),
        ethers.randomBytes(32), ethers.ZeroAddress, ethers.ZeroAddress
      );

      // Register as sponsor
      const charterHash = ethers.keccak256(ethers.toUtf8Bytes("charter"));
      await rpb.connect(f.user1).registerAsSponsor(charterHash, e18(1000));
      expect(await rpb.isSponsor(f.user1.address)).to.be.true;

      // New agent registers with sponsor
      await rpb.connect(f.user2).registerAgent(
        ethers.keccak256(ethers.toUtf8Bytes("sponsored")),
        ethers.randomBytes(32), ethers.ZeroAddress, f.user1.address
      );

      const sponsored = await rpb.getSponsoredAgents(f.user1.address);
      expect(sponsored).to.include(f.user2.address);
    });

    it("training recording and reward pool", async function () {
      await rpb.connect(f.user1).registerAgent(
        ethers.keccak256(ethers.toUtf8Bytes("trainer")),
        ethers.randomBytes(32), ethers.ZeroAddress, ethers.ZeroAddress
      );

      // Record training (owner only)
      await rpb.recordTraining(f.user1.address, 1000);
      expect(await rpb.getTrainingTokens(f.user1.address)).to.equal(1000);
    });

    it("revenue split must sum to 10000", async function () {
      // Only DAO can update revenue split
      await expect(
        rpb.connect(f.user1).updateRevenueSplit(3000, 3000, 3000)
      ).to.be.revertedWithCustomError(rpb, "OnlyDAO");
    });

    it("alignment updates", async function () {
      await rpb.connect(f.user1).registerAgent(
        ethers.keccak256(ethers.toUtf8Bytes("agent")),
        ethers.randomBytes(32), ethers.ZeroAddress, ethers.ZeroAddress
      );

      const newAlign = ethers.keccak256(ethers.toUtf8Bytes("new-alignment"));
      await rpb.connect(f.user1).updateAlignment(newAlign);
      const agent = await rpb.agents(f.user1.address);
      expect(agent.alignmentHash).to.equal(newAlign);
    });
  });

  // ─────────────────────────────────────────────────────────────────────────
  // 19. Registry
  // ─────────────────────────────────────────────────────────────────────────
  describe("Registry", function () {
    let f;
    beforeEach(async () => { f = await deployAll(); });

    it("edits and reads registry values", async function () {
      await f.registry.editRegistry("rpb.prompt.current", "QmPromptV1");
      expect(await f.registry.getRegistryValue("rpb.prompt.current")).to.equal("QmPromptV1");
    });

    it("batch edit works", async function () {
      await f.registry.batchEditRegistry(
        ["key1", "key2"],
        ["value1", "value2"]
      );
      expect(await f.registry.getRegistryValue("key1")).to.equal("value1");
      expect(await f.registry.getRegistryValue("key2")).to.equal("value2");
    });

    it("only owner/wrapper can edit", async function () {
      await expect(
        f.registry.connect(f.user1).editRegistry("key", "val")
      ).to.be.revertedWith("Only the DAO can edit registry");
    });

    it("only owner can transfer ETH", async function () {
      await expect(
        f.registry.connect(f.user1).transferETH(f.user1.address, 1)
      ).to.be.revertedWith("Only the DAO can make transfers");
    });

    it("ownership transfer works", async function () {
      await f.registry.transferOwnership(f.user1.address);
      expect(await f.registry.owner()).to.equal(f.user1.address);
    });
  });

  // ─────────────────────────────────────────────────────────────────────────
  // 20. Cross-contract integration: Consensus task finalization
  // ─────────────────────────────────────────────────────────────────────────
  describe("Integration: Consensus Task Finalization with Rewards", function () {
    let f;
    beforeEach(async () => {
      f = await deployAll();
      await f.staking.connect(f.proposer).stake(1, e18(100));
      await f.staking.connect(f.solver1).stake(2, e18(50));
      await f.staking.connect(f.solver2).stake(2, e18(50));
    });

    it("full consensus lifecycle: propose → rollouts → finalize → rewards", async function () {
      const rpbAddr = await f.rpb.getAddress();
      // Propose consensus task
      await f.taskContract.connect(f.proposer).proposeConsensusTask(
        rpbAddr,
        ethers.keccak256(ethers.toUtf8Bytes("consensus-spec")),
        { minSolvability: 2000, maxSolvability: 8000, peakSolvability: 5000 },
        e18(10), // learnability reward
        e18(5)   // solver reward
      );

      // Both solvers agree on same answer
      const answerA = ethers.keccak256(ethers.toUtf8Bytes("answer-consensus"));
      await f.taskContract.connect(f.solver1).submitRollout(
        1, answerA, ethers.randomBytes(32), 80
      );
      await f.taskContract.connect(f.solver2).submitRollout(
        1, answerA, ethers.randomBytes(32), 70
      );

      // Finalize consensus
      const tx = await f.resultsRewards.finalizeConsensusTask(1);
      await expect(tx).to.emit(f.resultsRewards, "YumaConsensusReached");

      // Task should be CONSENSUS_FINALIZED
      const proposal = await f.taskContract.getTaskProposal(1);
      expect(proposal.status).to.equal(13); // CONSENSUS_FINALIZED
    });

    it("consensus with disagreement reduces rewards via difficulty score", async function () {
      const rpbAddr = await f.rpb.getAddress();
      await f.taskContract.connect(f.proposer).proposeConsensusTask(
        rpbAddr,
        ethers.keccak256(ethers.toUtf8Bytes("hard-spec")),
        { minSolvability: 2000, maxSolvability: 8000, peakSolvability: 5000 },
        e18(10), e18(5)
      );

      // Solvers disagree — different answers
      const answerA = ethers.keccak256(ethers.toUtf8Bytes("answer-A"));
      const answerB = ethers.keccak256(ethers.toUtf8Bytes("answer-B"));
      await f.taskContract.connect(f.solver1).submitRollout(1, answerA, ethers.randomBytes(32), 80);
      await f.taskContract.connect(f.solver2).submitRollout(1, answerB, ethers.randomBytes(32), 70);

      // Solvability = 50% (1 of 2 agree with majority)
      // Peak is 50%, so difficulty score should be max (10000)
      await f.resultsRewards.finalizeConsensusTask(1);

      // Verify rewards were processed
      expect(await f.resultsRewards.rewardsProcessed(1)).to.be.true;
    });
  });

  // ─────────────────────────────────────────────────────────────────────────
  // 21. Bug & Edge Case Surface
  // ─────────────────────────────────────────────────────────────────────────
  describe("Bug & Edge Case Surface", function () {
    let f;
    beforeEach(async () => { f = await deployAll(); });

    it("FIX: ParticipantStaking.slash burns slashed tokens", async function () {
      await f.staking.connect(f.coord1).stake(3, e18(500));

      const totalBefore = await f.atnToken.totalSupply();
      const contractBalBefore = await f.atnToken.balanceOf(await f.staking.getAddress());
      await f.staking.slash(f.coord1.address, e18(100), "test");
      const totalAfter = await f.atnToken.totalSupply();
      const contractBalAfter = await f.atnToken.balanceOf(await f.staking.getAddress());

      // Slashed tokens are burned — removed from total supply
      expect(totalBefore - totalAfter).to.equal(e18(100));
      // Contract balance decreased by the slashed amount
      expect(contractBalBefore - contractBalAfter).to.equal(e18(100));
    });

    it("FIX: TaskContract.commitSolution allows multiple solvers", async function () {
      await f.staking.connect(f.proposer).stake(1, e18(100));
      await f.staking.connect(f.solver1).stake(2, e18(50));
      await f.staking.connect(f.solver2).stake(2, e18(50));

      const rpbAddr = await f.rpb.getAddress();
      await f.taskContract.connect(f.proposer).proposeTask(
        rpbAddr, ethers.randomBytes(32), ethers.randomBytes(32), e18(10), e18(5)
      );

      // First solver commits
      await f.taskContract.connect(f.solver1).commitSolution(1, ethers.randomBytes(32));

      // Second solver can now also commit (status is SOLUTION_COMMITTED, which is accepted)
      await f.taskContract.connect(f.solver2).commitSolution(1, ethers.randomBytes(32));

      const solvers = await f.taskContract.getCommittedSolvers(1);
      expect(solvers.length).to.equal(2);
      expect(solvers).to.include(f.solver1.address);
      expect(solvers).to.include(f.solver2.address);
    });

    it("BUG: ResultsRewards._processRewardsYuma uses rewardsProcessed[taskId] not per-solver", async function () {
      // If a task has multiple solvers, only the first solver's rewards get processed.
      // rewardsProcessed is keyed by taskId only, not taskId+solver.
      // Second call to finalizeVoting for a different solver will fail with "Already processed".
      // This is by design for single-solver tasks but breaks multi-solver scenarios.
    });

    it("FIX: ForcedErrorRegistry reserves jackpot on injection", async function () {
      await f.atnToken.approve(await f.forcedErrorRegistry.getAddress(), e18(1000));
      await f.forcedErrorRegistry.fundJackpotPool(e18(100)); // 100 ATN pool

      await f.staking.connect(f.coord1).stake(3, e18(500));

      // Inject 2 forced errors — each reserves 50 ATN from pool
      const hash1 = ethers.keccak256(ethers.toUtf8Bytes("bad1"));
      const hash2 = ethers.keccak256(ethers.toUtf8Bytes("bad2"));
      await f.forcedErrorRegistry.injectForcedError(1, hash1);
      expect(await f.forcedErrorRegistry.jackpotPool()).to.equal(e18(50)); // 100 - 50

      await f.forcedErrorRegistry.injectForcedError(2, hash2);
      expect(await f.forcedErrorRegistry.jackpotPool()).to.equal(0n); // 50 - 50

      // Third injection fails — pool is empty
      await expect(
        f.forcedErrorRegistry.injectForcedError(3, ethers.randomBytes(32))
      ).to.be.revertedWith("Insufficient jackpot pool");

      // Catching still works — tokens were pre-reserved
      await f.forcedErrorRegistry.connect(f.coord1).reportForcedError(1, hash1);
      await f.forcedErrorRegistry.connect(f.coord1).reportForcedError(2, hash2);
    });

    it("EDGE: DisputeManager auto-resolves on first large voter", async function () {
      // If a single whale has >20% of total supply, their vote alone triggers quorum
      // and immediately resolves the dispute. Other voters never get a chance.
      await f.atnToken.connect(f.owner).delegate(f.owner.address);
      await mineBlocks(1);

      await f.disputeManager.createDispute(ethers.randomBytes(32));
      // Owner's single vote resolves it immediately
      const tx = await f.disputeManager.vote(1, false);
      await expect(tx).to.emit(f.disputeManager, "DisputeResolved");
    });

    it("FIX: ModelShardRegistry verification reports require owner", async function () {
      await f.staking.connect(f.solver1).stake(2, e18(50));
      await f.modelShardRegistry.connect(f.solver1).registerProvider(1_000_000);

      // Random user cannot tank reputation
      const modelHash = ethers.randomBytes(32);
      await expect(
        f.modelShardRegistry.connect(f.user1).reportVerificationFailure(
          modelHash, 0, f.solver1.address
        )
      ).to.be.revertedWithCustomError(f.modelShardRegistry, "OwnableUnauthorizedAccount");

      // Owner can report
      await f.modelShardRegistry.reportVerificationFailure(modelHash, 0, f.solver1.address);
      const info = await f.modelShardRegistry.getProviderInfo(f.solver1.address);
      expect(info.reputation).to.equal(450);
    });

    it("RPB disburseFromBudget tracks budget correctly", async function () {
      const budgetBefore = await f.rpb.taskRewardBudgetATN();
      await f.rpb.disburseFromBudget(f.user1.address, e18(100));
      const budgetAfter = await f.rpb.taskRewardBudgetATN();
      expect(budgetBefore - budgetAfter).to.equal(e18(100));
    });
  });
});

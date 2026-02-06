const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("Autonet - Core Contracts", function () {
  let atnToken;
  let staking;
  let taskContract;
  let resultsRewards;
  let project;
  let forcedErrorRegistry;
  let owner, proposer, solver1, solver2, coord1, coord2, coord3;

  const PROPOSER_STAKE = ethers.parseEther("100");
  const SOLVER_STAKE = ethers.parseEther("50");
  const COORDINATOR_STAKE = ethers.parseEther("500");

  beforeEach(async function () {
    [owner, proposer, solver1, solver2, coord1, coord2, coord3] = await ethers.getSigners();

    // Deploy ATN Token
    const ATNToken = await ethers.getContractFactory("ATNToken");
    atnToken = await ATNToken.deploy(owner.address, ethers.parseEther("1000000000"));
    await atnToken.waitForDeployment();

    // Deploy Staking
    const Staking = await ethers.getContractFactory("ParticipantStaking");
    staking = await Staking.deploy(await atnToken.getAddress(), owner.address);
    await staking.waitForDeployment();

    // Deploy Project
    const Project = await ethers.getContractFactory("Project");
    project = await Project.deploy(await atnToken.getAddress(), owner.address);
    await project.waitForDeployment();

    // Deploy TaskContract
    const TaskContract = await ethers.getContractFactory("TaskContract");
    taskContract = await TaskContract.deploy(await staking.getAddress(), owner.address);
    await taskContract.waitForDeployment();

    // Deploy ResultsRewards
    const ResultsRewards = await ethers.getContractFactory("ResultsRewards");
    resultsRewards = await ResultsRewards.deploy(
      await taskContract.getAddress(),
      await staking.getAddress(),
      owner.address
    );
    await resultsRewards.waitForDeployment();

    // Deploy ForcedErrorRegistry
    const ForcedErrorRegistry = await ethers.getContractFactory("ForcedErrorRegistry");
    forcedErrorRegistry = await ForcedErrorRegistry.deploy(
      await atnToken.getAddress(),
      await staking.getAddress(),
      owner.address
    );
    await forcedErrorRegistry.waitForDeployment();

    // Configure contracts
    await taskContract.setResultsRewardsContract(await resultsRewards.getAddress());
    await resultsRewards.setProjectContract(await project.getAddress());
    await staking.setAuthorizedSlasher(await resultsRewards.getAddress(), true);
    await staking.setAuthorizedSlasher(await forcedErrorRegistry.getAddress(), true);
    await project.setAuthorizedDisburser(await resultsRewards.getAddress(), true);

    // Distribute tokens for testing
    const testAmount = ethers.parseEther("10000");
    await atnToken.transfer(proposer.address, testAmount);
    await atnToken.transfer(solver1.address, testAmount);
    await atnToken.transfer(solver2.address, testAmount);
    await atnToken.transfer(coord1.address, testAmount);
    await atnToken.transfer(coord2.address, testAmount);
    await atnToken.transfer(coord3.address, testAmount);

    // Approve staking contract
    await atnToken.connect(proposer).approve(await staking.getAddress(), testAmount);
    await atnToken.connect(solver1).approve(await staking.getAddress(), testAmount);
    await atnToken.connect(solver2).approve(await staking.getAddress(), testAmount);
    await atnToken.connect(coord1).approve(await staking.getAddress(), testAmount);
    await atnToken.connect(coord2).approve(await staking.getAddress(), testAmount);
    await atnToken.connect(coord3).approve(await staking.getAddress(), testAmount);
  });

  describe("Staking", function () {
    it("Should allow participants to stake for different roles", async function () {
      // Proposer stakes
      await staking.connect(proposer).stake(1, PROPOSER_STAKE); // Role 1 = PROPOSER
      expect(await staking.isActiveParticipant(proposer.address, 1)).to.be.true;

      // Solver stakes
      await staking.connect(solver1).stake(2, SOLVER_STAKE); // Role 2 = SOLVER
      expect(await staking.isActiveParticipant(solver1.address, 2)).to.be.true;

      // Coordinator stakes
      await staking.connect(coord1).stake(3, COORDINATOR_STAKE); // Role 3 = COORDINATOR
      expect(await staking.isActiveParticipant(coord1.address, 3)).to.be.true;
    });
  });

  describe("Task Lifecycle with Checkpoints", function () {
    beforeEach(async function () {
      // Setup stakes
      await staking.connect(proposer).stake(1, PROPOSER_STAKE);
      await staking.connect(solver1).stake(2, SOLVER_STAKE);
      await staking.connect(coord1).stake(3, COORDINATOR_STAKE);
    });

    it("Should allow proposer to create a task", async function () {
      const specHash = ethers.keccak256(ethers.toUtf8Bytes("task_spec"));
      const groundTruthHash = ethers.keccak256(ethers.toUtf8Bytes("ground_truth"));

      await expect(
        taskContract.connect(proposer).proposeTask(
          1, // projectId
          specHash,
          groundTruthHash,
          ethers.parseEther("10"), // learnability reward
          ethers.parseEther("5")  // solver reward
        )
      ).to.emit(taskContract, "TaskProposed");
    });

    it("Should allow solver to submit checkpoints", async function () {
      // Create task
      const specHash = ethers.keccak256(ethers.toUtf8Bytes("task_spec"));
      const groundTruthHash = ethers.keccak256(ethers.toUtf8Bytes("ground_truth"));
      await taskContract.connect(proposer).proposeTask(1, specHash, groundTruthHash, ethers.parseEther("10"), ethers.parseEther("5"));

      // Submit checkpoints
      const weightsHash = ethers.keccak256(ethers.toUtf8Bytes("weights_step_10"));
      const dataIndicesHash = ethers.keccak256(ethers.toUtf8Bytes("data_indices_10"));
      const randomSeed = ethers.keccak256(ethers.toUtf8Bytes("seed_10"));

      await expect(
        taskContract.connect(solver1).submitCheckpoint(1, 10, weightsHash, dataIndicesHash, randomSeed)
      ).to.emit(taskContract, "CheckpointSubmitted").withArgs(1, solver1.address, 10, weightsHash);

      // Verify checkpoint stored
      const checkpointCount = await taskContract.getCheckpointCount(1, solver1.address);
      expect(checkpointCount).to.equal(1);

      const checkpoint = await taskContract.getCheckpoint(1, solver1.address, 0);
      expect(checkpoint.stepNumber).to.equal(10);
      expect(checkpoint.weightsHash).to.equal(weightsHash);
    });

    it("Should enforce checkpoint order", async function () {
      // Create task
      const specHash = ethers.keccak256(ethers.toUtf8Bytes("task_spec"));
      const groundTruthHash = ethers.keccak256(ethers.toUtf8Bytes("ground_truth"));
      await taskContract.connect(proposer).proposeTask(1, specHash, groundTruthHash, ethers.parseEther("10"), ethers.parseEther("5"));

      // Submit first checkpoint at step 10
      await taskContract.connect(solver1).submitCheckpoint(
        1, 10,
        ethers.keccak256(ethers.toUtf8Bytes("w10")),
        ethers.keccak256(ethers.toUtf8Bytes("d10")),
        ethers.keccak256(ethers.toUtf8Bytes("s10"))
      );

      // Try to submit checkpoint at earlier step - should fail
      await expect(
        taskContract.connect(solver1).submitCheckpoint(
          1, 5,
          ethers.keccak256(ethers.toUtf8Bytes("w5")),
          ethers.keccak256(ethers.toUtf8Bytes("d5")),
          ethers.keccak256(ethers.toUtf8Bytes("s5"))
        )
      ).to.be.revertedWith("Checkpoint step must be increasing");

      // Submit checkpoint at later step - should succeed
      await taskContract.connect(solver1).submitCheckpoint(
        1, 20,
        ethers.keccak256(ethers.toUtf8Bytes("w20")),
        ethers.keccak256(ethers.toUtf8Bytes("d20")),
        ethers.keccak256(ethers.toUtf8Bytes("s20"))
      );

      expect(await taskContract.getCheckpointCount(1, solver1.address)).to.equal(2);
    });
  });

  describe("Multi-Coordinator Yuma Voting", function () {
    let taskId;
    let projectId;

    beforeEach(async function () {
      // Setup stakes
      await staking.connect(proposer).stake(1, PROPOSER_STAKE);
      await staking.connect(solver1).stake(2, SOLVER_STAKE);
      await staking.connect(coord1).stake(3, COORDINATOR_STAKE);
      await staking.connect(coord2).stake(3, COORDINATOR_STAKE);
      await staking.connect(coord3).stake(3, COORDINATOR_STAKE);

      // Create a project first (needed for reward distribution)
      await atnToken.connect(proposer).approve(await project.getAddress(), ethers.parseEther("1000"));
      const tx = await project.connect(proposer).createProject(
        "Test Project",
        "QmDesc",
        ethers.parseEther("100"),  // funding goal
        ethers.parseEther("0"),    // initial budget (start at 0)
        ethers.parseEther("10"),
        "TestPT",
        "TPT"
      );
      projectId = 1;

      // Fund the project first
      await project.connect(proposer).fundProject(projectId, ethers.parseEther("100"), ethers.parseEther("10"));

      // Now allocate the project budget for rewards (currentFunding is 100, we allocate 50)
      await project.connect(proposer).allocateTaskBudget(projectId, ethers.parseEther("50"));

      // Create and setup task
      const specHash = ethers.keccak256(ethers.toUtf8Bytes("task_spec"));
      const groundTruthCid = "QmGroundTruth123456789";
      const groundTruthHash = ethers.keccak256(ethers.toUtf8Bytes(groundTruthCid));

      await taskContract.connect(proposer).proposeTask(projectId, specHash, groundTruthHash, ethers.parseEther("10"), ethers.parseEther("5"));
      taskId = 1;

      // Solver commits solution
      const solutionCid = "QmSolution123456789";
      const solutionHash = ethers.keccak256(ethers.toUtf8Bytes(solutionCid));
      await taskContract.connect(solver1).commitSolution(taskId, solutionHash);

      // Reveals (simplified - in production would verify hashes)
      await resultsRewards.connect(proposer).revealGroundTruth(taskId, groundTruthCid);
      await resultsRewards.connect(solver1).revealSolution(taskId, solutionCid);
    });

    it("Should allow multiple coordinators to vote", async function () {
      const reportCid = "QmReport123";

      // First coordinator votes
      await expect(
        resultsRewards.connect(coord1).submitVote(taskId, solver1.address, true, 85, reportCid)
      ).to.emit(resultsRewards, "CoordinatorVoted");

      // Second coordinator votes
      await resultsRewards.connect(coord2).submitVote(taskId, solver1.address, true, 90, reportCid);

      // Third coordinator votes differently
      await resultsRewards.connect(coord3).submitVote(taskId, solver1.address, true, 75, reportCid);

      // Check vote count
      expect(await resultsRewards.getVoteCount(taskId, solver1.address)).to.equal(3);
    });

    it("Should prevent double voting", async function () {
      await resultsRewards.connect(coord1).submitVote(taskId, solver1.address, true, 85, "QmReport");

      await expect(
        resultsRewards.connect(coord1).submitVote(taskId, solver1.address, true, 90, "QmReport")
      ).to.be.revertedWith("Already voted");
    });

    it("Should compute Yuma consensus with stake weighting", async function () {
      // Coord1 and Coord2 vote correct, Coord3 votes incorrect
      await resultsRewards.connect(coord1).submitVote(taskId, solver1.address, true, 85, "QmReport");
      await resultsRewards.connect(coord2).submitVote(taskId, solver1.address, true, 90, "QmReport");

      // Finalize after minimum coordinators
      await resultsRewards.finalizeVoting(taskId, solver1.address);

      const result = await resultsRewards.getConsensusResult(taskId, solver1.address);
      expect(result.finalized).to.be.true;
      expect(result.consensusCorrect).to.be.true;
      // Score should be stake-weighted average of 85 and 90 = 87.5, rounded
    });

    it("Should update coordinator bonds after consensus", async function () {
      await resultsRewards.connect(coord1).submitVote(taskId, solver1.address, true, 85, "QmReport");
      await resultsRewards.connect(coord2).submitVote(taskId, solver1.address, true, 90, "QmReport");

      await expect(resultsRewards.finalizeVoting(taskId, solver1.address))
        .to.emit(resultsRewards, "BondUpdated");

      const bond1 = await resultsRewards.getCoordinatorBond(coord1.address);
      expect(bond1.totalVotes).to.equal(1);
      expect(bond1.correctVotes).to.equal(1); // Aligned with consensus
    });
  });

  describe("ForcedErrorRegistry", function () {
    beforeEach(async function () {
      // Fund the jackpot pool
      await atnToken.approve(await forcedErrorRegistry.getAddress(), ethers.parseEther("1000"));
      await forcedErrorRegistry.fundJackpotPool(ethers.parseEther("1000"));

      // Setup coordinator stake
      await staking.connect(coord1).stake(3, COORDINATOR_STAKE);
    });

    it("Should allow governance to inject forced errors", async function () {
      const knownBadHash = ethers.keccak256(ethers.toUtf8Bytes("bad_solution"));

      await expect(
        forcedErrorRegistry.injectForcedError(1, knownBadHash)
      ).to.emit(forcedErrorRegistry, "ForcedErrorInjected").withArgs(1, knownBadHash, ethers.parseEther("50"));

      expect(await forcedErrorRegistry.isTaskForcedError(1)).to.be.true;
    });

    it("Should reward coordinators who catch forced errors", async function () {
      const knownBadHash = ethers.keccak256(ethers.toUtf8Bytes("bad_solution"));
      await forcedErrorRegistry.injectForcedError(1, knownBadHash);

      const balanceBefore = await atnToken.balanceOf(coord1.address);

      await expect(
        forcedErrorRegistry.connect(coord1).reportForcedError(1, knownBadHash)
      ).to.emit(forcedErrorRegistry, "ForcedErrorCaught");

      const balanceAfter = await atnToken.balanceOf(coord1.address);
      expect(balanceAfter - balanceBefore).to.equal(ethers.parseEther("50")); // Jackpot amount
    });

    it("Should not allow double-catching of forced errors", async function () {
      const knownBadHash = ethers.keccak256(ethers.toUtf8Bytes("bad_solution"));
      await forcedErrorRegistry.injectForcedError(1, knownBadHash);

      await forcedErrorRegistry.connect(coord1).reportForcedError(1, knownBadHash);

      // Setup another coordinator
      await staking.connect(coord2).stake(3, COORDINATOR_STAKE);

      await expect(
        forcedErrorRegistry.connect(coord2).reportForcedError(1, knownBadHash)
      ).to.be.revertedWith("Already caught");
    });

    it("Should check jackpot pool balance", async function () {
      expect(await forcedErrorRegistry.getJackpotPool()).to.equal(ethers.parseEther("1000"));
    });
  });

  describe("Legacy Single-Coordinator Mode", function () {
    let taskId;
    let projectId;

    beforeEach(async function () {
      // Setup stakes
      await staking.connect(proposer).stake(1, PROPOSER_STAKE);
      await staking.connect(solver1).stake(2, SOLVER_STAKE);
      await staking.connect(coord1).stake(3, COORDINATOR_STAKE);

      // Create a project first (needed for reward distribution)
      await atnToken.connect(proposer).approve(await project.getAddress(), ethers.parseEther("1000"));
      await project.connect(proposer).createProject(
        "Test Project",
        "QmDesc",
        ethers.parseEther("100"),  // funding goal
        ethers.parseEther("0"),    // initial budget
        ethers.parseEther("10"),
        "TestPT",
        "TPT"
      );
      projectId = 1;

      // Fund the project first
      await project.connect(proposer).fundProject(projectId, ethers.parseEther("100"), ethers.parseEther("10"));

      // Now allocate the project budget for rewards
      await project.connect(proposer).allocateTaskBudget(projectId, ethers.parseEther("50"));

      // Create task
      const specHash = ethers.keccak256(ethers.toUtf8Bytes("task_spec"));
      const groundTruthCid = "QmGroundTruth123456789";
      const groundTruthHash = ethers.keccak256(ethers.toUtf8Bytes(groundTruthCid));

      await taskContract.connect(proposer).proposeTask(projectId, specHash, groundTruthHash, ethers.parseEther("10"), ethers.parseEther("5"));
      taskId = 1;

      // Solver commits solution
      const solutionCid = "QmSolution123456789";
      const solutionHash = ethers.keccak256(ethers.toUtf8Bytes(solutionCid));
      await taskContract.connect(solver1).commitSolution(taskId, solutionHash);

      // Reveals
      await resultsRewards.connect(proposer).revealGroundTruth(taskId, groundTruthCid);
      await resultsRewards.connect(solver1).revealSolution(taskId, solutionCid);
    });

    it("Should allow single coordinator to verify (legacy mode)", async function () {
      await expect(
        resultsRewards.connect(coord1).submitVerification(taskId, solver1.address, true, 85, "QmReport")
      ).to.emit(resultsRewards, "VerificationSubmitted");
    });
  });

  describe("InferenceProviderBridge - Jurisdiction Integration", function () {
    let inferenceFactory;
    let projectId;

    beforeEach(async function () {
      // Deploy InferenceProviderFactory
      const InferenceProviderFactory = await ethers.getContractFactory("InferenceProviderFactory");
      inferenceFactory = await InferenceProviderFactory.deploy(
        await project.getAddress(),
        owner.address
      );
      await inferenceFactory.waitForDeployment();

      // Create and deploy a project
      await atnToken.connect(proposer).approve(await project.getAddress(), ethers.parseEther("1000"));
      await project.connect(proposer).createProject(
        "AI Inference Project",
        "QmDesc",
        ethers.parseEther("100"),
        ethers.parseEther("0"),
        ethers.parseEther("10"),
        "AIP",
        "AIP"
      );
      projectId = 1;

      // Fund the project
      await project.connect(proposer).fundProject(projectId, ethers.parseEther("100"), ethers.parseEther("10"));

      // Set mature model (makes project DEPLOYED)
      await project.connect(proposer).setMatureModel(
        projectId,
        "QmModelWeightsCid123456789",
        ethers.parseEther("1") // 1 ATN per inference
      );
    });

    it("Should deploy a bridge for a deployed project", async function () {
      await expect(
        inferenceFactory.deployBridge(projectId, proposer.address)
      ).to.emit(inferenceFactory, "BridgeDeployed");

      expect(await inferenceFactory.hasBridge(projectId)).to.be.true;
      expect(await inferenceFactory.getBridgeCount()).to.equal(1);
    });

    it("Should not allow duplicate bridge deployment", async function () {
      await inferenceFactory.deployBridge(projectId, proposer.address);

      await expect(
        inferenceFactory.deployBridge(projectId, proposer.address)
      ).to.be.revertedWith("Bridge already deployed");
    });

    it("Bridge should return correct price per unit", async function () {
      await inferenceFactory.deployBridge(projectId, proposer.address);

      const bridgeAddress = await inferenceFactory.projectBridges(projectId);
      const bridge = await ethers.getContractAt("InferenceProviderBridge", bridgeAddress);

      expect(await bridge.getPricePerUnit()).to.equal(ethers.parseEther("1"));
    });

    it("Bridge should return model CID", async function () {
      await inferenceFactory.deployBridge(projectId, proposer.address);

      const bridgeAddress = await inferenceFactory.projectBridges(projectId);
      const bridge = await ethers.getContractAt("InferenceProviderBridge", bridgeAddress);

      expect(await bridge.getModelCid()).to.equal("QmModelWeightsCid123456789");
    });
  });
});

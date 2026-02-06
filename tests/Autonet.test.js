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

  describe("ModelShardRegistry - Distributed Model Storage", function () {
    let shardRegistry;
    let provider1, provider2, provider3;
    const MIN_STORAGE_STAKE = ethers.parseEther("50");

    beforeEach(async function () {
      [owner, proposer, solver1, solver2, coord1, coord2, coord3, provider1, provider2, provider3] = await ethers.getSigners();

      // Deploy ModelShardRegistry
      const ModelShardRegistry = await ethers.getContractFactory("ModelShardRegistry");
      shardRegistry = await ModelShardRegistry.deploy(
        await staking.getAddress(),
        owner.address
      );
      await shardRegistry.waitForDeployment();

      // Distribute tokens to providers
      const testAmount = ethers.parseEther("10000");
      await atnToken.transfer(provider1.address, testAmount);
      await atnToken.transfer(provider2.address, testAmount);
      await atnToken.transfer(provider3.address, testAmount);

      // Approve staking contract for providers
      await atnToken.connect(provider1).approve(await staking.getAddress(), testAmount);
      await atnToken.connect(provider2).approve(await staking.getAddress(), testAmount);
      await atnToken.connect(provider3).approve(await staking.getAddress(), testAmount);
    });

    describe("Provider Registration", function () {
      it("Should allow provider registration with sufficient stake", async function () {
        // Stake sufficient amount
        await staking.connect(provider1).stake(2, MIN_STORAGE_STAKE); // Using SOLVER role for minimum stake

        const capacityBytes = ethers.parseUnits("1000000", 0); // 1MB capacity
        await expect(
          shardRegistry.connect(provider1).registerProvider(capacityBytes)
        ).to.emit(shardRegistry, "ProviderRegistered").withArgs(provider1.address, capacityBytes);

        // Verify provider info
        const [capacity, used, reputation, successfulVerifications, failedVerifications, active] =
          await shardRegistry.getProviderInfo(provider1.address);
        expect(capacity).to.equal(capacityBytes);
        expect(used).to.equal(0);
        expect(reputation).to.equal(500); // Starts at 50%
        expect(successfulVerifications).to.equal(0);
        expect(failedVerifications).to.equal(0);
        expect(active).to.be.true;
      });

      it("Should reject provider registration without sufficient stake", async function () {
        // Try to register without staking (stake = 0 < 50 ATN minimum)
        await expect(
          shardRegistry.connect(provider1).registerProvider(ethers.parseUnits("1000000", 0))
        ).to.be.revertedWith("Insufficient stake");
      });

      it("Should reject zero capacity", async function () {
        await staking.connect(provider1).stake(2, MIN_STORAGE_STAKE);

        await expect(
          shardRegistry.connect(provider1).registerProvider(0)
        ).to.be.revertedWith("Capacity must be positive");
      });

      it("Should allow provider to update capacity", async function () {
        await staking.connect(provider1).stake(2, MIN_STORAGE_STAKE);
        await shardRegistry.connect(provider1).registerProvider(ethers.parseUnits("1000000", 0));

        const newCapacity = ethers.parseUnits("2000000", 0);
        await shardRegistry.connect(provider1).updateCapacity(newCapacity);

        const [capacity] = await shardRegistry.getProviderInfo(provider1.address);
        expect(capacity).to.equal(newCapacity);
      });

      it("Should allow provider to deactivate", async function () {
        await staking.connect(provider1).stake(2, MIN_STORAGE_STAKE);
        await shardRegistry.connect(provider1).registerProvider(ethers.parseUnits("1000000", 0));

        await shardRegistry.connect(provider1).deactivateProvider();

        const [, , , , , active] = await shardRegistry.getProviderInfo(provider1.address);
        expect(active).to.be.false;
      });
    });

    describe("Model Registration", function () {
      it("Should allow model registration with shard manifest", async function () {
        const modelHash = ethers.keccak256(ethers.toUtf8Bytes("model_v1"));
        const manifestCid = "QmManifest123456789";
        const merkleRoot = ethers.keccak256(ethers.toUtf8Bytes("merkle_root"));
        const dataShards = 10;
        const parityShards = 4;
        const totalSize = ethers.parseUnits("100000000", 0); // 100MB

        await expect(
          shardRegistry.connect(proposer).registerModel(
            modelHash,
            manifestCid,
            merkleRoot,
            dataShards,
            parityShards,
            totalSize,
            0, // StorageTier.IPFS_PUBLIC
            1, // ShardingStrategy.TENSOR_PARALLEL
            1  // projectId
          )
        ).to.emit(shardRegistry, "ModelRegistered").withArgs(modelHash, manifestCid, dataShards + parityShards, 1);

        // Verify model manifest
        const manifest = await shardRegistry.getModelManifest(modelHash);
        expect(manifest.manifestCid).to.equal(manifestCid);
        expect(manifest.merkleRoot).to.equal(merkleRoot);
        expect(manifest.totalShards).to.equal(dataShards + parityShards);
        expect(manifest.dataShards).to.equal(dataShards);
        expect(manifest.totalSize).to.equal(totalSize);
      });

      it("Should reject duplicate model registration", async function () {
        const modelHash = ethers.keccak256(ethers.toUtf8Bytes("model_v1"));
        const manifestCid = "QmManifest123456789";
        const merkleRoot = ethers.keccak256(ethers.toUtf8Bytes("merkle_root"));

        await shardRegistry.connect(proposer).registerModel(
          modelHash, manifestCid, merkleRoot, 10, 4,
          ethers.parseUnits("100000000", 0), 0, 1, 1
        );

        await expect(
          shardRegistry.connect(proposer).registerModel(
            modelHash, manifestCid, merkleRoot, 10, 4,
            ethers.parseUnits("100000000", 0), 0, 1, 1
          )
        ).to.be.revertedWith("Model already registered");
      });

      it("Should reject model with zero data shards", async function () {
        const modelHash = ethers.keccak256(ethers.toUtf8Bytes("model_v1"));

        await expect(
          shardRegistry.connect(proposer).registerModel(
            modelHash, "QmManifest", ethers.keccak256(ethers.toUtf8Bytes("root")),
            0, 4, ethers.parseUnits("100000000", 0), 0, 1, 1
          )
        ).to.be.revertedWith("Need at least 1 data shard");
      });
    });

    describe("Shard Announcements", function () {
      let modelHash;
      const shardSize = ethers.parseUnits("10000000", 0); // 10MB per shard

      beforeEach(async function () {
        // Register model
        modelHash = ethers.keccak256(ethers.toUtf8Bytes("model_v1"));
        await shardRegistry.connect(proposer).registerModel(
          modelHash,
          "QmManifest123",
          ethers.keccak256(ethers.toUtf8Bytes("merkle_root")),
          10, 4,
          ethers.parseUnits("100000000", 0),
          0, 1, 1
        );

        // Register providers
        await staking.connect(provider1).stake(2, MIN_STORAGE_STAKE);
        await staking.connect(provider2).stake(2, MIN_STORAGE_STAKE);
        await shardRegistry.connect(provider1).registerProvider(ethers.parseUnits("50000000", 0));
        await shardRegistry.connect(provider2).registerProvider(ethers.parseUnits("50000000", 0));
      });

      it("Should allow provider to announce shard", async function () {
        const shardIndex = 0;
        const shardHash = ethers.keccak256(ethers.toUtf8Bytes("shard_0"));

        await expect(
          shardRegistry.connect(provider1).announceShard(
            modelHash, shardIndex, shardHash, shardSize, false
          )
        ).to.emit(shardRegistry, "ShardAnnounced").withArgs(modelHash, shardIndex, provider1.address);

        // Verify provider list
        const providers = await shardRegistry.getShardProviders(modelHash, shardIndex);
        expect(providers.length).to.equal(1);
        expect(providers[0]).to.equal(provider1.address);

        // Verify storage used
        const [, used] = await shardRegistry.getProviderInfo(provider1.address);
        expect(used).to.equal(shardSize);
      });

      it("Should allow multiple providers to announce same shard", async function () {
        const shardIndex = 0;
        const shardHash = ethers.keccak256(ethers.toUtf8Bytes("shard_0"));

        await shardRegistry.connect(provider1).announceShard(
          modelHash, shardIndex, shardHash, shardSize, false
        );
        await shardRegistry.connect(provider2).announceShard(
          modelHash, shardIndex, shardHash, shardSize, false
        );

        const providers = await shardRegistry.getShardProviders(modelHash, shardIndex);
        expect(providers.length).to.equal(2);
        expect(providers).to.include(provider1.address);
        expect(providers).to.include(provider2.address);
      });

      it("Should reject announcement with mismatched shard hash", async function () {
        const shardIndex = 0;
        const shardHash1 = ethers.keccak256(ethers.toUtf8Bytes("shard_0"));
        const shardHash2 = ethers.keccak256(ethers.toUtf8Bytes("shard_0_different"));

        await shardRegistry.connect(provider1).announceShard(
          modelHash, shardIndex, shardHash1, shardSize, false
        );

        await expect(
          shardRegistry.connect(provider2).announceShard(
            modelHash, shardIndex, shardHash2, shardSize, false
          )
        ).to.be.revertedWith("Shard hash mismatch");
      });

      it("Should reject announcement if provider exceeds capacity", async function () {
        const largeShardSize = ethers.parseUnits("100000000", 0); // 100MB - exceeds provider capacity

        await expect(
          shardRegistry.connect(provider1).announceShard(
            modelHash, 0, ethers.keccak256(ethers.toUtf8Bytes("shard_0")), largeShardSize, false
          )
        ).to.be.revertedWith("Capacity exceeded");
      });

      it("Should reject duplicate announcement from same provider", async function () {
        const shardIndex = 0;
        const shardHash = ethers.keccak256(ethers.toUtf8Bytes("shard_0"));

        await shardRegistry.connect(provider1).announceShard(
          modelHash, shardIndex, shardHash, shardSize, false
        );

        await expect(
          shardRegistry.connect(provider1).announceShard(
            modelHash, shardIndex, shardHash, shardSize, false
          )
        ).to.be.revertedWith("Already storing this shard");
      });

      it("Should reject announcement for unregistered model", async function () {
        const fakeModelHash = ethers.keccak256(ethers.toUtf8Bytes("fake_model"));

        await expect(
          shardRegistry.connect(provider1).announceShard(
            fakeModelHash, 0, ethers.keccak256(ethers.toUtf8Bytes("shard")), shardSize, false
          )
        ).to.be.revertedWith("Model not registered");
      });

      it("Should reject announcement with invalid shard index", async function () {
        await expect(
          shardRegistry.connect(provider1).announceShard(
            modelHash, 100, ethers.keccak256(ethers.toUtf8Bytes("shard")), shardSize, false
          )
        ).to.be.revertedWith("Invalid shard index");
      });
    });

    describe("Shard Verification with Merkle Proofs", function () {
      let modelHash, merkleRoot;

      beforeEach(async function () {
        // Setup model with Merkle tree
        // For simplicity, create a small Merkle tree: 4 shards (2 data, 2 parity)
        const shard0Hash = ethers.keccak256(ethers.toUtf8Bytes("shard_0"));
        const shard1Hash = ethers.keccak256(ethers.toUtf8Bytes("shard_1"));
        const shard2Hash = ethers.keccak256(ethers.toUtf8Bytes("shard_2"));
        const shard3Hash = ethers.keccak256(ethers.toUtf8Bytes("shard_3"));

        // Build Merkle tree
        const level1_left = ethers.keccak256(ethers.concat([shard0Hash, shard1Hash]));
        const level1_right = ethers.keccak256(ethers.concat([shard2Hash, shard3Hash]));
        merkleRoot = ethers.keccak256(ethers.concat([level1_left, level1_right]));

        modelHash = ethers.keccak256(ethers.toUtf8Bytes("model_merkle"));
        await shardRegistry.connect(proposer).registerModel(
          modelHash, "QmManifest", merkleRoot, 2, 2,
          ethers.parseUnits("40000000", 0), 0, 1, 1
        );

        // Register provider and announce shard
        await staking.connect(provider1).stake(2, MIN_STORAGE_STAKE);
        await shardRegistry.connect(provider1).registerProvider(ethers.parseUnits("50000000", 0));
        await shardRegistry.connect(provider1).announceShard(
          modelHash, 0, shard0Hash, ethers.parseUnits("10000000", 0), false
        );
      });

      it("Should verify shard with correct Merkle proof", async function () {
        const shardHash = ethers.keccak256(ethers.toUtf8Bytes("shard_0"));
        const shard1Hash = ethers.keccak256(ethers.toUtf8Bytes("shard_1"));
        const shard2Hash = ethers.keccak256(ethers.toUtf8Bytes("shard_2"));
        const shard3Hash = ethers.keccak256(ethers.toUtf8Bytes("shard_3"));

        // Merkle proof for shard 0: [shard1Hash, level1_right]
        const level1_right = ethers.keccak256(ethers.concat([shard2Hash, shard3Hash]));
        const merkleProof = [shard1Hash, level1_right];

        await expect(
          shardRegistry.connect(owner).verifyShard(modelHash, 0, shardHash, merkleProof)
        ).to.emit(shardRegistry, "ShardVerified").withArgs(modelHash, 0, owner.address, true);
      });

      it("Should reject verification with wrong shard hash", async function () {
        const wrongHash = ethers.keccak256(ethers.toUtf8Bytes("wrong_shard"));
        const shard1Hash = ethers.keccak256(ethers.toUtf8Bytes("shard_1"));
        const shard2Hash = ethers.keccak256(ethers.toUtf8Bytes("shard_2"));
        const shard3Hash = ethers.keccak256(ethers.toUtf8Bytes("shard_3"));
        const level1_right = ethers.keccak256(ethers.concat([shard2Hash, shard3Hash]));
        const merkleProof = [shard1Hash, level1_right];

        await expect(
          shardRegistry.connect(owner).verifyShard(modelHash, 0, wrongHash, merkleProof)
        ).to.emit(shardRegistry, "ShardVerified").withArgs(modelHash, 0, owner.address, false);
      });
    });

    describe("Shard Availability Check", function () {
      let modelHash;

      beforeEach(async function () {
        modelHash = ethers.keccak256(ethers.toUtf8Bytes("model_availability"));
        await shardRegistry.connect(proposer).registerModel(
          modelHash, "QmManifest", ethers.keccak256(ethers.toUtf8Bytes("root")),
          10, 4, ethers.parseUnits("140000000", 0), 0, 1, 1
        );

        // Register providers
        await staking.connect(provider1).stake(2, MIN_STORAGE_STAKE);
        await staking.connect(provider2).stake(2, MIN_STORAGE_STAKE);
        await shardRegistry.connect(provider1).registerProvider(ethers.parseUnits("100000000", 0));
        await shardRegistry.connect(provider2).registerProvider(ethers.parseUnits("100000000", 0));
      });

      it("Should report insufficient shards when less than k available", async function () {
        // Announce only 5 shards (need 10 for reconstruction)
        for (let i = 0; i < 5; i++) {
          await shardRegistry.connect(provider1).announceShard(
            modelHash, i, ethers.keccak256(ethers.toUtf8Bytes(`shard_${i}`)),
            ethers.parseUnits("10000000", 0), false
          );
        }

        const [availableShards, sufficient] = await shardRegistry.checkShardAvailability(modelHash);
        expect(availableShards).to.equal(5);
        expect(sufficient).to.be.false;
      });

      it("Should report sufficient shards when k or more available", async function () {
        // Announce exactly 10 data shards (k = 10)
        for (let i = 0; i < 10; i++) {
          await shardRegistry.connect(provider1).announceShard(
            modelHash, i, ethers.keccak256(ethers.toUtf8Bytes(`shard_${i}`)),
            ethers.parseUnits("10000000", 0), false
          );
        }

        const [availableShards, sufficient] = await shardRegistry.checkShardAvailability(modelHash);
        expect(availableShards).to.equal(10);
        expect(sufficient).to.be.true;
      });

      it("Should report sufficient with mix of data and parity shards", async function () {
        // Announce 8 data shards + 3 parity shards = 11 total (>= k=10)
        for (let i = 0; i < 8; i++) {
          await shardRegistry.connect(provider1).announceShard(
            modelHash, i, ethers.keccak256(ethers.toUtf8Bytes(`shard_${i}`)),
            ethers.parseUnits("10000000", 0), false
          );
        }
        for (let i = 10; i < 13; i++) {
          await shardRegistry.connect(provider2).announceShard(
            modelHash, i, ethers.keccak256(ethers.toUtf8Bytes(`shard_${i}`)),
            ethers.parseUnits("10000000", 0), true
          );
        }

        const [availableShards, sufficient] = await shardRegistry.checkShardAvailability(modelHash);
        expect(availableShards).to.equal(11);
        expect(sufficient).to.be.true;
      });
    });

    describe("Shard Removal", function () {
      let modelHash;

      beforeEach(async function () {
        modelHash = ethers.keccak256(ethers.toUtf8Bytes("model_removal"));
        await shardRegistry.connect(proposer).registerModel(
          modelHash, "QmManifest", ethers.keccak256(ethers.toUtf8Bytes("root")),
          10, 4, ethers.parseUnits("140000000", 0), 0, 1, 1
        );

        await staking.connect(provider1).stake(2, MIN_STORAGE_STAKE);
        await shardRegistry.connect(provider1).registerProvider(ethers.parseUnits("100000000", 0));
        await shardRegistry.connect(provider1).announceShard(
          modelHash, 0, ethers.keccak256(ethers.toUtf8Bytes("shard_0")),
          ethers.parseUnits("10000000", 0), false
        );
      });

      it("Should allow provider to remove shard", async function () {
        await expect(
          shardRegistry.connect(provider1).removeShard(modelHash, 0)
        ).to.emit(shardRegistry, "ShardRemoved").withArgs(modelHash, 0, provider1.address);

        // Verify provider list is empty
        const providers = await shardRegistry.getShardProviders(modelHash, 0);
        expect(providers.length).to.equal(0);

        // Verify storage released
        const [, used] = await shardRegistry.getProviderInfo(provider1.address);
        expect(used).to.equal(0);
      });

      it("Should reject removal by non-provider", async function () {
        await expect(
          shardRegistry.connect(provider2).removeShard(modelHash, 0)
        ).to.be.revertedWith("Not storing this shard");
      });

      it("Should handle removal when multiple providers store same shard", async function () {
        // Add second provider
        await staking.connect(provider2).stake(2, MIN_STORAGE_STAKE);
        await shardRegistry.connect(provider2).registerProvider(ethers.parseUnits("100000000", 0));
        await shardRegistry.connect(provider2).announceShard(
          modelHash, 0, ethers.keccak256(ethers.toUtf8Bytes("shard_0")),
          ethers.parseUnits("10000000", 0), false
        );

        // Provider1 removes
        await shardRegistry.connect(provider1).removeShard(modelHash, 0);

        // Provider2 should still be listed
        const providers = await shardRegistry.getShardProviders(modelHash, 0);
        expect(providers.length).to.equal(1);
        expect(providers[0]).to.equal(provider2.address);
      });
    });

    describe("Provider Reputation Updates", function () {
      let modelHash;

      beforeEach(async function () {
        modelHash = ethers.keccak256(ethers.toUtf8Bytes("model_reputation"));
        await shardRegistry.connect(proposer).registerModel(
          modelHash, "QmManifest", ethers.keccak256(ethers.toUtf8Bytes("root")),
          10, 4, ethers.parseUnits("140000000", 0), 0, 1, 1
        );

        await staking.connect(provider1).stake(2, MIN_STORAGE_STAKE);
        await shardRegistry.connect(provider1).registerProvider(ethers.parseUnits("100000000", 0));
      });

      it("Should increase reputation on successful verification", async function () {
        const [, , initialReputation] = await shardRegistry.getProviderInfo(provider1.address);
        expect(initialReputation).to.equal(500);

        await shardRegistry.connect(owner).reportVerificationSuccess(modelHash, 0, provider1.address);

        const [, , updatedReputation, successfulVerifications] = await shardRegistry.getProviderInfo(provider1.address);
        expect(updatedReputation).to.equal(510);
        expect(successfulVerifications).to.equal(1);
      });

      it("Should decrease reputation on failed verification", async function () {
        const [, , initialReputation] = await shardRegistry.getProviderInfo(provider1.address);
        expect(initialReputation).to.equal(500);

        await shardRegistry.connect(owner).reportVerificationFailure(modelHash, 0, provider1.address);

        const [, , updatedReputation, , failedVerifications] = await shardRegistry.getProviderInfo(provider1.address);
        expect(updatedReputation).to.equal(450);
        expect(failedVerifications).to.equal(1);
      });

      it("Should cap reputation at maximum 1000", async function () {
        // Report many successful verifications
        for (let i = 0; i < 60; i++) {
          await shardRegistry.connect(owner).reportVerificationSuccess(modelHash, 0, provider1.address);
        }

        const [, , reputation] = await shardRegistry.getProviderInfo(provider1.address);
        expect(reputation).to.equal(1000);
      });

      it("Should cap reputation at minimum 0", async function () {
        // Report many failed verifications
        for (let i = 0; i < 15; i++) {
          await shardRegistry.connect(owner).reportVerificationFailure(modelHash, 0, provider1.address);
        }

        const [, , reputation] = await shardRegistry.getProviderInfo(provider1.address);
        expect(reputation).to.equal(0);
      });
    });
  });
});

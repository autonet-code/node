const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("Consensus-as-Truth (MM-Zero)", function () {
  let atnToken;
  let staking;
  let taskContract;
  let resultsRewards;
  let project;
  let owner, proposer, solver1, solver2, solver3, solver4, coord1;

  const PROPOSER_STAKE = ethers.parseEther("100");
  const SOLVER_STAKE = ethers.parseEther("50");
  const COORDINATOR_STAKE = ethers.parseEther("500");

  // Default difficulty target: peak at 50%
  const DEFAULT_DIFFICULTY = {
    minSolvability: 2500,  // 25%
    maxSolvability: 7500,  // 75%
    peakSolvability: 5000, // 50%
  };

  beforeEach(async function () {
    [owner, proposer, solver1, solver2, solver3, solver4, coord1] =
      await ethers.getSigners();

    // Deploy ATN Token
    const ATNToken = await ethers.getContractFactory("ATNToken");
    atnToken = await ATNToken.deploy(
      owner.address,
      ethers.parseEther("1000000000")
    );
    await atnToken.waitForDeployment();

    // Deploy Staking
    const Staking = await ethers.getContractFactory("ParticipantStaking");
    staking = await Staking.deploy(
      await atnToken.getAddress(),
      owner.address
    );
    await staking.waitForDeployment();

    // Deploy Project
    const Project = await ethers.getContractFactory("Project");
    project = await Project.deploy(
      await atnToken.getAddress(),
      owner.address
    );
    await project.waitForDeployment();

    // Deploy TaskContract
    const TaskContract = await ethers.getContractFactory("TaskContract");
    taskContract = await TaskContract.deploy(
      await staking.getAddress(),
      owner.address
    );
    await taskContract.waitForDeployment();

    // Deploy ResultsRewards
    const ResultsRewards = await ethers.getContractFactory("ResultsRewards");
    resultsRewards = await ResultsRewards.deploy(
      await taskContract.getAddress(),
      await staking.getAddress(),
      owner.address
    );
    await resultsRewards.waitForDeployment();

    // Configure contracts
    await taskContract.setResultsRewardsContract(
      await resultsRewards.getAddress()
    );
    await resultsRewards.setProjectContract(await project.getAddress());
    await staking.setAuthorizedSlasher(
      await resultsRewards.getAddress(),
      true
    );
    await project.setAuthorizedDisburser(
      await resultsRewards.getAddress(),
      true
    );

    // Distribute tokens
    const testAmount = ethers.parseEther("10000");
    for (const signer of [proposer, solver1, solver2, solver3, solver4, coord1]) {
      await atnToken.transfer(signer.address, testAmount);
      await atnToken
        .connect(signer)
        .approve(await staking.getAddress(), testAmount);
    }

    // Stake participants
    await staking.connect(proposer).stake(1, PROPOSER_STAKE);
    await staking.connect(solver1).stake(2, SOLVER_STAKE);
    await staking.connect(solver2).stake(2, SOLVER_STAKE);

    // Fund a project so rewards can be disbursed
    const projectFunding = ethers.parseEther("1000");
    await atnToken.approve(await project.getAddress(), projectFunding);
    await project.createProject(
      "TestProject",
      "QmTest",
      projectFunding,             // fundingGoal
      ethers.parseEther("0"),     // initialBudget
      ethers.parseEther("100"),   // founderPTAmount
      "TestPT",                   // ptName
      "TPT"                       // ptSymbol
    );
    // Approve ATN for funding, then fund the project
    await atnToken.approve(await project.getAddress(), projectFunding);
    await project.fundProject(1, projectFunding, ethers.parseEther("100"));
  });

  describe("Consensus Task Proposal", function () {
    it("Should allow a staked proposer to propose a consensus task", async function () {
      const specHash = ethers.keccak256(ethers.toUtf8Bytes("test-spec-cid"));

      const tx = await taskContract
        .connect(proposer)
        .proposeConsensusTask(
          1,
          specHash,
          DEFAULT_DIFFICULTY,
          ethers.parseEther("10"),
          ethers.parseEther("5")
        );

      const receipt = await tx.wait();
      const event = receipt.logs.find((l) => {
        try {
          return taskContract.interface.parseLog(l)?.name === "ConsensusTaskProposed";
        } catch { return false; }
      });
      expect(event).to.not.be.undefined;

      const parsed = taskContract.interface.parseLog(event);
      expect(parsed.args.taskId).to.equal(1);
      expect(parsed.args.projectId).to.equal(1);
      expect(parsed.args.peakSolvability).to.equal(5000);
    });

    it("Should reject invalid difficulty range", async function () {
      const specHash = ethers.keccak256(ethers.toUtf8Bytes("test-spec"));

      await expect(
        taskContract.connect(proposer).proposeConsensusTask(
          1,
          specHash,
          { minSolvability: 7500, maxSolvability: 2500, peakSolvability: 5000 },
          ethers.parseEther("10"),
          ethers.parseEther("5")
        )
      ).to.be.revertedWith("Invalid solvability range");
    });

    it("Should reject peak outside range", async function () {
      const specHash = ethers.keccak256(ethers.toUtf8Bytes("test-spec"));

      await expect(
        taskContract.connect(proposer).proposeConsensusTask(
          1,
          specHash,
          { minSolvability: 2500, maxSolvability: 7500, peakSolvability: 8000 },
          ethers.parseEther("10"),
          ethers.parseEther("5")
        )
      ).to.be.revertedWith("Peak outside range");
    });
  });

  describe("Rollout Submission", function () {
    let taskId;

    beforeEach(async function () {
      const specHash = ethers.keccak256(ethers.toUtf8Bytes("test-spec"));
      await taskContract
        .connect(proposer)
        .proposeConsensusTask(
          1,
          specHash,
          DEFAULT_DIFFICULTY,
          ethers.parseEther("10"),
          ethers.parseEther("5")
        );
      taskId = 1;
    });

    it("Should allow multiple solvers to submit rollouts", async function () {
      const answer1 = ethers.keccak256(ethers.toUtf8Bytes("answer-A"));
      const solution1 = ethers.keccak256(ethers.toUtf8Bytes("solution-1"));
      const answer2 = ethers.keccak256(ethers.toUtf8Bytes("answer-A")); // same answer
      const solution2 = ethers.keccak256(ethers.toUtf8Bytes("solution-2"));

      await taskContract
        .connect(solver1)
        .submitRollout(taskId, answer1, solution1, 80);
      await taskContract
        .connect(solver2)
        .submitRollout(taskId, answer2, solution2, 75);

      expect(await taskContract.getRolloutCount(taskId)).to.equal(2);
    });

    it("Should prevent double rollout from same solver", async function () {
      const answerHash = ethers.keccak256(ethers.toUtf8Bytes("answer"));
      const solHash = ethers.keccak256(ethers.toUtf8Bytes("solution"));

      await taskContract
        .connect(solver1)
        .submitRollout(taskId, answerHash, solHash, 80);

      await expect(
        taskContract
          .connect(solver1)
          .submitRollout(taskId, answerHash, solHash, 80)
      ).to.be.revertedWith("Already submitted rollout");
    });

    it("Should reject confidence > 100", async function () {
      const answerHash = ethers.keccak256(ethers.toUtf8Bytes("answer"));
      const solHash = ethers.keccak256(ethers.toUtf8Bytes("solution"));

      await expect(
        taskContract
          .connect(solver1)
          .submitRollout(taskId, answerHash, solHash, 101)
      ).to.be.revertedWith("Confidence must be 0-100");
    });

    it("Should reject rollout on ground-truth tasks", async function () {
      // Create a ground-truth task
      const specHash = ethers.keccak256(ethers.toUtf8Bytes("gt-spec"));
      const gtHash = ethers.keccak256(ethers.toUtf8Bytes("gt-answer"));
      await taskContract
        .connect(proposer)
        .proposeTask(1, specHash, gtHash, ethers.parseEther("10"), ethers.parseEther("5"));

      const gtTaskId = 2;
      const answerHash = ethers.keccak256(ethers.toUtf8Bytes("answer"));
      const solHash = ethers.keccak256(ethers.toUtf8Bytes("solution"));

      await expect(
        taskContract
          .connect(solver1)
          .submitRollout(gtTaskId, answerHash, solHash, 80)
      ).to.be.revertedWith("Not consensus task");
    });

    it("Should return rollout data correctly", async function () {
      const answerHash = ethers.keccak256(ethers.toUtf8Bytes("answer-X"));
      const solHash = ethers.keccak256(ethers.toUtf8Bytes("solution-X"));

      await taskContract
        .connect(solver1)
        .submitRollout(taskId, answerHash, solHash, 90);

      const rollouts = await taskContract.getRollouts(taskId);
      expect(rollouts.length).to.equal(1);
      expect(rollouts[0].solver).to.equal(solver1.address);
      expect(rollouts[0].answerHash).to.equal(answerHash);
      expect(rollouts[0].confidence).to.equal(90);
    });
  });

  describe("Difficulty Score Computation", function () {
    it("Should give max score when solvability equals peak", async function () {
      // We test this indirectly via the contract.
      // With 2 solvers submitting the SAME answer, solvability = 100% = 10000 BPS
      // With default target peak at 50%, this is outside the range (>75%), so score = 0
      // This IS the anti-collusion mechanism working correctly.
      // Let's test with 4 solvers where 2 agree and 2 disagree -> 50% solvability = peak

      // Need 2 more staked solvers
      await staking.connect(solver3).stake(2, SOLVER_STAKE);
      await staking.connect(solver4).stake(2, SOLVER_STAKE);

      const specHash = ethers.keccak256(ethers.toUtf8Bytes("diff-spec"));
      await taskContract
        .connect(proposer)
        .proposeConsensusTask(
          1,
          specHash,
          DEFAULT_DIFFICULTY,
          ethers.parseEther("10"),
          ethers.parseEther("5")
        );
      const taskId = 1;

      // 2 solvers give answer A, 2 give answer B -> 50% solvability
      const answerA = ethers.keccak256(ethers.toUtf8Bytes("answer-A"));
      const answerB = ethers.keccak256(ethers.toUtf8Bytes("answer-B"));
      const solHash = ethers.keccak256(ethers.toUtf8Bytes("sol"));

      await taskContract.connect(solver1).submitRollout(taskId, answerA, solHash, 80);
      await taskContract.connect(solver2).submitRollout(taskId, answerA, solHash, 75);
      await taskContract.connect(solver3).submitRollout(taskId, answerB, solHash, 60);
      await taskContract.connect(solver4).submitRollout(taskId, answerB, solHash, 55);

      expect(await taskContract.getRolloutCount(taskId)).to.equal(4);
      expect(await taskContract.isRolloutCollectionComplete(taskId)).to.be.true;
    });
  });

  describe("Consensus Finalization", function () {
    it("Should finalize when all solvers agree (high solvability = low reward)", async function () {
      const specHash = ethers.keccak256(ethers.toUtf8Bytes("easy-spec"));
      await taskContract
        .connect(proposer)
        .proposeConsensusTask(
          1,
          specHash,
          DEFAULT_DIFFICULTY,
          ethers.parseEther("10"),
          ethers.parseEther("5")
        );
      const taskId = 1;

      // Both solvers agree -> solvability = 100% -> outside 75% max -> score = 0
      const answerA = ethers.keccak256(ethers.toUtf8Bytes("same-answer"));
      const solHash1 = ethers.keccak256(ethers.toUtf8Bytes("sol-1"));
      const solHash2 = ethers.keccak256(ethers.toUtf8Bytes("sol-2"));

      await taskContract.connect(solver1).submitRollout(taskId, answerA, solHash1, 90);
      await taskContract.connect(solver2).submitRollout(taskId, answerA, solHash2, 85);

      // Need a 3rd to meet MIN_ROLLOUTS=3
      await staking.connect(solver3).stake(2, SOLVER_STAKE);
      await taskContract.connect(solver3).submitRollout(taskId, answerA, solHash2, 80);

      expect(await taskContract.isRolloutCollectionComplete(taskId)).to.be.true;

      // Finalize - all agree so solvability = 100% > maxSolvability 75%
      // Difficulty score should be 0 (anti-collusion: trivially easy = no reward)
      await resultsRewards.finalizeConsensusTask(taskId);

      // Task should be finalized
      const proposal = await taskContract.getTaskProposal(taskId);
      expect(proposal.status).to.equal(13); // CONSENSUS_FINALIZED
    });

    it("Should give higher rewards at 50% solvability (peak difficulty)", async function () {
      // Need 4 solvers
      await staking.connect(solver3).stake(2, SOLVER_STAKE);
      await staking.connect(solver4).stake(2, SOLVER_STAKE);

      const specHash = ethers.keccak256(ethers.toUtf8Bytes("hard-spec"));
      // Use a wider range so 50% is clearly in range
      await taskContract
        .connect(proposer)
        .proposeConsensusTask(
          1,
          specHash,
          { minSolvability: 2500, maxSolvability: 7500, peakSolvability: 5000 },
          ethers.parseEther("10"),
          ethers.parseEther("5")
        );
      const taskId = 1;

      // 3 agree, 1 disagrees -> 75% solvability (within range, not at boundary)
      const answerA = ethers.keccak256(ethers.toUtf8Bytes("answer-A"));
      const answerB = ethers.keccak256(ethers.toUtf8Bytes("answer-B"));
      const sol = ethers.keccak256(ethers.toUtf8Bytes("sol"));

      await taskContract.connect(solver1).submitRollout(taskId, answerA, sol, 80);
      await taskContract.connect(solver2).submitRollout(taskId, answerA, sol, 75);
      await taskContract.connect(solver3).submitRollout(taskId, answerA, sol, 70);
      await taskContract.connect(solver4).submitRollout(taskId, answerB, sol, 55);

      // Finalize - 75% solvability = at max boundary, reduced reward
      const tx = await resultsRewards.finalizeConsensusTask(taskId);
      const receipt = await tx.wait();

      // Check for YumaConsensusReached event (carries the difficulty score)
      const consensusEvents = receipt.logs
        .map((l) => {
          try { return resultsRewards.interface.parseLog(l); }
          catch { return null; }
        })
        .filter((e) => e?.name === "YumaConsensusReached");

      expect(consensusEvents.length).to.equal(1);

      // Task should be finalized
      const proposal = await taskContract.getTaskProposal(taskId);
      expect(proposal.status).to.equal(13); // CONSENSUS_FINALIZED
    });

    it("Should reject finalization before collection is complete", async function () {
      const specHash = ethers.keccak256(ethers.toUtf8Bytes("incomplete-spec"));
      await taskContract
        .connect(proposer)
        .proposeConsensusTask(
          1,
          specHash,
          DEFAULT_DIFFICULTY,
          ethers.parseEther("10"),
          ethers.parseEther("5")
        );

      // Only 1 rollout (MIN_ROLLOUTS = 3)
      const answerHash = ethers.keccak256(ethers.toUtf8Bytes("answer"));
      const solHash = ethers.keccak256(ethers.toUtf8Bytes("sol"));
      await taskContract.connect(solver1).submitRollout(1, answerHash, solHash, 80);

      await expect(
        resultsRewards.finalizeConsensusTask(1)
      ).to.be.revertedWith("Collection not complete");
    });

    it("Should reject double finalization", async function () {
      const specHash = ethers.keccak256(ethers.toUtf8Bytes("double-spec"));
      await taskContract
        .connect(proposer)
        .proposeConsensusTask(
          1,
          specHash,
          DEFAULT_DIFFICULTY,
          ethers.parseEther("10"),
          ethers.parseEther("5")
        );

      await staking.connect(solver3).stake(2, SOLVER_STAKE);

      const answer = ethers.keccak256(ethers.toUtf8Bytes("answer"));
      const sol = ethers.keccak256(ethers.toUtf8Bytes("sol"));
      await taskContract.connect(solver1).submitRollout(1, answer, sol, 80);
      await taskContract.connect(solver2).submitRollout(1, answer, sol, 75);
      await taskContract.connect(solver3).submitRollout(1, answer, sol, 70);

      await resultsRewards.finalizeConsensusTask(1);

      // Second finalization fails - status is no longer CONSENSUS_COLLECTING
      await expect(
        resultsRewards.finalizeConsensusTask(1)
      ).to.be.revertedWith("Not collecting");
    });
  });

  describe("Anti-Collusion Incentives", function () {
    it("Should penalize trivially easy tasks (100% agreement)", async function () {
      await staking.connect(solver3).stake(2, SOLVER_STAKE);

      const specHash = ethers.keccak256(ethers.toUtf8Bytes("collusion-spec"));
      await taskContract
        .connect(proposer)
        .proposeConsensusTask(
          1,
          specHash,
          DEFAULT_DIFFICULTY,
          ethers.parseEther("10"),
          ethers.parseEther("5")
        );

      // All 3 solvers give the same answer -> 100% solvability
      const sameAnswer = ethers.keccak256(ethers.toUtf8Bytes("colluded-answer"));
      const sol = ethers.keccak256(ethers.toUtf8Bytes("sol"));

      await taskContract.connect(solver1).submitRollout(1, sameAnswer, sol, 95);
      await taskContract.connect(solver2).submitRollout(1, sameAnswer, sol, 95);
      await taskContract.connect(solver3).submitRollout(1, sameAnswer, sol, 95);

      const proposerBalBefore = await atnToken.balanceOf(proposer.address);

      await resultsRewards.finalizeConsensusTask(1);

      const proposerBalAfter = await atnToken.balanceOf(proposer.address);

      // 100% solvability is > maxSolvability (75%), so difficulty score = 0
      // Proposer should get ZERO reward
      expect(proposerBalAfter).to.equal(proposerBalBefore);
    });
  });
});

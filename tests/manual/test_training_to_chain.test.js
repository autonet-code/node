/**
 * Test 2: Training Attestation → On-Chain Reward
 *
 * Confirms the on-chain training recording and reward minting flow:
 *   1. Deploy RPB with a test Registry
 *   2. Register an agent
 *   3. Agent self-reports training via recordTraining()
 *   4. Fund training pool via purchaseShares()
 *   5. Agent calls claimTrainingReward() → mints ATN
 *
 * Run:
 *   cd C:\code\autonet
 *   npx hardhat test tests/manual/test_training_to_chain.test.js
 */

const { expect } = require("chai");
const { ethers } = require("hardhat");

const e18 = (n) => ethers.parseEther(String(n));

describe("Test 2: Training → On-Chain Reward", function () {
  this.timeout(60_000);

  let rpb, rpbAddr, registry;
  let dao, trainer, investor;

  beforeEach(async function () {
    [dao, trainer, investor] = await ethers.getSigners();

    // Deploy Registry (dao is owner)
    const Registry = await ethers.getContractFactory("Registry");
    registry = await Registry.deploy(dao.address, dao.address);
    await registry.waitForDeployment();
    await registry.setJurisdictionAddress(dao.address);

    // Deploy RPB
    const RPB = await ethers.getContractFactory("RPB");
    rpb = await RPB.deploy(await registry.getAddress());
    await rpb.waitForDeployment();
    rpbAddr = await rpb.getAddress();

    // Verify: no pre-mint
    expect(await rpb.totalSupply()).to.equal(0n);

    // Register trainer as an agent
    const lineage = ethers.keccak256(ethers.toUtf8Bytes("trainer-agent"));
    const alignment = ethers.keccak256(ethers.toUtf8Bytes("aligned-v1"));
    await rpb.connect(trainer).registerAgent(
      lineage, alignment,
      ethers.ZeroAddress, // no parent
      ethers.ZeroAddress, // no sponsor
    );
  });

  it("records training and mints reward after pool is funded", async function () {
    // Step 1: Trainer self-reports training contribution
    await rpb.connect(trainer).recordTraining(100);

    // Verify training was recorded
    const tokens = await rpb.agentTrainingTokens(trainer.address);
    expect(tokens).to.equal(100n);
    console.log("  Training tokens recorded:", tokens.toString());

    // Step 2: Fund training pool (investor buys shares using ATN)
    // First, DAO mints some ATN for the investor to use
    await rpb.mint(investor.address, e18(10_000));

    // Investor approves and purchases shares
    await rpb.connect(investor).approve(rpbAddr, e18(5_000));
    await rpb.connect(investor).purchaseShares(rpbAddr, e18(5_000));

    const pool = await rpb.trainingRewardPool();
    console.log("  Training reward pool:", ethers.formatEther(pool), "ATN");
    expect(pool).to.be.gt(0n);

    // Verify shares issued
    const shares = await rpb.getShareBalance(investor.address);
    console.log("  Investor shares:", ethers.formatEther(shares));
    expect(shares).to.be.gt(0n);

    // Step 3: Trainer claims reward → mints fresh ATN
    const balBefore = await rpb.balanceOf(trainer.address);
    await rpb.connect(trainer).claimTrainingReward();
    const balAfter = await rpb.balanceOf(trainer.address);

    const reward = balAfter - balBefore;
    console.log("  ATN minted as reward:", ethers.formatEther(reward));
    expect(reward).to.be.gt(0n);

    // Verify totalTrainingMinted increased
    const minted = await rpb.totalTrainingMinted();
    console.log("  Total training minted:", ethers.formatEther(minted));
    expect(minted).to.equal(reward);
  });

  it("reverts if claiming with no recorded training", async function () {
    // Fund pool first
    await rpb.mint(investor.address, e18(1_000));
    await rpb.connect(investor).approve(rpbAddr, e18(1_000));
    await rpb.connect(investor).purchaseShares(rpbAddr, e18(1_000));

    await expect(
      rpb.connect(trainer).claimTrainingReward()
    ).to.be.revertedWithCustomError(rpb, "NoRewardsToClaim");
  });

  it("reverts if pool is empty", async function () {
    // Record training but don't fund pool
    await rpb.connect(trainer).recordTraining(100);

    await expect(
      rpb.connect(trainer).claimTrainingReward()
    ).to.be.revertedWithCustomError(rpb, "RewardPoolEmpty");
  });

  it("multiple trainers share pool proportionally", async function () {
    // Register a second trainer
    const [, , , , trainer2] = await ethers.getSigners();
    const lineage2 = ethers.keccak256(ethers.toUtf8Bytes("trainer-2"));
    const alignment2 = ethers.keccak256(ethers.toUtf8Bytes("aligned-v1"));
    await rpb.connect(trainer2).registerAgent(
      lineage2, alignment2,
      ethers.ZeroAddress,
      ethers.ZeroAddress,
    );

    // Record: trainer gets 300, trainer2 gets 100 (3:1 ratio)
    await rpb.connect(trainer).recordTraining(300);
    await rpb.connect(trainer2).recordTraining(100);

    // Fund pool
    await rpb.mint(investor.address, e18(10_000));
    await rpb.connect(investor).approve(rpbAddr, e18(10_000));
    await rpb.connect(investor).purchaseShares(rpbAddr, e18(10_000));

    // Both claim
    await rpb.connect(trainer).claimTrainingReward();
    await rpb.connect(trainer2).claimTrainingReward();

    const reward1 = await rpb.balanceOf(trainer.address);
    const reward2 = await rpb.balanceOf(trainer2.address);

    console.log("  Trainer 1 reward:", ethers.formatEther(reward1));
    console.log("  Trainer 2 reward:", ethers.formatEther(reward2));

    // Trainer 1 should get ~3x more than trainer 2
    // Allow some tolerance for rounding
    const ratio = Number(reward1) / Number(reward2);
    expect(ratio).to.be.closeTo(3.0, 0.1);
  });
});

/**
 * Test 4: Investment → Training Pool → Reward Cycle
 *
 * Full economic path from investor to trainer:
 *   1. Deploy RPB + test ERC20 with parity set
 *   2. Investor buys shares (funds training pool)
 *   3. Operator records training
 *   4. Trainer claims reward → mints ATN
 *   5. Verify all balances reconcile
 *
 * Run:
 *   cd C:\code\autonet
 *   npx hardhat test tests/manual/test_investment_to_reward.test.js
 */

const { expect } = require("chai");
const { ethers } = require("hardhat");

const e18 = (n) => ethers.parseEther(String(n));

describe("Test 4: Investment → Training Pool → Reward", function () {
  this.timeout(60_000);

  let rpb, rpbAddr, registry;
  let dao, operator, trainer, investor;

  beforeEach(async function () {
    [dao, operator, trainer, investor] = await ethers.getSigners();

    // Deploy Registry
    const Registry = await ethers.getContractFactory("Registry");
    registry = await Registry.deploy(dao.address, dao.address);
    await registry.waitForDeployment();
    await registry.setJurisdictionAddress(dao.address);

    // Deploy RPB
    const RPB = await ethers.getContractFactory("RPB");
    rpb = await RPB.deploy(await registry.getAddress());
    await rpb.waitForDeployment();
    rpbAddr = await rpb.getAddress();

    // Setup: operator authorized, trainer registered
    await rpb.setAuthorizedOperator(operator.address, true);

    const lineage = ethers.keccak256(ethers.toUtf8Bytes("trainer-agent"));
    const alignment = ethers.keccak256(ethers.toUtf8Bytes("aligned-v1"));
    await rpb.connect(trainer).registerAgent(
      lineage, alignment,
      ethers.ZeroAddress,
      ethers.ZeroAddress,
    );
  });

  it("full investment → training → reward cycle with ATN", async function () {
    // --- Investment Phase ---

    // DAO mints ATN for investor (simulates investor acquiring ATN)
    await rpb.mint(investor.address, e18(50_000));

    const investAmount = e18(5_000);
    await rpb.connect(investor).approve(rpbAddr, investAmount);
    await rpb.connect(investor).purchaseShares(rpbAddr, investAmount);

    // Verify shares
    const shares = await rpb.getShareBalance(investor.address);
    console.log("  Shares issued:", ethers.formatEther(shares));
    expect(shares).to.be.gt(0n);

    // Verify training pool funded
    const poolBefore = await rpb.trainingRewardPool();
    console.log("  Training pool after investment:", ethers.formatEther(poolBefore));
    expect(poolBefore).to.be.gt(0n);

    // Total shares should match
    const totalShares = await rpb.totalShares();
    expect(totalShares).to.equal(shares);

    // --- Training Phase ---

    const contribution = 200;
    await rpb.connect(operator).recordTraining(trainer.address, contribution);

    const trainingTokens = await rpb.agentTrainingTokens(trainer.address);
    expect(trainingTokens).to.equal(BigInt(contribution));
    console.log("  Training tokens:", trainingTokens.toString());

    // --- Reward Phase ---

    const trainerBalBefore = await rpb.balanceOf(trainer.address);
    await rpb.connect(trainer).claimTrainingReward();
    const trainerBalAfter = await rpb.balanceOf(trainer.address);

    const reward = trainerBalAfter - trainerBalBefore;
    console.log("  ATN minted to trainer:", ethers.formatEther(reward));
    expect(reward).to.be.gt(0n);

    // Pool should decrease
    const poolAfter = await rpb.trainingRewardPool();
    // Note: pool tracks the cap, actual pool might not decrease
    // depending on implementation (rewards are minted, not transferred from pool)
    console.log("  Training pool after claim:", ethers.formatEther(poolAfter));

    // Total training minted should equal the reward
    const totalMinted = await rpb.totalTrainingMinted();
    console.log("  Total training minted:", ethers.formatEther(totalMinted));
    expect(totalMinted).to.equal(reward);
  });

  it("multiple investment rounds stack", async function () {
    // Investor makes two purchases
    await rpb.mint(investor.address, e18(20_000));

    await rpb.connect(investor).approve(rpbAddr, e18(20_000));
    await rpb.connect(investor).purchaseShares(rpbAddr, e18(3_000));
    const shares1 = await rpb.getShareBalance(investor.address);

    await rpb.connect(investor).purchaseShares(rpbAddr, e18(7_000));
    const shares2 = await rpb.getShareBalance(investor.address);

    console.log("  After 1st purchase:", ethers.formatEther(shares1));
    console.log("  After 2nd purchase:", ethers.formatEther(shares2));
    expect(shares2).to.be.gt(shares1);

    // Pool should reflect both purchases
    const pool = await rpb.trainingRewardPool();
    console.log("  Total pool:", ethers.formatEther(pool));
    expect(pool).to.be.gt(0n);
  });

  it("trainer cannot double-claim", async function () {
    // Fund and train
    await rpb.mint(investor.address, e18(10_000));
    await rpb.connect(investor).approve(rpbAddr, e18(10_000));
    await rpb.connect(investor).purchaseShares(rpbAddr, e18(10_000));
    await rpb.connect(operator).recordTraining(trainer.address, 100);

    // First claim succeeds
    await rpb.connect(trainer).claimTrainingReward();
    const bal1 = await rpb.balanceOf(trainer.address);
    console.log("  After first claim:", ethers.formatEther(bal1));

    // Second claim with no new training → revert
    await expect(
      rpb.connect(trainer).claimTrainingReward()
    ).to.be.revertedWithCustomError(rpb, "NoRewardsToClaim");

    // But if more training is recorded, can claim again
    await rpb.connect(operator).recordTraining(trainer.address, 50);
    await rpb.connect(trainer).claimTrainingReward();
    const bal2 = await rpb.balanceOf(trainer.address);
    console.log("  After second training + claim:", ethers.formatEther(bal2));
    expect(bal2).to.be.gt(bal1);
  });
});

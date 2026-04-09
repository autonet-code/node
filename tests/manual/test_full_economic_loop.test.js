/**
 * Test 7: Full Economic Loop
 *
 * The complete cycle with balance reconciliation:
 *   1. Investor purchases shares → funds training pool
 *   2. Trainer trains, operator records on-chain
 *   3. Trainer claims reward → ATN minted
 *   4. Requester pays for inference (using ATN)
 *   5. Revenue splits: provider, treasury, dividend pool
 *   6. Shareholder claims dividends
 *   7. ALL BALANCES RECONCILE
 *
 * Run:
 *   cd C:\code\autonet
 *   npx hardhat test tests/manual/test_full_economic_loop.test.js
 */

const { expect } = require("chai");
const { ethers } = require("hardhat");

const e18 = (n) => ethers.parseEther(String(n));

describe("Test 7: Full Economic Loop", function () {
  this.timeout(120_000);

  let rpb, rpbAddr, registry;
  let dao, operator, trainer, requester, provider, investor;

  beforeEach(async function () {
    [dao, operator, trainer, requester, provider, investor] =
      await ethers.getSigners();

    // Deploy infrastructure
    const Registry = await ethers.getContractFactory("Registry");
    registry = await Registry.deploy(dao.address, dao.address);
    await registry.waitForDeployment();
    await registry.setJurisdictionAddress(dao.address);

    const RPB = await ethers.getContractFactory("RPB");
    rpb = await RPB.deploy(await registry.getAddress());
    await rpb.waitForDeployment();
    rpbAddr = await rpb.getAddress();

    // Authorize operator
    await rpb.setAuthorizedOperator(operator.address, true);

    // Register all agents
    for (const [signer, name] of [
      [trainer, "trainer"],
      [requester, "requester"],
      [provider, "provider"],
    ]) {
      const lineage = ethers.keccak256(ethers.toUtf8Bytes(`${name}-agent`));
      const alignment = ethers.keccak256(ethers.toUtf8Bytes("aligned-v1"));
      await rpb.connect(signer).registerAgent(
        lineage, alignment,
        ethers.ZeroAddress,
        ethers.ZeroAddress,
      );
    }
  });

  it("invest → train → mint → infer → earn → dividend — all balances reconcile", async function () {
    console.log("\n  === FULL ECONOMIC LOOP ===\n");

    // ─── Phase 1: Investment ───────────────────────────────────────────

    // DAO mints initial ATN for the investor (simulates acquisition)
    const investorSeed = e18(100_000);
    await rpb.mint(investor.address, investorSeed);

    const investAmount = e18(50_000);
    await rpb.connect(investor).approve(rpbAddr, investAmount);
    await rpb.connect(investor).purchaseShares(rpbAddr, investAmount);

    const shares = await rpb.getShareBalance(investor.address);
    const pool = await rpb.trainingRewardPool();
    console.log("  [Investment] Shares:", ethers.formatEther(shares));
    console.log("  [Investment] Training pool:", ethers.formatEther(pool));
    expect(shares).to.be.gt(0n);
    expect(pool).to.be.gt(0n);

    // ─── Phase 2: Training ──────────────────────────────────────────

    // Operator records 500 units of training for trainer
    await rpb.connect(operator).recordTraining(trainer.address, 500);

    const trainingTokens = await rpb.agentTrainingTokens(trainer.address);
    console.log("  [Training] Tokens recorded:", trainingTokens.toString());
    expect(trainingTokens).to.equal(500n);

    // ─── Phase 3: Claim Training Reward (mint ATN) ──────────────────

    await rpb.connect(trainer).claimTrainingReward();
    const trainerBalance = await rpb.balanceOf(trainer.address);
    const totalMinted = await rpb.totalTrainingMinted();

    console.log("  [Reward] ATN minted to trainer:", ethers.formatEther(trainerBalance));
    console.log("  [Reward] Total training minted:", ethers.formatEther(totalMinted));
    expect(trainerBalance).to.be.gt(0n);
    expect(totalMinted).to.equal(trainerBalance);

    // ─── Phase 4: Inference ─────────────────────────────────────────

    // Give requester some ATN to pay for inference
    await rpb.mint(requester.address, e18(50_000));

    const infCost = e18(10_000);
    await rpb.connect(requester).approve(rpbAddr, infCost);

    const providerBefore = await rpb.balanceOf(provider.address);
    const daoBefore = await rpb.balanceOf(dao.address);
    const requesterBefore = await rpb.balanceOf(requester.address);

    await rpb.connect(operator).recordInference(
      requester.address,
      provider.address,
      100,       // units
      rpbAddr,   // pay in ATN
      infCost,
    );

    const providerGain = (await rpb.balanceOf(provider.address)) - providerBefore;
    const daoGain = (await rpb.balanceOf(dao.address)) - daoBefore;
    const requesterSpent = requesterBefore - (await rpb.balanceOf(requester.address));
    const dividendPool = await rpb.getDividendPoolBalance(rpbAddr);

    console.log("  [Inference] Requester spent:", ethers.formatEther(requesterSpent));
    console.log("  [Inference] Provider earned:", ethers.formatEther(providerGain), "(60%)");
    console.log("  [Inference] DAO treasury:", ethers.formatEther(daoGain), "(15%)");
    console.log("  [Inference] Dividend pool:", ethers.formatEther(dividendPool), "(25%)");

    // Verify split
    expect(providerGain).to.equal(infCost * 6000n / 10000n);
    expect(daoGain).to.equal(infCost * 1500n / 10000n);
    expect(dividendPool).to.equal(infCost * 2500n / 10000n);

    // Revenue conservation: all pieces sum to the total cost
    expect(providerGain + daoGain + dividendPool).to.equal(infCost);
    console.log("  [Inference] Revenue conservation: PASS ✓");

    // ─── Phase 5: Dividends ─────────────────────────────────────────

    const investorBefore = await rpb.balanceOf(investor.address);
    await rpb.connect(investor).claimDividends(rpbAddr);
    const investorAfter = await rpb.balanceOf(investor.address);
    const dividendReceived = investorAfter - investorBefore;

    console.log("  [Dividend] Received:", ethers.formatEther(dividendReceived));
    expect(dividendReceived).to.be.gt(0n);

    // ─── Phase 6: Balance Reconciliation ────────────────────────────

    console.log("\n  === BALANCE RECONCILIATION ===\n");

    // Total supply = initial mints + training rewards
    const totalSupply = await rpb.totalSupply();
    // We minted: investorSeed + requester seed + training rewards
    const expectedMints = investorSeed + e18(50_000) + totalMinted;
    console.log("  Total supply:", ethers.formatEther(totalSupply));
    console.log("  Expected mints:", ethers.formatEther(expectedMints));
    expect(totalSupply).to.equal(expectedMints);

    // Inference was a transfer (not mint), so:
    // providerGain + daoGain + dividendPool = requesterSpent = infCost
    console.log("  Inference revenue conservation:", ethers.formatEther(providerGain + daoGain + dividendPool));
    expect(providerGain + daoGain + dividendPool).to.equal(infCost);

    // Training minted = trainer balance (trainer never spent)
    expect(await rpb.balanceOf(trainer.address)).to.equal(totalMinted);

    console.log("\n  === ALL BALANCES RECONCILED ✓ ===\n");
  });
});

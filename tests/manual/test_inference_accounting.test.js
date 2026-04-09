/**
 * Test 6: Inference Accounting On-Chain
 *
 * Confirms inference payment pull + revenue split + dividend claim:
 *   1. Deploy RPB, register agents
 *   2. Requester approves RPB to spend tokens
 *   3. Requester calls recordInference() → payment pulled
 *   4. Revenue splits: 60% provider, 15% treasury, 25% dividend pool
 *   5. Shareholder claims dividends
 *
 * Run:
 *   cd C:\code\autonet
 *   npx hardhat test tests/manual/test_inference_accounting.test.js
 */

const { expect } = require("chai");
const { ethers } = require("hardhat");

const e18 = (n) => ethers.parseEther(String(n));

describe("Test 6: Inference Accounting", function () {
  this.timeout(60_000);

  let rpb, rpbAddr, registry;
  let dao, provider, requester, investor;

  beforeEach(async function () {
    [dao, provider, requester, investor] = await ethers.getSigners();

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

    // Register agents
    for (const [signer, name] of [[provider, "provider"], [requester, "requester"]]) {
      const lineage = ethers.keccak256(ethers.toUtf8Bytes(`${name}-agent`));
      const alignment = ethers.keccak256(ethers.toUtf8Bytes("aligned-v1"));
      await rpb.connect(signer).registerAgent(
        lineage, alignment,
        ethers.ZeroAddress,
        ethers.ZeroAddress,
      );
    }

    // Mint ATN and distribute
    await rpb.mint(dao.address, e18(1_000_000));
    await rpb.transfer(requester.address, e18(100_000));
    await rpb.transfer(investor.address, e18(100_000));
  });

  it("inference payment is pulled and split correctly", async function () {
    const cost = e18(1_000);

    // Requester approves RPB to pull payment
    await rpb.connect(requester).approve(rpbAddr, cost);

    // Snapshot balances
    const providerBefore = await rpb.balanceOf(provider.address);
    const daoBefore = await rpb.balanceOf(dao.address);
    const requesterBefore = await rpb.balanceOf(requester.address);

    // Requester records inference (payment pulled from requester)
    await rpb.connect(requester).recordInference(
      requester.address,
      provider.address,
      10,       // units
      rpbAddr,  // payment in ATN
      cost,
    );

    // Check balances after
    const providerAfter = await rpb.balanceOf(provider.address);
    const daoAfter = await rpb.balanceOf(dao.address);
    const requesterAfter = await rpb.balanceOf(requester.address);

    const providerGain = providerAfter - providerBefore;
    const daoGain = daoAfter - daoBefore;
    const requesterSpent = requesterBefore - requesterAfter;

    console.log("  Requester spent:", ethers.formatEther(requesterSpent), "ATN");
    console.log("  Provider got:", ethers.formatEther(providerGain), "(60%)");
    console.log("  DAO got:", ethers.formatEther(daoGain), "(15%)");

    // Verify exact splits
    expect(requesterSpent).to.equal(cost);
    expect(providerGain).to.equal(cost * 6000n / 10000n);  // 60%
    expect(daoGain).to.equal(cost * 1500n / 10000n);        // 15%

    // Remaining 25% should be in dividend pool
    const dividendPool = await rpb.getDividendPoolBalance(rpbAddr);
    const expectedDividend = cost * 2500n / 10000n;          // 25%
    console.log("  Dividend pool:", ethers.formatEther(dividendPool), "(25%)");
    expect(dividendPool).to.equal(expectedDividend);

    // Verify total inference tracked
    const totalUnits = await rpb.totalInferenceUnits();
    expect(totalUnits).to.equal(10n);
  });

  it("shareholder can claim dividends from inference revenue", async function () {
    // Investor buys shares first
    await rpb.connect(investor).approve(rpbAddr, e18(10_000));
    await rpb.connect(investor).purchaseShares(rpbAddr, e18(10_000));

    const shares = await rpb.getShareBalance(investor.address);
    console.log("  Investor shares:", ethers.formatEther(shares));

    // Generate inference revenue
    const cost = e18(10_000);
    await rpb.connect(requester).approve(rpbAddr, cost);
    await rpb.connect(requester).recordInference(
      requester.address,
      provider.address,
      100,
      rpbAddr,
      cost,
    );

    const dividendPool = await rpb.getDividendPoolBalance(rpbAddr);
    console.log("  Dividend pool after inference:", ethers.formatEther(dividendPool));

    // Investor claims dividends
    const investorBefore = await rpb.balanceOf(investor.address);
    await rpb.connect(investor).claimDividends(rpbAddr);
    const investorAfter = await rpb.balanceOf(investor.address);

    const dividendReceived = investorAfter - investorBefore;
    console.log("  Dividend received:", ethers.formatEther(dividendReceived));
    expect(dividendReceived).to.be.gt(0n);
  });

  it("ATN payment uses internal _transfer (no approval needed)", async function () {
    // When paying in ATN (token == RPB address), the contract uses
    // internal _transfer() which doesn't check allowance.
    // So no approval is needed for ATN payments.
    const cost = e18(500);
    const requesterBefore = await rpb.balanceOf(requester.address);

    // No explicit approval — should still work because token == address(this)
    await rpb.connect(requester).recordInference(
      requester.address,
      provider.address,
      5,
      rpbAddr,
      cost,
    );

    const requesterAfter = await rpb.balanceOf(requester.address);
    expect(requesterBefore - requesterAfter).to.equal(cost);
    console.log("  ATN pulled without approval (internal _transfer):", ethers.formatEther(cost));
  });

  it("tracks inference metrics per provider", async function () {
    const cost = e18(500);
    await rpb.connect(requester).approve(rpbAddr, cost);
    await rpb.connect(requester).recordInference(
      requester.address,
      provider.address,
      25,
      rpbAddr,
      cost,
    );

    // Check agent record
    const record = await rpb.agents(provider.address);
    // record[7] is totalInferenceServed based on struct order
    console.log("  Provider inference served:", record[7].toString(), "units");
    expect(record[7]).to.equal(25n);
  });
});

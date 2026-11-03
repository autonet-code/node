const { expect } = require("chai");
const { ethers } = require("hardhat");
const { deploySubstrateWith, mintATN, SLOT_GOVERNOR } = require("./helpers");

// The service fee and its treasury share are set at deploy and governed
// afterwards: the right values change as the jurisdiction grows.

describe("Substrate fee settings", function () {
  let signers, gov, payer, provider;

  beforeEach(async function () {
    signers = await ethers.getSigners();
    [, , , , , , payer, provider] = signers;
    gov = signers[SLOT_GOVERNOR];
  });

  it("takes both from the constructor", async function () {
    const s = await deploySubstrateWith({ serviceFeeBps: 400, feeTreasuryBps: 2000 });
    expect(await s.serviceFeeBps()).to.equal(400);
    expect(await s.feeTreasuryBps()).to.equal(2000);
  });

  it("refuses out-of-bounds deploy values", async function () {
    await expect(deploySubstrateWith({ serviceFeeBps: 49 })).to.be.reverted;
    await expect(deploySubstrateWith({ serviceFeeBps: 1001 })).to.be.reverted;
    await expect(deploySubstrateWith({ feeTreasuryBps: 9001 })).to.be.reverted;
  });

  it("splits each fee by the treasury share", async function () {
    const s = await deploySubstrateWith({ serviceFeeBps: 1000, feeTreasuryBps: 2000 });
    await mintATN(s, payer, payer, 10_000n);
    const treasury = await s.treasury();
    const before = await s.balanceOf(treasury);
    await expect(s.connect(payer).payForService(provider.address, 10_000n, ethers.ZeroHash))
      .to.emit(s, "ServiceFee")
      .withArgs(payer.address, provider.address, 10_000n, 800n, 200n);
    expect(await s.balanceOf(treasury)).to.equal(before + 200n);
    expect(await s.balanceOf(provider.address)).to.equal(9_000n);
  });

  it("lets only the governor retune the treasury share, within bounds", async function () {
    const s = await deploySubstrateWith();
    await expect(s.connect(payer).setFeeTreasuryBps(1000))
      .to.be.revertedWithCustomError(s, "NotGovernor");
    await expect(s.connect(gov).setFeeTreasuryBps(9001))
      .to.be.revertedWithCustomError(s, "FeeOutOfBounds");
    await expect(s.connect(gov).setFeeTreasuryBps(0))
      .to.emit(s, "FeeTreasuryBpsSet").withArgs(5000, 0);
    expect(await s.feeTreasuryBps()).to.equal(0);
  });
});

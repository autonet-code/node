const { expect } = require("chai");
const { ethers } = require("hardhat");
const {
  ZERO,
  ONE,
  digest,
  deploySubstrateWith,
  deployIndexToken,
  fund,
  feeOf,
  anchorFields,
  anchorTx,
  anchorEpoch,
  SLOT_GOVERNOR,
} = require("./helpers");

// Substrate is not a token: it anchors epochs, registers agents and tools,
// takes the service fee in the value index's tokens and pays the pools out.

describe("Substrate: surface, deploy guards, fee and anchors", function () {
  let signers, deployer, gov, payer, recipient;

  beforeEach(async function () {
    signers = await ethers.getSigners();
    [deployer, , , , , , payer, recipient] = signers;
    gov = signers[SLOT_GOVERNOR];
  });

  it("has no token surface (no ERC-20, no vault mint, no genesis claim)", async function () {
    const s = await deploySubstrateWith();
    for (const f of [
      "balanceOf", "transfer", "transferFrom", "approve", "totalSupply",
      "name", "symbol", "decimals", "mintFromVault", "claimGenesis",
      "claimFor", "agentMintTotal", "atnTotalSupply", "serviceEarnings",
    ]) {
      expect(s.interface.hasFunction(f), f).to.equal(false);
    }
    for (const f of [
      "payForService", "collectFee", "claim", "withdraw", "withdrawTo",
      "withdrawTreasury", "sweep", "poolCap", "earnedValue", "parityOf",
      "acceptsToken", "valueOf", "renounceUpgrades",
    ]) {
      expect(s.interface.hasFunction(f), f).to.equal(true);
    }
  });

  it("refuses a zero governor and a zero registry at deploy", async function () {
    const F = await ethers.getContractFactory("Substrate");
    await expect(F.deploy(deployer.address, ZERO, ZERO, ONE, 250, 5000))
      .to.be.revertedWithCustomError(F, "NoAnchorAuthority");
    await expect(F.deploy(deployer.address, ONE, ZERO, ZERO, 250, 5000))
      .to.be.revertedWithCustomError(F, "RegistryRequired");
  });

  describe("setServiceFeeBps", function () {
    it("governor can retune within bounds; payments use the new value", async function () {
      const s = await deploySubstrateWith();
      const usd = await deployIndexToken(s);
      expect(await s.serviceFeeBps()).to.equal(250);
      await expect(s.connect(gov).setServiceFeeBps(1000))
        .to.emit(s, "ServiceFeeBpsSet").withArgs(250, 1000);
      await fund(usd, payer, s, 10_000n);
      await s.connect(payer).payForService(await usd.getAddress(), recipient.address, 10_000n, digest("x"));
      expect(await usd.balanceOf(recipient.address)).to.equal(9_000n);
      expect(await usd.balanceOf(await s.getAddress())).to.equal(1_000n);
      expect(await s.feeOf(10_000n)).to.equal(1_000n);
    });

    it("non-governor cannot retune", async function () {
      const s = await deploySubstrateWith();
      await expect(s.connect(payer).setServiceFeeBps(500))
        .to.be.revertedWithCustomError(s, "NotGovernor");
    });

    it("reverts below the lower bound and above the upper bound", async function () {
      const s = await deploySubstrateWith();
      await expect(s.connect(gov).setServiceFeeBps(49))
        .to.be.revertedWithCustomError(s, "FeeOutOfBounds");
      await expect(s.connect(gov).setServiceFeeBps(1001))
        .to.be.revertedWithCustomError(s, "FeeOutOfBounds");
      await s.connect(gov).setServiceFeeBps(50);
      expect(await s.serviceFeeBps()).to.equal(50);
      await s.connect(gov).setServiceFeeBps(1000);
      expect(await s.serviceFeeBps()).to.equal(1000);
    });

    it("feeOf rounds up", async function () {
      const s = await deploySubstrateWith();
      expect(await s.feeOf(1n)).to.equal(1n);
      expect(await s.feeOf(40n)).to.equal(1n);
      expect(await s.feeOf(41n)).to.equal(2n);
      expect(await s.feeOf(1_000_000n)).to.equal(feeOf(1_000_000n));
    });
  });

  describe("anchor cutoffs and the pool cutoff", function () {
    it("a cutoff must be after the latest one and before the anchoring block", async function () {
      const s = await deploySubstrateWith();
      const f = await anchorFields(s, "e1");
      // Not before this block: the next block's own timestamp is too late.
      const next = BigInt((await ethers.provider.getBlock("latest")).timestamp) + 100n;
      await ethers.provider.send("evm_setNextBlockTimestamp", [Number(next)]);
      await expect(anchorTx(s, gov, { ...f, cutoff: next }))
        .to.be.revertedWithCustomError(s, "CutoffOutOfOrder");
      // Not at or before the latest cutoff (deployedAt before any anchor).
      await expect(anchorTx(s, gov, { ...f, cutoff: await s.latestCutoff() }))
        .to.be.revertedWithCustomError(s, "CutoffOutOfOrder");
      await anchorTx(s, gov, f);
      expect(await s.latestCutoff()).to.equal(f.cutoff);
    });

    it("poolCutoff is the cutoff of the anchor before the latest (0 until two)", async function () {
      const s = await deploySubstrateWith();
      expect(await s.poolCutoff()).to.equal(0n);
      const a1 = await anchorEpoch(s, "e1");
      expect(await s.poolCutoff()).to.equal(0n);
      const a2 = await anchorEpoch(s, "e2");
      expect(await s.poolCutoff()).to.equal(a1.cutoff);
      await anchorEpoch(s, "e3");
      expect(await s.poolCutoff()).to.equal(a2.cutoff);
      expect(await s.anchorCount()).to.equal(3n);
      expect((await s.getAnchorByEpochId("e2")).cutoff).to.equal(a2.cutoff);
    });

    it("the anchor hash is the content hash anchorHashOf computes", async function () {
      const s = await deploySubstrateWith();
      const f = await anchorFields(s, "e1", digest("root"));
      await anchorTx(s, gov, f);
      expect(await s.latestAnchorHash()).to.equal(await s.anchorHashOf(
        f.epochId, f.epochRoot, f.prevEpochRoot, f.prevAnchorHash, f.claimCid,
        f.payloadHash, f.claimRoot, f.cutoff));
      expect(await s.anchorRoot(0)).to.equal(digest("root"));
    });

    it("the anchorer may anchor once the governor sets it", async function () {
      const s = await deploySubstrateWith();
      const anchorer = signers[5];
      await expect(anchorTx(s, anchorer, await anchorFields(s, "e1")))
        .to.be.revertedWithCustomError(s, "NotGovernor");
      await expect(s.connect(anchorer).setAnchorer(anchorer.address))
        .to.be.revertedWithCustomError(s, "NotGovernor");
      await s.connect(gov).setAnchorer(anchorer.address);
      await anchorTx(s, anchorer, await anchorFields(s, "e1"));
      expect(await s.isAnchored("e1")).to.equal(true);
    });
  });
});

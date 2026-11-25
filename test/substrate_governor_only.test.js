const { expect } = require("chai");
const { ethers } = require("hardhat");
const {
  ZERO,
  ONE,
  digest,
  registerAgent,
  deploySubstrateWith,
  deployIndexToken,
  fund,
  payFees,
  anchorFields,
  anchorTx,
  claimTree,
  claimTx,
  SLOT_GOVERNOR,
} = require("./helpers");

// Governor-only anchoring and the pools' on-chain cap.
//
// Substrate.anchor is callable only by the governor (the DAO timelock in
// production) or the anchorer (the AnchorElection). The hash chain and the
// lag-one cap on the pools stay enforced by Substrate. The full proposal
// lifecycle through an OZ Governor + timelock is covered on the Python side
// (tests/test_governed_anchor.py).

describe("Substrate governor-only anchoring + the pool cap", function () {
  let signers, attacker, a1, a2, provider, gov;

  beforeEach(async function () {
    signers = await ethers.getSigners();
    [, attacker, , a1, a2, , , provider] = signers;
    gov = signers[SLOT_GOVERNOR];
  });

  describe("anchor authority", function () {
    let substrate;
    beforeEach(async function () {
      substrate = await deploySubstrateWith();
      await registerAgent(substrate, attacker, "attacker");
    });

    it("rejects a zero governor at deploy", async function () {
      const F = await ethers.getContractFactory("Substrate");
      await expect(F.deploy(ZERO, ZERO, ZERO, ONE, 250, 5000))
        .to.be.revertedWithCustomError(F, "NoAnchorAuthority");
    });

    it("exposes the governor", async function () {
      expect(await substrate.governor()).to.equal(gov.address);
    });

    it("non-governor anchor reverts, even from a registered agent", async function () {
      const f = await anchorFields(substrate, "evil", digest("evil-root"));
      await expect(anchorTx(substrate, attacker, f)).to.be.revertedWithCustomError(
        substrate, "NotGovernor");
      expect(await substrate.anchorCount()).to.equal(0n);
    });

    it("governor anchors; EpochAnchored is emitted", async function () {
      const f = await anchorFields(substrate, "e1");
      await expect(anchorTx(substrate, gov, f)).to.emit(substrate, "EpochAnchored");
      expect(await substrate.isAnchored("e1")).to.equal(true);
      expect(await substrate.isAnchoredByHash(digest("e1"))).to.equal(true);
      expect(await substrate.anchorCount()).to.equal(1n);
    });

    it("hash chain binds the governor too: stale prev root / hash and re-anchor revert", async function () {
      await anchorTx(substrate, gov, await anchorFields(substrate, "e1"));
      const f = await anchorFields(substrate, "e2");
      await expect(
        anchorTx(substrate, gov, { ...f, prevEpochRoot: ethers.ZeroHash })
      ).to.be.revertedWithCustomError(substrate, "PrevEpochRootMismatch");
      await expect(
        anchorTx(substrate, gov, { ...f, prevAnchorHash: ethers.ZeroHash })
      ).to.be.revertedWithCustomError(substrate, "PrevAnchorHashMismatch");
      await expect(
        anchorTx(substrate, gov, { ...f, epochId: "e1" })
      ).to.be.revertedWithCustomError(substrate, "EpochAlreadyAnchored");
      await expect(
        anchorTx(substrate, gov, { ...f, epochId: "" })
      ).to.be.revertedWithCustomError(substrate, "EpochIdEmpty");
      await anchorTx(substrate, gov, f);
      expect(await substrate.anchorCount()).to.equal(2n);
    });
  });

  describe("on-chain cap (lag one)", function () {
    let substrate, usd, U;
    beforeEach(async function () {
      substrate = await deploySubstrateWith();
      usd = await deployIndexToken(substrate);
      U = await usd.getAddress();
    });

    it("claims of a pool never exceed its tool share up to the previous cutoff", async function () {
      // 1_000_000 gross, fee 2.5% = 25_000, half to the treasury: 12_500 tool share.
      const { toolShare } = await payFees(substrate, usd, 1_000_000n);
      expect(toolShare).to.equal(12_500n);
      expect(await substrate.toolShareTotal(U)).to.equal(12_500n);
      await anchorTx(substrate, gov, await anchorFields(substrate, "e1"));
      expect(await substrate.poolCap(U)).to.equal(0n);

      // The second anchor's root over-allocates (12_000 + 600 > 12_500).
      const tree = claimTree([
        { earner: a1.address, tokens: [U], amounts: [12_000n] },
        { earner: a2.address, tokens: [U], amounts: [600n] },
      ]);
      await anchorTx(substrate, gov, await anchorFields(substrate, "e2", tree.root));
      expect(await substrate.poolCap(U)).to.equal(12_500n);

      await claimTx(substrate, a1, tree, 0, "e2");
      await expect(claimTx(substrate, a2, tree, 1, "e2"))
        .to.be.revertedWithCustomError(substrate, "PoolExceeded")
        .withArgs(U, 12_500n, 12_600n);
      expect(await substrate.claimedTotal(U)).to.equal(12_000n);
    });

    it("fees after the previous cutoff wait for the next anchor", async function () {
      await payFees(substrate, usd, 1_000_000n);              // 12_500
      await anchorTx(substrate, gov, await anchorFields(substrate, "e1"));
      await payFees(substrate, usd, 2_000_000n);              // 25_000, epoch 2
      await anchorTx(substrate, gov, await anchorFields(substrate, "e2"));
      expect(await substrate.poolCap(U)).to.equal(12_500n);
      await anchorTx(substrate, gov, await anchorFields(substrate, "e3"));
      expect(await substrate.poolCap(U)).to.equal(37_500n);
    });

    it("a contract can pay a service and its fee feeds the pool", async function () {
      const M = await ethers.getContractFactory("ServicePayerMock");
      const payerContract = await M.deploy();
      await payerContract.waitForDeployment();
      await usd.mint(await payerContract.getAddress(), 400_000n);
      await expect(
        payerContract.pay(await substrate.getAddress(), U, provider.address, 400_000n, digest("pay"))
      ).to.emit(substrate, "ServicePayment");
      // fee 10_000; 5_000 owed to the treasury, 5_000 tool share.
      expect(await usd.balanceOf(provider.address)).to.equal(390_000n);
      expect(await substrate.treasuryOwed(U)).to.equal(5_000n);
      expect(await substrate.toolShareTotal(U)).to.equal(5_000n);
    });
  });
});

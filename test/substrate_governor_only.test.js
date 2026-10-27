const { expect } = require("chai");
const { ethers } = require("hardhat");
const {
  ZERO,
  digest,
  registerAgent,
  mintLeaf,
  mintATN,
  deploySubstrateWith,
  seedPool,
  anchorFields,
  anchorTx,
  SLOT_GOVERNOR,
} = require("./helpers");

// Governor-only anchoring + on-chain conservation (Decision 2026-10-16).
//
// Substrate.anchor is callable only by the governor (the DAO timelock in
// production: each anchor is an ordinary DAO proposal). The hash chain and
// the fees-only mint pool stay enforced by Substrate. The full proposal
// lifecycle through an OZ Governor + timelock is covered on the Python side
// (tests/test_governed_anchor.py).

describe("Substrate governor-only anchoring + conservation", function () {
  let signers, attacker, a1, a2, payer, provider, gov;

  beforeEach(async function () {
    signers = await ethers.getSigners();
    [, attacker, , a1, a2, , payer, provider] = signers;
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
      await expect(F.deploy(ZERO, ZERO, ZERO, ZERO)).to.be.revertedWithCustomError(
        F, "NoAnchorAuthority");
    });

    it("exposes the governor", async function () {
      expect(await substrate.governor()).to.equal(gov.address);
    });

    it("non-governor anchor reverts, even from a registered agent", async function () {
      const f = await anchorFields(substrate, "evil", mintLeaf(attacker.address, 10n ** 12n));
      await expect(anchorTx(substrate, attacker, f)).to.be.revertedWithCustomError(
        substrate, "NotGovernor");
      expect(await substrate.anchorCount()).to.equal(0n);
    });

    it("governor anchors; EpochAnchored is emitted", async function () {
      const f = await anchorFields(substrate, "e1");
      await expect(anchorTx(substrate, gov, f)).to.emit(substrate, "EpochAnchored");
      expect(await substrate.isAnchored("e1")).to.equal(true);
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
      await anchorTx(substrate, gov, f);
      expect(await substrate.anchorCount()).to.equal(2n);
    });
  });

  describe("on-chain conservation (fees-only pool)", function () {
    let substrate;
    beforeEach(async function () {
      substrate = await deploySubstrateWith();
      await registerAgent(substrate, a1, "a1");
      await registerAgent(substrate, a2, "a2");
    });

    it("first anchor has a zero pool: any mint reverts", async function () {
      const f = await anchorFields(substrate, "e1", mintLeaf(a1.address, 1n));
      await anchorTx(substrate, gov, f);
      await expect(
        substrate.connect(a1).claimFor(a1.address, 1n, digest("e1"), [])
      ).to.be.revertedWithCustomError(substrate, "EpochPoolExceeded");
    });

    it("pool = previous window's burned fees; mint over pool reverts", async function () {
      // 1_000_000 gross, fee 2.5% = 25_000, half to treasury -> 12_500 burned.
      await mintATN(substrate, null, payer, 1_000_000n);
      await substrate.connect(payer).payForService(provider.address, 1_000_000n, digest("x"));
      expect(await substrate.burnedSinceAnchor()).to.equal(12_500n);
      let f = await anchorFields(substrate, "e1");
      await anchorTx(substrate, gov, f);
      expect(await substrate.burnedSinceAnchor()).to.equal(0n);
      expect(await substrate.nextMintPool()).to.equal(12_500n);

      // Signed root over-allocates (12_000 + 600 > 12_500): the late claim reverts.
      const l1 = mintLeaf(a1.address, 12_000n);
      const l2 = mintLeaf(a2.address, 600n);
      const [lo, hi] = l1 <= l2 ? [l1, l2] : [l2, l1];
      const root = ethers.keccak256(ethers.solidityPacked(["bytes32", "bytes32"], [lo, hi]));
      f = await anchorFields(substrate, "e2", root);
      await anchorTx(substrate, gov, f);
      const anchor = await substrate.getAnchorByEpochId("e2");
      expect(anchor.mintPool).to.equal(12_500n);

      await substrate.connect(a1).claimFor(a1.address, 12_000n, digest("e2"), [l2]);
      await expect(substrate.connect(a2).claimFor(a2.address, 600n, digest("e2"), [l1]))
        .to.be.revertedWithCustomError(substrate, "EpochPoolExceeded")
        .withArgs(digest("e2"), 12_500n, 12_600n);
      expect((await substrate.getAnchorByEpochId("e2")).minted).to.equal(12_000n);
    });

    it("honest path: pool-exact mints all claim", async function () {
      const pool = await seedPool(substrate);
      const half = pool / 2n;
      const rest = pool - half;
      const l1 = mintLeaf(a1.address, half);
      const l2 = mintLeaf(a2.address, rest);
      const [lo, hi] = l1 <= l2 ? [l1, l2] : [l2, l1];
      const root = ethers.keccak256(ethers.solidityPacked(["bytes32", "bytes32"], [lo, hi]));
      const f = await anchorFields(substrate, "h1", root);
      await anchorTx(substrate, gov, f);
      await substrate.connect(a1).claimFor(a1.address, half, digest("h1"), [l2]);
      await substrate.connect(a2).claimFor(a2.address, rest, digest("h1"), [l1]);
      expect(await substrate.networkMintTotal()).to.equal(pool);
    });

    it("a contract can call payForService (ToolBond slashing path) and its fee feeds the pool", async function () {
      const M = await ethers.getContractFactory("ServicePayerMock");
      const payerContract = await M.deploy();
      await payerContract.waitForDeployment();
      const pcAddr = await payerContract.getAddress();
      await substrate.connect(signers[19]).mintFromVault(pcAddr, 400_000n);
      await expect(
        payerContract.pay(await substrate.getAddress(), provider.address, 400_000n, digest("slash"))
      ).to.emit(substrate, "ServicePayment");
      // fee 10_000; 5_000 to treasury, 5_000 burned.
      expect(await substrate.balanceOf(provider.address)).to.equal(390_000n);
      expect(await substrate.serviceEarnings(provider.address)).to.equal(390_000n);
      expect(await substrate.burnedSinceAnchor()).to.equal(5_000n);
    });
  });
});

const { expect } = require("chai");
const { ethers } = require("hardhat");
const { anyValue } = require("@nomicfoundation/hardhat-chai-matchers/withArgs");
const { digest, registerAgent, mintLeaf, mintATN } = require("./helpers");

const ZERO = "0x0000000000000000000000000000000000000000";

// Deploy Substrate with an explicit governor (default helper wires governor=0,
// which freezes the fee knob — these tests need a live governor).
async function deployWith(treasury, vaultMinter, governor) {
  const Substrate = await ethers.getContractFactory("Substrate");
  const s = await Substrate.deploy(treasury, vaultMinter, governor);
  await s.waitForDeployment();
  return s;
}

// Sorted-pair keccak of two leaves (OpenZeppelin convention).
function hashPair(a, b) {
  const [lo, hi] = a <= b ? [a, b] : [b, a];
  return ethers.keccak256(ethers.solidityPacked(["bytes32", "bytes32"], [lo, hi]));
}

describe("Substrate — fees-only emission + REP-from-earnings (2026-07-10)", function () {
  let deployer, gov, faucet, submitter, agentA, agentB, payer, recipient;

  beforeEach(async function () {
    [deployer, gov, faucet, submitter, agentA, agentB, payer, recipient] =
      await ethers.getSigners();
  });

  // -------------------------------------------------------------------------
  // Money-only merkle leaf round-trip.
  // -------------------------------------------------------------------------
  describe("recordTrainingForEpoch (money-only 2-field leaf)", function () {
    let substrate;
    beforeEach(async function () {
      substrate = await deployWith(deployer.address, ZERO, ZERO);
      await registerAgent(substrate, submitter, "submitter");
      await registerAgent(substrate, agentA, "agentA");
      await registerAgent(substrate, agentB, "agentB");
    });

    it("single-leaf tree: root == leaf, empty proof mints ATN and accrues earnings", async function () {
      const amount = 1_000_000n;
      const epochId = "epoch-single";
      const root = mintLeaf(agentA.address, amount);
      await substrate
        .connect(submitter)
        .submitAnchor(
          epochId,
          digest("root-" + epochId),
          await substrate.latestEpochRoot(),
          await substrate.latestAnchorHash(),
          "cid",
          digest("payload"),
          root
        );
      await expect(
        substrate
          .connect(agentA)
          .recordTrainingForEpoch(amount, digest(epochId), [])
      )
        .to.emit(substrate, "TrainingRecorded")
        .withArgs(agentA.address, digest(epochId), amount, amount);

      // ATN minted at the ratified amount; earnings ledger accrues it 1:1.
      expect(await substrate.balanceOf(agentA.address)).to.equal(amount);
      expect(await substrate.agentMintTotal(agentA.address)).to.equal(amount);
      expect(await substrate.networkMintTotal()).to.equal(amount);
      expect(await substrate.mintForEpoch(agentA.address, digest(epochId))).to.equal(amount);
    });

    it("two-leaf tree: each agent verifies with the sibling as proof", async function () {
      const amtA = 700_000n;
      const amtB = 300_000n;
      const leafA = mintLeaf(agentA.address, amtA);
      const leafB = mintLeaf(agentB.address, amtB);
      const root = hashPair(leafA, leafB);
      const epochId = "epoch-two";

      await substrate
        .connect(submitter)
        .submitAnchor(
          epochId,
          digest("root-" + epochId),
          await substrate.latestEpochRoot(),
          await substrate.latestAnchorHash(),
          "cid",
          digest("payload"),
          root
        );

      // agentA proves with leafB as its sibling.
      await substrate
        .connect(agentA)
        .recordTrainingForEpoch(amtA, digest(epochId), [leafB]);
      await substrate
        .connect(agentB)
        .recordTrainingForEpoch(amtB, digest(epochId), [leafA]);

      expect(await substrate.balanceOf(agentA.address)).to.equal(amtA);
      expect(await substrate.balanceOf(agentB.address)).to.equal(amtB);
      expect(await substrate.networkMintTotal()).to.equal(amtA + amtB);
    });

    it("wrong amount does not verify against the leaf", async function () {
      const amount = 500_000n;
      const epochId = "epoch-bad";
      const root = mintLeaf(agentA.address, amount);
      await substrate
        .connect(submitter)
        .submitAnchor(
          epochId,
          digest("root-" + epochId),
          await substrate.latestEpochRoot(),
          await substrate.latestAnchorHash(),
          "cid",
          digest("payload"),
          root
        );
      await expect(
        substrate
          .connect(agentA)
          .recordTrainingForEpoch(amount + 1n, digest(epochId), [])
      ).to.be.revertedWithCustomError(substrate, "MintProofInvalid");
    });

    it("double submit for the same epoch reverts", async function () {
      const amount = 123_456n;
      const epochId = "epoch-dup";
      const root = mintLeaf(agentA.address, amount);
      await substrate
        .connect(submitter)
        .submitAnchor(
          epochId,
          digest("root-" + epochId),
          await substrate.latestEpochRoot(),
          await substrate.latestAnchorHash(),
          "cid",
          digest("payload"),
          root
        );
      await substrate.connect(agentA).recordTrainingForEpoch(amount, digest(epochId), []);
      await expect(
        substrate.connect(agentA).recordTrainingForEpoch(amount, digest(epochId), [])
      ).to.be.revertedWithCustomError(substrate, "AlreadySubmittedForEpoch");
    });

    it("unanchored epoch reverts", async function () {
      await expect(
        substrate.connect(agentA).recordTrainingForEpoch(1n, digest("nope"), [])
      ).to.be.revertedWithCustomError(substrate, "EpochNotAnchored");
    });
  });

  // -------------------------------------------------------------------------
  // Reputation surface is gone (Substrate is pure money now).
  // -------------------------------------------------------------------------
  it("reputation views are removed from the ABI", async function () {
    const substrate = await deployWith(deployer.address, ZERO, ZERO);
    expect(substrate.interface.hasFunction("agentReputation")).to.equal(false);
    expect(substrate.interface.hasFunction("reputationOfAt")).to.equal(false);
    expect(substrate.interface.hasFunction("reputationTotalSupplyAt")).to.equal(false);
    // Money surface stays.
    expect(substrate.interface.hasFunction("balanceOfAt")).to.equal(true);
    expect(substrate.interface.hasFunction("atnTotalSupplyAt")).to.equal(true);
    expect(substrate.interface.hasFunction("agentMintTotal")).to.equal(true);
    expect(substrate.interface.hasFunction("serviceEarnings")).to.equal(true);
  });

  // -------------------------------------------------------------------------
  // serviceEarnings ledger (the second REP claim base).
  // -------------------------------------------------------------------------
  describe("serviceEarnings", function () {
    let substrate;
    const feeOf = (g, bps) => (g * bps) / 10000n;
    const netOf = (g, bps) => g - feeOf(g, bps);

    beforeEach(async function () {
      substrate = await deployWith(deployer.address, ZERO, gov.address);
      await registerAgent(substrate, faucet, "faucet");
      await registerAgent(substrate, submitter, "submitter");
      // Fund the payer with ATN via the faucet.
      await mintATN(substrate, submitter, faucet, 100_000n);
      await substrate.connect(faucet).transfer(payer.address, 100_000n);
    });

    it("accumulates net (post-fee) per recipient across payForService calls", async function () {
      const bps = await substrate.serviceFeeBps();
      await substrate.connect(payer).payForService(recipient.address, 10_000n, digest("r1"));
      expect(await substrate.serviceEarnings(recipient.address)).to.equal(netOf(10_000n, bps));

      await substrate.connect(payer).payForService(recipient.address, 4_000n, digest("r2"));
      expect(await substrate.serviceEarnings(recipient.address)).to.equal(
        netOf(10_000n, bps) + netOf(4_000n, bps)
      );
      // A different recipient tracks separately.
      expect(await substrate.serviceEarnings(payer.address)).to.equal(0n);
    });

    it("accumulates net when payForService is driven via PaymentChannel closeChannel", async function () {
      const bps = await substrate.serviceFeeBps();
      const Ch = await ethers.getContractFactory("PaymentChannel");
      const channel = await Ch.deploy(await substrate.getAddress(), 3600);
      await channel.waitForDeployment();

      // recipient must be a registered agent to be a channel provider.
      await registerAgent(substrate, recipient, "provider");
      await substrate.connect(payer).approve(await channel.getAddress(), 50_000n);
      await channel.connect(payer).openChannel(recipient.address, 5_000n);
      const id = await channel.channelCount();

      const net = await ethers.provider.getNetwork();
      const domain = {
        name: "AutonetPaymentChannel",
        version: "1",
        chainId: net.chainId,
        verifyingContract: await channel.getAddress(),
      };
      const types = {
        Voucher: [
          { name: "channelId", type: "uint256" },
          { name: "cumulativeAmount", type: "uint256" },
        ],
      };
      const cumulative = 3_000n;
      const sig = await payer.signTypedData(domain, types, {
        channelId: id,
        cumulativeAmount: cumulative,
      });
      await channel.connect(recipient).closeChannel(id, cumulative, sig);

      // The channel (as payer) called payForService(recipient, 3000) — the
      // recipient's earnings ledger reflects the post-fee net.
      expect(await substrate.serviceEarnings(recipient.address)).to.equal(
        netOf(cumulative, bps)
      );
    });
  });

  // -------------------------------------------------------------------------
  // Governed service fee.
  // -------------------------------------------------------------------------
  describe("setServiceFeeBps", function () {
    it("governor can retune within bounds; fee math uses the new value", async function () {
      const substrate = await deployWith(deployer.address, ZERO, gov.address);
      await registerAgent(substrate, faucet, "faucet");
      await registerAgent(substrate, submitter, "submitter");
      await mintATN(substrate, submitter, faucet, 100_000n);
      await substrate.connect(faucet).transfer(payer.address, 100_000n);

      expect(await substrate.serviceFeeBps()).to.equal(250);
      await expect(substrate.connect(gov).setServiceFeeBps(1000))
        .to.emit(substrate, "ServiceFeeBpsSet")
        .withArgs(250, 1000);
      expect(await substrate.serviceFeeBps()).to.equal(1000);

      // Fee now 10%: recipient nets 90% of a 10_000 payment.
      await substrate.connect(payer).payForService(recipient.address, 10_000n, digest("x"));
      expect(await substrate.balanceOf(recipient.address)).to.equal(9_000n);
      expect(await substrate.serviceEarnings(recipient.address)).to.equal(9_000n);
    });

    it("non-governor cannot retune", async function () {
      const substrate = await deployWith(deployer.address, ZERO, gov.address);
      await expect(
        substrate.connect(payer).setServiceFeeBps(500)
      ).to.be.revertedWithCustomError(substrate, "NotGovernor");
    });

    it("reverts below the lower bound and above the upper bound", async function () {
      const substrate = await deployWith(deployer.address, ZERO, gov.address);
      await expect(
        substrate.connect(gov).setServiceFeeBps(49)
      ).to.be.revertedWithCustomError(substrate, "FeeOutOfBounds");
      await expect(
        substrate.connect(gov).setServiceFeeBps(1001)
      ).to.be.revertedWithCustomError(substrate, "FeeOutOfBounds");
      // Bounds themselves are accepted.
      await substrate.connect(gov).setServiceFeeBps(50);
      expect(await substrate.serviceFeeBps()).to.equal(50);
      await substrate.connect(gov).setServiceFeeBps(1000);
      expect(await substrate.serviceFeeBps()).to.equal(1000);
    });

    it("zero-address governor freezes the knob (setter always reverts)", async function () {
      const substrate = await deployWith(deployer.address, ZERO, ZERO);
      // Even the zero address itself cannot call (governor==0 branch).
      await expect(
        substrate.connect(gov).setServiceFeeBps(500)
      ).to.be.revertedWithCustomError(substrate, "NotGovernor");
      expect(await substrate.serviceFeeBps()).to.equal(250);
    });
  });
});

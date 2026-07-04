const { expect } = require("chai");
const { ethers } = require("hardhat");

// Local helpers (mirrors test/helpers.js conventions, kept inline so this
// suite is self-contained).
const digest = (s) => ethers.keccak256(ethers.toUtf8Bytes(s));

async function registerAgent(substrate, signer, lineage) {
  return substrate
    .connect(signer)
    .registerAgent(digest(lineage), ethers.toUtf8Bytes("peer-" + lineage));
}

// Sign an EIP-712 Sponsorship{agent, parent} for Substrate as `owner`.
async function signSponsorship(substrate, ownerSigner, agent, parent) {
  const net = await ethers.provider.getNetwork();
  const domain = {
    name: "AutonetSubstrate",
    version: "1",
    chainId: net.chainId,
    verifyingContract: await substrate.getAddress(),
  };
  const types = {
    Sponsorship: [
      { name: "agent", type: "address" },
      { name: "parent", type: "address" },
    ],
  };
  return ownerSigner.signTypedData(domain, types, { agent, parent });
}

describe("Substrate owner-rooted registration (AgentSponsored)", function () {
  let substrate;
  let deployer, ownerW, otherOwnerW, agentA, agentB, agentC, stranger;
  const ZERO = ethers.ZeroAddress;

  beforeEach(async function () {
    [deployer, ownerW, otherOwnerW, agentA, agentB, agentC, stranger] =
      await ethers.getSigners();
    const Substrate = await ethers.getContractFactory("Substrate");
    substrate = await Substrate.deploy();
    await substrate.waitForDeployment();
  });

  // ---- happy path: fresh sponsored registration --------------------------

  it("fresh top-level sponsored registration binds owner, emits both events", async function () {
    const sig = await signSponsorship(substrate, ownerW, agentA.address, ZERO);
    await expect(
      substrate
        .connect(agentA)
        .registerAgentSponsored(
          digest("lineA"),
          ethers.toUtf8Bytes("peer-A"),
          ownerW.address,
          ZERO,
          sig
        )
    )
      .to.emit(substrate, "AgentRegistered")
      .and.to.emit(substrate, "AgentSponsored")
      .withArgs(agentA.address, ownerW.address, ZERO);

    expect(await substrate.isRegistered(agentA.address)).to.equal(true);
    expect(await substrate.getAgentOwner(agentA.address)).to.equal(ownerW.address);
    expect(await substrate.getAgentParent(agentA.address)).to.equal(ZERO);
  });

  it("fresh sponsored registration with a parent under the same owner", async function () {
    // Parent A first (top-level).
    const sigA = await signSponsorship(substrate, ownerW, agentA.address, ZERO);
    await substrate
      .connect(agentA)
      .registerAgentSponsored(
        digest("lineA"),
        ethers.toUtf8Bytes("peer-A"),
        ownerW.address,
        ZERO,
        sigA
      );

    // Child B under A, same owner.
    const sigB = await signSponsorship(
      substrate,
      ownerW,
      agentB.address,
      agentA.address
    );
    await expect(
      substrate
        .connect(agentB)
        .registerAgentSponsored(
          digest("lineB"),
          ethers.toUtf8Bytes("peer-B"),
          ownerW.address,
          agentA.address,
          sigB
        )
    )
      .to.emit(substrate, "AgentSponsored")
      .withArgs(agentB.address, ownerW.address, agentA.address);

    expect(await substrate.getAgentParent(agentB.address)).to.equal(agentA.address);
    expect(await substrate.getAgentOwner(agentB.address)).to.equal(ownerW.address);
  });

  // ---- happy path: retrofit on an existing ownerless agent ---------------

  it("retrofit: legacy agent attaches owner+parent once", async function () {
    // A registered the legacy way (ownerless).
    await registerAgent(substrate, agentA, "lineA");
    expect(await substrate.getAgentOwner(agentA.address)).to.equal(ZERO);

    const sig = await signSponsorship(substrate, ownerW, agentA.address, ZERO);
    await expect(
      substrate.connect(agentA).attachOwner(ownerW.address, ZERO, sig)
    )
      .to.emit(substrate, "AgentSponsored")
      .withArgs(agentA.address, ownerW.address, ZERO);

    expect(await substrate.getAgentOwner(agentA.address)).to.equal(ownerW.address);
  });

  it("retrofit with a parent under the same owner", async function () {
    // A is sponsored top-level.
    const sigA = await signSponsorship(substrate, ownerW, agentA.address, ZERO);
    await substrate
      .connect(agentA)
      .registerAgentSponsored(
        digest("lineA"),
        ethers.toUtf8Bytes("peer-A"),
        ownerW.address,
        ZERO,
        sigA
      );
    // B registered legacy, retrofits under A.
    await registerAgent(substrate, agentB, "lineB");
    const sigB = await signSponsorship(
      substrate,
      ownerW,
      agentB.address,
      agentA.address
    );
    await substrate.connect(agentB).attachOwner(ownerW.address, agentA.address, sigB);
    expect(await substrate.getAgentParent(agentB.address)).to.equal(agentA.address);
    expect(await substrate.getAgentOwner(agentB.address)).to.equal(ownerW.address);
  });

  // ---- bad signature paths -----------------------------------------------

  it("wrong signer (not the claimed owner) reverts", async function () {
    // otherOwnerW signs, but we claim ownerW as owner.
    const sig = await signSponsorship(substrate, otherOwnerW, agentA.address, ZERO);
    await expect(
      substrate
        .connect(agentA)
        .registerAgentSponsored(
          digest("lineA"),
          ethers.toUtf8Bytes("peer-A"),
          ownerW.address,
          ZERO,
          sig
        )
    ).to.be.revertedWithCustomError(substrate, "BadSponsorshipSignature");
  });

  it("tampered agent address in sig (signed for a different agent) reverts", async function () {
    // owner signed for agentB, but agentA submits.
    const sig = await signSponsorship(substrate, ownerW, agentB.address, ZERO);
    await expect(
      substrate
        .connect(agentA)
        .registerAgentSponsored(
          digest("lineA"),
          ethers.toUtf8Bytes("peer-A"),
          ownerW.address,
          ZERO,
          sig
        )
    ).to.be.revertedWithCustomError(substrate, "BadSponsorshipSignature");
  });

  it("wrong parent in sig vs arg reverts", async function () {
    // Register A top-level so it's a valid same-owner parent candidate.
    const sigA = await signSponsorship(substrate, ownerW, agentA.address, ZERO);
    await substrate
      .connect(agentA)
      .registerAgentSponsored(
        digest("lineA"),
        ethers.toUtf8Bytes("peer-A"),
        ownerW.address,
        ZERO,
        sigA
      );
    // owner signs B with parent = ZERO, but B submits parent = agentA.
    const sig = await signSponsorship(substrate, ownerW, agentB.address, ZERO);
    await expect(
      substrate
        .connect(agentB)
        .registerAgentSponsored(
          digest("lineB"),
          ethers.toUtf8Bytes("peer-B"),
          ownerW.address,
          agentA.address,
          sig
        )
    ).to.be.revertedWithCustomError(substrate, "BadSponsorshipSignature");
  });

  it("owner = address(0) reverts (OwnerRequired)", async function () {
    const sig = await signSponsorship(substrate, ownerW, agentA.address, ZERO);
    await expect(
      substrate
        .connect(agentA)
        .registerAgentSponsored(
          digest("lineA"),
          ethers.toUtf8Bytes("peer-A"),
          ZERO,
          ZERO,
          sig
        )
    ).to.be.revertedWithCustomError(substrate, "OwnerRequired");
  });

  // ---- parent constraint paths -------------------------------------------

  it("parent not registered reverts", async function () {
    // agentC is never registered; B claims it as parent.
    const sig = await signSponsorship(
      substrate,
      ownerW,
      agentB.address,
      agentC.address
    );
    await expect(
      substrate
        .connect(agentB)
        .registerAgentSponsored(
          digest("lineB"),
          ethers.toUtf8Bytes("peer-B"),
          ownerW.address,
          agentC.address,
          sig
        )
    ).to.be.revertedWithCustomError(substrate, "ParentNotRegistered");
  });

  it("parent with a DIFFERENT owner reverts", async function () {
    // A belongs to ownerW.
    const sigA = await signSponsorship(substrate, ownerW, agentA.address, ZERO);
    await substrate
      .connect(agentA)
      .registerAgentSponsored(
        digest("lineA"),
        ethers.toUtf8Bytes("peer-A"),
        ownerW.address,
        ZERO,
        sigA
      );
    // B claims otherOwnerW as owner but parent A (owned by ownerW).
    const sigB = await signSponsorship(
      substrate,
      otherOwnerW,
      agentB.address,
      agentA.address
    );
    await expect(
      substrate
        .connect(agentB)
        .registerAgentSponsored(
          digest("lineB"),
          ethers.toUtf8Bytes("peer-B"),
          otherOwnerW.address,
          agentA.address,
          sigB
        )
    ).to.be.revertedWithCustomError(substrate, "ParentOwnerMismatch");
  });

  // ---- owner immutability -------------------------------------------------

  it("second attach reverts (owner immutable) — retrofit after sponsored", async function () {
    const sigA = await signSponsorship(substrate, ownerW, agentA.address, ZERO);
    await substrate
      .connect(agentA)
      .registerAgentSponsored(
        digest("lineA"),
        ethers.toUtf8Bytes("peer-A"),
        ownerW.address,
        ZERO,
        sigA
      );
    // Try to re-attribute to a different owner via retrofit.
    const sig2 = await signSponsorship(substrate, otherOwnerW, agentA.address, ZERO);
    await expect(
      substrate.connect(agentA).attachOwner(otherOwnerW.address, ZERO, sig2)
    ).to.be.revertedWithCustomError(substrate, "OwnerAlreadySet");
  });

  it("second attach reverts even re-attaching the SAME owner", async function () {
    await registerAgent(substrate, agentA, "lineA");
    const sig = await signSponsorship(substrate, ownerW, agentA.address, ZERO);
    await substrate.connect(agentA).attachOwner(ownerW.address, ZERO, sig);
    await expect(
      substrate.connect(agentA).attachOwner(ownerW.address, ZERO, sig)
    ).to.be.revertedWithCustomError(substrate, "OwnerAlreadySet");
  });

  it("retrofit on an unregistered agent reverts (AgentNotActive)", async function () {
    const sig = await signSponsorship(substrate, ownerW, stranger.address, ZERO);
    await expect(
      substrate.connect(stranger).attachOwner(ownerW.address, ZERO, sig)
    ).to.be.revertedWithCustomError(substrate, "AgentNotActive");
  });

  it("sponsored registration cannot re-register an already-registered agent", async function () {
    await registerAgent(substrate, agentA, "lineA");
    const sig = await signSponsorship(substrate, ownerW, agentA.address, ZERO);
    await expect(
      substrate
        .connect(agentA)
        .registerAgentSponsored(
          digest("lineA2"),
          ethers.toUtf8Bytes("peer-A2"),
          ownerW.address,
          ZERO,
          sig
        )
    ).to.be.revertedWithCustomError(substrate, "AlreadyRegistered");
  });

  // ---- sameOwner truth table ---------------------------------------------

  it("sameOwner truth table including unset cases", async function () {
    // A and B under ownerW; C under otherOwnerW; stranger unset.
    const sigA = await signSponsorship(substrate, ownerW, agentA.address, ZERO);
    await substrate
      .connect(agentA)
      .registerAgentSponsored(
        digest("lineA"),
        ethers.toUtf8Bytes("peer-A"),
        ownerW.address,
        ZERO,
        sigA
      );
    const sigB = await signSponsorship(substrate, ownerW, agentB.address, ZERO);
    await substrate
      .connect(agentB)
      .registerAgentSponsored(
        digest("lineB"),
        ethers.toUtf8Bytes("peer-B"),
        ownerW.address,
        ZERO,
        sigB
      );
    const sigC = await signSponsorship(substrate, otherOwnerW, agentC.address, ZERO);
    await substrate
      .connect(agentC)
      .registerAgentSponsored(
        digest("lineC"),
        ethers.toUtf8Bytes("peer-C"),
        otherOwnerW.address,
        ZERO,
        sigC
      );

    // same owner → true
    expect(await substrate.sameOwner(agentA.address, agentB.address)).to.equal(true);
    expect(await substrate.sameOwner(agentB.address, agentA.address)).to.equal(true);
    // self is same owner
    expect(await substrate.sameOwner(agentA.address, agentA.address)).to.equal(true);
    // different owners → false
    expect(await substrate.sameOwner(agentA.address, agentC.address)).to.equal(false);
    // one side unset → false (both directions)
    expect(await substrate.sameOwner(agentA.address, stranger.address)).to.equal(false);
    expect(await substrate.sameOwner(stranger.address, agentA.address)).to.equal(false);
    // both unset → false
    expect(await substrate.sameOwner(stranger.address, deployer.address)).to.equal(false);
  });

  // ---- back-compat --------------------------------------------------------

  it("legacy registerAgent still works and leaves owner/parent unset", async function () {
    await expect(registerAgent(substrate, agentA, "lineA")).to.emit(
      substrate,
      "AgentRegistered"
    );
    expect(await substrate.isRegistered(agentA.address)).to.equal(true);
    expect(await substrate.getAgentOwner(agentA.address)).to.equal(ZERO);
    expect(await substrate.getAgentParent(agentA.address)).to.equal(ZERO);
  });

  it("registerTool unaffected: sponsored agent can register a tool", async function () {
    const sig = await signSponsorship(substrate, ownerW, agentA.address, ZERO);
    await substrate
      .connect(agentA)
      .registerAgentSponsored(
        digest("lineA"),
        ethers.toUtf8Bytes("peer-A"),
        ownerW.address,
        ZERO,
        sig
      );
    const d = digest("manifest-1");
    await expect(substrate.connect(agentA).registerTool(d)).to.emit(
      substrate,
      "ToolRegistered"
    );
    expect(await substrate.toolAuthor(d)).to.equal(agentA.address);
  });
});

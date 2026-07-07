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

// Sign an EIP-712 OwnerBinding{agent, parent, nonce} for Substrate as `owner`.
// `nonce` defaults to the agent's current on-chain binding nonce.
async function signBinding(substrate, ownerSigner, agent, parent, nonce) {
  const net = await ethers.provider.getNetwork();
  const domain = {
    name: "AutonetSubstrate",
    version: "1",
    chainId: net.chainId,
    verifyingContract: await substrate.getAddress(),
  };
  const types = {
    OwnerBinding: [
      { name: "agent", type: "address" },
      { name: "parent", type: "address" },
      { name: "nonce", type: "uint256" },
    ],
  };
  if (nonce === undefined) {
    nonce = await substrate.bindingNonce(agent);
  }
  return ownerSigner.signTypedData(domain, types, { agent, parent, nonce });
}

describe("Substrate owner-rooted registration (OwnerBound)", function () {
  let substrate;
  let deployer, ownerW, otherOwnerW, agentA, agentB, agentC, stranger;
  const ZERO = ethers.ZeroAddress;

  beforeEach(async function () {
    [deployer, ownerW, otherOwnerW, agentA, agentB, agentC, stranger] =
      await ethers.getSigners();
    const Substrate = await ethers.getContractFactory("Substrate");
    substrate = await Substrate.deploy((await ethers.getSigners())[0].address);
    await substrate.waitForDeployment();
  });

  // ---- happy path: fresh bound registration ------------------------------

  it("fresh top-level bound registration binds owner, emits both events", async function () {
    const sig = await signBinding(substrate, ownerW, agentA.address, ZERO);
    await expect(
      substrate
        .connect(agentA)
        .registerAgentBound(
          digest("lineA"),
          ethers.toUtf8Bytes("peer-A"),
          ownerW.address,
          ZERO,
          sig
        )
    )
      .to.emit(substrate, "AgentRegistered")
      .and.to.emit(substrate, "OwnerBound")
      .withArgs(agentA.address, ownerW.address, ZERO, ZERO, 0);

    expect(await substrate.isRegistered(agentA.address)).to.equal(true);
    expect(await substrate.getAgentOwner(agentA.address)).to.equal(ownerW.address);
    expect(await substrate.getAgentParent(agentA.address)).to.equal(ZERO);
    // nonce advanced past the consumed 0.
    expect(await substrate.bindingNonce(agentA.address)).to.equal(1);
  });

  it("fresh bound registration with a parent under the same owner", async function () {
    // Parent A first (top-level).
    const sigA = await signBinding(substrate, ownerW, agentA.address, ZERO);
    await substrate
      .connect(agentA)
      .registerAgentBound(
        digest("lineA"),
        ethers.toUtf8Bytes("peer-A"),
        ownerW.address,
        ZERO,
        sigA
      );

    // Child B under A, same owner.
    const sigB = await signBinding(
      substrate,
      ownerW,
      agentB.address,
      agentA.address
    );
    await expect(
      substrate
        .connect(agentB)
        .registerAgentBound(
          digest("lineB"),
          ethers.toUtf8Bytes("peer-B"),
          ownerW.address,
          agentA.address,
          sigB
        )
    )
      .to.emit(substrate, "OwnerBound")
      .withArgs(agentB.address, ownerW.address, agentA.address, ZERO, 0);

    expect(await substrate.getAgentParent(agentB.address)).to.equal(agentA.address);
    expect(await substrate.getAgentOwner(agentB.address)).to.equal(ownerW.address);
  });

  // ---- happy path: retrofit on an existing ownerless agent ---------------

  it("retrofit: legacy agent attaches owner+parent once", async function () {
    // A registered the legacy way (ownerless).
    await registerAgent(substrate, agentA, "lineA");
    expect(await substrate.getAgentOwner(agentA.address)).to.equal(ZERO);

    const sig = await signBinding(substrate, ownerW, agentA.address, ZERO);
    await expect(
      substrate.connect(agentA).attachOwner(ownerW.address, ZERO, sig)
    )
      .to.emit(substrate, "OwnerBound")
      .withArgs(agentA.address, ownerW.address, ZERO, ZERO, 0);

    expect(await substrate.getAgentOwner(agentA.address)).to.equal(ownerW.address);
  });

  it("retrofit with a parent under the same owner", async function () {
    // A is bound top-level.
    const sigA = await signBinding(substrate, ownerW, agentA.address, ZERO);
    await substrate
      .connect(agentA)
      .registerAgentBound(
        digest("lineA"),
        ethers.toUtf8Bytes("peer-A"),
        ownerW.address,
        ZERO,
        sigA
      );
    // B registered legacy, retrofits under A.
    await registerAgent(substrate, agentB, "lineB");
    const sigB = await signBinding(
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
    const sig = await signBinding(substrate, otherOwnerW, agentA.address, ZERO);
    await expect(
      substrate
        .connect(agentA)
        .registerAgentBound(
          digest("lineA"),
          ethers.toUtf8Bytes("peer-A"),
          ownerW.address,
          ZERO,
          sig
        )
    ).to.be.revertedWithCustomError(substrate, "BadOwnerBindingSignature");
  });

  it("tampered agent address in sig (signed for a different agent) reverts", async function () {
    // owner signed for agentB, but agentA submits.
    const sig = await signBinding(substrate, ownerW, agentB.address, ZERO);
    await expect(
      substrate
        .connect(agentA)
        .registerAgentBound(
          digest("lineA"),
          ethers.toUtf8Bytes("peer-A"),
          ownerW.address,
          ZERO,
          sig
        )
    ).to.be.revertedWithCustomError(substrate, "BadOwnerBindingSignature");
  });

  it("wrong parent in sig vs arg reverts", async function () {
    // Register A top-level so it's a valid same-owner parent candidate.
    const sigA = await signBinding(substrate, ownerW, agentA.address, ZERO);
    await substrate
      .connect(agentA)
      .registerAgentBound(
        digest("lineA"),
        ethers.toUtf8Bytes("peer-A"),
        ownerW.address,
        ZERO,
        sigA
      );
    // owner signs B with parent = ZERO, but B submits parent = agentA.
    const sig = await signBinding(substrate, ownerW, agentB.address, ZERO);
    await expect(
      substrate
        .connect(agentB)
        .registerAgentBound(
          digest("lineB"),
          ethers.toUtf8Bytes("peer-B"),
          ownerW.address,
          agentA.address,
          sig
        )
    ).to.be.revertedWithCustomError(substrate, "BadOwnerBindingSignature");
  });

  it("wrong nonce in sig reverts (stale/future nonce)", async function () {
    // owner signs at nonce 1, but the agent's current nonce is 0.
    const sig = await signBinding(substrate, ownerW, agentA.address, ZERO, 1);
    await expect(
      substrate
        .connect(agentA)
        .registerAgentBound(
          digest("lineA"),
          ethers.toUtf8Bytes("peer-A"),
          ownerW.address,
          ZERO,
          sig
        )
    ).to.be.revertedWithCustomError(substrate, "BadOwnerBindingSignature");
  });

  it("owner = address(0) reverts (OwnerRequired)", async function () {
    const sig = await signBinding(substrate, ownerW, agentA.address, ZERO);
    await expect(
      substrate
        .connect(agentA)
        .registerAgentBound(
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
    const sig = await signBinding(
      substrate,
      ownerW,
      agentB.address,
      agentC.address
    );
    await expect(
      substrate
        .connect(agentB)
        .registerAgentBound(
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
    const sigA = await signBinding(substrate, ownerW, agentA.address, ZERO);
    await substrate
      .connect(agentA)
      .registerAgentBound(
        digest("lineA"),
        ethers.toUtf8Bytes("peer-A"),
        ownerW.address,
        ZERO,
        sigA
      );
    // B claims otherOwnerW as owner but parent A (owned by ownerW).
    const sigB = await signBinding(
      substrate,
      otherOwnerW,
      agentB.address,
      agentA.address
    );
    await expect(
      substrate
        .connect(agentB)
        .registerAgentBound(
          digest("lineB"),
          ethers.toUtf8Bytes("peer-B"),
          otherOwnerW.address,
          agentA.address,
          sigB
        )
    ).to.be.revertedWithCustomError(substrate, "ParentOwnerMismatch");
  });

  // ---- owner rotation (key-loss recovery) ---------------------------------

  it("rotateOwner: leaked-owner recovery — rotate A→ownerB WITHOUT ownerA participating", async function () {
    // A is bound to ownerW (the wallet whose key later leaks).
    const sigA = await signBinding(substrate, ownerW, agentA.address, ZERO);
    await substrate
      .connect(agentA)
      .registerAgentBound(
        digest("lineA"),
        ethers.toUtf8Bytes("peer-A"),
        ownerW.address,
        ZERO,
        sigA
      );
    expect(await substrate.getAgentOwner(agentA.address)).to.equal(ownerW.address);
    expect(await substrate.bindingNonce(agentA.address)).to.equal(1);

    // Recovery: the daemon host (holds agent A's key) rotates to a fresh
    // wallet otherOwnerW. ONLY the NEW owner signs; ownerW never participates.
    const sigRot = await signBinding(substrate, otherOwnerW, agentA.address, ZERO);
    await expect(
      substrate.connect(agentA).rotateOwner(otherOwnerW.address, ZERO, sigRot)
    )
      .to.emit(substrate, "OwnerBound")
      // prevOwner is the leaked ownerW; nonce consumed is 1.
      .withArgs(agentA.address, otherOwnerW.address, ZERO, ownerW.address, 1);

    expect(await substrate.getAgentOwner(agentA.address)).to.equal(
      otherOwnerW.address
    );
    expect(await substrate.bindingNonce(agentA.address)).to.equal(2);
  });

  it("CRITICAL replay: a captured OLD binding sig cannot rotate back after rotation", async function () {
    // Bind A → ownerW. Capture ownerW's signature (the "old" leaked-owner sig).
    const oldSig = await signBinding(substrate, ownerW, agentA.address, ZERO);
    await substrate
      .connect(agentA)
      .registerAgentBound(
        digest("lineA"),
        ethers.toUtf8Bytes("peer-A"),
        ownerW.address,
        ZERO,
        oldSig
      );

    // Recover away from the leaked ownerW to otherOwnerW.
    const sigRot = await signBinding(substrate, otherOwnerW, agentA.address, ZERO);
    await substrate
      .connect(agentA)
      .rotateOwner(otherOwnerW.address, ZERO, sigRot);
    expect(await substrate.getAgentOwner(agentA.address)).to.equal(
      otherOwnerW.address
    );

    // The thief replays the captured oldSig (signed at nonce 0) to rotate the
    // fleet BACK to the leaked ownerW. Nonce is now 2, so it recovers to a
    // different (non-owner) signer and is rejected.
    await expect(
      substrate.connect(agentA).rotateOwner(ownerW.address, ZERO, oldSig)
    ).to.be.revertedWithCustomError(substrate, "BadOwnerBindingSignature");

    // Ownership unchanged: recovery holds.
    expect(await substrate.getAgentOwner(agentA.address)).to.equal(
      otherOwnerW.address
    );
  });

  it("rotateOwner first-attach works on a legacy ownerless agent", async function () {
    await registerAgent(substrate, agentA, "lineA");
    expect(await substrate.getAgentOwner(agentA.address)).to.equal(ZERO);
    const sig = await signBinding(substrate, ownerW, agentA.address, ZERO);
    await expect(
      substrate.connect(agentA).rotateOwner(ownerW.address, ZERO, sig)
    )
      .to.emit(substrate, "OwnerBound")
      .withArgs(agentA.address, ownerW.address, ZERO, ZERO, 0);
    expect(await substrate.getAgentOwner(agentA.address)).to.equal(ownerW.address);
  });

  it("rotateOwner on an unregistered agent reverts (AgentNotActive)", async function () {
    const sig = await signBinding(substrate, ownerW, stranger.address, ZERO);
    await expect(
      substrate.connect(stranger).rotateOwner(ownerW.address, ZERO, sig)
    ).to.be.revertedWithCustomError(substrate, "AgentNotActive");
  });

  it("rotation parent rules: new parent must be owned by the NEW owner", async function () {
    // A bound to ownerW top-level; B bound to ownerW under A.
    const sigA = await signBinding(substrate, ownerW, agentA.address, ZERO);
    await substrate
      .connect(agentA)
      .registerAgentBound(
        digest("lineA"),
        ethers.toUtf8Bytes("peer-A"),
        ownerW.address,
        ZERO,
        sigA
      );
    const sigB = await signBinding(substrate, ownerW, agentB.address, agentA.address);
    await substrate
      .connect(agentB)
      .registerAgentBound(
        digest("lineB"),
        ethers.toUtf8Bytes("peer-B"),
        ownerW.address,
        agentA.address,
        sigB
      );

    // Migrate the fleet to otherOwnerW. Rotate the PARENT first.
    const sigRotA = await signBinding(substrate, otherOwnerW, agentA.address, ZERO);
    await substrate
      .connect(agentA)
      .rotateOwner(otherOwnerW.address, ZERO, sigRotA);

    // Attempt to rotate B to otherOwnerW while keeping parent A BEFORE A is
    // under the new owner would fail; but A is now under otherOwnerW, so
    // rotating B under A succeeds. First prove the failure mode: rotating B
    // under a parent still owned by the OLD owner.
    // (Here A is already migrated, so use a fresh mismatch: parent = agentA is
    // fine now. Demonstrate mismatch via a parent owned by nobody-new.)
    // Rotate B under the now-migrated A — happy path.
    const sigRotB = await signBinding(
      substrate,
      otherOwnerW,
      agentB.address,
      agentA.address
    );
    await substrate
      .connect(agentB)
      .rotateOwner(otherOwnerW.address, agentA.address, sigRotB);
    expect(await substrate.getAgentOwner(agentB.address)).to.equal(
      otherOwnerW.address
    );
    expect(await substrate.getAgentParent(agentB.address)).to.equal(agentA.address);
  });

  it("rotation parent rules: parent owned by a different owner reverts", async function () {
    // A bound to ownerW; C bound to otherOwnerW. Try to rotate C's child under
    // A (owned by ownerW) while claiming otherOwnerW as owner → mismatch.
    const sigA = await signBinding(substrate, ownerW, agentA.address, ZERO);
    await substrate
      .connect(agentA)
      .registerAgentBound(
        digest("lineA"),
        ethers.toUtf8Bytes("peer-A"),
        ownerW.address,
        ZERO,
        sigA
      );
    // B is a fresh legacy agent we rotate; claim otherOwnerW as owner but
    // parent A (owned by ownerW) → ParentOwnerMismatch.
    await registerAgent(substrate, agentB, "lineB");
    const sigB = await signBinding(
      substrate,
      otherOwnerW,
      agentB.address,
      agentA.address
    );
    await expect(
      substrate.connect(agentB).rotateOwner(otherOwnerW.address, agentA.address, sigB)
    ).to.be.revertedWithCustomError(substrate, "ParentOwnerMismatch");
  });

  it("children keep their own bindings when a parent rotates (per-agent rotation)", async function () {
    // A (parent) and B (child under A), both under ownerW.
    const sigA = await signBinding(substrate, ownerW, agentA.address, ZERO);
    await substrate
      .connect(agentA)
      .registerAgentBound(
        digest("lineA"),
        ethers.toUtf8Bytes("peer-A"),
        ownerW.address,
        ZERO,
        sigA
      );
    const sigB = await signBinding(substrate, ownerW, agentB.address, agentA.address);
    await substrate
      .connect(agentB)
      .registerAgentBound(
        digest("lineB"),
        ethers.toUtf8Bytes("peer-B"),
        ownerW.address,
        agentA.address,
        sigB
      );

    // Rotate only A to otherOwnerW. B is untouched: its owner/parent stand.
    const sigRotA = await signBinding(substrate, otherOwnerW, agentA.address, ZERO);
    await substrate
      .connect(agentA)
      .rotateOwner(otherOwnerW.address, ZERO, sigRotA);

    expect(await substrate.getAgentOwner(agentA.address)).to.equal(
      otherOwnerW.address
    );
    // B still points at ownerW and parent A — a fleet migrates agent by agent.
    expect(await substrate.getAgentOwner(agentB.address)).to.equal(ownerW.address);
    expect(await substrate.getAgentParent(agentB.address)).to.equal(agentA.address);
  });

  // ---- nonce view ---------------------------------------------------------

  it("bindingNonce view starts at 0 and increments on each bind/rotate", async function () {
    expect(await substrate.bindingNonce(agentA.address)).to.equal(0);
    const sig0 = await signBinding(substrate, ownerW, agentA.address, ZERO);
    await substrate
      .connect(agentA)
      .registerAgentBound(
        digest("lineA"),
        ethers.toUtf8Bytes("peer-A"),
        ownerW.address,
        ZERO,
        sig0
      );
    expect(await substrate.bindingNonce(agentA.address)).to.equal(1);

    const sig1 = await signBinding(substrate, otherOwnerW, agentA.address, ZERO);
    await substrate.connect(agentA).rotateOwner(otherOwnerW.address, ZERO, sig1);
    expect(await substrate.bindingNonce(agentA.address)).to.equal(2);

    // sameOwner reflects the rotation immediately.
    // Bind B to otherOwnerW; A (rotated) and B now share an owner.
    const sigB = await signBinding(substrate, otherOwnerW, agentB.address, ZERO);
    await substrate
      .connect(agentB)
      .registerAgentBound(
        digest("lineB"),
        ethers.toUtf8Bytes("peer-B"),
        otherOwnerW.address,
        ZERO,
        sigB
      );
    expect(await substrate.sameOwner(agentA.address, agentB.address)).to.equal(true);
  });

  it("retrofit on an unregistered agent reverts (AgentNotActive)", async function () {
    const sig = await signBinding(substrate, ownerW, stranger.address, ZERO);
    await expect(
      substrate.connect(stranger).attachOwner(ownerW.address, ZERO, sig)
    ).to.be.revertedWithCustomError(substrate, "AgentNotActive");
  });

  it("bound registration cannot re-register an already-registered agent", async function () {
    await registerAgent(substrate, agentA, "lineA");
    const sig = await signBinding(substrate, ownerW, agentA.address, ZERO);
    await expect(
      substrate
        .connect(agentA)
        .registerAgentBound(
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
    const sigA = await signBinding(substrate, ownerW, agentA.address, ZERO);
    await substrate
      .connect(agentA)
      .registerAgentBound(
        digest("lineA"),
        ethers.toUtf8Bytes("peer-A"),
        ownerW.address,
        ZERO,
        sigA
      );
    const sigB = await signBinding(substrate, ownerW, agentB.address, ZERO);
    await substrate
      .connect(agentB)
      .registerAgentBound(
        digest("lineB"),
        ethers.toUtf8Bytes("peer-B"),
        ownerW.address,
        ZERO,
        sigB
      );
    const sigC = await signBinding(substrate, otherOwnerW, agentC.address, ZERO);
    await substrate
      .connect(agentC)
      .registerAgentBound(
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
    // no binding consumed
    expect(await substrate.bindingNonce(agentA.address)).to.equal(0);
  });

  it("registerTool unaffected: bound agent can register a tool", async function () {
    const sig = await signBinding(substrate, ownerW, agentA.address, ZERO);
    await substrate
      .connect(agentA)
      .registerAgentBound(
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

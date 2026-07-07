const { expect } = require("chai");
const { ethers } = require("hardhat");
const { anyUint } = require("@nomicfoundation/hardhat-chai-matchers/withArgs");

// Helpers ---------------------------------------------------------------------
const PEER_ID = ethers.toUtf8Bytes("peer-1234567890");
const digest = (s) => ethers.keccak256(ethers.toUtf8Bytes(s));

async function registerAgent(substrate, signer, lineage) {
  return substrate
    .connect(signer)
    .registerAgent(digest(lineage), ethers.toUtf8Bytes("peer-" + lineage));
}

describe("Substrate.registerTool (ToolRegistered)", function () {
  let substrate, owner, agentA, agentB, stranger;

  beforeEach(async function () {
    [owner, agentA, agentB, stranger] = await ethers.getSigners();
    const Substrate = await ethers.getContractFactory("Substrate");
    substrate = await Substrate.deploy((await ethers.getSigners())[0].address);
    await substrate.waitForDeployment();
    await registerAgent(substrate, agentA, "lineA");
    await registerAgent(substrate, agentB, "lineB");
  });

  it("registered agent can register a tool; emits ToolRegistered", async function () {
    const d = digest("manifest-1");
    await expect(substrate.connect(agentA).registerTool(d))
      .to.emit(substrate, "ToolRegistered")
      .withArgs(agentA.address, d, anyUint);
    expect(await substrate.toolAuthor(d)).to.equal(agentA.address);
    expect(await substrate.isToolRegistered(d)).to.equal(true);
    expect(await substrate.toolCountByAgent(agentA.address)).to.equal(1n);
  });

  it("non-agent caller reverts with AgentNotActive", async function () {
    await expect(
      substrate.connect(stranger).registerTool(digest("m"))
    ).to.be.revertedWithCustomError(substrate, "AgentNotActive");
  });

  it("zero digest reverts", async function () {
    await expect(
      substrate.connect(agentA).registerTool(ethers.ZeroHash)
    ).to.be.revertedWithCustomError(substrate, "ManifestDigestRequired");
  });

  it("duplicate digest reverts (even from the same author)", async function () {
    const d = digest("dup");
    await substrate.connect(agentA).registerTool(d);
    await expect(
      substrate.connect(agentA).registerTool(d)
    ).to.be.revertedWithCustomError(substrate, "ToolAlreadyRegistered");
  });

  it("a second agent cannot steal an already-registered digest", async function () {
    const d = digest("someone-elses-blob");
    await substrate.connect(agentA).registerTool(d);
    await expect(substrate.connect(agentB).registerTool(d))
      .to.be.revertedWithCustomError(substrate, "ToolAlreadyRegistered")
      .withArgs(d, agentA.address);
  });

  it("distinct digests from the same agent both register", async function () {
    await substrate.connect(agentA).registerTool(digest("m1"));
    await substrate.connect(agentA).registerTool(digest("m2"));
    expect(await substrate.toolCountByAgent(agentA.address)).to.equal(2n);
  });
});

const { expect } = require("chai");
const { ethers } = require("hardhat");
const { ZERO, deploySubstrate, registerAgent, signBinding } = require("./helpers");

// A rotation re-parents one agent and leaves its descendants where they
// were. The fleet tree must stay a tree, and revoking an ancestor must reach
// every descendant however the tree was rearranged.
describe("Substrate fleet tree after re-parenting", function () {
  let substrate, owner, agents;

  beforeEach(async function () {
    const signers = await ethers.getSigners();
    owner = signers[0];
    agents = signers.slice(1, 20);   // 19 agent keys, one owner wallet
    substrate = await deploySubstrate();
  });

  // Register `keys` as a chain: keys[0] top-level, each next under the last.
  async function chain(keys, tag) {
    for (let i = 0; i < keys.length; i++) {
      await registerAgent(substrate, keys[i], `${tag}${i}`, owner,
        i === 0 ? ZERO : keys[i - 1].address);
    }
  }

  async function reparent(agent, parent) {
    const sig = await signBinding(substrate, owner, agent.address, parent);
    return substrate.connect(agent).rotateOwner(owner.address, parent, sig);
  }

  it("an agent cannot bind under its own descendant", async function () {
    const [a, b, c] = agents;
    await chain([a, b, c], "x");
    await expect(reparent(a, c.address)).to.be.revertedWithCustomError(substrate, "FleetCycle");
    await expect(reparent(a, b.address)).to.be.revertedWithCustomError(substrate, "FleetCycle");
    await expect(reparent(a, a.address)).to.be.revertedWithCustomError(substrate, "FleetCycle");
    // Moving sideways is fine.
    await reparent(c, a.address);
    expect(await substrate.getAgentParent(c.address)).to.equal(a.address);
  });

  it("depth is counted by walking the tree, not read from a stale record", async function () {
    const c = agents.slice(0, 9);    // c0 .. c8, depths 0 .. 8
    const d = agents.slice(9, 19);   // d0 .. d9, depths 0 .. 9
    await chain(c, "c");
    await chain(d, "d");
    // c0 moves under d9: c0 is now at depth 10, its descendants at 11 .. 18,
    // though their recorded depths still say 1 .. 8.
    await reparent(c[0], d[9].address);
    expect(await substrate.agentDepth(c[8].address)).to.equal(8n);

    // Within the limit, descendants stay live.
    expect(await substrate.isLive(c[6].address)).to.equal(true);    // true depth 16
    // Past it, they are not live: a revoked ancestor could sit beyond the walk.
    expect(await substrate.isLive(c[7].address)).to.equal(false);   // true depth 17
    expect(await substrate.isLive(c[8].address)).to.equal(false);

    // A new child cannot be added below the real limit either, whatever the
    // parent's recorded depth says.
    const sig = await signBinding(substrate, owner, d[9].address, c[6].address);
    await expect(substrate.connect(d[9]).rotateOwner(owner.address, c[6].address, sig))
      .to.be.revertedWithCustomError(substrate, "FleetCycle");

    // Revoking the far ancestor reaches every descendant that was live.
    await substrate.connect(owner).revokeAgent(d[0].address, "0x");
    for (const k of [d[5], d[9], c[0], c[3], c[6]]) {
      expect(await substrate.isLive(k.address)).to.equal(false);
    }
  });

  it("moving the subtree back up brings its deep descendants back", async function () {
    const c = agents.slice(0, 9);
    const d = agents.slice(9, 19);
    await chain(c, "c");
    await chain(d, "d");
    await reparent(c[0], d[9].address);
    expect(await substrate.isLive(c[8].address)).to.equal(false);
    await reparent(c[0], ZERO);
    expect(await substrate.isLive(c[8].address)).to.equal(true);
  });
});

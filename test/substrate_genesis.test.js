const { expect } = require("chai");
const { ethers } = require("hardhat");
const {
  digest,
  deploySubstrateWith,
  mintLeaf,
  seedPool,
  anchorEpoch,
} = require("./helpers");

// Sorted-pair merkle tree over claimFor-style leaves (the
// nodes/common/mint_merkle.py convention: odd nodes promote unchanged).
function hashPair(a, b) {
  return a.toLowerCase() <= b.toLowerCase()
    ? ethers.keccak256(ethers.concat([a, b]))
    : ethers.keccak256(ethers.concat([b, a]));
}

function buildTree(entries) {
  const sorted = [...entries].sort((x, y) =>
    x[0].toLowerCase() < y[0].toLowerCase() ? -1 : 1);
  const levels = [sorted.map(([a, m]) => mintLeaf(a, m))];
  while (levels[levels.length - 1].length > 1) {
    const prev = levels[levels.length - 1];
    const next = [];
    for (let i = 0; i + 1 < prev.length; i += 2) next.push(hashPair(prev[i], prev[i + 1]));
    if (prev.length % 2 === 1) next.push(prev[prev.length - 1]);
    levels.push(next);
  }
  const root = levels[levels.length - 1][0];
  const proofs = {};
  sorted.forEach(([addr], i) => {
    const proof = [];
    let idx = i;
    for (const level of levels.slice(0, -1)) {
      const sib = idx % 2 === 0 ? idx + 1 : idx - 1;
      if (sib < level.length) proof.push(level[sib]);
      idx = Math.floor(idx / 2);
    }
    proofs[addr] = proof;
  });
  return { root, proofs };
}

describe("Substrate.claimGenesis (fork/split carry-over)", function () {
  let a, b, c, stranger;
  let tree, amounts;

  beforeEach(async function () {
    const signers = await ethers.getSigners();
    [, a, b, c, stranger] = signers;
    amounts = { [a.address]: 5_000_000n, [b.address]: 7n, [c.address]: 123_456_789n };
    tree = buildTree(Object.entries(amounts));
  });

  it("zero root: every claim reverts (fresh network)", async function () {
    const s = await deploySubstrateWith();
    expect(await s.genesisRoot()).to.equal(ethers.ZeroHash);
    await expect(s.connect(a).claimGenesis(amounts[a.address], tree.proofs[a.address]))
      .to.be.revertedWithCustomError(s, "NoGenesis");
    await expect(s.connect(a).claimGenesis(0, []))
      .to.be.revertedWithCustomError(s, "NoGenesis");
  });

  it("valid proof mints once to the caller, outside the earnings ledger", async function () {
    const s = await deploySubstrateWith({ genesisRoot: tree.root });
    const amt = amounts[c.address];
    await expect(s.connect(c).claimGenesis(amt, tree.proofs[c.address]))
      .to.emit(s, "GenesisClaimed").withArgs(c.address, amt)
      .and.to.emit(s, "ATNTransfer").withArgs(ethers.ZeroAddress, c.address, amt);
    expect(await s.balanceOf(c.address)).to.equal(amt);
    expect(await s.genesisClaimed(c.address)).to.equal(true);
    expect(await s.genesisMinted()).to.equal(amt);
    expect(await s.atnTotalSupply()).to.equal(amt);
    expect(await s.agentMintTotal(c.address)).to.equal(0n);
    expect(await s.networkMintTotal()).to.equal(0n);

    await s.connect(a).claimGenesis(amounts[a.address], tree.proofs[a.address]);
    expect(await s.genesisMinted()).to.equal(amt + amounts[a.address]);
  });

  it("double claim reverts", async function () {
    const s = await deploySubstrateWith({ genesisRoot: tree.root });
    await s.connect(b).claimGenesis(amounts[b.address], tree.proofs[b.address]);
    await expect(s.connect(b).claimGenesis(amounts[b.address], tree.proofs[b.address]))
      .to.be.revertedWithCustomError(s, "GenesisAlreadyClaimed").withArgs(b.address);
  });

  it("bad proof reverts: wrong amount, wrong caller, wrong path", async function () {
    const s = await deploySubstrateWith({ genesisRoot: tree.root });
    await expect(s.connect(a).claimGenesis(amounts[a.address] + 1n, tree.proofs[a.address]))
      .to.be.revertedWithCustomError(s, "GenesisProofInvalid");
    await expect(s.connect(stranger).claimGenesis(amounts[a.address], tree.proofs[a.address]))
      .to.be.revertedWithCustomError(s, "GenesisProofInvalid");
    await expect(s.connect(a).claimGenesis(amounts[a.address], tree.proofs[b.address]))
      .to.be.revertedWithCustomError(s, "GenesisProofInvalid");
    expect(await s.genesisClaimed(a.address)).to.equal(false);
    expect(await s.genesisMinted()).to.equal(0n);
  });

  it("epoch mint cap and agentMintTotal are unaffected by genesis", async function () {
    const s = await deploySubstrateWith({ genesisRoot: tree.root });
    const pool = await seedPool(s);
    await s.connect(c).claimGenesis(amounts[c.address], tree.proofs[c.address]);
    expect(await s.nextMintPool()).to.equal(pool);

    // The full pool is still claimable by an epoch leaf after genesis.
    await anchorEpoch(s, "e1", mintLeaf(a.address, pool));
    await s.connect(a).claimFor(a.address, pool, digest("e1"), []);
    expect(await s.agentMintTotal(a.address)).to.equal(pool);
    expect(await s.networkMintTotal()).to.equal(pool);
    expect(await s.agentMintTotal(c.address)).to.equal(0n);

    // Genesis on top of earned ATN does not touch the ledger either.
    await s.connect(a).claimGenesis(amounts[a.address], tree.proofs[a.address]);
    expect(await s.agentMintTotal(a.address)).to.equal(pool);
    expect(await s.networkMintTotal()).to.equal(pool);
    expect(await s.genesisMinted()).to.equal(amounts[a.address] + amounts[c.address]);
  });
});

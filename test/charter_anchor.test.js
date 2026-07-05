const { expect } = require("chai");
const { ethers } = require("hardhat");

// CharterAnchor: the deliberately-governed contrast to Substrate.sol. Anchors
// which 6-root charter version is canonical; governor-only, append-only chain.

const HASH_A = ethers.keccak256(ethers.toUtf8Bytes("charter-v1"));
const HASH_B = ethers.keccak256(ethers.toUtf8Bytes("charter-v2"));
const HASH_C = ethers.keccak256(ethers.toUtf8Bytes("charter-v3"));
const ZERO = ethers.ZeroHash;

async function deployAnchor(governorSigner) {
  const CharterAnchor = await ethers.getContractFactory("CharterAnchor");
  const a = await CharterAnchor.deploy(await governorSigner.getAddress());
  await a.waitForDeployment();
  return a;
}

describe("CharterAnchor", function () {
  let governor, other;

  beforeEach(async function () {
    [governor, other] = await ethers.getSigners();
  });

  it("rejects a zero governor at deploy", async function () {
    const CharterAnchor = await ethers.getContractFactory("CharterAnchor");
    await expect(CharterAnchor.deploy(ethers.ZeroAddress)).to.be.revertedWith(
      "governor=0"
    );
  });

  it("names the governor and starts with no versions", async function () {
    const a = await deployAnchor(governor);
    expect(await a.governor()).to.equal(await governor.getAddress());
    expect(await a.versionCount()).to.equal(0n);
    await expect(a.currentCharter()).to.be.revertedWithCustomError(
      a,
      "NoCharter"
    );
  });

  describe("governor-only", function () {
    it("lets the governor anchor", async function () {
      const a = await deployAnchor(governor);
      await expect(a.connect(governor).anchorCharter(HASH_A, "ipfs://v1"))
        .to.emit(a, "CharterAnchored")
        .withArgs(1n, HASH_A, "ipfs://v1", ZERO);
    });

    it("reverts when a non-governor tries to anchor", async function () {
      const a = await deployAnchor(governor);
      await expect(
        a.connect(other).anchorCharter(HASH_A, "ipfs://v1")
      ).to.be.revertedWithCustomError(a, "NotGovernor");
    });

    it("reverts on a zero charter hash", async function () {
      const a = await deployAnchor(governor);
      await expect(
        a.connect(governor).anchorCharter(ZERO, "ipfs://v1")
      ).to.be.revertedWithCustomError(a, "EmptyHash");
    });
  });

  describe("append-only chaining", function () {
    it("version 1 chains to bytes32(0)", async function () {
      const a = await deployAnchor(governor);
      await a.connect(governor).anchorCharter(HASH_A, "ipfs://v1");
      const [version, hash, uri, prevHash] = await a.currentCharter();
      expect(version).to.equal(1n);
      expect(hash).to.equal(HASH_A);
      expect(uri).to.equal("ipfs://v1");
      expect(prevHash).to.equal(ZERO);
    });

    it("version n+1 chains to version n's hash", async function () {
      const a = await deployAnchor(governor);
      await a.connect(governor).anchorCharter(HASH_A, "ipfs://v1");
      await expect(a.connect(governor).anchorCharter(HASH_B, "ipfs://v2"))
        .to.emit(a, "CharterAnchored")
        .withArgs(2n, HASH_B, "ipfs://v2", HASH_A);
      await expect(a.connect(governor).anchorCharter(HASH_C, "ipfs://v3"))
        .to.emit(a, "CharterAnchored")
        .withArgs(3n, HASH_C, "ipfs://v3", HASH_B);
      expect(await a.versionCount()).to.equal(3n);
    });

    it("counter is monotonic; old versions stay anchored (forward-only)", async function () {
      const a = await deployAnchor(governor);
      await a.connect(governor).anchorCharter(HASH_A, "ipfs://v1");
      await a.connect(governor).anchorCharter(HASH_B, "ipfs://v2");
      // version 1 is unchanged after v2 lands
      const [hash1, uri1, prev1] = await a.charterAt(1);
      expect(hash1).to.equal(HASH_A);
      expect(uri1).to.equal("ipfs://v1");
      expect(prev1).to.equal(ZERO);
    });
  });

  describe("view correctness", function () {
    it("charterAt returns each version", async function () {
      const a = await deployAnchor(governor);
      await a.connect(governor).anchorCharter(HASH_A, "ipfs://v1");
      await a.connect(governor).anchorCharter(HASH_B, "ipfs://v2");

      const [hA, uA, pA, tA] = await a.charterAt(1);
      expect(hA).to.equal(HASH_A);
      expect(uA).to.equal("ipfs://v1");
      expect(pA).to.equal(ZERO);
      expect(tA).to.be.gt(0n);

      const [hB, uB, pB] = await a.charterAt(2);
      expect(hB).to.equal(HASH_B);
      expect(uB).to.equal("ipfs://v2");
      expect(pB).to.equal(HASH_A);
    });

    it("charterAt reverts on version 0 or beyond the count", async function () {
      const a = await deployAnchor(governor);
      await a.connect(governor).anchorCharter(HASH_A, "ipfs://v1");
      await expect(a.charterAt(0)).to.be.revertedWithCustomError(
        a,
        "BadVersion"
      );
      await expect(a.charterAt(2)).to.be.revertedWithCustomError(
        a,
        "BadVersion"
      );
    });

    it("currentCharter tracks the latest version", async function () {
      const a = await deployAnchor(governor);
      await a.connect(governor).anchorCharter(HASH_A, "ipfs://v1");
      await a.connect(governor).anchorCharter(HASH_B, "ipfs://v2");
      const [version, hash] = await a.currentCharter();
      expect(version).to.equal(2n);
      expect(hash).to.equal(HASH_B);
    });
  });
});

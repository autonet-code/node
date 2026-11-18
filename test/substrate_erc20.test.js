const { expect } = require("chai");
const { ethers } = require("hardhat");
const { deploySubstrateWith, mintATN, mintEarned, digest } = require("./helpers");

// ATN is an ERC-20 as wallets, explorers and DEX indexers read one: the
// standard views, and Transfer events that account for every balance change
// (mints from address(0), the burned fee share to address(0)).
describe("Substrate as an ERC-20", function () {
  it("has the standard interface", async function () {
    const s = await deploySubstrateWith();
    const iface = s.interface;
    for (const sig of [
      "name()", "symbol()", "decimals()", "totalSupply()", "balanceOf(address)",
      "transfer(address,uint256)", "allowance(address,address)",
      "approve(address,uint256)", "transferFrom(address,address,uint256)",
    ]) {
      expect(iface.getFunction(sig), sig).to.not.equal(null);
    }
    expect(iface.getEvent("Transfer").topicHash)
      .to.equal(ethers.id("Transfer(address,address,uint256)"));
    expect(iface.getEvent("Approval").topicHash)
      .to.equal(ethers.id("Approval(address,address,uint256)"));
    expect(await s.decimals()).to.equal(6n);
  });

  it("Transfer events alone rebuild every balance and the total supply", async function () {
    const [treasury, a, b, provider, spender] = await ethers.getSigners();
    const s = await deploySubstrateWith({ treasury: treasury.address });

    await mintATN(s, null, a, 50_000_000n);                       // vault mint
    await mintEarned(s, b, 3_000_000n);                           // earned mint (seeds a pool via payForService)
    await s.connect(a).transfer(b.address, 1_000_000n);
    await s.connect(a).approve(spender.address, 2_000_000n);
    await expect(s.connect(spender).transferFrom(a.address, provider.address, 2_000_000n))
      .to.emit(s, "Transfer").withArgs(a.address, provider.address, 2_000_000n);
    const pay = s.connect(a).payForService(provider.address, 10_000_000n, digest("req"));
    const fee = (10_000_000n * 250n) / 10_000n;
    const burned = fee - (fee * 5000n) / 10_000n;
    await expect(pay).to.emit(s, "Transfer").withArgs(a.address, ethers.ZeroAddress, burned);

    const logs = await s.queryFilter(s.filters.Transfer(), 0);
    const bal = new Map();
    const add = (k, v) => bal.set(k, (bal.get(k) ?? 0n) + v);
    for (const l of logs) {
      const { from, to, value } = l.args;
      if (from !== ethers.ZeroAddress) add(from, -value);
      if (to !== ethers.ZeroAddress) add(to, value);
    }
    let sum = 0n;
    for (const [who, v] of bal) {
      expect(await s.balanceOf(who), who).to.equal(v);
      sum += v;
    }
    expect(await s.totalSupply()).to.equal(sum);
    expect(await s.totalSupply()).to.equal(await s.atnTotalSupply());
  });
});

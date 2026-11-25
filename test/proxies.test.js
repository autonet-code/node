const { expect } = require("chai");
const { ethers } = require("hardhat");
const {
  ZERO,
  digest,
  deploySubstrateWith,
  registerAgent,
  signVoucher,
  deployIndexToken,
  fund,
  payFees,
  anchorEpoch,
  claimTree,
  claimTx,
  SLOT_GOVERNOR,
} = require("./helpers");

// How a jurisdiction gets its contracts: Substrate as an ERC-1967 proxy the
// governor can upgrade, the other three as EIP-1167 clones. Each is set up
// by initialize() in the transaction that creates it, once.

const ONE = "0x0000000000000000000000000000000000000001";

async function cloneOf(factoryName, implementation, initArgs) {
  const F = await ethers.getContractFactory(factoryName);
  const Cloner = await ethers.getContractFactory("TestCloner");
  const cloner = await Cloner.deploy();
  await cloner.waitForDeployment();
  const data = F.interface.encodeFunctionData("initialize", initArgs);
  const tx = await cloner.cloneAndCall(await implementation.getAddress(), data);
  const rcpt = await tx.wait();
  const ev = rcpt.logs.map((l) => {
    try { return cloner.interface.parseLog(l); } catch (e) { return null; }
  }).find((e) => e && e.name === "Cloned");
  return F.attach(ev.args.clone);
}

describe("Substrate behind a proxy", function () {
  let substrate, gov, deployer, alice, faucet, stranger;
  const SubstrateArgs = (governor) => [ONE, governor, ZERO, ONE, 250, 5000];

  beforeEach(async function () {
    [deployer, alice, faucet, stranger] = await ethers.getSigners();
    gov = (await ethers.getSigners())[SLOT_GOVERNOR];
    substrate = await deploySubstrateWith();
  });

  it("is set up by initialize, at the proxy's own address", async function () {
    expect(await substrate.governor()).to.equal(gov.address);
    expect(await substrate.treasury()).to.equal(deployer.address);
    expect(await substrate.registry()).to.not.equal(ZERO);
    expect(await substrate.serviceFeeBps()).to.equal(250);
    expect(await substrate.deployedAt()).to.be.gt(0);
    expect(await substrate.latestCutoff()).to.equal(await substrate.deployedAt());
    // The EIP-712 domain names the proxy, not the implementation.
    const d = await substrate.eip712Domain();
    expect(d.verifyingContract).to.equal(await substrate.getAddress());
    expect(d.name).to.equal("AutonetSubstrate");
  });

  it("cannot be initialized twice, and neither can the implementation", async function () {
    await expect(substrate.initialize(...SubstrateArgs(stranger.address)))
      .to.be.revertedWithCustomError(substrate, "InvalidInitialization");
    const F = await ethers.getContractFactory("Substrate");
    const impl = await F.deploy(...SubstrateArgs(ONE));
    await impl.waitForDeployment();
    await expect(impl.initialize(...SubstrateArgs(stranger.address)))
      .to.be.revertedWithCustomError(impl, "InvalidInitialization");
    expect(await impl.governor()).to.equal(ONE);
  });

  it("a proxy refuses a zero governor like the constructor does", async function () {
    const F = await ethers.getContractFactory("Substrate");
    const impl = await F.deploy(...SubstrateArgs(ONE));
    await impl.waitForDeployment();
    const Proxy = await ethers.getContractFactory("SubstrateProxy");
    await expect(Proxy.deploy(
      await impl.getAddress(),
      F.interface.encodeFunctionData("initialize", SubstrateArgs(ZERO)),
    )).to.be.revertedWithCustomError(F, "NoAnchorAuthority");
  });

  it("the jurisdiction setup's path: an empty proxy first, initialized later in the same call", async function () {
    // The implementation as the suite deploys it: inert arguments.
    const F = await ethers.getContractFactory("Substrate");
    const DEAD = "0x000000000000000000000000000000000000dEaD";
    const impl = await F.deploy(ZERO, DEAD, ZERO, DEAD, 50, 0);
    await impl.waitForDeployment();
    const Proxy = await ethers.getContractFactory("SubstrateProxy");
    const proxy = await Proxy.deploy(await impl.getAddress(), "0x");
    await proxy.waitForDeployment();
    const s = F.attach(await proxy.getAddress());
    // Nothing is set until initialize runs.
    expect(await s.governor()).to.equal(ZERO);
    await s.initialize(...SubstrateArgs(gov.address));
    expect(await s.governor()).to.equal(gov.address);
    expect(await s.registry()).to.equal(ONE);
    await expect(s.initialize(...SubstrateArgs(stranger.address)))
      .to.be.revertedWithCustomError(s, "InvalidInitialization");
    await registerAgent(s, alice, "alice");
  });

  it("only the governor upgrades; pools, claims, agents and settings survive", async function () {
    await registerAgent(substrate, alice, "alice");
    const usd = await deployIndexToken(substrate);
    const U = await usd.getAddress();
    await payFees(substrate, usd, 1_000_000n);      // 12_500 tool share, 12_500 treasury
    await anchorEpoch(substrate, "e1");
    const tree = claimTree([{ earner: alice.address, tokens: [U], amounts: [777n], value: 777n }]);
    await anchorEpoch(substrate, "e2", tree.root);
    await claimTx(substrate, alice, tree, 0, "e2");
    await substrate.connect(gov).setServiceFeeBps(300);
    const deployedAt = await substrate.deployedAt();

    const V2 = await ethers.getContractFactory("SubstrateV2Mock");
    const v2 = await V2.deploy();
    await v2.waitForDeployment();
    const v2Addr = await v2.getAddress();

    await expect(substrate.connect(stranger).upgradeToAndCall(v2Addr, "0x"))
      .to.be.revertedWithCustomError(substrate, "NotGovernor");
    await expect(substrate.connect(alice).upgradeToAndCall(v2Addr, "0x"))
      .to.be.revertedWithCustomError(substrate, "NotGovernor");

    await expect(substrate.connect(gov).upgradeToAndCall(v2Addr, "0x"))
      .to.emit(substrate, "Upgraded").withArgs(v2Addr);

    const upgraded = V2.attach(await substrate.getAddress());
    expect(await upgraded.version2()).to.equal(2n);
    expect(await upgraded.owed(alice.address, U)).to.equal(777n);
    expect(await upgraded.earnedValue(alice.address)).to.equal(777n);
    expect(await upgraded.treasuryOwed(U)).to.equal(12_500n);
    expect(await upgraded.toolShareTotal(U)).to.equal(12_500n);
    expect(await upgraded.held(U)).to.equal(25_000n);
    expect(await upgraded.anchorCount()).to.equal(2n);
    expect(await upgraded.isRegistered(alice.address)).to.equal(true);
    expect(await upgraded.governor()).to.equal(gov.address);
    expect(await upgraded.serviceFeeBps()).to.equal(300);
    expect(await upgraded.deployedAt()).to.equal(deployedAt);
    expect(await upgraded.tokenDecimals(U)).to.equal(6);
    // Still set up: the new code cannot be initialized over the old state.
    await expect(upgraded.initialize(...SubstrateArgs(stranger.address)))
      .to.be.revertedWithCustomError(upgraded, "InvalidInitialization");
    // And what is owed still pays out.
    await upgraded.withdraw(alice.address, U);
    expect(await usd.balanceOf(alice.address)).to.equal(777n);
  });

  it("refuses an upgrade to a contract that is not a Substrate implementation", async function () {
    const Cloner = await ethers.getContractFactory("TestCloner");
    const notImpl = await Cloner.deploy();
    await notImpl.waitForDeployment();
    await expect(substrate.connect(gov).upgradeToAndCall(await notImpl.getAddress(), "0x"))
      .to.be.reverted;
    expect(await substrate.governor()).to.equal(gov.address);
  });

  it("a directly deployed Substrate works and cannot be upgraded", async function () {
    const direct = await deploySubstrateWith({ direct: true });
    expect(await direct.governor()).to.equal(gov.address);
    await registerAgent(direct, alice, "alice");
    const V2 = await ethers.getContractFactory("SubstrateV2Mock");
    const v2 = await V2.deploy();
    await v2.waitForDeployment();
    await expect(direct.connect(gov).upgradeToAndCall(await v2.getAddress(), "0x"))
      .to.be.revertedWithCustomError(direct, "UUPSUnauthorizedCallContext");
    await expect(direct.initialize(...SubstrateArgs(stranger.address)))
      .to.be.revertedWithCustomError(direct, "InvalidInitialization");
  });
});

describe("AnchorElection, ServiceRegistry and PaymentChannel as clones", function () {
  let substrate, deployer, client, provider, faucet, stranger;

  beforeEach(async function () {
    [deployer, client, provider, faucet, stranger] = await ethers.getSigners();
    substrate = await deploySubstrateWith();
    await registerAgent(substrate, provider, "prov");
    await registerAgent(substrate, faucet, "faucet");
  });

  it("AnchorElection: a clone gets its addresses and the default parameters, once", async function () {
    const F = await ethers.getContractFactory("AnchorElection");
    const impl = await F.deploy(ONE, ONE, ONE);
    await impl.waitForDeployment();
    const sub = await substrate.getAddress();
    const clone = await cloneOf("AnchorElection", impl, [sub, client.address, provider.address]);
    expect(await clone.substrate()).to.equal(sub);
    expect(await clone.governor()).to.equal(client.address);
    expect(await clone.repToken()).to.equal(provider.address);
    expect(await clone.proposers()).to.equal(3n);
    expect(await clone.proposerFloorPpm()).to.equal(0n);
    expect(await clone.maxOptions()).to.equal(8n);
    expect(await clone.openAfterAttempts()).to.equal(16n);
    await expect(clone.initialize(sub, stranger.address, stranger.address))
      .to.be.revertedWithCustomError(clone, "InvalidInitialization");
    await expect(impl.initialize(sub, stranger.address, stranger.address))
      .to.be.revertedWithCustomError(impl, "InvalidInitialization");
    // A directly deployed election has the same defaults.
    expect(await impl.proposers()).to.equal(3n);
    expect(await impl.maxOptions()).to.equal(8n);
  });

  it("ServiceRegistry: a clone registers services against its Substrate, and is set up once", async function () {
    const F = await ethers.getContractFactory("ServiceRegistry");
    const impl = await F.deploy(ONE);
    await impl.waitForDeployment();
    const sub = await substrate.getAddress();
    const clone = await cloneOf("ServiceRegistry", impl, [sub]);
    expect(await clone.substrate()).to.equal(sub);
    await expect(clone.connect(provider).registerService(digest("spec"), 5n))
      .to.emit(clone, "ServiceRegistered");
    expect(await clone.serviceCount()).to.equal(1n);
    await expect(clone.initialize(stranger.address))
      .to.be.revertedWithCustomError(clone, "InvalidInitialization");
    await expect(impl.initialize(sub))
      .to.be.revertedWithCustomError(impl, "InvalidInitialization");
    // The implementation's own state is untouched by the clone's.
    expect(await impl.serviceCount()).to.equal(0n);
  });

  it("ServiceRegistry and PaymentChannel clones refuse a zero Substrate", async function () {
    const R = await ethers.getContractFactory("ServiceRegistry");
    const rImpl = await R.deploy(ONE);
    await rImpl.waitForDeployment();
    await expect(cloneOf("ServiceRegistry", rImpl, [ZERO])).to.be.revertedWith("substrate=0");
    const C = await ethers.getContractFactory("PaymentChannel");
    const cImpl = await C.deploy(ONE, 3600);
    await cImpl.waitForDeployment();
    await expect(cloneOf("PaymentChannel", cImpl, [ZERO, 3600])).to.be.revertedWith("substrate=0");
    await expect(cloneOf("PaymentChannel", cImpl, [await substrate.getAddress(), 0]))
      .to.be.revertedWith("window=0");
  });

  it("PaymentChannel: a clone escrows, settles a voucher signed for its own address, and is set up once", async function () {
    const F = await ethers.getContractFactory("PaymentChannel");
    const impl = await F.deploy(ONE, 3600);
    await impl.waitForDeployment();
    const sub = await substrate.getAddress();
    const channel = await cloneOf("PaymentChannel", impl, [sub, 1800]);
    const chAddr = await channel.getAddress();
    expect(await channel.substrate()).to.equal(sub);
    expect(await channel.challengeWindow()).to.equal(1800n);
    expect((await channel.eip712Domain()).verifyingContract).to.equal(chAddr);

    const usd = await deployIndexToken(substrate);
    const U = await usd.getAddress();
    await fund(usd, client, chAddr, 10_000n);
    await channel.connect(client).openChannel(provider.address, U, 5000n);
    const id = await channel.channelCount();
    expect(await usd.balanceOf(chAddr)).to.equal(5000n);

    // A voucher signed for the implementation's address is not this clone's.
    const wrong = await signVoucher(impl, client, id, 3000n, U);
    await expect(channel.connect(provider).settle(id, 3000n, wrong, 1000n))
      .to.be.revertedWithCustomError(channel, "BadVoucherSignature");

    const sig = await signVoucher(channel, client, id, 3000n);
    const before = await usd.balanceOf(provider.address);
    await channel.connect(provider).settle(id, 3000n, sig, 1000n);
    // 2.5% fee: the provider receives 975 of the 1000 served, Substrate 25.
    expect((await usd.balanceOf(provider.address)) - before).to.equal(975n);
    expect(await usd.balanceOf(sub)).to.equal(25n);
    // The reentrancy guard keeps working call after call.
    await channel.connect(provider).settle(id, 3000n, sig, 2000n);
    await channel.connect(client).topUp(id, 100n);

    await expect(channel.initialize(sub, 60))
      .to.be.revertedWithCustomError(channel, "InvalidInitialization");
    await expect(impl.initialize(sub, 60))
      .to.be.revertedWithCustomError(impl, "InvalidInitialization");
  });

  it("an uninitialized clone can be set up by whoever calls first, so setup must share its transaction", async function () {
    // What the jurisdiction setup must never do: clone without initializing.
    const F = await ethers.getContractFactory("ServiceRegistry");
    const impl = await F.deploy(ONE);
    await impl.waitForDeployment();
    const Cloner = await ethers.getContractFactory("TestCloner");
    const cloner = await Cloner.deploy();
    await cloner.waitForDeployment();
    const rcpt = await (await cloner.cloneOnly(await impl.getAddress())).wait();
    const ev = rcpt.logs.map((l) => cloner.interface.parseLog(l)).find((e) => e.name === "Cloned");
    const bare = F.attach(ev.args.clone);
    await bare.connect(stranger).initialize(stranger.address);
    expect(await bare.substrate()).to.equal(stranger.address);
  });
});

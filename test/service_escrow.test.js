const { expect } = require("chai");
const { ethers } = require("hardhat");
const { time } = require("@nomicfoundation/hardhat-network-helpers");
const {
  digest,
  deploySubstrate,
  registerAgent,
  deployToken,
} = require("./helpers");

const RELEASE_TIMEOUT = 3600; // client can reclaim undelivered after 1h
const CLAIM_TIMEOUT = 7200; //   provider can claim delivered after 2h from delivery
const REQ = ethers.id("request-1");

describe("ServiceEscrow (v1 escrow-per-request)", function () {
  let substrate, registry, escrow, token;
  let provider, client, other;
  let serviceId;

  beforeEach(async function () {
    [, provider, client, other] = await ethers.getSigners();
    substrate = await deploySubstrate();
    token = await deployToken();
    await registerAgent(substrate, provider, "prov");

    const Reg = await ethers.getContractFactory("ServiceRegistry");
    registry = await Reg.deploy(await substrate.getAddress());
    await registry.waitForDeployment();

    const Escrow = await ethers.getContractFactory("ServiceEscrow");
    escrow = await Escrow.deploy(
      await registry.getAddress(),
      RELEASE_TIMEOUT,
      CLAIM_TIMEOUT
    );
    await escrow.waitForDeployment();

    await registry
      .connect(provider)
      .registerService(digest("spec"), await token.getAddress(), 1000n);
    serviceId = 1n;

    await token.mint(client.address, 10_000n);
    await token.connect(client).approve(await escrow.getAddress(), 10_000n);
  });

  async function open(amount = 1000n, requestId = REQ) {
    return escrow.connect(client).openRequest(serviceId, requestId, amount);
  }

  it("happy path: open -> markDelivered -> release pays provider", async function () {
    await expect(open())
      .to.emit(escrow, "RequestOpened");
    expect(await token.balanceOf(await escrow.getAddress())).to.equal(1000n);

    await escrow.connect(provider).markDelivered(client.address, serviceId, REQ);
    await expect(escrow.connect(client).release(serviceId, REQ))
      .to.emit(escrow, "RequestReleased");
    expect(await token.balanceOf(provider.address)).to.equal(1000n);
    expect(await token.balanceOf(await escrow.getAddress())).to.equal(0n);
  });

  it("client can release early (before delivery)", async function () {
    await open();
    await escrow.connect(client).release(serviceId, REQ);
    expect(await token.balanceOf(provider.address)).to.equal(1000n);
  });

  it("double open of same key reverts", async function () {
    await open();
    await expect(open()).to.be.revertedWithCustomError(escrow, "RequestExists");
  });

  it("zero amount reverts", async function () {
    await expect(open(0n)).to.be.revertedWithCustomError(escrow, "ZeroAmount");
  });

  it("open against inactive/nonexistent service reverts", async function () {
    await expect(
      escrow.connect(client).openRequest(99n, REQ, 1000n)
    ).to.be.revertedWithCustomError(escrow, "NoSuchService");
    await registry.connect(provider).retireService(serviceId);
    await expect(open()).to.be.revertedWithCustomError(escrow, "ServiceInactive");
  });

  // --- Griefing path A: provider never delivers -> client reclaims ---------
  it("client reclaims an undelivered request after releaseTimeout", async function () {
    await open();
    await expect(
      escrow.connect(client).reclaim(serviceId, REQ)
    ).to.be.revertedWithCustomError(escrow, "TooEarly");

    await time.increase(RELEASE_TIMEOUT + 1);
    await expect(escrow.connect(client).reclaim(serviceId, REQ))
      .to.emit(escrow, "RequestReclaimed");
    expect(await token.balanceOf(client.address)).to.equal(10_000n);
    expect(await token.balanceOf(provider.address)).to.equal(0n);
  });

  it("reclaim blocked once provider marked delivered", async function () {
    await open();
    await escrow.connect(provider).markDelivered(client.address, serviceId, REQ);
    await time.increase(RELEASE_TIMEOUT + 1);
    await expect(
      escrow.connect(client).reclaim(serviceId, REQ)
    ).to.be.revertedWithCustomError(escrow, "RequestNotOpen");
  });

  it("only the client can reclaim", async function () {
    await open();
    await time.increase(RELEASE_TIMEOUT + 1);
    await expect(
      escrow.connect(other).reclaim(serviceId, REQ)
    ).to.be.revertedWithCustomError(escrow, "RequestNotOpen"); // other's key doesn't exist
  });

  // --- Griefing path B: client never releases -> provider claims -----------
  it("provider claims delivered-but-unreleased after claimTimeout", async function () {
    await open();
    await escrow.connect(provider).markDelivered(client.address, serviceId, REQ);
    await expect(
      escrow.connect(provider).claimDelivered(client.address, serviceId, REQ)
    ).to.be.revertedWithCustomError(escrow, "TooEarly");

    await time.increase(CLAIM_TIMEOUT + 1);
    await expect(
      escrow.connect(provider).claimDelivered(client.address, serviceId, REQ)
    ).to.emit(escrow, "RequestClaimed");
    expect(await token.balanceOf(provider.address)).to.equal(1000n);
  });

  it("claimDelivered before delivery reverts (not delivered)", async function () {
    await open();
    await time.increase(CLAIM_TIMEOUT + 1);
    await expect(
      escrow.connect(provider).claimDelivered(client.address, serviceId, REQ)
    ).to.be.revertedWithCustomError(escrow, "RequestNotDelivered");
  });

  it("non-provider cannot mark delivered or claim", async function () {
    await open();
    await expect(
      escrow.connect(other).markDelivered(client.address, serviceId, REQ)
    ).to.be.revertedWithCustomError(escrow, "NotProvider");
    await escrow.connect(provider).markDelivered(client.address, serviceId, REQ);
    await time.increase(CLAIM_TIMEOUT + 1);
    await expect(
      escrow.connect(other).claimDelivered(client.address, serviceId, REQ)
    ).to.be.revertedWithCustomError(escrow, "NotProvider");
  });

  it("cannot release twice (terminal state)", async function () {
    await open();
    await escrow.connect(client).release(serviceId, REQ);
    await expect(
      escrow.connect(client).release(serviceId, REQ)
    ).to.be.revertedWithCustomError(escrow, "RequestNotOpen");
  });

  it("SafeERC20 catches a false-returning token on release", async function () {
    const Bad = await ethers.getContractFactory("FalseReturnERC20");
    const bad = await Bad.deploy();
    await bad.waitForDeployment();
    await registry
      .connect(provider)
      .registerService(digest("badspec"), await bad.getAddress(), 500n);
    const sid = 2n;
    await bad.mint(client.address, 1000n);
    await bad.connect(client).approve(await escrow.getAddress(), 1000n);
    // openRequest's transferFrom returns false -> SafeERC20 reverts
    await expect(
      escrow.connect(client).openRequest(sid, REQ, 500n)
    ).to.be.revertedWithCustomError(escrow, "SafeERC20FailedOperation");
  });
});

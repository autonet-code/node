const { expect } = require("chai");
const { ethers } = require("hardhat");
const { time } = require("@nomicfoundation/hardhat-network-helpers");
const { anyValue } = require("@nomicfoundation/hardhat-chai-matchers/withArgs");
const { deployToken, signVoucher } = require("./helpers");

const CHALLENGE_WINDOW = 3600;

describe("PaymentChannel (v1.5 prepaid credits)", function () {
  let channel, token, client, provider, other;

  beforeEach(async function () {
    [, client, provider, other] = await ethers.getSigners();
    token = await deployToken();
    const Ch = await ethers.getContractFactory("PaymentChannel");
    channel = await Ch.deploy(CHALLENGE_WINDOW);
    await channel.waitForDeployment();
    await token.mint(client.address, 10_000n);
    await token.connect(client).approve(await channel.getAddress(), 10_000n);
  });

  async function openChannel(deposit = 5000n) {
    await channel
      .connect(client)
      .openChannel(provider.address, await token.getAddress(), deposit);
    return 1n;
  }

  it("open escrows the deposit and emits", async function () {
    await expect(
      channel
        .connect(client)
        .openChannel(provider.address, await token.getAddress(), 5000n)
    )
      .to.emit(channel, "ChannelOpened")
      .withArgs(1n, client.address, provider.address, await token.getAddress(), 5000n);
    expect(await token.balanceOf(await channel.getAddress())).to.equal(5000n);
  });

  it("rejects zero provider / token / deposit", async function () {
    await expect(
      channel.connect(client).openChannel(ethers.ZeroAddress, await token.getAddress(), 1n)
    ).to.be.revertedWithCustomError(channel, "ProviderRequired");
    await expect(
      channel.connect(client).openChannel(provider.address, ethers.ZeroAddress, 1n)
    ).to.be.revertedWithCustomError(channel, "TokenRequired");
    await expect(
      channel.connect(client).openChannel(provider.address, await token.getAddress(), 0n)
    ).to.be.revertedWithCustomError(channel, "ZeroDeposit");
  });

  it("happy path: provider closes with voucher, remainder refunds after window", async function () {
    const id = await openChannel(5000n);
    const cumulative = 3000n;
    const sig = await signVoucher(channel, client, id, cumulative);

    await expect(channel.connect(provider).closeChannel(id, cumulative, sig))
      .to.emit(channel, "ChannelClosed")
      .withArgs(id, 3000n, 2000n, anyValue);
    expect(await token.balanceOf(provider.address)).to.equal(3000n);

    // remainder locked during the window
    await expect(
      channel.withdrawRemainder(id)
    ).to.be.revertedWithCustomError(channel, "WindowNotElapsed");

    await time.increase(CHALLENGE_WINDOW + 1);
    await expect(channel.withdrawRemainder(id))
      .to.emit(channel, "RemainderWithdrawn")
      .withArgs(id, client.address, 2000n);
    expect(await token.balanceOf(client.address)).to.equal(10_000n - 3000n);
    expect(await token.balanceOf(await channel.getAddress())).to.equal(0n);
  });

  it("only the provider can close", async function () {
    const id = await openChannel();
    const sig = await signVoucher(channel, client, id, 1000n);
    await expect(
      channel.connect(other).closeChannel(id, 1000n, sig)
    ).to.be.revertedWithCustomError(channel, "NotProvider");
  });

  it("voucher signed by a non-client is rejected", async function () {
    const id = await openChannel();
    const forged = await signVoucher(channel, other, id, 1000n);
    await expect(
      channel.connect(provider).closeChannel(id, 1000n, forged)
    ).to.be.revertedWithCustomError(channel, "BadVoucherSignature");
  });

  it("voucher for a different amount doesn't verify (tamper)", async function () {
    const id = await openChannel();
    const sig = await signVoucher(channel, client, id, 1000n);
    // provider tries to claim more than the signed cumulative
    await expect(
      channel.connect(provider).closeChannel(id, 2000n, sig)
    ).to.be.revertedWithCustomError(channel, "BadVoucherSignature");
  });

  it("replay: cannot close the same channel twice", async function () {
    const id = await openChannel();
    const sig = await signVoucher(channel, client, id, 1000n);
    await channel.connect(provider).closeChannel(id, 1000n, sig);
    await expect(
      channel.connect(provider).closeChannel(id, 1000n, sig)
    ).to.be.revertedWithCustomError(channel, "ChannelNotOpen");
  });

  it("stale (lower) voucher only shortchanges the provider; higher voucher after close is rejected", async function () {
    const id = await openChannel(5000n);
    // provider foolishly closes with an older, lower cumulative
    const staleSig = await signVoucher(channel, client, id, 1000n);
    await channel.connect(provider).closeChannel(id, 1000n, staleSig);
    expect(await token.balanceOf(provider.address)).to.equal(1000n);

    // it cannot then submit the newer higher voucher — channel is Closing
    const newerSig = await signVoucher(channel, client, id, 4000n);
    await expect(
      channel.connect(provider).closeChannel(id, 4000n, newerSig)
    ).to.be.revertedWithCustomError(channel, "ChannelNotOpen");

    // client's remainder reflects the (larger) unclaimed portion
    await time.increase(CHALLENGE_WINDOW + 1);
    await channel.withdrawRemainder(id);
    expect(await token.balanceOf(client.address)).to.equal(10_000n - 1000n);
  });

  it("over-claim beyond deposit is capped at the deposit", async function () {
    const id = await openChannel(2000n);
    const sig = await signVoucher(channel, client, id, 9999n);
    await channel.connect(provider).closeChannel(id, 9999n, sig);
    expect(await token.balanceOf(provider.address)).to.equal(2000n); // capped
    await time.increase(CHALLENGE_WINDOW + 1);
    await channel.withdrawRemainder(id);
    // remainder is zero; client got nothing back beyond deposit
    expect(await token.balanceOf(await channel.getAddress())).to.equal(0n);
  });

  it("cannot withdraw remainder before close, or twice", async function () {
    const id = await openChannel();
    await expect(
      channel.withdrawRemainder(id)
    ).to.be.revertedWithCustomError(channel, "ChannelNotClosing");
    const sig = await signVoucher(channel, client, id, 1000n);
    await channel.connect(provider).closeChannel(id, 1000n, sig);
    await time.increase(CHALLENGE_WINDOW + 1);
    await channel.withdrawRemainder(id);
    await expect(
      channel.withdrawRemainder(id)
    ).to.be.revertedWithCustomError(channel, "ChannelNotClosing");
  });

  it("zero-cumulative voucher pays nothing, refunds full deposit", async function () {
    const id = await openChannel(5000n);
    const sig = await signVoucher(channel, client, id, 0n);
    await channel.connect(provider).closeChannel(id, 0n, sig);
    expect(await token.balanceOf(provider.address)).to.equal(0n);
    await time.increase(CHALLENGE_WINDOW + 1);
    await channel.withdrawRemainder(id);
    expect(await token.balanceOf(client.address)).to.equal(10_000n);
  });
});

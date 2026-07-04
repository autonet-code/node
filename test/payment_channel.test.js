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

  // ---------------------------------------------------------------------------
  // Channel-only settlement: coverage the deleted escrow used to hold.
  // The channel is now the ONLY settlement rail, so the per-item economics
  // that escrow encoded (atomic pay-per-item, bounded loss) must be provable
  // here: one voucher per item, cumulative increments, and a theft ceiling of
  // exactly one item's increment.
  // ---------------------------------------------------------------------------

  it("one voucher covers many sequential items via cumulative increments", async function () {
    // A single deposit funds a multi-item relationship. The client signs a
    // fresh voucher AFTER each served item, each raising the cumulative by
    // that item's ask (item price = 800). The provider only ever needs the
    // latest voucher to collect the served total — two on-chain txs for N
    // items.
    const id = await openChannel(5000n);
    const ITEM = 800n;
    const items = 5; // 5 items served, cumulative 4000 < 5000 deposit
    let latestSig;
    let cumulative = 0n;
    for (let i = 0; i < items; i++) {
      // provider serves item i, THEN the client hands over the next voucher
      cumulative += ITEM;
      latestSig = await signVoucher(channel, client, id, cumulative);
    }
    expect(cumulative).to.equal(4000n);

    // provider closes with only the highest (latest) voucher.
    await channel.connect(provider).closeChannel(id, cumulative, latestSig);
    expect(await token.balanceOf(provider.address)).to.equal(4000n);

    await time.increase(CHALLENGE_WINDOW + 1);
    await channel.withdrawRemainder(id);
    // client paid exactly the served total; the unserved 1000 refunds.
    expect(await token.balanceOf(client.address)).to.equal(10_000n - 4000n);
    expect(await token.balanceOf(await channel.getAddress())).to.equal(0n);
  });

  it("client stops paying mid-relationship: provider closes with last voucher, gets exactly the served total", async function () {
    // Provider serves 3 items (cumulative 3000), then the client ghosts —
    // stops accepting items and stops signing new vouchers. The provider is
    // NOT stuck: it closes with the last voucher it holds and collects the
    // full served total, no more, no less. The unserved remainder refunds to
    // the (now-absent) client after the window — permissionless trigger.
    const id = await openChannel(5000n);
    const ITEM = 1000n;
    const served = 3;
    let lastSig;
    let cumulative = 0n;
    for (let i = 0; i < served; i++) {
      cumulative += ITEM;
      lastSig = await signVoucher(channel, client, id, cumulative);
    }
    // client vanishes here — no further vouchers exist.

    await channel.connect(provider).closeChannel(id, cumulative, lastSig);
    // provider got EXACTLY the served total, nothing withheld from honest work.
    expect(await token.balanceOf(provider.address)).to.equal(3000n);

    // remainder refunds even though the client is gone (timer, not an action).
    await time.increase(CHALLENGE_WINDOW + 1);
    await channel.connect(other).withdrawRemainder(id); // anyone can trigger
    expect(await token.balanceOf(client.address)).to.equal(10_000n - 3000n);
    expect(await token.balanceOf(await channel.getAddress())).to.equal(0n);
  });

  it("theft ceiling: a provider who takes a voucher and never serves steals at most one item's increment", async function () {
    // The client's exposure to a cheating provider is bounded by ONE voucher,
    // sized by the client. Model an item price the client chose to be tiny
    // (ITEM = 5). The client signs the voucher for item N+1 optimistically,
    // the provider pockets it and never serves. The most the provider can
    // steal is that single voucher's INCREMENT over the last honestly-served
    // cumulative — not the whole deposit.
    const id = await openChannel(5000n);
    const ITEM = 5n; // client sizes its per-item exposure deliberately small

    // 10 items served honestly: cumulative 50.
    let cumulative = 0n;
    for (let i = 0; i < 10; i++) {
      cumulative += ITEM;
      await signVoucher(channel, client, id, cumulative); // served + paid
    }
    const honestCumulative = cumulative; // 50

    // Client optimistically signs ONE more voucher for the next item; the
    // provider takes it and never delivers.
    cumulative += ITEM; // 55
    const stolenSig = await signVoucher(channel, client, id, cumulative);

    await channel.connect(provider).closeChannel(id, cumulative, stolenSig);
    await time.increase(CHALLENGE_WINDOW + 1);
    await channel.withdrawRemainder(id);

    // The provider collected 55; only ITEM (=5) of that was for the unserved
    // item. The client's LOSS beyond honestly-served value is exactly one
    // item's increment — bounded, not the deposit.
    const providerGot = await token.balanceOf(provider.address);
    expect(providerGot).to.equal(55n);
    const theft = providerGot - honestCumulative;
    expect(theft).to.equal(ITEM); // loss ceiling = one voucher increment
    // Deposit was 5000; the client kept 5000 - 55 = 4945.
    expect(await token.balanceOf(client.address)).to.equal(10_000n - 55n);
  });
});

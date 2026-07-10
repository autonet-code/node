const { expect } = require("chai");
const { ethers } = require("hardhat");
const { time } = require("@nomicfoundation/hardhat-network-helpers");
const { anyValue } = require("@nomicfoundation/hardhat-chai-matchers/withArgs");
const {
  deploySubstrate,
  registerAgent,
  signVoucher,
  fundATN,
} = require("./helpers");

const CHALLENGE_WINDOW = 3600;

// Substrate's service fee (fee-recycled emission): 2.5% of the gross payment,
// half to the treasury, half burned. Vouchers are GROSS-denominated; the
// provider receives net. These mirror Substrate's serviceFeeBps (default 2.5%).
const feeOf = (gross) => (gross * 250n) / 10000n;
const toTreasury = (gross) => (feeOf(gross) * 5000n) / 10000n;
const burned = (gross) => feeOf(gross) - toTreasury(gross);
const netOf = (gross) => gross - feeOf(gross);

describe("PaymentChannel (v1.5 prepaid credits, ATN-only + fee at settlement)", function () {
  let substrate, channel;
  let deployer, client, provider, other, faucet;
  let treasury;

  beforeEach(async function () {
    [deployer, client, provider, other, faucet] = await ethers.getSigners();
    substrate = await deploySubstrate();
    // deploySubstrate wires the treasury to signers[0] (the deployer).
    treasury = await substrate.treasury();
    expect(treasury).to.equal(deployer.address);

    await registerAgent(substrate, provider, "prov1");
    await registerAgent(substrate, faucet, "faucet");

    const Ch = await ethers.getContractFactory("PaymentChannel");
    channel = await Ch.deploy(await substrate.getAddress(), CHALLENGE_WINDOW);
    await channel.waitForDeployment();

    // Client holds 10_000 ATN and approves the channel to pull deposits.
    await fundATN(substrate, faucet, faucet, client.address, 10_000n);
    await substrate.connect(client).approve(await channel.getAddress(), 10_000n);
  });

  async function openChannel(deposit = 5000n) {
    await channel.connect(client).openChannel(provider.address, deposit);
    return await channel.channelCount();
  }

  it("open escrows the deposit and emits", async function () {
    await expect(channel.connect(client).openChannel(provider.address, 5000n))
      .to.emit(channel, "ChannelOpened")
      .withArgs(1n, client.address, provider.address, 5000n);
    expect(await substrate.balanceOf(await channel.getAddress())).to.equal(5000n);
  });

  it("rejects zero provider / deposit", async function () {
    await expect(
      channel.connect(client).openChannel(ethers.ZeroAddress, 1n)
    ).to.be.revertedWithCustomError(channel, "ProviderRequired");
    await expect(
      channel.connect(client).openChannel(provider.address, 0n)
    ).to.be.revertedWithCustomError(channel, "ZeroDeposit");
  });

  it("constructor rejects zero substrate", async function () {
    const Ch = await ethers.getContractFactory("PaymentChannel");
    await expect(Ch.deploy(ethers.ZeroAddress, CHALLENGE_WINDOW)).to.be.reverted;
  });

  it("happy path: provider closes with voucher (net of fee), remainder refunds after window", async function () {
    const id = await openChannel(5000n);
    const cumulative = 3000n;
    const sig = await signVoucher(channel, client, id, cumulative);

    const treasBefore = await substrate.balanceOf(treasury);
    const supplyBefore = await substrate.atnTotalSupply();

    // closeChannel routes the payout through payForService: provider gets net,
    // treasury gets half the fee, the burned half emits ServiceFee.
    await expect(channel.connect(provider).closeChannel(id, cumulative, sig))
      .to.emit(channel, "ChannelClosed")
      .withArgs(id, 3000n, 2000n, anyValue)
      .and.to.emit(substrate, "ServiceFee")
      .withArgs(
        await channel.getAddress(),
        provider.address,
        cumulative,
        burned(cumulative),
        toTreasury(cumulative)
      );

    // Provider receives GROSS minus the 2.5% service fee.
    expect(await substrate.balanceOf(provider.address)).to.equal(netOf(cumulative));
    // Treasury got its half of the fee.
    expect(await substrate.balanceOf(treasury)).to.equal(treasBefore + toTreasury(cumulative));
    // Supply dropped by the burned half.
    expect(await substrate.atnTotalSupply()).to.equal(supplyBefore - burned(cumulative));

    // remainder locked during the window
    await expect(
      channel.withdrawRemainder(id)
    ).to.be.revertedWithCustomError(channel, "WindowNotElapsed");

    await time.increase(CHALLENGE_WINDOW + 1);
    await expect(channel.withdrawRemainder(id))
      .to.emit(channel, "RemainderWithdrawn")
      .withArgs(id, client.address, 2000n);
    // Refund is fee-free: client's balance ends at 10_000 - gross-claimed.
    expect(await substrate.balanceOf(client.address)).to.equal(10_000n - 3000n);
    expect(await substrate.balanceOf(await channel.getAddress())).to.equal(0n);
  });

  it("remainder refund is fee-free (no ServiceFee on withdraw)", async function () {
    const id = await openChannel(5000n);
    const sig = await signVoucher(channel, client, id, 1000n);
    await channel.connect(provider).closeChannel(id, 1000n, sig);

    const supplyBefore = await substrate.atnTotalSupply();
    const treasBefore = await substrate.balanceOf(treasury);
    await time.increase(CHALLENGE_WINDOW + 1);
    await expect(channel.withdrawRemainder(id)).to.not.emit(substrate, "ServiceFee");
    // No burn, no treasury take on the refund.
    expect(await substrate.atnTotalSupply()).to.equal(supplyBefore);
    expect(await substrate.balanceOf(treasury)).to.equal(treasBefore);
    // Client got the full unclaimed remainder back (deposit - gross claimed).
    expect(await substrate.balanceOf(client.address)).to.equal(10_000n - 1000n);
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
    expect(await substrate.balanceOf(provider.address)).to.equal(netOf(1000n));

    // it cannot then submit the newer higher voucher — channel is Closing
    const newerSig = await signVoucher(channel, client, id, 4000n);
    await expect(
      channel.connect(provider).closeChannel(id, 4000n, newerSig)
    ).to.be.revertedWithCustomError(channel, "ChannelNotOpen");

    // client's remainder reflects the (larger) unclaimed GROSS portion
    await time.increase(CHALLENGE_WINDOW + 1);
    await channel.withdrawRemainder(id);
    expect(await substrate.balanceOf(client.address)).to.equal(10_000n - 1000n);
  });

  it("over-claim beyond deposit is capped at the deposit", async function () {
    const id = await openChannel(2000n);
    const sig = await signVoucher(channel, client, id, 9999n);
    await channel.connect(provider).closeChannel(id, 9999n, sig);
    // capped at deposit (gross), provider receives net of that
    expect(await substrate.balanceOf(provider.address)).to.equal(netOf(2000n));
    await time.increase(CHALLENGE_WINDOW + 1);
    await channel.withdrawRemainder(id);
    // remainder is zero; the fee's burn/treasury split accounts for the rest
    expect(await substrate.balanceOf(await channel.getAddress())).to.equal(0n);
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
    expect(await substrate.balanceOf(provider.address)).to.equal(0n);
    await time.increase(CHALLENGE_WINDOW + 1);
    await channel.withdrawRemainder(id);
    expect(await substrate.balanceOf(client.address)).to.equal(10_000n);
  });

  // ---------------------------------------------------------------------------
  // Channel-only settlement: coverage the deleted escrow used to hold.
  // The channel is now the ONLY settlement rail, so the per-item economics
  // that escrow encoded (atomic pay-per-item, bounded loss) must be provable
  // here: one voucher per item, cumulative increments, and a theft ceiling of
  // exactly one item's increment. Balances below are net of the service fee
  // (the fee is on the provider side of every voucher, so the theft bound is
  // unchanged and, if anything, tighter).
  // ---------------------------------------------------------------------------

  it("one voucher covers many sequential items via cumulative increments", async function () {
    const id = await openChannel(5000n);
    const ITEM = 800n;
    const items = 5; // 5 items served, cumulative 4000 < 5000 deposit
    let latestSig;
    let cumulative = 0n;
    for (let i = 0; i < items; i++) {
      cumulative += ITEM;
      latestSig = await signVoucher(channel, client, id, cumulative);
    }
    expect(cumulative).to.equal(4000n);

    await channel.connect(provider).closeChannel(id, cumulative, latestSig);
    expect(await substrate.balanceOf(provider.address)).to.equal(netOf(4000n));

    await time.increase(CHALLENGE_WINDOW + 1);
    await channel.withdrawRemainder(id);
    // client paid exactly the served GROSS total; the unserved 1000 refunds.
    expect(await substrate.balanceOf(client.address)).to.equal(10_000n - 4000n);
    expect(await substrate.balanceOf(await channel.getAddress())).to.equal(0n);
  });

  it("client stops paying mid-relationship: provider closes with last voucher, gets exactly the served total (net)", async function () {
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
    expect(await substrate.balanceOf(provider.address)).to.equal(netOf(3000n));

    await time.increase(CHALLENGE_WINDOW + 1);
    await channel.connect(other).withdrawRemainder(id); // anyone can trigger
    expect(await substrate.balanceOf(client.address)).to.equal(10_000n - 3000n);
    expect(await substrate.balanceOf(await channel.getAddress())).to.equal(0n);
  });

  it("theft ceiling: a provider who takes a voucher and never serves steals at most one item's increment (net)", async function () {
    const id = await openChannel(5000n);
    const ITEM = 100n; // client sizes its per-item exposure deliberately small

    // 10 items served honestly: cumulative 1000.
    let cumulative = 0n;
    for (let i = 0; i < 10; i++) {
      cumulative += ITEM;
      await signVoucher(channel, client, id, cumulative); // served + paid
    }
    const honestCumulative = cumulative; // 1000 (gross)

    // Client optimistically signs ONE more voucher for the next item; the
    // provider takes it and never delivers.
    cumulative += ITEM; // 1100
    const stolenSig = await signVoucher(channel, client, id, cumulative);

    await channel.connect(provider).closeChannel(id, cumulative, stolenSig);
    await time.increase(CHALLENGE_WINDOW + 1);
    await channel.withdrawRemainder(id);

    // The provider collected net(1100); the theft beyond honest work is bounded
    // by ONE item's increment (net), not the deposit.
    const providerGot = await substrate.balanceOf(provider.address);
    expect(providerGot).to.equal(netOf(cumulative));
    const theft = providerGot - netOf(honestCumulative);
    expect(theft).to.equal(netOf(cumulative) - netOf(honestCumulative)); // one increment, net
    // Client's balance: 10_000 - gross claimed (fee came out of the claimed).
    expect(await substrate.balanceOf(client.address)).to.equal(10_000n - cumulative);
  });
});

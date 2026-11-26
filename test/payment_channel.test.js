const { expect } = require("chai");
const { ethers } = require("hardhat");
const { time } = require("@nomicfoundation/hardhat-network-helpers");
const { anyValue } = require("@nomicfoundation/hardhat-chai-matchers/withArgs");
const {
  deploySubstrate,
  registerAgent,
  signVoucher,
  deployIndexToken,
  setParity,
  fund,
  SLOT_GOVERNOR,
} = require("./helpers");

const CHALLENGE_WINDOW = 3600;

// Substrate's service fee: 2.5% of the gross payout, rounded up; half of it
// is owed to the treasury, the rest is the tool share (the pool). Vouchers
// are GROSS-denominated; the provider receives net straight from the
// deposit and Substrate receives the fee.
const feeOf = (gross) => (gross * 250n + 9999n) / 10000n;
const toTreasury = (gross) => (feeOf(gross) * 5000n) / 10000n;
const toolShare = (gross) => feeOf(gross) - toTreasury(gross);
const netOf = (gross) => gross - feeOf(gross);

describe("PaymentChannel (prepaid credits in a value-index token, fee at each payout)", function () {
  let substrate, channel, usd, U, S;
  let deployer, client, provider, other, faucet;
  let treasury;

  beforeEach(async function () {
    [deployer, client, provider, other, faucet] = await ethers.getSigners();
    substrate = await deploySubstrate();
    S = await substrate.getAddress();
    // deploySubstrate wires the treasury to signers[0] (the deployer).
    treasury = await substrate.treasury();
    expect(treasury).to.equal(deployer.address);

    await registerAgent(substrate, provider, "prov1");
    await registerAgent(substrate, faucet, "faucet");

    const Ch = await ethers.getContractFactory("PaymentChannel");
    channel = await Ch.deploy(S, CHALLENGE_WINDOW);
    await channel.waitForDeployment();

    // A 6-decimal token of the index; the client holds 10_000 and approves
    // the channel to pull deposits.
    usd = await deployIndexToken(substrate);
    U = await usd.getAddress();
    await fund(usd, client, channel, 10_000n);
  });

  async function openChannel(deposit = 5000n) {
    await channel.connect(client).openChannel(provider.address, U, deposit);
    return await channel.channelCount();
  }

  it("open escrows the deposit and emits", async function () {
    await expect(channel.connect(client).openChannel(provider.address, U, 5000n))
      .to.emit(channel, "ChannelOpened")
      .withArgs(1n, client.address, provider.address, U, 5000n);
    expect(await usd.balanceOf(await channel.getAddress())).to.equal(5000n);
    expect((await channel.getChannel(1n)).token).to.equal(U);
    // Opening lists the token on Substrate, so its fees stay collectable.
    expect(await substrate.tokenDecimals(U)).to.equal(6);
  });

  it("refuses a token that is not in the index", async function () {
    const other = await deployIndexToken(substrate, { parity: 0n });
    await fund(other, client, channel, 1000n);
    await expect(channel.connect(client).openChannel(provider.address, await other.getAddress(), 1000n))
      .to.be.revertedWithCustomError(channel, "TokenNotAccepted");
  });

  it("refuses a token that takes a cut of the deposit", async function () {
    const cut = await deployIndexToken(substrate, { kind: "cut", cutBps: 100 });
    await fund(cut, client, channel, 10_000n);
    await expect(channel.connect(client).openChannel(provider.address, await cut.getAddress(), 10_000n))
      .to.be.revertedWithCustomError(channel, "ShortReceipt").withArgs(10_000n, 9_900n);
  });

  it("a voucher signed for another token does not verify", async function () {
    const id = await openChannel(5000n);
    const sig = await signVoucher(channel, client, id, 1000n, other.address);
    await expect(channel.connect(provider).settle(id, 1000n, sig, 1000n))
      .to.be.revertedWithCustomError(channel, "BadVoucherSignature");
    expect(await channel.voucherHash(id, U, 1000n)).to.not.equal(
      await channel.voucherHash(id, other.address, 1000n));
  });

  it("rejects zero provider / deposit", async function () {
    await expect(
      channel.connect(client).openChannel(ethers.ZeroAddress, U, 1n)
    ).to.be.revertedWithCustomError(channel, "ProviderRequired");
    await expect(
      channel.connect(client).openChannel(provider.address, U, 0n)
    ).to.be.revertedWithCustomError(channel, "ZeroDeposit");
  });

  it("governor sets the challenge window; open channels keep theirs", async function () {
    const governor = (await ethers.getSigners())[SLOT_GOVERNOR];
    expect(await substrate.governor()).to.equal(governor.address);
    const oldId = await openChannel(1000n);

    await expect(channel.connect(other).setChallengeWindow(60))
      .to.be.revertedWithCustomError(channel, "NotGovernor");
    await expect(channel.connect(governor).setChallengeWindow(0))
      .to.be.revertedWithCustomError(channel, "ZeroWindow");
    await expect(channel.connect(governor).setChallengeWindow(60))
      .to.emit(channel, "ChallengeWindowSet").withArgs(CHALLENGE_WINDOW, 60);
    expect(await channel.challengeWindow()).to.equal(60n);

    const newId = await openChannel(1000n);
    const t0 = BigInt(await time.latest());
    // The window applies to a client's close (a provider's close needs none).
    await channel.connect(client).startClose(oldId);
    await channel.connect(client).startClose(newId);
    const oldCh = await channel.getChannel(oldId);
    const newCh = await channel.getChannel(newId);
    expect(oldCh.window).to.equal(BigInt(CHALLENGE_WINDOW));
    expect(newCh.window).to.equal(60n);
    expect(oldCh.closesAt - t0).to.be.gte(BigInt(CHALLENGE_WINDOW));
    expect(newCh.closesAt - t0).to.be.lte(70n);
  });

  it("constructor rejects zero substrate", async function () {
    const Ch = await ethers.getContractFactory("PaymentChannel");
    await expect(Ch.deploy(ethers.ZeroAddress, CHALLENGE_WINDOW)).to.be.reverted;
  });

  it("happy path: provider closes with voucher (net of fee), remainder refunds at once", async function () {
    const id = await openChannel(5000n);
    const cumulative = 3000n;
    const sig = await signVoucher(channel, client, id, cumulative);

    // closeChannel pays the provider the net from the deposit and Substrate
    // the fee (collectFee), split between the treasury and the pool.
    await expect(channel.connect(provider).closeChannel(id, cumulative, sig, cumulative))
      .to.emit(channel, "ChannelClosed")
      .withArgs(id, 3000n, 2000n, anyValue)
      .and.to.emit(substrate, "ServiceFee")
      .withArgs(
        await channel.getAddress(),
        provider.address,
        U,
        cumulative,
        toolShare(cumulative),
        toTreasury(cumulative)
      );

    // Provider receives GROSS minus the 2.5% service fee.
    expect(await usd.balanceOf(provider.address)).to.equal(netOf(cumulative));
    // Substrate holds the fee: the treasury's share owed, the rest the pool's.
    expect(await usd.balanceOf(S)).to.equal(feeOf(cumulative));
    expect(await substrate.held(U)).to.equal(feeOf(cumulative));
    expect(await substrate.treasuryOwed(U)).to.equal(toTreasury(cumulative));
    expect(await substrate.toolShareTotal(U)).to.equal(toolShare(cumulative));

    // The provider's close collected everything it can: no window, the
    // remainder is the client's at once.
    await expect(channel.withdrawRemainder(id))
      .to.emit(channel, "RemainderWithdrawn")
      .withArgs(id, client.address, 2000n);
    // Refund is fee-free: client's balance ends at 10_000 - gross-claimed.
    expect(await usd.balanceOf(client.address)).to.equal(10_000n - 3000n);
    expect(await usd.balanceOf(await channel.getAddress())).to.equal(0n);
  });

  it("a vanished provider never locks the deposit: the client closes and takes it all back", async function () {
    const id = await openChannel(5000n);
    await expect(channel.connect(provider).startClose(id))
      .to.be.revertedWithCustomError(channel, "NotClient");
    await expect(channel.connect(client).startClose(id))
      .to.emit(channel, "ChannelCloseStarted");
    await expect(channel.withdrawRemainder(id))
      .to.be.revertedWithCustomError(channel, "WindowNotElapsed");
    await time.increase(CHALLENGE_WINDOW + 1);
    await expect(channel.withdrawRemainder(id))
      .to.emit(channel, "RemainderWithdrawn")
      .withArgs(id, client.address, 5000n);
    expect(await usd.balanceOf(client.address)).to.equal(10_000n);
  });

  it("a provider's close inside the client's window releases the remainder at once", async function () {
    const id = await openChannel(5000n);
    await channel.connect(client).startClose(id);
    await channel.connect(provider).closeChannel(id, 1200n, await signVoucher(channel, client, id, 1200n), 1200n);
    await expect(channel.withdrawRemainder(id))
      .to.emit(channel, "RemainderWithdrawn").withArgs(id, client.address, 3800n);
    expect(await usd.balanceOf(client.address)).to.equal(10_000n - 1200n);
  });

  it("the provider can still settle in a client's close until the client withdraws", async function () {
    const id = await openChannel(5000n);
    const sig = await signVoucher(channel, client, id, 3000n);
    await channel.connect(client).startClose(id);
    await expect(channel.connect(provider).closeChannel(id, 3000n, sig, 3000n))
      .to.emit(channel, "ChannelClosed");
    expect(await usd.balanceOf(provider.address)).to.equal(netOf(3000n));

    const id2 = await openChannel(1000n);
    const sig2 = await signVoucher(channel, client, id2, 500n);
    await channel.connect(client).startClose(id2);
    await time.increase(CHALLENGE_WINDOW + 1);
    // Past the window but not yet withdrawn (a paused token can hold both
    // sides through it): the provider still collects what was served.
    await expect(channel.connect(provider).closeChannel(id2, 500n, sig2, 500n))
      .to.emit(channel, "ChannelClosed");

    const id3 = await openChannel(1000n);
    const sig3 = await signVoucher(channel, client, id3, 500n);
    await channel.connect(client).startClose(id3);
    await time.increase(CHALLENGE_WINDOW + 1);
    await channel.withdrawRemainder(id3);
    await expect(channel.connect(provider).closeChannel(id3, 500n, sig3, 500n))
      .to.be.revertedWithCustomError(channel, "ChannelNotOpen");
  });

  it("remainder refund is fee-free (no ServiceFee on withdraw)", async function () {
    const id = await openChannel(5000n);
    const sig = await signVoucher(channel, client, id, 1000n);
    await channel.connect(provider).closeChannel(id, 1000n, sig, 1000n);

    const heldBefore = await usd.balanceOf(S);
    const owedBefore = await substrate.treasuryOwed(U);
    await time.increase(CHALLENGE_WINDOW + 1);
    await expect(channel.withdrawRemainder(id)).to.not.emit(substrate, "ServiceFee");
    // No fee on the refund: Substrate's balance and books are unchanged.
    expect(await usd.balanceOf(S)).to.equal(heldBefore);
    expect(await substrate.treasuryOwed(U)).to.equal(owedBefore);
    expect(await substrate.toolShareTotal(U)).to.equal(toolShare(1000n));
    // Client got the full unclaimed remainder back (deposit - gross claimed).
    expect(await usd.balanceOf(client.address)).to.equal(10_000n - 1000n);
  });

  it("only the provider can close", async function () {
    const id = await openChannel();
    const sig = await signVoucher(channel, client, id, 1000n);
    await expect(
      channel.connect(other).closeChannel(id, 1000n, sig, 1000n)
    ).to.be.revertedWithCustomError(channel, "NotProvider");
  });

  it("voucher signed by a non-client is rejected", async function () {
    const id = await openChannel();
    const forged = await signVoucher(channel, other, id, 1000n);
    await expect(
      channel.connect(provider).closeChannel(id, 1000n, forged, 1000n)
    ).to.be.revertedWithCustomError(channel, "BadVoucherSignature");
  });

  it("voucher for a different amount doesn't verify (tamper)", async function () {
    const id = await openChannel();
    const sig = await signVoucher(channel, client, id, 1000n);
    // provider tries to claim more than the signed cumulative
    await expect(
      channel.connect(provider).closeChannel(id, 2000n, sig, 2000n)
    ).to.be.revertedWithCustomError(channel, "BadVoucherSignature");
  });

  it("replay: cannot close the same channel twice", async function () {
    const id = await openChannel();
    const sig = await signVoucher(channel, client, id, 1000n);
    await channel.connect(provider).closeChannel(id, 1000n, sig, 1000n);
    await expect(
      channel.connect(provider).closeChannel(id, 1000n, sig, 1000n)
    ).to.be.revertedWithCustomError(channel, "ChannelNotOpen");
  });

  it("stale (lower) voucher only shortchanges the provider; higher voucher after close is rejected", async function () {
    const id = await openChannel(5000n);
    // provider foolishly closes with an older, lower cumulative
    const staleSig = await signVoucher(channel, client, id, 1000n);
    await channel.connect(provider).closeChannel(id, 1000n, staleSig, 1000n);
    expect(await usd.balanceOf(provider.address)).to.equal(netOf(1000n));

    // it cannot then submit the newer higher voucher Ã¢â‚¬â€ channel is Closing
    const newerSig = await signVoucher(channel, client, id, 4000n);
    await expect(
      channel.connect(provider).closeChannel(id, 4000n, newerSig, 4000n)
    ).to.be.revertedWithCustomError(channel, "ChannelNotOpen");

    // client's remainder reflects the (larger) unclaimed GROSS portion
    await time.increase(CHALLENGE_WINDOW + 1);
    await channel.withdrawRemainder(id);
    expect(await usd.balanceOf(client.address)).to.equal(10_000n - 1000n);
  });

  it("over-claim beyond deposit is capped at the deposit", async function () {
    const id = await openChannel(2000n);
    const sig = await signVoucher(channel, client, id, 9999n);
    await channel.connect(provider).closeChannel(id, 9999n, sig, 9999n);
    // capped at deposit (gross), provider receives net of that
    expect(await usd.balanceOf(provider.address)).to.equal(netOf(2000n));
    await time.increase(CHALLENGE_WINDOW + 1);
    await channel.withdrawRemainder(id);
    // remainder is zero; the provider's net and Substrate's fee account for the rest
    expect(await usd.balanceOf(await channel.getAddress())).to.equal(0n);
  });

  it("cannot withdraw remainder before close, or twice", async function () {
    const id = await openChannel();
    await expect(
      channel.withdrawRemainder(id)
    ).to.be.revertedWithCustomError(channel, "ChannelNotClosing");
    const sig = await signVoucher(channel, client, id, 1000n);
    await channel.connect(provider).closeChannel(id, 1000n, sig, 1000n);
    await time.increase(CHALLENGE_WINDOW + 1);
    await channel.withdrawRemainder(id);
    await expect(
      channel.withdrawRemainder(id)
    ).to.be.revertedWithCustomError(channel, "ChannelNotClosing");
  });

  it("zero-cumulative voucher pays nothing, refunds full deposit", async function () {
    const id = await openChannel(5000n);
    const sig = await signVoucher(channel, client, id, 0n);
    await channel.connect(provider).closeChannel(id, 0n, sig, 0n);
    expect(await usd.balanceOf(provider.address)).to.equal(0n);
    await time.increase(CHALLENGE_WINDOW + 1);
    await channel.withdrawRemainder(id);
    expect(await usd.balanceOf(client.address)).to.equal(10_000n);
  });

  // ---------------------------------------------------------------------------
  // settle: cash in without ending the channel
  // ---------------------------------------------------------------------------

  it("settle claims only up to what was served; the rest stays the client's", async function () {
    const id = await openChannel(5000n);
    // The client paid 3000 ahead; the provider has served 1200 of it.
    const sig = await signVoucher(channel, client, id, 3000n);
    await expect(channel.connect(provider).settle(id, 3000n, sig, 1200n))
      .to.emit(channel, "ChannelPaid")
      .withArgs(id, client.address, provider.address, 1200n, 1200n);
    // Later it has served 2000 under the same voucher: only the new part.
    await expect(channel.connect(provider).settle(id, 3000n, sig, 2000n))
      .to.emit(channel, "ChannelPaid")
      .withArgs(id, client.address, provider.address, 800n, 2000n);
    // upTo above the voucher is capped by the voucher.
    await expect(channel.connect(provider).settle(id, 3000n, sig, 9999n))
      .to.emit(channel, "ChannelPaid")
      .withArgs(id, client.address, provider.address, 1000n, 3000n);
    // Closing: the unserved part of a voucher never leaves the deposit.
    const id2 = await openChannel(5000n);
    const sig2 = await signVoucher(channel, client, id2, 3000n);
    await channel.connect(client).startClose(id2);
    await channel.connect(provider).settle(id2, 3000n, sig2, 1000n);
    await time.increase(CHALLENGE_WINDOW + 1);
    const before = await usd.balanceOf(client.address);
    await channel.withdrawRemainder(id2);
    expect(await usd.balanceOf(client.address)).to.equal(before + 4000n);
  });

  it("settle pays the difference net of fee and leaves the channel Open", async function () {
    const id = await openChannel(5000n);
    const sig1 = await signVoucher(channel, client, id, 1000n);
    await expect(channel.connect(provider).settle(id, 1000n, sig1, 1000n))
      .to.emit(channel, "ChannelPaid")
      .withArgs(id, client.address, provider.address, 1000n, 1000n)
      .and.to.emit(substrate, "ServiceFee")
      .withArgs(await channel.getAddress(), provider.address, U, 1000n,
        toolShare(1000n), toTreasury(1000n));
    expect(await usd.balanceOf(provider.address)).to.equal(netOf(1000n));
    // The fee is taken at the settle, not at the close.
    expect(await usd.balanceOf(S)).to.equal(feeOf(1000n));
    let ch = await channel.getChannel(id);
    expect(ch.status).to.equal(1n); // Open
    expect(ch.claimed).to.equal(1000n);
    expect(ch.closesAt).to.equal(0n);

    // A higher voucher later pays only the new part.
    const sig2 = await signVoucher(channel, client, id, 2500n);
    await expect(channel.connect(provider).settle(id, 2500n, sig2, 2500n))
      .to.emit(channel, "ChannelPaid")
      .withArgs(id, client.address, provider.address, 1500n, 2500n);
    expect(await usd.balanceOf(provider.address))
      .to.equal(netOf(1000n) + netOf(1500n));
    ch = await channel.getChannel(id);
    expect(ch.status).to.equal(1n);
    expect(ch.claimed).to.equal(2500n);
  });

  it("settling the same (or a lower) voucher again reverts NothingToSettle", async function () {
    const id = await openChannel(5000n);
    const sig = await signVoucher(channel, client, id, 1000n);
    await channel.connect(provider).settle(id, 1000n, sig, 1000n);
    await expect(channel.connect(provider).settle(id, 1000n, sig, 1000n))
      .to.be.revertedWithCustomError(channel, "NothingToSettle");
    const lower = await signVoucher(channel, client, id, 400n);
    await expect(channel.connect(provider).settle(id, 400n, lower, 400n))
      .to.be.revertedWithCustomError(channel, "NothingToSettle");
    // A zero voucher on a fresh channel is nothing too.
    const id2 = await openChannel(100n);
    const zero = await signVoucher(channel, client, id2, 0n);
    await expect(channel.connect(provider).settle(id2, 0n, zero, 0n))
      .to.be.revertedWithCustomError(channel, "NothingToSettle");
  });

  it("settle is provider-only, checks the voucher and caps at the deposit", async function () {
    const id = await openChannel(2000n);
    const sig = await signVoucher(channel, client, id, 9999n);
    await expect(channel.connect(other).settle(id, 9999n, sig, 9999n))
      .to.be.revertedWithCustomError(channel, "NotProvider");
    await expect(channel.connect(provider).settle(id, 9000n, sig, 9000n))
      .to.be.revertedWithCustomError(channel, "BadVoucherSignature");
    const forged = await signVoucher(channel, other, id, 1000n);
    await expect(channel.connect(provider).settle(id, 1000n, forged, 1000n))
      .to.be.revertedWithCustomError(channel, "BadVoucherSignature");
    await expect(channel.connect(provider).settle(id, 9999n, sig, 9999n))
      .to.emit(channel, "ChannelPaid")
      .withArgs(id, client.address, provider.address, 2000n, 2000n);
    // Spent: the over-deposit voucher has nothing left to pay.
    await expect(channel.connect(provider).settle(id, 9999n, sig, 9999n))
      .to.be.revertedWithCustomError(channel, "NothingToSettle");
  });

  it("close after settle pays only the rest; remainder is deposit - claimed", async function () {
    const id = await openChannel(5000n);
    await channel.connect(provider).settle(id, 1000n, await signVoucher(channel, client, id, 1000n), 1000n);
    const sig = await signVoucher(channel, client, id, 3000n);
    await expect(channel.connect(provider).closeChannel(id, 3000n, sig, 3000n))
      .to.emit(channel, "ChannelPaid")
      .withArgs(id, client.address, provider.address, 2000n, 3000n)
      .and.to.emit(channel, "ChannelClosed")
      .withArgs(id, 3000n, 2000n, anyValue);
    expect(await usd.balanceOf(provider.address))
      .to.equal(netOf(1000n) + netOf(2000n));
    // Closed: no more settles.
    await expect(channel.connect(provider).settle(id, 3000n, sig, 3000n))
      .to.be.revertedWithCustomError(channel, "ChannelNotOpen");
    await time.increase(CHALLENGE_WINDOW + 1);
    await expect(channel.withdrawRemainder(id))
      .to.emit(channel, "RemainderWithdrawn")
      .withArgs(id, client.address, 2000n);
    expect(await usd.balanceOf(client.address)).to.equal(10_000n - 3000n);
    expect(await usd.balanceOf(await channel.getAddress())).to.equal(0n);
  });

  it("close pays only up to upTo (what was served); the unserved rest of the voucher is the client's", async function () {
    const id = await openChannel(5000n);
    const sig = await signVoucher(channel, client, id, 3000n);
    // A 3000 voucher paid ahead, 1200 of it served.
    await expect(channel.connect(provider).closeChannel(id, 3000n, sig, 1200n))
      .to.emit(channel, "ChannelPaid")
      .withArgs(id, client.address, provider.address, 1200n, 1200n)
      .and.to.emit(channel, "ChannelClosed")
      .withArgs(id, 1200n, 3800n, anyValue);
    expect(await usd.balanceOf(provider.address)).to.equal(netOf(1200n));
    await expect(channel.withdrawRemainder(id))
      .to.emit(channel, "RemainderWithdrawn").withArgs(id, client.address, 3800n);
    // upTo at or below what settle already paid pays nothing more.
    const id2 = await openChannel(1000n);
    const sig2 = await signVoucher(channel, client, id2, 800n);
    await channel.connect(provider).settle(id2, 800n, sig2, 500n);
    await expect(channel.connect(provider).closeChannel(id2, 800n, sig2, 400n))
      .to.emit(channel, "ChannelClosed").withArgs(id2, 500n, 500n, anyValue)
      .and.not.to.emit(channel, "ChannelPaid");
  });

  it("close with nothing new pays nothing more and emits no ChannelPaid", async function () {
    const id = await openChannel(5000n);
    const sig = await signVoucher(channel, client, id, 1000n);
    await channel.connect(provider).settle(id, 1000n, sig, 1000n);
    await expect(channel.connect(provider).closeChannel(id, 1000n, sig, 1000n))
      .to.emit(channel, "ChannelClosed")
      .withArgs(id, 1000n, 4000n, anyValue)
      .and.not.to.emit(channel, "ChannelPaid");
    // A lower voucher than claimed does not lower claimed either.
    const id2 = await openChannel(1000n);
    await channel.connect(provider).settle(id2, 800n, await signVoucher(channel, client, id2, 800n), 800n);
    await expect(channel.connect(provider).closeChannel(
      id2, 300n, await signVoucher(channel, client, id2, 300n), 300n))
      .to.emit(channel, "ChannelClosed").withArgs(id2, 800n, 200n, anyValue);
    expect((await channel.getChannel(id2)).claimed).to.equal(800n);
  });

  it("settle inside a client's close keeps its deadline; the client gets the rest after it", async function () {
    const id = await openChannel(5000n);
    await channel.connect(client).startClose(id);
    const deadline = (await channel.getChannel(id)).closesAt;
    await channel.connect(provider).settle(id, 1200n, await signVoucher(channel, client, id, 1200n), 1200n);
    const ch = await channel.getChannel(id);
    expect(ch.status).to.equal(4n); // ClientClosing
    expect(ch.closesAt).to.equal(deadline);
    await time.increase(CHALLENGE_WINDOW + 1);
    await expect(channel.withdrawRemainder(id))
      .to.emit(channel, "RemainderWithdrawn").withArgs(id, client.address, 3800n);
    expect(await usd.balanceOf(client.address)).to.equal(10_000n - 1200n);
    await expect(channel.connect(provider).settle(
      id, 1500n, await signVoucher(channel, client, id, 1500n), 1500n))
      .to.be.revertedWithCustomError(channel, "ChannelNotOpen");
  });

  // ---------------------------------------------------------------------------
  // topUp: add to the buyer's credit
  // ---------------------------------------------------------------------------

  it("topUp grows the deposit, pulls the token and emits", async function () {
    const id = await openChannel(1000n);
    await expect(channel.connect(client).topUp(id, 500n))
      .to.emit(channel, "ChannelToppedUp")
      .withArgs(id, client.address, 500n, 1500n);
    expect((await channel.getChannel(id)).deposit).to.equal(1500n);
    expect(await usd.balanceOf(await channel.getAddress())).to.equal(1500n);
    expect(await usd.balanceOf(client.address)).to.equal(10_000n - 1500n);
    // A voucher above the old deposit is now collectable up to the new one.
    await expect(channel.connect(provider).settle(
      id, 1400n, await signVoucher(channel, client, id, 1400n), 1400n))
      .to.emit(channel, "ChannelPaid")
      .withArgs(id, client.address, provider.address, 1400n, 1400n);
  });

  it("topUp is client-only, non-zero, and only while Open", async function () {
    const id = await openChannel(1000n);
    await expect(channel.connect(provider).topUp(id, 100n))
      .to.be.revertedWithCustomError(channel, "NotClient");
    await expect(channel.connect(client).topUp(id, 0n))
      .to.be.revertedWithCustomError(channel, "ZeroDeposit");
    await expect(channel.connect(client).topUp(999n, 100n))
      .to.be.revertedWithCustomError(channel, "ChannelNotOpen");
    await channel.connect(client).startClose(id);
    await expect(channel.connect(client).topUp(id, 100n))
      .to.be.revertedWithCustomError(channel, "ChannelNotOpen");
    const id2 = await openChannel(1000n);
    await channel.connect(provider).closeChannel(
      id2, 100n, await signVoucher(channel, client, id2, 100n), 100n);
    await expect(channel.connect(client).topUp(id2, 100n))
      .to.be.revertedWithCustomError(channel, "ChannelNotOpen");
  });

  it("topUp needs the allowance", async function () {
    const id = await openChannel(1000n);
    await usd.connect(client).approve(await channel.getAddress(), 0n);
    await expect(channel.connect(client).topUp(id, 100n)).to.be.reverted;
  });

  it("theft ceiling holds across settles: one unserved voucher's increment", async function () {
    const id = await openChannel(5000n);
    const ITEM = 100n;
    let cumulative = 0n;
    for (let i = 0; i < 10; i++) {
      cumulative += ITEM;
      const s = await signVoucher(channel, client, id, cumulative);
      if (i % 3 === 2) await channel.connect(provider).settle(id, cumulative, s, cumulative);
    }
    const honest = cumulative; // 1000
    cumulative += ITEM;        // one voucher the provider never serves
    const stolen = await signVoucher(channel, client, id, cumulative);
    await channel.connect(provider).closeChannel(id, cumulative, stolen, cumulative);
    await time.increase(CHALLENGE_WINDOW + 1);
    await channel.withdrawRemainder(id);
    expect((await channel.getChannel(id)).claimed - honest).to.equal(ITEM);
    expect(await usd.balanceOf(client.address)).to.equal(10_000n - cumulative);
    expect(await usd.balanceOf(await channel.getAddress())).to.equal(0n);
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

    await channel.connect(provider).closeChannel(id, cumulative, latestSig, cumulative);
    expect(await usd.balanceOf(provider.address)).to.equal(netOf(4000n));

    await time.increase(CHALLENGE_WINDOW + 1);
    await channel.withdrawRemainder(id);
    // client paid exactly the served GROSS total; the unserved 1000 refunds.
    expect(await usd.balanceOf(client.address)).to.equal(10_000n - 4000n);
    expect(await usd.balanceOf(await channel.getAddress())).to.equal(0n);
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
    // client vanishes here Ã¢â‚¬â€ no further vouchers exist.

    await channel.connect(provider).closeChannel(id, cumulative, lastSig, cumulative);
    expect(await usd.balanceOf(provider.address)).to.equal(netOf(3000n));

    await time.increase(CHALLENGE_WINDOW + 1);
    await channel.connect(other).withdrawRemainder(id); // anyone can trigger
    expect(await usd.balanceOf(client.address)).to.equal(10_000n - 3000n);
    expect(await usd.balanceOf(await channel.getAddress())).to.equal(0n);
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

    await channel.connect(provider).closeChannel(id, cumulative, stolenSig, cumulative);
    await time.increase(CHALLENGE_WINDOW + 1);
    await channel.withdrawRemainder(id);

    // The provider collected net(1100); the theft beyond honest work is bounded
    // by ONE item's increment (net), not the deposit.
    const providerGot = await usd.balanceOf(provider.address);
    expect(providerGot).to.equal(netOf(cumulative));
    const theft = providerGot - netOf(honestCumulative);
    expect(theft).to.equal(netOf(cumulative) - netOf(honestCumulative)); // one increment, net
    // Client's balance: 10_000 - gross claimed (fee came out of the claimed).
    expect(await usd.balanceOf(client.address)).to.equal(10_000n - cumulative);
  });
});

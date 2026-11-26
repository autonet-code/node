const { expect } = require("chai");
const { ethers } = require("hardhat");
const { time } = require("@nomicfoundation/hardhat-network-helpers");
const {
  ZERO,
  digest,
  deploySubstrateWith,
  registerAgent,
  deployIndexToken,
  setParity,
  fund,
  feeOf,
  payFees,
  anchorEpoch,
  claimLeaf,
  claimTree,
  claimTx,
  sortBasket,
  signVoucher,
  SLOT_GOVERNOR,
} = require("./helpers");

// Services are paid in the tokens of the jurisdiction's value index. Each
// payment's fee is split: the treasury's share is owed to it, the rest
// fills that token's pool. Each epoch's close commits one claim leaf per
// earner (a basket over the pools plus its REP value); a claim credits it
// and pays every token in the same transaction. A token whose transfer
// fails stays owed and is withdrawn on its own later.

describe("Value-index payments, pools and claims", function () {
  let signers, deployer, payer, recipient, alice, bob, stranger, gov;
  let substrate, S, usd, U, dai, D;

  beforeEach(async function () {
    signers = await ethers.getSigners();
    [deployer, payer, recipient, alice, bob, stranger] = signers;
    gov = signers[SLOT_GOVERNOR];
    substrate = await deploySubstrateWith();
    S = await substrate.getAddress();
    usd = await deployIndexToken(substrate, { decimals: 6, name: "USD6", symbol: "USD6" });
    U = await usd.getAddress();
    dai = await deployIndexToken(substrate, { decimals: 18, name: "DAI18", symbol: "DAI18" });
    D = await dai.getAddress();
  });

  // Fees in both tokens, then two anchors so both pools are claimable.
  // Returns the tool share per token.
  async function fillPools(usdGross = 1_000_000n, daiGross = 10n ** 18n) {
    const a = await payFees(substrate, usd, usdGross);
    const b = await payFees(substrate, dai, daiGross);
    await anchorEpoch(substrate, "fill-1");
    return { usdPool: a.toolShare, daiPool: b.toolShare };
  }

  describe("payments", function () {
    it("pays in a 6-decimal token: net straight to the recipient, the fee to Substrate", async function () {
      await fund(usd, payer, substrate, 1_000_000n);
      await expect(substrate.connect(payer).payForService(U, recipient.address, 1_000_000n, digest("r1")))
        .to.emit(substrate, "ServicePayment")
        .withArgs(payer.address, recipient.address, U, 1_000_000n, digest("r1"))
        .and.to.emit(substrate, "ServiceFee")
        .withArgs(payer.address, recipient.address, U, 1_000_000n, 12_500n, 12_500n);
      expect(await usd.balanceOf(recipient.address)).to.equal(975_000n);
      expect(await usd.balanceOf(S)).to.equal(25_000n);
      expect(await usd.balanceOf(payer.address)).to.equal(0n);
      expect(await substrate.held(U)).to.equal(25_000n);
      expect(await substrate.feeTotal(U)).to.equal(25_000n);
    });

    it("pays in an 18-decimal token", async function () {
      const gross = 3n * 10n ** 18n;
      await fund(dai, payer, substrate, gross);
      await substrate.connect(payer).payForService(D, recipient.address, gross, digest("r"));
      const fee = feeOf(gross);
      expect(fee).to.equal(75n * 10n ** 15n);
      expect(await dai.balanceOf(recipient.address)).to.equal(gross - fee);
      expect(await dai.balanceOf(S)).to.equal(fee);
      expect(await substrate.treasuryOwed(D)).to.equal(fee / 2n);
      expect(await substrate.toolShareTotal(D)).to.equal(fee - fee / 2n);
      expect(await substrate.tokenDecimals(D)).to.equal(18);
      // Value is 6-decimal units at the parity: 3 DAI at parity 1e18 is 3e6.
      expect(await substrate["valueOf(address,uint256)"](D, gross)).to.equal(3_000_000n);
      expect(await substrate["valueOf(address,uint256)"](U, 3_000_000n)).to.equal(3_000_000n);
    });

    it("rounds the fee up: a gross of 1 pays a fee of 1 and nets nothing", async function () {
      await fund(usd, payer, substrate, 1n);
      await expect(substrate.connect(payer).payForService(U, recipient.address, 1n, digest("tiny")))
        .to.emit(substrate, "ServiceFee")
        .withArgs(payer.address, recipient.address, U, 1n, 1n, 0n);
      expect(await usd.balanceOf(recipient.address)).to.equal(0n);
      expect(await usd.balanceOf(S)).to.equal(1n);
      // Splitting a payment never escapes the fee.
      await fund(usd, payer, substrate, 2n);
      await substrate.connect(payer).payForService(U, recipient.address, 1n, digest("a"));
      await substrate.connect(payer).payForService(U, recipient.address, 1n, digest("b"));
      expect(await substrate.feeTotal(U)).to.equal(3n);
    });

    it("splits the fee: the treasury share is owed, the tool share fills the pool, checkpointed", async function () {
      const s = await deploySubstrateWith({ serviceFeeBps: 1000, feeTreasuryBps: 9000 });
      const t = await deployIndexToken(s);
      const T = await t.getAddress();
      await fund(t, payer, s, 20_000n);
      await s.connect(payer).payForService(T, recipient.address, 10_000n, digest("1"));
      const t1 = BigInt((await ethers.provider.getBlock("latest")).timestamp);
      await s.connect(payer).payForService(T, recipient.address, 10_000n, digest("2"));
      const t2 = BigInt((await ethers.provider.getBlock("latest")).timestamp);
      // fee 1_000 each: 900 to the treasury, 100 tool share.
      expect(await s.treasuryOwed(T)).to.equal(1_800n);
      expect(await s.toolShareTotal(T)).to.equal(200n);
      expect(await s.feeTotal(T)).to.equal(2_000n);
      expect(await s.toolShareUpTo(T, t1)).to.equal(100n);
      expect(await s.toolShareUpTo(T, t1 - 1n)).to.equal(0n);
      expect(await s.toolShareBetween(T, t1, t2)).to.equal(100n);
      expect(await s.feesUpTo(T, t1)).to.equal(1_000n);
      expect(await s.feesBetween(T, t1, t2)).to.equal(1_000n);
      expect(await s.feesBetween(T, t2, t1)).to.equal(0n);
      // The governor cannot push the treasury share past 9000.
      await expect(s.connect(gov).setFeeTreasuryBps(9001))
        .to.be.revertedWithCustomError(s, "FeeOutOfBounds");
    });

    it("refuses a token that is not in the index", async function () {
      const t = await deployIndexToken(substrate, { parity: 0n });
      await fund(t, payer, substrate, 1000n);
      await expect(substrate.connect(payer).payForService(await t.getAddress(), recipient.address, 1000n, digest("x")))
        .to.be.revertedWithCustomError(substrate, "NotInIndex");
      expect(await substrate.acceptsToken(await t.getAddress())).to.equal(false);
      expect(await substrate["valueOf(address,uint256)"](await t.getAddress(), 1000n)).to.equal(0n);
      await expect(substrate.listToken(await t.getAddress()))
        .to.be.revertedWithCustomError(substrate, "NotInIndex");
    });

    it("accepts a token with any number of decimals, 0 included", async function () {
      for (const d of [0, 2, 24]) {
        const t = await deployIndexToken(substrate, { decimals: d });
        const T = await t.getAddress();
        expect(await substrate.acceptsToken(T)).to.equal(true);
        await fund(t, payer, substrate, 1000n);
        await substrate.connect(payer).payForService(T, recipient.address, 1000n, digest(`x${d}`));
        expect(await substrate.tokenListed(T)).to.equal(true);
        expect(await substrate.tokenDecimals(T)).to.equal(BigInt(d));
      }
      // An address with no code is refused too.
      await setParity(substrate, { getAddress: async () => stranger.address }, 1n);
      expect(await substrate.acceptsToken(stranger.address)).to.equal(false);
    });

    it("refuses a token that takes a cut on transfer (ShortReceipt)", async function () {
      const cut = await deployIndexToken(substrate, { kind: "cut", cutBps: 100 });
      const C = await cut.getAddress();
      await fund(cut, payer, substrate, 1_000_000_000n);
      // fee 25_000_000, of which the token keeps 1%.
      await expect(substrate.connect(payer).payForService(C, recipient.address, 1_000_000_000n, digest("x")))
        .to.be.revertedWithCustomError(substrate, "ShortReceipt")
        .withArgs(C, 25_000_000n, 24_750_000n);
    });

    it("refuses a zero recipient and a zero amount", async function () {
      await fund(usd, payer, substrate, 10n);
      await expect(substrate.connect(payer).payForService(U, ZERO, 10n, digest("x")))
        .to.be.revertedWithCustomError(substrate, "ZeroAddress");
      await expect(substrate.connect(payer).payForService(U, recipient.address, 0n, digest("x")))
        .to.be.revertedWithCustomError(substrate, "ZeroAmount");
    });

    it("collectFee needs a listed token and pulls exactly the fee", async function () {
      const t = await deployIndexToken(substrate);
      const T = await t.getAddress();
      await fund(t, payer, substrate, 1_000n);
      await expect(substrate.connect(payer).collectFee(T, recipient.address, 1_000n, digest("x")))
        .to.be.revertedWithCustomError(substrate, "TokenNotListed");
      await substrate.listToken(T);
      expect(await substrate.poolTokens()).to.include(T);
      await substrate.connect(payer).collectFee(T, recipient.address, 1_000n, digest("x"));
      expect(await t.balanceOf(S)).to.equal(25n);
      expect(await t.balanceOf(payer.address)).to.equal(975n);
    });
  });

  describe("the pools and the claims", function () {
    it("nothing is claimable after only one anchor", async function () {
      await payFees(substrate, usd, 1_000_000n);
      const tree = claimTree([{ earner: alice.address, tokens: [U], amounts: [1n], value: 1n }]);
      await anchorEpoch(substrate, "e1", tree.root);
      expect(await substrate.poolCap(U)).to.equal(0n);
      await expect(claimTx(substrate, alice, tree, 0, "e1"))
        .to.be.revertedWithCustomError(substrate, "PoolExceeded")
        .withArgs(U, 0n, 1n);
    });

    it("claims above poolCap revert; up to it they succeed (lag one)", async function () {
      const { usdPool } = await fillPools();
      expect(usdPool).to.equal(12_500n);
      // Fees paid in the second epoch are not in this epoch's pool.
      await payFees(substrate, usd, 1_000_000n);
      const over = claimTree([{ earner: alice.address, tokens: [U], amounts: [usdPool + 1n], value: 0n }]);
      await anchorEpoch(substrate, "e2", over.root);
      expect(await substrate.poolCap(U)).to.equal(usdPool);
      await expect(claimTx(substrate, alice, over, 0, "e2"))
        .to.be.revertedWithCustomError(substrate, "PoolExceeded")
        .withArgs(U, usdPool, usdPool + 1n);

      const exact = claimTree([{ earner: alice.address, tokens: [U], amounts: [usdPool], value: 0n }]);
      await anchorEpoch(substrate, "e3", exact.root);
      // Now the second epoch's fees are in too: the cap grows to both.
      expect(await substrate.poolCap(U)).to.equal(2n * usdPool);
      await claimTx(substrate, alice, exact, 0, "e3");
      expect(await substrate.claimedTotal(U)).to.equal(usdPool);
    });

    it("a basket claim over two tokens; double claim, bad proof and unsorted tokens refused", async function () {
      const { usdPool, daiPool } = await fillPools();
      const tree = claimTree([
        { earner: alice.address, basket: [[U, usdPool / 2n], [D, daiPool / 2n]], value: 500n },
        { earner: bob.address, basket: [[D, daiPool / 4n], [U, usdPool / 4n]], value: 250n },
        { earner: stranger.address, basket: [[U, 1n]], value: 1n },
      ]);
      await anchorEpoch(substrate, "e2", tree.root);
      // The JS leaf matches the contract's.
      const e0 = tree.entries[0];
      expect(await substrate.claimLeaf(e0.earner, e0.tokens, e0.amounts))
        .to.equal(claimLeaf(e0.earner, e0.tokens, e0.amounts));

      // Anyone submits; the leaf fixes the earner.
      await expect(claimTx(substrate, stranger, tree, 0, "e2"))
        .to.emit(substrate, "Claimed").withArgs(alice.address, digest("e2"), 12_500n, 12_500n)
        .and.to.emit(substrate, "ClaimedToken").withArgs(alice.address, digest("e2"), U, usdPool / 2n)
        .and.to.emit(substrate, "ClaimedToken").withArgs(alice.address, digest("e2"), D, daiPool / 2n)
        .and.to.emit(substrate, "Withdrawn").withArgs(alice.address, U, alice.address, usdPool / 2n)
        .and.to.emit(substrate, "Withdrawn").withArgs(alice.address, D, alice.address, daiPool / 2n);
      // The same transaction paid the whole basket.
      expect(await usd.balanceOf(alice.address)).to.equal(usdPool / 2n);
      expect(await dai.balanceOf(alice.address)).to.equal(daiPool / 2n);
      expect(await substrate.owed(alice.address, U)).to.equal(0n);
      expect(await substrate.owed(alice.address, D)).to.equal(0n);
      expect(await substrate.claimedEpoch(alice.address, digest("e2"))).to.equal(true);

      await expect(claimTx(substrate, alice, tree, 0, "e2"))
        .to.be.revertedWithCustomError(substrate, "AlreadyClaimed");

      // Bad proof: bob's amounts with alice's proof, or inflated amounts.
      const b = tree.entries[1];
      await expect(substrate.claim(b.earner, digest("e2"), b.tokens, b.amounts, tree.proof(0)))
        .to.be.revertedWithCustomError(substrate, "ClaimProofInvalid");
      await expect(substrate.claim(b.earner, digest("e2"), b.tokens,
        b.amounts.map((a) => a + 1n), tree.proof(1)))
        .to.be.revertedWithCustomError(substrate, "ClaimProofInvalid");

      // Unsorted (or repeated) tokens are refused before the proof.
      await expect(substrate.claim(b.earner, digest("e2"), [...b.tokens].reverse(),
        [...b.amounts].reverse(), tree.proof(1)))
        .to.be.revertedWithCustomError(substrate, "TokensNotSorted");
      await expect(substrate.claim(b.earner, digest("e2"), [U, U], [1n, 1n], tree.proof(1)))
        .to.be.revertedWithCustomError(substrate, "TokensNotSorted");
      await expect(substrate.claim(b.earner, digest("e2"), b.tokens, [1n], tree.proof(1)))
        .to.be.revertedWithCustomError(substrate, "LengthMismatch");

      await claimTx(substrate, bob, tree, 1, "e2");
      await claimTx(substrate, bob, tree, 2, "e2");
      expect(await dai.balanceOf(bob.address)).to.equal(daiPool / 4n);
      expect(await substrate.claimedTotal(U)).to.equal(usdPool / 2n + usdPool / 4n + 1n);
      // 12_500 (alice) + 6_250 (bob) + 1 (stranger), valued on chain.
      expect(await substrate.totalEarnedValue()).to.equal(18_751n);
    });

    it("refuses claims on an unanchored epoch or one with no claim root", async function () {
      await fillPools();
      await expect(substrate.claim(alice.address, digest("nope"), [U], [1n], []))
        .to.be.revertedWithCustomError(substrate, "EpochNotAnchored");
      await anchorEpoch(substrate, "empty");
      await expect(substrate.claim(alice.address, digest("empty"), [U], [1n], []))
        .to.be.revertedWithCustomError(substrate, "ClaimRootMissing");
    });

    it("earnedValue accumulates across epochs, valued on chain at the anchor's parities", async function () {
      await fillPools();
      const t2 = claimTree([{ earner: alice.address, tokens: [U], amounts: [100n] }]);
      await anchorEpoch(substrate, "e2", t2.root);
      await claimTx(substrate, alice, t2, 0, "e2");
      const t3 = claimTree([
        { earner: alice.address, tokens: [U], amounts: [50n] },
        { earner: bob.address, basket: [[D, 2n * 10n ** 12n]] },
      ]);
      await anchorEpoch(substrate, "e3", t3.root);
      expect(await substrate.anchorParity(2, U)).to.equal(10n ** 30n);
      expect(await substrate.anchorParity(2, D)).to.equal(10n ** 18n);
      // A parity edit after the anchor changes nothing for its claims.
      await setParity(substrate, dai, 5n * 10n ** 18n);
      await expect(claimTx(substrate, alice, t3, 0, "e3"))
        .to.emit(substrate, "Claimed").withArgs(alice.address, digest("e3"), 50n, 150n);
      await claimTx(substrate, bob, t3, 1, "e3");
      expect(await substrate.earnedValue(alice.address)).to.equal(150n);
      expect(await substrate.earnedValue(bob.address)).to.equal(2n);
      expect(await substrate.totalEarnedValue()).to.equal(152n);
      expect(await usd.balanceOf(alice.address)).to.equal(150n);
    });

    it("no leaf can grant more REP value than the capped amounts are worth", async function () {
      const { usdPool } = await fillPools();
      // Take the token out of the index before the epoch is anchored: its
      // amounts still pay out, and are worth no REP.
      await setParity(substrate, usd, 0n);
      const t = claimTree([{ earner: alice.address, tokens: [U], amounts: [usdPool] }]);
      await anchorEpoch(substrate, "e2", t.root);
      await claimTx(substrate, alice, t, 0, "e2");
      expect(await usd.balanceOf(alice.address)).to.equal(usdPool);
      expect(await substrate.earnedValue(alice.address)).to.equal(0n);
      // A leaf with no amounts is worth nothing.
      const t3 = claimTree([{ earner: bob.address, tokens: [], amounts: [] }]);
      await anchorEpoch(substrate, "e3", t3.root);
      await claimTx(substrate, bob, t3, 0, "e3");
      expect(await substrate.earnedValue(bob.address)).to.equal(0n);
    });

    it("a malformed parity reads as out of the index and never blocks an anchor", async function () {
      await fillPools();
      const reg = await ethers.getContractAt("MockRegistry", await substrate.registry());
      await reg.setValue(`jurisdiction.parity.${U.toLowerCase()}`, "not a number");
      expect(await substrate.parityOf(U)).to.equal(0n);
      await anchorEpoch(substrate, "e2");
      expect(await substrate.anchorParity(1, U)).to.equal(0n);
    });
  });

  describe("payouts and withdrawals", function () {
    let usdPool, p, P, tree;
    // A basket over two plain tokens and a pausable one.
    beforeEach(async function () {
      ({ usdPool } = await fillPools());
      p = await deployIndexToken(substrate, { kind: "pausable" });
      P = await p.getAddress();
      await payFees(substrate, p, 1_000_000n);
      await anchorEpoch(substrate, "p-fill");
      tree = claimTree([
        { earner: alice.address, basket: [[U, 1_000n], [D, 10n ** 12n], [P, 500n]], value: 1n },
      ]);
      await anchorEpoch(substrate, "e2", tree.root);
    });

    it("a claim pays every token of the basket in one transaction", async function () {
      await expect(claimTx(substrate, stranger, tree, 0, "e2"))
        .to.emit(substrate, "Withdrawn").withArgs(alice.address, U, alice.address, 1_000n)
        .and.to.emit(substrate, "Withdrawn").withArgs(alice.address, D, alice.address, 10n ** 12n)
        .and.to.emit(substrate, "Withdrawn").withArgs(alice.address, P, alice.address, 500n)
        .and.not.to.emit(substrate, "PayDeferred");
      expect(await usd.balanceOf(alice.address)).to.equal(1_000n);
      expect(await dai.balanceOf(alice.address)).to.equal(10n ** 12n);
      expect(await p.balanceOf(alice.address)).to.equal(500n);
      expect(await usd.balanceOf(stranger.address)).to.equal(0n);
      for (const t of [U, D, P]) expect(await substrate.owed(alice.address, t)).to.equal(0n);
      await expect(substrate.withdraw(alice.address, U))
        .to.be.revertedWithCustomError(substrate, "NothingOwed");
    });

    it("a paused token is deferred while the others pay; withdraw pays it after unpause", async function () {
      await p.setPaused(true);
      await expect(claimTx(substrate, stranger, tree, 0, "e2"))
        .to.emit(substrate, "PayDeferred").withArgs(alice.address, P, 500n)
        .and.to.emit(substrate, "Withdrawn").withArgs(alice.address, U, alice.address, 1_000n)
        .and.to.emit(substrate, "Withdrawn").withArgs(alice.address, D, alice.address, 10n ** 12n);
      expect(await usd.balanceOf(alice.address)).to.equal(1_000n);
      expect(await dai.balanceOf(alice.address)).to.equal(10n ** 12n);
      // The paused token stays owed, and the books still match.
      expect(await substrate.owed(alice.address, P)).to.equal(500n);
      expect(await substrate.held(P)).to.equal(await p.balanceOf(S));
      await expect(substrate.withdraw(alice.address, P)).to.be.revertedWithCustomError(p, "TokenPaused");
      expect(await substrate.owed(alice.address, P)).to.equal(500n);
      // Payments in the other tokens still go through.
      await fund(usd, payer, substrate, 1_000n);
      await substrate.connect(payer).payForService(U, recipient.address, 1_000n, digest("still"));
      // Unpaused, anyone's withdraw pays it to the earner.
      await p.setPaused(false);
      await expect(substrate.connect(stranger).withdraw(alice.address, P))
        .to.emit(substrate, "Withdrawn").withArgs(alice.address, P, alice.address, 500n);
      expect(await p.balanceOf(alice.address)).to.equal(500n);
      expect(await substrate.owed(alice.address, P)).to.equal(0n);
      await expect(substrate.withdraw(alice.address, P))
        .to.be.revertedWithCustomError(substrate, "NothingOwed");
    });

    it("withdrawTo sends the caller's own deferred money elsewhere", async function () {
      await p.setPaused(true);
      await claimTx(substrate, alice, tree, 0, "e2");
      await p.setPaused(false);
      await expect(substrate.connect(alice).withdrawTo(P, ZERO))
        .to.be.revertedWithCustomError(substrate, "ZeroAddress");
      await expect(substrate.connect(stranger).withdrawTo(P, stranger.address))
        .to.be.revertedWithCustomError(substrate, "NothingOwed");
      await expect(substrate.connect(alice).withdrawTo(P, bob.address))
        .to.emit(substrate, "Withdrawn").withArgs(alice.address, P, bob.address, 500n);
      expect(await p.balanceOf(bob.address)).to.equal(500n);
    });

    it("withdrawTreasury pays the treasury its share, once", async function () {
      await claimTx(substrate, alice, tree, 0, "e2");
      const owed = await substrate.treasuryOwed(U);
      expect(owed).to.equal(12_500n);
      await expect(substrate.connect(stranger).withdrawTreasury(U))
        .to.emit(substrate, "TreasuryWithdrawn").withArgs(U, owed);
      expect(await usd.balanceOf(deployer.address)).to.equal(owed);
      await expect(substrate.withdrawTreasury(U))
        .to.be.revertedWithCustomError(substrate, "NothingOwed");
      // The books still match what Substrate holds: the pool less the payout.
      expect(await substrate.held(U)).to.equal(await usd.balanceOf(S));
      expect(await substrate.held(U)).to.equal(usdPool - 1_000n);
    });

    it("sweep moves a stray transfer to the treasury's share", async function () {
      await expect(substrate.sweep(U)).to.be.revertedWithCustomError(substrate, "NothingOwed");
      await usd.mint(S, 333n);
      const before = await substrate.treasuryOwed(U);
      await expect(substrate.connect(stranger).sweep(U)).to.emit(substrate, "Swept").withArgs(U, 333n);
      expect(await substrate.treasuryOwed(U)).to.equal(before + 333n);
      expect(await substrate.held(U)).to.equal(await usd.balanceOf(S));
      // The pools take only fees: a sweep never raises the cap.
      expect(await substrate.poolCap(U)).to.equal(usdPool);
      // With no treasury there is nowhere to sweep to.
      const s0 = await deploySubstrateWith({ treasury: ZERO });
      await usd.mint(await s0.getAddress(), 5n);
      await expect(s0.sweep(U)).to.be.revertedWithCustomError(s0, "ZeroAddress");
    });
  });

  describe("a token whose transfer returns false", function () {
    it("is deferred at the claim, the books restored, and paid by withdraw later", async function () {
      const f = await deployIndexToken(substrate, { kind: "false" });
      const F = await f.getAddress();
      await payFees(substrate, f, 1_000_000n);
      await payFees(substrate, usd, 1_000_000n);
      await anchorEpoch(substrate, "f1");
      const tree = claimTree([{ earner: bob.address, basket: [[F, 700n], [U, 300n]], value: 1n }]);
      await anchorEpoch(substrate, "f2", tree.root);
      const heldBefore = await substrate.held(F);

      await f.setFailing(true);
      await expect(claimTx(substrate, bob, tree, 0, "f2"))
        .to.emit(substrate, "PayDeferred").withArgs(bob.address, F, 700n)
        .and.to.emit(substrate, "Withdrawn").withArgs(bob.address, U, bob.address, 300n);
      expect(await usd.balanceOf(bob.address)).to.equal(300n);
      expect(await f.balanceOf(bob.address)).to.equal(0n);
      expect(await substrate.owed(bob.address, F)).to.equal(700n);
      expect(await substrate.held(F)).to.equal(heldBefore);
      expect(await substrate.held(F)).to.equal(await f.balanceOf(S));
      // withdraw uses safeTransfer: a false return reverts and changes nothing.
      await expect(substrate.withdraw(bob.address, F)).to.be.reverted;
      expect(await substrate.owed(bob.address, F)).to.equal(700n);

      await f.setFailing(false);
      await expect(substrate.withdraw(bob.address, F))
        .to.emit(substrate, "Withdrawn").withArgs(bob.address, F, bob.address, 700n);
      expect(await f.balanceOf(bob.address)).to.equal(700n);
      expect(await substrate.held(F)).to.equal(await f.balanceOf(S));
    });
  });

  describe("a USDT-style token (no return values)", function () {
    it("works through payment, channel and a claim that pays it", async function () {
      const nr = await deployIndexToken(substrate, { kind: "noreturn" });
      const N = await nr.getAddress();
      const provider = signers[7];
      await registerAgent(substrate, provider, "prov");

      // Direct payment.
      await fund(nr, payer, substrate, 1_000_000n);
      await substrate.connect(payer).payForService(N, recipient.address, 1_000_000n, digest("x"));
      expect(await nr.balanceOf(recipient.address)).to.equal(975_000n);
      expect(await nr.balanceOf(S)).to.equal(25_000n);

      // Channel: open, settle, close, refund.
      const Ch = await ethers.getContractFactory("PaymentChannel");
      const channel = await Ch.deploy(S, 3600);
      await channel.waitForDeployment();
      await fund(nr, alice, channel, 10_000n);
      await channel.connect(alice).openChannel(provider.address, N, 10_000n);
      const id = await channel.channelCount();
      await channel.connect(provider).settle(id, 4_000n, await signVoucher(channel, alice, id, 4_000n), 4_000n);
      expect(await nr.balanceOf(provider.address)).to.equal(3_900n);
      await channel.connect(provider).closeChannel(id, 6_000n, await signVoucher(channel, alice, id, 6_000n), 6_000n);
      expect(await nr.balanceOf(provider.address)).to.equal(5_850n);
      await channel.withdrawRemainder(id);
      expect(await nr.balanceOf(alice.address)).to.equal(4_000n);
      expect(await nr.balanceOf(S)).to.equal(25_150n);
      expect(await substrate.held(N)).to.equal(25_150n);

      // The claim pays it in the same transaction (empty return data is success).
      await anchorEpoch(substrate, "n1");
      const tree = claimTree([{ earner: bob.address, tokens: [N], amounts: [1_234n], value: 1n }]);
      await anchorEpoch(substrate, "n2", tree.root);
      await expect(claimTx(substrate, bob, tree, 0, "n2"))
        .to.emit(substrate, "Withdrawn").withArgs(bob.address, N, bob.address, 1_234n)
        .and.not.to.emit(substrate, "PayDeferred");
      expect(await nr.balanceOf(bob.address)).to.equal(1_234n);
      expect(await substrate.owed(bob.address, N)).to.equal(0n);
      // Treasury share: 12_500 + 50 + 25 from the payment and the two payouts.
      expect(await substrate.treasuryOwed(N)).to.equal(12_575n);
      await substrate.withdrawTreasury(N);
      expect(await nr.balanceOf(deployer.address)).to.equal(12_575n);
    });
  });

  describe("a token removed from the index", function () {
    it("refuses new payments, opens and top-ups; an open channel still settles and refunds", async function () {
      const provider = signers[7];
      await registerAgent(substrate, provider, "prov");
      const Ch = await ethers.getContractFactory("PaymentChannel");
      const channel = await Ch.deploy(S, 3600);
      await channel.waitForDeployment();
      await fund(usd, alice, channel, 10_000n);
      await channel.connect(alice).openChannel(provider.address, U, 5_000n);
      const id = await channel.channelCount();

      await setParity(substrate, usd, 0n);
      expect(await substrate.parityOf(U)).to.equal(0n);
      expect(await substrate.acceptsToken(U)).to.equal(false);

      await fund(usd, payer, substrate, 1_000n);
      await expect(substrate.connect(payer).payForService(U, recipient.address, 1_000n, digest("x")))
        .to.be.revertedWithCustomError(substrate, "NotInIndex");
      await expect(channel.connect(alice).openChannel(provider.address, U, 1_000n))
        .to.be.revertedWithCustomError(channel, "TokenNotAccepted");
      await expect(channel.connect(alice).topUp(id, 1_000n))
        .to.be.revertedWithCustomError(channel, "TokenNotAccepted");

      // The open channel still pays the provider net and Substrate the fee.
      await channel.connect(provider).settle(id, 2_000n, await signVoucher(channel, alice, id, 2_000n), 2_000n);
      expect(await usd.balanceOf(provider.address)).to.equal(1_950n);
      expect(await usd.balanceOf(S)).to.equal(50n);
      // And the client's close refunds the rest after the window.
      await channel.connect(alice).startClose(id);
      await time.increase(3601);
      await channel.withdrawRemainder(id);
      expect(await usd.balanceOf(alice.address)).to.equal(10_000n - 2_000n);
      // Earners keep withdrawing what they are owed in it.
      expect(await substrate.treasuryOwed(U)).to.equal(25n);
      await substrate.withdrawTreasury(U);
    });
  });

  describe("renounceUpgrades", function () {
    it("is refused outside an upgrade, works inside upgradeToAndCall, then upgrades revert", async function () {
      await expect(substrate.connect(stranger).renounceUpgrades())
        .to.be.revertedWithCustomError(substrate, "NotGovernor");
      await expect(substrate.connect(gov).renounceUpgrades())
        .to.be.revertedWithCustomError(substrate, "NotDuringUpgrade");
      expect(await substrate.upgradesRenounced()).to.equal(false);

      const V2 = await ethers.getContractFactory("SubstrateV2Mock");
      const v2 = await V2.deploy();
      await v2.waitForDeployment();
      const call = substrate.interface.encodeFunctionData("renounceUpgrades", []);
      await expect(substrate.connect(gov).upgradeToAndCall(await v2.getAddress(), call))
        .to.emit(substrate, "UpgradesRenounced");
      expect(await substrate.upgradesRenounced()).to.equal(true);
      expect(await V2.attach(S).version2()).to.equal(2n);

      const v3 = await V2.deploy();
      await v3.waitForDeployment();
      await expect(substrate.connect(gov).upgradeToAndCall(await v3.getAddress(), "0x"))
        .to.be.revertedWithCustomError(substrate, "UpgradesAreRenounced");
      // The flag does not outlive its transaction.
      await expect(substrate.connect(gov).renounceUpgrades())
        .to.be.revertedWithCustomError(substrate, "NotDuringUpgrade");
    });
  });
});

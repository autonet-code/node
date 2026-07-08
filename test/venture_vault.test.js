const { expect } = require("chai");
const { ethers } = require("hardhat");
const { time } = require("@nomicfoundation/hardhat-network-helpers");

// ---------------------------------------------------------------------------
// Helpers (VentureVault uses ATN = Substrate.sol as the only currency, so we
// mint ATN through the substrate's only mint path: recordTrainingForEpoch over
// an anchored epoch with a single-leaf merkle root, then distribute by transfer).
// ---------------------------------------------------------------------------

const digest = (s) => ethers.keccak256(ethers.toUtf8Bytes(s));

// Substrate's service fee (fee-recycled emission) is taken from every
// payForService at SERVICE_FEE_BPS. Revenue tests want exact round
// amounts landing at the vault, so gross-up: smallest g whose net
// (g - fee(g)) equals the desired amount.
function grossFor(net) {
  let g = (net * 10000n) / 9750n;
  while (g - (g * 250n) / 10000n < net) g += 1n;
  return g;
}

async function deploySubstrate() {
  const Substrate = await ethers.getContractFactory("Substrate");
  const s = await Substrate.deploy((await ethers.getSigners())[0].address, "0x0000000000000000000000000000000000000000");
  await s.waitForDeployment();
  return s;
}

async function registerAgent(substrate, signer, lineage) {
  return substrate
    .connect(signer)
    .registerAgent(digest(lineage), ethers.toUtf8Bytes("peer-" + lineage));
}

// EIP-712 OwnerBinding{agent, parent, nonce} signed by `owner` for Substrate.
async function signBinding(substrate, ownerSigner, agent, parent, nonce) {
  const net = await ethers.provider.getNetwork();
  const domain = {
    name: "AutonetSubstrate",
    version: "1",
    chainId: net.chainId,
    verifyingContract: await substrate.getAddress(),
  };
  const types = {
    OwnerBinding: [
      { name: "agent", type: "address" },
      { name: "parent", type: "address" },
      { name: "nonce", type: "uint256" },
    ],
  };
  if (nonce === undefined) nonce = await substrate.bindingNonce(agent);
  return ownerSigner.signTypedData(domain, types, { agent, parent, nonce });
}

// Register `agentSigner` bound to `ownerWallet` (top-level).
async function registerBound(substrate, agentSigner, ownerWallet, lineage) {
  const sig = await signBinding(
    substrate,
    ownerWallet,
    agentSigner.address,
    ethers.ZeroAddress
  );
  await substrate
    .connect(agentSigner)
    .registerAgentBound(
      digest(lineage),
      ethers.toUtf8Bytes("peer-" + lineage),
      ownerWallet.address,
      ethers.ZeroAddress,
      sig
    );
}

// The single-leaf merkle leaf Substrate.recordTrainingForEpoch expects.
function mintLeaf(agentAddr, amount) {
  const inner = ethers.keccak256(
    ethers.AbiCoder.defaultAbiCoder().encode(
      ["address", "uint256"],
      [agentAddr, amount]
    )
  );
  return ethers.keccak256(ethers.solidityPacked(["bytes32"], [inner]));
}

// Mint `amount` ATN to `faucetSigner` (a registered agent) via one anchored
// epoch, then it is free to transfer. Returns nothing; balance lands on faucet.
let _epochCounter = 0;
async function mintATN(substrate, submitterSigner, faucetSigner, amount) {
  _epochCounter += 1;
  const epochId = "epoch-" + _epochCounter;
  const root = mintLeaf(faucetSigner.address, amount);
  const prevEpochRoot = await substrate.latestEpochRoot();
  const prevAnchorHash = await substrate.latestAnchorHash();
  await substrate
    .connect(submitterSigner)
    .submitAnchor(
      epochId,
      digest("root-" + epochId),
      prevEpochRoot,
      prevAnchorHash,
      "cid-" + epochId,
      digest("payload-" + epochId),
      root
    );
  const epochIdHash = digest(epochId);
  await substrate
    .connect(faucetSigner)
    .recordTrainingForEpoch(amount, epochIdHash, []);
}

// Fund `to` with `amount` ATN by minting to the faucet and transferring.
async function fundATN(substrate, submitter, faucet, to, amount) {
  await mintATN(substrate, submitter, faucet, amount);
  await substrate.connect(faucet).transfer(to, amount);
}

describe("VentureVault", function () {
  let substrate, factory;
  let deployer, ventureOwner, ventureAgent, faucet;
  let b1, b2, b3; // backers
  let v1o, v2o, v3o; // verifier owner wallets
  let v1, v2, v3, v4; // verifier agents
  let timelockW, stranger;
  const ZERO = ethers.ZeroAddress;

  const RAISE_MIN = 1000n;
  const RAISE_MAX = 10_000n;
  const TERMS_BPS = 3000; // 30% of net revenue to backers
  const PLANNED_TRANCHES = 4n;
  const PERIOD = 86400n; // 1 day = "epoch duration"
  const QUORUM = 2n;

  async function deployVault(overrides = {}) {
    const now = await time.latest();
    const trialDeadline = overrides.trialDeadline ?? BigInt(now) + 7n * PERIOD;
    const Vault = await ethers.getContractFactory("VentureVault");
    const vault = await Vault.deploy(
      await substrate.getAddress(),
      ventureAgent.address,
      overrides.prospectus ?? digest("prospectus-1"),
      overrides.termsBps ?? TERMS_BPS,
      overrides.raiseMin ?? RAISE_MIN,
      overrides.raiseMax ?? RAISE_MAX,
      overrides.plannedTranches ?? PLANNED_TRANCHES,
      overrides.period ?? PERIOD,
      overrides.quorum ?? QUORUM,
      trialDeadline,
      overrides.timelock ?? ZERO
    );
    await vault.waitForDeployment();
    return vault;
  }

  // approve + contribute helper
  async function contribute(vault, backer, amount) {
    await substrate.connect(backer).approve(await vault.getAddress(), amount);
    await vault.connect(backer).contribute(amount);
  }

  beforeEach(async function () {
    [
      deployer,
      ventureOwner,
      ventureAgent,
      faucet,
      b1,
      b2,
      b3,
      v1o,
      v2o,
      v3o,
      v1,
      v2,
      v3,
      v4,
      timelockW,
      stranger,
    ] = await ethers.getSigners();

    substrate = await deploySubstrate();

    // Faucet is a plain registered agent used only to mint & distribute ATN.
    await registerAgent(substrate, faucet, "faucet");

    // Venture agent: registered + owner-bound (VentureVault requires a set owner).
    await registerBound(substrate, ventureAgent, ventureOwner, "venture");

    // Verifier agents, each bound to a DISTINCT owner wallet.
    await registerBound(substrate, v1, v1o, "verifier1");
    await registerBound(substrate, v2, v2o, "verifier2");
    await registerBound(substrate, v3, v3o, "verifier3");
    // v4 is bound to the SAME owner as v1 (sock-puppet).
    await registerBound(substrate, v4, v1o, "verifier4");

    const Factory = await ethers.getContractFactory("VentureVaultFactory");
    factory = await Factory.deploy(await substrate.getAddress());
    await factory.waitForDeployment();

    // Pre-fund the three backers with ATN.
    await fundATN(substrate, faucet, faucet, b1.address, 20_000n);
    await fundATN(substrate, faucet, faucet, b2.address, 20_000n);
    await fundATN(substrate, faucet, faucet, b3.address, 20_000n);
  });

  // ==========================================================================
  // Construction / config validation
  // ==========================================================================

  describe("construction", function () {
    it("stores config and the venture agent's owner", async function () {
      const vault = await deployVault();
      expect(await vault.agent()).to.equal(ventureAgent.address);
      expect(await vault.agentOwner()).to.equal(ventureOwner.address);
      expect(await vault.termsBps()).to.equal(TERMS_BPS);
      expect(await vault.raiseMin()).to.equal(RAISE_MIN);
      expect(await vault.raiseMax()).to.equal(RAISE_MAX);
      expect(await vault.plannedTranches()).to.equal(PLANNED_TRANCHES);
      expect(await vault.tranchePeriod()).to.equal(PERIOD);
      expect(await vault.greenlightQuorum()).to.equal(QUORUM);
      expect(await vault.phase()).to.equal(0); // Fundraising
      expect(await vault.atn()).to.equal(await substrate.getAddress());
    });

    it("reverts on unregistered agent", async function () {
      const Vault = await ethers.getContractFactory("VentureVault");
      const now = await time.latest();
      await expect(
        Vault.deploy(
          await substrate.getAddress(),
          stranger.address, // not registered
          digest("p"),
          TERMS_BPS,
          RAISE_MIN,
          RAISE_MAX,
          PLANNED_TRANCHES,
          PERIOD,
          QUORUM,
          BigInt(now) + PERIOD,
          ZERO
        )
      ).to.be.revertedWithCustomError(Vault, "AgentNotRegistered");
    });

    it("reverts when the agent has no bound owner", async function () {
      // faucet is registered but ownerless.
      const Vault = await ethers.getContractFactory("VentureVault");
      const now = await time.latest();
      await expect(
        Vault.deploy(
          await substrate.getAddress(),
          faucet.address,
          digest("p"),
          TERMS_BPS,
          RAISE_MIN,
          RAISE_MAX,
          PLANNED_TRANCHES,
          PERIOD,
          QUORUM,
          BigInt(now) + PERIOD,
          ZERO
        )
      ).to.be.revertedWithCustomError(Vault, "AgentOwnerUnset");
    });

    it("reverts on bad terms / bounds / tranches / period / quorum / deadline", async function () {
      const Vault = await ethers.getContractFactory("VentureVault");
      const now = await time.latest();
      const base = [
        await substrate.getAddress(),
        ventureAgent.address,
        digest("p"),
      ];
      const okTail = [
        RAISE_MIN,
        RAISE_MAX,
        PLANNED_TRANCHES,
        PERIOD,
        QUORUM,
        BigInt(now) + PERIOD,
        ZERO,
      ];
      // termsBps > 10000
      await expect(
        Vault.deploy(...base, 10_001, ...okTail)
      ).to.be.revertedWithCustomError(Vault, "BadTerms");
      // raiseMax < raiseMin
      await expect(
        Vault.deploy(...base, TERMS_BPS, 5000n, 4000n, PLANNED_TRANCHES, PERIOD, QUORUM, BigInt(now) + PERIOD, ZERO)
      ).to.be.revertedWithCustomError(Vault, "BadRaiseBounds");
      // zero tranches
      await expect(
        Vault.deploy(...base, TERMS_BPS, RAISE_MIN, RAISE_MAX, 0n, PERIOD, QUORUM, BigInt(now) + PERIOD, ZERO)
      ).to.be.revertedWithCustomError(Vault, "BadTranches");
      // zero period
      await expect(
        Vault.deploy(...base, TERMS_BPS, RAISE_MIN, RAISE_MAX, PLANNED_TRANCHES, 0n, QUORUM, BigInt(now) + PERIOD, ZERO)
      ).to.be.revertedWithCustomError(Vault, "BadPeriod");
      // zero quorum
      await expect(
        Vault.deploy(...base, TERMS_BPS, RAISE_MIN, RAISE_MAX, PLANNED_TRANCHES, PERIOD, 0n, BigInt(now) + PERIOD, ZERO)
      ).to.be.revertedWithCustomError(Vault, "BadQuorum");
      // deadline in past
      await expect(
        Vault.deploy(...base, TERMS_BPS, RAISE_MIN, RAISE_MAX, PLANNED_TRANCHES, PERIOD, QUORUM, BigInt(now), ZERO)
      ).to.be.revertedWithCustomError(Vault, "DeadlineInPast");
    });

    it("factory deploys a vault and emits VentureCreated", async function () {
      const now = await time.latest();
      await expect(
        factory.createVenture(
          ventureAgent.address,
          digest("prospectus-fac"),
          TERMS_BPS,
          RAISE_MIN,
          RAISE_MAX,
          PLANNED_TRANCHES,
          PERIOD,
          QUORUM,
          BigInt(now) + PERIOD,
          ZERO
        )
      ).to.emit(factory, "VentureCreated");
    });
  });

  // ==========================================================================
  // Contribution / raise
  // ==========================================================================

  describe("contribution", function () {
    it("accepts contributions, tracks weight, escrows ATN", async function () {
      const vault = await deployVault();
      await contribute(vault, b1, 600n);
      await contribute(vault, b2, 400n);
      expect(await vault.raised()).to.equal(1000n);
      expect(await vault.contribution(b1.address)).to.equal(600n);
      expect(await vault.contribution(b2.address)).to.equal(400n);
      expect(await substrate.balanceOf(await vault.getAddress())).to.equal(1000n);
    });

    it("caps at raiseMax", async function () {
      const vault = await deployVault();
      await contribute(vault, b1, RAISE_MAX);
      await substrate.connect(b2).approve(await vault.getAddress(), 1n);
      await expect(vault.connect(b2).contribute(1n)).to.be.revertedWithCustomError(
        vault,
        "RaiseMaxExceeded"
      );
    });

    it("rejects zero amount and post-deadline contributions", async function () {
      const vault = await deployVault();
      await expect(vault.connect(b1).contribute(0n)).to.be.revertedWithCustomError(
        vault,
        "ZeroAmount"
      );
      await time.increase(8n * PERIOD);
      await substrate.connect(b1).approve(await vault.getAddress(), 100n);
      await expect(vault.connect(b1).contribute(100n)).to.be.revertedWithCustomError(
        vault,
        "DeadlinePassed"
      );
    });
  });

  // ==========================================================================
  // Under-min refund
  // ==========================================================================

  describe("raise failure (under min)", function () {
    it("below raiseMin at deadline → everyone withdraws fully", async function () {
      const vault = await deployVault();
      await contribute(vault, b1, 300n);
      await contribute(vault, b2, 200n); // total 500 < 1000 min

      await time.increase(8n * PERIOD);
      await expect(vault.failRaise())
        .to.emit(vault, "RaiseFailed")
        .withArgs(500n, RAISE_MIN);
      expect(await vault.phase()).to.equal(3); // Refunding

      const b1Before = await substrate.balanceOf(b1.address);
      await vault.connect(b1).refund();
      expect(await substrate.balanceOf(b1.address)).to.equal(b1Before + 300n);
      await vault.connect(b2).refund();
      expect(await vault.contribution(b1.address)).to.equal(0n);
      // vault drained back to backers
      expect(await substrate.balanceOf(await vault.getAddress())).to.equal(0n);
    });

    it("failRaise reverts if raiseMin was met", async function () {
      const vault = await deployVault();
      await contribute(vault, b1, RAISE_MIN);
      await time.increase(8n * PERIOD);
      await expect(vault.failRaise()).to.be.revertedWithCustomError(vault, "WrongPhase");
    });

    it("failRaise reverts before the deadline", async function () {
      const vault = await deployVault();
      await contribute(vault, b1, 100n);
      await expect(vault.failRaise()).to.be.revertedWithCustomError(
        vault,
        "DeadlineNotReached"
      );
    });
  });

  // ==========================================================================
  // Verifier trials + greenlight
  // ==========================================================================

  describe("verifier trials", function () {
    async function raised(vault) {
      await contribute(vault, b1, 6000n);
      await contribute(vault, b2, 4000n); // 10000 total, >= min
      await vault.openTrials();
    }

    it("distinct-owner passes greenlight; verifier fee = 1% split equally", async function () {
      const vault = await deployVault();
      await raised(vault); // 10000 raised

      await expect(vault.connect(v1).attestTrial(true, digest("r1")))
        .to.emit(vault, "TrialAttested")
        .withArgs(v1.address, v1o.address, true, digest("r1"));
      await vault.connect(v2).attestTrial(true, digest("r2"));

      await expect(vault.greenlight())
        .to.emit(vault, "Greenlit");
      expect(await vault.phase()).to.equal(2); // Active
      // 1% of 10000 = 100, split across 2 verifiers = 50 each.
      expect(await vault.verifierFeePool()).to.equal(100n);
      expect(await vault.verifierFeeSlice()).to.equal(50n);

      // verifiers claim their fee
      const before = await substrate.balanceOf(v1.address);
      await vault.connect(v1).claimVerifierFee();
      expect(await substrate.balanceOf(v1.address)).to.equal(before + 50n);
      await vault.connect(v2).claimVerifierFee();
      // double claim rejected
      await expect(vault.connect(v1).claimVerifierFee()).to.be.revertedWithCustomError(
        vault,
        "FeeAlreadyClaimed"
      );
    });

    it("sock-puppet: a second verifier under the SAME owner is rejected as a pass", async function () {
      const vault = await deployVault();
      await raised(vault);
      await vault.connect(v1).attestTrial(true, digest("r1")); // owner v1o
      // v4 shares owner v1o → its PASS is rejected (distinct-owner).
      await expect(
        vault.connect(v4).attestTrial(true, digest("r4"))
      ).to.be.revertedWithCustomError(vault, "VerifierOwnerNotDistinct");
      expect(await vault.passerCount()).to.equal(1n);
    });

    it("self-verify: an agent under the venture owner is rejected", async function () {
      const vault = await deployVault();
      await raised(vault);
      // Register a verifier agent bound to the VENTURE owner.
      await registerBound(substrate, stranger, ventureOwner, "self-verifier");
      await expect(
        vault.connect(stranger).attestTrial(true, digest("self"))
      ).to.be.revertedWithCustomError(vault, "VerifierIsVentureOwner");
    });

    it("unregistered verifier rejected", async function () {
      const vault = await deployVault();
      await raised(vault);
      // deployer is not a registered agent.
      await expect(
        vault.connect(deployer).attestTrial(true, digest("x"))
      ).to.be.revertedWithCustomError(vault, "VerifierNotRegistered");
    });

    it("ownerless verifier rejected (VerifierOwnerUnset)", async function () {
      const vault = await deployVault();
      await raised(vault);
      // register a bare ownerless agent
      await registerAgent(substrate, stranger, "ownerless-verifier");
      await expect(
        vault.connect(stranger).attestTrial(true, digest("x"))
      ).to.be.revertedWithCustomError(vault, "VerifierOwnerUnset");
    });

    it("greenlight reverts below quorum; failTrials refunds after deadline", async function () {
      const vault = await deployVault();
      await raised(vault);
      await vault.connect(v1).attestTrial(true, digest("r1")); // only 1 pass, quorum 2
      await expect(vault.greenlight()).to.be.revertedWithCustomError(
        vault,
        "QuorumNotMet"
      );
      await time.increase(8n * PERIOD);
      await expect(vault.failTrials()).to.emit(vault, "RaiseFailed");
      expect(await vault.phase()).to.equal(3); // Refunding
      // full refund of contributions
      const b1Before = await substrate.balanceOf(b1.address);
      await vault.connect(b1).refund();
      expect(await substrate.balanceOf(b1.address)).to.equal(b1Before + 6000n);
    });

    it("a FAIL verdict does not count toward quorum", async function () {
      const vault = await deployVault();
      await raised(vault);
      await vault.connect(v1).attestTrial(false, digest("fail"));
      await vault.connect(v2).attestTrial(true, digest("pass"));
      expect(await vault.passerCount()).to.equal(1n);
      await expect(vault.greenlight()).to.be.revertedWithCustomError(vault, "QuorumNotMet");
    });

    it("cannot attest twice", async function () {
      const vault = await deployVault();
      await raised(vault);
      await vault.connect(v1).attestTrial(true, digest("r1"));
      await expect(
        vault.connect(v1).attestTrial(true, digest("r1b"))
      ).to.be.revertedWithCustomError(vault, "AlreadyAttested");
    });
  });

  // ==========================================================================
  // Tranches
  // ==========================================================================

  describe("tranche drip", function () {
    // Sets up a greenlit vault with `total` raised. Returns the vault.
    async function greenlit(total = 10_000n) {
      const vault = await deployVault();
      await contribute(vault, b1, (total * 6n) / 10n);
      await contribute(vault, b2, total - (total * 6n) / 10n);
      await vault.openTrials();
      await vault.connect(v1).attestTrial(true, digest("r1"));
      await vault.connect(v2).attestTrial(true, digest("r2"));
      await vault.greenlight();
      return vault;
    }

    it("first tranche claimable immediately, then one per period", async function () {
      const vault = await greenlit(10_000n);
      // pool = 10000 - 100 fee = 9900; perTranche = 9900/4 = 2475
      expect(await vault.tranchePool()).to.equal(9900n);
      expect(await vault.tranchesVested()).to.equal(1n);

      const agentBefore = await substrate.balanceOf(ventureAgent.address);
      await vault.connect(ventureAgent).claimTranche();
      expect(await substrate.balanceOf(ventureAgent.address)).to.equal(
        agentBefore + 2475n
      );
      expect(await vault.tranchesClaimed()).to.equal(1n);

      // no new tranche yet
      await expect(
        vault.connect(ventureAgent).claimTranche()
      ).to.be.revertedWithCustomError(vault, "NoTrancheDue");

      // advance 1 period → tranche 2
      await time.increase(PERIOD);
      expect(await vault.tranchesVested()).to.equal(2n);
      await vault.connect(ventureAgent).claimTranche();
      expect(await vault.tranchesClaimed()).to.equal(2n);

      // advance past the end → all 4 vested, last absorbs remainder
      await time.increase(10n * PERIOD);
      expect(await vault.tranchesVested()).to.equal(4n);
      await vault.connect(ventureAgent).claimTranche();
      expect(await vault.tranchesClaimed()).to.equal(4n);
      // agent got the whole pool
      expect(await substrate.balanceOf(ventureAgent.address)).to.equal(
        agentBefore + 9900n
      );
      expect(await vault.undrippedTranchePrincipal()).to.equal(0n);
    });

    it("only the agent can claim tranches", async function () {
      const vault = await greenlit();
      await expect(
        vault.connect(stranger).claimTranche()
      ).to.be.revertedWithCustomError(vault, "NotAgent");
    });
  });

  // ==========================================================================
  // Halt vote math (incl. the 1/3 boundary)
  // ==========================================================================

  describe("halt vote", function () {
    async function greenlitThree() {
      // three backers so we can hit fractions cleanly: 3000/3000/3000 = 9000
      const vault = await deployVault({ raiseMin: 1000n, raiseMax: 20_000n });
      await contribute(vault, b1, 3000n);
      await contribute(vault, b2, 3000n);
      await contribute(vault, b3, 3000n);
      await vault.openTrials();
      await vault.connect(v1).attestTrial(true, digest("r1"));
      await vault.connect(v2).attestTrial(true, digest("r2"));
      await vault.greenlight();
      return vault; // raised = 9000
    }

    it("exactly 1/3 of raised weight halts (boundary: >= 1/3)", async function () {
      const vault = await greenlitThree(); // raised 9000, 1/3 = 3000
      // one backer with 3000 == exactly 1/3 → halts (>= boundary).
      await expect(vault.connect(b1).setHaltVote(true)).to.emit(vault, "Halted");
      expect(await vault.halted()).to.equal(true);
      expect(await vault.haltWeight()).to.equal(3000n);
    });

    it("just under 1/3 does NOT halt", async function () {
      // raised 9001 so 1/3 = 3000.33; weight 3000 < that.
      const vault = await deployVault({ raiseMin: 1000n, raiseMax: 20_000n });
      await contribute(vault, b1, 3000n);
      await contribute(vault, b2, 3000n);
      await contribute(vault, b3, 3001n); // raised 9001
      await vault.openTrials();
      await vault.connect(v1).attestTrial(true, digest("r1"));
      await vault.connect(v2).attestTrial(true, digest("r2"));
      await vault.greenlight();

      await vault.connect(b1).setHaltVote(true); // weight 3000, 3000*3=9000 < 9001
      expect(await vault.halted()).to.equal(false);
      // add b2 → 6000, halts
      await expect(vault.connect(b2).setHaltVote(true)).to.emit(vault, "Halted");
      expect(await vault.halted()).to.equal(true);
    });

    it("symmetric un-halt: dropping below 1/3 un-halts", async function () {
      const vault = await greenlitThree();
      await vault.connect(b1).setHaltVote(true); // 3000 == 1/3 → halted
      expect(await vault.halted()).to.equal(true);
      await expect(vault.connect(b1).setHaltVote(false)).to.emit(vault, "Unhalted");
      expect(await vault.halted()).to.equal(false);
      expect(await vault.haltWeight()).to.equal(0n);
    });

    it("halt freezes drip: no new tranche vests while halted", async function () {
      const vault = await greenlitThree(); // raised 9000, pool 8910
      // claim tranche 1 at period 0
      await vault.connect(ventureAgent).claimTranche();
      expect(await vault.tranchesClaimed()).to.equal(1n);

      // halt during period 0 (haltPeriodVested snapshots to 1)
      await vault.connect(b1).setHaltVote(true);
      expect(await vault.halted()).to.equal(true);

      // advance several periods; drip is frozen at the halt boundary.
      await time.increase(5n * PERIOD);
      await expect(
        vault.connect(ventureAgent).claimTranche()
      ).to.be.revertedWithCustomError(vault, "NoTrancheDue");

      // un-halt → drip resumes; vested catches up.
      await vault.connect(b1).setHaltVote(false);
      expect(await vault.tranchesVested()).to.equal(4n);
      await vault.connect(ventureAgent).claimTranche();
      expect(await vault.tranchesClaimed()).to.equal(4n);
    });

    it("non-backer cannot vote", async function () {
      const vault = await greenlitThree();
      await expect(
        vault.connect(stranger).setHaltVote(true)
      ).to.be.revertedWithCustomError(vault, "NotABacker");
    });
  });

  // ==========================================================================
  // Revenue split + pro-rata claims (multiple backers, multiple drops)
  // ==========================================================================

  describe("revenue split", function () {
    async function greenlitVault() {
      // b1: 6000, b2: 4000 → raised 10000, termsBps 30%
      const vault = await deployVault();
      await contribute(vault, b1, 6000n);
      await contribute(vault, b2, 4000n);
      await vault.openTrials();
      await vault.connect(v1).attestTrial(true, digest("r1"));
      await vault.connect(v2).attestTrial(true, digest("r2"));
      await vault.greenlight();
      return vault;
    }

    // Revenue lands via payForService (a labeled transfer to the vault).
    // Grossed-up so the vault receives exactly `amount` net of the fee.
    async function dropRevenue(vault, amount) {
      const gross = grossFor(amount);
      await fundATN(substrate, faucet, faucet, faucet.address, gross);
      await substrate
        .connect(faucet)
        .payForService(await vault.getAddress(), gross, digest("req"));
    }

    it("splits termsBps to backers pro-rata, remainder to agent; multiple drops", async function () {
      const vault = await greenlitVault();

      // Drop 1: 1000 revenue → 300 to backers (30%), 700 to agent.
      await dropRevenue(vault, 1000n);
      await expect(vault.claimRevenue())
        .to.emit(vault, "RevenueSplit")
        .withArgs(1000n, 300n, 700n);

      // backer shares of the 300: b1 6000/10000 = 180, b2 = 120.
      expect(await vault.pendingBackerRevenue(b1.address)).to.equal(180n);
      expect(await vault.pendingBackerRevenue(b2.address)).to.equal(120n);

      // Drop 2: 500 revenue → 150 backers, 350 agent (accumulates).
      await dropRevenue(vault, 500n);
      await vault.claimRevenue();
      // b1 total 180 + 90 = 270; b2 120 + 60 = 180.
      expect(await vault.pendingBackerRevenue(b1.address)).to.equal(270n);
      expect(await vault.pendingBackerRevenue(b2.address)).to.equal(180n);

      // Backers pull.
      const b1Before = await substrate.balanceOf(b1.address);
      await vault.connect(b1).claimBackerRevenue();
      expect(await substrate.balanceOf(b1.address)).to.equal(b1Before + 270n);
      const b2Before = await substrate.balanceOf(b2.address);
      await vault.connect(b2).claimBackerRevenue();
      expect(await substrate.balanceOf(b2.address)).to.equal(b2Before + 180n);

      // Agent pulls its 700 + 350 = 1050.
      const agentBefore = await substrate.balanceOf(ventureAgent.address);
      await vault.connect(ventureAgent).claimAgentRevenue();
      expect(await substrate.balanceOf(ventureAgent.address)).to.equal(
        agentBefore + 1050n
      );
    });

    it("revenue does not touch tranche principal", async function () {
      const vault = await greenlitVault();
      // pool = 9900. Drop revenue then claim a tranche; both are independent.
      await dropRevenue(vault, 1000n);
      await vault.claimRevenue();

      // agent can still claim the first tranche (2475) fully.
      const agentBefore = await substrate.balanceOf(ventureAgent.address);
      await vault.connect(ventureAgent).claimTranche();
      expect(await substrate.balanceOf(ventureAgent.address)).to.equal(
        agentBefore + 2475n
      );
      // and its revenue share still owed separately.
      await vault.connect(ventureAgent).claimAgentRevenue();
      expect(await substrate.balanceOf(ventureAgent.address)).to.equal(
        agentBefore + 2475n + 700n
      );
    });

    it("claimRevenue reverts when nothing new arrived", async function () {
      const vault = await greenlitVault();
      await expect(vault.claimRevenue()).to.be.revertedWithCustomError(
        vault,
        "NothingToClaim"
      );
    });

    it("backer with no pending revenue reverts on claim", async function () {
      const vault = await greenlitVault();
      await dropRevenue(vault, 1000n);
      await vault.claimRevenue();
      await vault.connect(b1).claimBackerRevenue();
      await expect(
        vault.connect(b1).claimBackerRevenue()
      ).to.be.revertedWithCustomError(vault, "NothingToClaim");
    });
  });

  // ==========================================================================
  // DAO veto
  // ==========================================================================

  describe("dao veto", function () {
    async function greenlitWithTimelock() {
      const vault = await deployVault({ timelock: timelockW.address, raiseMax: 20_000n });
      await contribute(vault, b1, 6000n);
      await contribute(vault, b2, 4000n); // raised 10000
      await vault.openTrials();
      await vault.connect(v1).attestTrial(true, digest("r1"));
      await vault.connect(v2).attestTrial(true, digest("r2"));
      await vault.greenlight();
      return vault;
    }

    it("timelock veto → permanent halt + pro-rata refund of UNSPENT raise", async function () {
      const vault = await greenlitWithTimelock(); // pool 9900
      // agent drips tranche 1 (2475) → spent.
      await vault.connect(ventureAgent).claimTranche();
      expect(await vault.trancheDistributed()).to.equal(2475n);

      // veto
      await expect(vault.connect(timelockW).daoVeto()).to.emit(vault, "Vetoed");
      expect(await vault.phase()).to.equal(4); // Vetoed
      expect(await vault.halted()).to.equal(true);

      // Refundable = unspent tranche principal = pool - distributed = 9900-2475 = 7425.
      // Split pro-rata by contribution: b1 6000/10000 → 4455, b2 4000/10000 → 2970.
      const b1Before = await substrate.balanceOf(b1.address);
      await vault.connect(b1).refund();
      expect(await substrate.balanceOf(b1.address)).to.equal(b1Before + 4455n);
      const b2Before = await substrate.balanceOf(b2.address);
      await vault.connect(b2).refund();
      expect(await substrate.balanceOf(b2.address)).to.equal(b2Before + 2970n);
    });

    it("veto never touches distributed revenue", async function () {
      const vault = await greenlitWithTimelock();
      // drop revenue (grossed-up past the service fee), split, so
      // agent+backers are owed.
      const gross = grossFor(1000n);
      await fundATN(substrate, faucet, faucet, faucet.address, gross);
      await substrate
        .connect(faucet)
        .payForService(await vault.getAddress(), gross, digest("req"));
      await vault.claimRevenue(); // 300 backers, 700 agent owed

      await vault.connect(timelockW).daoVeto();

      // Backer revenue claims still work post-veto.
      const b1Before = await substrate.balanceOf(b1.address);
      await vault.connect(b1).claimBackerRevenue(); // 180
      expect(await substrate.balanceOf(b1.address)).to.equal(b1Before + 180n);
      // Agent revenue still claimable.
      const agentBefore = await substrate.balanceOf(ventureAgent.address);
      await vault.connect(ventureAgent).claimAgentRevenue();
      expect(await substrate.balanceOf(ventureAgent.address)).to.equal(agentBefore + 700n);
    });

    it("only the timelock may veto; reverts when no timelock set", async function () {
      const withTl = await greenlitWithTimelock();
      await expect(withTl.connect(stranger).daoVeto()).to.be.revertedWithCustomError(
        withTl,
        "NotTimelock"
      );

      const noTl = await deployVault(); // timelock = 0
      await contribute(noTl, b1, RAISE_MIN);
      await expect(noTl.connect(timelockW).daoVeto()).to.be.revertedWithCustomError(
        noTl,
        "NoTimelock"
      );
    });

    it("veto during fundraising refunds the whole raise", async function () {
      const vault = await deployVault({ timelock: timelockW.address });
      await contribute(vault, b1, 500n);
      await contribute(vault, b2, 300n);
      await vault.connect(timelockW).daoVeto();
      expect(await vault.phase()).to.equal(4);
      // never greenlit → whole contribution refundable.
      const b1Before = await substrate.balanceOf(b1.address);
      await vault.connect(b1).refund();
      expect(await substrate.balanceOf(b1.address)).to.equal(b1Before + 500n);
    });
  });

  // ==========================================================================
  // Full lifecycle end to end
  // ==========================================================================

  it("full lifecycle: raise → trials → greenlight → tranches → revenue → claims", async function () {
    const vault = await deployVault();
    // raise
    await contribute(vault, b1, 6000n);
    await contribute(vault, b2, 4000n);
    expect(await vault.raised()).to.equal(10_000n);
    await vault.openTrials();
    // trials
    await vault.connect(v1).attestTrial(true, digest("r1"));
    await vault.connect(v2).attestTrial(true, digest("r2"));
    await vault.connect(v3).attestTrial(true, digest("r3"));
    await vault.greenlight();
    expect(await vault.phase()).to.equal(2);
    // fee: 100 / 3 = 33 each
    expect(await vault.verifierFeeSlice()).to.equal(33n);
    await vault.connect(v1).claimVerifierFee();
    await vault.connect(v2).claimVerifierFee();
    await vault.connect(v3).claimVerifierFee();

    // tranches (pool = 10000 - 100 = 9900 over 4)
    await time.increase(10n * PERIOD);
    await vault.connect(ventureAgent).claimTranche();
    expect(await vault.tranchesClaimed()).to.equal(4n);

    // revenue (grossed-up past the service fee)
    const gross = grossFor(2000n);
    await fundATN(substrate, faucet, faucet, faucet.address, gross);
    await substrate
      .connect(faucet)
      .payForService(await vault.getAddress(), gross, digest("req"));
    await vault.claimRevenue(); // 600 backers, 1400 agent
    await vault.connect(b1).claimBackerRevenue(); // 360
    await vault.connect(b2).claimBackerRevenue(); // 240
    await vault.connect(ventureAgent).claimAgentRevenue(); // 1400

    // vault drained down to at most rounding dust (verifier fee 100/3=33 each
    // leaves 1 wei un-split; equal-split dust stays in the vault by design).
    expect(await substrate.balanceOf(await vault.getAddress())).to.be.lessThanOrEqual(2n);
  });
});

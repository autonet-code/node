const { ethers } = require("hardhat");
const { time } = require("@nomicfoundation/hardhat-network-helpers");

const ZERO = "0x0000000000000000000000000000000000000000";
const ONE = "0x0000000000000000000000000000000000000001";
const digest = (s) => ethers.keccak256(ethers.toUtf8Bytes(s));

// Reserved signer slots (tests use indices 0..15).
//   16: fee payer for payFees, 17: fee recipient for payFees,
//   18: default Substrate governor (anchors; production uses the DAO
//       timelock).
const SLOT_POOL_PAYER = 16;
const SLOT_POOL_RECIPIENT = 17;
const SLOT_GOVERNOR = 18;

// substrate address (lowercase) -> { governor, registry }
const _ctx = {};
const ctxOf = async (substrate) =>
  _ctx[(await substrate.getAddress()).toLowerCase()] || {};

// The implementation behind every proxied Substrate: deployed directly
// with placeholder arguments, which also locks it against initialize().
// One per hardhat network state (a snapshot revert drops it, so it is
// looked up by code, not cached blindly).
let _impl = null;
async function substrateImplementation() {
  if (_impl && (await ethers.provider.getCode(await _impl.getAddress())) !== "0x") {
    return _impl;
  }
  const Substrate = await ethers.getContractFactory("Substrate");
  _impl = await Substrate.deploy(ONE, ONE, ZERO, ONE, 250, 5000);
  await _impl.waitForDeployment();
  return _impl;
}

async function deployRegistry() {
  const R = await ethers.getContractFactory("MockRegistry");
  const r = await R.deploy();
  await r.waitForDeployment();
  return r;
}

// Deploy Substrate(treasury, governor, anchorer, registry, serviceFeeBps,
// feeTreasuryBps). Defaults: treasury = signers[0], governor = signers[18],
// anchorer = none, registry = a fresh MockRegistry, fee 2.5%, half of it to
// the treasury. As a proxy unless `direct`.
async function deploySubstrateWith(opts = {}) {
  const signers = await ethers.getSigners();
  const treasury = opts.treasury !== undefined ? opts.treasury : signers[0].address;
  const governor =
    opts.governor !== undefined ? opts.governor : signers[SLOT_GOVERNOR].address;
  const anchorer = opts.anchorer !== undefined ? opts.anchorer : ZERO;
  const registry = opts.registry !== undefined
    ? opts.registry
    : await (await deployRegistry()).getAddress();
  const Substrate = await ethers.getContractFactory("Substrate");
  const args = [
    treasury, governor, anchorer, registry,
    opts.serviceFeeBps !== undefined ? opts.serviceFeeBps : 250,
    opts.feeTreasuryBps !== undefined ? opts.feeTreasuryBps : 5000];
  let substrate;
  if (opts.direct) {
    // A plain contract, set up by its constructor.
    substrate = await Substrate.deploy(...args);
    await substrate.waitForDeployment();
  } else {
    // As a jurisdiction gets it: an ERC-1967 proxy over the shared
    // implementation, set up by initialize() in the creating transaction.
    const impl = await substrateImplementation();
    const Proxy = await ethers.getContractFactory("SubstrateProxy");
    const proxy = await Proxy.deploy(
      await impl.getAddress(),
      Substrate.interface.encodeFunctionData("initialize", args));
    await proxy.waitForDeployment();
    substrate = Substrate.attach(await proxy.getAddress());
  }
  _ctx[(await substrate.getAddress()).toLowerCase()] = {
    governor:
      signers.find((x) => x.address.toLowerCase() === String(governor).toLowerCase()) || null,
  };
  return substrate;
}

async function deploySubstrate() {
  return deploySubstrateWith();
}

// The MockRegistry a Substrate reads its value index from.
async function registryOf(substrate) {
  return (await ethers.getContractFactory("MockRegistry")).attach(await substrate.registry());
}

// The parity that makes one whole token worth 1e6 value units.
const unitParity = (decimals) => 10n ** BigInt(36 - Number(decimals));

// Set a token's parity in the Substrate's index (0 removes it).
async function setParity(substrate, token, parity) {
  const reg = await registryOf(substrate);
  await (await reg.setParity(await token.getAddress(), parity)).wait();
}

// Deploy a MockIndexToken and put it in the Substrate's index (unless
// `parity` is 0). `kind` picks another mock: "cut", "pausable", "noreturn".
async function deployIndexToken(substrate, opts = {}) {
  const decimals = opts.decimals !== undefined ? opts.decimals : 6;
  let t;
  if (opts.kind === "cut") {
    t = await (await ethers.getContractFactory("CutToken")).deploy(opts.cutBps || 100);
  } else if (opts.kind === "pausable") {
    t = await (await ethers.getContractFactory("PausableToken")).deploy();
  } else if (opts.kind === "noreturn") {
    t = await (await ethers.getContractFactory("NoReturnToken")).deploy();
  } else {
    t = await (await ethers.getContractFactory("MockIndexToken"))
      .deploy(opts.name || "Index USD", opts.symbol || "IUSD", decimals);
  }
  await t.waitForDeployment();
  // The cut, pausable and no-return mocks all have 6 decimals.
  const d = opts.kind ? 6 : decimals;
  const parity = opts.parity !== undefined ? BigInt(opts.parity) : unitParity(d);
  if (substrate && parity !== 0n) await setParity(substrate, t, parity);
  return t;
}

// Mint `amount` of `token` to `owner` and approve `spender` for it.
async function fund(token, owner, spender, amount) {
  await (await token.mint(owner.address, amount)).wait();
  await (await token.connect(owner).approve(
    typeof spender === "string" ? spender : await spender.getAddress(), amount)).wait();
}

// Fee math, mirroring Substrate (fee rounded up).
function feeOf(gross, bps = 250n) {
  return (gross * bps + 9999n) / 10000n;
}
function splitFee(fee, treasuryBps = 5000n) {
  const toTreasury = (fee * treasuryBps) / 10000n;
  return { toTreasury, toolShare: fee - toTreasury };
}

// Pay a service in `token` from the reserved payer to the reserved
// recipient, so fees reach the pool. Returns the fee split.
async function payFees(substrate, token, gross = 1_000_000_000n) {
  const signers = await ethers.getSigners();
  const payer = signers[SLOT_POOL_PAYER];
  const recipient = signers[SLOT_POOL_RECIPIENT];
  await fund(token, payer, substrate, gross);
  await (await substrate.connect(payer).payForService(
    await token.getAddress(), recipient.address, gross, digest("seed"))).wait();
  const fee = feeOf(gross, await substrate.serviceFeeBps());
  return { fee, ...splitFee(fee, await substrate.feeTreasuryBps()) };
}

// ---------------------------------------------------------------------------
// Anchoring: governor or anchorer; the cutoff is after the latest one and
// strictly before the anchoring block.
// ---------------------------------------------------------------------------

// Anchor fields chained onto the current chain head. The cutoff is the
// head block's timestamp (the anchoring block comes later); a block is
// mined first when the head is not yet past the latest cutoff.
async function anchorFields(substrate, epochId, claimRoot) {
  const latest = await substrate.latestCutoff();
  let head = BigInt((await ethers.provider.getBlock("latest")).timestamp);
  if (head <= latest) {
    await time.increase(Number(latest - head) + 1);
    head = BigInt((await ethers.provider.getBlock("latest")).timestamp);
  }
  return {
    epochId,
    epochRoot: digest("root-" + epochId),
    prevEpochRoot: await substrate.latestEpochRoot(),
    prevAnchorHash: await substrate.latestAnchorHash(),
    claimCid: "cid-" + epochId,
    payloadHash: digest("payload-" + epochId),
    claimRoot: claimRoot || ethers.ZeroHash,
    cutoff: head,
  };
}

function anchorTx(substrate, sender, f) {
  return substrate
    .connect(sender)
    .anchor(
      f.epochId,
      f.epochRoot,
      f.prevEpochRoot,
      f.prevAnchorHash,
      f.claimCid,
      f.payloadHash,
      f.claimRoot,
      f.cutoff
    );
}

// Anchor `epochId` from `sender` (default: the deploy's governor).
async function anchorEpoch(substrate, epochId, claimRoot, sender) {
  const f = await anchorFields(substrate, epochId, claimRoot);
  const s = sender || (await ctxOf(substrate)).governor;
  if (!s) throw new Error("anchorEpoch: no governor signer for this substrate");
  await (await anchorTx(substrate, s, f)).wait();
  return f;
}

// ---------------------------------------------------------------------------
// Claims: one leaf per earner,
//   keccak256(bytes.concat(keccak256(abi.encode(
//       CLAIM_TAG, earner, tokens, amounts))))
// in a sorted-pair merkle tree.
// ---------------------------------------------------------------------------

const CLAIM_TAG = digest("autonet.claim");

function claimLeaf(earner, tokens, amounts) {
  const inner = ethers.keccak256(
    ethers.AbiCoder.defaultAbiCoder().encode(
      ["bytes32", "address", "address[]", "uint256[]"],
      [CLAIM_TAG, earner, tokens, amounts]
    )
  );
  return ethers.keccak256(inner);
}

function hashPair(a, b) {
  const [lo, hi] = BigInt(a) <= BigInt(b) ? [a, b] : [b, a];
  return ethers.keccak256(ethers.solidityPacked(["bytes32", "bytes32"], [lo, hi]));
}

// A basket in the order a leaf needs: tokens strictly ascending.
// `pairs` is [[tokenAddress, amount], ...].
function sortBasket(pairs) {
  const sorted = [...pairs].sort((x, y) =>
    BigInt(x[0]) < BigInt(y[0]) ? -1 : BigInt(x[0]) > BigInt(y[0]) ? 1 : 0);
  return { tokens: sorted.map((p) => p[0]), amounts: sorted.map((p) => p[1]) };
}

// Merkle tree over leaves; an odd node is carried up unchanged.
function merkleTree(leaves) {
  if (leaves.length === 0) throw new Error("merkleTree: no leaves");
  const layers = [leaves];
  while (layers[layers.length - 1].length > 1) {
    const prev = layers[layers.length - 1];
    const next = [];
    for (let i = 0; i < prev.length; i += 2) {
      next.push(i + 1 < prev.length ? hashPair(prev[i], prev[i + 1]) : prev[i]);
    }
    layers.push(next);
  }
  const proof = (index) => {
    const out = [];
    let idx = index;
    for (let l = 0; l < layers.length - 1; l++) {
      const sib = idx ^ 1;
      if (sib < layers[l].length) out.push(layers[l][sib]);
      idx >>= 1;
    }
    return out;
  };
  return { root: layers[layers.length - 1][0], proof };
}

// Build an epoch's claim tree. `entries` is [{earner, tokens,
// amounts}] (tokens already sorted, or give `basket` pairs instead).
// Returns { root, entries, proof(i) }.
function claimTree(entries) {
  const norm = entries.map((e) => {
    const b = e.basket ? sortBasket(e.basket) : { tokens: e.tokens, amounts: e.amounts };
    return { earner: e.earner, tokens: b.tokens, amounts: b.amounts };
  });
  const t = merkleTree(norm.map((e) => claimLeaf(e.earner, e.tokens, e.amounts)));
  return { root: t.root, entries: norm, proof: t.proof };
}

// Submit entry `i` of a claim tree for epoch `epochId`.
function claimTx(substrate, sender, tree, i, epochId) {
  const e = tree.entries[i];
  return substrate.connect(sender).claim(
    e.earner, digest(epochId), e.tokens, e.amounts, tree.proof(i));
}

// ---------------------------------------------------------------------------
// PaymentChannel vouchers: EIP-712 Voucher(channelId, token, cumulative).
// The token defaults to the channel's own.
// ---------------------------------------------------------------------------

async function signVoucher(channel, signer, channelId, cumulativeAmount, token) {
  const net = await ethers.provider.getNetwork();
  const domain = {
    name: "AutonetPaymentChannel",
    version: "1",
    chainId: net.chainId,
    verifyingContract: await channel.getAddress(),
  };
  const types = {
    Voucher: [
      { name: "channelId", type: "uint256" },
      { name: "token", type: "address" },
      { name: "cumulativeAmount", type: "uint256" },
    ],
  };
  if (token === undefined) {
    try {
      token = (await channel.getChannel(channelId)).token;
    } catch (e) {
      token = ZERO;
    }
  }
  return signer.signTypedData(domain, types, { channelId, token, cumulativeAmount });
}

// ---------------------------------------------------------------------------
// Owner binding and registration.
// ---------------------------------------------------------------------------

// EIP-712 OwnerBinding{agent, parent, nonce} for Substrate, signed by
// `signer` (the owner wallet, or the parent agent for a parent-signed child).
// `nonce` defaults to the agent's current on-chain binding nonce.
async function signBinding(substrate, signer, agent, parent = ZERO, nonce) {
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
  return signer.signTypedData(domain, types, { agent, parent, nonce });
}

// EIP-712 RevokeAgent(address agent), signed by the owner wallet.
async function signRevoke(substrate, ownerSigner, agent) {
  const net = await ethers.provider.getNetwork();
  const domain = {
    name: "AutonetSubstrate",
    version: "1",
    chainId: net.chainId,
    verifyingContract: await substrate.getAddress(),
  };
  const types = { RevokeAgent: [{ name: "agent", type: "address" }] };
  return ownerSigner.signTypedData(domain, types, { agent });
}

// Register `signer` through registerAgentBound (the only registration path).
// Owner defaults to the agent itself. The binding signature comes from
// `ownerSigner`, or from `parentSigner` when given (parent-signed child).
async function registerAgent(
  substrate,
  signer,
  label,
  ownerSigner = signer,
  parent = ZERO,
  parentSigner
) {
  const sig = await signBinding(
    substrate,
    parentSigner || ownerSigner,
    signer.address,
    parent
  );
  return substrate
    .connect(signer)
    .registerAgentBound(
      ethers.toUtf8Bytes("peer-" + label),
      ownerSigner.address,
      parent,
      sig
    );
}

module.exports = {
  ZERO,
  ONE,
  digest,
  deploySubstrate,
  deploySubstrateWith,
  deployRegistry,
  registryOf,
  unitParity,
  setParity,
  deployIndexToken,
  fund,
  feeOf,
  splitFee,
  payFees,
  registerAgent,
  signBinding,
  signRevoke,
  signVoucher,
  anchorFields,
  anchorTx,
  anchorEpoch,
  CLAIM_TAG,
  claimLeaf,
  hashPair,
  sortBasket,
  merkleTree,
  claimTree,
  claimTx,
  SLOT_GOVERNOR,
  SLOT_POOL_PAYER,
  SLOT_POOL_RECIPIENT,
};

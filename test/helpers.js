const { ethers } = require("hardhat");

const digest = (s) => ethers.keccak256(ethers.toUtf8Bytes(s));

async function deploySubstrate() {
  const Substrate = await ethers.getContractFactory("Substrate");
  const substrate = await Substrate.deploy((await ethers.getSigners())[0].address, "0x0000000000000000000000000000000000000000");
  await substrate.waitForDeployment();
  return substrate;
}

async function registerAgent(substrate, signer, lineage) {
  return substrate
    .connect(signer)
    .registerAgent(digest(lineage), ethers.toUtf8Bytes("peer-" + lineage));
}

async function deployToken(name = "TestUSD", symbol = "TUSD") {
  const Mock = await ethers.getContractFactory("MockERC20");
  const t = await Mock.deploy(name, symbol);
  await t.waitForDeployment();
  return t;
}

// ---------------------------------------------------------------------------
// ATN faucet: ATN is minted only via Substrate.recordTrainingForEpoch over an
// anchored epoch (v4.1 gradient trust — the leaf commits agent/amount/repAmount).
// These mint to a registered faucet agent, which then transfers freely.
// ---------------------------------------------------------------------------

function mintLeaf(agentAddr, amount, repAmount = amount) {
  const inner = ethers.keccak256(
    ethers.AbiCoder.defaultAbiCoder().encode(
      ["address", "uint256", "uint256"],
      [agentAddr, amount, repAmount]
    )
  );
  return ethers.keccak256(ethers.solidityPacked(["bytes32"], [inner]));
}

// Substrate's service fee (fee-recycled emission) is taken from every
// payForService at SERVICE_FEE_BPS. Callers wanting an exact net at the
// recipient gross-up: smallest g whose net (g - fee(g)) equals `net`.
function grossFor(net) {
  let g = (net * 10000n) / 9750n;
  while (g - (g * 250n) / 10000n < net) g += 1n;
  return g;
}

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
  await substrate
    .connect(faucetSigner)
    .recordTrainingForEpoch(amount, amount, digest(epochId), []);
}

// Fund `to` with `amount` ATN by minting to the faucet and transferring.
async function fundATN(substrate, submitter, faucet, to, amount) {
  await mintATN(substrate, submitter, faucet, amount);
  await substrate.connect(faucet).transfer(to, amount);
}

// Sign an EIP-712 voucher (channelId, cumulativeAmount) for a PaymentChannel.
async function signVoucher(channel, signer, channelId, cumulativeAmount) {
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
      { name: "cumulativeAmount", type: "uint256" },
    ],
  };
  return signer.signTypedData(domain, types, {
    channelId,
    cumulativeAmount,
  });
}

module.exports = {
  digest,
  deploySubstrate,
  registerAgent,
  deployToken,
  signVoucher,
  mintLeaf,
  grossFor,
  mintATN,
  fundATN,
};

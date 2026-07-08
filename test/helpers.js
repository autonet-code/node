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
};

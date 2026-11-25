/**
 * Deploy contracts/core/Substrate.sol to a Hardhat-configured network.
 *
 * Constructor: (treasury, governor, anchorer, registry, serviceFeeBps,
 * feeTreasuryBps), all from env below. The governor (the DAO timelock) is
 * the only address that may anchor an epoch or retune the service fee.
 * REGISTRY is the DAO Registry, where the value index lives
 * (jurisdiction.parity.<token>). On a local chain without one, a test
 * Registry and a 6-decimal test token at parity 1e30 are deployed, so
 * services can be paid at once.
 * Writes the deployed address + network info to
 * deployments/<network>.json so daemons + tests can pick it up.
 *
 * Usage:
 *   PRIVATE_KEY=<hex> npx hardhat run scripts/deploy_substrate.js --network shadownet
 *   npx hardhat run scripts/deploy_substrate.js --network localhost
 *
 * The PRIVATE_KEY env var is read by hardhat.config.js. For shadownet
 * use the funded test wallet at C:\code\dao\trustless-contracts\pk.txt.
 */

const { ethers, network } = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  const [deployer] = await ethers.getSigners();
  const balance = await ethers.provider.getBalance(deployer.address);
  console.log(`Network: ${network.name} (chainId=${network.config.chainId})`);
  console.log(`Deployer: ${deployer.address}`);
  console.log(`Balance: ${ethers.formatEther(balance)} ETH`);

  if (balance === 0n) {
    throw new Error("Deployer has zero balance");
  }

  // DAO treasury: the fee's treasury share is owed to it (set once). Defaults to the deployer for local runs; pass TREASURY for
  // real networks (shadownet: the trustless DAO address in registry.json).
  const treasury = process.env.TREASURY || deployer.address;
  console.log(`Treasury: ${treasury}`);

  // Governor: the DAO timelock. The only address allowed to anchor an
  // epoch (Substrate.anchor, reached through an ordinary DAO proposal) and
  // to retune the service fee (setServiceFeeBps). Immutable and non-zero.
  // TIMELOCK wins over GOVERNOR. Dev networks may pass an EOA (the
  // daemon's key) so the daemon anchors directly.
  const governor = process.env.TIMELOCK || process.env.GOVERNOR || "";
  if (!governor || /^0x0{40}$/i.test(governor)) {
    throw new Error("set TIMELOCK (or GOVERNOR): the address that anchors epochs");
  }
  console.log(`Governor: ${governor}`);

  let registry = process.env.REGISTRY || "";
  let indexToken = null;
  if (!registry) {
    if (network.name !== "localhost" && network.name !== "hardhat") {
      throw new Error("set REGISTRY: the DAO Registry that holds the value index");
    }
    const Registry = await ethers.getContractFactory("MockRegistry");
    const reg = await Registry.deploy();
    await reg.waitForDeployment();
    registry = await reg.getAddress();
    const Token = await ethers.getContractFactory("MockIndexToken");
    const tok = await Token.deploy("Test USD", "TUSD", 6);
    await tok.waitForDeployment();
    indexToken = await tok.getAddress();
    await (await reg.setParity(indexToken, 10n ** 30n)).wait();
    console.log(`Test Registry: ${registry}, index token TUSD: ${indexToken}`);
  }
  console.log(`Registry: ${registry}`);

  console.log("Deploying Substrate.sol...");
  const Substrate = await ethers.getContractFactory("Substrate");
  const substrate = await Substrate.deploy(
    treasury,
    governor,
    process.env.ANCHORER || ethers.ZeroAddress,
    registry,
    // Service fee and the fee's treasury share (the rest is the tool
    // share), in bps. Set here, governed afterwards.
    Number(process.env.SERVICE_FEE_BPS || 250),
    Number(process.env.FEE_TREASURY_BPS || 5000)
  );
  await substrate.waitForDeployment();
  const address = await substrate.getAddress();
  const tx = substrate.deploymentTransaction();
  console.log(`  address: ${address}`);
  console.log(`  tx: ${tx ? tx.hash : "?"}`);

  // Confirm a few view calls work
  const anchorCount = await substrate.anchorCount();
  const agentCount = await substrate.registeredAgentCount();
  console.log(`  anchorCount: ${anchorCount}`);
  console.log(`  registeredAgentCount: ${agentCount}`);

  const out = {
    network: network.name,
    chainId: network.config.chainId,
    rpc_url: network.config.url || null,
    substrate_address: address,
    treasury: treasury,
    registry: registry,
    index_token: indexToken,
    governor: governor,
    deployer: deployer.address,
    deployment_tx: tx ? tx.hash : null,
    deployed_at: new Date().toISOString(),
  };
  const dir = path.join(__dirname, "..", "deployments");
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  const outPath = path.join(dir, `${network.name}.json`);
  fs.writeFileSync(outPath, JSON.stringify(out, null, 2));
  console.log(`Deployment record written to ${outPath}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

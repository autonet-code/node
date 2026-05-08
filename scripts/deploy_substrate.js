/**
 * Deploy contracts/core/Substrate.sol to a Hardhat-configured network.
 *
 * No constructor args. Writes the deployed address + network info to
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

  console.log("Deploying Substrate.sol...");
  const Substrate = await ethers.getContractFactory("Substrate");
  const substrate = await Substrate.deploy();
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

/**
 * Deploy the standing economy contracts around Substrate.sol:
 *
 *   - ServiceRegistry(substrate) — the ask-priced remote-service market.
 *
 * NOT deployed here:
 *   - PaymentChannel: its challengeWindow constructor arg is an
 *     unblessed economics parameter (docs/BACKLOG.md decision #4).
 *
 * The charter version is NOT deployed here: it lives in the DAO Registry keys
 * jurisdiction.<id>.charter / .charter_uri (docs/charter_anchor.md).
 *
 * Usage:
 *   npx hardhat run scripts/deploy_economy.js --network shadownet
 *
 * Reads the substrate address from deployments/<network>.json (run
 * deploy_substrate.js first). Writes deployments/<network>_economy.json.
 */

const { ethers, network } = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  const [deployer] = await ethers.getSigners();
  console.log(`Network: ${network.name} (chainId=${network.config.chainId})`);
  console.log(`Deployer: ${deployer.address}`);

  const depPath = path.join(__dirname, "..", "deployments", `${network.name}.json`);
  const substrate = JSON.parse(fs.readFileSync(depPath, "utf-8")).substrate_address;
  if (!substrate) throw new Error(`no substrate_address in ${depPath}`);
  console.log(`Substrate: ${substrate}`);

  console.log("Deploying ServiceRegistry...");
  const ServiceRegistry = await ethers.getContractFactory("ServiceRegistry");
  const registry = await ServiceRegistry.deploy(substrate);
  await registry.waitForDeployment();
  const registryAddr = await registry.getAddress();
  console.log(`  address: ${registryAddr}`);

  const out = {
    network: network.name,
    chainId: network.config.chainId,
    substrate_address: substrate,
    service_registry_address: registryAddr,
    payment_channel_address: null, // pending challengeWindow blessing
    deployer: deployer.address,
    deployed_at: new Date().toISOString(),
  };
  const outPath = path.join(__dirname, "..", "deployments", `${network.name}_economy.json`);
  fs.writeFileSync(outPath, JSON.stringify(out, null, 2));
  console.log(`Deployment record written to ${outPath}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

/**
 * Deploy the standing economy contracts around Substrate.sol:
 *
 *   - CharterAnchor(governor = deployer)  + anchor the current charter
 *     hash (pass CHARTER_HASH / CHARTER_URI env vars; hash comes from
 *     `python -c "from nodes.common.world_model_substrate.charter_version
 *     import charter_hash; print(charter_hash())"`).
 *   - ServiceRegistry(substrate) — the ask-priced remote-service market.
 *
 * NOT deployed here:
 *   - PaymentChannel: its challengeWindow constructor arg is an
 *     unblessed economics parameter (docs/BACKLOG.md decision #4).
 *   - VentureVault: deploy-per-venture by design.
 *
 * Usage:
 *   CHARTER_HASH=<64-hex> npx hardhat run scripts/deploy_economy.js --network shadownet
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

  const charterHash = process.env.CHARTER_HASH || "";
  if (!/^[0-9a-fA-F]{64}$/.test(charterHash)) {
    throw new Error("CHARTER_HASH env var must be 64 hex chars (sha256 of the canonical charter)");
  }
  const charterUri = process.env.CHARTER_URI ||
    "https://github.com/autonet-code/node/blob/master/nodes/common/world_model_substrate/adapter.py";

  console.log("Deploying CharterAnchor...");
  const CharterAnchor = await ethers.getContractFactory("CharterAnchor");
  const anchor = await CharterAnchor.deploy(deployer.address);
  await anchor.waitForDeployment();
  const anchorAddr = await anchor.getAddress();
  console.log(`  address: ${anchorAddr} (governor: ${deployer.address})`);

  console.log(`Anchoring charter v1 (${charterHash.slice(0, 12)}...)`);
  const tx = await anchor.anchorCharter("0x" + charterHash, charterUri);
  await tx.wait();
  const current = await anchor.currentCharter();
  console.log(`  anchored version ${current[0]}, hash ${current[1]}`);

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
    charter_anchor_address: anchorAddr,
    charter_hash: charterHash,
    charter_uri: charterUri,
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

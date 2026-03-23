/**
 * Deploy the main Autonet economic contract to Shadownet.
 *
 * Requires: deployment-addresses.json from the initial deploy.
 * Adds: Autonet contract address to deployment-addresses.json.
 */

const hre = require("hardhat");
const fs = require("fs");

async function main() {
  const [deployer] = await hre.ethers.getSigners();
  console.log("Deploying Autonet contract with account:", deployer.address);
  console.log("Balance:", (await deployer.provider.getBalance(deployer.address)).toString());

  // Load existing deployment addresses
  const addresses = JSON.parse(fs.readFileSync("deployment-addresses.json", "utf8"));
  console.log("\nUsing existing deployments:");
  console.log("  ATNToken:", addresses.ATNToken);
  console.log("  ParticipantStaking:", addresses.ParticipantStaking);

  // Deploy a minimal timelock (deployer acts as timelock for testnet)
  // The Autonet contract needs a timelock address — use deployer for now
  const timelockAddress = deployer.address;

  // Deploy the Autonet contract
  console.log("\nDeploying Autonet...");
  const Autonet = await hre.ethers.getContractFactory("Autonet");
  const autonet = await Autonet.deploy(
    addresses.ATNToken,              // _economyAddress (ATN token)
    addresses.ParticipantStaking,    // _registryAddress (staking registry)
    addresses.ATNToken,              // _repTokenAddress (reuse ATN as rep token for testnet)
    timelockAddress,                 // _timelockAddress
    "Autonet",                       // name
    "ANET"                           // symbol
  );
  await autonet.waitForDeployment();
  const autonetAddress = await autonet.getAddress();
  console.log("  Autonet deployed to:", autonetAddress);

  // Update deployment-addresses.json
  addresses.Autonet = autonetAddress;
  addresses.timestamp = new Date().toISOString();
  fs.writeFileSync("deployment-addresses.json", JSON.stringify(addresses, null, 2));
  console.log("\nUpdated deployment-addresses.json");

  console.log("\n" + "=".repeat(50));
  console.log("AUTONET CONTRACT DEPLOYED");
  console.log("=".repeat(50));
  console.log("Address:", autonetAddress);
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });

const { ethers } = require("hardhat");

async function main() {
  const registryAddress = process.env.REGISTRY_ADDRESS;
  if (!registryAddress) {
    throw new Error("Set REGISTRY_ADDRESS env var");
  }

  const [deployer] = await ethers.getSigners();
  console.log("Deployer:", deployer.address);
  console.log("Registry:", registryAddress);
  console.log("Network:", (await ethers.provider.getNetwork()).chainId.toString());

  const balance = await ethers.provider.getBalance(deployer.address);
  console.log("Balance:", ethers.formatEther(balance), "XTZ");

  // Deploy RPB
  const RPB = await ethers.getContractFactory("RPB");
  console.log("\nDeploying RPB...");
  const rpb = await RPB.deploy(registryAddress);
  await rpb.waitForDeployment();
  const rpbAddr = await rpb.getAddress();
  console.log("RPB deployed at:", rpbAddr);

  // Verify basic state
  const name = await rpb.name();
  const symbol = await rpb.symbol();
  const dao = await rpb.dao();
  const totalSupply = await rpb.totalSupply();
  const constitution = await rpb.getConstitution();

  console.log("\n--- RPB State ---");
  console.log("Token:", name, "(" + symbol + ")");
  console.log("DAO:", dao);
  console.log("Total supply:", ethers.formatEther(totalSupply), "ATN (should be 0)");
  console.log("Constitution:", constitution || "(empty)");
  console.log("Current epoch:", (await rpb.currentEpoch()).toString());
  console.log("Provider share:", (await rpb.inferenceProviderShare()).toString(), "bps");
  console.log("Shareholder share:", (await rpb.shareholderShare()).toString(), "bps");
  console.log("Treasury share:", (await rpb.treasuryShare()).toString(), "bps");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });

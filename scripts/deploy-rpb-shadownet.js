const { ethers } = require("hardhat");

async function main() {
  const registryAddress = "0xA2Ec6A1Aa7bd2bfF4f7AFF8d40247F302cFBBb2F";

  const [deployer] = await ethers.getSigners();
  console.log("Deployer:", deployer.address);

  const balance = await ethers.provider.getBalance(deployer.address);
  console.log("Balance:", ethers.formatEther(balance), "XTZ");

  // Deploy RPB
  console.log("\nDeploying RPB with registry:", registryAddress);
  const RPB = await ethers.getContractFactory("RPB");
  const rpb = await RPB.deploy(registryAddress);
  await rpb.waitForDeployment();

  const rpbAddr = await rpb.getAddress();
  console.log("RPB deployed at:", rpbAddr);

  // Verify basics
  console.log("Token name:", await rpb.name());
  console.log("Token symbol:", await rpb.symbol());
  console.log("Total supply:", ethers.formatEther(await rpb.totalSupply()), "ATN");
  console.log("DAO address:", await rpb.dao());
  console.log("Registry:", await rpb.registry());
  console.log("Deployer balance:", ethers.formatEther(await rpb.balanceOf(deployer.address)), "ATN");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });

/**
 * Autonet Contract Deployment Script
 *
 * Deploys all core contracts to the configured network.
 */

const hre = require("hardhat");

async function main() {
  const [deployer] = await hre.ethers.getSigners();

  console.log("Deploying Autonet contracts with account:", deployer.address);
  console.log("Account balance:", (await deployer.provider.getBalance(deployer.address)).toString());

  // 1. Deploy ATN Token
  console.log("\n1. Deploying ATN Token...");
  const ATNToken = await hre.ethers.getContractFactory("ATNToken");
  const initialSupply = hre.ethers.parseEther("1000000000"); // 1 billion
  const atnToken = await ATNToken.deploy(
    deployer.address, // initial holder
    initialSupply
  );
  await atnToken.waitForDeployment();
  const atnAddress = await atnToken.getAddress();
  console.log("   ATN Token deployed to:", atnAddress);

  // 2. Deploy ParticipantStaking
  console.log("\n2. Deploying ParticipantStaking...");
  const ParticipantStaking = await hre.ethers.getContractFactory("ParticipantStaking");
  const staking = await ParticipantStaking.deploy(
    atnAddress,
    deployer.address // governance
  );
  await staking.waitForDeployment();
  const stakingAddress = await staking.getAddress();
  console.log("   ParticipantStaking deployed to:", stakingAddress);

  // 3. Deploy Project Contract
  console.log("\n3. Deploying Project...");
  const Project = await hre.ethers.getContractFactory("Project");
  const project = await Project.deploy(
    atnAddress,
    deployer.address // governance
  );
  await project.waitForDeployment();
  const projectAddress = await project.getAddress();
  console.log("   Project deployed to:", projectAddress);

  // 4. Deploy TaskContract
  console.log("\n4. Deploying TaskContract...");
  const TaskContract = await hre.ethers.getContractFactory("TaskContract");
  const taskContract = await TaskContract.deploy(
    stakingAddress,
    deployer.address // governance
  );
  await taskContract.waitForDeployment();
  const taskAddress = await taskContract.getAddress();
  console.log("   TaskContract deployed to:", taskAddress);

  // 5. Deploy ResultsRewards
  console.log("\n5. Deploying ResultsRewards...");
  const ResultsRewards = await hre.ethers.getContractFactory("ResultsRewards");
  const results = await ResultsRewards.deploy(
    taskAddress,
    stakingAddress,
    deployer.address // governance
  );
  await results.waitForDeployment();
  const resultsAddress = await results.getAddress();
  console.log("   ResultsRewards deployed to:", resultsAddress);

  // 6. Deploy AnchorBridge
  console.log("\n6. Deploying AnchorBridge...");
  const AnchorBridge = await hre.ethers.getContractFactory("AnchorBridge");
  const bridge = await AnchorBridge.deploy(atnAddress);
  await bridge.waitForDeployment();
  const bridgeAddress = await bridge.getAddress();
  console.log("   AnchorBridge deployed to:", bridgeAddress);

  // 7. Deploy DisputeManager
  console.log("\n7. Deploying DisputeManager...");
  const DisputeManager = await hre.ethers.getContractFactory("DisputeManager");
  const disputes = await DisputeManager.deploy(atnAddress);
  await disputes.waitForDeployment();
  const disputesAddress = await disputes.getAddress();
  console.log("   DisputeManager deployed to:", disputesAddress);

  // 8. Deploy AutonetDAO
  console.log("\n8. Deploying AutonetDAO...");
  const AutonetDAO = await hre.ethers.getContractFactory("AutonetDAO");
  const dao = await AutonetDAO.deploy(
    atnAddress,
    deployer.address // treasury
  );
  await dao.waitForDeployment();
  const daoAddress = await dao.getAddress();
  console.log("   AutonetDAO deployed to:", daoAddress);

  // 9. Deploy ForcedErrorRegistry
  console.log("\n9. Deploying ForcedErrorRegistry...");
  const ForcedErrorRegistry = await hre.ethers.getContractFactory("ForcedErrorRegistry");
  const forcedErrors = await ForcedErrorRegistry.deploy(
    atnAddress,
    stakingAddress,
    deployer.address // governance
  );
  await forcedErrors.waitForDeployment();
  const forcedErrorsAddress = await forcedErrors.getAddress();
  console.log("   ForcedErrorRegistry deployed to:", forcedErrorsAddress);

  // 10. Deploy InferenceProviderFactory (Jurisdiction Bridge)
  console.log("\n10. Deploying InferenceProviderFactory...");
  const InferenceProviderFactory = await hre.ethers.getContractFactory("InferenceProviderFactory");
  const inferenceFactory = await InferenceProviderFactory.deploy(
    projectAddress,
    deployer.address // owner
  );
  await inferenceFactory.waitForDeployment();
  const inferenceFactoryAddress = await inferenceFactory.getAddress();
  console.log("   InferenceProviderFactory deployed to:", inferenceFactoryAddress);

  // 11. Configure contract relationships
  console.log("\n11. Configuring contract relationships...");

  // Set ResultsRewards in TaskContract
  await taskContract.setResultsRewardsContract(resultsAddress);
  console.log("   TaskContract linked to ResultsRewards");

  // Set Project in ResultsRewards
  await results.setProjectContract(projectAddress);
  console.log("   ResultsRewards linked to Project");

  // Authorize ResultsRewards as disburser on Project
  await project.setAuthorizedDisburser(resultsAddress, true);
  console.log("   ResultsRewards authorized as disburser");

  // Authorize ResultsRewards as a slasher
  await staking.setAuthorizedSlasher(resultsAddress, true);
  console.log("   ResultsRewards authorized as slasher");

  // Authorize ForcedErrorRegistry as slasher
  await staking.setAuthorizedSlasher(forcedErrorsAddress, true);
  console.log("   ForcedErrorRegistry authorized as slasher");

  // Add deployer as validator for AnchorBridge
  await bridge.addValidator(deployer.address);
  console.log("   Deployer added as bridge validator");

  // Update ATN token DAO address
  await atnToken.setDaoAddress(daoAddress);
  console.log("   ATN Token DAO updated");

  // 11. Print deployment summary
  console.log("\n" + "=".repeat(60));
  console.log("DEPLOYMENT COMPLETE");
  console.log("=".repeat(60));
  console.log(`
Contract Addresses:
-------------------
ATN Token:                  ${atnAddress}
ParticipantStaking:         ${stakingAddress}
Project:                    ${projectAddress}
TaskContract:               ${taskAddress}
ResultsRewards:             ${resultsAddress}
AnchorBridge:               ${bridgeAddress}
DisputeManager:             ${disputesAddress}
AutonetDAO:                 ${daoAddress}
ForcedErrorRegistry:        ${forcedErrorsAddress}
InferenceProviderFactory:   ${inferenceFactoryAddress}

Save these addresses to your .env file!
  `);

  // Save addresses to a JSON file
  const fs = require("fs");
  const addresses = {
    ATNToken: atnAddress,
    ParticipantStaking: stakingAddress,
    Project: projectAddress,
    TaskContract: taskAddress,
    ResultsRewards: resultsAddress,
    AnchorBridge: bridgeAddress,
    DisputeManager: disputesAddress,
    AutonetDAO: daoAddress,
    ForcedErrorRegistry: forcedErrorsAddress,
    InferenceProviderFactory: inferenceFactoryAddress,
    deployer: deployer.address,
    network: hre.network.name,
    timestamp: new Date().toISOString(),
  };

  fs.writeFileSync(
    "deployment-addresses.json",
    JSON.stringify(addresses, null, 2)
  );
  console.log("Addresses saved to deployment-addresses.json");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });

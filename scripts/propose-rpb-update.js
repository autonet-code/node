/**
 * Governance proposal: update rpb.contract in Registry + set constitution on new RPB.
 * Only works on testnets with short voting periods and deployer holding majority tokens.
 */
const { ethers } = require("hardhat");
const fs = require("fs");
const path = require("path");

const GOVERNOR = "0x7c83FF7b0356DbE332BFC527F1Ea73283974aEA2";
const TIMELOCK = "0x4a988674D4d4372C6cE133858beE2a717957D6a1";
const REGISTRY = "0xA2Ec6A1Aa7bd2bfF4f7AFF8d40247F302cFBBb2F";
const NEW_RPB = "0x132169EA108a52d21F168d6906c8F2Eb6Ef1E18A";

async function main() {
  const [deployer] = await ethers.getSigners();
  console.log("Proposer:", deployer.address);

  // Load constitution
  const constitution = fs.readFileSync(
    path.join(__dirname, "../constitution/v1_udhr.txt"), "utf8"
  ).trim();
  console.log("Constitution length:", constitution.length, "chars");

  // Governor contract
  const governorAbi = [
    "function propose(address[],uint256[],bytes[],string) returns (uint256)",
    "function castVote(uint256,uint8) returns (uint256)",
    "function queue(address[],uint256[],bytes[],bytes32) returns (uint256)",
    "function execute(address[],uint256[],bytes[],bytes32) payable returns (uint256)",
    "function state(uint256) view returns (uint8)",
    "function proposalDeadline(uint256) view returns (uint256)",
    "function hashProposal(address[],uint256[],bytes[],bytes32) view returns (uint256)",
    "function votingPeriod() view returns (uint256)",
    "function votingDelay() view returns (uint256)",
  ];
  const governor = new ethers.Contract(GOVERNOR, governorAbi, deployer);

  // Encode calldata for 2 actions:
  // 1. Registry.setRegistryValue("rpb.contract", NEW_RPB)
  // 2. RPB.setConstitution(constitution)
  const registryIface = new ethers.Interface([
    "function editRegistry(string,string)"
  ]);
  const rpbIface = new ethers.Interface([
    "function setConstitution(string)"
  ]);

  const targets = [REGISTRY, NEW_RPB];
  const values = [0, 0];
  const calldatas = [
    registryIface.encodeFunctionData("editRegistry", ["rpb.contract", NEW_RPB]),
    rpbIface.encodeFunctionData("setConstitution", [constitution]),
  ];
  const description = "RPB v0.1.6: update registry pointer + set UDHR constitution on-chain";
  const descHash = ethers.keccak256(ethers.toUtf8Bytes(description));

  // Compute proposal ID
  const proposalId = await governor.hashProposal(targets, values, calldatas, descHash);

  // 1. Propose (skip if already proposed)
  const currentState = await governor.state(proposalId).catch(() => null);
  if (currentState === null || currentState === undefined) {
    console.log("\n1. Creating proposal...");
    const proposeTx = await governor.propose(targets, values, calldatas, description);
    await proposeTx.wait();
    console.log("Proposal created.");
  } else {
    console.log("\n1. Proposal already exists, state:", ["Pending","Active","Canceled","Defeated","Succeeded","Queued","Expired","Executed"][Number(currentState)]);
  }
  console.log("Proposal ID:", proposalId.toString());

  // Wait for voting delay (if any)
  const delay = await governor.votingDelay();
  if (delay > 0n) {
    console.log(`Waiting ${delay} blocks for voting delay...`);
    for (let i = 0; i < Number(delay) + 1; i++) {
      await ethers.provider.send("evm_mine", []);
    }
  }

  // 2. Vote (skip if already voted or not active)
  const stateBeforeVote = await governor.state(proposalId);
  if (stateBeforeVote === 1n) { // Active
    console.log("\n2. Casting vote (For)...");
    try {
      const voteTx = await governor.castVote(proposalId, 1); // 1 = For
      await voteTx.wait();
      console.log("Vote cast.");
    } catch (e) {
      console.log("Vote may already be cast:", e.reason || e.message.slice(0, 100));
    }
  } else {
    console.log("\n2. Skipping vote, state:", stateBeforeVote.toString());
  }

  // Wait for voting period to end (deadline is a timestamp on OZ5 Governor)
  const deadline = await governor.proposalDeadline(proposalId);
  console.log("Proposal deadline (timestamp):", deadline.toString());

  while (true) {
    const block = await ethers.provider.getBlock("latest");
    const now = BigInt(block.timestamp);
    if (now > deadline) break;
    const secsLeft = Number(deadline - now);
    console.log(`Waiting ${secsLeft}s for voting period to end...`);
    await new Promise(r => setTimeout(r, Math.min(secsLeft * 1000, 10000)));
  }
  console.log("Voting period ended.");

  // Check state: 4 = Succeeded
  const state = await governor.state(proposalId);
  console.log("Proposal state:", ["Pending","Active","Canceled","Defeated","Succeeded","Queued","Expired","Executed"][Number(state)]);
  if (state !== 4n) {
    console.error("Proposal did not succeed, state:", state.toString());
    process.exit(1);
  }

  // 3. Queue
  console.log("\n3. Queueing...");
  const queueTx = await governor.queue(targets, values, calldatas, descHash);
  await queueTx.wait();
  console.log("Queued.");

  // 4. Execute (delay is 0 on testnet)
  console.log("\n4. Executing...");
  const execTx = await governor.execute(targets, values, calldatas, descHash);
  await execTx.wait();
  console.log("Executed!");

  // Verify
  const rpbAbi = ["function constitution() view returns (string)", "function getConstitution() view returns (string)"];
  const rpb = new ethers.Contract(NEW_RPB, rpbAbi, deployer);
  const onChain = await rpb.getConstitution();
  console.log("\n--- Verification ---");
  console.log("Constitution on-chain:", onChain.slice(0, 100) + "...");
  console.log("Length:", onChain.length);
  console.log("Matches file:", onChain === constitution);
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });

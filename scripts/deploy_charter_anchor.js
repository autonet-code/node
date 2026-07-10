/**
 * Deploy a fresh CharterAnchor.sol for the new DAO jurisdiction on Etherlink
 * shadownet, governed by the NEW timelock. Standalone raw-ethers (does not
 * depend on hardhat runtime), modeled on scripts/submit-autonet-attach-e2e.js.
 *
 * Cross-checks the old anchor's currentCharter() hash before deploying, and
 * verifies the local charter_hash helper matches the expected hash.
 *
 * Signer key from C:\code\autonet\.env (PRIVATE_KEY). NEVER printed.
 *   node scripts/deploy_charter_anchor.js
 */
const fs = require("fs");
const path = require("path");
const { ethers } = require("ethers");

const RPC = "https://node.shadownet.etherlink.com";
const CHAIN_ID = 127823;
const NEW_TIMELOCK = "0xdB2B6098356f80304Ec78D51799d4e5a377e81dA";
const OLD_ANCHOR = "0x5342aA08EaDE0241fADA97938334Aa4b2B7bEC1D";
const EXPECT_HASH = "5756ed3aa1831533c6ae7a1728cd6af73241c787049080ac3469d5b40f841cd5";
const EXPECT_SIGNER = "0x06E5b15Bc39f921e1503073dBb8A5dA2Fc6220E9";

const ANCHOR_ABI = [
  "function currentCharter() view returns (uint256 version, bytes32 hash, string uri, bytes32 prevHash, uint256 timestamp)",
  "function governor() view returns (address)",
  "function versionCount() view returns (uint256)",
];

function loadKey() {
  const raw = fs.readFileSync("C:\\code\\autonet\\.env", "utf8");
  for (const line of raw.split(/\r?\n/)) {
    const m = line.match(/^\s*PRIVATE_KEY\s*=\s*(.+?)\s*$/);
    if (m) {
      let k = m[1].trim().replace(/^["']|["']$/g, "");
      if (!k.startsWith("0x")) k = "0x" + k;
      return k;
    }
  }
  throw new Error("PRIVATE_KEY not found");
}

async function main() {
  const provider = new ethers.JsonRpcProvider(RPC, CHAIN_ID);
  const wallet = new ethers.Wallet(loadKey(), provider);
  if (wallet.address.toLowerCase() !== EXPECT_SIGNER.toLowerCase())
    throw new Error(`signer mismatch: ${wallet.address}`);
  console.log("signer:", wallet.address);

  // Cross-check old anchor's anchored hash.
  const oldAnchor = new ethers.Contract(OLD_ANCHOR, ANCHOR_ABI, provider);
  const oldCur = await oldAnchor.currentCharter();
  const oldHashHex = oldCur[1].slice(2); // strip 0x
  console.log("OLD anchor currentCharter version:", oldCur[0].toString());
  console.log("OLD anchor hash:", oldHashHex);
  console.log("OLD anchor uri: ", oldCur[2]);
  console.log("expected hash:  ", EXPECT_HASH);
  console.log("OLD == expected:", oldHashHex.toLowerCase() === EXPECT_HASH.toLowerCase());

  // Deploy new anchor.
  const art = JSON.parse(fs.readFileSync(
    path.join(__dirname, "..", "artifacts", "contracts", "core", "CharterAnchor.sol", "CharterAnchor.json"), "utf8"));
  const factory = new ethers.ContractFactory(art.abi, art.bytecode, wallet);
  console.log("Deploying CharterAnchor with governor =", NEW_TIMELOCK, "...");
  const anchor = await factory.deploy(NEW_TIMELOCK);
  const dtx = anchor.deploymentTransaction();
  console.log("DEPLOY tx:", dtx.hash);
  await anchor.waitForDeployment();
  const addr = await anchor.getAddress();
  const rc = await provider.getTransactionReceipt(dtx.hash);
  console.log("NEW CharterAnchor address:", addr);
  console.log("  deploy block:", rc.blockNumber);

  // Verify governor + empty state.
  const anchorR = new ethers.Contract(addr, ANCHOR_ABI, provider);
  const gov = await anchorR.governor();
  const vc = await anchorR.versionCount();
  console.log("  governor():", gov, "(expected", NEW_TIMELOCK + ")");
  console.log("  versionCount():", vc.toString(), "(expected 0)");
  if (gov.toLowerCase() !== NEW_TIMELOCK.toLowerCase())
    throw new Error("governor mismatch on deployed anchor");

  console.log("\n=== SUMMARY (json) ===");
  console.log(JSON.stringify({
    new_charter_anchor: addr,
    deploy_tx: dtx.hash,
    deploy_block: rc.blockNumber,
    governor: gov,
    old_anchor_hash: oldHashHex,
    expected_hash: EXPECT_HASH,
    old_matches_expected: oldHashHex.toLowerCase() === EXPECT_HASH.toLowerCase(),
  }));
}

main().catch((e) => { console.error(e); process.exit(1); });

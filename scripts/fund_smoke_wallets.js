/**
 * Fund a second wallet on shadownet for the cross-machine smoke test.
 *
 * The smoke runs two daemons. Each needs a wallet for chain submission
 * (the anchor submitter). The deployer wallet is the funded one; we
 * fund a second wallet here and print both keys + addresses.
 *
 * The "second wallet" is generated fresh each run unless TARGET_PK
 * is set in env. For idempotency, callers should pin TARGET_PK.
 *
 * Usage:
 *   PRIVATE_KEY=<deployer hex> npx hardhat run scripts/fund_smoke_wallets.js --network shadownet
 *
 * Optional env:
 *   TARGET_PK=<hex>   The second wallet's private key. If unset, a new
 *                     random one is generated. The script tops it up to
 *                     SMOKE_BALANCE_ETH (default 1.0).
 *   SMOKE_BALANCE_ETH (default 1.0)
 *   AUTONET_KEYS_DIR  Where the wallet record (with its private key) is
 *                     written. Default ~/.autonet/keys. Never inside the
 *                     repo: deployments/ holds addresses only.
 *
 * The record lands at <AUTONET_KEYS_DIR>/<network>-smoke-wallet.json,
 * mode 0600 where the OS supports it. The key is not printed.
 */

const { ethers, network } = require("hardhat");
const fs = require("fs");
const os = require("os");
const path = require("path");

function keysDir() {
  return process.env.AUTONET_KEYS_DIR || path.join(os.homedir(), ".autonet", "keys");
}

async function main() {
  const [deployer] = await ethers.getSigners();
  console.log(`Network: ${network.name} (chainId=${network.config.chainId})`);
  console.log(`Funder: ${deployer.address}`);
  const funderBal = await ethers.provider.getBalance(deployer.address);
  console.log(`Funder balance: ${ethers.formatEther(funderBal)} ETH`);

  const targetBalance = ethers.parseEther(process.env.SMOKE_BALANCE_ETH || "1.0");

  // Either accept a provided target wallet or mint a fresh one.
  let targetWallet;
  if (process.env.TARGET_PK) {
    targetWallet = new ethers.Wallet(
      process.env.TARGET_PK.startsWith("0x")
        ? process.env.TARGET_PK
        : "0x" + process.env.TARGET_PK,
      ethers.provider,
    );
    console.log(`Target (existing): ${targetWallet.address}`);
  } else {
    targetWallet = ethers.Wallet.createRandom().connect(ethers.provider);
    console.log(`Target (fresh):    ${targetWallet.address}`);
    console.log(`  (its key is saved to ${keysDir()} after funding)`);
  }

  const targetBal = await ethers.provider.getBalance(targetWallet.address);
  console.log(`Target balance:    ${ethers.formatEther(targetBal)} ETH`);

  if (targetBal >= targetBalance) {
    console.log("Already at or above target balance; no funding needed.");
    return;
  }

  const topUp = targetBalance - targetBal;
  console.log(`Sending ${ethers.formatEther(topUp)} ETH to ${targetWallet.address}...`);
  const tx = await deployer.sendTransaction({
    to: targetWallet.address,
    value: topUp,
  });
  console.log(`  tx: ${tx.hash}`);
  await tx.wait();

  const finalBal = await ethers.provider.getBalance(targetWallet.address);
  console.log(`Target balance after: ${ethers.formatEther(finalBal)} ETH`);

  // Persist the wallet record outside the repo so the smoke script can
  // pick it up (scripts/testnet_chain_smoke.py reads the same path).
  const out = {
    network: network.name,
    chainId: network.config.chainId,
    funder: deployer.address,
    target_address: targetWallet.address,
    target_private_key: targetWallet.privateKey,
    funded_at: new Date().toISOString(),
    final_balance_eth: ethers.formatEther(finalBal),
  };
  const dir = keysDir();
  fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
  const outPath = path.join(dir, `${network.name}-smoke-wallet.json`);
  fs.writeFileSync(outPath, JSON.stringify(out, null, 2), { mode: 0o600 });
  try {
    fs.chmodSync(outPath, 0o600); // writeFileSync's mode skips existing files
  } catch (_) {
    // Windows: POSIX modes are not enforced; the file stays in the user profile.
  }
  console.log(`Wallet record (with private key) written to ${outPath}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

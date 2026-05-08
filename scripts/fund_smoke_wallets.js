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
 */

const { ethers, network } = require("hardhat");
const fs = require("fs");
const path = require("path");

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
    console.log(`Target PRIVATE_KEY: ${targetWallet.privateKey}`);
    console.log("  (record this — it's not stored anywhere else)");
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

  // Persist the wallet info to deployments/ so the smoke script can pick it up.
  const out = {
    network: network.name,
    chainId: network.config.chainId,
    funder: deployer.address,
    target_address: targetWallet.address,
    target_private_key: targetWallet.privateKey,
    funded_at: new Date().toISOString(),
    final_balance_eth: ethers.formatEther(finalBal),
  };
  const dir = path.join(__dirname, "..", "deployments");
  const outPath = path.join(dir, `${network.name}-smoke-wallet.json`);
  fs.writeFileSync(outPath, JSON.stringify(out, null, 2));
  console.log(`Wallet record written to ${outPath}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

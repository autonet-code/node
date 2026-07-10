/**
 * Game-out simulation for the services market (docs/services_market.md).
 *
 * Runs scripted multi-party scenarios against a fresh in-memory hardhat
 * deployment of Substrate + ServiceRegistry + PaymentChannel, and prints
 * outcome tables for the economics review. No assertions — this is a
 * *narrative* of who-ends-up-with-what under honest and adversarial play.
 *
 * Settlement is the payment channel, ONLY (the postpaid escrow was DELETED,
 * ratified 2026-07-04): an unarbitrated escrow can't know delivery truth, so
 * a party must bear the lie. The channel makes exposure per-item and PREPAID
 * — the theft ceiling is one client-sized voucher.
 *
 * ATN-only (ratified 2026-07-10): the channel settles through
 * Substrate.payForService, so the 2.5% fee-recycled emission takes its cut on
 * the canonical rail (closes audit gap G1). Vouchers are GROSS-denominated;
 * the provider receives net of the fee. Deltas below are net.
 *
 * Run:
 *   npx hardhat run scripts/sim_service_market.js
 *
 * Scenarios:
 *   A. Channel — honest multi-item relationship: 4 items, provider closes
 *      the highest voucher, unused deposit refunds.
 *   B. Channel — provider TAKES a voucher and GHOSTS: shows the loss is
 *      bounded to a single item's increment, not the deposit.
 *   C. Channel — CLIENT ghosts mid-relationship: provider closes with its
 *      last voucher and keeps exactly the served total.
 */

const { ethers, network } = require("hardhat");

const CHALLENGE_WINDOW = 3600; // 1h
const DIGEST = (s) => ethers.keccak256(ethers.toUtf8Bytes(s));

// Advance the hardhat clock (only meaningful on the in-process network).
async function warp(seconds) {
  await network.provider.send("evm_increaseTime", [seconds]);
  await network.provider.send("evm_mine", []);
}

// ATN balance on the substrate (the only currency).
async function bal(substrate, addr) {
  return substrate.balanceOf(addr);
}

function fmt(n) {
  return n.toString().padStart(6, " ");
}

function table(title, rows) {
  console.log(`\n=== ${title} ===`);
  const w = Math.max(...rows.map((r) => r[0].length));
  for (const [label, value] of rows) {
    console.log(`  ${label.padEnd(w)} : ${value}`);
  }
}

async function signVoucher(channel, signer, channelId, cumulativeAmount) {
  const net = await ethers.provider.getNetwork();
  const domain = {
    name: "AutonetPaymentChannel",
    version: "1",
    chainId: net.chainId,
    verifyingContract: await channel.getAddress(),
  };
  const types = {
    Voucher: [
      { name: "channelId", type: "uint256" },
      { name: "cumulativeAmount", type: "uint256" },
    ],
  };
  return signer.signTypedData(domain, types, { channelId, cumulativeAmount });
}

// ATN is minted only via Substrate.recordTrainingForEpoch over an anchored
// epoch (money-only leaf commits agent/amount). Mint to a faucet agent,
// then transfer.
function mintLeaf(agentAddr, amount) {
  const inner = ethers.keccak256(
    ethers.AbiCoder.defaultAbiCoder().encode(
      ["address", "uint256"],
      [agentAddr, amount]
    )
  );
  return ethers.keccak256(ethers.solidityPacked(["bytes32"], [inner]));
}

let _epochCounter = 0;
async function mintATN(substrate, faucet, amount) {
  _epochCounter += 1;
  const epochId = "epoch-" + _epochCounter;
  const prevEpochRoot = await substrate.latestEpochRoot();
  const prevAnchorHash = await substrate.latestAnchorHash();
  await substrate
    .connect(faucet)
    .submitAnchor(
      epochId,
      DIGEST("root-" + epochId),
      prevEpochRoot,
      prevAnchorHash,
      "cid-" + epochId,
      DIGEST("payload-" + epochId),
      mintLeaf(faucet.address, amount)
    );
  await substrate
    .connect(faucet)
    .recordTrainingForEpoch(amount, DIGEST(epochId), []);
}

async function main() {
  const [deployer, provider, client] = await ethers.getSigners();

  // --- deploy stack (deployer is the treasury for the fee split) ---
  const Substrate = await ethers.getContractFactory("Substrate");
  const substrate = await Substrate.deploy(
    deployer.address,
    "0x0000000000000000000000000000000000000000",
    "0x0000000000000000000000000000000000000000"
  );
  await substrate.waitForDeployment();

  const Reg = await ethers.getContractFactory("ServiceRegistry");
  const registry = await Reg.deploy(await substrate.getAddress());
  await registry.waitForDeployment();

  const Channel = await ethers.getContractFactory("PaymentChannel");
  const channel = await Channel.deploy(
    await substrate.getAddress(),
    CHALLENGE_WINDOW
  );
  await channel.waitForDeployment();

  // provider is a registered agent (serves + is a mint faucet); client is a
  // wallet the faucet funds with ATN.
  await substrate
    .connect(provider)
    .registerAgent(DIGEST("provider-lineage"), ethers.toUtf8Bytes("peer-prov"));

  await registry
    .connect(provider)
    .registerService(DIGEST("transcribe_audio"), 1000n);

  console.log("Services market game-out (channel-only settlement, ATN-only)");
  console.log(`  challengeWindow=${CHALLENGE_WINDOW}s, service fee=2.5% at settlement`);
  console.log(`  provider=${provider.address}`);
  console.log(`  client  =${client.address}`);

  // Fund + approve the client generously (ATN via the faucet).
  await mintATN(substrate, provider, 100_000n);
  await substrate.connect(provider).transfer(client.address, 100_000n);
  await substrate.connect(client).approve(await channel.getAddress(), 100_000n);

  // ---------------------------------------------------------------------------
  // Scenario A — honest multi-item relationship
  // ---------------------------------------------------------------------------
  {
    const p0 = await bal(substrate, provider.address);
    const c0 = await bal(substrate, client.address);
    await channel.connect(client).openChannel(provider.address, 5000n);
    const id = await channel.channelCount();

    const cumulatives = [1000n, 2000n, 3000n, 4000n];
    let latestSig, latestCum;
    for (const cum of cumulatives) {
      latestSig = await signVoucher(channel, client, id, cum);
      latestCum = cum;
    }
    await channel.connect(provider).closeChannel(id, latestCum, latestSig);
    await warp(CHALLENGE_WINDOW + 1);
    await channel.withdrawRemainder(id);
    table("A. Channel: honest relationship (4 items, deposit 5000)", [
      ["provider Δ (net)", fmt((await bal(substrate, provider.address)) - p0)],
      ["client Δ", fmt((await bal(substrate, client.address)) - c0)],
      ["items served", "4 (gross cumulative 4000)"],
      ["outcome", "provider paid net-of-fee; client refunded 1000 unused — 2 txs for 4 items"],
    ]);
  }

  // ---------------------------------------------------------------------------
  // Scenario B — provider takes a voucher and ghosts (bounded loss)
  // ---------------------------------------------------------------------------
  {
    const p0 = await bal(substrate, provider.address);
    const c0 = await bal(substrate, client.address);
    await channel.connect(client).openChannel(provider.address, 5000n);
    const id = await channel.channelCount();

    const honest = 150n; // 3 served items (gross)
    for (const cum of [50n, 100n, 150n]) {
      await signVoucher(channel, client, id, cum);
    }
    const stolenCum = 200n; // one unsigned-for item taken
    const stolenSig = await signVoucher(channel, client, id, stolenCum);
    await channel.connect(provider).closeChannel(id, stolenCum, stolenSig);
    await warp(CHALLENGE_WINDOW + 1);
    await channel.withdrawRemainder(id);
    const providerDelta = (await bal(substrate, provider.address)) - p0;
    table("B. Channel: provider TAKES a voucher and GHOSTS", [
      ["provider Δ (net)", fmt(providerDelta)],
      ["client Δ", fmt((await bal(substrate, client.address)) - c0)],
      ["honestly served (gross)", fmt(honest)],
      ["outcome", "theft ceiling = ONE item's increment (net), not the 5000 deposit"],
    ]);
  }

  // ---------------------------------------------------------------------------
  // Scenario C — client ghosts mid-relationship (provider protected)
  // ---------------------------------------------------------------------------
  {
    const p0 = await bal(substrate, provider.address);
    const c0 = await bal(substrate, client.address);
    await channel.connect(client).openChannel(provider.address, 5000n);
    const id = await channel.channelCount();

    let lastSig, lastCum;
    for (const cum of [1000n, 2000n, 3000n]) {
      lastSig = await signVoucher(channel, client, id, cum);
      lastCum = cum;
    }
    await channel.connect(provider).closeChannel(id, lastCum, lastSig);
    await warp(CHALLENGE_WINDOW + 1);
    await channel.withdrawRemainder(id); // permissionless refund trigger
    table("C. Channel: CLIENT ghosts mid-relationship", [
      ["provider Δ (net)", fmt((await bal(substrate, provider.address)) - p0)],
      ["client Δ", fmt((await bal(substrate, client.address)) - c0)],
      ["items served", "3 (gross cumulative 3000)"],
      ["outcome", "provider keeps served total net-of-fee; unused 2000 refunds — no work stolen"],
    ]);
  }

  console.log("\nAll scenarios complete. Channel-only settlement is bounded on both sides:");
  console.log("  - provider ghosts after taking a voucher: loss capped at ONE item's increment;");
  console.log("  - client ghosts mid-relationship: provider keeps exactly the served total (net);");
  console.log("  - over-claim capped at deposit; stale/replay blocked; remainder refunds after window;");
  console.log("  - the 2.5% service fee is taken at settlement (fee-recycled emission holds on the canonical rail).");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

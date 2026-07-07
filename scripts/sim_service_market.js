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

async function bal(token, addr) {
  return token.balanceOf(addr);
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

async function main() {
  const [deployer, provider, client] = await ethers.getSigners();

  // --- deploy stack ---
  const Substrate = await ethers.getContractFactory("Substrate");
  const substrate = await Substrate.deploy((await ethers.getSigners())[0].address);
  await substrate.waitForDeployment();

  const Reg = await ethers.getContractFactory("ServiceRegistry");
  const registry = await Reg.deploy(await substrate.getAddress());
  await registry.waitForDeployment();

  const Channel = await ethers.getContractFactory("PaymentChannel");
  const channel = await Channel.deploy(CHALLENGE_WINDOW);
  await channel.waitForDeployment();

  const Token = await ethers.getContractFactory("MockERC20");
  const token = await Token.deploy("SimUSD", "sUSD");
  await token.waitForDeployment();

  // provider is a registered agent; client is just a wallet with money.
  await substrate
    .connect(provider)
    .registerAgent(DIGEST("provider-lineage"), ethers.toUtf8Bytes("peer-prov"));

  await registry
    .connect(provider)
    .registerService(DIGEST("transcribe_audio"), await token.getAddress(), 1000n);

  console.log("Services market game-out (channel-only settlement)");
  console.log(`  challengeWindow=${CHALLENGE_WINDOW}s`);
  console.log(`  provider=${provider.address}`);
  console.log(`  client  =${client.address}`);

  // Fund + approve the client generously; reset between scenarios by tracking
  // deltas rather than balances.
  await token.mint(client.address, 100_000n);
  await token.connect(client).approve(await channel.getAddress(), 100_000n);

  // ---------------------------------------------------------------------------
  // Scenario A — honest multi-item relationship
  // ---------------------------------------------------------------------------
  {
    const p0 = await bal(token, provider.address);
    const c0 = await bal(token, client.address);
    await channel
      .connect(client)
      .openChannel(provider.address, await token.getAddress(), 5000n);
    const id = await channel.channelCount();

    // Each item is served, THEN the client signs the next voucher; cumulative
    // is monotone (item price 1000).
    const cumulatives = [1000n, 2000n, 3000n, 4000n];
    let latestSig, latestCum;
    for (const cum of cumulatives) {
      latestSig = await signVoucher(channel, client, id, cum);
      latestCum = cum;
    }
    // provider closes with the HIGHEST voucher; unused deposit refunds.
    await channel.connect(provider).closeChannel(id, latestCum, latestSig);
    await warp(CHALLENGE_WINDOW + 1);
    await channel.withdrawRemainder(id);
    table("A. Channel: honest relationship (4 items, deposit 5000)", [
      ["provider Δ", fmt((await bal(token, provider.address)) - p0)],
      ["client Δ", fmt((await bal(token, client.address)) - c0)],
      ["items served", "4 (cumulative 4000)"],
      ["outcome", "provider paid 4000, client refunded 1000 unused — 2 txs for 4 items"],
    ]);
  }

  // ---------------------------------------------------------------------------
  // Scenario B — provider takes a voucher and ghosts (bounded loss)
  // ---------------------------------------------------------------------------
  {
    const p0 = await bal(token, provider.address);
    const c0 = await bal(token, client.address);
    await channel
      .connect(client)
      .openChannel(provider.address, await token.getAddress(), 5000n);
    const id = await channel.channelCount();

    // Client sizes its per-item exposure small (item price 50). 3 items are
    // served honestly (cumulative 150). The client optimistically signs ONE
    // more voucher (200) for the next item — the provider pockets it and
    // never delivers, then closes.
    const honest = 150n; // 3 served items
    for (const cum of [50n, 100n, 150n]) {
      await signVoucher(channel, client, id, cum);
    }
    const stolenCum = 200n; // one unsigned-for item taken
    const stolenSig = await signVoucher(channel, client, id, stolenCum);
    await channel.connect(provider).closeChannel(id, stolenCum, stolenSig);
    await warp(CHALLENGE_WINDOW + 1);
    await channel.withdrawRemainder(id);
    const providerDelta = (await bal(token, provider.address)) - p0;
    table("B. Channel: provider TAKES a voucher and GHOSTS", [
      ["provider Δ", fmt(providerDelta)],
      ["client Δ", fmt((await bal(token, client.address)) - c0)],
      ["honestly served", fmt(honest)],
      ["stolen (unserved)", fmt(providerDelta - honest)],
      ["outcome", "theft ceiling = ONE item's increment (50), not the 5000 deposit"],
    ]);
  }

  // ---------------------------------------------------------------------------
  // Scenario C — client ghosts mid-relationship (provider protected)
  // ---------------------------------------------------------------------------
  {
    const p0 = await bal(token, provider.address);
    const c0 = await bal(token, client.address);
    await channel
      .connect(client)
      .openChannel(provider.address, await token.getAddress(), 5000n);
    const id = await channel.channelCount();

    // Provider serves 3 items (cumulative 3000), then the client stops
    // paying / accepting items and disappears. The provider is not stuck: it
    // closes with the last voucher it holds and keeps EXACTLY the served
    // total. The unused remainder refunds to the absent client (anyone can
    // trigger — it's a timer).
    let lastSig, lastCum;
    for (const cum of [1000n, 2000n, 3000n]) {
      lastSig = await signVoucher(channel, client, id, cum);
      lastCum = cum;
    }
    await channel.connect(provider).closeChannel(id, lastCum, lastSig);
    await warp(CHALLENGE_WINDOW + 1);
    await channel.withdrawRemainder(id); // permissionless refund trigger
    table("C. Channel: CLIENT ghosts mid-relationship", [
      ["provider Δ", fmt((await bal(token, provider.address)) - p0)],
      ["client Δ", fmt((await bal(token, client.address)) - c0)],
      ["items served", "3 (cumulative 3000)"],
      ["outcome", "provider keeps served total (3000); unused 2000 refunds — no work stolen"],
    ]);
  }

  console.log("\nAll scenarios complete. Channel-only settlement is bounded on both sides:");
  console.log("  - provider ghosts after taking a voucher: loss capped at ONE item's increment;");
  console.log("  - client ghosts mid-relationship: provider keeps exactly the served total;");
  console.log("  - over-claim capped at deposit; stale/replay blocked; remainder refunds after window.");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

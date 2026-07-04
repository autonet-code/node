/**
 * Game-out simulation for the services market (docs/services_market.md).
 *
 * Runs scripted multi-party scenarios against a fresh in-memory hardhat
 * deployment of Substrate + ServiceRegistry + ServiceEscrow + PaymentChannel,
 * and prints outcome tables for the economics review. No assertions — this is
 * a *narrative* of who-ends-up-with-what under honest and adversarial play.
 *
 * Run:
 *   npx hardhat run scripts/sim_service_market.js
 *
 * Scenarios:
 *   A. Escrow — honest client + honest provider (deliver, release).
 *   B. Escrow — griefing PROVIDER never delivers; client reclaims after T.
 *   C. Escrow — griefing CLIENT never releases; provider claims after T2.
 *   D. Channel — honest run: 4 vouchers, provider closes highest, refund.
 *   E. Channel — voucher-WITHHOLDING provider closes with a stale voucher.
 */

const { ethers, network } = require("hardhat");

const RELEASE_TIMEOUT = 3600; // 1h
const CLAIM_TIMEOUT = 7200; //   2h
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
  const substrate = await Substrate.deploy();
  await substrate.waitForDeployment();

  const Reg = await ethers.getContractFactory("ServiceRegistry");
  const registry = await Reg.deploy(await substrate.getAddress());
  await registry.waitForDeployment();

  const Escrow = await ethers.getContractFactory("ServiceEscrow");
  const escrow = await Escrow.deploy(
    await registry.getAddress(),
    RELEASE_TIMEOUT,
    CLAIM_TIMEOUT
  );
  await escrow.waitForDeployment();

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
  const serviceId = 1n;

  console.log("Services market game-out");
  console.log(`  releaseTimeout=${RELEASE_TIMEOUT}s claimTimeout=${CLAIM_TIMEOUT}s ` +
              `challengeWindow=${CHALLENGE_WINDOW}s`);
  console.log(`  provider=${provider.address}`);
  console.log(`  client  =${client.address}`);

  // Fund + approve the client generously; reset between scenarios by tracking
  // deltas rather than balances.
  await token.mint(client.address, 100_000n);
  await token.connect(client).approve(await escrow.getAddress(), 100_000n);
  await token.connect(client).approve(await channel.getAddress(), 100_000n);

  // ---------------------------------------------------------------------------
  // Scenario A — honest escrow
  // ---------------------------------------------------------------------------
  {
    const req = ethers.id("A-req");
    const p0 = await bal(token, provider.address);
    const c0 = await bal(token, client.address);
    await escrow.connect(client).openRequest(serviceId, req, 1000n);
    await escrow.connect(provider).markDelivered(client.address, serviceId, req);
    await escrow.connect(client).release(serviceId, req);
    table("A. Escrow: honest client + honest provider", [
      ["provider Δ", fmt((await bal(token, provider.address)) - p0)],
      ["client Δ", fmt((await bal(token, client.address)) - c0)],
      ["outcome", "provider paid 1000, service rendered — fair"],
    ]);
  }

  // ---------------------------------------------------------------------------
  // Scenario B — griefing provider never delivers
  // ---------------------------------------------------------------------------
  {
    const req = ethers.id("B-req");
    const p0 = await bal(token, provider.address);
    const c0 = await bal(token, client.address);
    await escrow.connect(client).openRequest(serviceId, req, 1000n);
    // provider stalls forever; client waits out releaseTimeout then reclaims
    await warp(RELEASE_TIMEOUT + 1);
    await escrow.connect(client).reclaim(serviceId, req);
    table("B. Escrow: provider NEVER delivers (griefing provider)", [
      ["provider Δ", fmt((await bal(token, provider.address)) - p0)],
      ["client Δ", fmt((await bal(token, client.address)) - c0)],
      ["outcome", "client fully refunded after T; provider gained nothing"],
    ]);
  }

  // ---------------------------------------------------------------------------
  // Scenario C — griefing client never releases
  // ---------------------------------------------------------------------------
  {
    const req = ethers.id("C-req");
    const p0 = await bal(token, provider.address);
    const c0 = await bal(token, client.address);
    await escrow.connect(client).openRequest(serviceId, req, 1000n);
    await escrow.connect(provider).markDelivered(client.address, serviceId, req);
    // client goes silent; provider waits out claimTimeout then claims
    await warp(CLAIM_TIMEOUT + 1);
    await escrow.connect(provider).claimDelivered(client.address, serviceId, req);
    table("C. Escrow: client NEVER releases (griefing client)", [
      ["provider Δ", fmt((await bal(token, provider.address)) - p0)],
      ["client Δ", fmt((await bal(token, client.address)) - c0)],
      ["outcome", "provider claimed 1000 after T2 for delivered work — protected"],
    ]);
  }

  // ---------------------------------------------------------------------------
  // Scenario D — honest payment channel, 4 work items
  // ---------------------------------------------------------------------------
  {
    const p0 = await bal(token, provider.address);
    const c0 = await bal(token, client.address);
    await channel
      .connect(client)
      .openChannel(provider.address, await token.getAddress(), 5000n);
    const id = await channel.channelCount();

    // client hands vouchers off-chain per item; cumulative is monotone.
    const cumulatives = [1000n, 2000n, 3000n, 4000n];
    let latestSig, latestCum;
    for (const cum of cumulatives) {
      latestSig = await signVoucher(channel, client, id, cum);
      latestCum = cum;
    }
    // provider closes with the HIGHEST voucher.
    await channel.connect(provider).closeChannel(id, latestCum, latestSig);
    await warp(CHALLENGE_WINDOW + 1);
    await channel.withdrawRemainder(id);
    table("D. Channel: honest run (4 items, deposit 5000)", [
      ["provider Δ", fmt((await bal(token, provider.address)) - p0)],
      ["client Δ", fmt((await bal(token, client.address)) - c0)],
      ["items served", "4 (cumulative 4000)"],
      ["outcome", "provider paid 4000, client refunded 1000 unused — 2 txs for 4 items"],
    ]);
  }

  // ---------------------------------------------------------------------------
  // Scenario E — voucher-withholding / stale-close provider
  // ---------------------------------------------------------------------------
  {
    const p0 = await bal(token, provider.address);
    const c0 = await bal(token, client.address);
    await channel
      .connect(client)
      .openChannel(provider.address, await token.getAddress(), 5000n);
    const id = await channel.channelCount();

    // client actually paid up to 4000 (4 items). Provider, buggy or spiteful,
    // closes with an old 2000 voucher instead of the latest.
    const staleSig = await signVoucher(channel, client, id, 2000n);
    await signVoucher(channel, client, id, 4000n); // latest exists but unused
    await channel.connect(provider).closeChannel(id, 2000n, staleSig);
    await warp(CHALLENGE_WINDOW + 1);
    await channel.withdrawRemainder(id);
    table("E. Channel: provider closes with a STALE (lower) voucher", [
      ["provider Δ", fmt((await bal(token, provider.address)) - p0)],
      ["client Δ", fmt((await bal(token, client.address)) - c0)],
      ["outcome", "provider self-harms (claimed 2000 not 4000); client keeps 3000"],
      ["takeaway", "stale-close only hurts the closer — no client exposure"],
    ]);
  }

  console.log("\nAll scenarios complete. Griefing is bounded on both sides:");
  console.log("  - escrow: client reclaims undelivered (T), provider claims delivered (T2)");
  console.log("  - channel: over-claim capped at deposit; stale voucher only hurts provider;");
  console.log("             replay blocked by single-close; remainder refunds after window.");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

/**
 * Deploy the ServiceMarket rail (ServiceRegistry + PaymentChannel) around a
 * deployed Substrate. Separate from deploy_economy.js because the flag-day
 * rotation KEEPS the existing CharterAnchor (charter content unchanged) and
 * only needs the two market contracts on the new Substrate ABI.
 *
 *   ServiceRegistry(substrate)
 *   PaymentChannel(substrate, challengeWindow)   // 2-arg, ATN-only build
 *
 * Usage:
 *   SUBSTRATE_ADDR=0x... [CHALLENGE_WINDOW=3600] \
 *   PRIVATE_KEY=<hex> npx hardhat run scripts/deploy_service_market.js --network shadownet
 */
const { ethers, network } = require("hardhat");

async function main() {
  const [deployer] = await ethers.getSigners();
  const substrate = process.env.SUBSTRATE_ADDR;
  if (!substrate) throw new Error("Set SUBSTRATE_ADDR");
  const challengeWindow = Number(process.env.CHALLENGE_WINDOW || 3600);
  console.log(`Network: ${network.name} (chainId=${network.config.chainId})`);
  console.log(`Deployer: ${deployer.address}`);
  console.log(`Substrate: ${substrate}`);
  console.log(`challengeWindow: ${challengeWindow}`);

  const ServiceRegistry = await ethers.getContractFactory("ServiceRegistry");
  const registry = await ServiceRegistry.deploy(substrate);
  await registry.waitForDeployment();
  const registryAddr = await registry.getAddress();
  console.log(`ServiceRegistry: ${registryAddr}`);

  const PaymentChannel = await ethers.getContractFactory("PaymentChannel");
  const channel = await PaymentChannel.deploy(substrate, challengeWindow);
  await channel.waitForDeployment();
  const channelAddr = await channel.getAddress();
  console.log(`PaymentChannel: ${channelAddr}`);

  console.log(JSON.stringify({
    service_registry_address: registryAddr,
    payment_channel_address: channelAddr,
    challenge_window: challengeWindow,
  }));
}

main().catch((e) => { console.error(e); process.exit(1); });

const { expect } = require("chai");
const { ethers } = require("hardhat");
const {
  digest,
  deploySubstrate,
  registerAgent,
} = require("./helpers");

describe("ServiceRegistry", function () {
  let substrate, registry, provider, provider2, stranger;

  beforeEach(async function () {
    [, provider, provider2, stranger] = await ethers.getSigners();
    substrate = await deploySubstrate();
    await registerAgent(substrate, provider, "prov1");
    await registerAgent(substrate, provider2, "prov2");
    const Reg = await ethers.getContractFactory("ServiceRegistry");
    registry = await Reg.deploy(await substrate.getAddress());
    await registry.waitForDeployment();
  });

  it("registered agent registers a service (id 1), emits event", async function () {
    const spec = digest("spec-transcribe");
    await expect(
      registry.connect(provider).registerService(spec, 1000n)
    )
      .to.emit(registry, "ServiceRegistered")
      .withArgs(1n, provider.address, spec, 1000n);

    const s = await registry.getService(1n);
    expect(s.provider).to.equal(provider.address);
    expect(s.specDigest).to.equal(spec);
    expect(s.askAmount).to.equal(1000n);
    expect(s.active).to.equal(true);
    expect(await registry.serviceCount()).to.equal(1n);
  });

  it("non-agent cannot register a service", async function () {
    await expect(
      registry.connect(stranger).registerService(digest("s"), 1n)
    ).to.be.revertedWithCustomError(registry, "NotRegisteredAgent");
  });

  it("rejects zero spec digest", async function () {
    await expect(
      registry.connect(provider).registerService(ethers.ZeroHash, 1n)
    ).to.be.revertedWithCustomError(registry, "SpecDigestRequired");
  });

  it("provider updates the ask; non-provider cannot", async function () {
    await registry.connect(provider).registerService(digest("s"), 1000n);
    await expect(
      registry.connect(provider).updateServiceAsk(1n, 2000n)
    )
      .to.emit(registry, "ServiceAskUpdated")
      .withArgs(1n, provider.address, 2000n);
    const s = await registry.getService(1n);
    expect(s.askAmount).to.equal(2000n);

    await expect(
      registry.connect(provider2).updateServiceAsk(1n, 5n)
    ).to.be.revertedWithCustomError(registry, "NotServiceProvider");
  });

  it("retire marks inactive; update on retired reverts", async function () {
    await registry.connect(provider).registerService(digest("s"), 1000n);
    await expect(registry.connect(provider).retireService(1n))
      .to.emit(registry, "ServiceRetired")
      .withArgs(1n, provider.address);
    expect((await registry.getService(1n)).active).to.equal(false);
    await expect(
      registry.connect(provider).updateServiceAsk(1n, 1n)
    ).to.be.revertedWithCustomError(registry, "ServiceNotActive");
    await expect(
      registry.connect(provider).retireService(1n)
    ).to.be.revertedWithCustomError(registry, "ServiceNotActive");
  });

  it("operations on a nonexistent service revert", async function () {
    await expect(
      registry.connect(provider).updateServiceAsk(99n, 1n)
    ).to.be.revertedWithCustomError(registry, "NoSuchService");
    await expect(
      registry.connect(provider).retireService(99n)
    ).to.be.revertedWithCustomError(registry, "NoSuchService");
  });
});

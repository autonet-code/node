const { expect } = require("chai");
const { deploySubstrateWith } = require("./helpers");

// Wallet-facing token metadata: name and symbol per jurisdiction (set at
// deploy, derived by the setup from the DAO), decimals fixed at 6.
describe("Substrate ATN metadata", function () {
  it("reports the deploy's name and symbol, and 6 decimals", async function () {
    const substrate = await deploySubstrateWith({});
    expect(await substrate.name()).to.equal("Test ATN");
    expect(await substrate.symbol()).to.equal("REPATN");
    expect(await substrate.decimals()).to.equal(6);
  });
});

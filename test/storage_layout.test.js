const { expect } = require("chai");
const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

// Substrate is upgradeable in place: a jurisdiction's proxy keeps its
// storage while the governor points it at new code. A new version may
// therefore only ADD state after what exists; it must never move, retype or
// remove a variable. This test holds the layout of record
// (test/fixtures/substrate_storage_layout.json) and fails when the compiled
// Substrate no longer starts with it.
//
// After adding state at the end, record it:
//   UPDATE_STORAGE_LAYOUT=1 npx hardhat test test/storage_layout.test.js
// Before the first mainnet deploy the file may be rewritten freely; after
// it, only appended to.

const FIXTURE = path.join(__dirname, "fixtures", "substrate_storage_layout.json");

// A type's full shape, so a change inside a struct or a mapping's value
// shows up too.
function shape(types, id) {
  const t = types[id];
  const out = { label: t.label, bytes: t.numberOfBytes, encoding: t.encoding };
  if (t.key) out.key = shape(types, t.key);
  if (t.value) out.value = shape(types, t.value);
  if (t.base) out.base = shape(types, t.base);
  if (t.members) {
    out.members = t.members.map((m) => ({
      label: m.label, slot: m.slot, offset: m.offset, type: shape(types, m.type),
    }));
  }
  return out;
}

async function currentLayout() {
  const info = await hre.artifacts.getBuildInfo("contracts/core/Substrate.sol:Substrate");
  const layout = info.output.contracts["contracts/core/Substrate.sol"].Substrate.storageLayout;
  return layout.storage.map((v) => ({
    label: v.label, slot: v.slot, offset: v.offset, type: shape(layout.types, v.type),
  }));
}

describe("Substrate storage layout (upgrade safety)", function () {
  it("starts with the layout of record: state is only ever appended", async function () {
    const now = await currentLayout();
    if (process.env.UPDATE_STORAGE_LAYOUT === "1") {
      fs.writeFileSync(FIXTURE, JSON.stringify(now, null, 1) + "\n");
    }
    const recorded = JSON.parse(fs.readFileSync(FIXTURE, "utf8"));
    expect(now.length, "a state variable was removed").to.be.gte(recorded.length);
    for (let i = 0; i < recorded.length; i++) {
      expect(now[i], `storage entry ${i} (${recorded[i].label}) changed`).to.deep.equal(recorded[i]);
    }
    expect(now.length, "new state was appended: record it (see the top of this file)")
      .to.equal(recorded.length);
  });

  it("keeps OpenZeppelin's own state out of the numbered slots", async function () {
    // EIP712 keeps two fallback strings in slots 0 and 1 (unused: the name
    // and version are short constants); Initializable and UUPSUpgradeable
    // use namespaced slots. So Substrate's own first variable sits at 2.
    const now = await currentLayout();
    expect(now[0].label).to.equal("_nameFallback");
    expect(now[1].label).to.equal("_versionFallback");
    expect(now[2].slot).to.equal("2");
  });
});

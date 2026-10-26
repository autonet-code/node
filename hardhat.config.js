require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config();

const SOLC_SETTINGS = {
  optimizer: {
    enabled: true,
    runs: 50,
  },
  viaIR: true,
};

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: {
    // 0.8.20 is the production compiler. 0.8.24 exists only for the
    // test-only DAO stand-in (contracts/test/AnchorTestGovernance.sol,
    // tests/test_governed_anchor.py): OZ 5.4's Governor needs >= 0.8.24.
    // Every other source is pinned to 0.8.20 below so its bytecode does
    // not change with the compiler list.
    compilers: [
      { version: "0.8.20", settings: SOLC_SETTINGS },
      // OZ 5.4 utils/Bytes.sol uses mcopy (Cancun).
      { version: "0.8.24", settings: { ...SOLC_SETTINGS, evmVersion: "cancun" } },
    ],
    overrides: Object.fromEntries(
      [
        "contracts/core/Substrate.sol",
        "contracts/core/ServiceMarket.sol",
        "contracts/test/MockERC20.sol",
        "contracts/test/MockRepToken.sol",
        "contracts/test/ServicePayerMock.sol",
      ].map((f) => [f, { version: "0.8.20", settings: SOLC_SETTINGS }])
    ),
  },
  networks: {
    localhost: {
      // E2E scripts set HARDHAT_LOCAL_URL to run on another port.
      url: process.env.HARDHAT_LOCAL_URL || "http://127.0.0.1:8545",
    },
    hardhat: {
      chainId: 1337,
      allowUnlimitedContractSize: true,
    },
    sepolia: {
      url: process.env.SEPOLIA_RPC_URL || "",
      accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : [],
    },
    shadownet: {
      url: "https://node.shadownet.etherlink.com",
      chainId: 127823,
      accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : [],
      timeout: 180000,
    },
  },
  paths: {
    sources: "./contracts",
    // JS/Solidity hardhat tests live in ./test. The ./tests directory is the
    // Python (pytest) suite — pointing hardhat there would make it try to
    // parse .py files as Mocha specs.
    tests: "./test",
    cache: "./cache",
    artifacts: "./artifacts",
  },
  mocha: {
    timeout: 40000,
  },
};

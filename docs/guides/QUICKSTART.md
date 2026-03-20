# Autonet Quickstart Guide

## Prerequisites

- Node.js 18+
- Python 3.10+
- Git

## Installation

```bash
# Clone
git clone <repo> autonet
cd autonet

# Install Node.js dependencies (contracts)
npm install

# Install Python dependencies (nodes)
pip install -r requirements.txt
```

## Run the Demo

```bash
python demo.py
```

This simulates a complete training cycle without needing a blockchain node.

## Local Development Setup

### 1. Start local blockchain

```bash
npx hardhat node
```

Leave this running. It provides a local Ethereum node at `http://localhost:8545`.

### 2. Deploy contracts

In a new terminal:
```bash
npx hardhat run scripts/deploy.js --network localhost
```

Note the deployed addresses printed to console.

### 3. Update configuration

Copy `.env.example` to `.env` and fill in the contract addresses.

### 4. Run a node

```bash
# Run a solver node
python -c "
from nodes.solver import SolverNode
node = SolverNode()
node.run(max_cycles=10)
"
```

## Docker Compose (Full Stack)

```bash
docker-compose up
```

This starts:
- Hardhat node (blockchain)
- One of each node type (proposer, solver, coordinator, aggregator)

## Contract Interaction (Hardhat Console)

```bash
npx hardhat console --network localhost
```

```javascript
// Get deployed contracts
const ATN = await ethers.getContractAt("ATNToken", "0x...");
const Project = await ethers.getContractAt("Project", "0x...");

// Check balance
await ATN.balanceOf("0x...");

// Create a project
await Project.createProject(
  "My AI Project",
  "QmDescriptionCid",
  ethers.parseEther("1000"),
  ethers.parseEther("100"),
  ethers.parseEther("10000"),
  "My Project Token",
  "MPT"
);
```

## Running Tests

```bash
# Contract tests
npx hardhat test

# Python tests
pytest tests/
```

## Next Steps

1. Read `CLAUDE.md` for full architecture understanding
2. Read `docs/architecture/CONTRACTS.md` for contract details
3. Read `docs/architecture/NODES.md` for node implementation
4. Modify `demo.py` to experiment with the training loop

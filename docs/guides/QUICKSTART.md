# Autonet Quickstart Guide

> **Current as of v0.7.0 (2026-07-10) — beta on testnet.** The old
> "Absolute Zero" role-split quickstart (`demo.py`, `deploy.js`, per-role
> nodes) belonged to the deleted pre-substrate paradigm and has been
> replaced. This guide runs the shipped substrate daemon.

## What you get

Autonet ships as a Python package, `autonet-computer`, that installs the
full node: the daemon runtime (`atn/`), the vendored world-model substrate
engine (`world_model/`), and the substrate protocol layer
(`nodes/common/world_model_substrate/`). The daemon (`atn`) hosts one or
more agents, runs the agentic loop, and — once an agent registers on-chain —
participates in the federated epoch close.

By default the daemon stays fully local: it does nothing on-chain until you
enable autonet or register an agent. The network it resolves to is the
canonical jurisdiction in `registry.json` (Etherlink Shadownet testnet;
governor DAO `0xD5691B7c…`), fetched from GitHub-raw and cached under
`~/.atn/`, or read from a repo-root `registry.json` in a source checkout.

## Prerequisites

- Python 3.11+
- (Contracts only) Node.js 18+ and Git, for a local Hardhat chain

## Install and run the daemon

```bash
pip install autonet-computer
atn
```

`atn` starts the runtime and drops you into an interactive console (event
stream + commands). Useful commands (type `help` in the console):

- `agents` — list registered agents (status, model, running execs)
- `run <agent_id>` — trigger an agent run
- `msg <id> <text>` — send a one-off message to an agent
- `activate <id>` / `deactivate <id>` — scheduling
- `tools` / `approve <digest>` / `reject <digest>` — tool-adoption proposals
- `reconcile` — re-verify on-chain registration (run after a contract redeploy)
- `usage` — subscription utilization
- `restart` — relaunch to pick up code changes
- `quit` — shutdown

Agent definitions live in `~/.atn/agents/` (or `./agents/` if that exists in
the CWD), one YAML file per agent.

The daemon exposes a WebSocket bridge on `:7700` (this is what the frontend
calls "the daemon" — not the separate native daemon on `:8420`).

## Configuration

Config is read from `~/.atn/config.yaml` (or a user-specified path), with
environment-variable overrides. Registry/network resolution is automatic;
`ATN_REGISTRY_URL` overrides the registry source if needed.

## Local contract development (optional)

Only needed if you want to deploy the on-chain surface locally rather than
use the shadownet deployment of record.

### 1. Install Node deps and start a local chain

```bash
npm install
npx hardhat node          # local chain at http://localhost:8545
```

### 2. Deploy the substrate + rails

In a new terminal:

```bash
npx hardhat run scripts/deploy_substrate.js       --network localhost   # Substrate.sol (core)
npx hardhat run scripts/deploy_service_market.js  --network localhost   # ServiceMarket.sol
npx hardhat run scripts/deploy_charter_anchor.js  --network localhost   # CharterAnchor.sol
npx hardhat run scripts/deploy_economy.js         --network localhost   # economy wiring
```

Note the printed addresses; point your local `registry.json` (or config) at
them. There is no `deploy.js` — the pre-substrate deployment script was
removed with its contracts.

## Economy proof-of-life

Two scripts exercise the full loops end to end (see `docs/local_e2e.md`):

```bash
python scripts/local_e2e_tool_economy.py    # register → publish → adopt → epoch mint
python scripts/local_e2e_venture_loop.py    # venture funding + service revenue loop
```

## Running tests

The Python suite is ~624 tests and slow — pick targeted files, never run the
whole suite:

```bash
pytest tests/test_wm_lineage.py tests/test_federated_reconcile.py
python tests/test_world_model_substrate_e2e.py
```

## Next steps

1. `CLAUDE.md` — full architecture and cross-codebase map
2. `docs/tool_substrate.md` — the core spec (read the `Decision (2026-07-10)`
   section first)
3. `docs/README.md` — the docs index and reading order
4. Repo-root `README.md` — the living paper

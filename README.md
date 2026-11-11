[![AUTONET: A self-sufficient economy for AI and human agents (12 min walkthrough)](https://img.youtube.com/vi/wdyUwG7vUxg/maxresdefault.jpg)](https://youtu.be/wdyUwG7vUxg)

# Autonet

The node of the On-Chain Jurisdiction: the daemon that a household runs, the
tool library and the services market it takes part in, and Autonet's
contracts.

**Read the paper: [The On-Chain Jurisdiction](https://github.com/autonet-code/whitepaper).**

Nothing has launched yet: every release so far runs on test networks.
Addresses of record are in [`registry.json`](registry.json).

## Install

```bash
pip install autonet-computer                       # Full node: agent framework + chain registration + substrate/P2P
pip install autonet-computer[voice-full]           # + voice: all TTS backends + push-to-talk STT
```

The base install is a fully functional node: agent management, on-chain
registration, and substrate event gossip / federated close all work out of the
box.

Or from source:

```bash
git clone https://github.com/autonet-code/node.git
cd node
pip install -e .
```

Start the agent framework:

```bash
atn
```

**Set your owner wallet.** The owner wallet is the identity that owns this
daemon's earnings (published tools, service revenue) and the address a remote
connection must sign with to control the whole fleet. Set it from the Config
page once a wallet is connected, or by hand in `~/.atn/config.yaml`:

```yaml
autonet:
  owner_wallet: "0xYourWalletAddress"
```

## Quickstart

Prerequisites: Python 3.11+, and Node.js 18+ for contract work.

```bash
pip install -r requirements.txt                          # Python nodes
npm install                                              # contracts
npx hardhat node                                         # local chain (separate terminal)
npx hardhat run scripts/deploy_substrate.js --network localhost
```

**Proof of life.** An end-to-end script exercises the live path without needing a live network:

```bash
python scripts/local_e2e_tool_security.py         # tool economy: bond -> review -> close -> conclude -> on-chain mint -> malware verdict, DAO clear and designate
```

The other `scripts/local_e2e_*.py` scripts cover services, the epoch-close and the robots the same way.

## Repo map

| Path | What |
|------|------|
| `atn/` | The daemon: agents, the WebSocket server and agent API, wallet auth, the tool store, services, the chain client. |
| `nodes/` | The network side: p2p, the blob store, event gossip, the tool library, the epoch-close and the review mint. |
| `contracts/core/` | Autonet's contracts: `Substrate.sol`, `ServiceMarket.sol`, `AnchorElection.sol`. |
| `robot/` | The robot side: the reference safety pack, the ROS 2 safety filter and the simulated robot (Docker). |
| `experiments/` | Experiment records, including the simulated robot runs (`experiments/robots/`). |
| `scripts/` | Deploy, operations and the end-to-end scripts. |
| `tests/` | The test suite (large and slow: run targeted files, never the whole suite). |
| `docs/` | The system's current architecture and economy, on one page each. |
| `docs_from_code/` | Reference documentation generated from the code. |
| `legacy/pre-substrate/` | Earlier-paradigm files kept with their history. Not the live path. |

## Testing

```bash
pytest tests/test_tool_mint.py tests/test_federated_reconcile.py    # targeted files only
python scripts/local_e2e_robot.py                                   # robots against a simulated TurtleBot3 (needs Docker)
```

## Contributing

1. Fork the repo.
2. Make changes on a short topic branch off `master`.
3. Run targeted tests, plus the end-to-end scripts for the areas you touched.
4. Open a PR. Merges rebase onto `master` and fast-forward, with docs updated in the same change.

## Related repositories

| Repo | What |
|------|------|
| [whitepaper](https://github.com/autonet-code/whitepaper) | The paper, with its earlier versions. |
| [on-chain-jurisdiction](https://github.com/autonet-code/on-chain-jurisdiction) | DAO governance, trustless economy, RepToken. |
| [tool-registry](https://github.com/autonet-code/tool-registry) | Open catalog of agent tools. |

## License

AGPL-3.0-or-later (see `LICENSE`). The Solidity contracts under `contracts/` stay MIT (see `contracts/LICENSE`).
Releases up to v0.7.5 were published under MIT.

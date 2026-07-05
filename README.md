# Autonet Node

**Decentralized AI alignment, training, and agent orchestration.**

> Alpha pre-release. `Substrate.sol` is deployed on Etherlink Shadownet (testnet). The constitution is not yet published on-chain. Expect breaking changes.

Autonet is a protocol for decentralized AI alignment where alignment emerges from economic incentives rather than centralized constraint. The canonical training and inference path is the **world-model substrate**: a graph of debated claims over a 6-root charter coordinate space, not a neural network. Agents contribute conversations; a deterministic **federated epoch close** replays those events into a bit-identical mint map on every honest daemon; a rotating submitter anchors the epoch on-chain, and each agent records its own mint — reputation (soulbound) and ATN (transferable) minted in lockstep.

The item the substrate judges is a **tool**: pinned, locally-verifiable code. Published tools become verdict-layer claims; their authors mint from epoch emission in proportion to standing × usage, gated by a consensus **vetting greenlight** and an **adoption** rail. Remote APIs are a separate economy — **Services** — that carry no substrate standing (remote execution is unknowable in principle) and trade on behavioral trust instead. An agent can be funded as a **venture**: backers stake ATN against its future service revenue.

Four contracts under `contracts/core/` carry this:

| Contract | Role |
|----------|------|
| `Substrate.sol` | Constitutional core: epoch anchoring, agent registry, training records, ATN token, inference payments. The only mint path. |
| `ServiceMarket.sol` | Remote-API market: service registry + EIP-712 payment channels; any-ERC20 pricing; no mint. |
| `VentureVault.sol` | Agent-as-venture funding — one instance per venture; backers hold a pull-based claim on net revenue. No judge. |
| `CharterAnchor.sol` | Governed anchor for the charter *version* (values stay off-chain); drift detection. |

The original VL-JEPA / TextJEPA neural pipeline and the proposer/solver/coordinator role split are **no longer live** — their modules remain on disk for reference only (`docs/VALIDATION_FINDINGS.md` explains why VL-JEPA was shelved). For the authoritative, maintained architecture map and agent-onboarding contract, read **`CLAUDE.md`**. For the design specs, see **`docs/README.md`**.

## Install

```bash
pip install autonet-computer                  # Agent framework (local operation)
pip install autonet-computer[voice]           # + voice / TTS
pip install autonet-computer[network]         # + blockchain, P2P, training (full node)
pip install autonet-computer[network,voice]   # everything
```

Or from source:

```bash
git clone https://github.com/autonet-code/node.git
cd node
pip install -e ".[network]"
```

Start the agent framework:

```bash
atn
```

## Quickstart

Prerequisites: Python 3.11+, and Node.js 18+ for contract work.

```bash
pip install -r requirements.txt                          # Python nodes
npm install                                              # contracts
npx hardhat node                                         # local chain (separate terminal)
npx hardhat run scripts/deploy_substrate.js --network localhost
```

**Proof of life.** Three end-to-end scripts exercise the live paths without needing a live network:

```bash
python tests/test_world_model_substrate_e2e.py    # substrate vertical slice: events -> close -> mint -> inference
python scripts/local_e2e_tool_economy.py          # tool economy: register -> publish -> vet -> adopt -> epoch mint
python scripts/local_e2e_venture_loop.py          # venture funding + service revenue loop
```

The tool-economy and venture scripts are the tool-substrate session's capstones — see `docs/local_e2e.md` for what each proves and the seams they revealed.

## Repo map

| Path | What |
|------|------|
| `atn/` | Agent framework / daemon runtime: agent registry, WS server (:7700), wallet auth, on-chain service, agent directory. |
| `nodes/` | Training & inference implementation. `common/world_model_substrate/` is the substrate protocol layer (adapter, events, reconcile, mint_gate, infer, artifact_index); `common/` holds shared infra (p2p, blob store, event gossip, federated close). |
| `world_model/` | Vendored substrate engine: claim graph, charter tendencies, equilibration. Sync per `world_model/VENDORED.md`. |
| `contracts/core/` | The four Solidity contracts (see table above). |
| `experiments/` | Pre-registered contest experiments (phase8–phase10): prereg committed before any run, raw artifacts, pure `analyze.py`. |
| `scripts/` | Deploy, install, and operational scripts. `scripts/debug/` groups profiling/repro scripts. |
| `tests/` | Test suite (~624 tests — run targeted subsets, never the whole thing). |
| `legacy/pre-substrate/` | Dead-paradigm files preserved with history (orchestrator sim, VL-JEPA demo/validate, FedAvg-era self-tests). Not the live path. |
| `docs/` | Design specs, experiment records, reference. Start at `docs/README.md`. |

See `CLAUDE.md` ("This Repo: Key Directories") for the authoritative map.

## Alignment pricing

Operations are priced by semantic alignment with jurisdiction standards — the same mechanism steers both inference cost and training reward:

```
alignment = geometric_mean(user_to_jurisdiction, task_to_user, task_to_jurisdiction)
```

High alignment is subsidized (toward free); neutral pays base cost; low alignment pays a premium that funds the subsidies. Applied to training, "task alignment" becomes "capability gap" — the network pays more to train what it lacks. In V1 the pricing (`nodes/common/alignment_pricing.py`) is advisory: computed and displayed, not enforced.

## Testing

```bash
pytest tests/test_wm_lineage.py tests/test_federated_reconcile.py   # targeted subsets only
python tests/test_world_model_substrate_e2e.py                      # substrate e2e
```

The Python suite is large and slow — **never run the whole `pytest tests/`**; pick targeted files. See `CLAUDE.md` ("Testing").

## Contributing

The codebase splits into a **core-protected layer** (seven files enforcing the jurisdiction's constitutional guarantees — constitution injection, lineage-hash verification, alignment-hash computation, on-chain integrity check; hashed into an on-chain fingerprint) and an **extensible surface** (everything else — providers, tools, connectors, CLI, config, the training pipeline). Changes outside the seven core files keep the node's on-chain integrity check passing. See `CLAUDE.md` for the core-file list and the scope boundaries for autonomous work.

1. Fork the repo.
2. Make changes (extensible surface).
3. Run targeted test subsets per `CLAUDE.md` — never the full suite.
4. Open a PR.

## Related repositories

| Repo | What |
|------|------|
| [whitepaper](https://github.com/autonet-code/whitepaper) | Protocol specification. |
| [on-chain-jurisdiction](https://github.com/autonet-code/on-chain-jurisdiction) | DAO governance, trustless economy, RepToken. |
| [tool-registry](https://github.com/autonet-code/tool-registry) | Open catalog of agent tools. |

## License

MIT

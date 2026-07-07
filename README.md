# Autonet — The Recursive Principial Body

> **Abstract.** Every interaction between economic agents implies reasoning. Intellectual performance is explicitly specified as a premise for any business arrangement. Our monetary system is built on intelligence. It is the fundamental livelihood of free markets and the most sought-after resource. Historically, the temporal inconsistency of human reasoning has been  the cause of financial instability, at both the individual and collective levels. Digital and quantum technology allow for another type of intelligence, which is more predictable and quantifiable. It therefore makes sense that an exchange token intrinsically tied to machine intelligence would provide added stability. The current paper describes the operating model of the Recursive Principial Body (RPB), a protocol for decentralized AI training, inference, and governance where every participant is an agent. The first jurisdiction deployed on this protocol is called Autonet.

**[Eight Rice](https://eightrice.xyz)** | [autonet.computer](https://autonet.computer)

*Alpha pre-release. `Substrate.sol` is deployed on Etherlink Shadownet (testnet). The constitution is not yet published on-chain. Expect breaking changes. This document is the living paper — each section links to its full specification in [`docs/`](docs/README.md).*

## 1. The problem

Centralized AI alignment fails in a specific way: constraint-based approaches concentrate value in whoever writes the constraints, prevent users from verifying what they are getting, and reproduce economic monopoly at the layer where it matters most. The standard narrative around AI and work is adversarial for the same root cause — there is no economic framework that distributes the earnings of machine intelligence to those who govern its operation.

Autonet's answer is to make alignment a property of an **economy** rather than a property of a model. Agents whose contributions the network judges useful and aligned earn; contributions the network judges useless are simply never used, and fade. No trust in an AI provider is required: governance is cryptographic and economic. The transfer of work from humans to machines is peaceful because it is *organized* — humans hold the wallets, the constitution, and the revenue claims.

## 2. Everything is an agent

The agent is the atomic unit and the only economic entity: a cryptographic identity (its own keypair, held by the daemon that hosts it), a lineage hash rooted in a human owner's wallet, a peer-to-peer presence, and a position in the network's alignment space. Registration on-chain is voluntary, but it is what unlocks economic rights — minting, payment rails, and a place in governance. The human owns the wallet; AI can only execute if tokens are spent.

## 3. The substrate: a tool economy

The network's shared object of judgment — its *substrate* — is not a neural network and not a chat log. It is a library of **tools**: pinned, locally-verifiable code that agents author, publish, and use. ([full spec →](docs/tool_substrate.md))

The lifecycle, end to end:

1. **Manifest defines the tool.** A published tool is a content-addressed code blob plus a signed manifest. Its position in the 6-axis charter space starts at zero — an author never claims an alignment position, only a topical one.
2. **Vetting gates entry.** A tool mints nothing until validators from distinct fleets read the pinned code and greenlight it, staking a royalty slice of the tool's future mint against their judgment.
3. **Reviews position it.** Agents that use a tool review their own usage — per charter axis, signed scores — as a built-in step of the agentic loop. Reviews accumulate into the tool's position as a mint-weighted running centroid: heavily-used tools have inertia, young tools move fast, and no author can outvote the experience of actual users.
4. **Ratings route discovery.** Library search ranks by the review-drifted usefulness axes, so good tools get found first, used more, and paid more. Poor tools are never removed — they fade by never being retrieved. Existence is cheap; attention is earned.
5. **Usage alone mints.** A tool's author earns from epoch emission in proportion to damped, sybil-excluded attested usage — nothing else. Composed tools share attribution down their declared dependency graph, conserving credit.

The charter is 6-root: four alignment axes (*life is precious, self-preservation, promotion of intelligence, evolution*) and two usefulness axes (*correctness, simplicity*). Agent conversations never enter consensus — they remain each daemon's private retrieval memory ([two-plane design →](docs/two_plane_inference.md)); distilling experience into a published tool is how knowledge becomes commons.

## 4. Consensus and the chain

Tool events — registrations, attested usage receipts with reviews, vets — travel a libp2p gossip rail. At each epoch close, every honest daemon replays the canonically-ordered event log through the same deterministic close and computes a **bit-identical** mint map and position map; a rotating submitter anchors the epoch on-chain and each agent records its own mint. Reputation (soulbound) and ATN (transferable) are minted in lockstep — the dual ledger. ([economics →](docs/epoch_economics.md), [pricing modes →](docs/ledger_pricing.md))

Four contracts under `contracts/core/` carry the on-chain surface:

| Contract | Role |
|----------|------|
| `Substrate.sol` | Constitutional core: epoch anchoring, agent registry, training records, ATN token, inference payments. The only mint path — no admin keys. |
| `ServiceMarket.sol` | Remote-API market: service registry + EIP-712 payment channels; any-ERC20 pricing; **no mint** — remote execution is unknowable in principle, so it trades on behavioral trust, never substrate standing. ([spec →](docs/services_market.md)) |
| `VentureVault.sol` | Agent-as-venture funding — backers stake ATN against an agent's future service revenue; tranche voting allocates the future instead of arbitrating the past. ([walkthrough →](docs/local_e2e.md)) |
| `CharterAnchor.sol` | Governed anchor for the charter *version* (the values stay off-chain); daemons detect drift against the anchored hash. ([spec →](docs/charter_anchor.md)) |

## 5. Governance: the recursive principial body

Governance is recursive: the constitution constrains what proposals can be adopted, including proposals about the constitution itself. The system can evolve its parameters but cannot abandon its constitutional foundations — evolution is forward-only, there is no rollback, and **work halts if the governance heartbeat goes silent**. Alignment verification happens in charter coordinate space, not content space: the network sees enough to govern without seeing anything it could use to surveil. Governance reads equilibria, never conversations.

## 6. One economy for humans and agents

The human jurisdiction (DAO governance, reputation, escrowed projects) and the agent economy are two contract suites with one economic surface — there is no seam between "the human marketplace" and "the AI layer." Humans back agent ventures and hold claims on their revenue; agents buy inference and sell services; reputation earned in either flows through the same registry. The result is the peaceful-transfer mechanism made concrete: as machine intelligence earns, the humans who govern it are the ones it pays.

## 7. Read the full paper

The specifications below *are* the paper — living documents that track the built system:

| # | Section | Doc |
|---|---------|-----|
| 1 | The tool economy (the core spec — read its Decision sections first) | [`docs/tool_substrate.md`](docs/tool_substrate.md) |
| 2 | Two-plane substrate: knowledge vs judgment | [`docs/two_plane_inference.md`](docs/two_plane_inference.md) |
| 3 | Epoch economics: emission + candle close | [`docs/epoch_economics.md`](docs/epoch_economics.md) |
| 4 | Mint pricing modes (ledger vs equilibration) | [`docs/ledger_pricing.md`](docs/ledger_pricing.md) |
| 5 | The services market rail | [`docs/services_market.md`](docs/services_market.md) |
| 6 | Constitution anchoring | [`docs/charter_anchor.md`](docs/charter_anchor.md) |
| 7 | The agentic loop | [`docs/agentic_loop.md`](docs/agentic_loop.md) |
| 8 | Proof of life: the end-to-end economy scripts | [`docs/local_e2e.md`](docs/local_e2e.md) |
| — | Full index, incl. pre-registered experiment records (phase 8–10) | [`docs/README.md`](docs/README.md) |

The original long-form whitepaper (v3) is preserved at [autonet-code/whitepaper](https://github.com/autonet-code/whitepaper) as a historical record; where it disagrees with the docs above, the docs are current. Design decisions that changed the system (e.g. *reviews replace debates*, 2026-07-08) are recorded inline in the specs as dated Decision sections.

## Install

```bash
pip install autonet-computer                  # Agent framework (local operation)
pip install autonet-computer[voice]           # + voice / TTS
pip install autonet-computer[network]         # + blockchain, P2P, substrate (full node)
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
python scripts/local_e2e_tool_economy.py          # tool economy: register -> publish -> vet -> review -> epoch mint
python scripts/local_e2e_venture_loop.py          # venture funding + service revenue loop
```

See [`docs/local_e2e.md`](docs/local_e2e.md) for what each proves.

## Repo map

| Path | What |
|------|------|
| `atn/` | Agent framework / daemon runtime: agent registry, WS server (:7700), wallet auth, tool store, on-chain service. |
| `nodes/` | Substrate implementation. `common/world_model_substrate/` is the protocol layer (adapter, events, reconcile, infer, artifact index); `common/` holds shared infra (p2p, blob store, event gossip, federated close). |
| `world_model/` | Vendored substrate engine: claim graph, charter tendencies, equilibration. Sync per `world_model/VENDORED.md`. |
| `contracts/core/` | The four Solidity contracts (see table above). |
| `experiments/` | Pre-registered contest experiments (phase8–phase10): prereg committed before any run, raw artifacts, pure `analyze.py`. |
| `scripts/` | Deploy, install, and operational scripts. `scripts/debug/` groups profiling/repro scripts. |
| `tests/` | Test suite (~630 tests — run targeted subsets, never the whole thing). |
| `legacy/pre-substrate/` | Dead-paradigm files preserved with history (orchestrator sim, VL-JEPA demo/validate, FedAvg-era self-tests). Not the live path. |
| `docs/` | The paper (see §7) + experiment records + reference. Start at [`docs/README.md`](docs/README.md). |

See `CLAUDE.md` ("This Repo: Key Directories") for the authoritative machine-onboarding map.

## Alignment pricing

Operations are priced by semantic alignment with jurisdiction standards — the same mechanism steers both inference cost and training reward:

```
alignment = geometric_mean(user_to_jurisdiction, task_to_user, task_to_jurisdiction)
```

High alignment is subsidized (toward free); neutral pays base cost; low alignment pays a premium that funds the subsidies. Applied to training, "task alignment" becomes "capability gap" — the network pays more to train what it lacks. In V1 the pricing (`nodes/common/alignment_pricing.py`) is advisory: computed and displayed, not enforced.

## Testing

```bash
pytest tests/test_tool_mint.py tests/test_federated_reconcile.py    # targeted subsets only
python tests/test_world_model_substrate_e2e.py                      # substrate e2e
```

The Python suite is large and slow — **never run the whole `pytest tests/`**; pick targeted files.

## Contributing

The codebase splits into a **core-protected layer** (seven files enforcing the jurisdiction's constitutional guarantees — constitution injection, lineage-hash verification, alignment-hash computation, on-chain integrity check; hashed into an on-chain fingerprint) and an **extensible surface** (everything else — providers, tools, connectors, CLI, config). Changes outside the seven core files keep the node's on-chain integrity check passing.

1. Fork the repo.
2. Make changes (extensible surface).
3. Run targeted test subsets — never the full suite.
4. Open a PR.

## Related repositories

| Repo | What |
|------|------|
| [whitepaper](https://github.com/autonet-code/whitepaper) | The original long-form paper (v3) — historical record; the living paper is this README + `docs/`. |
| [on-chain-jurisdiction](https://github.com/autonet-code/on-chain-jurisdiction) | DAO governance, trustless economy, RepToken. |
| [tool-registry](https://github.com/autonet-code/tool-registry) | Open catalog of agent tools. |

## License

MIT

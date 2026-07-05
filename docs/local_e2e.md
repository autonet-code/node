# Local end-to-end: tool economy composes

`scripts/local_e2e_tool_economy.py` is the tool-substrate session's
capstone. It proves the whole tool-economy architecture — daemon,
substrate consensus, and chain — composes on a single local deployment,
with each hop verified against the next.

## Run

```bash
python scripts/local_e2e_tool_economy.py
```

Prerequisites (already true in this repo): `npm install` done
(`node_modules/` present, incl. Hardhat), contracts compiled
(`npx hardhat compile`), and the Python deps installed
(`web3`, `eth-account`, the `atn` + `nodes` packages importable).
Windows and POSIX both supported; the script manages the Hardhat node
subprocess and kills it (taskkill /T on Windows) on teardown.

The script prints a stage-by-stage scoreboard and exits nonzero if any
stage FAILs. A stage may legitimately SKIP (with a printed reason)
rather than sink time — currently only the ServiceMarket escrow leg is
allowed to SKIP (if `MockERC20` is absent from the repo).

## What it proves

Seven stages, each verified:

1. **Chain up** — launches `npx hardhat node` (localhost:8545), deploys
   `Substrate.sol` + `ServiceMarket.sol` (ServiceRegistry + ServiceEscrow)
   + `MockERC20` via a tiny inline `hardhat run` deployer. Hardhat's
   funded `account[0]` is the author's human OWNER wallet, `account[1]`
   the caller's.
2. **Fleet** — two real ATN `Runtime`s in tmp dirs (`autonet.enabled=False`;
   we drive the substrate pieces directly). Runtime A hosts `author-1`
   (owner = account[0]); it also hosts `caller-1` (owner = account[1] on
   chain) so tool invocation stays in one daemon (tools are daemon-local)
   while the two DISTINCT on-chain owners the damper needs still exist.
   Runtime B stands up its own agent to prove the two-runtime fleet.
   Verifies `author-1`'s parentless lineage is **owner-rooted** (hashes
   from the owner wallet).
3. **Chain identity** — owner-bound registration on chain. Reads each
   agent's key from `runtime.registry.get_agent_key`, funds the agent
   address with gas from its owner, signs the owner-binding EIP-712, and
   calls the registration method from the agent's own key. Verifies
   `getAgentOwner` returns the two distinct owners on chain.
4. **Tool economy (daemon)** — `author-1` registers a pinned echo tool
   (`register_tool`, `publish=True`); `caller-1` invokes it via
   `tool_registry.call_tool` and then runs `attest_tools` (the cognitive
   attestation — the only mint-counting usage). A `WorldService` is wired
   to the tool_store's `manifest_sink`/`event_sink` exactly like
   `atn/autonet_service.py` does; the tool_used receipts are captured off
   the event sink and the registration sprout is read from the
   WorldService epoch buffer.
5. **Consensus** — the REAL federated close. Registration sprouts (author
   signing key) and receipts (caller signing key) become canonical
   `EventBatch` chains, driven through `FederatedCloseDriver.run()` with a
   mocked gossip (the `tests/test_tool_mint` TestDriverCarryOver pattern).
   The `agent_owner_map` is **read back from chain** (`getAgentOwner` for
   both agents), keyed by the same 0x addresses the events now carry.
   Verifies the mint credits `author-1`, `attesters == 1`, and a **control
   case**: setting the caller's owner equal to the author's in the map
   drops the mint to zero — proving the chain-fed same-owner exclusion.
6. **Settlement** — the winning submitter anchors the epoch
   (`submitAnchor` via `EpochAnchorer`), `author-1` records its own mint
   (`recordTrainingForEpoch` via `AuthoritativeChainSubmitter`, mint-merkle
   proof against the anchored root). Verifies `agentReputation` and the
   ATN `balanceOf` both moved on chain by the scaled `agent_mint[author]`
   amount. Then registers one Service on `ServiceRegistry` and runs one
   `ServiceEscrow` round-trip with MockERC20 (client funds → provider
   delivers → client releases → provider balance +ask).
7. **Teardown** — kills the Hardhat node, removes tmp dirs, prints the
   scoreboard.

## Expected output (sketch)

```
  [PASS] Stage 1 — chain up (hardhat node + deploy)
  [PASS] Stage 2 — fleet (two ATN runtimes, owner-rooted lineage)
           author_lineage_root=0xf39Fd6e5…   author_addr=0x…
  [PASS] Stage 3 — chain identity (owner-bound registration)
           author_owner_onchain=0xf39Fd6e5…  caller_owner_onchain=0x7099797…
           distinct_owners=True
  [PASS] Stage 4 — tool economy (register + invoke + attest)
           registration_events=1  attested_receipts=1
  [PASS] Stage 5 — consensus (federated close + owner-map exclusion)
           tool_mint_raw=4.15888308  agent_mint_raw=8.15888308
           control_same_owner_mint=0 (excluded)
  [PASS] Stage 6 — settlement (anchor + recordTraining + service)
           reputation_delta=8158883  atn_delta=8158883
           service_id=1  escrow_provider_delta=1000
  [PASS] Stage 7 — teardown
  7 PASS / 0 FAIL / 0 SKIP
```

## Seams this e2e revealed

These are the composition frictions found while wiring the loop
end-to-end. None block the loop (the script works around each), but each
is a real observation about where the pieces don't yet click cleanly.

1. **Registration sprout carries `author_post=False` → zero-standing tool
   mint.** `WorldService.submit_tool_manifest` emits the registration
   `SubClaimSprouted` with `author_post=False`. In the default
   **ledger-pricing** federated close (no equilibration), the manifest
   claim node then has zero `net_score` standing, so
   `mint = standing × usage_term = 0` — a freshly-registered, genuinely-
   attested pinned tool mints NOTHING. The unit fixture
   (`tests/test_tool_mint._registration_event`) hand-sets
   `author_post=True` to get standing 1, which is why the tests pass while
   the live daemon path would mint zero until PRO debate accrues standing
   organically. The script sets `author_post=True` on the buffered
   registration events (documented inline) to exercise the mint. **Open
   question for the owner:** should tool registration stamp an author post
   (giving the manifest immediate unit standing, matching `submit_con`'s
   `author_post=True` convention), or is a tool author supposed to earn
   standing only through subsequent debate? This is the single most
   load-bearing seam — the whole "tools mint to their author" claim hinges
   on it.

2. **Tool registration events don't reach the `event_sink`.** Only
   `tool_used` receipts flow through `tool_store.event_sink`; the manifest
   registration goes through `manifest_sink → submit_tool_manifest`, whose
   sprout lands in the WorldService's epoch buffer, not the sink. So a
   consumer that only taps `event_sink` (the natural place to collect
   "everything for the epoch batch") is missing the registration and gets
   an empty mint. The script reads registration events out of
   `ws._epoch_events` instead. A single unified "all substrate events for
   this epoch" tap would remove this footgun.

3. **String agent-id vs 0x address at the mint→chain boundary.** The tool
   events carry the daemon-local agent id (`author-1`, `caller-1`) in
   `author`/`tool_author`/`author_agent`. `federated_epoch_close` keys
   `agent_mint` by that string. But `mint_merkle` only admits keys that
   parse as 0x addresses (the contract can only verify `msg.sender`), so a
   string-keyed mint has **no on-chain claim path** — it silently drops at
   the merkle-leaf stage. Production bridges this with
   `world_substrate_feed.ResolvedAgentIdentity` (local id → 0x). The script
   replicates that remap explicitly before building the canonical batches.
   The seam: nothing in the tool-mint path itself enforces or documents
   that the author field must be an address by the time it reaches the
   anchor; it's an implicit contract between the feed resolver and the
   merkle builder.

4. **On-chain amount is `agent_mint[author]`, not `tool_mint[digest]`.**
   The tool-author mint is merged into the broader `agent_mint` map
   (through the violator-pays gate) BEFORE anchoring, so the amount that
   hits the chain (`8.15…`) is larger than the standalone tool_mint entry
   (`4.15…`). Verifying the chain delta against `tool_mint[digest]` fails;
   the correct source is `agent_mint[author]`. Obvious in hindsight, but a
   trap for anyone reasoning "the tool minted X, so X should appear on
   chain."

5. **Hardhat node's `chainId` is 1337, not 31337.** `npx hardhat node`
   serves the in-process `hardhat` network (`hardhat.config.js`:
   `chainId: 1337`), while the daemon's chain config defaults to `31337`
   (the eth_tester default). Signing txs for 31337 gets rejected with
   "incompatible EIP-155 transaction, signed for another chain." The
   script pins `CHAIN_ID = 1337`. Worth noting for anyone pointing the
   real daemon at a local `hardhat node`.

6. **Hardhat node stdout must be drained.** `npx hardhat node` logs every
   RPC call; if its stdout PIPE buffer fills, the process blocks and the
   deploy side sees an undici `HeadersTimeoutError`. The script sends the
   node's stdout/stderr to DEVNULL. (A subtle subprocess-management trap,
   not an architecture seam, but it cost a run.)

7. **Owner-binding EIP-712 is under active rename.** During this work the
   contract's owner-binding struct was renamed Sponsorship → OwnerBinding
   (`registerAgentSponsored` → `registerAgentBound`, `sponsorshipHash` →
   `bindingHash`, `sponsorshipNonce` → `bindingNonce`, event
   `AgentSponsored` → `OwnerBound`) and gained a mandatory `nonce` in the
   typed struct. The script is written to survive either shape: it
   discovers the method/view/event names by ABI introspection and — rather
   than reconstruct the typed-data struct itself — asks the **contract's
   own `bindingHash`/`sponsorshipHash` view** for the exact digest and
   signs that raw 32-byte digest with the owner key. This sidesteps all
   struct-field/typehash drift: whatever the contract hashes is what gets
   signed. Any future field addition to the binding struct needs no script
   change as long as a hash view is exposed.
```

# Local end-to-end: the venture loop composes

`scripts/local_e2e_venture_loop.py` is the sibling capstone for the
venture-loop contracts + rails (VentureVault + factory, CharterAnchor,
TrialRunner, the evidence CON rail). It **reuses the tool-economy
harness** wholesale (`start_hardhat`, `deploy_contracts`,
`resolve_binding_shape`/`sponsored_register`, `send_agent_tx`,
`make_runtime`, `register_local_agent`, the `Board` scoreboard and
`_fake_embedder`) and bolts CharterAnchor + VentureVaultFactory onto the
base deploy. It stands up ONE daemon runtime, registers a fleet of
owner-bound agents against a live Hardhat chain, and drives the whole
venture lifecycle end-to-end.

## Run

```bash
python scripts/local_e2e_venture_loop.py
```

Same prerequisites as the tool-economy e2e (`npm install`,
`npx hardhat compile`, Python deps). `--from-stage` / `--to-stage`
letter flags exist for iteration, but stages A–F share on-chain + daemon
state, so a real green run uses no flags.

## What it proves

Eight stages (0 bootstrap + A–G), each verified:

- **0 chain up** — base deploy (Substrate + ServiceMarket + MockERC20)
  plus a second inline deployer for `CharterAnchor` (governor =
  `account[0]`) and `VentureVaultFactory`.
- **A CharterAnchor** — anchor the live `charter_hash()` (v1), assert the
  daemon's `verify_charter_against_anchor` MATCHES; anchor a bogus hash
  (v2, append-only) and assert the drift-warning path fires
  (`match=False`); re-anchor the real hash (v3) and assert current
  matches again.
- **B venture** — the venture agent publishes a `venture_prospectus` blob
  (echo battery, `pass_threshold=1.0`); a VentureVault is deployed via the
  factory bound to that agent (`termsBps=6000`, the 60% backer split).
- **C backers** — two funded accounts mint ATN via the substrate's only
  mint path (single-leaf `recordTrainingForEpoch`), approve + contribute
  to `raiseMin`, `openTrials`.
- **D trials** — two verifier agents (distinct owners) run
  `TrialRunner.run_trial` against the prospectus through a local echo
  transport, submit `attestTrial` on chain from their own keys;
  greenlight; a verifier claims its fee slice.
- **E tranches + revenue** — agent claims tranche 1; a customer pays via
  `payForInference(vault, …)`; `claimRevenue` splits 60/40; backers claim
  pro-rata (360/240, asserted to the wei); a ≥1/3 backer halts the next
  tranche (asserted unclaimable), un-halts, and the tranche is claimable
  again.
- **F evidence rail** — the author publishes a deliberately-buggy pinned
  tool; a disputer files an evidence-bearing CON; a third daemon runs
  `check_evidence` → confirms → posts support. Two mini-closes (with vs
  without recruited support) show the supported CON's standing is double
  the naked CON's (52 vs 26) and drags the disputed tool's net_score
  strictly further negative (+2 → −24) — the close prices the CON above
  the naked equivalent.
- **G** — totals + teardown (Hardhat killed, tmp + deploy records removed).

## Seams this e2e revealed

Additional to the tool-economy seams above:

8. **`close_epoch(apply_gate=False)` leaves `node_mint` empty for a
   single-epoch evidence comparison.** The reliable pricing signal for
   "did the recruited support move standing" is the LIVE `net_score` read
   via `WorldService.read_node_scores()` (equilibrated state), not the
   close record's `node_mint` diagnostic — which stays 0.0 for a graph
   that has not accumulated an emission-priced usage term. The script
   asserts on `read_node_scores()` (CON node standing and the disputed
   tool's net_score) and keeps the mini-close `node_mint` figures as
   labeled diagnostics only.

9. **Tranche auto-vesting from tx latency makes the halt boundary
   nondeterministic.** With a short `tranchePeriod` (seconds), the real
   block-time that elapses across the greenlight → claim → halt tx
   sequence silently vests a second tranche, so `haltPeriodVested`
   snapshots past the tranche the test wants to freeze. The script uses a
   large `tranchePeriod` (100_000s) and drives vesting purely with
   `evm_increaseTime`, so the halt boundary is exact.

10. **`submit_tool_manifest` fails SILENTLY (`{"error": …}`, no
    `manifest_digest`) on an incomplete manifest.** Seeding a fresh
    WorldService with the disputed tool's claim node needs a FULLY valid
    manifest — `kind: "tool_manifest"` and (for pinned) a non-empty
    `code_digest` — or `validate_manifest` rejects it and the receipt has
    no `manifest_digest` key. Easy to miss because the daemon path builds
    the manifest for you; hand-building one for a direct
    `submit_tool_manifest` call must satisfy `_REQUIRED_FIELDS`.

11. **CharterAnchor is append-only, so "fix the drift" is a NEW version.**
    Re-anchoring the correct charter hash after a bogus anchor does not
    edit v2 — it appends v3. `currentCharter()` then tracks v3, and
    `versionCount` is 3. The forward-only doctrine (no rollback) is
    load-bearing: the test asserts `chain_version == 3` after the
    realignment, not a return to v1.

12. **The scoreboard's arrow/box glyphs crash a cp1252 Windows console.**
    The final `print(board.render())` carries `→` and `═`; on a stock
    Windows console (`cp1252`) that raises `UnicodeEncodeError`. The
    script reconfigures `sys.stdout`/`sys.stderr` to utf-8 before running.
```

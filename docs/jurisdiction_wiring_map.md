# Jurisdiction Wiring Map

**Status: research report, 2026-07-05 — since SUPERSEDED IN PART by the live
attach (2026-07-10). Read the "Live attach" banner below before relying on
any "design, not built" / "future work" framing in the body.** The original
survey mapped the "tokenized project → on-chain jurisdiction" attachment across
three codebases and the three wiring points the user named, with a `file:line`
citation per claim. Much of it now describes what WAS built.

## Live attach (2026-07-10) — what actually shipped

The canonical Autonet jurisdiction is a **werule-platform-created DAO** (a
HomebaseDAO governor + timelock + RepToken + Registry suite), with the autonet
contract stack **attached post-hoc**: the Substrate/vault/rep-claim/charter
contracts were deployed with the jurisdiction's governor/timelock as their
constructor params, then wired in by governance proposals —
`RepToken.setMinter` (grants the `AutonetRepClaim` contract the right to mint
REP), plus Registry self-description keys (`jurisdiction.parity.*` for the
vault buy rail, `jurisdiction.autonet.{substrate,vault,rep_claim}` naming the
attached contracts).

**Addresses of record live in `registry.json`** at the repo root (daemons
fetch it from GitHub raw master, cached at `~/.atn/registry.json`); do not
trust any address literal in the body of this doc. Discovery root is the DAO
governor (`atn/jurisdiction.py: GOVERNOR_ADDRESS`); `autonet.yaml`
`blockchain.contracts.AutonetDAO` overrides it.

Two claims in the body are now **outdated by the fees-only / REP-from-earnings
decision (2026-07-10)** and are corrected in section B.3 below: Substrate.sol
is now a PURE MONEY contract — it has **no `agentReputation` surface** and mints
**no reputation**. REP is claimed DAO-side (`AutonetRepClaim`, 1:1 on ratified
ATN earnings — `agentMintTotal` + `serviceEarnings`); the read-only
"reputation mirror" this report contemplated was **retired** in favor of that
pull-claim rail. Treat every "mints reputation+ATN in lockstep" /
"`agentReputation` soulbound" phrasing below as historical.

Codebases surveyed:

- **trustless-contracts** — `C:\code\dao\trustless-contracts`, branch
  `feature/coin-offering` (the boats branch; confirmed checked out). The DAO /
  jurisdiction / project economy. All `contracts/*.sol` citations below are
  relative to this repo.
- **autonet** — `C:\code\autonet`, branch `master`. The tool-substrate economy:
  `contracts/core/Substrate.sol`, `contracts/core/ServiceMarket.sol`.
- **boats** — `D:\videos\SF\manifesting\from_now\autonomous-charter`. Design
  docs only (see finding below).

---

## Finding 0 — where the boats attachment logic actually lives

**The boats "attachment logic" is NOT in the boats directory.** The tree at
`D:\videos\SF\manifesting\from_now\autonomous-charter\` contains **zero**
`.sol` / `.js` / `.ts` / `.py` files — only Markdown design docs, one geometry
file (`boat_spec.json`, pure hull/wingsail data), and Blender assets. The only
occurrences of "jurisdiction" in that tree are *maritime regulatory*
jurisdictions (Malta/Norway/UK) in `feasibility.md`, not the on-chain DAO.

The docs point to the code: `BUILD_STATE.md:8-12` states the contracts live in
`c:\code\dao\trustless-contracts` branch `feature/coin-offering` and the
frontend in `c:\code\boats` branch `feature/token-reframe`. That branch is
checked out and the code is real: `CoinOffering.sol`'s header docstring is
literally titled **"THE BOATS MODEL"** (`CoinOffering.sol:23`). So "I already
created the attachment logic" is accurate — it is built + locally-tested code,
parked in the contracts repo, not the strategy folder.

**Two stale-doc traps** for anyone reading only the boats folder:

1. `ARCHITECTURE_NOTES.md:9` describes a **two-token split** (transferable ATNB
   for value, non-transferable earned rep for governance). This is
   **superseded**: `BUILD_STATE.md:28` records the reversal — **ATNB *is* the
   transferable governance token** (`ERC20Votes`), buying grants governance,
   auto-delegated to the team, reclaimable. The shipped code follows
   BUILD_STATE (`CoinOffering.sol:29-40`).
2. The trustless.business **milestone-Project + escrow + arbiter** path
   (`investment_scenarios.md` Path C, 7 milestone projects, €620k) is
   **described but not deployed** — only the CoinOffering raise + dividend loop
   is actually built and tested.

`governance_and_safety.md` in the boats tree is about the **physical boat's
onboard AI** (Coast Guard > onboard AI > passenger governance authority
hierarchy, hard-constraint geofences, non-removable human mayday baseline). It
is a *separate* governance layer from the token DAO and contains no on-chain
heartbeat / work-halt mechanism (that concept lives in the autonet repo, not
here).

---

## A. The attachment mechanism as it exists today

### A.1 The jurisdiction is a DAO suite

A "jurisdiction" in trustless-contracts is a five-contract suite, brought up by
`TrustlessFactory` in three transactions (`scripts/deploy-autonet-jurisdiction.js`):

| Contract | Role |
|---|---|
| `HomebaseDAO` (`Dao.sol`) | The governor. `GovernorVotes(IVotes(repToken))` + timelock (`Dao.sol:12-26`). |
| `TimelockController` | The executor / **owner-of-everything** after finalize. Governance acts only through it. |
| `RepToken` | Governance + reputation token (IVotes, `RepToken.sol:31`). The vote weight AND the project-creation / appeal gate. |
| `Registry` | Treasury + key-value config store, with a rolling-window spend cap (`Registry.sol:157-175`). |
| `Economy` | Project ledger + clone factory for work projects; holds all DAO-governed fee/quorum params (`IGovernedEconomy`). |

Bring-up flow (`TrustlessFactory.sol`):

1. `deployInfrastructure(...)` (`:94`) → Economy + Timelock + Registry.
2. `deployDAOToken(...)` (`:105`) → RepToken (via `RepTokenFactory`, **hardcoded
   `transferrable=false` for Economy DAOs**, `:119-120`) + HomebaseDAO.
3. `configureAndFinalize(...)` (`:132`) — **the attach + handoff step**:
   `economy.setImplementations(native, erc20)` (`:151`);
   `economy.setDaoAddresses(timelock, registry, governor, repToken)` (`:167`,
   **one-time-only**, `Economy.sol:240`); `repToken.setEconomyAddress` (`:159`);
   `registry.setJurisdictionAddress(repToken)` (`:160`); then transfers admin /
   ownership to the Timelock and grants the DAO `PROPOSER_ROLE`, `address(0)`
   `EXECUTOR_ROLE` (`:168-176`). After this the Timelock owns the jurisdiction.

### A.2 Two flavors of "tokenized project" attach to that suite

**Flavor 1 — work projects** (`NativeProject` / `ERC20Project`, cloned by
`Economy.createProject` / `createERC20Project`, `Economy.sol:163,187`). Each
clone is initialized with `daoTimelock` + `daoGovernor` (`NativeProject.sol:113-114`)
and calls back `economy.registerProjectRoles(...)` (`:118`). "Tokenized" here =
the payment rail is an ERC20 (`ERC20Project`) vs native coin (`NativeProject`).
**These carry an arbiter + appeal-to-DAO state machine** (detail in B.2).

**Flavor 2 — the boats CoinOffering** (`CoinOffering.sol`, "THE BOATS MODEL").
Not a clone of Project — a standalone router:

```
buyer --USDC--> CoinOffering.buy(paymentToken, amount)
                  |  atnbOut = amount * parity / 1e18   (CoinOffering.sol:173)
                  |  parity read from Registry key "jurisdiction.parity.<tokenHex>" (:229-237)
                  |
                  +--1--> USDC routed straight to Registry treasury (:193)  [never custodied]
                  +--2--> ATNB minted to buyer (:199), grants governance,
                          token auto-delegates votes to execution team
                          (RepToken.defaultDelegate, :326-334) on first receipt
                  |
        saleCap (:83) is the fundraising goal; buy reverts past it (:177)
```

Revenue return path (`RepToken` + `Registry`, demonstrated in `demoDividends.js`):
project revenue → Registry treasury → `earmarkFunds` + `startNewPassiveIncomeEpoch`
(`RepToken.sol:182`) → each holder `claimPassiveIncome(epochId)` (`:204`) pays
`balanceAtSnapshot × budget / totalAtSnapshot` (`:210-214`). **Dividends track
balance, not votes** — delegating your vote to the team does not forfeit your
revenue share (`demoDividends.js:64,95-102`).

### A.3 Assumptions that still hold vs. ones the new economy broke

**Still hold (jurisdiction side):**

- The DAO-suite topology (governor + timelock + votes-token + registry-treasury)
  is intact and is what any tokenized project attaches to.
- `Registry` as treasury + parity store + spend-cap is reusable as-is by any new
  instrument, including a venture vault (the parity key convention is already
  shared between CoinOffering and RepToken, `CoinOffering.sol:44-54`).
- RepToken-as-vote-weight and the passive-income / delegate-reward loops are
  independent of what the money-raising instrument is.

**Broken / superseded by the new economy:**

- **Arbiter model vs. no-judge doctrine.** `ERC20Project` / `NativeProject` are
  built around a named `arbiter` + `daoOverrule` appeal court (`NativeProject.sol:330,378`).
  The autonet venture-vault doctrine **deletes arbitration wholesale**
  (`venture_vault_design.md`: "DELETE arbiter/appeals/overrule (escrow-deletion
  doctrine)"). ServiceMarket already committed to this: the postpaid escrow was
  deleted because "an unarbitrated escrow cannot know delivery truth"
  (`ServiceMarket.sol:26-33`); settlement is prepaid channel vouchers, no oracle.
- **Old Autonet.sol epoch/service concepts vs. Substrate.sol.** The
  trustless-contracts `Autonet.sol` (RPB) models epochs, services referencing a
  `projectContract`, capability-scorecard-weighted emission, backer/participant
  reward claims. Autonet's `Substrate.sol` replaced all of this from scratch
  (`Substrate.sol:9-13,32-42`): epoch **anchoring only** (the mint math is the
  off-chain deterministic close), agent registry with owner-binding, training
  records minting reputation+ATN in lockstep (`:710-760`) — **superseded
  2026-07-10: Substrate now mints ATN ONLY (money-only 2-field leaf) and
  accrues the `agentMintTotal` earnings ledger; REP is a DAO-side pull claim,
  see the banner and B.3** — and a labeled `payForService` transfer that takes
  the 2.5% service fee and accrues a per-recipient `serviceEarnings` ledger.
  Capability scoring, evolution
  proposals, sponsorship hierarchies are explicitly called out as
  "pre-substrate concepts" that do **not** live on Substrate.sol (`:40-42`).
- **Dead bridge script.** `autonet/scripts/e2e_jurisdiction_integration.py`
  models the OLD attachment: `Project.sol`, `InferenceProviderFactory`,
  `InferenceProviderBridge`, `ATNToken`, `setMatureModel`, PT-holder revenue —
  "The bridge implements IInferenceProvider compatible with Jurisdiction's
  Autonet.sol" (`:383`). **None of those contracts exist in autonet anymore**
  (`contracts/core/` holds only `Substrate.sol` + `ServiceMarket.sol`; grep for
  `InferenceProviderBridge`/`IInferenceProvider` in `*.sol` returns nothing).
  This script is dead pre-substrate history, useful only as a record of the
  *original* bridge-based attachment intent.

---

## B. The three wiring points

### B.1 Governance changing CORE VALUES (the 6-root charter)

**BUILT since this survey.** The "design, not built" anchor below shipped as
`contracts/core/CharterAnchor.sol` — a standalone governed anchor (NOT the
Registry-key option this report leaned toward, and NOT a governed slot on
Substrate.sol, which stays ungoverned). It is deployed on the live jurisdiction
with charter v1 anchored on-chain (governor = the DAO timelock); the daemon
verifies its local `charter_hash()` against `currentCharter()` and warns on
drift. See `docs/charter_anchor.md`. The forward-only / migration-deferred
reasoning below still holds. The rest is the historical design survey.

**Where the charter lives today.** The charter axes are a Python constant in the
off-chain deterministic close: `CHARTER` in
`nodes/common/world_model_substrate/adapter.py:58-92` — six roots
(`life_precious`, `self_preservation`, `promotion_of_intelligence`, `evolution`,
`correctness`, `simplicity`), each an axis index + thesis string + `veto_floor`.
`build_charter_world()` (`:96-131`) materializes them into the world every daemon
equilibrates against. The identical list is re-declared for the LLM scorer
(`llm_score_turn.py:31-36`). **There is no charter version, hash, or anchor
anywhere on-chain today** — a repo-wide grep for
`charter_version|charterVersion|charter_hash|charterHash|CHARTER_VERSION` returns
nothing. The charter is a hardcoded constant, changed only by editing the file
and every daemon re-pulling the code.

**What a governor proposal changing it would look like.** The values live in the
off-chain close, so the on-chain act cannot *be* the charter — it can only
**anchor which charter version is canonical**, exactly as the DAO already anchors
the constitutional prompt in trustless-contracts:
`EvolutionProposal.updateRPBPrompt(promptCid)` is `onlyTimelock` and writes
`rpb.prompt.v<n>` / `rpb.prompt.current` / `rpb.prompt.version` into the Registry
(`EvolutionProposal.sol:552-565`). The charter is the substrate-native analogue
of that RPB prompt.

**Minimal on-chain anchor (design, not built):** a single governed value — a
`charterVersionHash` (sha256 of the canonical `CHARTER` blob) plus a version
counter — written by governance. Two placement options:

- On **Substrate.sol**: add a governed `charterVersion` / `charterHash` pair set
  by a privileged setter. But Substrate.sol today has **no admin keys and no
  governance surface at all** (it is deliberately just the attribution layer,
  `Substrate.sol:36-42`), so this would introduce the first governed slot on an
  intentionally ungoverned contract. It also has no owner/timelock wired.
- In the **jurisdiction Registry** (trustless-contracts), reusing the exact
  `updateRPBPrompt` pattern: a `charter.v<n>` / `charter.current` /
  `charter.version` key set `onlyTimelock`. This keeps Substrate.sol pure and
  puts value-governance where the governor already lives.

The Registry option is the cleaner fit: the charter is a *jurisdiction standard*
(same category as the RPB prompt), and CLAUDE.md explicitly flags "constitutional
prompt wording" as a user-only / governance decision. Daemons would read the
anchored version hash and refuse to close against a charter blob whose sha256
doesn't match (fail-closed, mirroring `manifest_meta`-vs-chain mismatch being
CON-able, `tool_substrate.md:405-408`).

**Migration / forward-only implications.** The charter *is* the coordinate space
every historical claim was embedded in (6-dim head, `adapter.py:113-116`). Change
the roots and every past node's alignment coordinates change meaning — you cannot
re-equilibrate history against a new basis and keep bit-identical replay
(replay/float-order is consensus-relevant, `epoch_economics.md:130-135`). So a
charter change is inherently a **forward-only fork boundary**, consistent with
the project's "no rollback, move forward through governance" principle
(CLAUDE.md). Practical shape: version N+1 takes effect at an epoch boundary; old
epochs stay anchored under version N; the version counter monotonically
increases; there is no re-scoring of the past. This matches `tool_substrate.md`'s
note that protocol-floor / charter changes are "redeploy/governance" events
deliberately deferred (`tool_substrate.md:262-264`).

### B.2 Attaching the venture vault as a "tokenized project"

**BUILT since this survey.** The venture vault shipped as
`contracts/core/VentureVault.sol` (deploy-per-venture via `VentureVaultFactory`,
ATN-only, no arbiter — halt vote instead of appeals), exercised end to end by
`scripts/local_e2e_venture_loop.py`. The port analysis below is the historical
design reasoning; the arbiter WAS deleted and tranche-gating WAS adopted as
described. The following is the survey as written.

The venture vault (`venture_vault_design.md`) is the boats CoinOffering's
successor instrument. Attaching it under the jurisdiction is mostly a **port from
`ERC20Project` + CoinOffering with the arbiter surgically removed**.

**What ports over (keep):**

- The **jurisdiction wiring block** from Project init: store `daoTimelock` +
  `daoGovernor` at init (`NativeProject.sol:113-114`), register with the Economy
  (`registerProjectRoles`, `:118`). The vault attaches the same way a work
  project does — the DAO learns about it, it learns its governor/timelock.
- The **contribution ledger + pro-rata math + `BackerVoteCast`** (design doc
  names these explicitly as "keep"). Backer contributions and pro-rata revenue
  claims map directly to CoinOffering's `totalMinted` / per-holder-balance
  dividend math (`CoinOffering.sol:88`, `RepToken.claimPassiveIncome:210-214`).
- The **Registry-treasury routing + spend-cap** (`CoinOffering.buy` routes cash
  straight to Registry, `:193`; Registry rolling-window cap, `Registry.sol:157-175`).
  Boats already identified "schedule-gated treasury release" as the one missing
  platform feature (`ARCHITECTURE_NOTES.md:14-28`), which became the Registry
  spend-guard (`BUILD_STATE.md:21`). The vault's **tranche gating** is the same
  idea one level up.
- The **parity / registry-key convention** for raise pricing
  (`CoinOffering.sol:44-54,229-237`).
- **`daoVeto()`** as the governance kill-switch (`NativeProject.sol:534-540`):
  100%-refund close callable only by the Timelock. This survives the arbiter
  deletion because it is a *future*-facing halt (refund un-spent funds), not a
  *past*-judging ruling — squarely inside the doctrine's allowed set.

**What gets deleted (the arbiter):**

- `arbitrate(percent, rulingHash)` (`NativeProject.sol:330` / `ERC20Project.sol:324`)
- `appeal(proposalId, targets)` (`NativeProject.sol:344` / `:338`)
- `daoOverrule(percent, rulingHash)` (`NativeProject.sol:378` / `:372`)
- `finalizeArbitration()` (`NativeProject.sol:388`)
- the whole `Stage.Dispute/Appealable/Appeal` machine and
  `disputeResolution`/`_finalizeDispute` (`NativeProject.sol:399-424`)
- Economy's arbiter-related params (`setArbitrationFee`, `Economy.sol:260`) become
  dead for vaults.

Replaced by **tranche gating** (`venture_vault_design.md`): "backers vote
continue-the-mission, can halt future tranches, never claw back spent ones."
This is the third instance of the ratified theft-ceiling pattern
(voucher→item in ServiceMarket, channel→client, **tranche→investor** here). Note
the doctrine's *reason* the judge is deletable: "arbitration judges the past
(needs shared truth); tranche voting allocates the future (each backer's private
judgment suffices)."

**What the DAO would actually govern about a vault:**

- **Parameters, not disputes.** By analogy to what the DAO governs on Economy
  today — fees, quorum thresholds, cooling-off/appeal periods, project threshold,
  `maxImmediateBps` (`Economy.sol:248-296`) — the vault's governed knobs would be
  the *raise/tranche* parameters: min/max caps, tranche schedule bounds,
  verifier-trial quorum (`N-of-M` greenlight threshold), backer-halt quorum.
  These are "direction" decisions (see B.3).
- **The greenlight gate**, if the DAO owns the verifier-trial pipeline (the
  vetting pipeline generalized: "moats can't be read so they're PROBED").
- **The kill-switch** (`daoVeto`-style forward halt).
- **NOT disputes** — there is no dispute surface to govern; that is the whole
  point of the doctrine. Revenue arrival settles the private phase retroactively;
  no judge is ever seated.

**One flag carried from the design doc:** revenue lands in the **vault**, not the
agent wallet, and backers hold a claim on **vault net revenue, never the
artifact** (handing over pinned code leaks the moat, since published code is a
commons — `venture_vault_design.md`). Any port must keep revenue custody in the
vault contract, distinct from the agent's ATN balance on Substrate.sol.

### B.3 Reputation as governance weight

**RESOLVED 2026-07-10 (this section's question is now answered in code).**
The decision went further than the "mirror, don't bridge" lean below:
Substrate.sol's `agentReputation` was **deleted**. Substrate is a pure money
contract that mints ATN only and accrues two earnings ledgers
(`agentMintTotal` = cumulative tool-pool mint, `serviceEarnings` = cumulative
net service revenue). **REP is minted DAO-side by `AutonetRepClaim`** as a pull
claim on those ledgers at a hardcoded **1:1** (you cannot lie about REP because
you cannot lie about ATN you provably received). REP is the DAO's `RepToken`
(IVotes) — so REP is voice/direction weight, held via the claim rail, and the
federated close weights reviews by REP share read from `RepToken` checkpoints
pinned to the previous anchor. The read-only "reputation mirror" contemplated
below was **retired** in favor of this claim rail (`registry.json`
`reputation_mirror` is empty; `rep_claim` holds the live address). The rest of
this section is the historical reasoning that led there.

Two reputation ledgers exist, in different repos, with different properties:

| | Substrate.sol `agentReputation` | RepToken (trustless-contracts) |
|---|---|---|
| Holder | **agent** (on-chain agent key) | **human** (wallet) |
| Transferable | No — soulbound, monotonic, no transfer fn (`Substrate.sol:663-680`) | Configurable; **Economy-DAO instances hardcoded non-transferable** (`TrustlessFactory.sol:120`) |
| Minted by | training close, in lockstep with ATN (`Substrate.sol:741-752`) | economic activity pull (`claimReputationFromEconomy`, `RepToken.sol:120`), CoinOffering mint, admin |
| Is it IVotes? | **No** — Substrate.sol has no voting/governance surface | **Yes** — it *is* the DAO vote token (`Dao.sol:23`) |

**Bridge, mirror, or replace?** The two measure different things and belong to
different entities (agent vs. human), so a naive replace is wrong. The design
doc's ratified wall is the deciding constraint:

> **"Stake votes direction; reputation prices quality — never let money buy
> standing (the verdict layer dies otherwise)."** (`venture_vault_design.md`)

Mapping DAO functions onto that wall:

- **Direction decisions** (which way the jurisdiction goes — allocate the
  future): DAO proposals/votes on parameters, tranche continuation, charter
  version adoption, treasury allocation, greenlight thresholds. These are
  **stake/vote-weighted** — RepToken's existing IVotes role. Keep RepToken as
  the direction-voting instrument.
- **Quality decisions** (what is good work — judge the produced): substrate
  standing / `agentReputation`. This is the verdict layer, priced by debate, and
  the wall says money must never buy it. It must **not** become vote weight,
  because that would let training-mint (which mints ATN too) translate into
  governance direction, collapsing the wall.

**Therefore: mirror, don't bridge or replace.** `agentReputation` stays soulbound
and off the DAO vote path; RepToken stays the direction vote. The one useful
on-chain link is a **read-only mirror**: the indexer already mirrors
Substrate.sol agent/endpoint events to Firestore (CLAUDE.md cross-codebase map);
a jurisdiction that *wanted* to weight a specific quality-gated decision by
substrate standing could read `agentReputation(agent)` as an input, but it should
never be minted into RepToken (that would make quality buy direction). The
cleanest statement: **RepToken = direction weight (human, IVotes);
agentReputation = quality standing (agent, soulbound, verdict layer); no token
flows between them.**

Open sub-point: agents are not humans and RepToken votes are human-held. If
agent quality should ever influence direction, it must route through the agent's
*owner* wallet (the owner-binding on Substrate.sol, `:394,449-481`), not through
minting RepToken to an agent key — but that is a direction the user must set, not
a code fact (see D).

---

## C. Porting inventory

### Carries over (reuse largely as-is)

| Piece | Source | Note |
|---|---|---|
| DAO-suite topology (governor+timelock+votes+registry+economy) | `TrustlessFactory.sol`, `Dao.sol` | The thing any tokenized project attaches to. Intact. |
| Registry treasury + rolling spend-cap | `Registry.sol:157-175,177-204` | Vault routes revenue here; cap survives vote capture. |
| Parity / registry-key pricing convention | `CoinOffering.sol:44-54,229-237` | Shared by CoinOffering + RepToken; reuse for vault raise pricing. |
| Passive-income dividend loop (balance-snapshot pro-rata) | `RepToken.sol:182-218`, `demoDividends.js` | Model for vault net-revenue splits; dividends track balance not votes. |
| Delegate-reward ("incentivized representation") loop | `RepToken.sol:193,222-234` | Pays accumulated delegated weight; independent of raise instrument. |
| Jurisdiction-wiring init block (store timelock+governor, registerProjectRoles) | `NativeProject.sol:113-118`, `Economy.sol:230` | How the vault announces itself to the DAO. |
| CoinOffering router (mint-to-buyer, cash-to-treasury, saleCap goal) | `CoinOffering.sol` (whole) | The boats raise; vault generalizes it (tranches instead of single cap). |
| Governance kill-switch (forward refund halt) | `NativeProject.sol:534-540` (`daoVeto`) | Survives arbiter deletion — future-facing, not a ruling. |
| RepToken as IVotes direction weight | `Dao.sol:23`, `RepToken.sol:31` | Stays the direction-vote instrument. |
| Charter-anchor pattern (governed version key in Registry) | `EvolutionProposal.sol:552-565` (`updateRPBPrompt`) | Exact template for a `charter.version` anchor. |

### Needs rewriting

| Piece | Why |
|---|---|
| Single `saleCap` → **tranche schedule + backer-halt vote** | Vault funds a burn rate metered over time, not a one-shot cap (`venture_vault_design.md`). |
| Raise flow → **prospectus manifest → escrowed refundable raise → verifier trials → N-of-M greenlight → tranches → splits** | The no-judge fundraise sequence; escrow must be refundable with mechanical full-refund on greenlight-fail, no cashback judgment. |
| Economy work-project vetting → **verifier-trial pipeline** (vetting generalized to probe moats) | Moats "can't be read so they're PROBED" (black-box trials, distinct fleets). Ports the tool-substrate vetting (`tool_substrate.md:266-332`) up a level. |
| Charter as hardcoded constant → **charter + governed version anchor** | `adapter.py:58-92` has no version; add anchor + fail-closed daemon check. |
| RepToken↔agentReputation relationship → **explicit read-only mirror, no token flow** | Enforce the direction/quality wall in code (B.3). |

### Dead (do not port)

| Piece | Where | Status |
|---|---|---|
| `arbitrate` / `appeal` / `daoOverrule` / `finalizeArbitration` + Dispute state machine | `NativeProject.sol:330-424`, `ERC20Project.sol:324-380` | **Deleted by no-judge doctrine.** Only `daoVeto` (forward halt) survives. |
| `InferenceProviderBridge` / `InferenceProviderFactory` / `Project.sol` / `ATNToken` bridge attachment | `autonet/scripts/e2e_jurisdiction_integration.py` (refs only; contracts gone) | Pre-substrate; contracts no longer exist in autonet. Dead history. |
| trustless-contracts `Autonet.sol` epoch/service/capability-scorecard emission | `Autonet.sol`, `CapabilityScorecard.sol` | Superseded by Substrate.sol's anchor-only + off-chain close. The capability-gap *pricing idea* survives conceptually (`tool_substrate.md:129-139` atlas gaps), but not this contract. |
| Two-token split (transferable value + soulbound gov) for boats | `ARCHITECTURE_NOTES.md:9` | Reversed; `BUILD_STATE.md:28` — ATNB is the single transferable gov token. |
| Milestone-Project + escrow + arbiter crowdfunding path (boats Path C) | `investment_scenarios.md` | Designed, never deployed; the vault replaces it (and deletes its arbiter). |

---

## D. Open decisions (user only)

1. **Charter anchor placement.** On Substrate.sol (introduces the first governed
   slot on a deliberately ungoverned contract, no timelock wired) vs. in the
   jurisdiction Registry (reuses `updateRPBPrompt`, keeps Substrate pure).
   Report leans Registry; the user decides — this is "constitutional prompt
   wording" territory.

2. **Does the DAO govern the charter at all, and via which governor?** The
   charter is a *network*-level constitution; the trustless-contracts DAO is a
   *jurisdiction*. Confirm whether charter-version governance sits at the network
   level or is delegated per jurisdiction (the fractal-governance question).

3. **Vault raise = securities.** The design doc flags once: revenue-split funding
   is "textbook securities territory." A go/no-go and jurisdiction-of-issuance
   call the user must make before any deploy.

4. **agentReputation → direction weight: never, or only via owner wallet?** The
   wall says quality must not buy direction. Confirm the hard rule: does agent
   standing ever influence DAO direction, and if so does it route strictly
   through the owner wallet, never by minting RepToken to an agent key?

5. **Vault economic parameters** (per CLAUDE.md, user-only): tranche
   sizing/schedule bounds, verifier-trial `N-of-M` greenlight threshold,
   backer-halt quorum, min/max raise caps, whether verifier rewards are
   mint-funded at bootstrap then migrate to bps-of-raise.

6. **One paid service category with real buyers.** The design doc's named
   weakest link — the vault's viability rests on demand, not mechanism. The
   user's call on the beachhead (hardware/compute rental for agents is the
   doc's most-plausible candidate).

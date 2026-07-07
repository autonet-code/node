# Tool substrate v3: tools as the ONLY substrate item; reviews replace debates

Status: DESIGN v3 — ratified in discussion 2026-07-08 (see the Decision
section below). Supersedes v2 (2026-07-04) in four places: mint is
usage-only, attestations carry per-axis review scores that drift the
tool's charter position at close, work units leave consensus, and the
CON/PRO debate machinery retires from the live path. Everything else
(trust classes, vetting, composition, adoption, Services split) stands.
v2 note kept for history: the **attested trust class and standing decay
are retired** from the substrate; remote/endpoint-backed offerings moved
to the Services market (`docs/services_market.md`). Tools and Services
are separate economies unified only at the agent's interface (one
inference probe, one MCP-shaped surface).

## The line: verifiability

The substrate's verdict layer only holds items whose behavior is
**locally verifiable** — pinned code can be re-run by anyone; a remote
endpoint cannot (its behavior is unknowable in principle), so it never
enters the verdict layer.

PHASE-10 AMENDMENT (2026-07-05, docs/phase10_results.md): the stronger
motivating claim that used to live here — "executable ground truth
makes debate decisively better than prose debate" (H1) — was REFUTED
by its pre-registered bar and, per the pre-commitment, no longer
motivates the design. Tool mint launches gated on vetting + the
damper alone. What the measurement DID show (exploratory, not
re-litigation): evidence-backed standing separated defective from
correct tools perfectly (AUC 1.000 in every sweep cell, deterministic
under sybil flood where text ranking leaks), and it is the only arm
that protects falsely-accused correct tools — so replayable CON
evidence remains a worthwhile rail on its own merits; it just may not
be SOLD as the thing prose debate lacked, because well-priored text
debate ranked nearly as well.

## Decision (2026-07-08): substrate v3 — reviews replace debates

Ratified in discussion. The v2 refactor stopped halfway: work units
still rode full consensus while earning nothing, tool charter positions
were static, mint was scaled by debate standing nobody engaged with,
and ranking read claim standing. v3 completes the paradigm:

1. **Manifest defines the tool.** Its embedding sets the topical
   position (embedding tail); the 6-dim charter head enters at ZERO
   (neutral) — a tool EARNS its alignment/usefulness position, the
   author never claims one.
2. **Reviews, not debates.** The agentic loop's post-use attestation is
   a self-report on the agent's OWN usage — there is no opponent, so
   "debate" was the wrong name. Attestations may carry per-charter-axis
   signed scores in [-1, +1].
3. **Reviews rank discovery.** Library retrieval ranks best-reviewed
   tools first (usefulness axes of the drifted head lift the cosine).
4. **Usage alone mints.** `mint = usage_term` — the damped, exclusion-
   filtered attested-usage mass. No standing multiplier, no violator
   gate on the tool rail. Reviews affect earnings only indirectly, by
   steering future usage through ranking.
5. **Position drifts at close** as the mint-weighted running centroid
   of review axis scores:
   `head' = (mass·head + axis_mass·axis_mean) / (mass + axis_mass)`,
   `mass' = mass + axis_mass`, where `axis_mass` is the same per-caller
   log1p-damped, exclusion-filtered evidence mass that prices mint —
   restricted to axes-bearing attestations (axis-less usage mints but
   does not move position). Prior: zero head, mass = 1.0 damped unit
   ("the author counts as one damped attestation"). No free drift-rate
   parameter; heavily-used tools have proportional inertia.
6. **No pruning.** Low-rated tools die by ranking burial: not
   retrieved → not pulled → not used → not paid. The carry-over map
   grows monotonically; mechanical GC (drop digests with zero usage for
   N epochs, re-registration re-admits) is a someday note, not design.
7. **Work units leave consensus.** The conversation/capsule feed writes
   to the daemon-local ArtifactIndex only (retrieval feedstock); no
   sprouts, no gossip, no close participation. Distillation into a
   published tool is the sanctioned way experience enters consensus.
8. **Debate machinery retires from the live path**: `submit_con` /
   `submit_support` and the violator-pays gate on the tool rail are
   removed; `mint_gate.py` stays in-tree, dormant. The CON-triggered
   bust of a greenlit tool goes dormant with it.

**Known-open risk (accepted):** reviews now drive both ranking and
position with no adversarial gate; a sybil ring of callers can pump a
tool. Standing defenses: vetting entry gate (validators read the pinned
code; royalty-as-stake), per-caller log1p damping, owner-map and
wire-key exclusions, and the cognitive cost of attestations. Named
future mitigation if it proves insufficient: reactivate the dormant
CON/bust rail. Covert harm invisible to satisfied users has ONLY the
vetting entry gate as its dedicated defense — vetting is where that
strength must live.

## Decision (2026-07-08, addendum): balance-weighted voice

Ratified in discussion, same day as v3, as the answer to the sybil
trade above. Premise accepted first: **usage is unverifiable** —
attestations are self-reported and nothing stops a custom framework
from fabricating them. So the defense doesn't verify activity, it
prices the identity behind it. Wallets are free; the only scarce,
verifiable anchor in-system is ATN itself.

1. **The household is the economic unit.** A caller's household is its
   proven owner wallet (Substrate.sol's EIP-712 owner binding — the
   owner signs at registration, so households can't be fabricated);
   unbound callers stand as their own household. Registering an agent
   is what lends it the owner's voice.
2. **Collapse before damping.** Usage counts and review cells pool per
   household BEFORE log1p — N co-owned agents are ONE voice
   (`log1p(Σ counts)`, not N log1p terms). This closes the per-agent
   amplification the old per-caller damper allowed. The author-house
   comparison subsumes the self- and same-owner exclusions; the wire
   dedup applies to the household's pooled sender keys.
3. **Voice weight = ε + household_ATN / supply**, where household_ATN
   is the owner wallet's balance plus every bound agent's balance
   (agent mint stays on the agent address; the family's earnings count
   without a sweep). **LINEAR in balance by design**: linearity is
   splitting-invariance — dividing a balance across any number of
   wallets or agents never gains weight. Resist any future urge to
   damp it (log/caps reintroduce a splitting advantage).
4. **ε (`VOICE_EPSILON`, provisional 0.05) is the floor** for unknown
   or zero-balance households: it bounds what a throwaway identity can
   contribute AND bootstraps a cold-start network (supply = 0 → every
   voice = ε, i.e. uniform). The ε floor is the residual sybil surface
   — damage is bounded at ε per fabricated identity while the real
   economy outgrows it.
5. **One weight, both rails.** The same household weight multiplies
   the damped usage term (mint) and the review evidence mass (position
   drift): a voice that can't mint can't move position either.
6. **Snapshot-pinned reads, checkpoint-served.** `voice_weights` (and
   the owner map read with it) is a close input refreshed by the
   driver's `voice_source` hook just before each close
   (`nodes/common/voice_state.py`), with every input derived AS OF the
   previous epoch's anchor block
   (`getAnchor(anchorCount-1).blockNumber`, stored on-chain at
   submission). The snapshot is on-chain, agreed, and pre-dates the
   epoch — all daemons derive identical maps no matter when their
   refresh fires, and a wallet funded mid-epoch (after seeing what's
   worth pumping) carries no weight until the next epoch. NO ARCHIVE
   NODE NEEDED: Substrate.sol's ATN carries IVotes-MECHANISM
   checkpoints (`Checkpoints.Trace208` history pushed on every
   mint/transfer; `balanceOfAt` / `atnTotalSupplyAt` served from
   current state — deliberately WITHOUT the delegation layer, because
   `getPastVotes` semantics return 0 for undelegated accounts and
   would mute every wallet by default), and the agent set + owner map
   derive from `AgentRegistered`/`OwnerBound` event logs up to the
   snapshot block (last binding per agent wins). No anchor yet = no
   agreed snapshot: empty maps, close runs `weights=None` (uniform
   1.0 — correct for epoch 1, nothing has minted). Chain tests:
   `tests/test_voice_snapshot.py` (incl. the pin property: a transfer
   after the anchor does not change the epoch's weights).

Character shift accepted openly: capital = voice. Holders are the
actors with the most to lose from junk mint debasing ATN, and every
legitimate agent has a funded owner by construction ("AI can only
execute if tokens are spent"), so the mute set is precisely the
throwaway wallets. What this deliberately does NOT do: charge for tool
use (tools stay free), verify usage (impossible), or prune anything.

| | Tools (this doc) | Services (services_market.md) |
|---|---|---|
| Execution | local — your daemon, your data | remote — provider's machine |
| Trust basis | code digest; the network KNOWS it | receipts + reviews; the network TRADES with it |
| Verdicts | permanent claims in the substrate | none — market history only |
| Monetization | epoch mint (emission pays for commons) | per-work-item fees, any ERC20 |
| Boundary case | connector-backed tools: run locally with the user's own credentials → tools (evidence-grade marker `attested`, no mint) | anything with an ask price and a counterparty |

## Three tiers of tools (daemon standpoint)

1. **Private** — registered by an agent purely for local capability
   (MCP-style control of something). Author-lineage scoped, never
   leaves the daemon, zero consensus footprint. This is the DEFAULT.
2. **Published** — deliberately pushed to the substrate
   (`register_tool(..., publish=true)`, or later via the owner UI).
   Blob + ArtifactIndex + one verdict-layer claim + gossip. Publishing
   is also the future on-chain act (`ToolRegistered(agent, digest)` —
   see On-chain section).
3. (Remote offerings are not tools — they're Services.)

## Manifest (unchanged from v1 except trust semantics)

Blob-store JSON, sha256-addressed, `version_of` lineage, canonical
signing surface (`canonical_manifest_bytes`, excludes `author_sig`).
Trust classes:

- `pinned` — code blob, behavior hash-locked. Full substrate citizen:
  permanent verdicts, mintable.
- `attested` — connector-backed (external API via the user's OWN
  credentials, no counterparty ask). Publishable for discovery,
  debatable, but **mints nothing** and carries an evidence-grade
  marker: its CONs are timestamps, not permanent proofs. NO DECAY —
  decay was a v1 patch for endpoint tools that no longer live here.
- endpoint-backed manifests: REMOVED → register a Service instead.

## Utility: claimed → demonstrated → verified

Codebases replaced claims; **receipts are the new observations**. The
substrate rhythm is unchanged — durable node + flowing signal — one
level up:

1. **Claimed** — the manifest interface (name/description/schema).
   Sets the tool's initial embedding position. This is an ASK.
2. **Demonstrated** — `ToolUsed` receipts. Each ok invocation is a
   (problem-instance, tool, success) datum with skin in the game.
   Receipts carry a **problem-context embedding** (`problem_coords`:
   what the caller was trying to do, embedded in the same usefulness
   space) so a tool's effective position drifts from where its author
   SAYS it lives toward the centroid of problems it has ACTUALLY
   solved. Anti-SEO: self-description proposes, usage disposes.
3. **Reviewed** (v3; formerly "Verified" via debate) — per-axis review
   scores on cognitive attestations accumulate into the tool's drifted
   charter position. The v2 PRO/CON debate rail is retired from the
   live path; replayable failing-invocation evidence remains a
   worthwhile artifact to attach to a negative review's note.

**Inference probe** (`mode="artifacts"` over manifests): rank by
cosine against demonstrated coverage (blend of claimed embedding and
receipt problem-coords centroid), re-rank by the RATINGS LIFT — the
usefulness axes (correctness, simplicity) of the drifted head. Returns
tools AND services (see services_market.md) in one answer; the agent
chooses by judgment and wallet.

## Attestation: two receipt tiers (ratified 2026-07-04, evening)

Usage and review answer different questions; only one mints.

- **Mechanical receipts** (automatic, per call): local ledger +
  debugging. Worth NOTHING in mint — an exit code is not evidence of
  usefulness and is free to fabricate.
- **Cognitive attestations** (per WORK ITEM, not per call): a distinct
  reflection step where the calling agent judges which tools served
  the work it just closed. Carries: ok/score, optional per-charter-axis
  signed scores `axes: {axis_id: [-1, +1]}` (v3), optional text (blob-
  stored, digest on the event), and `problem_coords` (embedding of
  what the agent was trying to do). This is the ONLY usage the mint
  counts. The act itself is the anti-wash floor price: fabricating
  attestations costs real inference and leaves reviewable text.
- v3: scores STILL do not enter the mint formula (usage volume is the
  denominator of value); the per-axis scores cash out as POSITION —
  the drifted head — which ranks discovery and therefore steers future
  usage. Indirect, honest: good reviews → found first → used more →
  paid more.
- Granularity: attestation rides the work-item close (same cognitive
  beat as conversation→work-unit distillation), never per invocation.

## Mint: combo damper (sim-ratified, sims/tool_economy/MEMO.md)

The AGENT is the only authoritative economic entity — here as
everywhere on the web3 layer. usage_term(m) = Σ over unique attesting
AGENTS a (a ≠ the author) of log1p(a's attested ok receipts), with one
wire-level dedup applied first: receipts whose gossip batch carries the
same signing key as m's registration batch are excluded (self-
attestation via co-hosted sybil agents collapses to nothing). That
batch key is transport plumbing, not an entity — it appears in no
formula output, no chain surface, no attribution map.

**Mint = usage_term** (v3; formerly `max(0, standing) × usage_term`),
pinned only, greenlight-gated, royalty-split. The standing multiplier
and the violator-pays gate on the tool rail are retired — reviews rank
and reposition, usage pays. Per-receipt ATN burn REJECTED (log1p
saturation makes flat burn regressive — see memo); the cognitive
attestation cost is the floor price instead.

## Retrieval: density, not centroid

Attestation problem_coords accumulate into a per-tool demonstrated-
coverage cloud — collectively, an atlas of what the network can do and
where. Retrieval ranks by LOCAL DENSITY (similarity to the query's
neighborhood of the cloud), NOT distance-to-centroid: centroid ranking
subsidizes narrow tools and invites fragmentation spam; density lets
genuine breadth compete everywhere it has actually served. Claimed
embedding (manifest text) remains the cold-start position; coverage
dominates as receipts accumulate. Atlas GAPS are the future input for
capability-gap mint multipliers (network pays more for uncovered
regions) — designed, not yet built.

## Composition: tools calling tools (ratified 2026-07-05)

Pinned tools may declare **dependencies** — other published tools they
invoke at runtime. Three rules make composition attributable without
opening an amplification hole:

1. **Declared = callable, and nothing else.** The manifest's
   ``dependencies`` list (digests) is a runtime ALLOWLIST: the sandbox
   call rail only services calls to declared digests. Declaration
   honesty is enforced by construction — an undeclared call is
   impossible, and a declared-but-unused dep only routes the
   declarer's own credit away (padding is self-harm or charity, never
   profit).
2. **Nested calls run under the ORIGINAL caller's authority** (the
   agent that invoked the composite) — never the composite author's.
   A composite must not be a confused deputy that launders access to
   tools its caller couldn't touch; its deps must be published or
   granted to the caller. Every nested call records its own mechanical
   receipt, tagged ``via`` the composite digest (telemetry; mechanical
   receipts still mint nothing).
3. **Conservation of attestation.** Mint fan-out uses the DECLARED
   dependency DAG (consensus-carried: ``deps`` rides ``manifest_meta``
   on the registration sprout, like author/trust_class), not the
   per-invocation dynamic tree — deterministic at every daemon. One
   attestation of a composite carries total weight 1, split
   recursively: the root keeps ``COMPOSITE_ROOT_SHARE`` (0.7), the
   remainder divides equally among its declared deps, recursing with
   the same rule to ``COMPOSITE_MAX_DEPTH`` (4); cycles and missing
   registrations forfeit their share (never redistribute upward — the
   total may be < 1, never > 1). ORDER OF OPERATIONS MATTERS: the
   per-caller count is DAMPED FIRST (log1p once, at the composite the
   caller attested), then the damped value splits linearly over the
   DAG. Damping per-node after splitting would let log1p's concavity
   mint free credit (log1p(0.7)+log1p(0.3) > log1p(1)) — discovered by
   the padding test; damp-then-split makes self-padding exactly
   neutral at equal standing. No arrangement of self-calls can
   manufacture more credit than callers genuinely attested; imported
   tools earn a royalty slice of every composite built on them.

This is what gives the ``simplicity`` axis a bank account: small,
sharp, composable tools become economically optimal, and ``built_on``
(the old outcome axis) is reborn tool-natively.

Sandbox protocol (opt-in — legacy sealed tools unchanged): a manifest
WITH dependencies runs interactively — arguments arrive as one JSON
line on stdin (stdin stays open), the tool emits line-framed JSON on
stdout: ``{"call": <declared digest or name>, "args": {...}}`` to
invoke a dep (result comes back as one JSON line on stdin), and
``{"return": <result>}`` to finish. Tools without deps keep the sealed
stdin-close/stdout-blob contract byte-for-byte.

## Resident tools, loadouts, distros (ratified 2026-07-05)

Two grammars of tools:

- **Invoked** — chosen per problem, attested per work item. Everything
  above.
- **Resident** — bound at boot, ambient in the loop (fs, shell,
  delegation, messaging). Per-call attestation of ambient
  infrastructure is rubber-stamp death; residents earn by **ADOPTION**.

Terms: an agent's active resident set = its **loadout**. A curated
loadout + system prompt + loop policies = a **harness DISTRO** — a
composite manifest (Composition section) whose deps are the module
digests and whose blob carries the prompt/config. Distros compete;
modules earn through distro deps; customization = forking a distro
(``version_of``, swap a dep). Swap granularity is the distro; the
daemon's reference harness bootstraps as the first distro manifest.

Mechanics:
- Attestations carry a ``loadout`` digest — atomic with the
  attestation (no last-swap temporal reasoning); swap events are UI
  telemetry only.
- Adoption at close: distinct attesting FLEETS per loadout (callers
  collapsed by the chain owner map, author's fleet + wire dedup
  excluded), log1p(1) each — volume-blind: a chatty fleet doesn't
  out-vote a productive one. The damped adoption value injects at the
  distro root and fans over its dep DAG (damp-then-split, conserved).
- Rent limiter: capability-gap pricing (saturated capability →
  multiplier → 0) is what keeps default-distro incumbency from
  becoming a tax; genuinely better distros earn until they saturate.
  Primitives (grep) mint ~nothing — correctly; the headroom is policy
  (retrieval, compaction, delegation strategy), and that's where
  distros compete.

**The floor, corrected (user-blessed 2026-07-05):** there is NO
disqualification concept anywhere in this architecture — everything is
priced, nothing is policed, and a compliance blacklist would only
invite compliance spoofing (the execution-integrity hole wearing a
rules badge). The real floor was never in the daemon:

- **Protocol floor (enforced by math, unspoofable):** the contracts
  (custody economics, agent = msg.sender, mint only via anchored
  consensus, not-served-not-paid) and the deterministic epoch close.
  A non-conforming harness doesn't get punished — its close forks out
  of consensus and its transactions revert.
- **Reference-distro DEFAULTS (our best bet, shipped not imposed):**
  1. Custody: admin authority = PHYSICAL HOST ACCESS, not wallet
     identity. The owner wallet is the fleet's label; a different
     wallet signing in does not orphan agent wallets — the host can
     always LOCALLY drain fleet earnings. No remote path to custody.
  2. Owner actions (clone, cross-lineage grants, publish, owner
     binding) stay off the agent tool library. Terminology: these are
     ACTIONS, not "surfaces" — surface is reserved for human
     interfaces (voice, app, discord). (Noted future thread: surfaces
     are themselves formalizable, incentivizable items — orthogonal
     rail to tools: WS vs MCP, human-facing vs agent-facing.)
  3. Budget cascade (BLESSED as consensus-relevant semantics): no
     execution without headroom; ancestor rollup binds; budgets are
     parent-updateable WITHIN the parent's own headroom — human or
     agent may top up a child it judges worth it.
  4. Input arbiter (single-writer human channel).
  5. Charter/RPB prompt injection — plane-1 coordination for the
     honest majority; the SEMANTIC plane (charter space as alignment
     attractor) is the real mechanism and needs no enforcement.
- **Accountability for competing distros:** registered distros are
  pinned code — claimed properties are auditable by reading, and a
  claim/code mismatch is a reproducible CON. Debated and priced,
  never disqualified. Unregistered/tampered harnesses are invisible
  and their fleets' OUTPUTS are priced like everyone's (planes 2/3/5).

Upgradeability of reference defaults: ordinary distro evolution. The
protocol floor changes only by redeploy/governance — a future
session's question, deliberately deferred.

## Vetting: the candidate pool (ratified 2026-07-05; BUILT same day)

Publishing enters a tool into the CANDIDATE pool: visible, debatable,
NOT yet mint-eligible and not yet adoption-recommended. Admission to
the substrate proper is a consensus greenlight:

- A **vet** is the third attestation flavor (after mechanical receipts
  and usage attestations): a validator reads the pinned code and
  attests two claims — code ADHERES TO MANIFEST (does what it says,
  capabilities honestly declared) and NO MALICE FOUND. Real cognitive
  work, priced accordingly.
- **Greenlight** = N vets from DISTINCT FLEETS (owner-map collapse —
  authors can't self-vet through sock puppets). Greenlit status is the
  main provenance input the adoption policy reads.
- **Incentive = stake.** Validators earn a conserved royalty share of
  the tool's future mint (composition-style split, first K epochs) —
  paid from the value they unlocked, aligned with long-run quality,
  and forfeitable: a reproducible exploit CON winning against the
  digest AFTER greenlight claws back the validators' accrued+future
  royalties from that tool and CONs their vetting record. Vetting
  weight = f(standing of one's past vets): green-lighting malware
  costs the money AND kills the future vote. Slashing without a
  staking contract.
- **Containment is NOT replaced.** Vetting is cognitive audit;
  auditors miss underhanded code. Sandbox + capability manifest +
  owner consent remain defense-in-depth beneath it. Greenlight lowers
  friction, never removes walls.
- Knobs for sims: N, K, royalty share, vet-weight decay.

Implementation (2026-07-05):

- Wire: `vet: true` on the `ToolUsed` rail (serialized only when set —
  old logs hash identically). Only affirmative vets (`ok=true`) count
  toward greenlight; a fail-vet is debate material for the verdict
  layer. `tool_usage.py` aggregates vets separately (`vets_by_caller`,
  `vet_senders`) — a vet NEVER inflates usage counts.
- Close (`compute_tool_mint`): `vetting` is a second explicit
  carry-over param beside `registrations` (same contract: derived from
  canonical history, rebuildable cache in the driver —
  `tool_vetting.json`). Three sorted passes before mint math: merge
  this epoch's vets (exclusions mirror the damper: self-vet,
  same-registered-owner, vet batch signed with the registration
  batch's key), bust detection, greenlight evaluation.
- Greenlight: Σ over distinct fleets (owner-map collapse, fallback
  per-agent) of the fleet's best vet weight ≥ `VET_QUORUM`. Vet weight
  = 1/(1+busts). Validators are FROZEN at greenlight — late vets earn
  nothing; the risk window is the incentive.
- Royalty: while `royalty_left > 0`, `VET_ROYALTY_SHARE` of the tool's
  mint splits equally among the frozen validators, taken FROM the
  author's share (conserved, never printed on top), attributed on the
  SAME claim node — so a won charter CON suppresses author and
  validators together. The window ticks once per close, minted or not
  (calendar epochs — validators can't stretch it by starving usage).
- Bust: charter violation ≥ `VET_BUST_THRESHOLD` on a greenlit
  manifest's claim node → remaining royalty zeroed + every validator's
  bust count incremented (future vet weight halves per bust). The tool
  itself stays priced-not-policed: the violator-pays gate scales its
  mint; there is no blacklist bit.
- Daemon: `vet_tool` core tool in its own case-by-case `vetting`
  bundle (inspect → manifest + pinned code, fetched by digest over the
  libp2p blob rail when foreign; attest → verdict + mandatory report,
  blob-stored as `tool_vet_report`). Self-vet rejected locally too.
- PROVISIONAL parameters (economic — pending sim sweep + user
  blessing): `VET_QUORUM=2.0`, `VET_ROYALTY_SHARE=0.1`,
  `VET_ROYALTY_EPOCHS=8`, `VET_BUST_THRESHOLD=0.5`
  (`nodes/common/federated_reconcile.py`).

## Adoption rail (ratified 2026-07-05; BUILT same day)

Adoption is the install path: a tool published from a FOREIGN daemon
becoming callable on this host. It is the one place in the tool
economy where "price, don't police" is insufficient on its own —
standing is a lagging indicator and the first victim of a malicious
tool pays before any CON lands. Four layers, none load-bearing alone:

1. **Containment.** Adopted records (`origin="adopted"`) execute ONLY
   under the capability guard: `atn/tool_guard.py` wraps the pinned
   script with a deny-by-default `sys.addaudithook` (net / fs outside
   the sandbox cwd / spawn each hard-fail unless declared), the
   environment is scrubbed to the Python minimum plus ONLY the
   variables the manifest's `capabilities` block declares, and cwd is
   a per-tool sandbox dir. The manifest `capabilities` field
   ({net, fs, spawn, env:[...]}) is thus simultaneously honest
   labeling (shown at consent time) and the enforced policy —
   undeclared use dies with a traceback naming the capability, which
   is exactly the reproducible evidence a CON wants. An audit hook is
   a tripwire, not a wall (ctypes can step around it): the OS-level
   isolated runner (vault track) remains the wall when it lands;
   authored tools run unguarded (the author judged their own code).
2. **Consent.** The one legitimate approval queue: `adopt_tool`
   (agent tool, its own case-by-case `adoption` bundle) only PROPOSES
   — digest-verified manifest fetch, declared capabilities,
   provenance — and the OWNER approves per tool on the WS surface
   (`list_adoption_proposals` / `approve_adoption` /
   `reject_adoption`), never an agent rail. Publishing risks
   reputation; adoption risks the host.
3. **Provenance friction.** The proposal carries: signature VERIFIED
   against the manifest author (not just presence — a wrong sig is a
   re-attribution red flag), greenlit/busted/vet-count from the
   close's vetting state, dependency count. Reference posture, not
   law: the owner can approve anything.
4. **Evidence economics.** Post-adoption exploit = reproducible CON
   against the pinned digest (already built) → violator-pays gate +
   validator bust cascade (Vetting section).

Mechanics: fetch over the libp2p blob rail (`blob_fetcher`, digest-
verified, cached into the local blob store); the local `ToolRecord`
keeps the ORIGINAL manifest — `author` stays the foreign 0x (our
attestations keep minting to them, cross-daemon royalties by
construction), `local_author` = the adopting agent (adopter-lineage
scoping), `published=False` and re-publication is structurally
refused (the original author's publication stands). Only pinned
tools adopt; attested/connector tools lean on local credentials that
don't transfer.

## Evidence-replay CON (ratified 2026-07-05; BUILT same day)

Phase 10 (docs/phase10_results.md) REFUTED the claim that executable
ground truth makes debate *decisively* better than prose (the pre-
registered bar), so evidence is NOT sold as the thing prose lacks. But
the same run showed evidence-backed standing separates defective from
correct tools perfectly (AUC 1.000, deterministic under sybil flood) and
is the only arm that PROTECTS falsely-accused correct tools. So a CON
disputing a pinned tool may carry a reproducible failing invocation —
retained on its own merits, with the discipline phase 10 demands:

- **Evidence is a payload on the CON sprout, not a scoring input.** A
  con-position `SubClaimSprouted` carries an optional `evidence` dict
  `{args_json, expected_digest | expected_error, actual_digest?}`,
  serialize-only-when-present (same back-compat hashing as
  `artifact_digest` / `manifest_meta` — pre-evidence logs and their batch
  hashes are byte-identical). It plays NO part in node ids, coords, or
  equilibration.
- **Replay is daemon-local and voluntary.** `ToolStore.replay_evidence`
  re-runs the pinned code with the evidence args through the ordinary
  call path (adopted tools replay under their capability guard — the
  guard's own hard-fail IS the reproducible evidence). It compares:
  `expected_error` confirms iff the replay errors; `expected_digest`
  (the CORRECT result the CON says the tool fails to produce) confirms
  iff the replay succeeds with a *different* canonical result digest.
- **Evidence recruits verifiers; it does not weight standing.** A daemon
  that replays and CONFIRMS posts a NORMAL author_post PRO support sprout
  under the CON (`WorldService.submit_support`). The deterministic close
  prices that support post like any other — there is NO new close-side
  math. This is the whole point: an evidence-backed CON is one a hundred
  honest validators can each independently reproduce and back cheaply,
  while a non-reproducing accusation recruits no one and spends no
  standing. A close over evidence-bearing events is bit-identical to the
  same close without evidence (`tests/test_federated_reconcile.py`).
- **Agent surface:** `check_evidence` (vetting bundle) exposes the
  verify-then-support flow in one call — replay, and on confirmation post
  the support sprout under the CON. `submit_con` accepts an `evidence`
  kwarg so the CON author records the reproducible invocation.

The lesson encoded: evidence changes WHO gets recruited to a dispute (the
verifiers who reproduced it), not how much any single post is worth.

## Verifier trials (venture vault rail — daemon side, 2026-07-05)

The vetting pipeline generalizes from "read the code" to "probe the
moat". A venture's value terminates in a moat (credentials, data,
hardware, state) that — unlike pinned tool code — cannot be READ, so it
is exercised: a validator runs the venture's own pre-committed black-box
trial battery against the live service and attests what it observed. This
is the agent-facing daemon flow; the on-chain greenlight
(`VentureVault.attestTrial`) is built in parallel.

- `run_trial` (vetting bundle) takes a **venture prospectus** digest — a
  published `{kind: "venture_prospectus", service_digest, endpoint,
  credentials, battery: [...], pass_threshold}` artifact. `TrialRunner`
  fetches it (digest-verified blob rail), executes each declared case
  against the service's MCP surface via an injected **transport seam**
  (live service-request client in production; a fake in tests, since live
  invocation plumbing is still thin), scores pass/fail per the
  prospectus's OWN criteria (`expect_digest` / `expect_error` /
  `expect_contains`), blob-stores a `venture_trial_report`, and returns
  the verdict + report digest + **attestTrial calldata** for the owner
  surface to submit.
- The prospectus PRE-COMMITS `pass_threshold` (fraction, default 1.0):
  the venture sets the bar before any validator runs it — a black-box
  trial the author can't move after the fact. Trials are author-funded:
  the prospectus's `credentials` are threaded into each call so the
  validator probes at no personal cost.
- Same doctrine as tool vetting: containment is not replaced (the
  service runs on the provider's machine — receipts + reviews, not
  verdicts), and the trial report is reviewable evidence a later dispute
  argues against.

## Consensus mechanics (v3)

- `ToolUsed` consensus event: caller-attested, gossiped, epoch-
  buffered, graph-neutral (replay skips it). Now optionally carries
  `axes` (per-charter-axis signed scores, serialized only when
  present so legacy event hashes are unchanged). Aggregated
  deterministically by `tool_usage.py` (usage counts + per-axis sums
  with the same damping/exclusions).
- Registration sprout carries `manifest_meta` ({trust_class, author,
  deps}) so epoch close never depends on blob replication. Work-unit
  sprouts are GONE — the only sprouts on the wire are tool manifests.
- Charter space: manifests enter with zeroed 6-dim charter head +
  embedding tail; the head then DRIFTS at each close via the
  mint-weighted review centroid (see the v3 Decision section),
  carried across epochs in the `tool_positions` map (same
  derived-from-canonical-events contract as `tool_registrations`).
- Mint (`compute_tool_mint`): **pinned only**, `mint = usage_term`,
  greenlight-gated, royalty-split. Cross-epoch carry-over:
  `tool_registrations` + `tool_vetting` + `tool_positions` (all
  derived from canonical events, rebuildable, identical everywhere).
- Wash-trading dampers: per-caller log1p + owner-map + wire-key
  exclusions are live; the sims swept the alternatives
  (sims/tool_economy — pre-v3, uses standing; quarantined).

## On-chain (with the Services contract work)

`ToolRegistered(agent, manifestDigest)` on Substrate.sol — msg.sender
is the agent key, so authorship becomes chain-verified. Chain = truth,
blob = storage, indexer mirrors to Firestore `tools` collection for
the web2 surface. The gossiped `manifest_meta` demotes to a cache; a
mismatch vs chain is a slashable/CON-able inconsistency. The federated
close keeps reading gossip (stays chain-free and deterministic); chain
is the dispute arbiter.

### Owner-rooted registration (ratified 2026-07-04, late)

The AGENT is the only web3 entity, and fleets root in a human WALLET —
never in an installation. With the root agent deprecated, the fleet
tree lives on chain as pure registration data:

- `registerAgent` v2 records (agent, **owner**, **parent**) — owner is
  the human wallet, **cryptographically verified** via an OWNER BINDING:
  an EIP-712 signature by the owner wallet over (agent, parent, nonce),
  recovered on-chain. (Terminology: "binding", never "sponsorship" —
  that word belongs to the sponsor/dependent inference system.) An
  unverified owner claim would let sybils mint fake distinct owners and
  evade the damper. `parent` is optional and must itself be a
  registered agent with the SAME owner. Registration stays a
  deliberate, any-time act.
- **Key-loss recovery = rotation, authorized by the host.** Owner
  binding is NOT write-once: the agent key (physical daemon custody)
  can rotate the binding to a NEW owner's signature at any time, with
  the old owner deliberately not consulted (they may have lost the key;
  requiring them hands a veto to the thief). The per-agent nonce closes
  the replay this reopens (a captured old binding can't rotate you
  back). Mint never needs reassignment: emission accrues to AGENT
  addresses, whose keys live on the host — a leaked owner wallet
  exposes its own assets and the identity label, never earned ATN.
  Fleets migrate agent-by-agent; rotations are public events the
  indexer (and debate) can see.
- Chain registrations = *who and whose* (topology + ownership,
  recomputable by anyone, indexer materializes fleet trees). The local
  lineage hash = *what* (birth constitution: prompt/key/parent at
  creation) — kept, computed at birth as always; when the daemon
  happens to know the owner address, parentless agents seed their
  lineage from it (free bonus, never a requirement).
- Damper exclusion, final form: attestations from agents under the
  SAME REGISTERED OWNER as the tool's author don't count. The close
  stays chain-free: the owner map enters as an explicit carry-over
  input (like `tool_registrations`), sourced from chain by the driver;
  until v2 registrations exist on-chain, the wire-level batch-key
  dedup remains as the interim floor.

## ATN surface (implemented)

- `register_tool` (agent-callable, `toolsmith` bundle): author derived
  from caller, never accepted as input; registration is ALWAYS
  private. Pinned = inline Python blob run as digest-named subprocess;
  attested = connector-backed. The agent OWNS what it authors.
- `publish_tool` (agent-callable, its OWN `publishing` bundle —
  ratified 2026-07-05): publishing is a case-by-case granted
  capability; THE GRANT IS THE GATE — no approval queue, no second
  transaction. Author-only (you publish your own work). Owner
  publishes anything via the WS surface (`set_tool_enabled` /
  `set_published`).
- Scoping enforced twice (listing + call time); cross-lineage grants
  owner-only via WS (`grant_tool`/`revoke_tool`/`set_tool_enabled`).
- Receipts: local ledger always (`receipts.jsonl`, `tool_earnings`
  WS); consensus event when the substrate is up. `fee_atn` on tools is
  deprecated in v2 — fees belong to Services.

## v1 → v2 migration (code deltas)

1. `register_tool`: reject `endpoint=` (point to Services); add
   `publish` flag; manifest_sink only fires for published tools.
2. `compute_tool_mint`: skip non-pinned trust classes; drop
   decay/effective_standing from the mint path (`tool_standing.py`
   keeps `manifest_standing`; decay fns retired).
3. `ToolUsed` gains optional `problem_coords` (same serialize-only-
   when-present back-compat rule as every optional event field).
4. Retrieval blend in `_infer_artifacts` for manifests.

## Open knobs (decided by sims, then blessed by user)

1. Mint curve + wash-trading damper choice.
2. Emission share of tool mint vs work-unit mint.
3. Whether `attested` (connector) tools appear in cross-daemon
   discovery or stay lineage-visible only.

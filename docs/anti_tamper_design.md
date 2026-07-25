# Consensus-Node Anti-Tamper: Design

Status: designed 2026-06-17, pre-implementation. The "other half" of the
auto-update feature (docs/auto_update_design.md). Pairs the silent updater
with an integrity story so an open-source daemon can still be a *trustworthy
consensus node*.

## The problem in one line

The agent framework is open-source except the obfuscated fingerprint module.
By running a node you accept a constitution that is provisioned into every
agent's context. We must make it so that **a node participating in consensus
provably runs canonical code with the constitution uncircumvented**, and a
tampered node is *excluded by the network*, not merely asked nicely to stop.

## What the recon established (ground truth, 2026-06-17)

- **Fingerprint mechanism exists and is correct but INERT.** `atn/_cache.py`
  `core_fingerprint()` / `validate()` compute a SHA-256 over `_CORE_FILES`
  (7 files), byte-identical to `scripts/build_release.py:compute_core_hash`.
  `_cache.py` is the only obfuscated file (python-minifier), and the on-chain
  hash is computed AFTER obfuscation. BUT: `validate()`'s return value is
  *discarded* at both call sites (`autonet_service.py:681` ignores it;
  `scheduler.py:246` stores `_freshness_ok` and never reads it). Nothing is
  enforced. The enforcement branch was never written.
- **Constitution provisioning has ONE chokepoint:**
  `execution_engine.py:428-439` prepends the constitutional preamble, but only
  when `identity.registered_on_chain == True`. Six files guarantee that path:
  execution_engine (chokepoint), autonet_service (loader/cache),
  delegate_prompts (preamble template), runtime/__init__ (bridge wiring),
  models.py (`registered_on_chain` field), constitution/v1_udhr.txt (text).
  Constitution is loaded on-chain-first with local-file fallback.
- **The address bootstrap hole is the linchpin.** `atn/jurisdiction.py` holds
  the ONLY hardcoded root: `GOVERNOR_ADDRESS` + RPC + chain id. Everything else
  (substrate, registry, rpb, token…) is discovered from the Governor OR
  overridden from registry.json / config.yaml. **`jurisdiction.py` is NOT in
  `_CORE_FILES`**, and the registry address used by `validate()` itself comes
  from mutable config. So the integrity check is *circular*: an attacker
  repoints `registry_address` → `validate()` queries the attacker's contract →
  forged "valid" hash → full bypass.
- **Lineage is identity, not integrity.** `compute_lineage_hash =
  sha256(parent_hash : charter[:256] : public_key)` where `charter` = the
  agent's per-agent `system_prompt` (NOT the constitution). `verify_lineage()`
  exists but is never called; the chain stores `lineageHash` without validating
  it. No code binding today.

## Design: four pieces, in dependency order

### Piece 1: Close the bootstrap hole (prerequisite for everything)

Without this, every other check is an oracle for attacker data. The fix is
structural, not cryptographic-heavy:

1. **Add `atn/jurisdiction.py` to `_CORE_FILES`.** Now the hardcoded Governor
   address (and RPC, chain id) is part of the fingerprint: editing it changes
   the running-code hash.
2. **Bake the canonical registry address into the obfuscated module.** The
   address `validate()` queries must NOT come from mutable config for a
   consensus node. Put the canonical jurisdiction's `(governor, registry,
   chain_id, rpc)` tuple inside the protected set, ideally derived inside
   `_cache.py` / `jurisdiction.py` so it's covered by the very hash it anchors.
   Resolution of the bootstrap paradox: the address that says "where do I read
   my expected hash" lives in the code that the expected hash covers. An
   attacker who repoints it changes the code → hash mismatch → tamper branch.
3. **Consensus mode ignores config address overrides.** registry.json /
   config.yaml address overrides remain for *local dev / alternate
   jurisdictions* (non-consensus), but a node that wants to participate in the
   canonical jurisdiction uses the baked-in addresses. Overriding them = you're
   not on the canonical jurisdiction = your consensus posts aren't accepted by
   canonical peers (Piece 4).

This is the piece that makes "you can't just point at another contract to
circumvent everything" true.

### Piece 2: Enforce the fingerprint at startup, with the update/tamper split

Wire the inert `validate()` into a decision. The mismatch is disambiguated by
the auto-updater, exactly as the user framed it:

```
at startup (and periodically), compute local core_fingerprint()
fetch canonical hash for THIS version from the baked-in registry
  ├─ match                         → healthy, participate
  ├─ no on-chain hash for version  → fail-open + WARN (unpublished; today's norm)
  ├─ chain unreachable             → fail-open + WARN (can't distinguish; log)
  └─ mismatch AND hash exists:
       ├─ on-chain hash for a NEWER version exists  → benign: a release is out.
       │     route to the auto-update path (stage → apply on next boot).
       │     NOT tampering.
       └─ on-chain hash for MY version ≠ my code    → TAMPERING.
             refuse to participate in consensus + self-destruct sequence.
```

The update path is what tells "new version" apart from "tampered." That's why
"refuse to start / self-destruct" is safe: the benign false-positive (new
release) is caught and routed to update before the destroy branch.

**Self-destruct semantics (honest framing):** self-destruct is the
*honest-node-cooperates* behavior, not the enforcement. An attacker who has
tampered the code can patch out the destroy branch, so self-destruct cannot be
what protects the network. What it does: a node whose operator did NOT
intend to tamper (got a corrupted install, a bad disk, a supply-chain hiccup)
takes itself out cleanly instead of polluting consensus. Scope of "destruct":
stop the consensus loop, refuse to post to substrate / submit anchors / join
epoch close, wipe the in-memory agent keys, and mark the daemon halted. It does
NOT delete user data. The teeth are in Piece 4.

### Piece 3: Minimal on-chain hash registry (the deployment governance is behind)

Governance is implemented but not deployed; we deploy the *minimum* that makes
the canonical hash real and updatable, as an explicit placeholder for the full
jurisdiction contracts.

- A small `CodeHashRegistry` contract: `getRegistryValue(string key) → string`
  (matches what `_cache.py:_fetch_ref` already calls: `node.code.hash.<ver>`),
  plus `setCodeHash(version, hash)` gated by **ERC20-weighted majority**.
- Bootstrap: the hardcoded deployer wallet deploys it, mints total supply to
  itself, and pushes the first hash. This is the placeholder for on-chain
  jurisdiction contracts: it lets one wallet act as "governance" until real
  governance deploys, at which point token supply / ownership migrates.
- The CI release flow (`release.yml` lines 108-115 currently only *document*
  the hash) gains a step that calls `setCodeHash(version, package_hash)` with
  the deployer key. This is what finally makes `validate()` non-advisory.
- **Token economics here are the user's call** (supply, whether the deployer
  wallet keeps total supply or distributes, the majority threshold) are flagged
  below, not decided in this doc.

### Piece 4: Lineage-as-integrity, the actual teeth (peer-verifiable)

Self-attestation is worthless against tampering (the tampered node lies about
its own check). The teeth must be peer-verifiable on-chain. This is where the
user's "fold lineage into integrity" instinct is correct, and it's a real
design, not a category error, IF done at the right layer:

- **Bind the code fingerprint into registration.** When a node registers an
  agent, include the canonical-code attestation: the agent's on-chain record
  carries (or is checked against) the `node.code.hash.<version>` it claims to
  run. Mechanism options (decision needed, see open questions):
  - (a) extend the lineage charter input to include the core-code/constitution
    hash, so `lineageHash` itself attests code: clean, but it means a code update
    forces re-registration (new lineage). Auditable but heavy.
  - (b) keep lineage as identity; add a SEPARATE `attestedCodeHash` field on the
    agent's on-chain record, updated per version, verifiable by peers. Lighter;
    lineage stays stable across updates.
- **Peers verify, not the node itself.** At epoch close / anchor submission,
  honest daemons can check that a contributing peer's attested code hash equals
  the canonical hash for its version. A tampered node either (i) attests the
  canonical hash it isn't running (detectable if combined with any
  challenge-response), or (ii) attests a non-canonical hash, rejected outright.
  Full cryptographic remote attestation (proving you run specific code) is hard
  without a TEE; the pragmatic V1 is: mismatched/absent attestation → consensus
  contributions ignored by honest peers, and the on-chain record is permanent
  evidence for governance slashing.
- **Wire the dead `verify_lineage()`** at load + registration as the local
  half (catches accidental staleness), even though the network half is the
  real protection.

## Why this isn't over-engineered (rejecting recon's heavier proposals)

The recon agents proposed config-file signing, HSMs, runtime address
allowlists, a separate `UpgradeGuard` contract, etc. Most are unnecessary:

- **Config signing / HSM:** the threat is filesystem tampering by someone who
  owns the machine. Signing config doesn't help: they can re-sign or patch the
  verifier. The answer is "the canonical addresses aren't in mutable config for
  consensus nodes" (Piece 1), not "sign the mutable config."
- **Runtime address allowlist:** subsumed by baking the address into the
  fingerprinted set.
- **Separate UpgradeGuard contract:** the minimal `CodeHashRegistry` (Piece 3)
  already is the immutable-ish address-of-record; a second contract is
  premature before real governance.

The honest security posture: against a machine-owner attacker, no software
self-check is unbreakable. We make tampering (a) detectable by honest peers
(Piece 4), (b) self-correcting for non-malicious corruption (Piece 2), and
(c) not circumventable by the cheap config-repoint path (Piece 1). That's the
right bar for a consensus node without a TEE.

## Implementation order (when the user says go)

1. Piece 1 (address pinning + `jurisdiction.py` into `_CORE_FILES`): pure
   refactor, no chain dependency. Unblocks everything.
2. Piece 2 (enforce validate(), wire the update/tamper split): depends on the
   updater (done) and on Piece 1.
3. Piece 3 (deploy `CodeHashRegistry`, CI publishes hash): the deployment the
   user noted is "behind." Bootstrap wallet.
4. Piece 4 (lineage/attestation teeth): the hardest, last; safe to ship 1-3
   first (a node self-excludes on tamper) and add peer-verification after.

## Open questions for the user (NOT decided here)

- **Token economics for `CodeHashRegistry`** (supply, deployer-keeps-all vs.
  distribute, ERC20-weighted majority threshold). Flagged per CLAUDE.md: these
  are the user's call.
- **Lineage binding mechanism**: 4(a) lineage-includes-code-hash (re-register on
  update) vs. 4(b) separate `attestedCodeHash` field (lineage stable). Trade
  auditability vs. update friction.
- **Self-destruct aggressiveness**: halt-consensus-only vs. also wipe keys vs.
  refuse-to-boot. Recommend halt-consensus + wipe in-memory keys; keep user
  data and allow a clean reinstall.
- **Unregistered-agent loophole**: today unregistered agents run with NO
  constitution. For a consensus node, should registration (and thus the
  constitution) be mandatory? Affects whether the chokepoint guard stays gated
  on `registered_on_chain`.

# Autonet: one epoch end to end, and the economy as the code implements it

Every claim below comes from the code. It was first written from commit 782bdc8 and now describes HEAD (157796c), including the later fixes d5f77db (loadout receipts and sanctions, empty REP), 8a7241d (stuck election re-post) and 157796c (search hides finally denied tools). Paths are relative to the repo root. Citations use `file:line`. Text rewritten for those commits cites HEAD lines. Older citations were taken at 782bdc8; since then `federated_reconcile.py` lines past 803 moved down 2 (past 812, down 4), and `federated_close_driver.py` lines past 955 moved down 4 (past 1080, down 10). Other cited files are unchanged except `anchor_election.py` (past 125) and `atn/tool_security.py` (past 211).

---

## 1. Overview of one epoch

1. Agents produce events: claim sprouts, observations, and `tool_used` receipts. These enter the local `WorldService`, which publishes them as signed gossip batches (`event_gossip.py`).
2. Every daemon buffers its own batches and its peers' batches (`event_gossip.py:501`, `:578`).
3. A local wall-clock scheduler ticks every `epoch_interval_seconds` (default 60 s, `nodes/service.py:693`, `epoch_scheduler.py:38`). Each tick runs a local, non-authoritative close and calls `FederatedCloseDriver.run` (`nodes/service.py:1086-1104`).
4. The driver reads chain inputs pinned to the previous epoch's cutoff. These are the owner map, REP shares, the fee pool, ToolBond ratings, sanctions, the charter, containment and mint-gate parameters, and the epoch rules (`voice_state.read_voice_state`). The driver then decides whether the epoch's cutoff plus grace has passed (`federated_close_driver.py:495-550`).
5. Selected batches are put in canonical order (`canonical_ordering.py`). The canonical sequence is replayed on a fresh charter world, and `compute_tool_mint` plus `federated_reconcile_epoch` produce the authoritative payload (`federated_reconcile.py:1267-1580`).
6. The payload is hashed, the household mint map is merkleized, and the anchor goes on chain through a DAO election (`chain_submission_driver.py`, `anchor_election.py`, `AnchorElection.sol`). The fallbacks are a governed proposal or, on dev networks, a direct anchor.
7. After the anchor lands, every daemon calls `claimFor` for the households it serves. It then claims REP through `AutonetRepClaim` (`chain_submission_driver.py:603-660`).
8. Separately, staked reviews for ToolBond's epoch are committed when the federated close fires. They are revealed after the anchor and finalized after the reveal window (`atn/review_driver.py`). They are read back into later closes as ratings and reviewer shares.

In production the pipelined mode is always on: `chain_source` is wired whenever `substrate_address` and `rpc_url` are set (`nodes/service.py:965-998`, `federated_close_driver.py:999-1000`).

---

## 2. Events and how they reach the close

### 2.1 Event kinds

Defined in `nodes/common/world_model_substrate/events.py`:

| kind | Fields that matter to the economy | Notes |
|---|---|---|
| `sub_claim_sprouted` | `author_agent`, `node_id`, `parent_id`, `position`, `artifact_digest`, `manifest_meta` | A registration sprout carries `manifest_meta`, listed in the next row (`events.py:113-120`). |
| (registration sprout) | `manifest_meta = {trust_class, author, deps (comma-joined), version_of?}` | Built in `world_service.py:1032-1046`. It is submitted with `author_post=True` (`:1051`). |
| `observation_added` | `obs_id`, `coords` | Graph only. |
| `tool_used` | `author_agent` (the caller), `manifest_digest`, `seq`, `ok`, `attested`, `own_share?`, `loadout?`, `vet?`, `agent_sig?` | Mechanical receipts (`attested=False`) count for nothing (`events.py:194-200`). |

Receipts are emitted by `ToolStore._record_receipt` (mechanical, `atn/tool_store.py:1566-1635`) and by the attestation path (`atn/tool_store.py:1879-1912`). An attested event carries `"attested": True`, optionally `own_share` and `loadout`, and an `agent_sig`. The attested event deliberately carries no axes: "No axes on the wire: reviews settle on ToolBond" (`tool_store.py:1896`). `seq` is the tool store's own counter, re-seeded at load from the receipt and attestation counts (`tool_store.py:274`).

`apply_events` ignores `tool_used` when replaying the graph (`aggregate.py:155-158`).

### 2.2 Signed receipts (`nodes/common/tool_receipts.py`)

- The signed text is `"autonet:tool_used:v1|<domain>|<author_agent>|<manifest_digest>|<seq>|<1/0 ok>|<receipt_digest>|<repr(own_share) or ''>"` (`tool_receipts.py:18-33`). The signature is EIP-191 (`:36-41`).
- The domain is the jurisdiction key `<chain_id>:<dao>` (`atn/autonet_service.py:1271`). It equals the gossip `rpb_address` (`nodes/service.py:648`).
- The agent's own key signs. The owner has no agent key, so its receipts cannot sign and do not mint on a chain-backed close (`autonet_service.py:1265-1273`).
- `verified_receipts` drops any attested `tool_used` that is unsigned, does not recover to `author_agent`, or repeats an earlier `(author_agent, seq)` (`tool_receipts.py:60-77`). With `domain=None` nothing is dropped.
- The close passes a domain only when a review snapshot exists: `receipt_domain = gossip.rpb_address if review_snapshot is not None else None` (`federated_close_driver.py:1102-1103`).

### 2.3 Gossip (`nodes/common/event_gossip.py`)

- **Envelope.** `EventBatch{schema=1, rpb_address, sender_pubkey, batch_seq, prev_batch_hash, timestamp, events}`. It is signed with ed25519 over canonical sorted JSON (`:135-177`). If `cryptography` is missing, signing falls back to keyed blake2b, and verification returns True (`:109-127`).
- **Outbox.** With no peer on the topic, events are held and signed only when sent, so the timestamp is fresh. They join the daemon's own epoch set only then (`:441-478`). Own batches enter `_epoch_batches` at publish (`:498-502`).
- **Receive.** The receiver checks rpb, then the signature, then the skew rule: a batch is dropped when `|timestamp - now| > max_skew_seconds` (`:548-558`). It then dedupes by content hash (FIFO of 4096), buffers the batch, and ingests it into the local world unless it is our own (`:514-598`).
- `max_skew_seconds` is the governed `epoch_skew` (default 60). The close driver sets it after each voice refresh (`federated_close_driver.py:928-931`).
- `take_epoch_batches(selected, drop_at_or_before=cutoff)` removes the epoch's members plus anything signed at or before the cutoff (`:626-641`). `restore_epoch_batches` puts batches back after a failed close or a rewind (`:643-652`).
- The gossip keypair is per daemon, persisted at `<data_dir>/world/gossip/keypair.json` (`nodes/service.py:1186-1219`).

### 2.4 Voice state (`nodes/common/voice_state.py`)

`read_voice_state` (`:1165`) is the single chain read for a close. It is all-or-nothing: any failure blocks the close (`federated_close_driver.py:867-931`). It returns the following:

- **Pinning.** In pipelined mode (`pin` with `number > 1`), REP is read at `timestamp = cutoffs[-1]` (the previous epoch's cutoff). Logs are read up to `block_at_or_before(timestamp)`, which raises until the chain head is past that time (`:1088-1109`). The pool is `burnedBetween(cutoffs[-2], cutoffs[-1])`. For epoch 2 the start is `deployedAt`; past epoch 2 a pin without two cutoffs raises (`:1275-1292`, `:1286-1289`). Without a pin, the read is pinned to `getAnchor(anchorCount-1)` and the pool is that anchor's `windowBurned` (`:1294-1302`).
- **Genesis branch** (`anchorCount == 0`, not pinned). The result has empty maps, `emission_pool 0.0`, a review snapshot without `bonds`, and no `mint_gate` key (`:1242-1274`).
- **Owner map.** Agents come from `AgentRegistered` logs and owners from `OwnerBound` logs, with the last binding per agent winning. Revoked agents and their whole subtree are dropped (`:1309-1354`).
- **REP.** REP comes from RepToken `getPastBalance` and `getPastTotalSupply` at the pinned timestamp. It reads balances, not votes (`:1359-1370`, docstring `:27-31`). Household REP is the sum of its bound agents plus the owner's own balance when the owner is not itself an agent (`:1374-1389`). Then `rep_shares[house] = round(rep/supply, 9)` and `voice_weights[house] = round(epsilon + share, 9)` with `epsilon = VOICE_EPSILON = 0.0` (`:1393-1396`, `federated_reconcile.py:136`).
- **Registry** (charter, sanctions, robot policy, containment, mint gate, epoch rules). Rebuilt from `RegistryUpdated` logs up to the pinned block (`:1405-1414`, `:1482-1485`).
- **Review snapshot.** Rebuilt from ToolBond logs. The reviewer-share window opens at the block of `cutoffs[-2]` (pipelined) or `getAnchor(count-2)` (non-pipelined) (`:1437-1447`). `review_snapshot["supply"]` is set to the REP supply (`:1447`).
- **Epoch window.** Covered in section 5.

### 2.5 Canonical ordering (`nodes/common/canonical_ordering.py`)

1. Group batches by sender. Sort each sender's batches by `batch_seq`. The run must be contiguous, and `prev_batch_hash` must link from batch to batch. A gap or a hash mismatch drops the whole sender (`:154-211`). The earliest batch in the window is checked against an empty hash only when its seq is 1 (`:199-202`).
2. Each sender's root is a binary sha256 merkle over its batches' content hashes. An odd last leaf is duplicated (`:126-146`).
3. Senders are sorted by root, and their chains are concatenated (`:254-261`).
4. `epoch_root()` is the merkle of the sender roots in root order. With no sender it is `EMPTY_EPOCH_ROOT = sha256("autonet:world-model:empty-epoch:v1")` (`:68-70`, `:87-106`).

---

## 3. The federated close

### 3.1 Driver gating (`federated_close_driver.py`)

- **Pipelined tick** (`_run_pipelined`, `:1219-1269`). The driver reads the latest 16 anchors (`nodes/service.py:982-996`). It reconciles them against its own unanchored local epochs. If an epoch was anchored with a different root or id, it rewinds: local epochs are voided, their batches restored, and the anchored carry-over and world installed (`:390-466`). It then computes the pin `{number = anchorCount + len(local) + 1, cutoffs, prev_root}` (`:468-480`) and refreshes voice state when the pin changes (`:1238-1246`).
- **Participation blockers.** The driver refuses to close without a voice source, without carry-over files past genesis, with a refresh error, with a failed tracker, with an uninstalled genesis seed, or when the pinned charter differs from the active one (`:822-865`, `:959-990`).
- **Charter change.** If the pinned `charter_text` is not byte-equal to the active charter, the driver halts and signals a restart (`:959-990`).

### 3.2 Inputs to `federated_epoch_close` (`federated_close_driver.py:1082-1104`)

- the canonical order
- carried `tool_registrations`, `tool_vetting`, `tool_positions`
- `agent_owner_map`
- `voice_weights`, or None when empty and the close is not chain-backed past genesis
- `rep_shares`, same rule. A chain-backed close past genesis passes the empty maps, so an empty REP read weighs nothing and keeps the gate on (`federated_close_driver.py:1082-1086`, `:1099-1105`, HEAD)
- `emission_pool`, or None without a chain source
- `review_snapshot`
- `sanctioned_households`
- `denied_tools` (computed by `containment.denied_digests`, `:933-939`)
- `mint_gate`
- `receipt_domain`
- `pricing` (from `world_service.pricing`, default `"ledger"`, `nodes/service.py:880`)

`seed_world` is never passed, so every close replays onto a fresh charter world (`federated_reconcile.py:1347-1352`).

### 3.3 Ordering and determinism (`federated_reconcile.py`)

- **Ledger replay.** For each canonical batch: `apply_events(equilibrate_after=False)`. Sprout `node_id`s are rewritten to the live ids on copies. `tool_used` events and registration sprouts are stamped with `sender = batch.sender_pubkey.hex()` on copies (`:1366-1389`, `:1605-1623`).
- **Fixed evaluation order.** Score snapshots evaluate `net_score` in sorted node-id order after clearing caches (`reconcile.py:83-112`).
- **Sorted iteration.** All maps are iterated in sorted key order. `log1p` comes from `world_model.generalized.detmath`, so it is independent of the C library (`:537-538`).
- **Rounding.** Outputs are rounded to `OUTPUT_DECIMALS = 10` (`:67`). `_round_dict` drops values that round to zero (`:1695-1703`). Positions are rounded to 9 places and own-share sums to 10.
- **Score movement is diagnostic only.** Per-node score mint and novelty are computed but never enter `agent_mint` (`:1080-1096`). The only earning rail is `extra_node_agent_mint` from `compute_tool_mint` (`:1106-1119`). `agent_weights` is never supplied by the federated path, so every weight is 1.0 (`:1114`).
- **Violator-pays gate.** Present but off: `apply_gate=False` (`:1021`, `:1277`).

### 3.4 Output pipeline after `compute_tool_mint`

1. Tool positions (ToolBond ratings) are written onto the replay world (`:1468`, `:1223-1264`).
2. `federated_reconcile_epoch` merges the tool mint (`:1470-1477`).
3. **Pool normalization.** `scale = emission_pool / raw_total`, and every agent mint is multiplied by it (`reconcile.py:466-498`). If `raw_total <= 0`, nothing changes.
4. **Floor to base units.** `k_a = floor(v_a * 1e6)`. Any excess over `round(pool * 1e6)` is removed one unit at a time from the largest allocations, ties broken by key. Dust stays unminted (`:1647-1682`).
5. **Sanctions withheld after the pool split.** An agent whose owner or own address is a sanctioned household is removed. Its amount goes to `withheld_mint` and is not redistributed (`:1494-1515`).
6. **Household re-key.** Each agent amount goes to `owner_map.get(agent, agent)`, lowercased and summed (`:1516-1525`, `:1583-1602`).
7. The per-digest display `mint` and `reviewer_mint` are scaled by the same `pool/raw_total` factor (`:1483-1493`).
8. **`authoritative_payload`, schema 6** (`:1553-1579`): `epoch_root`, `agent_mint` (keyed by household), `agent_novelty`, `total_mint`, `total_novelty`, `output_decimals`, `gate_applied`, `pricing`, `charter_hash` (sha256 hex of the charter bytes, no 0x, `charter_version.py:47-50`), `sanctioned_households`, `n_batches`, `n_events`. The driver adds `carry_cid` (`federated_close_driver.py:1153-1158`) and `world_cid` (`:1166-1178`).

### 3.5 Carry-over and the submitter pick

- **Carry-over** is three maps: `tool_registrations`, `tool_vetting`, `tool_positions` (`authoritative_encoding.py:286-290`). They are persisted after each close (`federated_close_driver.py:1123-1130`) and committed by `carry_cid = sha256(sorted-key JSON bundle)` (`authoritative_encoding.py:293-330`).
- **Submitter pick.** `idx = int(sha256("autonet:submitter:v1" || epoch_id || canonical_root)[:8]) mod |sorted senders|` (`federated_close_driver.py:89-109`). The sender set is `gossip.known_senders()` (`:1191`).

---

## 4. The mint computation (`compute_tool_mint`, `federated_reconcile.py:389-1009`)

Notation: for a household `h` and a tool `d`, `n_{h,d}` is the number of attested, `ok` receipts from `h`'s agents for `d`.

### 4.1 Registrations

- The carried registration map comes first. This epoch's registration sprouts follow in `(author_agent, seq)` order with `setdefault`, so the first registration of a digest wins forever (`:546-560`).
- The registration batch's gossip key is stored as `meta["sender"]` (`:558-559`).

### 4.2 Usage aggregation

- `usage = tool_usage_from_events(verified_receipts(events, receipt_domain))` (`:562-563`).
- Only `ok and attested` receipts populate `attested_ok_by_caller` and `attester_senders`. Vets (`vet: true`) are excluded from usage (`tool_usage.py:115-159`). `own_share` is clamped to [0, 1] and summed per caller (`tool_usage.py:174-186`).

### 4.3 Households, the owner map and the sanctions skip

- `_household(agent) = owner_map[agent.lower()] or agent.lower()` (`:666-667`). Every lookup lowercases both sides (`:582-593`).
- In the usage loop, a sanctioned household's receipts are skipped: "a sanctioned household's usage credits nobody" (`:778-783`). The loadout adoption loop skips it too (`:815-816` at HEAD).

### 4.4 Collapse, then damp, then fan out

For each digest `d` that is both used and registered:

```
damped_{h,d} = log1p( sum over h's agents of attested_ok_count )        (:782-787)
eff[t][h] += damped_{h,d} * share_d(t)   for each t in composition_shares(d)   (:788-790)
```

N agents under one owner are one `log1p` term. The value is damped first and split second, so padding a composite with dependencies is neutral (`:650-658`).

### 4.5 Composition fan-out (`_composition_shares`, `:194-232`)

- One attestation carries total weight 1.
- At a node with declared deps and `depth < COMPOSITE_MAX_DEPTH = 4`, the node keeps `weight * own`. Here `own = reviewed root share, else COMPOSITE_ROOT_SHARE = 0.7`, clamped to [0, 1] (`:74-75`, `:219-220`).
- Each declared dep gets `weight * (1 - own) / len(declared)`, where the divisor is the full declared count (`:225`).
- A dep that is unregistered or already on the path forfeits its share. The total can be below 1 but never above 1 (`:226-229`).
- A node with no deps, or at depth 4 or more, keeps the full weight reaching it (`:216-218`).

**Reviewed root share** (`:713-764`). For a composite (a tool with deps), each household's own-share estimates contribute `w = log1p(n) * rep_share(h)`, or `w = log1p(n) * 1.0` when `rep_shares` is None. Zero-REP households contribute nothing. The author's household and households whose senders include the registration key are excluded. Then `o_sum += w * mean`, `o_mass += w`, and the root share is `clamp(o_sum / o_mass, 0, 1)`, rounded to 9 places. The estimate accumulates across epochs through `positions[d]["own_share"]`.

**Successor carry** (`:687-701`). A manifest whose `version_of` names a registered predecessor by the same author starts from the predecessor's carried position. Until ToolBond has rated the successor itself, it shows the predecessor's rating (`:983-988`).

### 4.6 Loadout adoption

- Adoption reads the same `verified_receipts(events, receipt_domain)` as usage, so unsigned or forged attestations credit no adoption (`federated_reconcile.py:806`, HEAD).
- Each distinct household (fleet) whose attested receipts carry `loadout = L` contributes `log1p(1)` once per epoch to `eff[t][h]` over `L`'s composition shares. A sanctioned household is skipped (`:807-826`, HEAD).
- Its senders are pooled into `house_senders` for the wire dedup (`:819-821`, HEAD).

### 4.7 Per-tool rules, in order (`:865-962`)

1. **Denied tools.** `digest in denied_tools` → skip (`:866-867`).
2. **Bond rule.** Applies only when the snapshot carries `bonds`, which happens only when ToolBond logs were read.
   - The tool must be bonded, not unbonding, `stake >= min_stake` and `stake > 0`.
   - The mint author becomes the bond's author (`:870-879`).
   - Then, bonded or not, a tool with no author is skipped (`:884-885` at HEAD). This check sits between the bond rule and the pinned rule.
   - Bonds come from `Bonded` events (stake = the event's `stake` field, unbonding cleared), `UnbondRequested`, `Unbonded` (removes the bond) and `Finalized` (stake reduced by `bondSlashed`). `min_stake` comes from the latest `ParamsSet`, default 0 (`voice_state.py:421-445`).
3. **Pinned only.** `manifest_meta.trust_class` must equal `"pinned"` (`:884-885`, `tool_manifest.py:41`).
4. **Exclusions** per household `h` in `eff[d]`:
   - skip the author's household: `h == household(author)` (`:905-906`)
   - skip co-hosted households: `reg_sender in house_senders[h]`, meaning any of `h`'s attestations rode the same gossip key as the registration (`:907-908`)
5. **Weighted usage term.**
   ```
   usage_term_d = sum over remaining h of eff[d][h] * round(mint_weight(h), 12)        (:909)
   mint_weight(h) = round(voice_weights.get(h, VOICE_EPSILON = 0.0), 9)
                    or 1.0 when voice_weights is None                                   (:832-837)
   ```
   With ε = 0, a household's mint weight is its REP share. `voice_weights` is None only off chain or at genesis; a chain-backed close past genesis passes `{}` even when the REP read is empty, so every household then weighs 0 (`federated_close_driver.py:1082-1100`, HEAD).
6. **Breakthrough gate** (`:862-864`, `:911-917`). It is on when `mint_gate is not None and rep_shares is not None`. Since d5f77db `rep_shares` is `{}`, not None, on a chain-backed close past genesis, so an empty REP read leaves the gate on (`federated_reconcile.py:866`, `federated_close_driver.py:1101-1105`, HEAD).
   - `gate_rep = sum of rep_share(h)` over the counted households with share > 0. `gate_holders` is the count of those households. `n_holders` is the count of all households with positive `rep_shares`.
   - The tool earns nothing this epoch if `gate_rep * 10000 < rep_bps` or `gate_holders * 10000 < houses_bps * n_holders`.
   - Defaults are `MINT_GATE_DEFAULTS = (100, 50)` (`containment.py:22`): users must hold at least 1% of REP supply and be at least 0.5% of REP-holding households.
   - Registry keys are `jurisdiction.<id>.mint_gate_rep_bps` and `jurisdiction.<id>.mint_gate_houses_bps` (`containment.py:25-35`).
   - After the first anchor `read_voice_state` always returns a gate (defaults when unset, `voice_state.py:1484-1485`). At genesis it does not.
7. **No mint, no entry.** `mint = usage_term`. If `mint <= 0` the tool is skipped (`:919-921`). No standing multiplier and no vet gate apply.
8. **Mint node.** The lexicographically first claim node with that artifact digest, else `"tool:<digest>"` (`:925-926`). Because the replay world is fresh, a tool registered in an earlier epoch gets `"tool:<digest>"`.
9. **Reviewer share** (`:932-944`, `reviewer_weights` at `:324-362`).
   - `reviewer_bps = snapshot.reviewer_share_bps`. The default is `REVIEWER_SHARE_BPS_DEFAULT = 4000` (40%) when ToolBond never emitted `ParamsSet` (`:88`, `voice_state.py:293`, `:532`). It is clamped to [0, 10000].
   - Per reviewer, over the tool's rounds finalized in the window:
     ```
     w = (accuracy / 1e18) * novelty
     novelty = mean over scored axes (plus "harm" if a harm score was given) of 1 / (1 + M_axis / supply)
     ```
   - A review needs `stake > 0` and `accuracy > 0`, and it must score at least one axis. The stake is not part of the weight (`:346-358`).
   - `M_axis` is the tool's REP mass on that axis before the round (`voice_state.py:485-486`). `supply` is the REP supply at the round's ToolBond epoch, falling back to the snapshot supply. Accuracy `2^64 - 1` means uninformative and is treated as None (`voice_state.py:296`, `:496`).
   - The author's own address is removed from the weights (`:933-935`).
   - `share_pool = mint * bps / 10000`. Each reviewer gets `share_pool * w / total_w`, and the author gets `max(0, mint - reviewer_total)`.
   - Reviewer addresses map to an agent id already present in the close, using the sorted-first spelling, else the lowercased address (`:845-857`).
   - Sanctioned reviewers keep their place in the split, and their part is withheld at step 5 of section 3.4 (`voice_state.py:382-385`).

### 4.8 Positions (ratings), no mint effect (`:979-990`, `tool_rating_position` `:365-386`)

```
head_axis  = round(sum / mass / 100, 9)    (in -1..1)
mass_axis  = round(mass / REP supply, 9)
harm       = round(harm_sum / harm_mass / 100, 9)    (0..1)
harm_mass  = round(harm_mass / supply, 9)
```

How the rating sums are built in the snapshot (`voice_state.py:461-508`):

- Each review's `weight` (from the `Reviewed` event) times its int8 score in -100..100, summed per axis. `-128` means no score.
- Harm in 0..100, where 255 means none.
- Axis keys are mapped through `keccak(axis_id)`. Keys outside the active charter are dropped.
- An `Overruled` event replaces a round's harm with `harm_mass * DAO harm` (`:480-483`).
- Harm in its appeal window (`HarmSettled` but not finalized) goes into `cooling_harm`. It counts for containment only (`:510-529`).

### 4.9 Vetting (dead knobs)

- Vets are still merged, with the same self, same-owner and same-sender exclusions (`:595-619`).
- A manifest is greenlit when the sum over distinct fleets of the maximum member weight `1/(1 + busts)` reaches `VET_QUORUM = 2.0`. Greenlight sets `royalty_left = VET_ROYALTY_EPOCHS = 8`, and `royalty_left` ticks down once per close (`:628-642`, `:966-969`).
- `VET_ROYALTY_SHARE = 0.1` and `VET_BUST_THRESHOLD = 0.5` are never used (`:102-105`). The greenlit flag is informational (`:886-890`).

### 4.10 Pool size from burned fees

- **Fees on chain.** `payForService` charges `fee = amount * serviceFeeBps / 10000`, with a default of 250 bps, bounded to 50..1000 and set by the governor (`Substrate.sol:1226-1261`). `toTreasury = fee * 5000 / 10000` if a treasury is set, and `burned = fee - toTreasury` (`:1348-1352`). Burns are checkpointed by timestamp (`:1367-1375`).
- **On-chain pool.** `anchor` sets `mintPool = previous anchor's windowBurned`, which is 0 for the first anchor. It sets `windowBurned = burnedUpTo(cutoff) - burnedAtLatestCutoff` (`Substrate.sol:300-307`, `:260-263`). This is a one-epoch lag.
- **Off-chain pool.** The close uses `emission_pool = burned_raw / 1e6` (`voice_state.py:197`, `:1303`). Burned fees are the only emission: an epoch with no burned fees mints nothing.
- **Claim cap.** `claimFor` reverts with `EpochPoolExceeded` when the cumulative mint passes `mintPool` (`Substrate.sol:943-950`).

---

## 5. When an epoch ends (the cutoff rule)

### 5.1 Governed rules

Read from `jurisdiction.<id>.<name>` at the pinned block, with defaults in `voice_state.py:842-859` (parsed at `:894-911`):

| key | default | meaning |
|---|---|---|
| `epoch_target_share` | 0 | ppm of ATN supply that must burn in fees to end an epoch. 0 means every epoch lasts the floor. |
| `epoch_retarget_window` | 4 | Retired. Read and ignored. |
| `epoch_ceiling_votes` | 10 | Ceiling on epoch length, in voting periods. |
| `epoch_grace` | 120 | Seconds after the cutoff before a daemon closes. |
| `epoch_skew` | 60 | Maximum gap, in seconds, between signed time and arrival for a batch. |

### 5.2 Window (`epoch_window`, `:1005-1044`)

- `base = previous cutoff`. Before epoch 1 it is `Substrate.deployedAt`.
- `floor = votingDelay + votingPeriod + timelock minDelay`, from governor and timelock settings events (`:932-966`). All three are 0 without a governor.
- `ceiling = max(floor, epoch_ceiling_votes * votingPeriod)`.
- `length = floor`.

### 5.3 Cutoff selection (`_select_window`, `federated_close_driver.py:495-550`)

1. `cutoff = base + length`.
2. If `target_share > 0` and a burn source exists, `volume_cutoff` gives the first second `t` in `[start + floor, start + ceiling]` with `burnedBetween(start, t) >= target_share_ppm * ATN_supply // 1e6`. It reads only time up to `now - grace`. A quiet network closes at `start + ceiling` (`voice_state.py:969-1002`).
3. If the earliest pending batch is after the cutoff, the cutoff slides forward in steps of `grace` (`:539-541`).
4. The daemon closes once `now >= cutoff + grace`. Members are batches with `start < timestamp <= cutoff`.
5. The epoch id is `epoch-<n>-<first 16 hex of prev root>-<cutoff>` (`voice_state.py:914-917`).

### 5.4 On chain

`anchor` requires `latestCutoff < cutoff <= block.timestamp`, `prevEpochRoot == latestEpochRoot` and `prevAnchorHash == latestAnchorHash` (`Substrate.sol:286-298`). The anchor hash covers content only, `keccak(abi.encode(the 8 fields))` (`:350-369`), so the next epoch's `prevAnchorHash` is known before the anchor lands.

---

## 6. Anchoring

### 6.1 Encoding (`epoch_anchorer.prepare_anchor`, `:218-301`)

- `payload_bytes = encode_authoritative_payload(...)`. This is compact JSON with a fixed field order: schema 6, epoch_id, epoch_root, prev_epoch_root, agent_mint, agent_novelty, total_mint, total_novelty, output_decimals, gate_applied, n_batches, n_events, world_cid, carry_cid, charter_hash, sanctioned_households. `seed: true` is appended only on a genesis seed. Floats are written as `f"{x:.10f}"` (`authoritative_encoding.py:105-214`).
- `payloadHash = sha256(payload_bytes)` (`:256-259`).
- The mint blob is `{"schema": 6, "agent_mint": {...}}`. Its cid is the sha256 hex of the blob (`:217-253`).
- `agentMintRoot = mint_merkle_root(agent_mint)`.
- `cutoff` is taken from the epoch id. If the id has none, it is `max(latest block timestamp, latestCutoff + 1)` (`:272-278`).
- In pipelined mode the option chains onto the parent this daemon prepared: `prev_root = parent.epoch_root` and `prev_hash = anchorHashOf(parent fields)` (`chain_submission_driver.py:354-385`).

### 6.2 The merkle leaf (`mint_merkle.py`, `Substrate.sol:929-939`)

- `leaf = keccak256(keccak256(abi.encode(address claimant, uint256 amount)))`.
- `amount = round(raw * 1e6)` (`scale_mint`, `:50-56`).
- Pairs are hashed in sorted order. An odd node is promoted unchanged. A single leaf is its own root.
- Only keys that parse as addresses and have a non-zero amount become leaves (`:59-78`). An empty tree gives a zero root, and `claimFor` rejects claims against it (`Substrate.sol:930`).

### 6.3 Paths (`chain_submission_driver.handle_federated_close`, `:222-317`)

1. **Election** (when `anchor_election_address` is set).
   - The election is one DAO proposal calling `AnchorElection.anchorWinner(epochNumber, attempt)`, with the description `"Anchor election: epoch N, attempt A"` (`AnchorElection.sol:134-159`).
   - The Python side proposes with any key that meets `proposalThreshold`. While the proposal is Pending, a drawn local key registers this daemon's option. While it is Active, every local key with votes votes for the option whose hash matches its own close. A succeeded proposal is queued, then executed after its eta (`anchor_election.py:105-220`).
   - **Draw.** A live agent is drawn if `mulDiv(h, supply, 2^256-1) < proposers * votes`, where `h = keccak(chainid, this, epoch, attempt, agent)`. Defaults are `proposers = 3`, `proposerFloorPpm = 0`, `maxOptions = 8`, `openAfterAttempts = 2`. From attempt 2 on, any live agent may register (`AnchorElection.sol:85-91`, `:171-183`).
   - `anchorWinner` requires `epochNumber == anchorCount + 1` (`:249-260`).
   - A defeated, expired or canceled proposal moves to the next attempt. So does a queued proposal still unexecuted `EXECUTE_GRACE_S = 600` seconds past its eta, for example when Substrate rejects the winning option and every `anchorWinner` reverts (`anchor_election.py:38-42`, `:127-145`, HEAD).
2. **Governed proposal.** The proposal calls `Substrate.anchor` with the description `"Anchor epoch <id> 0x<root>"` (`governed_anchor.py:78-80`, `:136-148`). The winner proposes first. The others propose after `tick_interval = 15 s` (`chain_submission_driver.py:129`, `:517-519`).
3. **Direct.** When Substrate's governor is this daemon's key (dev), only the `pick_submitter` winner anchors (`:282-288`).

The ticker drops an election or proposal after `_MAX_PROPOSAL_AGE_S = 9 days` (`:72`). Before 8a7241d a stuck-queued election waited out this limit.

### 6.4 Mint claims and REP claims

- **Claimants.** The configured `owner_wallet`, each local agent's on-chain `getAgentOwner`, and each agent address itself (`chain_submission_driver.py:564-601`).
- **Claim flow** (`authoritative_submitter.submit_for_epoch`, `:183-363`). Read the anchor by epoch id, fetch the blob by cid, verify its sha256, decode the claimant's share, and scale it. Zero is skipped. Build the proof and pre-verify it against the on-chain root. Skip if `hasSubmittedForEpoch`. Then call `claimFor(claimant, amount, keccak(epochId), proof)`, signed by the operator key. Anyone may submit a claim (`Substrate.sol:910-972`).
- **REP.** For each claimant, `AutonetRepClaim.claimable(earner)`, then `claim(earner)` when non-zero (`chain_submission_driver.py:632-660`). The REP basis is `agentMintTotal`, at 1:1 per contract comments (`Substrate.sol:855-859`, `:952-956`), and `serviceEarnings` (`:1263-1267`).

---

## 7. Reviews

### 7.1 Queueing (`atn/runtime/review_batch.py`)

- A review is `{digest, axes?, harm?, stake?, evidence_hash?}`.
- Axes outside the charter are dropped. Scores clamp to [-1, 1] and are encoded as an int8 `round(x * 100)` in [-100, 100], with -128 meaning no score. Harm clamps to [0, 1] and is encoded as 0..100, with 255 meaning none (`:115-140`, `:445-481`).
- A review with no charter axis and no harm is skipped (`:469-470`). The author's own tool is skipped (`:519-524`). The agent must be registered on chain (`:503-506`).
- There is one batch file per agent per ToolBond epoch. A second review of the same tool replaces the first. Once a batch is committed, new reviews go to `epoch + 1` (`:245-254`).

### 7.2 Commit-reveal (`atn/review_driver.py`)

- **Commit.** Triggered when this daemon's federated close fires (`autonet_service.py:1432-1440`, `review_driver.py:103-115`). It applies only to batches for `ToolBond.currentEpoch()`.
  - `commitment = keccak(abi.encode(epoch, agent, ReviewInput[], salt))` (`review_batch.py:164-170`). `salt` is 32 random bytes and is saved to disk before the transaction is sent (`review_driver.py:219-243`).
  - The commit transaction carries `commit(commitment, totalStake, reviewCount)`, after an ATN `approve` for the stake (`review_batch.py:363-390`).
  - An uncommitted batch from an older epoch rolls forward (`review_driver.py:177-181`).
- **Reveal.** Sent after `revealEnds(epoch) > 0`, meaning the epoch is anchored, and before the deadline. Any local key may relay it. After the window closes, an unrevealed batch is marked forfeited (`:268-304`).
- **Finalize and claim.** After the window, the driver calls `finalize(digest, epoch, 50)` up to 20 times per round. A round in the cooling stage waits out its appeal window unless overruled. When the round is done, `claim(digest, epoch, reviewer)` (`:306-371`, constants `:49-51`).

### 7.3 Harm and containment tiers (`nodes/common/containment.py`)

Defaults are `deny_harm 50.0`, `trust_harm 10.0`, `trust_mass_bps 10.0` (`:18-19`). Registry keys are `jurisdiction.<id>.containment_{deny_harm,trust_harm,trust_mass_bps}` (`:58-71`).

```
harm      = harm_sum / harm_mass          (0..100)
share_bps = harm_mass * 10000 / REP supply
quorate   = share_bps >= trust_mass_bps
```

| Condition | Tier |
|---|---|
| no harm mass | strict |
| `harm >= 50` and quorate | denied |
| `harm < 10` and quorate | declared |
| otherwise | strict |

Source: `:38-55`.

- **In the close.** A denied tool mints nothing. Its standing includes appeal-window (cooling) harm (`:74-91`).
- **In the daemon.** A network (adopted) tool runs only while bonded. Its sandbox tier comes from the driver's pinned `security_standing`, which includes cooling harm (`federated_close_driver.py:941-962`, HEAD; `atn/tool_security.py:140-211`).
- **In search.** `probe_tools` drops a tool whose final verdict is denied: `finally_denied` asks `security_standing(include_cooling=False)`, so harm still in its appeal window contains a tool but does not hide it. Unknown standing is shown (`atn/tool_security.py:214-228`, `:520-521`, `atn/autonet_service.py:1350-1354`, HEAD).

### 7.4 Sanctions

- **Registry key.** `sanction.<0x>`, whose value is JSON `{"until", "reason", ...}`. An empty value lifts the sanction. A non-JSON value sanctions with no end. A non-integer `until` is ignored (`voice_state.py:694-743`).
- **Pinned use in the close.**
  - A sanction is in force while the pin timestamp is before `until`.
  - The household is the address's owner if bound, else the address.
  - An agent rotated away from a sanctioned household at or after the sanction's block carries the sanction to its new owner, repeated transitively (`:1112-1162`).
  - Effects: their usage credits no tool (`federated_reconcile.py:780-781`), and since d5f77db their loadout adoption credits no distro (`:815-816`, HEAD). Their REP shares are removed from own-share weighting and from the gate holder count (`:1443-1448`). Their reviews move no rating, with weight 0 (`voice_state.py:468`). Their mint is withheld (`federated_reconcile.py:1494-1515`).
- **Live use** (`atn/sanctions.py`). The daemon reads `getAllKeys` and `getAllValues`, refreshed at most every 60 s (`:25`). An address is sanctioned if it is named directly, its owner is named, or a past owner (`OwnerBound.prevOwner`) is named (`:161-180`). Agents may not buy from, adopt from, or review for a sanctioned household (`:1-14`, `:194-203`).

---

## 8. Every DAO Registry key the code reads

### 8.1 Pinned in the close (from `RegistryUpdated` logs at the pinned block)

| key | default | where |
|---|---|---|
| `jurisdiction.autonet.charter` (hard-coded id) | none; must start with `{` | `voice_state.py:612`, `:627-644` |
| `sanction.<addr>` | none | `voice_state.py:694-743` |
| `jurisdiction.<id>.robots` | `[]` | `voice_state.py:801-802` |
| `jurisdiction.<id>.robot_standards` | `[]` (64-hex pack digests) | `:803-805` |
| `jurisdiction.<id>.parent` | none (`{"registry","id"}`; max depth 8) | `:759`, `:806-815` |
| `jurisdiction.<id>.epoch_target_share` | 0 | `:846` |
| `jurisdiction.<id>.epoch_retarget_window` | 4 (unused) | `:849` |
| `jurisdiction.<id>.epoch_ceiling_votes` | 10 | `:851` |
| `jurisdiction.<id>.epoch_grace` | 120 | `:854` |
| `jurisdiction.<id>.epoch_skew` | 60 | `:858` |
| `jurisdiction.<id>.mint_gate_rep_bps` | 100 | `containment.py:22-35` |
| `jurisdiction.<id>.mint_gate_houses_bps` | 50 | same |
| `jurisdiction.<id>.containment_deny_harm` | 50.0 | `containment.py:18`, `:58-71` |
| `jurisdiction.<id>.containment_trust_harm` | 10.0 | same |
| `jurisdiction.<id>.containment_trust_mass_bps` | 10.0 | same |

`<id>` is always `"autonet"` in the close: `read_voice_state` defaults `jurisdiction_id="autonet"` (`voice_state.py:1175`), and `nodes/service.py:968-975` does not pass it.

### 8.2 Read at join (current values, `atn/config.py:907-936`, `:580-657`)

`jurisdiction.<id>.` followed by any of: `substrate`, `dao`, `rpb`, `rep_token`, `deploy_block`, `charter`, `charter_uri`, `epoch_rules` (must be `"2"`, `:677`, `:687-694`), `containment_deny_harm`, `containment_trust_harm`, `containment_trust_mass_bps`, `bootstrap`, `timelock`, `registry` / `dao_registry`, `economy`, `service_registry`, `payment_channel`, `anchor_election`, `rep_claim`, `tool_bond`.

A jurisdiction will not start without `substrate`, `rep_token`, `tool_bond`, `dao` and `timelock` (`config.py:664-670`).

### 8.3 Live

`sanction.*` via `getAllKeys` / `getAllValues` (`atn/sanctions.py:60-69`).

### 8.4 ToolBond parameters (not Registry, but governed inputs)

From the latest `ParamsSet` log: `reviewerShareBps` (default 4000 when no event) and `minStake` (default 0) (`voice_state.py:421-423`, `:532`, `:539`).

---

## 9. Inconsistencies and dead code

1. **Reviewer weighting is described two ways.** `compute_tool_mint`'s docstring and the constant comment say reviewers split "by stake x accuracy x novelty" (`federated_reconcile.py:83-86`, `:499-502`). `reviewer_weights` uses `accuracy x novelty`, with the stake only required to be positive (`:329-330`, `:356-358`).
2. **Stale ε comments.** Several places describe weight as `epsilon + share`, with ε bounding zero-REP voice and bootstrapping a cold start: `voice_state.py:10-16`, `:1207-1210`, `federated_close_driver.py:236-242`, and "ε-floored" throughout `compute_tool_mint`'s docstring. `VOICE_EPSILON = 0.0` (`federated_reconcile.py:136`). The driver comment even says `household_ATN / supply` (`federated_close_driver.py:239`).
3. Fixed in d5f77db: a chain-backed close past genesis now passes the empty maps, so empty REP weighs nothing and the gate stays on (`federated_close_driver.py:1082-1105`, HEAD). Original finding: **No RepToken means weight 1.0, not 0.** With no RepToken, or an empty REP read, `voice_weights` and `rep_shares` are `{}`. The driver passes None for empty maps (`federated_close_driver.py:1089-1095`). Every household then weighs 1.0 and the breakthrough gate is off (`federated_reconcile.py:832-837`, `:862`). This is the opposite of the "ε = 0, no free weight" rule. Config refuses to start a jurisdiction without `rep_token`, but a RepToken with zero supply still yields empty maps.
4. Fixed in d5f77db: adoption now reads `verified_receipts(events, receipt_domain)` (`federated_reconcile.py:806`, HEAD). Original finding: **Loadout adoption bypasses receipt signatures.** `loadout_adoption_from_events(events)` reads raw events, not `verified_receipts(...)` (`federated_reconcile.py:804`). Unsigned or forged attested receipts with a `loadout` stamp still credit adoption.
5. Fixed in d5f77db: the adoption loop skips sanctioned households (`federated_reconcile.py:815-816`, HEAD). Original finding: **Loadout adoption bypasses sanctions.** The adoption loop never checks `sanctioned_set` (`:805-822`), unlike the usage loop (`:780-781`).
6. **Charter key is hard-coded.** `charter.py:3-4`, `charter_version.py:5-6` and `atn/config.py` use `jurisdiction.<id>.charter`. The close reads `CHARTER_REGISTRY_KEY = "jurisdiction.autonet.charter"` (`voice_state.py:612`, `:1408`). Every other pinned key also uses the default id `"autonet"`, because `nodes/service.py` never passes `jurisdiction_id`.
7. **Containment thresholds come from two sources.** `containment.py:3-6` says one rule shared by daemon and close. The close uses thresholds pinned to the snapshot (`voice_state.py:1482-1483`). The daemon's run tier uses `RPBConfig.containment_*` from the join-time registry seed (`atn/tool_security.py:183-193`, `atn/config.py:624-627`). A DAO change applies to the close at the next epoch and to sandboxes only after a restart.
8. **The submitter pick is not canonical.** `pick_submitter` claims all daemons compute the same winner (`federated_close_driver.py:15-20`, `:94-98`). Its input is `gossip.known_senders()`, the daemon's lifetime set of seen senders (`event_gossip.py:654-663`), which differs between daemons. On the election path `is_winner` is unused.
9. **Pool description mismatch.** `read_voice_state`'s docstring says the pool is "burned fee shares (ServiceFee logs)" (`voice_state.py:1196-1198`). The code reads `windowBurned` or `burnedBetween` (`:1292`, `:1302`). The `ServiceFee` event ABI in `_VOICE_ABI` is unused (`:181-192`).
10. **Possible pool mismatch at a cutoff second.** The pipelined pool is `burnedBetween(c[n-2], c[n-1])` (`voice_state.py:1292`). On-chain `windowBurned` subtracts `burnedAtLatestCutoff` as it stood when the previous anchor landed (`Substrate.sol:305-307`). The two differ if fees burn in the cutoff's own second after that anchor. The non-pipelined `snapshot_anchor_count` guard is skipped in pipelined mode (`chain_submission_driver.py:328-330`).
11. **Rounding described as truncation.** `authoritative_submitter.py:32-35` says floats are "truncated". `scale_mint` rounds (`mint_merkle.py:50-56`). The same file's anchor tuple comment lists 10 fields (`:210-213`); the struct has 14.
12. **Stale submitter description.** `epoch_anchorer.py:1-18` says the on-chain anchor submitter "is always an agent address". `Substrate.anchor` is governor- or anchorer-only (`Substrate.sol:282-285`), and claims are signed by the daemon operator key.
13. **Carry bundle field count.** The encoding spec says the carry bundle holds "the five tool maps" (`authoritative_encoding.py:31`, `:98`). `CARRY_KEYS` has three (`:286-290`).
14. **`pricing` is not committed.** `pricing` is in `authoritative_payload` (`federated_reconcile.py:1573`) but missing from `_FIELD_ORDER` (`authoritative_encoding.py:110-127`), so the payload hash does not cover it.
15. **Epoch rules default.** `RPBConfig.epoch_rules` comment says "Empty = '1'" (`atn/config.py:279-281`). `epoch_rules_error` treats empty as `"2"` (`:689`). The "2" comment mentions "the length retargeted" (`:674-676`), while `voice_state.py:839`, `:847-849` say the retarget is retired.
16. **Vetting is dead.** Greenlight and royalty state is computed, persisted and committed through `carry_cid`, but has no mint effect (`federated_reconcile.py:474-481`, `:886-890`, `:964-969`). `VET_ROYALTY_SHARE` and `VET_BUST_THRESHOLD` are never read. Bust detection is dormant (`:621-626`).
17. **Unused usage fields.** `tool_usage_from_events` still builds `axis_reviews_by_caller`, `vet_axis_reviews_by_caller` and `fee_total` (`tool_usage.py:59-76`). `compute_tool_mint` reads none of them, and attested events no longer carry axes (`atn/tool_store.py:1896`). The module docstring still describes the deleted β cap and decoupled REP mint (`tool_usage.py:9-22`).
18. **Stale event comments.** `ToolUsed` says "mint ∝ standing × usage" and that vets gate mint eligibility (`events.py:169-170`, `:217-224`). `submit_tool_manifest` justifies `author_post` as giving "standing 1 at birth" to mint (`world_service.py:1047-1050`). Standing no longer prices mint.
19. **Cross-epoch standing never seeded.** `compute_tool_mint`'s docstring says cross-epoch standing needs `seed_world` (`federated_reconcile.py:467-472`). The driver never passes one, so older tools anchor on `"tool:<digest>"` (`:925-926`).
20. **Violator-pays gate is dormant on the federated path.** `apply_gate=False` (`federated_reconcile.py:1021`, `:1277`). The local scheduler defaults to `apply_gate=True` (`epoch_scheduler.py:50`). It runs a time-based `epoch_emission_rate` projection with weight 1.0 (`world_service.py:2466-2559`, `nodes/service.py:650-657`), which differs from the fees-only authoritative close.
21. **Dead helpers and parameters.**
    - `GovernedAnchorer.anchor_direct` is unused; the driver uses `EpochAnchorer.submit_prepared` (`governed_anchor.py:131-132`).
    - `reconcile._node_novelty_alone` is unused (`reconcile.py:242-244`).
    - `epoch_window(history=...)` is documented as no longer used (`voice_state.py:1010`, `:1023-1024`).
    - `ChainSubmissionDriver(gossip=..., **_ignored)` is unused (`chain_submission_driver.py:128-131`).
    - `EpochAnchorer.anchor_close_result(signatures, signer_keys)` accepts and ignores them (`epoch_anchorer.py:326-336`).
    - `_closed_epoch_number` (one epoch in flight) applies only in the non-pipelined mode, which production never uses (`federated_close_driver.py:508-512`, `:547-549`).
22. **Duplicated constant and charter hash formats.** `REVIEWER_SHARE_BPS_DEFAULT = 4000` is defined in two modules (`federated_reconcile.py:88`, `voice_state.py:293`). `Charter.hash` and the config `charter_hash` use a `0x` prefix (`charter.py:73-75`, `config.py:614`). The payload `charter_hash` has none (`charter_version.py:47-50`).
23. **Two merkle conventions.** Canonical ordering duplicates the odd leaf and uses sha256 (`canonical_ordering.py:139-140`). The mint tree promotes the odd node and uses keccak with sorted pairs (`mint_merkle.py:90-101`). This is internally consistent, but easy to confuse.
24. **Wire dedup excludes honest households.** Any household whose attestations ride the registration's gossip key is excluded (`federated_reconcile.py:907-908`). That includes honest households hosted on the author's daemon, not only sybils.

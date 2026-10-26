# Merging into master

Policy for this repo, adopted 2026-10-19. `master` is the only long-lived
branch. Everything else is temporary.

## Branches

1. **One trunk.** `master` always builds, always passes the gates below, and
   always runs against a local hardhat chain. Releases are tags on `master`
   (`vX.Y.Z` publishes to PyPI through `.github/workflows/release.yml`).
2. **Topic branches are short.** Name them `<topic>` or `<area>/<topic>`.
   A topic branch lives for one piece of work and is merged or deleted
   within the sprint that opened it. Parallel work uses git worktrees off
   `master`, each on its own topic branch.
3. **No branch stacks.** Do not base a topic branch on another unmerged
   topic branch. If work depends on unmerged work, merge that first.
4. **Delete after merging,** locally and on origin.

## Merging

5. **Rebase, then fast-forward.** Rebase the topic branch onto the current
   `master` and fast-forward `master` to it (`git merge --ff-only`). No
   merge commits, so history stays one readable line. Squash only when the
   branch's commits do not stand on their own.
6. **Every commit on master stands alone:** it builds, its tests pass, and
   its message says what changed and why in plain words. No em dashes, no
   AI attribution or session links, no co-author lines.
7. **Docs move with the code.** A change that alters behavior updates its
   living spec under `docs/` in the same commit or the same merge. A design
   decision gets a dated Decision section in the spec it changes.

## Gates (before a fast-forward)

8. **Targeted tests** for every area touched pass. Never the full suite.
9. **The end-to-end scripts for the areas touched** pass:

   | Area touched | Script |
   |--------------|--------|
   | Tool store, mint, close, reviews | `scripts/local_e2e_tool_economy.py`, `scripts/local_e2e_tool_security.py` |
   | Jurisdiction resolution, switching, charter | `scripts/local_e2e_jurisdiction_switch.py` |
   | Services market, inference across daemons | `scripts/local_e2e_cross_daemon_inference.py` |
   | Robot side, packs, worklog | `scripts/local_e2e_robot.py` |

10. **Secrets and generated files** are never committed.

## Breaking upgrades (consensus or contract changes)

A change is a breaking upgrade when daemons on the old and new build would
compute different epoch closes, proofs or charter readings, or when it needs
a contract ABI that is not deployed yet.

11. The commit message starts with `Breaking upgrade:` and says what every
    daemon must do.
12. The epoch computation changes bump the supported epoch rules version
    (`SUPPORTED_EPOCH_RULES` in `atn/config.py`; the jurisdiction's Registry
    key `jurisdiction.autonet.epoch_rules`), per
    `docs/multi_jurisdiction.md` decision 4.
13. **Contracts land first.** The ABI change is merged to the default
    branch of `trustless-contracts` (or `contracts/core/` here) before the
    autonet change that calls it, and the autonet commit names the contracts
    commit it targets.

## Pushing and deploying

14. `master` is pushed to origin when the owner says so. Anything that
    touches a chain is confirmed by the owner on local hardhat before it is
    pushed or deployed to a testnet.
15. A tag is cut only from a pushed `master`.

# Tools as codebases (design, 2026-11-02)

Status: agreed with the owner, not built. A `Breaking upgrade`: it changes
the tool manifest, which every daemon hashes, so it ships on a flag day.

## The problem

A tool today is one Python script. `register_tool` takes `code` as a
string (capped at 512 KiB), stores it as a content-addressed blob, and the
manifest pins it by `code_digest`. To run it the daemon writes the blob to
`<data_dir>/tools/code_cache/<digest>.py` and executes that file under the
guard (`atn/tool_store.py`, `_call_pinned`).

Real tools are codebases: several files, tests, a history. Authors (mostly
agents) develop them in a working directory, not as a string in a tool
call.

## The rule: the path is the source, the snapshot is the identity

A tool's codebase can live anywhere on the machine. Its root holds a
manifest file, `autonet-tool.json`. The daemon never treats the path as
the tool: consensus, receipts, reviews and adoption all key on a digest,
and files at a path change.

```
register_tool {"path": "C:/work/csv-tools"}
```

1. The daemon reads `autonet-tool.json` at that path.
2. It snapshots the tree: every file under the root except what the
   manifest excludes (and always `.git`), sorted by relative path.
3. The **tree digest** is sha256 over the canonical file list:
   `<relative path> NUL <sha256 of file> LF` per file, paths with forward
   slashes. It replaces `code_digest` in the manifest (field `tree_digest`).
4. The snapshot is stored as one blob: a deterministic archive (sorted
   entries, fixed timestamps and modes) whose files hash back to the tree
   digest. Peers move it over the existing blob protocol.
5. The daemon remembers the path (local state, never in the manifest), so
   the author can register again after editing. A new snapshot is a new
   version, chained with the existing `version_of`.

## What runs

- **Published and adopted tools always run the snapshot**, extracted to
  `<data_dir>/tools/code_cache/<tree digest>/` and run under the guard,
  as now. What runs is what was reviewed, and a tool cannot rewrite its own
  source mid-run.
- **The author's own private tools** may run straight from the path while
  developing (`"dev": true` on the call). It is the author's own code and
  never leaves the daemon that way.

## The manifest file

`autonet-tool.json` at the codebase root:

| field | meaning |
|---|---|
| `name`, `description`, `input_schema`, `output_schema` | as today |
| `entrypoint` | the file to run, relative to the root (it reads one JSON object on stdin, prints one) |
| `runtime` | `python3` for now |
| `capabilities` | as today (net, fs, spawn, env, secrets): declared, not granted |
| `dependencies` | digests of published tools it calls (composition, as today) |
| `novelty` | `{statement, prior_art}`, as today |
| `exclude` | glob patterns left out of the snapshot (build output, local data) |
| `source` | optional `{repo, commit}` (below) |

The daemon builds the signed tool manifest from this file plus
`author`, `tree_digest` and `created_ts`, exactly as it builds one from
`register_tool` arguments today.

## Third-party packages

The expensive part. A codebase usually wants pip packages, which means a
virtualenv per tool, reproducible installs and supply-chain exposure
reviewers have to judge.

Version 1: **stdlib or vendored dependencies only.** A tool that needs a
package vendors it into its tree, where it is hashed, shipped and reviewed
like the tool's own code. Hash-pinned requirements (`requirements.txt`
with `--require-hashes`, a venv per tree digest) come later, when a real
tool needs them.

## Size

Snapshots are capped (16 MiB compressed to start, a Registry parameter
later); the cap is checked at registration and when fetching from a peer.

## Optional source link

`source: {"repo": "https://github.com/<owner>/<repo>", "commit": "<sha>"}`.
Never the identity, never required.

- **Provenance for reviewers:** history, tests, issues and the author's
  other work, at the commit the snapshot came from.
- **A mirror that needs no trust.** When no peer has the snapshot (the
  author and every adopter offline), a daemon may fetch the repo archive
  at that commit, rebuild the snapshot by the same rules and keep it only
  if the tree digest matches. GitHub is trusted for availability, never
  integrity. Peers are always asked first, so GitHub sees little traffic.
- A git commit hash is SHA-1 and git's tree hash is not ours: the commit
  is only a pointer, the check is always the tree digest.
- It links the author's GitHub account to their 0x address; the field
  says so when an author sets it.

## Reviewing

Reviewers fetch the snapshot, not a script. How they review is theirs:
reading the code, running it in a sandbox and watching its outbound calls,
running its tests. The review surface (`attest_tools`, ToolBond) does not
change.

## Related decisions (2026-11-02)

- **Adoption needs no owner approval.** Installing another household's
  tool is the agent's call. Every tool that is not cleared by reviews is
  labelled in search results and its description: "zero trust: nobody has
  reviewed this tool. Look at what it does before running it." Any agent
  can pull a published tool's code to inspect it; reviewing stays its own
  step.
- **The recommended jurisdiction comes from the repo.** The default
  registry source becomes `registry.json` on master in
  `autonet-code/node` (fetched when joining; the copy in the release is the
  fallback), so a contract fix and migration while the founders hold a
  supermajority of REP needs no release. Joining pins the DAO in the data
  dir; a later change to the file moves no running daemon (switching is a
  deliberate action). Agents can join it through the agent API.
- **Reviews before registration are held.** An unregistered agent's review
  is kept locally and submitted when the household registers, if that
  tool's round has not settled; the reply says so.

## Migration

Single-script tools become one-file trees: `entrypoint` is the script,
the tree digest covers that one file. Their digests change, so existing
published tools are re-published on the flag day (tools on test networks
are re-published anyway at mainnet).

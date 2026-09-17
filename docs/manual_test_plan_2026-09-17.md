# Manual end-to-end test plan (2026-09-17)

Walk the platform the way a new user would, in the order of the video script.
Start from an empty fleet. Every step has an expected result; anything else is
a bug. Report failures by section and step number.

Keep the daemon running in the terminal I own so I can read its log; tell me
when you need it restarted.

## 0. Clean slate

1. Remove every agent from the Agents page. Restart the daemon (ask me).
   Expected: the boot log shows no DecryptError line, and exactly one agent
   comes back: Kevin. (The seed stamp is present, so an empty fleet restores
   the guide agent once.)
2. Terminal banner reads `ATN Runtime  v0.7.5+src`.
3. Open autonet.computer in the browser with the daemon running. Expected: the
   UI appears on its own, no manual connect step, and the connection info
   shows endpoint `ws://localhost:7700` and node version 0.7.5.
4. Stop the daemon (ask me). Expected: the site shows its disconnected state
   with install instructions, and the Docs and Whitepaper pages still render.
   Start it again.

## 1. Install path (optional, 5 minutes)

1. In a fresh venv: `pip install autonet-computer` then `atn --version`.
   Expected: 0.7.5 installs from PyPI and prints its version.
2. Run `atn` from that venv with the tree daemon stopped. Expected: boots,
   listens on 7700, autonet.computer connects to it. Stop it and go back to
   the tree daemon for the rest of the plan.

## 2. AI Input

1. AI Input page with nothing configured. Expected: one card per known
   provider, each "Not configured"; no card that is blank or shows a wrong
   provider's error (Ollama text on a non-Ollama card).
2. Claude Max bridge: configure it. Expected: badge goes Active, Anthropic
   models appear in an agent's model picker.
3. An API-key provider (Anthropic key, OpenAI, DeepSeek, any you have): paste
   the key. Expected: Active, models listed. Paste junk: rejected at save with
   a readable error, card stays Not configured.
4. Ollama: with Ollama running, the card lists your local models; with Ollama
   stopped, Retry reports it cannot connect.
5. Custom provider (the "catch all"): add ID `my-llm`, name `My LLM`, an
   OpenAI-compatible base URL, key optional. Expected: card titled "My LLM"
   with model chips. Restart the daemon: the card is still there and can be
   deleted.
6. Bridge usage: after one agent turn on the bridge, leave the page and come
   back. Expected: Subscription Usage is populated without pressing Refresh.

## 3. Kevin and onboarding

1. With NO provider configured (temporarily remove them, or do this before
   section 2). Expected: Agents page shows the "No AI provider configured"
   banner and Kevin's send button is disabled.
2. With a provider: talk to Kevin. Expected: he debriefs you one theme at a
   time (background, skills, goals, constraints), no intake form, and the
   Profile page fills in as you go, with dated claims.
3. Ask Kevin for a plan toward one goal. Expected: a concrete plan, and he
   offers to set up agents for it.
4. Remove Kevin while other agents exist. Restart. Expected: he stays gone.

## 4. Building the fractal

1. Add Agent (top level): name, model, system prompt. Expected: card appears,
   chat works on the first message.
2. Add a child under it via the parent picker. Expected: the chart draws the
   tree; the child's config shows the parent.
3. Add a boss: create an agent with "set as parent of" over two existing top
   level agents. Expected: the new agent sits above both.
4. Reparent an agent by editing its parent. Expected: chart redraws; a second
   client (desktop app or another browser tab) redraws without reconnecting.
5. Remove a parent that has children. Expected: the dialog says how many
   children will be promoted; they become top level; survives a restart.
6. Advanced grants: in Add Agent, expand the advanced grants. Expected: the
   bundles (profile, toolsmith, publishing, and the rest) are listed with
   captions; tick one, create, reopen Config: still ticked.
7. Ask an agent in chat to create an agent using its own id as the new id.
   Expected: a clear refusal, chart still renders.

## 5. Running agents

1. Chat with an agent that has tools. Expected: tool calls are visible in
   the transcript, the answer follows, no stuck spinner.
2. Delegation: ask a parent to hand a task to its child. Expected: the child
   runs, the parent receives the result and reports it to you.
3. Wake on child: parent panel has the "Wake on child" switch. With it on, a
   child finishing on its own wakes the parent; with it off, it does not.
4. Wake by user and by parent: a message to an idle agent starts a run; a
   parent messaging a child starts the child's run.
5. Heartbeats: set 1 h + 30 m, Save. Expected: no validation error, panel
   re-reads 1h30m, card chip reads the schedule (not On-demand). Clear both
   fields: Save disabled. Set 5 m and watch one fire.
6. Disable mid-run. Expected: the run ends, the card stays disabled, no
   countdown restarts, no Run Now button on a disabled agent.
7. Kill: kill a running execution from the UI. Expected: it stops within a
   few seconds, agent returns to idle, daemon keeps running.
8. Credit budget: set a tiny budget, run until it trips. Expected: the card's
   dot goes amber with "budget reached". Raise the budget: active again
   without a restart.

## 6. Context and cost

1. Context view after two turns on an Anthropic-backed agent. Expected: cached
   input tokens > 0 (prompt caching).
2. Context view shows the model, the window size and the auto-compact
   threshold; a never-run agent shows no context line.
3. Cost chip: an API-key agent shows a non-zero cost after a turn; a bridge or
   Ollama agent shows none.
4. Compact from the chat header on an idle agent. Expected: compactions count
   +1, model and window unchanged, the agent still remembers the gist of the
   conversation on the next turn.
5. Sliding window: run a long conversation (or a low-window model) until
   trimming kicks in. Expected: the run keeps going, no provider error about
   context length.

## 7. Voice (voice extras installed)

1. Start voice mode, message the root agent. Expected: the reply is spoken;
   the focus rows in the mic popup name your agents (not "Orchestrator").
2. Push to talk: hold the PTT key (Page Down or Insert). Expected: a red
   Recording chip; release sends the transcript to the selected agent.
3. Select a different agent to listen to. Expected: only that agent is spoken.
4. "Everything" vs "responses only": with everything on, tool calls are
   narrated in a different voice from the agent's speech; untick it and only
   the tool tone plays.
5. During a long reply: Pause, Resume, Skip sentence, Mute speech all behave
   as labelled; the mic icon pulses while speaking and goes idle after.
6. On a machine without the extras, the mic button shows the "Voice support
   is not installed" message.

## 8. Desktop app (Windows)

1. Launch the desktop build, it connects to the daemon like the web does.
2. Transparency toggle works and persists across restart.
3. Pop out an agent window. Expected: live chat, messaging it does not open
   a duplicate window in the main app; switch the pop-out to Config and close
   it: the docked window lands on Config.
4. Connectors page: the Download Releases and pip install buttons open real
   pages.

## 9. Sponsor-dependent inference

Needs two daemons (or the same daemon acting as both, for the config half).

1. Sponsor side: Sponsor Inference panel, add a dependent identity and a
   token grant from one of your providers. Expected: the row reads Serving,
   config.yaml has `autonet.sponsor_inference: true`; Stop serving flips it.
2. Dependent side: agent Config > Dependent Inference > Sponsored, Save.
   Expected: no address box; the panel shows the sponsor or a discovery
   warning; the choice survives a reopen.
3. Run the dependent agent. Expected: the answer comes back through the
   sponsor, the sponsor's remaining-grant chip shows a real number that went
   down, and the sponsor's log shows the request.
4. Limits: ask for a huge max_tokens on a small grant. Expected: capped. A
   prompt bigger than the grant: refused with a readable reason.

## 10. Security: isolation and secrets

1. Security dialog: note the worker isolation default (it ships OFF; decide).
   Turn it on for the rest of this section and restart.
2. Run an API-provider agent. Expected: its detail header shows `pid NNNNN`
   matching `atn agents` in a terminal. Kill that PID from Task Manager.
   Expected: the run ends with an error, the daemon and the other agents
   survive.
3. Secrets page: add a secret with a name, value and authorized hosts.
   Expected: it appears masked, the value is never shown again.
4. Host scan. Expected: it runs and reports exposed secrets on the machine
   (or none), with paths.
5. Access log opens and is empty for the new secret.
6. Grant: give a top-level agent the secret in its Config allowance picker
   and set the root allowance in the Security dialog to include it. Run it
   on a task that needs the secret. Expected: the agent has `secret_*` tools,
   uses the secret as a temporary variable, the transcript never shows the
   value, and the access log shows a session minted.
7. Fractal propagation: a child's picker greys out secrets its parent lacks;
   a parent can extend its own allowance to a child.
8. Clamp: set the root allowance to a different secret. Expected: the Secrets
   page strikes through the agent's request with a "clamped" tooltip.
9. Revoke the secret from the agent. Expected: next run has no `secret_*`
   tools.
10. Alarm: have the agent print the secret value into the transcript, or call
    a host outside the authorized list. Expected: the call fails and an alarm
    appears on the Secrets page.
11. Not built: automatic key rotation and wallet transfer on alarm. Do not
    look for them.

## 11. Tools, single player

1. Tools page lists the built-in bundles and one ATN Harness card (not
   thirteen copies). Search "summarize a web page". Expected: ranked partial
   matches.
2. Ask an agent to build a tool for a task it has repeated. Expected: it
   registers one; the Tools page shows it under your local scope.
3. Grant that tool to a sibling agent. Expected: if the sibling lacks the
   unified tools bundle the dialog offers to grant it; the sibling can call
   the tool.
4. Dynamic expansion: with an MCP connector running, the agent sees a short
   list of tools; calling the connector's discovery tool exposes the
   connector's operations, named `mcp_<id>_<op>`, and using one works.
5. Gate: an agent without the publishing bundle asked to publish a tool is
   refused with "not granted"; add the bundle and it passes.
6. Tool window shows Calls and OK rate after use.

## 12. Web3 identity

1. Network page with MetaMask disconnected: Register on the root agent says
   to connect a wallet first.
2. Connect a wallet, register the root agent. Expected: the on-chain badge
   reads registered, the record block fills in, the address links resolve on
   Etherlink Shadownet to the current Substrate.
3. Sponsor panel "Use 0x..." writes `autonet.owner_wallet` into config.yaml.
4. Network page query of an unregistered address says registered: false; a
   broken RPC URL shows an RPC error, not a hang.

## 13. Tools, multiplayer (two daemons or the shadownet)

1. Publish the tool from section 11 on daemon A. Expected: within about 30 s
   daemon B logs indexed remote manifests and B's Tools page, Network scope,
   finds it by description; results reorder by relevance to the query.
2. Agent on B uses the tool via probe_tools. Expected: after the run, the
   harness adds one closing review turn naming that tool; an agent that used
   only a connector gets no review turn.
3. Review scores land: the tool's per-axis scores update on both daemons.
4. Epoch close: after one federated close, both daemons' world state dirs
   hold matching tool_positions, credibility and review book files, and the
   tool window shows "Mint (recent)".
5. Composition: publish tool B that imports tool A; use B. Expected: A's
   usage count also moves (attribution follows the import).

## 14. Services

1. Publish a service (what, backed by, unit of work, price). Expected: the
   card reads "N ATN per item" and, with a service registry configured, the
   snackbar says listed on the market and another daemon's Market tab shows
   it; without one, "Published on this daemon only".
2. Publish one backed by "This machine's model". Expected: inference-backed
   card with an enabled Purchase button.
3. Buy one unit of your own listing. Expected: the wallet prompt asks for
   exactly the ask price in ATN, you get a result plus a receipt, and after
   reload the card reads Requests 1, Success 100%.
4. Agent-side: ask an agent to find and request a service. Expected:
   find_services returns the listing, request_service pays one unit and
   returns the result.
5. Retire the service. Expected: market row goes inactive (or the "still
   listed" note without a registry).
6. Not served: point a listing at a backend that fails. Expected: the buyer
   loses at most one unit and the failure is recorded on the card.

## 15. Owner, docs, website

1. Owner page: Fleet earnings and Network mint total in ATN; fleet voice
   weights listed per household.
2. Docs page: Full index links open; content is current; no em dashes.
3. Whitepaper page opens scrolled past the video thumbnail; scrolling up with
   the mouse wheel reveals it (unverified by me, please check).
4. About, Privacy, Terms open.
5. README as a newcomer: the three claims the audit flagged as not backed by
   code are "work halts if the governance heartbeat goes silent", the
   alignment score being "computed and displayed", and the missing mention of
   the UI at autonet.computer. Decide the wording.

## 16. Known gaps (owner decisions, not bugs to report)

- Worker isolation ships OFF while the script says every agent is isolated.
- No published Windows build download.
- Sponsor path has no browsable audit trail and no semantic alignment check.
- No key rotation or wallet transfer hooks on alarm.
- Dependent identity is self-declared, not signed.
- The vault holds about 125 `agent-key.<test-id>` entries from old test
  runs; safe to delete once you confirm none is registered on a chain you
  care about.

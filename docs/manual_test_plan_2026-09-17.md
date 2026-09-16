# Manual test plan, prime-time pass (2026-09-17)

Walk the platform in the order of the video script. Every step has an expected
result. Steps tagged `F<n>` verify a fix from the 2026-09-16 audit pass; the
number is the audit finding index (see the pass notes in the session). Steps
without a tag check a script claim that the audit found solid but that only
you can confirm end to end.

## 0. Before you start

- [ ] Backend: `cd C:\code\autonet` and confirm `git log -1` shows the
      prime-time commit. Restart the daemon from this tree (`atn` in a terminal
      or `python -m atn`). The boot log must show NO `DecryptError` lines
      (your vault was torn by concurrent test writes yesterday and recovered;
      backups sit next to it as `vault.age.bak-2026-09-16-2153`).
- [ ] Frontend: rebuild atn_web (web and, if you want to test section 8, the
      Windows desktop build) from the committed tree.
- [ ] `python -c "import atn; print(atn.__version__)"` prints 0.7.4 (F105).
      The terminal banner says `ATN Runtime  v0.7.4+src` (F44).
- [ ] Open autonet.computer in the browser with the daemon running: the UI
      appears without any manual step (script: "as soon as it detects atn").

## 1. AI input (Config > AI Input)

- [ ] F13 / F14: no "World-Model Substrate" card saying "Ollama not detected",
      no empty "Marketplace Service" card. The real Ollama card still has a
      working Retry.
- [ ] F12: paste a DeepSeek key: badge goes Active, DeepSeek models appear in an
      agent's model picker. A junk key is rejected at save time.
- [ ] F11 / F15 / F16: add a custom provider (ID `my-llm`, display name
      `My LLM`, base URL of any OpenAI-compatible endpoint, key optional). The
      card title reads "My LLM" and lists model chips. Restart the daemon: the
      card is still there with its delete button.
- [ ] F18 / F119: with the Claude Max bridge, run one agent turn, leave and
      return to the AI Input tab: the Subscription Usage section is already
      populated (no Refresh click).
- [ ] Prompt caching claim: after two turns on an Anthropic-backed agent, the
      Context view's cache stat shows cached input tokens > 0.

## 2. Kevin and onboarding

- [ ] F29: with no provider configured, the Agents tab shows the red "No AI
      provider configured" banner above Kevin, and Kevin's composer send arrow
      stays disabled.
- [ ] Talk to Kevin: he debriefs you (strengths, weaknesses, goals) and the
      Profile page fills in.
- [ ] Remove Kevin, restart the daemon: he stays gone (fleet was non-empty) or
      returns once if he was your only agent (F28, by design). Note which you
      saw.

## 3. Fractal org chart

- [ ] Add Agent: create a child under any existing agent via the parent picker.
- [ ] Add Agent with "SET AS PARENT OF" over two existing top-level agents: the
      chart shows the new boss above them.
- [ ] F30 / F81 / F96: in Add Agent, expand "advanced grants": `profile`,
      `toolsmith`, `publishing` and the rest of the 9 advanced bundles are
      listed with captions. Tick `toolsmith`, create, reopen Config: still
      ticked.
- [ ] F33: create a parent with one child, remove the parent. The child is now
      top-level and survives a daemon restart. The remove dialog told you how
      many children would be promoted.
- [ ] F35: with the browser and the desktop app both connected, reparent in
      one; the other redraws without a reconnect.
- [ ] F32: ask an agent (via chat) to create an agent with its own id as the
      id; it gets a clear refusal and the chart keeps rendering.

## 4. Heartbeats and wake-ups

- [ ] F45: heartbeat editor, 1 h + 30 m, Save: no "Invalid schedule" error,
      the panel re-reads 1h30m.
- [ ] F46: clear both fields: Save is disabled.
- [ ] F49: the agent card shows a pink `heartbeat 5m` chip, not "On-demand".
- [ ] F47: with a heartbeat running, press Disable mid-run; after the run
      ends the card stays disabled and no countdown restarts.
- [ ] F54: a disabled agent has no Run Now button.
- [ ] F50: parent agent panel has a "Wake on child" switch; it round-trips.
- [ ] Child finishes a delegated task: parent wakes (with the switch on) and
      you can see the child's result via the parent.
- [ ] F117: set a tiny credit budget, run until it trips: the card's dot goes
      amber with "budget reached". F48: raise the budget: status returns to
      active without a restart.

## 5. Context and cost

- [ ] F37: Context view on a 200k Claude model shows "auto-compact 92%"; on an
      ollama model about 50%; a never-run agent shows no line.
- [ ] F38: an API-key agent (not the bridge) shows a non-zero cost chip after a
      turn; an ollama agent shows none.
- [ ] F39: Compact an idle agent from the chat header: Context view keeps the
      model and window, compactions count +1.

## 6. Voice (needs the voice extras installed)

- [ ] F59: on a machine without the extras, the mic button shows the "Voice
      support is not installed" snackbar.
- [ ] F55: start voice, message the root agent: the reply is spoken; the mic
      popup focus rows name the root agent (not "Orchestrator").
- [ ] F60: root agent chat has a PTT mic icon; holding Page Down / Insert turns
      it into a red Recording chip; releasing sends the transcript.
- [ ] F57: untick "Narrate tool calls": tool tone still plays, no narration.
- [ ] F61: during a long reply: Pause, Resume, Skip sentence, Mute speech all
      behave as labelled.
- [ ] F62: the mic icon pulses while speaking and returns to idle after.
- [ ] Two-voice claim: agent speech and tool narration use different voices
      (F58 if you set `voice.tools_voice` in config.yaml).

## 7. Desktop build (Windows app)

- [ ] Transparency toggle works and persists across restart.
- [ ] Pop out an agent window: it shows live chat (F0 also with a remote
      daemon URL in the header).
- [ ] F3: message the popped-out agent: no duplicate in-app window appears.
- [ ] F4: switch the pop-out to Config, close it: the docked window lands on
      Config.
- [ ] F2: Connectors page: "Download Releases" and "pip install" buttons open
      real pages, not 404s.

## 8. Sponsor-dependent inference

- [ ] F22 / F114: agent Config > Dependent Inference > "Sponsored" + Save: no
      address box; the panel shows the daemon-level sponsor or the discovery
      warning; the choice survives a reopen.
- [ ] F23: Sponsor Inference panel: add a dependent; the row says "Serving",
      config.yaml has `autonet.sponsor_inference: true`; "Stop serving" flips
      it to false.
- [ ] F24: after one answered request, the remaining-grant chip renders with a
      real number.
- [ ] F27 (two daemons): dependent asks for 64k max_tokens on a 500-token
      grant: sponsor caps it; a prompt bigger than the grant is refused.

## 9. Isolation and secrets

- [ ] Security dialog: worker isolation is OFF by default. Decide whether to
      ship it on (owner decision, see the pass notes). Turn it on for the
      next steps.
- [ ] F8: run an API-provider agent; its detail header shows `pid NNNNN`
      matching `atn agents` in the terminal. Kill that PID: the run ends, the
      daemon survives.
- [ ] Secrets tab: add a secret, run the host scan, open the access log.
- [ ] F63 / F7: create a top-level agent, give it a secret in its Config
      allowance picker, set the root allowance in the Security dialog to that
      secret, trigger it twice: the log shows a session minted both times and
      the agent has `secret_*` tools.
- [ ] F64 / F68: root allowance set to a different secret: the Secrets tab
      strikes through the agent's request with a "clamped" tooltip; a child's
      picker greys out secrets its parent lacks.
- [ ] Revoke the secret from the agent's Config: the next run has no
      `secret_*` tools.
- [ ] F66: a tool that calls a host outside a secret's authorized hosts fails
      AND raises an alarm in the Secrets tab.
- [ ] Not built, do not look for it: automatic key rotation and wallet
      transfer on alarm (script claim; owner decision).

## 10. Tools, single player

- [ ] F122: Tools page search "summarize a web page" returns ranked partial
      matches (was empty).
- [ ] F79: an agent with only `unified_tools` calling `publish_tool` via
      use_tool is refused with "not granted"; add `publishing` and it passes
      the gate.
- [ ] F80: with an MCP connector running, `list_tools(category='connector')`
      names are `mcp_<id>_<op>` and `use_tool` on them works.
- [ ] F82: Grant a tool to an agent lacking `unified_tools`: the dialog offers
      to grant the bundle.
- [ ] F84: with `autonet.owner_wallet` set, the Tools page still shows exactly
      one ATN Harness card.
- [ ] Ask an agent to build a tool for a repeated task: it registers one; a
      sibling agent granted that tool can call it.

## 11. Tools, multiplayer (needs a peer daemon or the shadownet)

- [ ] F98: Tools page, Network scope: a task description reorders results by
      relevance; digest-only tools show real names.
- [ ] F97: an agent that used only a connector gets no closing review turn;
      one that used a registered network tool gets the review prompt naming
      that tool.
- [ ] F93: publish a tool on daemon A; within ~30 s daemon B logs "indexed N
      remote tool manifest(s)" and B's probe finds it.
- [ ] F95 / F94: after one federated close, the world state dir has
      tool_positions.json, tool_credibility.json, tool_review_book.json, and
      local_tool_positions.json matches the driver's file.
- [ ] F100: a tool window shows Calls and OK rate, plus "Mint (recent)"; no
      "Fees earned 0.00".

## 12. Services

- [ ] F76 / F75: Publish a service backed by "This machine's model": the card
      shows inference-backed and its Purchase button is enabled.
- [ ] F74: publish with ask price `1`: the card reads "1 ATN per item" and the
      wallet prompt asks for 1 ATN, not 1 wei. Older listings show the
      legacy-scale note.
- [ ] F70 / F115: with `service_registry_address` configured the snackbar
      says "Published and listed on the market" and another daemon's Market
      tab lists it; without it, "Published on this daemon only".
- [ ] F71 / F116: buy one unit of your own listing: you get a result plus a
      receipt, and after reload the card reads Requests 1, Success 100%.
- [ ] F72: retire it: the market row goes inactive (or the "still listed"
      snackbar appears without a registry).
- [ ] F77 (two machines): with no chain config, a remote service_request is
      refused with "no chain configuration"; a local buy still works.

## 13. Web3 identity and earnings

- [ ] F90: with MetaMask disconnected, Register says "Connect a wallet to fund
      the agent address, then register."
- [ ] F85 / F89: connect a wallet, register the root agent: the on-chain tab
      badge reads registered and the record block fills in.
- [ ] F86: the Network tab address links resolve on Etherlink Shadownet to the
      current Substrate (0x4C4dAEd1...).
- [ ] F87: sponsor panel "Use 0x..." writes `autonet.owner_wallet` into
      config.yaml.
- [ ] F102: Owner page tiles read "Fleet earnings" and "Network mint total"
      in ATN, not "Reputation".
- [ ] F126: Network page query of an unregistered address says
      registered:false; with a broken RPC URL it shows an RPC error.

## 14. Docs and website

- [ ] Whitepaper page opens scrolled past the YouTube thumbnail; scrolling up
      reveals it. (Not verified by me: the browser probe was cut off, and it
      reported that mouse-wheel scrolling did nothing on Whitepaper and
      Secrets; please check with a real mouse.)
- [ ] F103: with the daemon stopped, the Docs tab still renders the bundled
      paper, current content, no em dashes.
- [ ] F106 / F109 / F112: Docs "Full index" links open; the Create Agent
      dialog's Docs link shows the 2026-09 banner; the Add Provider Docs link
      lists Marketplace Service.
- [ ] Read README.md once as a newcomer: the three claims the audit flagged as
      not backed by code are "work halts if the governance heartbeat goes
      silent", the alignment score being "computed and displayed", and no
      mention of the UI / autonet.computer / port 7700. Decide the wording.

## 15. Not covered by this pass (owner decisions)

- Worker isolation ships OFF by default while the script says every agent is
  process-isolated.
- No published Windows build or download link exists.
- Sponsor path has no audit trail the sponsor can browse and no semantic
  alignment check (script claims both).
- Key rotation / wallet transfer hooks on alarm do not exist.
- The README's 1:1 DAO-side REP claim has no daemon or UI implementation.
- Dependent identity is self-declared (no signature); the doc was corrected
  to say so.
- The vault holds about 125 `agent-key.<test-id>` entries from months of tests
  writing to the real keystore. Safe to delete the ones with no matching
  `~/.atn/agents/<id>` directory, once you confirm none is registered on a
  chain you care about.

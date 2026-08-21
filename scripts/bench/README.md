# Benchmark rig (SWE-bench Verified, standalone daemon)

**THE GOAL (do not drift):** one number that places the ATN FRAMEWORK in
the public comparison — same model, same 500 tasks, official grader as
published entries of other frameworks. The reference must be CURRENT
capabilities (entries ≤ ~6 months old), not historic tables. As of
2026-08: the live seat is **Claude 4.5 Opus** — live-SWE-agent 79.2
(2025-12), Sonar Foundation 79.2 (2025-12), mini-SWE-agent 76.8
(2026-02). Model runs via the Claude Max bridge (user decision: NO
Anthropic API key), so throughput paces in Max windows and the run is
multi-day resumable. A run with no current published reference config is
worthless — pin the reference FIRST, then run.

`swe_driver.py` runs SWE-bench Verified instances against a standalone
ATN daemon (`autonet.enabled: false`, unregistered — zero network tax):
one fresh agent tree per task over loopback WS, git diff as the
prediction, exact per-tree token ledger (prompt + completion, every
agent), full execution records as transcripts. Grading is a separate,
off-pod step.

Status 2026-08-05: 10-task smoke COMPLETE on RunPod 4090 +
Qwen3-Coder-30B-A3B-AWQ (vLLM). 10/10 executed, all patches non-empty,
mean 59s wall, 3.79M in / 66k out tokens total, ~1.2¢ GPU per attempt.

## RUN 11/12 (2026-08-16): fixes validated; resolve rate is model-limited

Three-run arc, same 10 astropy instances, same model
(Qwen3-Coder-30B-A3B-AWQ @ 28k ctx), graded with the official harness
(swebench 4.1.0, WSL Docker):

| run | change | overflow crashes | resolved |
|-----|--------|------------------|----------|
| 10  | baseline | 5/10 | 0/10 |
| 11  | prompt: verify-before-DONE, delegate-early; driver: abort detection + py_compile | 6/10 | 1/10 (13236) |
| 12  | framework: tier-3 hard context reset in `atn/providers/base.py` (prune → compact → trim → RESET → abort) | **0/10** | 1/10 (14309) |

Conclusions: (a) the loop is now crash-free on small-context models —
the hard reset (rebuild stack from task + tool-call digest, 2/run cap)
eliminated every overflow abort and is in the repo with tests
(`tests/atn/test_loop_hardening.py`); (b) ~10% resolve on this slice is
the MODEL's ceiling (30B 4-bit, 28k ctx), with run-to-run variance in
which instance lands — N seeds required for any citable number;
(c) next lever is the model seat: Nemotron 3.5 Lightning 30B-A3B
(hybrid Mamba, huge ctx in 24GB, self-reported 52.8% SWE-bench V) —
needs GGUF or non-NVFP4 quant on Ada + vLLM ≥0.27.

Ops notes for the next resurrection: custom provider registration does
NOT survive pod recreation (persists in container home, not the volume)
— run `/workspace/add_provider.py` after the daemon is up, or the first
create_agent fails with "unknown model". Driver teardown misses
grandchildren: wipe `/workspace/bench/atn-agents` between runs.

## GRADED 2026-08-16: run10 0/10 resolved (official harness, swebench 4.1.0, WSL)

All 10 patches applied cleanly; none resolved. Reports:
`results/run10/grading_report.json` + `grading_logs/<iid>.report.json`.

Failure taxonomy (from transcripts + test output — the real findings):

- **5/10 context-overflow aborts** ("Aborted: context overflow could not
  be reduced (max compactions reached)") at turns 14–46 of 60. The 28672
  vLLM context cannot hold SWE-bench exploration under the current
  compaction ladder. Correlation that matters: ALL five aborts were
  tree_size 1–2 (solo agents); the three trees with 4–6 children never
  overflowed. Delegation is the context-relief valve — but the loop
  aborts instead of delegating when full.
- **5/10 claimed DONE without verification.** None ran pytest; 3 of them
  (12907, 13236, 14309) shipped code with SyntaxError/IndentationError
  while claiming success. The daemon reports `execution.completed` for
  overflow-aborts too, so the driver can't tell success from abort
  without parsing `output.result`.
- **Near-miss:** 13977 (6-agent tree) got 12/20 FAIL_TO_PASS, 318/322
  PASS_TO_PASS — a genuine partial fix, and the most delegation-heavy run.

Fixes before the full slice: (1) overflow → auto-delegate instead of
abort (framework); (2) task prompt must REQUIRE running the repro/tests
before DONE; (3) driver should mark `output.result` aborts as failed;
(4) consider Devstral/larger ctx or a 48GB pod for a second seat.

## Pod resurrection (everything durable is on the volume)

Create pod: RTX 4090, SECURE cloud, network volume `r76qgts96s`
(EU-RO-1) mounted at /workspace, **`allowedCudaVersions: ["13.0","12.9"]`**
(torch in the venv is cu130; 12.8 hosts fail). Then:
`bash /workspace/boot.sh` (apt deps + ~/.atn/config.yaml), 
`bash /workspace/restart_llm.sh` (kills GPU orphans incl. the renamed
`VLLM::EngineCore`, serves model), `tmux new -d -s atn
"/workspace/atn-env/bin/atn"` (tmux, NOT nohup — the CLI exits on stdin
EOF), then drive with `/workspace/swe_driver.py`.

## Dogfood findings from the smoke (fix in framework)

- `pip install autonet-computer` fails on clean Ubuntu without `libgmp-dev` (fastecdsa build).
- Keyless `openai_compat` providers are skipped ("no API key") — hostile to local vLLM/ollama-style endpoints; dummy key works around.
- `atn` CLI exits on stdin EOF — needs a `--headless` flag.
- `openai_compat` never requests `stream_options.include_usage` → token accounting reads zero against vLLM/OpenAI streaming. Server-side workaround: `--enable-force-include-usage` (vLLM).
- `create_agent` auto-starts an execution; a follow-up `trigger_run` hits "At concurrency limit".
- Execution records are dropped on `remove_agent` — export ledgers/transcripts BEFORE teardown.
- `get_history` carries execution metadata only; full transcript = `get_execution` per execution id, pre-teardown.

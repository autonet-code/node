# Benchmark rig (SWE-bench Verified, standalone daemon)

`swe_driver.py` runs SWE-bench Verified instances against a standalone
ATN daemon (`autonet.enabled: false`, unregistered — zero network tax):
one fresh agent tree per task over loopback WS, git diff as the
prediction, exact per-tree token ledger (prompt + completion, every
agent), full execution records as transcripts. Grading is a separate,
off-pod step.

Status 2026-08-05: 10-task smoke COMPLETE on RunPod 4090 +
Qwen3-Coder-30B-A3B-AWQ (vLLM). 10/10 executed, all patches non-empty,
mean 59s wall, 3.79M in / 66k out tokens total, ~1.2¢ GPU per attempt.
**UNGRADED** — resolve rate unknown. Artifacts in
`results/run10/` (local, uncommitted until graded).

## NEXT SESSION: grade run10

Predictions file: `scripts/bench/results/run10/predictions.jsonl`
(`model_name_or_path` = `atn+cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit`).

Route A — hosted (fast, needs one-time signup):
`pip install sb-cli`, get token per https://www.swebench.com/sb-cli/,
`sb-cli submit swe-bench_verified test --predictions_path predictions.jsonl`.

Route B — local WSL (no signup, Docker pulls ~10-20GB):
`pip install swebench`, then
`python -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Verified --predictions_path predictions.jsonl --instance_ids <10 ids from ledger> --run_id atn-run10`.

Caution flag for interpretation: 59s mean wall is fast — check
transcripts for premature DONE before celebrating any resolve number.

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

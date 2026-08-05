#!/usr/bin/env python3
"""SWE-bench driver for the ATN daemon (headless, loopback WS).

Runs one fresh agent tree per task against a standalone daemon
(autonet.enabled=false), collects the git diff as the prediction, and
writes an exact per-tree token ledger (prompt + completion, every agent
in the tree) — the benchmark's primary cost metric.

Usage (on the pod):
    python swe_driver.py --limit 10 \
        --provider vllm --model cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit \
        --out /workspace/bench/run1

Grading happens elsewhere (sb-cli or a Docker host) from predictions.jsonl.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path

import websockets

DATASET_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=princeton-nlp%2FSWE-bench_Verified&config=default&split=test"
    "&offset={offset}&length={length}"
)

TASK_PROMPT = """You are a software engineer fixing a real reported issue.

Repository checkout: {workdir}
Work ONLY inside that directory. The repo is at the exact commit where the
issue was reported. Your fix will be evaluated by the repo's own test suite.

<issue>
{problem}
</issue>

Rules:
- Modify source code only. Do NOT modify tests or add new test files.
- Make the minimal change that resolves the issue.
- You may delegate subtasks to child agents when that is more efficient.
- When the fix is complete and coherent, state DONE and stop.
"""


class DaemonClient:
    """Persistent WS connection, msg_id demux + terminal-event queue."""

    def __init__(self, url: str):
        self.url = url
        self.ws = None
        self.pending: dict[str, asyncio.Future] = {}
        self.events: asyncio.Queue = asyncio.Queue()
        self._reader_task = None

    async def connect(self):
        self.ws = await websockets.connect(self.url, max_size=None, open_timeout=10,
                                           ping_interval=30, ping_timeout=120)
        self._reader_task = asyncio.create_task(self._reader())

    async def _reader(self):
        async for frame in self.ws:
            try:
                msg = json.loads(frame)
            except Exception:
                continue
            if not isinstance(msg, dict):
                continue
            mid = msg.get("msg_id")
            if mid and mid in self.pending:
                self.pending.pop(mid).set_result(msg)
            elif msg.get("type") == "event" and msg.get("event_type", "").startswith("execution."):
                await self.events.put(msg)
            # else: unsolicited snapshot / other events — drop

    async def call(self, msg_type: str, timeout: float = 120.0, **args):
        mid = uuid.uuid4().hex
        fut = asyncio.get_event_loop().create_future()
        self.pending[mid] = fut
        payload = json.dumps({"type": msg_type, "msg_id": mid, **args})
        try:
            await self.ws.send(payload)
        except websockets.ConnectionClosed:
            if self._reader_task:
                self._reader_task.cancel()
            await self.connect()  # reconnect once; loopback daemon needs no re-auth
            await self.ws.send(payload)
        msg = await asyncio.wait_for(fut, timeout)
        if not msg.get("ok"):
            raise RuntimeError(f"{msg_type}: {msg.get('error')}")
        return msg.get("result")

    async def wait_terminal(self, execution_id: str, timeout: float):
        """Wait for execution.completed/failed/killed for execution_id."""
        deadline = time.monotonic() + timeout
        while True:
            remain = deadline - time.monotonic()
            if remain <= 0:
                raise asyncio.TimeoutError()
            evt = await asyncio.wait_for(self.events.get(), remain)
            data = evt.get("data", {})
            if (data.get("execution_id") == execution_id
                    and evt.get("event_type") in
                    ("execution.completed", "execution.failed", "execution.killed")):
                return evt.get("event_type"), data

    async def close(self):
        if self._reader_task:
            self._reader_task.cancel()
        if self.ws:
            await self.ws.close()


def fetch_tasks(limit: int, offset: int, cache: Path) -> list[dict]:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        rows = [json.loads(l) for l in cache.read_text().splitlines() if l.strip()]
    else:
        rows = []
        got, off = 0, offset
        while got < limit:
            n = min(100, limit - got)
            with urllib.request.urlopen(DATASET_ROWS_URL.format(offset=off, length=n)) as r:
                batch = json.load(r)["rows"]
            if not batch:
                break
            rows.extend(row["row"] for row in batch)
            got += len(batch)
            off += len(batch)
        cache.write_text("\n".join(json.dumps(r) for r in rows))
    return rows[:limit]


def checkout(task: dict, work_root: Path, repo_cache: Path) -> Path:
    """Bare-clone cache per repo; fresh worktree per instance."""
    repo = task["repo"]  # e.g. "django/django"
    commit = task["base_commit"]
    bare = repo_cache / (repo.replace("/", "__") + ".git")
    if not bare.exists():
        subprocess.run(
            ["git", "clone", "--bare", f"https://github.com/{repo}.git", str(bare)],
            check=True, capture_output=True,
        )
    workdir = work_root / task["instance_id"]
    if workdir.exists():
        subprocess.run(["rm", "-rf", str(workdir)], check=True)
    subprocess.run(
        ["git", "clone", "--shared", str(bare), str(workdir)],
        check=True, capture_output=True,
    )
    subprocess.run(["git", "-C", str(workdir), "checkout", "-q", commit],
                   check=True, capture_output=True)
    return workdir


def collect_patch(workdir: Path) -> str:
    subprocess.run(["git", "-C", str(workdir), "add", "-N", "."],
                   check=True, capture_output=True)
    out = subprocess.run(
        ["git", "-C", str(workdir), "diff", "--", ".", ":(exclude)tests/", ":(exclude)*/tests/*"],
        check=True, capture_output=True, text=True,
    )
    return out.stdout


async def tree_ledger(client: DaemonClient, root_id: str) -> tuple[dict, dict]:
    """Sum token_usage over every agent in root's delegate tree.

    Returns (ledger, records) — records is {agent_id: [full get_execution
    results]} for transcript export. MUST run before agent teardown: the
    daemon drops execution records with the agent.
    """
    ids = {root_id}
    try:
        tree = await client.call("get_delegates")
        nodes = tree if isinstance(tree, list) else (
            tree.get("nodes") or tree.get("tree") or tree.get("delegates") or [])
        def flatten(ns):
            for n in ns:
                if not isinstance(n, dict):
                    continue
                nid = n.get("agent_id") or n.get("id")
                parent = n.get("parent_id") or n.get("parent")
                if nid and parent in ids:
                    ids.add(nid)
                kids = n.get("children") or []
                if nid in ids and kids:
                    flatten(kids)
        for _ in range(6):  # tolerate flat lists in any order
            flatten(nodes if isinstance(nodes, list) else [])
    except RuntimeError:
        pass
    per_agent, records = {}, {}
    totals = {"input_tokens": 0, "output_tokens": 0, "total": 0,
              "cache_read_tokens": 0, "cache_creation_tokens": 0}
    for aid in ids:
        try:
            hist = await client.call("get_history", agent_id=aid)
        except RuntimeError:
            continue
        for exmeta in (hist or {}).get("executions", []) or []:
            eid = exmeta.get("execution_id")
            if not eid:
                continue
            try:
                ex = await client.call("get_execution", execution_id=eid)
            except RuntimeError:
                ex = exmeta
            records.setdefault(aid, []).append(ex)
            for prov, u in (ex.get("token_usage") or {}).items():
                pa = per_agent.setdefault(aid, {})
                for k in totals:
                    v = int(u.get(k, 0) or 0)
                    pa[k] = pa.get(k, 0) + v
                    totals[k] += v
    return {"totals": totals, "per_agent": per_agent, "tree_size": len(ids)}, records


async def run_task(client: DaemonClient, task: dict, args, workdir: Path) -> dict:
    iid = task["instance_id"]
    agent_id = f"bench-{iid}"[:64]
    prompt = TASK_PROMPT.format(workdir=workdir, problem=task["problem_statement"])
    started = time.monotonic()
    res = await client.call(
        "create_agent",
        id=agent_id, name=f"bench {iid}", mode="cognitive",
        prompt=prompt,
        agent_type="general", provider=args.provider, model=args.model,
        max_turns=args.max_turns,
        tools=["shell", "delegation", "observation", "messaging"],
    )
    # create_agent auto-starts the first execution — do NOT trigger_run again
    eid = res.get("execution_id")
    if not eid:
        raise RuntimeError(f"create_agent returned no execution_id: {res}")
    try:
        status, _ = await client.wait_terminal(eid, args.task_timeout)
    except asyncio.TimeoutError:
        await client.call("kill_execution", execution_id=eid)
        status = "execution.timeout"
    wall = time.monotonic() - started
    ledger, records = await tree_ledger(client, agent_id)
    patch = await asyncio.to_thread(collect_patch, workdir)
    # export full execution records BEFORE teardown deletes them
    tdir = Path(args.out) / "transcripts"
    tdir.mkdir(parents=True, exist_ok=True)
    out_records = dict(records)
    try:
        out_records["_output"] = await client.call("get_output", agent_id=agent_id)
    except RuntimeError as e:
        out_records["_output"] = {"error": str(e)}
    (tdir / f"{iid}.json").write_text(json.dumps(out_records, indent=1))
    # teardown: children first (delegate tree), then root
    for aid in list(ledger["per_agent"]):
        if aid != agent_id:
            try:
                await client.call("remove_agent", agent_id=aid)
            except RuntimeError:
                pass
    try:
        await client.call("kill_agent", agent_id=agent_id)
        await client.call("remove_agent", agent_id=agent_id)
    except RuntimeError:
        pass
    return {"instance_id": iid, "status": status, "wall_s": round(wall, 1),
            "patch": patch, "ledger": ledger}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", default="ws://localhost:7700")
    ap.add_argument("--provider", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--task-timeout", type=float, default=1800.0)
    ap.add_argument("--work-root", default="/workspace/bench/work")
    ap.add_argument("--repo-cache", default="/workspace/bench/repos")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    preds_path = out / "predictions.jsonl"
    ledger_path = out / "ledger.jsonl"
    done = set()
    if preds_path.exists():
        done = {json.loads(l)["instance_id"] for l in preds_path.read_text().splitlines() if l.strip()}

    tasks = fetch_tasks(args.limit, args.offset, out / "tasks.jsonl")
    client = DaemonClient(args.ws)
    await client.connect()
    try:
        for i, task in enumerate(tasks):
            iid = task["instance_id"]
            if iid in done:
                continue
            print(f"[{i+1}/{len(tasks)}] {iid}", flush=True)
            workdir = await asyncio.to_thread(
                checkout, task, Path(args.work_root), Path(args.repo_cache))
            try:
                r = await run_task(client, task, args, workdir)
            except Exception as e:
                print(f"  ERROR {iid}: {e}", flush=True)
                r = {"instance_id": iid, "status": "driver.error", "error": str(e),
                     "wall_s": 0, "patch": "", "ledger": {}}
            with preds_path.open("a") as f:
                f.write(json.dumps({"instance_id": iid,
                                    "model_name_or_path": f"atn+{args.model}",
                                    "model_patch": r["patch"]}) + "\n")
            with ledger_path.open("a") as f:
                row = {k: v for k, v in r.items() if k != "patch"}
                f.write(json.dumps(row) + "\n")
            t = r.get("ledger", {}).get("totals", {})
            print(f"  {r['status']}  wall={r['wall_s']}s  "
                  f"tokens={t.get('total', 0)} (in={t.get('input_tokens', 0)} "
                  f"out={t.get('output_tokens', 0)})  patch={len(r['patch'])}B", flush=True)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())

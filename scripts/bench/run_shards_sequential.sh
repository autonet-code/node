#!/bin/bash
# Run the four run14 shards SEQUENTIALLY against ONE daemon (ws://localhost:7700).
# Each shard is re-invoked until every instance has a good prediction (the
# driver skips done instances and its backstop leaves exhaustion-era instances
# un-done), with a stall guard so a hard failure can't loop forever.
cd /c/code/autonet || exit 1
# Singleton lock: a second wrapper must be impossible no matter how the
# launcher misbehaves (2026-08-18: retries on a false failure signal started
# five of these against one daemon).
LOCK=/tmp/run14_wrapper.lock
if [ -f "$LOCK" ]; then
  oldpid=$(cat "$LOCK" 2>/dev/null)
  if [ -n "$oldpid" ] && kill -0 "$oldpid" 2>/dev/null; then
    echo "wrapper already running (pid $oldpid); refusing to start"
    exit 1
  fi
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT
# Completion = RECORDED attempts (empty-patch attempts count: pass@1 means
# one attempt, and a real attempt that produced no patch is a miss, not a
# do-over). Only zero-token non-attempts are left unrecorded to re-run.
count_good() {
  python -c "
import json,sys
try:
    rows=[json.loads(l) for l in open('scripts/bench/results/run14$1/predictions.jsonl',encoding='utf-8') if l.strip()]
    print(len({r['instance_id'] for r in rows}))
except FileNotFoundError:
    print(0)"
}
declare -A OFFSETS=( [a]=0 [b]=125 [c]=250 [d]=375 )
for s in a b c d; do
  for pass in 1 2 3 4 5 6; do
    n=$(count_good $s)
    if [ "$n" -ge 125 ]; then break; fi
    echo "=== shard $s pass $pass ($n/125 good) ==="
    python -u scripts/bench/swe_driver.py \
      --ws ws://localhost:7700 --limit 125 --offset "${OFFSETS[$s]}" \
      --provider claude_max --model claude-opus-4-5 \
      --out "scripts/bench/results/run14$s" \
      --work-root "C:/bench_work" \
      --repo-cache "C:/Users/astmo/AppData/Local/Temp/claude/bench_repos" \
      --usage-threshold 82
    n2=$(count_good $s)
    if [ "$n2" -le "$n" ]; then
      echo "=== shard $s made no progress on pass $pass (still $n2); cooling 1800s ==="
      sleep 1800
    fi
  done
  echo "=== shard $s finished with $(count_good $s)/125 good ==="
done
echo "=== ALL SHARDS DONE ==="

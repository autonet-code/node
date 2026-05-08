#!/bin/bash
# Args: $1 = bootstrap multiaddr
cd ~/autonet-test
pkill -f cross_machine_smoke 2>/dev/null
sleep 1
rm -rf /tmp/sub_ec2 /tmp/ec2_smoke.log
nohup env \
  SUBSTRATE_DATA=/tmp/sub_ec2 \
  SUBSTRATE_LISTEN_PORT=4002 \
  SUBSTRATE_RPB=cross-smoke \
  SUBSTRATE_OBS_INTERVAL=3 \
  SUBSTRATE_CLOSE_INTERVAL=15 \
  SUBSTRATE_DURATION=55 \
  SUBSTRATE_OBS_AXIS=2 \
  SUBSTRATE_BOOTSTRAP="$1" \
  ./venv/bin/python cross_machine_smoke.py \
  > /tmp/ec2_smoke.log 2>&1 < /dev/null &
disown
echo "launched pid=$!"

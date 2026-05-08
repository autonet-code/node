#!/bin/bash
set -e
cd ~/autonet-test
rm -rf /tmp/sub_ec2 /tmp/ec2_smoke.log /tmp/ec2_fifo
mkfifo /tmp/ec2_fifo
nohup bash -c "(cat /tmp/ec2_fifo) | SUBSTRATE_DATA=/tmp/sub_ec2 SUBSTRATE_LISTEN_PORT=4002 SUBSTRATE_RPB=cross-smoke SUBSTRATE_BOOTSTRAP=$1 ./venv/bin/python cross_machine_smoke.py" > /tmp/ec2_smoke.log 2>&1 < /dev/null &
disown
echo "launched, pid=$!"

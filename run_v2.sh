#!/bin/bash
# BAH 2026 v2 Pipeline — runs in background, logs to logs/pipeline_v2_*.log
set -e
cd "$(dirname "$0")"
mkdir -p logs

ARGS="${@:---skip-nowcast}"
echo "Launching v2 pipeline at $(date) with args: $ARGS"
nohup .venv/bin/python -m bah2026.scripts.run_full_pipeline $ARGS > /dev/null 2>&1 &
PID=$!
echo "PID: $PID"
echo "$PID" > /tmp/bah2026_pid.txt
echo "Monitor: tail -f logs/pipeline_v2_*.log"
#!/usr/bin/env bash
# Wait for E30's training process (by PID) to exit, then run E29 (run_e29_e30.sh e29).
set -u
cd "$(dirname "$0")"
PID=$1
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
./run_e29_e30.sh e29

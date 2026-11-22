#!/bin/bash
# Round-1 review experiments (CPU pool): Step 4b (M1, m3), Step 5b ci (M6), Step 5b mat (M5).
set -e
cd "$(dirname "$0")"
mkdir -p artifacts/logs
run() { echo "== $* ($(date +%H:%M))"; python3 -u "$@" > "artifacts/logs/$(basename "$1" .py)${2:+_$2}.log" 2>&1; }
run step4b_mle_far.py
run step5b_detect_more.py ci
run step5b_detect_more.py mat
echo "ALL DONE ($(date +%H:%M))"

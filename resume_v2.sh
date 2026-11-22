#!/bin/bash
# Rerun Step 5 (v2, global nuisance grid), 9, 10 after the project move broke the 23:58 run.
set -e
cd "$(dirname "$0")"
mkdir -p artifacts/logs
run() { echo "== $* ($(date +%H:%M))"; python3 -u "$@" > "artifacts/logs/$(basename "$1" .py)${2:+_$2}.log" 2>&1; }
run step5_detect.py
run step9_material.py
run step10_learned.py
echo "ALL DONE ($(date +%H:%M))"

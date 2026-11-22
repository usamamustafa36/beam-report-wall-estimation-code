#!/bin/bash
# Resume run_all.sh from Step 5 (Steps 1-4 and 6a already done on the corrected pools).
set -e
cd "$(dirname "$0")"
mkdir -p artifacts/logs
run() { echo "== $* ($(date +%H:%M))"; python3 -u "$@" > "artifacts/logs/$(basename "$1" .py)${2:+_$2}.log" 2>&1; }
run step5_detect.py
run step6_mismatch.py b
run step9_material.py
run step10_learned.py
echo "ALL DONE ($(date +%H:%M))"

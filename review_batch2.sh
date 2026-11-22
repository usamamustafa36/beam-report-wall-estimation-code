#!/bin/bash
# Round-1 review runs left after the 16:48 reboot: Step 10 (fixed), Step 13 policy, facade E (4b, 5b ci).
cd "$(dirname "$0")"
mkdir -p artifacts/logs
run() { echo "== $* ($(date +%H:%M))"; python3 -u "$@" > "artifacts/logs/$(basename "$1" .py)${2:+_$2}${3:+_$3}.log" 2>&1 || echo "FAILED $*"; }
run step10_learned.py quick
grep -q '"cls": "NLOS"' artifacts/logs/step10_learned_quick.log && run step10_learned.py || echo "step10 quick failed, skipping full"
run step13_policy.py
run step4b_mle_far.py E
run step5b_detect_more.py ci E
echo "ALL DONE ($(date +%H:%M))"

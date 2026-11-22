#!/bin/bash
# Resumes rerun_whitenfix.sh after the 2026-10-30 08:54 reboot (runs 1-2 finished; starts at run 3; short runs first).
cd "$(dirname "$0")"
mkdir -p artifacts/logs
run() { echo "== $* ($(date '+%d %H:%M'))"; python3 -u "$@" > "artifacts/logs/$(basename "$1" .py)${2:+_$2}${3:+_$3}.log" 2>&1 || echo "FAILED $*"; }
run step6_mismatch.py b          # pseudo-true bias of the physics model
run step9_material.py            # material classification
run step10_learned.py            # learned estimator + hybrid
run step4b_mle_far.py E          # facade E
run step13_policy.py             # test-then-fuse policy
run step5_detect.py              # N = 5, 20 curves and full-Sionna false alarms (Fig. 2)
run step11_moved_detect.py       # re-traced moved facade (C1)
run step5b_detect_more.py mat    # unknown material (M5)
run step5b_detect_more.py ci E
echo "ALL DONE ($(date '+%d %H:%M'))"

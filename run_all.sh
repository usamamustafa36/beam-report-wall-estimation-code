#!/usr/bin/env bash
# Reruns every analysis from the traced scenes (step3_trace.py / step3_e2e_trace.py must have run).
# Each step writes artifacts/<step>.txt (or .json); logs go to artifacts/logs/.
set -e
cd "$(dirname "$0")"
mkdir -p artifacts/logs
run() { echo "== $* ($(date +%H:%M))"; python3 "$@" > "artifacts/logs/$(basename "$1" .py)${2:+_$2}.log" 2>&1; }
run step1_checks.py
run step2_checks.py
run step3_validate.py
run step3_checks.py
run step3_crb.py
run step3_fig.py
run step6_mismatch.py a
run step4_mle.py
run step5_detect.py
run step6_mismatch.py b
run step9_material.py
run step10_learned.py
echo "ALL DONE ($(date +%H:%M))"

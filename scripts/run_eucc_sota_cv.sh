#!/usr/bin/env bash
# Relaunch wrapper for scripts/run_eucc_sota_cv.py.
#
# The driver is idempotent (config hash -> parquet, skips done cells), so on a
# crash / segfault / freeze-reboot this just relaunches it until every cell is
# done. Extra args are passed through (e.g. --smoke, --models tabpfn_3).
#
#   ./scripts/run_eucc_sota_cv.sh
#   nohup ./scripts/run_eucc_sota_cv.sh >/dev/null 2>&1 &   # unattended

set -u
cd "$(dirname "$0")/.."

TABPFN_TOKEN=$(grep TABPFN_TOKEN ~/.bashrc | head -1 | sed 's/.*TABPFN_TOKEN="\(.*\)".*/\1/')
export TABPFN_TOKEN

LOG=results/runs_sota_cv/driver.log
mkdir -p results/runs_sota_cv

for attempt in $(seq 1 50); do
    echo "=== attempt ${attempt} $(date -Is) ===" | tee -a "$LOG"
    uv run python scripts/run_eucc_sota_cv.py "$@" 2>&1 | tee -a "$LOG"
    status=${PIPESTATUS[0]}
    if [ "$status" -eq 0 ]; then
        pending=$(uv run python scripts/run_eucc_sota_cv.py --dry-run | grep -oP 'pending: \K[0-9]+')
        if [ "${pending:-1}" -eq 0 ]; then
            echo "Complete after ${attempt} attempt(s)." | tee -a "$LOG"
            exit 0
        fi
    fi
    echo "exit=${status}, cells still pending -- relaunching in 30s" | tee -a "$LOG"
    sleep 30
done
echo "Gave up after 50 attempts; check ${LOG}." | tee -a "$LOG"
exit 1

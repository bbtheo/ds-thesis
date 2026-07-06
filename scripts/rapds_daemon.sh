#!/usr/bin/env bash
# Self-resuming daemon for the trimmed RAP design-space sweep (720 cells).
# The driver is idempotent (config-hash -> parquet, .failed sentinels), so on any
# death this just relaunches and resumes. Detach with: setsid nohup bash this.sh &
cd /home/theo/ds-gradu || exit 1
TOK=$(grep TABPFN_TOKEN ~/.bashrc | head -1 | sed 's/.*TABPFN_TOKEN="\(.*\)".*/\1/')
export TABPFN_TOKEN="$TOK"
# Cap CPU threads: the uncapped all-24-core kNN retrieval (paysim 6.3M rows) running
# concurrently with the 250W GPU hard-froze + rebooted the machine 2026-06-20 (nothing
# logged = power/hard-hang, not OOM/VRAM). Limiting BLAS/sklearn threads cuts the combined
# CPU+GPU power peak and keeps the desktop responsive. Result-neutral (thread count does
# not change kNN/BLAS outputs); retrieval is cached per (metric,G) so the slowdown is minor.
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
TARGET=720
for i in $(seq 1 400); do
  n=$(ls results/runs_rapds/*.parquet 2>/dev/null | wc -l)
  if [ "$n" -ge "$TARGET" ]; then
    echo "[$(date '+%F %T')] ALL $TARGET DONE (iter $i)"
    break
  fi
  echo "[$(date '+%F %T')] daemon iter $i: launching driver (done=$n/$TARGET)"
  uv run python scripts/run_rap_designspace.py --datasets eu_cc paysim banksim
  echo "[$(date '+%F %T')] driver exited (done=$(ls results/runs_rapds/*.parquet 2>/dev/null | wc -l)/$TARGET); resume in 15s"
  sleep 15
done
echo "[$(date '+%F %T')] daemon end (done=$(ls results/runs_rapds/*.parquet 2>/dev/null | wc -l)/$TARGET)"

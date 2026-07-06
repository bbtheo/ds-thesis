#!/usr/bin/env bash
# Self-resuming daemon for the C2 cross-tab sweep (360 cells). Idempotent; on any
# death it relaunches and resumes. Detach with: setsid nohup bash this.sh &
cd /home/theo/ds-gradu || exit 1
TOK=$(grep TABPFN_TOKEN ~/.bashrc | head -1 | sed 's/.*TABPFN_TOKEN="\(.*\)".*/\1/')
export TABPFN_TOKEN="$TOK"
# Thread caps (the 3 hard freezes were a parallel C compile, not us, but keep headroom).
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
TARGET=810
for i in $(seq 1 200); do
  n=$(ls results/runs_c2grid/*.parquet 2>/dev/null | wc -l)
  if [ "$n" -ge "$TARGET" ]; then echo "[$(date '+%F %T')] ALL $TARGET DONE (iter $i)"; break; fi
  echo "[$(date '+%F %T')] daemon iter $i: launching driver (done=$n/$TARGET)"
  uv run python scripts/run_c2_grid.py
  echo "[$(date '+%F %T')] driver exited (done=$(ls results/runs_c2grid/*.parquet 2>/dev/null | wc -l)/$TARGET); resume in 10s"
  sleep 10
done
echo "[$(date '+%F %T')] daemon end (done=$(ls results/runs_c2grid/*.parquet 2>/dev/null | wc -l)/$TARGET)"

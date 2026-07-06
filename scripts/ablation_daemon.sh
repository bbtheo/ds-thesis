#!/usr/bin/env bash
# Self-resuming daemon for the relevance-ablation batch 1 (KR/RK at G=32, ctx=50k).
# Idempotent; relaunches+resumes on any death. Detach: setsid nohup bash this.sh &
cd /home/theo/ds-gradu || exit 1
TOK=$(grep TABPFN_TOKEN ~/.bashrc | head -1 | sed 's/.*TABPFN_TOKEN="\(.*\)".*/\1/')
export TABPFN_TOKEN="$TOK"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
TARGET=72
for i in $(seq 1 200); do
  n=$(ls results/runs_ablation/*.parquet 2>/dev/null | wc -l)
  if [ "$n" -ge "$TARGET" ]; then echo "[$(date '+%F %T')] ALL $TARGET DONE (iter $i)"; break; fi
  echo "[$(date '+%F %T')] daemon iter $i: launch (done=$n/$TARGET)"
  uv run python scripts/run_relevance_ablation.py --granularities 32 --contexts 50000
  echo "[$(date '+%F %T')] driver exited (done=$(ls results/runs_ablation/*.parquet 2>/dev/null | wc -l)/$TARGET); resume 10s"
  sleep 10
done
echo "[$(date '+%F %T')] daemon end (done=$(ls results/runs_ablation/*.parquet 2>/dev/null | wc -l)/$TARGET)"

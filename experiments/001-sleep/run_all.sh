#!/usr/bin/env bash
# Phase 0 batch: re-run C2/C4/C5 with the fixed code, then 3 seeds each, with ARC-Easy
# before sleep (base model) and after each sleep run (its adapter).
# Ordered by priority and resumable: a run whose JSON already exists is skipped,
# so the script can be stopped in the morning and relaunched the next night.
#
#   systemd-run --user --unit=sleep-exp --working-directory="$PWD" -E PATH="$PATH" bash run_all.sh
#   journalctl --user -u sleep-exp -f                   # follow progress
set -u
cd "$(dirname "$0")"
mkdir -p runs/logs runs/arc runs/arc_cloze

JOBS=(
  "c2_rag 0" "c0_none 0" "c4_sleep_replay 0" "c5_sleep_integrate 0"
  "c2_rag 1" "c2_rag 2"
  "c4_sleep_replay 1" "c5_sleep_integrate 1"
  "c4_sleep_replay 2" "c5_sleep_integrate 2"
)

run_jobs() {
  set -o pipefail
  if [[ ! -f runs/arc/base.json ]]; then
    echo "=== $(date +%H:%M) ARC-Easy base"
    uv run python arc.py 2>&1 | tee runs/logs/arc_base.log
  fi
  if [[ ! -f runs/arc_cloze/base.json ]]; then
    echo "=== $(date +%H:%M) ARC-Easy cloze base"
    uv run python arc_cloze.py 2>&1 | tee runs/logs/arc_cloze_base.log
  fi
  for job in "${JOBS[@]}"; do
    read -r cond seed <<<"$job"
    name="${cond}_seed${seed}"
    if [[ -f "runs/${name}.json" ]]; then
      echo "skip $cond seed $seed (already done)"
    else
      echo "=== $(date +%H:%M) $cond seed $seed"
      uv run python run.py "$cond" --seed "$seed" --eval-days 1 3 5 7 10 2>&1 | tee "runs/logs/${name}.log" \
        || echo "!!! $cond seed $seed failed, see runs/logs"
    fi
    if [[ -d "adapters/${name}" && ! -f "runs/arc/${name}.json" ]]; then
      echo "=== $(date +%H:%M) ARC-Easy $name"
      uv run python arc.py "adapters/${name}" 2>&1 | tee "runs/logs/arc_${name}.log"
    fi
    if [[ -d "adapters/${name}" && ! -f "runs/arc_cloze/${name}.json" ]]; then
      echo "=== $(date +%H:%M) ARC-Easy cloze $name"
      uv run python arc_cloze.py "adapters/${name}" 2>&1 | tee "runs/logs/arc_cloze_${name}.log"
    fi
  done
  uv run python analyze.py
  echo "=== $(date +%H:%M) done"
}

# Keep the laptop awake while the batch runs.
export -f run_jobs
systemd-inhibit --what=sleep:idle:handle-lid-switch --why="sleep experiment batch" bash -c "$(declare -p JOBS); run_jobs"

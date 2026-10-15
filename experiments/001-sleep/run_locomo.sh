#!/usr/bin/env bash
# LoCoMo batch: one conversation, quick conditions first, then the two nights-heavy ones. Resumable:
# finished runs are skipped, and an interrupted sleep run resumes from its last night.
#
#   systemd-run --user --unit=locomo-exp --working-directory="$PWD" -E PATH="$PATH" bash run_locomo.sh 0
#   journalctl --user -u locomo-exp -f
# Run it with the project environment first on PATH (uv sync, or an existing venv).
set -u
cd "$(dirname "$0")"
CONV="${1:-0}"
mkdir -p runs/locomo runs/logs runs/arc runs/arc_cloze

JOBS=(
  "c0_none" "c2_rag --top-k 4" "c2_rag --top-k 10" "c3_lora_raw"
  "c1_full_context" "c4_sleep_replay" "c5_sleep_integrate"
)

run_jobs() {
  for job in "${JOBS[@]}"; do
    read -r cond opts <<<"$job"
    case "$opts" in "--top-k "*) suffix="_k${opts#--top-k }" ;; *) suffix="" ;; esac
    name="${cond}${suffix}_conv${CONV}"
    if [[ -f "runs/locomo/${name}.json" ]]; then
      echo "skip ${name}"
    else
      echo "=== $(date +%H:%M) ${name}"
      # shellcheck disable=SC2086
      python locomo_run.py "$cond" --conv "$CONV" $opts 2>&1 | tee "runs/logs/locomo_${name}.log" \
        || echo "!!! ${name} failed, see runs/logs"
    fi
    adapter="adapters/locomo_${name}"
    if [[ -d "$adapter" && ! -f "runs/arc/locomo_${name}.json" ]]; then
      echo "=== $(date +%H:%M) ARC-Easy ${name}"
      python arc.py "$adapter" 2>&1 | tee "runs/logs/arc_locomo_${name}.log"
      python arc_cloze.py "$adapter" 2>&1 | tee "runs/logs/arc_cloze_locomo_${name}.log"
    fi
  done
  echo "=== $(date +%H:%M) done"
}

export -f run_jobs
systemd-inhibit --what=sleep:idle:handle-lid-switch --why="locomo batch" bash -c "$(declare -p JOBS CONV); run_jobs"

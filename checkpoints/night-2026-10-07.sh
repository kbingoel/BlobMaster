#!/usr/bin/env bash
# Night of 2026-10-07: two Phase-5 runs back to back (gen-2.md §6 Phase 5, day 2), each from run 1's step 7600.
# Launch detached from the repo root:
#   setsid nohup checkpoints/night-2026-10-07.sh > checkpoints/night-2026-10-07.log 2>&1 < /dev/null &
# Stop everything: kill this script first (pkill -f night-2026-10-07.sh), then `touch checkpoints/<run>/STOP`.
set -u
cd "$(dirname "$0")/.."
for run in rl-2026-10-07a rl-2026-10-07b; do
  echo "== $run start $(date '+%F %T')"
  mkdir -p checkpoints/$run
  scripts/blobmaster-train.sh train --config checkpoints/$run.toml --output checkpoints/$run \
    > checkpoints/$run/train.log 2>&1 < /dev/null
  echo "== $run exit $? $(date '+%F %T')"
  ( unset LD_PRELOAD; .venv/bin/python scripts/plot_rl_run.py checkpoints/$run ) > checkpoints/$run/plot.log 2>&1
done
echo "== night done $(date '+%F %T')"

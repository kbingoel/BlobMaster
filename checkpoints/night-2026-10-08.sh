#!/usr/bin/env bash
# Night of 2026-10-08: two runs of policy iteration by rollouts back to back (gen-2.md §6 Phase 5, day 3), each from
# run 2b's final model: a (6 h, validation 2's settings) then b (5.5 h, a third of the LR and twice the batch).
# ~11.7 h in all.
# Launch detached from the repo root:
#   setsid nohup checkpoints/night-2026-10-08.sh > checkpoints/night-2026-10-08.log 2>&1 < /dev/null &
# Stop everything: kill this script first (pkill -f night-2026-10-08.sh), then `touch checkpoints/<run>/STOP`.
set -u
cd "$(dirname "$0")/.."
for run in pi-2026-10-08a pi-2026-10-08b; do
  echo "== $run start $(date '+%F %T')"
  mkdir -p checkpoints/$run
  scripts/blobmaster-train.sh train --config checkpoints/$run.toml --output checkpoints/$run \
    > checkpoints/$run/train.log 2>&1 < /dev/null
  echo "== $run exit $? $(date '+%F %T')"
  ( unset LD_PRELOAD; .venv/bin/python scripts/plot_rl_run.py checkpoints/$run ) > checkpoints/$run/plot.log 2>&1
done
echo "== night done $(date '+%F %T')"

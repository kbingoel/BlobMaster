#!/usr/bin/env bash
# Chain of 2026-10-09 (gen-2.md §6 Phase 5, day 4), ~70 h back to back:
#   e1      (4 h)       exploiter of last night's best model (pi-2026-10-08b's final): how exploitable is it?
#   l01-l10 (6 h each)  league legs (chain-2026-10-09-league.toml.in), each from the previous leg's final; every
#                       leg's final joins the frozen opponents of the next ones
#   e2      (4 h)       exploiter of the league's final: did the league make P harder to exploit?
# A leg that leaves no final model, or whose final lost more than 3 points against rule bot 2 since its start,
# ends the league; e2 then targets the last good model.
# Launch detached from the repo root:
#   setsid nohup checkpoints/chain-2026-10-09.sh > checkpoints/chain-2026-10-09.log 2>&1 < /dev/null &
# Stop after the running run: touch checkpoints/chain-2026-10-09.STOP. Stop now: that, then
# touch checkpoints/<running run>/STOP (it saves; `train --resume` continues it).
set -u
cd "$(dirname "$0")/.."
ACTORS=28
START=checkpoints/pi-2026-10-08b/models/step-002931
POOL=(
  checkpoints/pi-2026-10-08b/models/step-002931/model
  checkpoints/pi-2026-10-08a/models/step-006403/model
  checkpoints/rl-2026-10-07b/models/step-003285/model
  checkpoints/rl-2026-10-06/models/step-007600/model
  checkpoints/pretrain-2026-10-06/model
)
STOPFILE=checkpoints/chain-2026-10-09.STOP

run() { # name template hours init: write checkpoints/<name>.toml and train it
  local name=$1 template=$2 hours=$3 init=$4
  if [[ -e $STOPFILE ]]; then echo "== stop file: not starting $name"; return 1; fi
  local opp=""
  for m in "${POOL[@]}"; do opp+="  \"$m\",\n"; done
  sed -e "s|@HOURS@|$hours|" -e "s|@INIT@|$init|g" -e "s|@ACTORS@|$ACTORS|" -e "s|@OPPONENTS@|$opp|" \
    "$template" > checkpoints/$name.toml
  mkdir -p checkpoints/$name
  echo "== $name start $(date '+%F %T') from $init"
  scripts/blobmaster-train.sh train --config checkpoints/$name.toml --output checkpoints/$name \
    > checkpoints/$name/train.log 2>&1 < /dev/null
  echo "== $name exit $? $(date '+%F %T')"
  ( unset LD_PRELOAD; .venv/bin/python scripts/plot_rl_run.py checkpoints/$name ) > checkpoints/$name/plot.log 2>&1
  return 0
}

final_of() { # the run's last published model (models/step-N), if complete
  local f
  f=$(ls -d checkpoints/$1/models/step-* 2>/dev/null | sort | tail -1)
  [[ -n "$f" && -f "$f/model/policy.onnx" && -f "$f/checkpoint/policy.ot" ]] && echo "$f"
}

rb2_change() { # the last net-rb2 bench's change since the run's start, paired
  python3 - "checkpoints/$1/metrics.jsonl" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
b = [r for r in rows if r.get("kind") == "bench" and r.get("name") == "net-rb2" and r.get("paired")]
print(next((p["diff"] for p in b[-1]["paired"] if p.get("vs") == "start"), 0.0) if b else 0.0)
PY
}

run chain-2026-10-09-e1 checkpoints/chain-2026-10-09-exploit.toml.in 4.0 $START
init=$START
for k in 01 02 03 04 05 06 07 08 09 10; do
  name=chain-2026-10-09-l$k
  run $name checkpoints/chain-2026-10-09-league.toml.in 6.0 $init || break
  if ! f=$(final_of $name); then echo "== $name left no final model: the league ends"; break; fi
  d=$(rb2_change $name)
  if awk -v d="$d" 'BEGIN{exit !(d < -3.0)}'; then echo "== $name lost $d vs rule bot 2: the league ends"; break; fi
  POOL+=("$f/model")
  init=$f
done
run chain-2026-10-09-e2 checkpoints/chain-2026-10-09-exploit.toml.in 4.0 $init
echo "== chain done $(date '+%F %T')"

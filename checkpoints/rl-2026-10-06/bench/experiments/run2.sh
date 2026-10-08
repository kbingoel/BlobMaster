#!/usr/bin/env bash
# Queued after run.sh: c_puct 2.0, and c_puct 1.0 with plays 5 x 200, on the step-7600 networks.
set -u
cd "$(dirname "$0")/../../../.."
R=checkpoints/rl-2026-10-06; E=$R/bench/experiments; B=./target/release/blobmaster
until grep -q "ALL DONE" $E/run.log; do sleep 15; done
base=$R/bench/search-rb2-step-007600.csv
run() { # name c_puct extra-args...
  local name=$1 c=$2; shift 2
  echo "== $name start $(date +%T)"
  $B bench $R/models/step-007600/model --mode search --opponent rulebot2 --deals 128 --seed 7 --c-puct "$c" "$@" \
    --per-deal-out $E/$name.csv --compare $base > $E/$name.txt 2> $E/$name.err
  echo "== $name done $(date +%T): $(grep -E 'points/game|paired' $E/$name.txt | tr '\n' ' ')"
}
run search-c2.0-step7600 2.0
run search-c1.0-plays5x200-step7600 1.0 --sims 200
echo ALL DONE 2

#!/usr/bin/env bash
# Search vs rule bot 2, seed 7, 128 deals, on the step-7600 networks: c_puct sweep and a hybrid
# (step-7600 P with the warm start's V), each paired with c_puct 0.2 (bench/search-rb2-step-007600.csv).
set -u
cd "$(dirname "$0")/../../../.."
R=checkpoints/rl-2026-10-06; E=$R/bench/experiments; B=./target/release/blobmaster
base=$R/bench/search-rb2-step-007600.csv
run() { # name model c_puct
  echo "== $1 start $(date +%T)"
  $B bench "$2" --mode search --opponent rulebot2 --deals 128 --seed 7 --c-puct "$3" \
    --per-deal-out $E/$1.csv --compare $base > $E/$1.txt 2> $E/$1.err
  echo "== $1 done $(date +%T): $(grep -E 'points/game|paired' $E/$1.txt | tr '\n' ' ')"
}
run search-c1.0-step7600 $R/models/step-007600/model 1.0
run search-c0.5-step7600 $R/models/step-007600/model 0.5
run search-c0.2-hybrid-p7600-v0 $E/hybrid-p7600-v0 0.2
echo ALL DONE

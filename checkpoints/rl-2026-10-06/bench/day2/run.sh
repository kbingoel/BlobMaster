#!/usr/bin/env bash
# Day-2 decomposition (2026-10-07): step-7600 networks, seed 7, 128 deals vs rule bot 2,
# each paired against P alone on the same deals (bench/experiments/net-s7-7600.csv).
set -u
cd "$(dirname "$0")/../../../.."
R=checkpoints/rl-2026-10-06; M=$R/models/step-007600/model; E=$R/bench/day2; B=$E/blobmaster-frozen
b() { local n=$1; shift
  echo "== $n start $(date +%T)"
  $B bench $M --mode search --opponent rulebot2 --deals 128 --seed 7 \
    --compare $R/bench/experiments/net-s7-7600.csv --per-deal-out $E/$n.csv "$@" > $E/$n.txt 2> $E/$n.err
  echo "== $n done $(date +%T): $(grep -E 'diff|paired' $E/$n.txt | tr '\n' ' ')"
}
b p-via-search --dets 1 --sims 1 --bid-dets 1 --bid-sims 1 --one-card-bids policy
b bids-only    --c-puct 1.0 --dets 1 --sims 1
b plays-only   --c-puct 1.0 --bid-dets 1 --bid-sims 1 --one-card-bids policy
b plays-20x25  --c-puct 1.0 --dets 20 --sims 25
echo "DAY2 DECOMPOSITION FINISHED"

#!/usr/bin/env bash
# Q-rule sweep 1 (2026-10-07), queued after run.sh: step-7600 networks, seed 7, 128 deals vs
# rule bot 2, paired against P alone. Plays and bids searched separately, the other from P.
set -u
cd "$(dirname "$0")/../../../.."
R=checkpoints/rl-2026-10-06; M=$R/models/step-007600/model; E=$R/bench/day2; B=$E/blobmaster-q
until grep -q "DAY2 DECOMPOSITION FINISHED" $E/run.log; do sleep 20; done
b() { local n=$1; shift
  echo "== $n start $(date +%T)"
  $B bench $M --mode search --opponent rulebot2 --deals 128 --seed 7 \
    --compare $R/bench/experiments/net-s7-7600.csv --per-deal-out $E/$n.csv "$@" > $E/$n.txt 2> $E/$n.err
  echo "== $n done $(date +%T): $(grep -E 'diff|paired|table' $E/$n.txt | tr '\n' ' ')"
}
PB="--bid-dets 1 --bid-sims 1 --one-card-bids policy"   # bids from P
PP="--dets 1 --sims 1"                                  # plays from P
b q-plays-32x16-T0.05  --root q --q-temp 0.05 --dets 32 --sims 16 --bid-candidates 2 $PB
b q-plays-64x8-T0.05   --root q --q-temp 0.05 --dets 64 --sims 8 --bid-candidates 2 $PB
b q-plays-32x16-T0     --root q --q-temp 0 --dets 32 --sims 16 --bid-candidates 2 $PB
b q-bids-32x16-T0.05   --root q --q-temp 0.05 --bid-dets 32 --bid-sims 16 --bid-candidates 2 $PP
echo "SWEEP1 FINISHED"

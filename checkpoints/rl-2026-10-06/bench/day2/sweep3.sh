#!/usr/bin/env bash
# Queue 3 (2026-10-07, 15:00): search against four copies of P first (self-play's setting; P alone scores 0).
# Against rule bot 2 every search models the opponents as P (play and bid reading), which stopped fitting as
# P left rule bot 2 behind; so the improvement step is judged against P.
set -u
cd "$(dirname "$0")/../../../.."
R=checkpoints/rl-2026-10-06; M=$R/models/step-007600/model; E=$R/bench/day2; B=$E/blobmaster-r
while kill -0 1312618 2>/dev/null; do sleep 15; done
echo "== qfull-vsP done $(date +%T): $(grep -E 'diff|table' $E/qfull-vsP.txt | tr '\n' ' ')"
b() { local n=$1 model=$2 opp=$3; shift 3
  echo "== $n start $(date +%T)"
  $B bench $model --opponent $opp --seed 7 --per-deal-out $E/$n.csv "$@" > $E/$n.txt 2> $E/$n.err
  echo "== $n done $(date +%T): $(grep -E 'diff|paired|table' $E/$n.txt | tr '\n' ' ')"
}
Q="--mode search --root q --q-temp 0.05 --c-puct 1.0 --bid-dets 32 --bid-sims 16 --dets 32 --sims 16 --bid-candidates 2"
H=$E/hybrid-p7600-vfit; mkdir -p $H
VM=$(python3 -c "import json; print(json.load(open('checkpoints/vfit-2026-10-07/state.json'))['model'])")
cp $M/policy.onnx $H/policy.onnx; cp $VM/value.onnx $H/value.onnx; cp $M/meta.json $H/meta.json
b qfull-vfit-vsP $H $M --deals 128 $Q --compare $E/qfull-vsP.csv
b visits-c1-vsP $M $M --deals 128 --mode search --c-puct 1.0 --compare $E/net-vsP.csv
b rollouts-vsP $M $M --deals 32 --mode search --root rollouts --q-temp 0 --dets 32 --bid-dets 32 --bid-candidates 2
b qfull-vfit-rb2 $H rulebot2 --deals 128 $Q --compare $E/qfull-rb2.csv
echo "SWEEP3 FINISHED"

#!/usr/bin/env bash
# Queue 2 (2026-10-07): full search with the Q rule, against rule bot 2 and against four copies of P
# (self-play's setting: P alone scores exactly 0 there); then the same with V from the V-stream test
# (checkpoints/vfit-2026-10-07) on step 7600's P; then the rest of queue 1.
set -u
cd "$(dirname "$0")/../../../.."
R=checkpoints/rl-2026-10-06; M=$R/models/step-007600/model; E=$R/bench/day2; B=$E/blobmaster-q
while kill -0 1255073 2>/dev/null; do sleep 15; done
echo "== q-plays-64x8-T0.05 done $(date +%T): $(grep -E 'paired' $E/q-plays-64x8-T0.05.txt)"
b() { local n=$1 model=$2 opp=$3; shift 3
  echo "== $n start $(date +%T)"
  $B bench $model --opponent $opp --deals 128 --seed 7 --per-deal-out $E/$n.csv "$@" > $E/$n.txt 2> $E/$n.err
  echo "== $n done $(date +%T): $(grep -E 'diff|paired|table' $E/$n.txt | tr '\n' ' ')"
}
Q="--mode search --root q --q-temp 0.05 --c-puct 1.0 --bid-dets 32 --bid-sims 16 --dets 32 --sims 16 --bid-candidates 2"
b net-vsP $M $M --mode network
b qfull-rb2 $M rulebot2 $Q --compare $R/bench/experiments/net-s7-7600.csv
b qfull-vsP $M $M $Q --compare $E/net-vsP.csv
b visits-c1-vsP $M $M --mode search --c-puct 1.0 --compare $E/net-vsP.csv
# The V-stream test's V on step 7600's P.
until [ -f checkpoints/vfit-2026-10-07/state.json ] && grep -q '"finished": true' checkpoints/vfit-2026-10-07/state.json; do sleep 30; done
H=$E/hybrid-p7600-vfit; mkdir -p $H
VM=$(python3 -c "import json; print(json.load(open('checkpoints/vfit-2026-10-07/state.json'))['model'])")
ln -sf "$(realpath $M/policy.onnx)" $H/policy.onnx; ln -sf "$(realpath $VM/value.onnx)" $H/value.onnx; cp $M/meta.json $H/meta.json
echo "== hybrid: P of step 7600, V of $VM"
b qfull-vfit-rb2 $H rulebot2 $Q --compare $E/qfull-rb2.csv
b qfull-vfit-vsP $H $M $Q --compare $E/qfull-vsP.csv
b q-plays-32x16-T0 $M rulebot2 --mode search --root q --q-temp 0 --c-puct 1.0 --dets 32 --sims 16 --bid-candidates 2 --bid-dets 1 --bid-sims 1 --one-card-bids policy --compare $R/bench/experiments/net-s7-7600.csv
b q-bids-32x16-T0.05 $M rulebot2 --mode search --root q --q-temp 0.05 --c-puct 1.0 --bid-dets 32 --bid-sims 16 --bid-candidates 2 --dets 1 --sims 1 --compare $R/bench/experiments/net-s7-7600.csv
echo "SWEEP2 FINISHED"

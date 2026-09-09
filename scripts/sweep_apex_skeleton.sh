#!/bin/bash
# S2 apex skeleton sweep (E62): Q0 fidelity grid + bounded-Q diagnostic.
# Question: how much of warden's 5.07 is recoverable by act()-path fidelity
# (mask-always, WAIT penalty), given the learned Q is net-negative (E61
# Q-delta -0.42 on both seeds)?
#   arm0 q0/mask1/wait.30  = A4 Q0 replication (expect ~3.76 pooled; E61)
#   arm1 q0/mask0/wait.30  = warden move filter (mask binds only must_flee)
#   arm2 q0/mask1/wait0    = no WAIT penalty (warden WAITs 27% of the time)
#   arm3 q0/mask0/wait0    = full warden fidelity
#   arm4 q1/mask1/wait.30 + Q_CLIP=0.5 = does bounded authority stop harming?
# Base: pinned A4 weights (ep3400, my-saved-model.pt sha de8c26d4...3f).
# RELAX_TIER excluded: E42 exonerated the mask (zero vetoed) and E59 holds
# it non-binding in combat (~20 bombs/rd). Decision rule: any Q0 arm >= 4.8
# becomes ARBITER's fallback skeleton (5.07 bar met at S0); all ~3.8 ->
# re-diagnose before P0.
# Protocol: frozen CPU, apex vs 3x rule_based, classic, 40rd x seeds 0,1
# (paired; seed spread is ~+-0.5, so single-seed screens are insufficient).
set -e
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
PIN="de8c26d44a0694c21b64f0c0ad203382fe6c53504d7035c8bca3cfb4fb73af3f"
got=$(sha256sum agent_code/apex/my-saved-model.pt | cut -d' ' -f1)
if [ "$got" != "$PIN" ]; then
  echo "PIN MISMATCH: my-saved-model.pt is $got, want $PIN (E34 hygiene)"; exit 1
fi
echo "pin OK: A4 weights ($got)"

run() { # arm tag, env assignments..., seed
  local arm="$1"; local tag="$2"; shift 2
  local seed="$1"; shift
  local out="results/gate_apex_s2_${arm}_${tag}_s${seed}.json"
  echo "=== S2/${arm}/${tag} seed ${seed} -> ${out} ==="
  env "$@" $PY main.py play --no-gui \
    --agents apex rule_based_agent rule_based_agent rule_based_agent \
    --train 0 --continue-without-training --scenario classic \
    --n-rounds 40 --seed "$seed" --save-stats "$out" 2>&1 | tail -1
}

for seed in 0 1; do
  run arm0 q0base        "$seed" APEX_Q_WEIGHT=0
  run arm1 mask0         "$seed" APEX_Q_WEIGHT=0 APEX_MASK_ALWAYS=0
  run arm2 wait0         "$seed" APEX_Q_WEIGHT=0 APEX_WAIT=0
  run arm3 fullwarden    "$seed" APEX_Q_WEIGHT=0 APEX_MASK_ALWAYS=0 APEX_WAIT=0
  run arm4 qclip05       "$seed" APEX_Q_WEIGHT=1 APEX_Q_CLIP=0.5
done

got2=$(sha256sum agent_code/apex/my-saved-model.pt | cut -d' ' -f1)
if [ "$got2" != "$PIN" ]; then
  echo "PIN CHANGED DURING SWEEP: $got2 (E34 hygiene violation)"; exit 1
fi
echo "pin intact after sweep. Done."

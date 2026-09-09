#!/bin/bash
# A1->A2 stage reset: archive best/last, clear cross-regime EMA, eps re-warm 0.30.
# Mimics train_overlord_curriculum.sh reset_for_next_stage + train_reaper.sh.
# Keeps q_net/target/optimizer/total_steps momentum; only best_ema/ema_reward
# cleared and epsilon_steps rewound. Atomic tmp+replace, loss <=5 rounds.
# Usage: bash scripts/apex_archive_A1_for_A2.sh
set -e
cd "$(dirname "$0")/.."
TS=$(date +%Y%m%d_%H%M)
mkdir -p results/archive logs
if [ ! -f agent_code/apex/checkpoints/last.pt ]; then
  echo "ERROR: agent_code/apex/checkpoints/last.pt missing, nothing to archive"
  exit 1
fi
cp agent_code/apex/checkpoints/best.pt "results/archive/apex_A1_best_${TS}.pt"
cp agent_code/apex/checkpoints/last.pt "results/archive/apex_A1_last_${TS}.pt"
echo "archived A1 best+last with TS=$TS"
python3 - <<'EOF'
import torch, os
p = 'agent_code/apex/checkpoints/last.pt'
ckpt = torch.load(p, map_location='cpu', weights_only=False)
print(f"before: ep={ckpt.get('episode')} eps_steps={ckpt.get('epsilon_steps')} ema={ckpt.get('ema_reward')} best_ema={ckpt.get('best_ema')}")
ckpt['best_ema'] = None
ckpt['ema_reward'] = None
ckpt['epsilon_steps'] = 73684  # 0.05+0.95*(1-73684/100000)=0.30 on 100k decay
torch.save(ckpt, p + '.tmp')
os.replace(p + '.tmp', p)
v = torch.load(p, map_location='cpu', weights_only=False)
eps = 0.05 + 0.95 * (1 - min(1.0, v['epsilon_steps'] / 100000))
print(f"after: ep={v.get('episode')} eps_steps={v.get('epsilon_steps')} eps_now={eps:.4f}")
assert abs(eps - 0.30) < 0.005, eps
print('reset OK: archived A1, EMA cleared, eps=0.30')
EOF
ls -lh "results/archive/apex_A1_best_${TS}.pt" "results/archive/apex_A1_last_${TS}.pt" agent_code/apex/checkpoints/last.pt

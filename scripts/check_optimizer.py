"""Correctness gates for dml_optimizer.py (run before any benchmark).

1. Stock Adam trajectory == DMLAdam trajectory (CPU, fixed seed, math parity).
2. Lion deterministic across two same-seed runs (CUDA when available, else CPU).
3. Factory returns stock Adam/AdamW on CUDA/CPU, Lion on request.
Usage: python3 scripts/check_optimizer.py
"""
import os
import sys

import torch

sys.path.insert(0, '.')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from dml_optimizer import DMLAdam, DMLLion, build_optimizer_for_device  # noqa: E402


def main():
    fails = []

    # Gate 1: parity on CPU (isolated subprocess: each gate is a subprocess
    # so device queues don't leak between CPU and CUDA runs).
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)

    def run_gate(code):
        p = subprocess.run([sys.executable, '-c', code], capture_output=True,
                           text=True, cwd=root)
        print(p.stdout.strip())
        if p.returncode != 0:
            print(p.stderr.strip()[-2000:])
        return p.returncode == 0

    pre = ('import sys; sys.path.insert(0, %r); sys.path.insert(0, %r); sys.path.insert(0, %r); '
           'import torch; ' % (os.path.join(root, 'agent_code', 'sentinel'), root, here))

    ok1 = run_gate(pre + '''
from check_gate_lib import run_traj
import torch.optim as optim
l1, p1 = run_traj(__import__("dml_optimizer").DMLAdam, "cpu", lr=1e-3)
l2, p2 = run_traj(optim.Adam, "cpu", lr=1e-3)
dl = max(abs(a - b) for a, b in zip(l1, l2))
dp = float((p1 - p2).abs().max())
print(f"gate1 DMLAdam==torch.Adam: max loss diff {dl:.2e}, max param diff {dp:.2e}")
assert dl < 1e-5 and dp < 1e-5, "parity"
''')
    if not ok1:
        fails.append('parity')

    # Gate 2: Lion determinism on CUDA (or CPU when no GPU).
    ok2 = run_gate(pre + '''
import torch
from dml_optimizer import DMLLion
from check_gate_lib import run_traj
dev = "cuda" if torch.cuda.is_available() else "cpu"
_, q1 = run_traj(DMLLion, dev, lr=3e-4)
_, q2 = run_traj(DMLLion, dev, lr=3e-4)
dd = float((q1 - q2).abs().max())
print(f"gate2 Lion determinism on {dev}: max param diff {dd:.2e}")
assert dd == 0.0, "determinism"
''')
    if not ok2:
        fails.append('determinism')

    # Gate 3: factory returns stock optimizers on CUDA/CPU.
    ok3 = run_gate(pre + '''
import torch
from dml_optimizer import build_optimizer_for_device, DMLLion
m = torch.nn.Linear(4, 4)
cpu = torch.device("cpu")
o = build_optimizer_for_device(cpu, m.parameters(), lr=1e-3)
assert type(o).__name__ == "Adam", f"CPU must use torch Adam, got {type(o).__name__}"
o = build_optimizer_for_device(cpu, m.parameters(), lr=1e-3, adamw=True)
assert type(o).__name__ == "AdamW", f"CPU must use torch AdamW, got {type(o).__name__}"
o = build_optimizer_for_device(cpu, m.parameters(), lr=3e-4, name="lion")
assert isinstance(o, DMLLion), f"Lion must use DMLLion impl, got {type(o).__name__}"
if torch.cuda.is_available():
    dev = torch.device("cuda")
    o = build_optimizer_for_device(dev, m.parameters(), lr=1e-3)
    assert type(o).__name__ == "Adam", f"CUDA must use torch Adam, got {type(o).__name__}"
    print("gate3 factory: CPU->Adam/AdamW, CUDA->Adam, lion->DMLLion CLEAN")
else:
    print("gate3 factory: CPU->Adam/AdamW, lion->DMLLion CLEAN (no CUDA here)")
''')
    if not ok3:
        fails.append('factory')

    if fails:
        print('FAIL:', fails)
        return 1
    print('ALL GATES PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())

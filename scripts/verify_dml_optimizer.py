"""Strict verification for dml_optimizer.py + CUDA training path.

Checks:
  1. DMLAdam numerical equivalence vs torch.optim.Adam on CPU (max diff < 1e-5).
  2. DMLAdamW numerical equivalence vs torch.optim.AdamW on CPU.
  3. Factory returns stock Adam/AdamW on CUDA/CPU, Lion impl on request.
  4. On CUDA (if available): model + optimizer states on cuda:0,
     5 training steps (AMP default path + fp32 path), batch tensors on device.
  5. Sentinel/overlord wiring: setup_training picks CUDA factory + AMP.

Usage:
    python3 scripts/verify_dml_optimizer.py
"""

import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "agent_code", "sentinel"))

import torch

from dml_optimizer import (
    DMLAdam,
    DMLAdamW,
    DMLLion,
    build_optimizer_for_device,
    is_cuda_device,
    assert_params_on_device,
)


def _net(seed=0):
    g = torch.Generator().manual_seed(seed)
    m = torch.nn.Sequential(torch.nn.Linear(46, 64), torch.nn.ReLU(), torch.nn.Linear(64, 6))
    for p in m.parameters():
        torch.nn.init.normal_(p, generator=g)
    return m


def check_equivalence(cls, ref_cls, tag, **kw):
    a, b = _net(0), _net(0)
    b.load_state_dict(a.state_dict())
    o1, o2 = cls(a.parameters(), **kw), ref_cls(b.parameters(), **kw)
    torch.manual_seed(0)
    for i in range(50):
        x = torch.randn(32, 46)
        t = torch.randint(0, 6, (32,))
        for m, o in ((a, o1), (b, o2)):
            o.zero_grad()
            loss = torch.nn.functional.cross_entropy(m(x), t)
            loss.backward()
            o.step()
    diffs = [(pa - pb).abs().max().item() for pa, pb in zip(a.parameters(), b.parameters())]
    md = max(diffs)
    print(f"[{tag}] max param diff after 50 steps vs torch: {md:.2e}")
    assert md < 1e-5, f"{tag} diverges from torch reference (diff={md:.2e})"
    print(f"[{tag}] equivalence OK")


def check_factory():
    cpu = torch.device("cpu")
    o = build_optimizer_for_device(cpu, _net().parameters(), lr=1e-3)
    assert type(o).__name__ == "Adam", f"CPU must use torch Adam, got {type(o).__name__}"
    o = build_optimizer_for_device(cpu, _net().parameters(), lr=1e-3, adamw=True)
    assert type(o).__name__ == "AdamW", f"CPU must use torch AdamW, got {type(o).__name__}"
    o = build_optimizer_for_device(cpu, _net().parameters(), lr=3e-4, name="lion")
    assert isinstance(o, DMLLion), f"Lion must use Lion impl, got {type(o).__name__}"
    print("[factory] CPU -> torch Adam/AdamW, lion -> Lion OK")

    if torch.cuda.is_available():
        dev = torch.device("cuda")
        o = build_optimizer_for_device(dev, _net().parameters(), lr=1e-3)
        assert type(o).__name__ == "Adam", f"CUDA must use torch Adam, got {type(o).__name__}"
        o = build_optimizer_for_device(dev, _net().parameters(), lr=1e-3, adamw=True)
        assert type(o).__name__ == "AdamW", f"CUDA must use torch AdamW, got {type(o).__name__}"
        assert is_cuda_device(dev) and not is_cuda_device(cpu)
        print("[factory] CUDA -> torch Adam/AdamW OK")
    else:
        print("[factory] no CUDA here — CUDA branch skipped, CPU checks still valid")


def check_cuda_device():
    if not torch.cuda.is_available():
        print("[cuda] no CUDA device here (CPU box) — live-GPU checks skipped, CPU checks still valid")
        return False
    dev = torch.device("cuda")
    print(f"[cuda] device={dev} ({torch.cuda.get_device_name(0)})")
    for tag, use_amp in (("fp32", False), ("amp", True)):
        m = _net().to(dev)
        assert_params_on_device(m, dev, what="verify_net")
        opt = build_optimizer_for_device(dev, m.parameters(), lr=1e-3)
        assert type(opt).__name__ == "Adam", f"expected stock Adam on {dev}, got {type(opt).__name__}"
        scaler = torch.amp.GradScaler("cuda") if use_amp else None
        m.train()
        t0 = time.perf_counter()
        for i in range(5):
            x = torch.randn(256, 46, device=dev)  # sentinel BATCH=256, FEATURE_DIM=46
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss = m(x).pow(2).mean()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss = m(x).pow(2).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
            for p in m.parameters():
                assert p.grad is None or str(p.grad.device) == str(dev), (
                    f"grad on {p.grad.device}, expected {dev}"
                )
            for st in opt.state.values():
                for v in st.values():
                    if isinstance(v, torch.Tensor):
                        assert str(v.device) == str(dev), (
                            f"optimizer state on {v.device}, expected {dev}"
                        )
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 5 * 1000
        print(f"[cuda] 5x fwd/bwd/step ({tag}) on {dev}: {dt:.1f} ms/iter OK")
    return True


def check_wiring():
    for agent in ("sentinel", "overlord"):
        path = os.path.join(REPO_ROOT, "agent_code", agent, "train.py")
        with open(path) as f:
            src = f.read()
        assert "build_optimizer_for_device" in src, f"{agent} train.py not wired to factory"
        assert "torch_directml" not in src, f"{agent} train.py still references torch_directml"
        assert "privateuse" not in src, f"{agent} train.py still references privateuse"
        assert "amp_enabled" in src or "autocast" in src, f"{agent} train.py missing AMP path"
        assert "load_state_dict(ckpt['optimizer'])" in src, "checkpoint resume missing"
    print("[wiring] sentinel+overlord train.py use CUDA factory + AMP + resume OK")


def main():
    print(f"torch: {torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
    live = check_cuda_device()
    check_equivalence(DMLAdam, torch.optim.Adam, "DMLAdam", lr=1e-3)
    check_equivalence(
        DMLAdamW, torch.optim.AdamW, "DMLAdamW", lr=1e-3, weight_decay=1e-2
    )
    check_factory()
    check_wiring()
    print(f"PASS (live CUDA exercised: {live})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

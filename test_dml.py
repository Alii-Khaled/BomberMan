"""CUDA probe for sentinel (Colab NVIDIA GPUs). CPU fallback, never hangs.

Legacy name kept for compat (was DirectML probe for WSL2 + RX 6900 XT).
Canonical probe is test_gpu.py. This file forwards to the CUDA path.

Checks: device creation, matmul + autograd on CUDA, sentinel DuelingMLP
forward/backward on CUDA, atomic checkpoint round-trip.
Usage: python3 test_dml.py  (or python3 test_gpu.py)
"""
import os
import sys
import tempfile


def main() -> int:
    sys.path.insert(0, 'agent_code/sentinel')
    try:
        from device import get_device, is_cuda
    except ImportError:
        from agent_code.sentinel.device import get_device, is_cuda
    import torch

    print(f'torch: {torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}')

    dev = get_device()
    print(f'device: {dev} cuda={is_cuda(dev)}')

    # 1. matmul + sync via .item() / cuda.synchronize
    a = torch.randn(1024, 1024, device=dev)
    b = torch.randn(1024, 1024, device=dev)
    c = (a @ b).sum().item()
    if str(getattr(dev, 'type', dev)) == 'cuda':
        torch.cuda.synchronize()
    print(f'matmul 1024^3 on {dev}: sum={c:.1f} OK')

    # 2. sentinel MLP forward/backward on device (AMP when CUDA)
    from model import build_model, FEATURE_DIM
    m = build_model().to(dev)
    m.train()
    x = torch.randn(32, FEATURE_DIM, device=dev)
    use_amp = str(getattr(dev, 'type', dev)) == 'cuda'
    if use_amp:
        scaler = torch.amp.GradScaler('cuda')
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            loss = m(x).pow(2).mean()
        scaler.scale(loss).backward()
        scaler.step(torch.optim.SGD(m.parameters(), lr=1e-3))
        scaler.update()
    else:
        loss = m(x).pow(2).mean()
        loss.backward()
    print(f'DuelingMLP fwd/bwd (AMP={use_amp}) on {dev}: loss={loss.detach().cpu().item():.4f} OK')

    # 3. atomic checkpoint round-trip (temp dir, does not touch real checkpoints)
    from checkpointing import _atomic_save, load_checkpoint
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'probe.pt')
        _atomic_save(p, {'q_net': {k: v.cpu() for k, v in m.state_dict().items()}})
        rt = load_checkpoint(p)
        assert rt and 'q_net' in rt, 'checkpoint round-trip failed'
    print('checkpoint atomic save/load OK')
    print(f'PASS (active device: {dev})')
    return 0


if __name__ == '__main__':
    sys.exit(main())

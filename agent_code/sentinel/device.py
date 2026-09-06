"""CUDA device helper for sentinel (Google Colab / NVIDIA GPUs).

Colab-safe: defaults to CUDA when available, else CPU. Never raises.
act() always stays on CPU; only q_net/target_net in train.py move to CUDA.
"""

import os


def _resolve_want(env_flag='SENTINEL_DEVICE'):
    """Resolve desired device string from env.

    Primary: SENTINEL_DEVICE=cuda|cpu|auto (default auto).
    Legacy: SENTINEL_DML / SENTINEL_CUDA honored for compat:
      SENTINEL_DML=0 -> cpu, SENTINEL_CUDA=0 -> cpu.
    """
    raw = os.environ.get(env_flag, 'auto').strip().lower()
    if raw in ('cuda', 'cpu'):
        return raw
    # legacy compat (no DirectML path anymore — maps to cpu/cuda choice)
    if os.environ.get('SENTINEL_DML', '1') == '0':
        return 'cpu'
    if os.environ.get('SENTINEL_CUDA', '1') == '0':
        return 'cpu'
    return 'auto'


def get_device(logger=None, env_flag='SENTINEL_DEVICE'):
    """Return a torch.device: CUDA when usable, else CPU.

    Set SENTINEL_DEVICE=cpu to force CPU (tournament / debugging).
    Never raises: falls back to CPU with a log line.
    """
    try:
        import torch
    except Exception as ex:
        if logger:
            logger.warning(f'sentinel torch import failed, CPU fallback: {ex}')
        return 'cpu'
    want = _resolve_want(env_flag)
    if want == 'cpu':
        return torch.device('cpu')
    try:
        if torch.cuda.is_available():
            dev = torch.device('cuda')
            # Fast probe: create tiny tensor, run op, sync.
            probe = torch.zeros(8, device=dev)
            _ = (probe + 1).sum().item()
            torch.cuda.synchronize()
            del probe
            if logger:
                logger.info(f'sentinel device=cuda ({torch.cuda.get_device_name(0)})')
            return dev
        raise RuntimeError('torch.cuda.is_available() is False')
    except Exception as ex:
        if logger:
            logger.warning(f'sentinel CUDA unavailable, CPU fallback: {ex}')
        try:
            return torch.device('cpu')
        except Exception:
            return 'cpu'


def is_cuda(device) -> bool:
    """True if device is a CUDA device."""
    try:
        t = getattr(device, 'type', device)
        return str(t).lower() == 'cuda'
    except Exception:
        return False


def is_dml(device) -> bool:
    """Legacy shim: DirectML no longer supported — always False."""
    return False

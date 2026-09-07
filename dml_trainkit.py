"""Shared training-kit for CUDA agents (sentinel + overlord).

Single home for the GPU-utilization mechanics so both agents stay in sync:
  * env-driven update config (UTD ratio, batch override, round-end updates)
  * GPU duty-cycle timer (torch.cuda.synchronize around measured sections)
  * LR-schedule application (stateless: pure function of total_steps)

Import pattern (SequentialAgentBackend chdirs into the agent dir)::

    try:
        from dml_trainkit import update_config, DutyTimer, apply_schedule
    except ImportError:
        import sys, os
        _root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from dml_trainkit import update_config, DutyTimer, apply_schedule

Env (PREFIX = SENTINEL / OVERLORD):
  {PREFIX}_UTD          updates per env step (default 1 = legacy behavior)
  {PREFIX}_BATCH        batch override (default 0 = agent constant)
  {PREFIX}_EOR_UPDATES  extra updates at round end (default = agent constant)
  {PREFIX}_DEVICE       cuda|cpu|auto (default auto = CUDA when available)
  {PREFIX}_AMP          1/0 (default 1 = autocast+GradScaler on CUDA)
All defaults reproduce pre-enhancement behavior exactly, except the
device default is now CUDA (was DirectML) and AMP is on for CUDA.
"""
import csv
import glob
import os
import time


def update_config(prefix, default_batch, default_eor):
    """(utd, batch, eor_updates) from env, validated to sane ranges."""
    def _int(name, default, lo, hi):
        try:
            v = int(os.environ.get(name, str(default)))
        except ValueError:
            v = default
        return max(lo, min(hi, v))

    utd = _int(f'{prefix}_UTD', 1, 1, 8)
    batch = _int(f'{prefix}_BATCH', 0, 0, 4096)
    batch = batch or default_batch
    eor = _int(f'{prefix}_EOR_UPDATES', default_eor, 0, 128)
    return utd, batch, eor


class DutyTimer:
    """Accumulates synchronized GPU-busy ms vs wall ms per round.

    Usage: with timer.measure(device): ... synchronized op ...
    or legacy: with timer.measure(): ... (auto-syncs CUDA if available).
    Overhead: two perf_counter calls + one cuda synchronize per update
    (the .item() sync is already paid by loss logging).
    """

    def __init__(self):
        self.gpu_ms = 0.0
        self.round_t0 = time.perf_counter()

    def measure(self, device=None):
        """Context manager timing GPU-busy section (syncs CUDA inside)."""
        return self._Section(self, device)

    class _Section:
        def __init__(self, timer, device=None):
            self.timer = timer
            self.device = device

        def __enter__(self):
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                pass
            self.t0 = time.perf_counter()
            return self

        def __exit__(self, *exc):
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                pass
            self.timer.gpu_ms += (time.perf_counter() - self.t0) * 1000.0
            return False

    def end_round(self):
        """Return (gpu_ms, wall_ms, duty_pct); resets accumulators."""
        wall_ms = (time.perf_counter() - self.round_t0) * 1000.0
        gpu_ms, self.gpu_ms = self.gpu_ms, 0.0
        self.round_t0 = time.perf_counter()
        duty = (100.0 * gpu_ms / wall_ms) if wall_ms > 0 else 0.0
        return round(gpu_ms, 1), round(wall_ms, 1), round(duty, 1)


def apply_schedule(optimizer, base_lr, total_steps,
                   warmup=5000, total=200000, min_ratio=0.1):
    """Set group LRs from the stateless warmup+cosine schedule.

    Returns the applied LR. Pure function of total_steps (already
    checkpointed) -> resume-safe, no extra state.
    """
    try:
        from dml_optimizer import schedule_factor
    except ImportError:
        import sys
        _root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from dml_optimizer import schedule_factor
    lr_now = base_lr * schedule_factor(
        total_steps, warmup=warmup, total=total, min_ratio=min_ratio)
    for g in optimizer.param_groups:
        g['lr'] = lr_now
    return lr_now


__all__ = ["update_config", "DutyTimer", "apply_schedule",
           "get_dml_device", "get_cuda_device", "get_device",
           "is_dml_device", "is_cuda_device", "CheckpointStore",
           "log_metrics_row", "append_metrics_row", "save_every_config"]


def is_cuda_device(device) -> bool:
    """True if ``device`` is a CUDA device."""
    try:
        t = getattr(device, 'type', device)
        return str(t).lower() == 'cuda'
    except Exception:
        return False


def is_dml_device(device) -> bool:
    """Legacy shim: DirectML removed — True for any GPU (CUDA) device.

    Kept so old call sites that branched on "GPU vs CPU" (e.g. overlord
    full-batch vs CPU-batch) keep working: CUDA counts as GPU.
    """
    return is_cuda_device(device)


def _want_cuda(env_flag, default_on=True):
    """Parse device want from env. Supports NEW {PREFIX}_DEVICE and legacy flags."""
    # New canonical: {PREFIX}_DEVICE=cuda|cpu|auto
    prefix = env_flag[:-len('_DML')] if env_flag.endswith('_DML') else env_flag
    dev_key = f'{prefix}_DEVICE'
    raw = os.environ.get(dev_key, '').strip().lower()
    if raw == 'cpu':
        return False
    if raw == 'cuda':
        return True
    # Legacy: *_DML=0, *_CUDA=0 force CPU; default tries CUDA.
    default = '1' if default_on else '0'
    if os.environ.get(env_flag, default) == '0':
        return False
    cuda_flag = f'{prefix}_CUDA'
    if cuda_flag in os.environ and os.environ.get(cuda_flag) == '0':
        return False
    return True


def get_cuda_device(env_flag, logger=None, default_on=True):
    """CUDA device or CPU. Never raises; Colab/tournament-safe.

    env_flag e.g. 'OVERLORD_DML' (legacy) or 'OVERLORD_DEVICE' via prefix:
    '{PREFIX}_DEVICE=cpu' forces CPU; default (auto) tries CUDA with a
    tiny probe op + synchronize.
    """
    try:
        import torch
    except Exception as ex:
        if logger:
            logger.warning(f'torch import failed, CPU fallback: {ex}')
        return 'cpu'
    if not _want_cuda(env_flag, default_on):
        return torch.device('cpu')
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('torch.cuda.is_available() is False')
        dev = torch.device('cuda')
        probe = torch.zeros(8, device=dev)
        _ = (probe + 1).sum().item()
        torch.cuda.synchronize()
        del probe
        if logger:
            logger.info(f'device=cuda ({torch.cuda.get_device_name(0)}) via {env_flag}')
        return dev
    except Exception as ex:
        if logger:
            logger.warning(f'CUDA unavailable, CPU fallback: {ex}')
        try:
            return torch.device('cpu')
        except Exception:
            return 'cpu'


def get_device(env_flag, logger=None, default_on=True):
    """Canonical alias for :func:`get_cuda_device`."""
    return get_cuda_device(env_flag, logger, default_on)


def get_dml_device(env_flag, logger=None, default_on=True):
    """Legacy alias for :func:`get_cuda_device` (DirectML removed)."""
    return get_cuda_device(env_flag, logger, default_on)


def amp_enabled(prefix, default_on=True):
    """True when AMP autocast+GradScaler should be used for PREFIX.

    Default ON for Colab CUDA; ``{PREFIX}_AMP=0`` forces fp32.
    Returns False on CPU-only hosts or when torch AMP is unavailable.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        if not hasattr(torch, 'autocast'):
            return False
    except Exception:
        return False
    default = '1' if default_on else '0'
    return os.environ.get(f'{prefix}_AMP', default) != '0'


class CheckpointStore:
    """Atomic versioned checkpoints for one agent.

    last.pt (every save), best.pt (caller decides), ep_NNNNNN.pt every
    snapshot_every (pruned to keep_snapshots). Payload schema matches
    sentinel's checkpointing.py: q_net/target_net/optimizer/episode/
    total_steps + extras. Never raises on save (logs via logger or pass).
    """

    def __init__(self, agent_dir, save_every=1, snapshot_every=200,
                 keep_snapshots=5, logger=None):
        self.dir = os.path.join(agent_dir, 'checkpoints')
        os.makedirs(self.dir, exist_ok=True)
        self.save_every = save_every
        self.snapshot_every = snapshot_every
        self.keep_snapshots = keep_snapshots
        self.logger = logger

    def _atomic(self, path, payload):
        import torch
        tmp = path + '.tmp'
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def save_last(self, payload):
        try:
            self._atomic(os.path.join(self.dir, 'last.pt'), payload)
        except Exception as ex:
            if self.logger:
                self.logger.warning(f'checkpoint save failed: {ex}')

    def save_best(self, payload):
        try:
            self._atomic(os.path.join(self.dir, 'best.pt'), payload)
        except Exception as ex:
            if self.logger:
                self.logger.warning(f'best save failed: {ex}')

    def save_snapshot(self, episode, payload):
        try:
            self._atomic(os.path.join(self.dir, f'ep_{episode:06d}.pt'), payload)
            snaps = sorted(glob.glob(os.path.join(self.dir, 'ep_*.pt')))
            for old in snaps[:-self.keep_snapshots]:
                try:
                    os.remove(old)
                except OSError:
                    pass
        except Exception as ex:
            if self.logger:
                self.logger.warning(f'snapshot save failed: {ex}')

    def maybe_save(self, episode, payload, improved=False):
        if episode % max(1, self.save_every) == 0:
            self.save_last(payload)
        if improved:
            self.save_best(payload)
        if episode % max(1, self.snapshot_every) == 0:
            self.save_snapshot(episode, payload)

    def load(self, name='last.pt', map_location='cpu'):
        try:
            import torch
            path = os.path.join(self.dir, name)
            if not os.path.isfile(path):
                return None
            return torch.load(path, map_location=map_location, weights_only=False)
        except Exception:
            return None


def log_metrics_row(path, row):
    """Append row to CSV; widens header when new columns appear."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    if not os.path.isfile(path):
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)
        return
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        old_fields = list(reader.fieldnames or [])
        old_rows = list(reader)
    fields = old_fields + [k for k in row.keys() if k not in old_fields]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(old_rows)
        w.writerow({k: row.get(k, '') for k in fields})


def append_metrics_row(path, row):
    """Fast append-only path for hot training loops (O(1) per round).

    If the file exists and its header already covers ``row`` keys, appends
    without rewriting (no read-back). Otherwise falls back to
    :func:`log_metrics_row` (header widen). Never raises.
    """
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        if not os.path.isfile(path):
            return log_metrics_row(path, row)
        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            old_fields = list(reader.fieldnames or [])
        if all(k in old_fields for k in row.keys()):
            with open(path, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=old_fields)
                w.writerow({k: row.get(k, '') for k in old_fields})
            return
        return log_metrics_row(path, row)
    except Exception:
        try:
            return log_metrics_row(path, row)
        except Exception:
            return


def save_every_config(prefix, default=5):
    """Rounds between full ``last.pt`` saves (env ``{PREFIX}_SAVE_EVERY``)."""
    try:
        v = int(os.environ.get(f'{prefix}_SAVE_EVERY', str(default)))
    except ValueError:
        v = default
    return max(1, min(100, v))

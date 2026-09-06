"""Optimizer benchmark: speed + convergence proxy (offline-safe).

--micro : synchronized median optimizer-step time per candidate
          (real sentinel DuelingMLP, batch 256, CUDA when available).
--proxy : fixed-seed optimization trajectories on a TD-scale regression
          task (targets x10 to mimic TD-error magnitudes); compares final
          loss + loss AUC across candidates. Same init + data order.
Live 50-round game validation runs at Stage-4 adoption (not here: live
runs would clobber the running Stage-3 checkpoints/metrics).
Usage: python3 scripts/bench_optim.py [--micro] [--proxy]
"""
import statistics
import sys
import time

import torch

CANDIDATES = ['adam', 'adam_tuned', 'lion']


def _device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def build(name, params, schedule=False):
    sys.path.insert(0, '.')
    from dml_optimizer import build_optimizer_for_device, schedule_factor
    dev = _device()
    if name == 'adam':
        opt = build_optimizer_for_device(dev, params, lr=1e-3)
        base = 1e-3
    elif name == 'adam_tuned':
        opt = build_optimizer_for_device(
            dev, params, lr=1e-3, tuned=True,
            lookahead={'alpha': 0.5, 'k': 5})
        base = 1e-3
    elif name == 'lion':
        opt = build_optimizer_for_device(dev, params, lr=3e-4, name='lion',
                                         betas=(0.9, 0.99))
        base = 3e-4
    else:
        raise ValueError(name)
    sched = None
    if schedule:
        sched = lambda step: base * schedule_factor(step)  # noqa: E731
    return opt, sched


def sync(t):
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
    return float(t.detach().cpu().sum())


def micro(steps=60, warmup=10, batch=256):
    sys.path.insert(0, '.')
    sys.path.insert(0, 'agent_code/sentinel')
    from model import build_model, FEATURE_DIM
    dev = _device()
    print(f'{len(CANDIDATES)} candidates, batch={batch}, device={dev}')
    rows = []
    for name in CANDIDATES:
        m = build_model().to(dev)
        from dml_optimizer import build_optimizer_for_device as _b
        opt = _b(dev, m.parameters(), **_kw(name))
        import numpy as np
        Xn = np.random.randn(batch, FEATURE_DIM).astype('float32')
        ts = []
        for i in range(steps + warmup):
            S = torch.from_numpy(Xn).to(dev)
            loss = m(S).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            t0 = time.perf_counter()
            opt.step()
            sync(m.trunk[0].weight)
            dt = (time.perf_counter() - t0) * 1e3
            if i >= warmup:
                ts.append(dt)
        med = statistics.median(ts)
        rows.append((name, med, 1000.0 / med))
        print(f'{name:12s} step {med:6.2f} ms  ({1000.0 / med:6.1f} steps/s)')
    base = rows[0][1]
    for name, med, _ in rows:
        print(f'{name:12s} vs adam: {base / med:.2f}x')
    return rows


def _kw(name):
    if name == 'adam':
        return {'lr': 1e-3}
    if name == 'adam_tuned':
        return {'lr': 1e-3, 'tuned': True,
                'lookahead': {'alpha': 0.5, 'k': 5}}
    if name == 'lion':
        return {'lr': 3e-4, 'name': 'lion', 'betas': (0.9, 0.99)}
    raise ValueError(name)


def proxy(steps=400, batch=256, seed=7, schedule=True):
    """Ill-conditioned noisy regression (condition ~1e4, TD-scale noise).

    Same init + same data order per candidate. Separates adaptivity:
    plain SGD-style dynamics stall in the valley; good adaptive methods
    don't. Final loss + mean loss decide.
    """
    sys.path.insert(0, '.')
    import numpy as np
    dev = _device()
    rng = np.random.RandomState(seed)
    U, _ = np.linalg.qr(rng.randn(batch, batch))
    V, _ = np.linalg.qr(rng.randn(46, 46))
    spectrum = np.logspace(-2, 2, 46).astype('float32')
    X = ((U[:, :46] * spectrum) @ V).astype('float32')
    Wtrue = rng.randn(46, 6).astype('float32')
    Y = (X @ Wtrue + 3.0 * rng.randn(batch, 6)).astype('float32')
    Xt = torch.from_numpy(X)
    Yt = torch.from_numpy(Y)
    results = {}
    for name in CANDIDATES:
        torch.manual_seed(1000 + seed)
        m = torch.nn.Linear(46, 6).to(dev)
        from dml_optimizer import build_optimizer_for_device as _b
        from dml_optimizer import schedule_factor
        opt = _b(dev, m.parameters(), **_kw(name))
        base_lr = {'adam': 1e-3, 'adam_tuned': 1e-3, 'lion': 3e-4}[name]
        torch.manual_seed(seed)
        order = torch.randperm(batch, device='cpu')
        losses = []
        for i in range(steps):
            idx = order[(i * 64) % batch:((i * 64) % batch) + 64]
            x = Xt[idx].to(dev)
            y = Yt[idx].to(dev)
            if schedule:
                lr = base_lr * schedule_factor(i, warmup=50, total=steps)
                for g in opt.param_groups:
                    g['lr'] = lr
            pred = m(x)
            loss = ((pred - y) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
            with torch.no_grad():
                full = ((m(Xt.to(dev)) - Yt.to(dev)) ** 2).mean()
            losses.append(float(full.detach().cpu()))
        auc = sum(losses) / len(losses)
        results[name] = (losses[-1], auc)
        print(f'{name:12s} final loss {losses[-1]:10.2f}  mean loss {auc:10.2f}')
    return results


if __name__ == '__main__':
    if '--micro' in sys.argv or len(sys.argv) == 1:
        micro()
    if '--proxy' in sys.argv or len(sys.argv) == 1:
        proxy()

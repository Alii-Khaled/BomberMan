"""Shared trajectory helper for optimizer gates/benchmarks (importable)."""
import torch


def run_traj(opt_cls, device, steps=20, seed=0, clip=5.0, batch=32, **kw):
    torch.manual_seed(seed)
    m = torch.nn.Sequential(torch.nn.Linear(46, 256), torch.nn.ReLU(),
                            torch.nn.Linear(256, 6)).to(device)
    opt = opt_cls(m.parameters(), **kw)
    torch.manual_seed(1234)
    losses = []
    for _ in range(steps):
        x = torch.randn(batch, 46, device=device)
        loss = m(x).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(m.parameters(), clip)
        opt.step()
        losses.append(float(loss.detach().cpu()))
    flat = torch.cat([p.detach().cpu().flatten() for p in m.parameters()])
    return losses, flat

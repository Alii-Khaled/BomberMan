"""Overlord CNN probe on CUDA (L40S path): fwd/bwd, channels-last, AMP, save/load.

Usage: python3 scripts/probe_overlord_cuda.py
Covers the upgraded OverlordNet defaults (base-96/fc-512/BN) + GN + deep
variants, mirroring test_cuda.py for sentinel. Never touches real checkpoints.
"""
import os
import sys

sys.path.insert(0, 'agent_code/overlord')
sys.path.insert(0, '.')


def main() -> int:
    import torch
    from model import build_model, N_CHANNELS

    print(f'torch: {torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}')
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dev.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print(f'device: cuda ({torch.cuda.get_device_name(0)})')
    else:
        print('device: cpu (CUDA unavailable — probe degrades gracefully)')

    for tag, kw in [('default', {}), ('gn', {'norm': 'gn'}), ('deep', {'deep': True})]:
        m = build_model(**kw).to(dev)
        if dev.type == 'cuda' and os.environ.get('OVERLORD_CHANNELS_LAST', '1') == '1':
            m = m.to(memory_format=torch.channels_last)
        m.train()
        n = sum(p.numel() for p in m.parameters())
        x = torch.randn(32, N_CHANNELS, 17, 17, device=dev)
        if dev.type == 'cuda':
            x = x.to(memory_format=torch.channels_last)
        s = torch.randn(32, 8, device=dev)
        use_amp = dev.type == 'cuda'
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4, foreach=True)
        if use_amp:
            scaler = torch.amp.GradScaler('cuda')
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                q, aux = m(x, s)
                loss = q.pow(2).mean() + 0.1 * aux.pow(2).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            q, aux = m(x, s)
            loss = q.pow(2).mean() + 0.1 * aux.pow(2).mean()
            loss.backward()
            opt.step()
        # aux head alive + dueling zero-init sanity (initial Q ~ 0)
        with torch.no_grad():
            m.eval()
            q0, _ = m(x[:4], s[:4])
            qmax = q0.abs().max().item()
        print(f'[{tag}] params={n} fwd/bwd AMP={use_amp} OK loss={loss.detach().cpu().item():.4f} init|Q|max={qmax:.4f}')
        del m, x, s, opt

    # checkpoint round-trip (temp dir only)
    import tempfile
    sys.path.insert(0, 'agent_code/sentinel')
    from checkpointing import _atomic_save, load_checkpoint
    m = build_model().cpu()
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

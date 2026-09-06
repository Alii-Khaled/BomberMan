"""Atomic, resumable checkpoints for sentinel (light: no replay buffer).

Layout (all under agent_code/sentinel/checkpoints/):
  last.pt   - full training state, overwritten every SAVE_EVERY rounds
  best.pt   - copy of best EMA-reward state so far
  ep_NNNNNN.pt - versioned snapshot every SNAPSHOT_EVERY rounds (pruned)

Each file stores: q_net, target_net, optimizer, episode, total_steps,
epsilon_steps, best_ema, config. Tournament export my-saved-model.pt
(state_dict only) is written separately and is NOT the source of truth.
"""
import glob
import os

SAVE_EVERY = 25
SNAPSHOT_EVERY = 200
KEEP_SNAPSHOTS = 5


def checkpoint_dir(here=None):
    base = here or os.path.dirname(os.path.abspath(__file__))
    d = os.path.join(base, 'checkpoints')
    os.makedirs(d, exist_ok=True)
    return d


def _atomic_save(path, payload):
    import torch
    tmp = path + '.tmp'
    torch.save(payload, tmp)
    os.replace(tmp, path)


def save_last(here, payload):
    _atomic_save(os.path.join(checkpoint_dir(here), 'last.pt'), payload)


def save_best(here, payload):
    _atomic_save(os.path.join(checkpoint_dir(here), 'best.pt'), payload)


def save_snapshot(here, episode, payload):
    d = checkpoint_dir(here)
    _atomic_save(os.path.join(d, f'ep_{episode:06d}.pt'), payload)
    snaps = sorted(glob.glob(os.path.join(d, 'ep_*.pt')))
    for old in snaps[:-KEEP_SNAPSHOTS]:
        try:
            os.remove(old)
        except OSError:
            pass


def load_checkpoint(path, map_location='cpu'):
    """Return payload dict or None. Never raises."""
    try:
        import torch
        if not path or not os.path.isfile(path):
            return None
        return torch.load(path, map_location=map_location, weights_only=False)
    except Exception:
        return None


def last_path(here=None):
    return os.path.join(checkpoint_dir(here), 'last.pt')


def best_path(here=None):
    return os.path.join(checkpoint_dir(here), 'best.pt')

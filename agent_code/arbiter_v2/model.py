"""ARBITER policy/value net: shared MLP trunk over the 98-dim engineered
features, with a policy prior head pi (6 logits) and a state-value head
V (1 scalar, score-margin-to-go).

Design constraint from A4 (E62): the net NEVER votes on root actions
directly. pi prunes the search and is the S0 fallback; V evaluates
search leaves only. This is structural immunity to the apex failure
(E61 Q-delta -0.42 from a dueling Q given veto authority over an
equivalent heuristic).

Convention (repo-wide): zero-init heads, so pi starts uniform and V
starts at 0 — the heuristic skeleton drives early play.
"""
import os
import torch
import torch.nn as nn

try:
    from .features import FEATURE_DIM
except ImportError:  # direct execution fallback
    from features import FEATURE_DIM

N_ACTIONS = 6
ACTION_LIST = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def _env_int(name, default, lo, hi):
    try:
        v = int(os.environ.get(name, str(default)))
    except ValueError:
        v = default
    return max(lo, min(hi, v))


class ArbiterNet(nn.Module):
    def __init__(self, in_dim=FEATURE_DIM, hid1=256, hid2=256, hid3=256,
                 n_actions=N_ACTIONS):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hid1), nn.ReLU(inplace=True),
            nn.Linear(hid1, hid2), nn.ReLU(inplace=True),
            nn.Linear(hid2, hid3), nn.ReLU(inplace=True),
        )
        self.pi = nn.Linear(hid3, n_actions)
        self.v = nn.Linear(hid3, 1)
        for m in (self.pi, self.v):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.trunk(x)
        return self.pi(h), self.v(h).squeeze(-1)


def build_model(in_dim=FEATURE_DIM, hid1=None, hid2=None, hid3=None):
    """Env-overridable constructor (ARBITER_HID1/2/3)."""
    if hid1 is None:
        hid1 = _env_int('ARBITER_HID1', 256, 64, 2048)
    if hid2 is None:
        hid2 = _env_int('ARBITER_HID2', 256, 32, 1024)
    if hid3 is None:
        hid3 = _env_int('ARBITER_HID3', 256, 32, 1024)
    return ArbiterNet(in_dim=in_dim, hid1=hid1, hid2=hid2, hid3=hid3)

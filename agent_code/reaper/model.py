"""Reaper: small dueling MLP over engineered features.

~400K params, CPU-forward <1 ms. Zero-init value/advantage heads so Q
starts near zero (behavior cloned / heuristically seeded policy dominates
early, learned Q grows from scratch — same stabilization as overlord).
"""
import os
import torch
import torch.nn as nn

from .features import FEATURE_DIM

N_ACTIONS = 6
ACTION_LIST = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def _env_int(name, default, lo, hi):
    try:
        v = int(os.environ.get(name, str(default)))
    except ValueError:
        v = default
    return max(lo, min(hi, v))


class ReaperNet(nn.Module):
    def __init__(self, in_dim=FEATURE_DIM, hid1=512, hid2=256, hid3=256,
                 n_actions=N_ACTIONS):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hid1), nn.ReLU(inplace=True),
            nn.Linear(hid1, hid2), nn.ReLU(inplace=True),
            nn.Linear(hid2, hid3), nn.ReLU(inplace=True),
        )
        self.v = nn.Linear(hid3, 1)
        self.adv = nn.Linear(hid3, n_actions)
        for m in (self.v, self.adv):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.trunk(x)
        v = self.v(h)
        a = self.adv(h)
        return v + a - a.mean(dim=-1, keepdim=True)


def build_model(in_dim=FEATURE_DIM, hid1=None, hid2=None, hid3=None):
    """Env-overridable constructor (REAPER_HID1/HID2/HID3)."""
    if hid1 is None:
        hid1 = _env_int('REAPER_HID1', 512, 64, 2048)
    if hid2 is None:
        hid2 = _env_int('REAPER_HID2', 256, 32, 1024)
    if hid3 is None:
        hid3 = _env_int('REAPER_HID3', 256, 32, 1024)
    return ReaperNet(in_dim=in_dim, hid1=hid1, hid2=hid2, hid3=hid3)

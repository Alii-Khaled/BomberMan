"""Harvey model: a 12-channel board CNN fused with a 98-value scalar MLP.

The flat input contains the raveled board tensor followed by scalars. The
network has policy pi(6) and value V(1) heads plus a training-only margin
head. Zero-initialized heads let the safety skeleton guide early training.
"""
import os

import torch
import torch.nn as nn

try:
    from .features import FEATURE_DIM, SCALAR_DIM, N_CHANNELS
except ImportError:  # direct execution fallback
    from features import FEATURE_DIM, SCALAR_DIM, N_CHANNELS

N_ACTIONS = 6
ACTION_LIST = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
TENSOR_DIM = N_CHANNELS * 17 * 17


def _env_int(name, default, lo, hi):
    try:
        v = int(os.environ.get(name, str(default)))
    except ValueError:
        v = default
    return max(lo, min(hi, v))


class HarveyNet(nn.Module):
    def __init__(self, in_dim=FEATURE_DIM, ch=32, ch2=64, chead=8,
                 hid=256, n_actions=N_ACTIONS):
        super().__init__()
        self.tensor_dim = TENSOR_DIM
        self.cnn = nn.Sequential(
            nn.Conv2d(N_CHANNELS, ch, 3, padding=1),
            nn.GroupNorm(4, ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(4, ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch2, 3, padding=1),
            nn.GroupNorm(8, ch2), nn.ReLU(inplace=True),
            nn.Conv2d(ch2, chead, 1),
            nn.GroupNorm(4, chead), nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(chead * 17 * 17, hid), nn.ReLU(inplace=True),
        )
        self.mlp = nn.Sequential(
            nn.Linear(SCALAR_DIM, hid), nn.ReLU(inplace=True),
            nn.Linear(hid, hid), nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(
            nn.Linear(2 * hid, hid), nn.ReLU(inplace=True),
        )
        self.pi = nn.Linear(hid, n_actions)
        self.v = nn.Linear(hid, 1)
        self.aux = nn.Linear(hid, 1)
        for m in (self.pi, self.v, self.aux):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def _fuse(self, x):
        t = x[:, :self.tensor_dim].reshape(-1, N_CHANNELS, 17, 17)
        s = x[:, self.tensor_dim:]
        return self.trunk(torch.cat([self.cnn(t), self.mlp(s)], dim=-1))

    def forward(self, x):
        h = self._fuse(x)
        return self.pi(h), self.v(h).squeeze(-1)

    def forward_all(self, x):
        h = self._fuse(x)
        return self.pi(h), self.v(h).squeeze(-1), self.aux(h).squeeze(-1)


def build_model(in_dim=FEATURE_DIM, hid1=None, hid2=None, hid3=None):
    """Constructor with env overrides (HARVEY_CH/CH2/CHEAD/HID).

    hid1/2/3 are accepted so the pretraining scripts can call this the
    same way as the earlier builder (they pass HARVEY_HID*); the trunk
    uses HID for the fused width."""
    ch = _env_int('HARVEY_CH', 32, 8, 128)
    ch2 = _env_int('HARVEY_CH2', 64, 8, 256)
    chead = _env_int('HARVEY_CHEAD', 8, 2, 64)
    hid = _env_int('HARVEY_TRUNK_HID', 256, 64, 1024)
    return HarveyNet(in_dim=in_dim, ch=ch, ch2=ch2, chead=chead, hid=hid)

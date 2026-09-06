"""Overlord CNN: spatial encoder + dueling head + aux danger predictor."""
import torch
import torch.nn as nn

N_CHANNELS = 12
N_ACTIONS = 6
ACTION_LIST = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        r = x
        x = self.relu(self.b1(self.c1(x)))
        x = self.b2(self.c2(x))
        return self.relu(x + r)


class OverlordNet(nn.Module):
    def __init__(self, in_ch=N_CHANNELS, base=64, n_actions=N_ACTIONS):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base), nn.ReLU(inplace=True),
        )
        self.res1 = ResBlock(base)
        self.down = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base * 2), nn.ReLU(inplace=True),
        )
        self.res2 = ResBlock(base * 2)
        # global + local pooling; amax is used instead of AdaptiveMaxPool2d
        # (CUDA-native, verified fast on Colab GPUs).
        self.gap = nn.AdaptiveAvgPool2d(1)
        feat = base * 2 * 2 + 8  # gap+gmp + small scalar head input
        # scalar head: step, bombs_left, crates, coins, opps, escape, crates_hit, opps_hit
        self.fc = nn.Sequential(nn.Linear(feat, 256), nn.ReLU(inplace=True))
        self.v = nn.Linear(256, 1)
        self.adv = nn.Linear(256, n_actions)
        # aux: predict danger_t1 mean (dense gradient)
        self.aux = nn.Linear(256, 1)

    def forward(self, img, scalars=None):
        # img: [B,C,X,Y] with X=Y=17
        x = self.stem(img)
        x = self.res1(x)
        x = self.down(x)   # 9x9
        x = self.res2(x)
        g1 = self.gap(x).flatten(1)
        g2 = x.amax(dim=(2, 3))
        if scalars is None:
            scalars = torch.zeros((img.shape[0], 8), device=img.device, dtype=img.dtype)
        h = torch.cat([g1, g2, scalars], dim=1)
        h = self.fc(h)
        v = self.v(h)
        a = self.adv(h)
        q = v + a - a.mean(dim=-1, keepdim=True)
        aux = self.aux(h).squeeze(-1)
        return q, aux


def build_model():
    return OverlordNet()


def scalars_from_state(game_state, safety_info=None):
    import numpy as np
    if game_state is None:
        return np.zeros(8, dtype=np.float32)
    arena = game_state['field']
    _, _, bombs_left, _ = game_state['self']
    step = float(game_state.get('step', 0))
    coins = game_state.get('coins', []) or []
    others = game_state.get('others', []) or []
    return np.array([
        min(step, 400) / 400,
        1.0 if bombs_left else 0.0,
        min(float((arena == 1).sum()), 100) / 100,
        min(float(len(coins)), 20) / 20,
        min(float(len(others)), 3) / 3,
        1.0 if (safety_info or {}).get('can_escape_if_bomb') else 0.0,
        min(float((safety_info or {}).get('crates_hit_if_bomb', 0)), 4) / 4,
        min(float((safety_info or {}).get('opps_hit_if_bomb', 0)), 2) / 2,
    ], dtype=np.float32)

"""Apex CNN: spatial encoder + dueling head + aux danger predictor."""
import os
import torch
import torch.nn as nn

N_CHANNELS = 12
N_SCALARS = 16
N_ACTIONS = 6
ACTION_LIST = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def _env_int(name, default, lo, hi):
    try:
        v = int(os.environ.get(name, str(default)))
    except ValueError:
        v = default
    return max(lo, min(hi, v))


def _env_str(name, default):
    return os.environ.get(name, default).strip().lower() or default


class ResBlock(nn.Module):
    def __init__(self, c, norm='bn', groups=16):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        if norm == 'gn':
            g = min(groups, c)
            while c % g != 0 and g > 1:
                g //= 2
            self.b1 = nn.GroupNorm(g, c)
            self.b2 = nn.GroupNorm(g, c)
        else:
            self.b1 = nn.BatchNorm2d(c)
            self.b2 = nn.BatchNorm2d(c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        r = x
        x = self.relu(self.b1(self.c1(x)))
        x = self.b2(self.c2(x))
        return self.relu(x + r)


def _norm_layer(c, norm='bn', groups=16):
    if norm == 'gn':
        g = min(groups, c)
        while c % g != 0 and g > 1:
            g //= 2
        return nn.GroupNorm(g, c)
    return nn.BatchNorm2d(c)


class ApexNet(nn.Module):
    def __init__(self, in_ch=N_CHANNELS, base=96, n_actions=N_ACTIONS,
                 fc_dim=512, norm='bn', deep=False, groups=16):
        super().__init__()
        self._norm = norm
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1, bias=False),
            _norm_layer(base, norm, groups), nn.ReLU(inplace=True),
        )
        self.res1 = ResBlock(base, norm, groups)
        self.down = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1, bias=False),
            _norm_layer(base * 2, norm, groups), nn.ReLU(inplace=True),
        )
        self.res2 = ResBlock(base * 2, norm, groups)
        # Optional 3rd block at full depth (APEX_DEEP=1). Adds ~0.6M
        # params at base=96; gated so default shape stays loadable.
        self.res3 = ResBlock(base * 2, norm, groups) if deep else None
        # global + local pooling; amax is used instead of AdaptiveMaxPool2d
        # (CUDA-native, verified fast on Colab GPUs).
        self.gap = nn.AdaptiveAvgPool2d(1)
        feat = base * 2 * 2 + N_SCALARS  # gap+gmp + extended scalar head (8 overlord + 8 apex extras)
        # scalar head: step, bombs_left, crates, coins, opps, escape, crates_hit, opps_hit
        self.fc = nn.Sequential(nn.Linear(feat, fc_dim), nn.ReLU(inplace=True))
        self.v = nn.Linear(fc_dim, 1)
        self.adv = nn.Linear(fc_dim, n_actions)
        # aux: predict danger_t1 mean (dense gradient)
        self.aux = nn.Linear(fc_dim, 1)
        # Dueling stabilization: zero-init both heads so Q starts as ~0
        # (heuristic prior drives early play; learned Q grows from scratch).
        nn.init.zeros_(self.v.weight)
        nn.init.zeros_(self.v.bias)
        nn.init.zeros_(self.adv.weight)
        nn.init.zeros_(self.adv.bias)

    def forward(self, img, scalars=None):
        # img: [B,C,X,Y] with X=Y=17
        if img.dim() == 4 and img.is_contiguous(memory_format=torch.contiguous_format):
            # allow channels_last inputs without forcing a copy here;
            # convs handle either format (autotuner picks fastest).
            pass
        x = self.stem(img)
        x = self.res1(x)
        x = self.down(x)   # 9x9
        x = self.res2(x)
        if self.res3 is not None:
            x = self.res3(x)
        g1 = self.gap(x).flatten(1)
        g2 = x.amax(dim=(2, 3))
        if scalars is None:
            scalars = torch.zeros((img.shape[0], N_SCALARS), device=img.device, dtype=img.dtype)
        h = torch.cat([g1, g2, scalars], dim=1)
        h = self.fc(h)
        v = self.v(h)
        a = self.adv(h)
        q = v + a - a.mean(dim=-1, keepdim=True)
        aux = self.aux(h).squeeze(-1)
        return q, aux


def build_model(in_ch=N_CHANNELS, base=None, n_actions=N_ACTIONS,
                fc_dim=None, norm=None, deep=None):
    """Env-overridable constructor (all args optional, backward compatible).

    APEX_BASE (default 96, was 64) · APEX_FC (default 512, was 256)
    APEX_NORM=bn|gn (default bn) · APEX_DEEP=1 adds res3.
    Old checkpoints with base=64 load with strict=False (shape-mismatched
    layers re-init) — fresh-init runs are unaffected.
    """
    if base is None:
        base = _env_int('APEX_BASE', 96, 32, 256)
    if fc_dim is None:
        fc_dim = _env_int('APEX_FC', 512, 128, 2048)
    if norm is None:
        norm = _env_str('APEX_NORM', 'bn')
        if norm not in ('bn', 'gn'):
            norm = 'bn'
    if deep is None:
        deep = os.environ.get('APEX_DEEP', '0') == '1'
    return ApexNet(in_ch=in_ch, base=base, n_actions=n_actions,
                       fc_dim=fc_dim, norm=norm, deep=deep)


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

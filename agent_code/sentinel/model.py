"""Dueling MLP for sentinel. CPU-optimized, single-thread friendly."""
import torch
import torch.nn as nn

FEATURE_DIM = 46
N_ACTIONS = 6
ACTION_LIST = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


class DuelingMLP(nn.Module):
    def __init__(self, in_dim=FEATURE_DIM, hidden=256, n_actions=N_ACTIONS):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.v = nn.Linear(hidden, 1)
        self.adv = nn.Linear(hidden, n_actions)

    def forward(self, x):
        h = self.trunk(x)
        v = self.v(h)
        a = self.adv(h)
        return v + a - a.mean(dim=-1, keepdim=True)


def build_model(in_dim=FEATURE_DIM, hidden=256):
    return DuelingMLP(in_dim, hidden)

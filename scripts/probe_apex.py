"""E49 apex scaffold probes (static parts; callbacks/train next).

Groups (fail loudly; never touch training/checkpoints):
 1. shapes: ApexNet fwd (1,12,17,17)+(1,16) -> Q(1,6) + aux(1,).
 2. zero-init: fresh |Q|max tiny (heuristic drives early play).
 3. extras: determinism + finiteness + bounds on random states.
 4. extras rotation-invariance (aggregate-only design claim).
 5. safety parity: apex action_safety == overlord action_safety.
 6. trunk transfer: o3sbest loads with ONLY fc/v/aux mismatched.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from agent_code.apex.model import build_model as build_apex, N_SCALARS
from agent_code.apex.features_extra import extra_scalars


def check(name, cond):
    print(('PASS' if cond else 'FAIL') + ': ' + name)
    if not cond:
        raise SystemExit('probe failed: ' + name)


print('N_SCALARS =', N_SCALARS)
check('scalars dim 16', N_SCALARS == 16)

# 1+2. shapes + zero-init
m = build_apex()
m.eval()
with torch.no_grad():
    q, aux = m(torch.randn(1, 12, 17, 17), torch.randn(1, 16))
check('Q shape (1,6)', tuple(q.shape) == (1, 6))
check('aux shape (1,)', tuple(aux.shape) == (1,))
check('zero-init |Q|max<0.05', float(q.abs().max()) < 0.05)


def rand_state(rng):
    arena = np.zeros((17, 17), dtype=int)
    arena[0, :] = arena[-1, :] = arena[:, 0] = arena[:, -1] = -1
    arena[1:-1, 1:-1][rng.random((15, 15)) < 0.3] = 1
    free = [tuple(p) for p in zip(*np.where(arena == 0))]
    pos = free[rng.integers(len(free))]
    bombs = [((int(free[rng.integers(len(free))][0]),
               int(free[rng.integers(len(free))][1])), int(rng.integers(0, 5)))
             for _ in range(int(rng.integers(0, 3)))]
    others = [('r%d' % j, int(rng.integers(0, 9)), True,
               (int(free[rng.integers(len(free))][0]),
                int(free[rng.integers(len(free))][1])))
              for j in range(int(rng.integers(0, 3)))]
    coins = [(int(free[rng.integers(len(free))][0]),
              int(free[rng.integers(len(free))][1]))
             for _ in range(int(rng.integers(0, 5)))]
    return {'field': arena, 'coins': coins, 'step': int(rng.integers(0, 400)),
            'self': ('a', int(rng.integers(0, 9)), True, tuple(pos)),
            'others': others, 'bombs': bombs,
            'explosion_map': np.zeros((17, 17), dtype=int)}


rng = np.random.default_rng(11)
states = [rand_state(rng) for _ in range(30)]

# 3. determinism + finiteness + bounds
e0 = [extra_scalars(s) for s in states]
e1 = [extra_scalars(s) for s in states]
check('extras deterministic', all(np.array_equal(a, b) for a, b in zip(e0, e1)))
check('extras finite', all(np.isfinite(a).all() for a in e0))
E = np.stack(e0)
check('trap/margin/hunt/deadend binary',
      set(np.unique(E[:, [0, 3, 5, 6]]).tolist()) <= {0.0, 1.0})
check('bounded dims in range', bool((E[:, [1, 2, 4, 7]] >= -1.0).all()
                                    and (E[:, [1, 2, 4, 7]] <= 1.0).all()))


def rot_state(gs):
    """CW rotation of a full game state: (x,y) -> (16-y,x)."""
    def rp(p):
        return (16 - int(p[1]), int(p[0]))
    g = dict(gs)
    # CW: out[x,y] = in[y,16-x]
    g['field'] = np.ascontiguousarray(
        [[gs['field'][y, 16 - x] for y in range(17)] for x in range(17)])
    n, s, b, xy = gs['self']
    g['self'] = (n, s, b, rp(xy))
    g['others'] = [(n2, s2, b2, rp(xy2)) for (n2, s2, b2, xy2) in gs['others']]
    g['coins'] = [rp(c) for c in gs['coins']]
    g['bombs'] = [(rp(xy2), t) for (xy2, t) in gs['bombs']]
    g['explosion_map'] = np.ascontiguousarray(
        [[gs['explosion_map'][y, 16 - x] for y in range(17)] for x in range(17)])
    return g


# 4. rotation invariance (exact equality expected)
bad = 0
for s in states:
    a, b = extra_scalars(s), extra_scalars(rot_state(s))
    if not np.array_equal(a, b):
        bad += 1
        if bad <= 2:
            print('  mismatch:', a, b)
check('extras rotation-invariant (30/30 exact)', bad == 0)

# 5. safety parity apex vs overlord
sys.path.insert(0, '.')
from agent_code.apex.safety import action_safety as apex_safety
from agent_code.overlord.safety import action_safety as over_safety
os.environ.pop('OVERLORD_G_PAYOFF_TIER', None)
os.environ.pop('OVERLORD_G_MUSTFLEE1', None)
mismatch = 0
for s in states:
    a, b = apex_safety(s), over_safety(s)
    if a['valid'] != b['valid'] or a['safe'] != b['safe'] \
            or a['can_escape_if_bomb'] != b['can_escape_if_bomb'] \
            or a['dist_hyp'] != b['dist_hyp'] \
            or a['crates_hit_if_bomb'] != b['crates_hit_if_bomb'] \
            or a['opps_hit_if_bomb'] != b['opps_hit_if_bomb']:
        mismatch += 1
check('mask parity apex==overlord (30/30)', mismatch == 0)

# 6. trunk transfer: every non-head key loads byte-exact; head re-inits.
# (strict=False skips missing keys but NOT shape mismatches, so the head
# keys — whose shapes legitimately change 392->400 — are excluded first.)
ckpt = torch.load('results/archive/overlord_val_best_O3s.pt',
                   map_location='cpu', weights_only=False)['q_net']
HEAD = {'fc.0.weight', 'fc.0.bias', 'v.weight', 'v.bias',
        'adv.weight', 'adv.bias', 'aux.weight', 'aux.bias'}
body = {k: v.cpu() for k, v in ckpt.items() if k not in HEAD}
res = m.load_state_dict(body, strict=False)
missing, unexpected = set(res.missing_keys), set(res.unexpected_keys)
check('transfer: only head layers re-init',
      missing == HEAD and not unexpected)
n_trunk = len([k for k in body])
print('INFO: trunk params transferred: %d (head re-inits: %d)'
      % (n_trunk, len(HEAD)))

print('PROBE PASS (E49 apex scaffold static parts)')

"""Reaper training: N-step Double Dueling DQN + PER + Huber on a small MLP.

Design (cheap-to-train, strong):
  * engineered 68-dim features (BFS pathfinding + placement value +
    own-bomb + opponent model) — navigation comes from features, not
    from learning pixels
  * optional BC pretrain (scripts/pretrain_reaper.py) from teacher demos
    (warden/sentinel/overlord), then RL fine-tune with a demo-replay mix
    (REAPER_DEMOS + REAPER_DEMO_RATIO) to resist catastrophic forgetting
  * 8-fold dihedral symmetry augmentation of every transition at sample
    time (features permute exactly; see features.apply_aug/map_action)
  * proven anti-divergence preset: N_STEP=5 (suicide inside the return
    window), Huber delta=1, grad clip 1, tuned Adam (eps/decay 1e-4)

Device: CUDA when available (auto), else CPU. Env: REAPER_DEVICE / _AMP /
_UTD / _BATCH / _EOR_UPDATES / _OPT / _LR / _SCHEDULE / _TUNED /
_SAVE_EVERY / _EPS_START / _EPS_DECAY / _DEMOS / _DEMO_RATIO / _BC_W.
"""
from collections import deque, namedtuple
import glob
import os
import numpy as np
import events as e

try:
    from .callbacks import ACTION_LIST, ACTION_TO_IDX
    from .features import (state_to_features, FEATURE_DIM, apply_aug,
                           map_action, N_SYMS)
    from .model import build_model
    from .safety import action_safety
    from dml_trainkit import (update_config, DutyTimer, get_cuda_device,
                              get_device, is_cuda_device,
                              CheckpointStore, append_metrics_row, apply_schedule,
                              amp_enabled, save_every_config)
except ImportError:
    from callbacks import ACTION_LIST, ACTION_TO_IDX
    from features import (state_to_features, FEATURE_DIM, apply_aug,
                          map_action, N_SYMS)
    from model import build_model
    from safety import action_safety
    try:
        from dml_trainkit import (update_config, DutyTimer, get_cuda_device,
                                  get_device, is_cuda_device,
                                  CheckpointStore, append_metrics_row, apply_schedule,
                                  amp_enabled, save_every_config)
    except ImportError:
        import sys
        _root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from dml_trainkit import (update_config, DutyTimer, get_cuda_device,
                                  get_device, is_cuda_device,
                                  CheckpointStore, append_metrics_row, apply_schedule,
                                  amp_enabled, save_every_config)

Transition = namedtuple('Transition', ('feats', 'action', 'next_feats', 'reward', 'done'))

LR = 1e-3
BATCH = 256
BATCH_CPU = 128
TARGET_SYNC = 1000
BUFFER_SIZE = 200000
MIN_REPLAY = 2000
EMA_ALPHA = 0.05


def _env_int(name, default, lo, hi):
    try:
        v = int(os.environ.get(name, str(default)))
    except ValueError:
        v = default
    return max(lo, min(hi, v))


def _env_float(name, default):
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


EPS_START = _env_float('REAPER_EPS_START', 1.0)
EPS_END = 0.05
EPS_DECAY = _env_int('REAPER_EPS_DECAY', 50000, 5000, 1000000)
UTD, BATCH_EFF, EOR_UPDATES = update_config('REAPER', BATCH, 4)
SAVE_EVERY = save_every_config('REAPER', 5)
OPT_NAME = os.environ.get('REAPER_OPT', 'adam')
OPT_LR = float(os.environ.get('REAPER_LR', '0.0') or 0.0)
OPT_SCHEDULE = os.environ.get('REAPER_SCHEDULE', '0') == '1'
OPT_TUNED = os.environ.get('REAPER_TUNED', '1') == '1'
OPT_LR_DEFAULTS = {'adam': LR, 'lion': 1e-3}
SCHED_WARMUP, SCHED_TOTAL, SCHED_MIN = 5000, 200000, 0.1
DEMOS_GLOB = os.environ.get('REAPER_DEMOS', '').split(':') if os.environ.get('REAPER_DEMOS', '') else []
DEMO_RATIO = _env_float('REAPER_DEMO_RATIO', 0.0)
BC_W = _env_float('REAPER_BC_W', 0.5)
# E37/P3: longer horizon so kill chains (10-25 steps of pursuit) stay inside
# the return window (gamma^5=0.86 was strangling pursuit credit).
GAMMA = _env_float('REAPER_GAMMA', 0.99)
N_STEP = _env_int('REAPER_N_STEP', 8, 1, 20)
# Potential-shaping weights (Ng et al. 1999; policy-invariant: the per-round
# total telescopes to Phi_T - Phi_0, so shaping can guide without ever
# dominating the engine score). Potentials read off the feature vector:
# f[93] PHI_kill, f[94] coin closeness, f[85] mobility.
W_PHI_KILL = _env_float('REAPER_W_PHI_KILL', 2.0)
W_PHI_COIN = _env_float('REAPER_W_PHI_COIN', 1.0)
W_PHI_MOB = _env_float('REAPER_W_PHI_MOB', 0.5)


class PERBuffer:
    """Prioritized replay on preallocated numpy arrays (E37/P1).

    The old version rebuilt a 200k-element probability vector from a Python
    list and sampled with replace=False on EVERY update — O(N) with a huge
    constant, dominating step time at capacity. This version keeps a fixed
    ring + float64 priority array: add is O(1), sample is one vectorized
    sum + cumsum-searchsorted (~0.3 ms at 200k). replace=True duplicates
    are rare at batch << buffer and harmless for SGD.
    """

    def __init__(self, cap=BUFFER_SIZE, alpha=0.6):
        self.cap = cap
        self.alpha = alpha
        self.buf = [None] * cap
        self.prio = np.zeros(cap, dtype=np.float64)
        self.n = 0
        self.pos = 0

    def __len__(self):
        return self.n

    def add(self, tr, td_error=1.0):
        p = (abs(float(td_error)) + 1e-5) ** self.alpha
        self.buf[self.pos] = tr
        self.prio[self.pos] = p
        self.pos = (self.pos + 1) % self.cap
        if self.n < self.cap:
            self.n += 1

    def sample(self, n, beta=0.4):
        n = max(1, min(int(n), self.n))
        p = self.prio[:self.n]
        s = float(p.sum())
        if not (s > 0.0):
            idx = np.random.randint(0, self.n, size=n)
            return [self.buf[i] for i in idx], idx.astype(np.int64), \
                np.ones(n, dtype=np.float32)
        p = p / s
        idx = np.searchsorted(np.cumsum(p),
                              np.random.random_sample(n)).clip(0, self.n - 1)
        w = (self.n * p[idx]) ** (-beta)
        w = w / w.max()
        return [self.buf[i] for i in idx], idx.astype(np.int64), \
            w.astype(np.float32)

    def update(self, idx, td_errors):
        for i, td in zip(idx, td_errors):
            self.prio[int(i)] = (abs(float(td)) + 1e-5) ** self.alpha


def epsilon_now(epsilon_steps):
    frac = min(1.0, epsilon_steps / EPS_DECAY)
    return EPS_END + (EPS_START - EPS_END) * (1.0 - frac)


def _pick_device(logger=None):
    try:
        # NB: pass 'REAPER_DML' (not 'REAPER_DEVICE'): _want_cuda derives
        # the canonical key as {prefix}_DEVICE, so this is what makes
        # REAPER_DEVICE=cpu actually work (passing 'REAPER_DEVICE' would
        # make it look for 'REAPER_DEVICE_DEVICE' and silently stay on
        # CUDA — same convention as overlord's 'OVERLORD_DML' call).
        return get_cuda_device('REAPER_DML', logger, default_on=True)
    except Exception:
        try:
            import torch
            return torch.device('cpu')
        except Exception:
            return 'cpu'


def _load_demos(logger=None):
    """Load demo (feats, action) pairs from npz files matching DEMOS_GLOB.

    Paths resolve against CWD, ~, AND the repo root: the engine chdir's
    into agent_code/<name>/ around every callback, so a repo-relative
    pattern like results/demos/... would otherwise silently match nothing
    (and the demo mix would be a no-op without any error).
    """
    try:
        _root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    except Exception:
        _root = ''
    feats, acts = [], []
    for pat in DEMOS_GLOB:
        files = (sorted(glob.glob(pat))
                 or sorted(glob.glob(os.path.expanduser(pat)))
                 or (sorted(glob.glob(os.path.join(_root, pat))) if _root else []))
        if not files and logger:
            logger.warning(f'reaper demo pattern matched nothing: {pat}')
        for f in files:
            try:
                d = np.load(f)
                F = np.asarray(d['feats'], dtype=np.float32)
                A = np.asarray(d['acts'])
                if F.ndim != 2 or F.shape[1] != FEATURE_DIM:
                    continue
                feats.append(F)
                acts.append(A)
            except Exception as ex:
                if logger:
                    logger.warning(f'reaper demo load failed {f}: {ex}')
    if not feats:
        return None, None
    F = np.concatenate(feats, axis=0)
    A = np.concatenate(acts, axis=0)
    A = np.array([ACTION_TO_IDX[a] if isinstance(a, str) else int(a) for a in A],
                 dtype=np.int64)
    return F, A


def _engine_score_delta(events):
    """Exact tournament score delta from one step's engine events.

    The tournament ranks by total score = coins x1 + kills x5 (settings.py
    REWARD_COIN/REWARD_KILL, scored in environment.py:186,256). Everything
    else in the reward function is shaping. Logging both separately (see
    end_of_round metrics) keeps shaping from silently dominating the true
    objective — the Phase-3 acceptance gate.
    """
    d = 0.0
    for ev in events:
        if ev == e.COIN_COLLECTED:
            d += 1.0
        elif ev == e.KILLED_OPPONENT:
            d += 5.0
    return d


def _potential(feats):
    """Shaping potential Phi(s) off the feature vector (E37/P3).

    Phi = W_PHI_KILL * f[93] + W_PHI_COIN * f[94] + W_PHI_MOB * f[85].
    Bounded in [0, ~3.5]; the per-round total telescopes to Phi_T - Phi_0
    (Ng et al. 1999), so shaping guides every step without ever dominating
    the engine score no matter the weights.
    """
    if feats is None:
        return 0.0
    try:
        return (W_PHI_KILL * float(feats[93]) + W_PHI_COIN * float(feats[94])
                + W_PHI_MOB * float(feats[85]))
    except (IndexError, TypeError, ValueError):
        return 0.0


def _phi_shaping(feats, nfeats):
    """F(s, a, s') = gamma * Phi(s') - Phi(s). Zero if either end is missing
    (encode failure / terminal), so shaping never injects NaNs."""
    if feats is None:
        return 0.0
    nxt = _potential(nfeats) if nfeats is not None else 0.0
    return GAMMA * nxt - _potential(feats)


def reward_from_events(self, events, old_state=None, action=None, new_state=None):
    """Base reward = the tournament objective, exactly (E37/P3).

    +1 coin / +5 kill match the engine score one-to-one; death penalties
    price the forfeited future score (suicide worse: it also hands the
    round to the opponents). INVALID teaches legality (the engine scores
    it 0, so the net must learn it from here). WAITED is a mild trickle
    against idle-survival policies (which would otherwise farm SURVIVED).
    Everything else the old shaping did (crate spam, coin greed, flat trap
    bonuses into a saturated coin pie) is gone — replaced by the
    potential-based shaping in game_events_occurred, which preserves the
    optimal policy by construction.
    """
    r = 0.0
    for ev in events:
        if ev == e.COIN_COLLECTED:
            r += 1.0
        elif ev == e.KILLED_OPPONENT:
            r += 5.0
        elif ev == e.KILLED_SELF:
            r -= 8.0
        elif ev == e.GOT_KILLED:
            r -= 6.0
        elif ev == e.INVALID_ACTION:
            r -= 0.6
        elif ev == e.WAITED:
            r -= 0.05
        elif ev == e.SURVIVED_ROUND:
            r += 1.0
    return r


def _custom(old_state, action, new_state):
    # E37/P3: no synthetic events. The old MOVE_TOWARD/AWAY_COIN,
    # BOMB_NO_ESCAPE/BOMB_GOOD/TRAP_LAID events were flat (non-potential)
    # shaping that distorted the objective AND cost an extra action_safety
    # + up to 3 opp_can_escape calls on every BOMB step. Their guidance now
    # comes from the potential shaping (Phi_kill/Phi_coin/Phi_mob), which
    # is policy-invariant and free (it reuses the encoded features).
    return []


def _encode(state, own_bomb):
    if state is None:
        return None
    try:
        s = action_safety(state)
    except Exception:
        s = None
    return state_to_features(state, s, own_bomb).astype(np.float32)


def setup_training(self):
    self.logger.info('reaper setup_training')
    import torch
    # REAPER_RUN_DIR isolates parallel sweeps (E37/P4): checkpoints, metrics
    # and model exports land under the run dir instead of the agent dir, so
    # K concurrent `main.py play --train 1` processes never clobber each
    # other. Only the promoted candidate is copied into agent_code/reaper/.
    # NOTE: SequentialAgentBackend chdirs into agent_code/<name>/ around
    # every callback, so a RELATIVE run dir must be resolved against the
    # repo root here (absolute __file__), never against the process cwd.
    _rd = os.environ.get('REAPER_RUN_DIR', '').strip()
    if _rd and not os.path.isabs(_rd):
        _repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _rd = os.path.normpath(os.path.join(_repo, _rd))
    here = _rd or os.path.dirname(os.path.abspath(__file__))
    self._reaper_here = here
    # Optional seeding for sweep arms (E37 sw02): seeds numpy/python RNG
    # so parallel jobs differ reproducibly. Torch init noise remains
    # unseeded (cuDNN nondeterminism would make exact repro misleading).
    _seed = os.environ.get('REAPER_SEED', '').strip()
    if _seed:
        try:
            import random as _r
            _s = int(_seed)
            np.random.seed(_s)
            _r.seed(_s)
            self.logger.info(f'reaper seeded RNGs with {_s}')
        except Exception as ex:
            self.logger.warning(f'reaper seeding failed: {ex}')
    self.device = _pick_device(self.logger)
    if os.environ.get('REAPER_BATCH', '0') in ('0', ''):
        self.batch_size = BATCH if is_cuda_device(self.device) else BATCH_CPU
    else:
        self.batch_size = BATCH_EFF
    self.utd = UTD
    self.eor_updates = EOR_UPDATES
    self.duty = DutyTimer()
    self.ckpts = CheckpointStore(here, logger=self.logger, save_every=SAVE_EVERY)
    self.logger.info(
        f'reaper device={self.device} batch={self.batch_size} '
        f'utd={self.utd} eor={self.eor_updates}'
    )
    self.q_net = build_model().to(self.device)
    if getattr(self, 'model', None) is None:
        self.model = build_model()
    try:
        self.model.cpu()
        self.model.eval()
    except Exception:
        pass
    self.target_net = build_model().to(self.device)
    self.target_net.load_state_dict(self.q_net.state_dict())
    self.target_net.eval()
    try:
        from dml_optimizer import build_optimizer_for_device, assert_params_on_device
    except ImportError:
        import sys
        _root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from dml_optimizer import build_optimizer_for_device, assert_params_on_device
    self.opt_base_lr = OPT_LR or OPT_LR_DEFAULTS.get(OPT_NAME, LR)
    self.opt_schedule = OPT_SCHEDULE
    self.optimizer = build_optimizer_for_device(
        self.device, self.q_net.parameters(), lr=self.opt_base_lr,
        weight_decay=1e-4, adamw=False, tuned=OPT_TUNED,
        name=OPT_NAME if OPT_NAME in ('adam', 'lion') else 'adam',
    )
    try:
        assert_params_on_device(self.q_net, self.device, what="q_net")
    except AssertionError as ex:
        self.logger.warning(f'reaper device mismatch: {ex}')
    try:
        _dev_is_cuda = is_cuda_device(self.device)
    except Exception:
        _dev_is_cuda = str(getattr(self.device, 'type', self.device)) == 'cuda'
    self.use_amp = bool(_dev_is_cuda and amp_enabled('REAPER', default_on=True))
    self.scaler = None
    if self.use_amp:
        try:
            self.scaler = torch.amp.GradScaler('cuda')
        except Exception:
            try:
                self.scaler = torch.cuda.amp.GradScaler()
            except Exception as ex:
                self.logger.warning(f'reaper AMP scaler init failed, fp32 fallback: {ex}')
                self.use_amp = False
                self.scaler = None
    self.logger.info(
        f'reaper optimizer={type(self.optimizer).__name__} lr={self.opt_base_lr} '
        f'schedule={self.opt_schedule} amp={self.use_amp} device={self.device}'
    )
    self.buffer = PERBuffer()
    self.n_step_buf = deque(maxlen=N_STEP)
    self.total_steps = 0
    self.episode = 0
    self.epsilon_steps = 0
    self.epsilon = EPS_START
    self.best_ema = None
    self.ema_reward = None
    self.last_loss = 0.0
    self.last_bc_loss = 0.0
    self._round_reward = 0.0
    self._round_rew_engine = 0.0
    self._round_rew_shaping = 0.0
    self._round_coins = 0
    self._round_kills = 0
    self._round_suicides = 0
    self._round_killed_self = 0
    self._round_got_killed = 0
    self._own_bomb = None
    self._feat_cache = None

    self._demo_feats, self._demo_acts = None, None
    if DEMOS_GLOB and DEMO_RATIO > 0:
        self._demo_feats, self._demo_acts = _load_demos(self.logger)
        if self._demo_feats is not None:
            self.logger.info(f'reaper demos loaded: {len(self._demo_feats)} samples')

    resumed = False
    ckpt = self.ckpts.load('last.pt')
    if ckpt and isinstance(ckpt, dict) and 'q_net' in ckpt:
        try:
            dev = self.device
            self.q_net.load_state_dict(
                {k: v.to(dev) if hasattr(v, 'to') else v for k, v in ckpt['q_net'].items()},
                strict=False,
            )
            self.target_net.load_state_dict(self.q_net.state_dict())
            try:
                self.optimizer.load_state_dict(ckpt.get('optimizer', {}))
            except Exception as ex:
                self.logger.warning(
                    f'reaper optimizer state incompatible '
                    f'({type(self.optimizer).__name__}), fresh optimizer: {ex}'
                )
            if OPT_TUNED and OPT_NAME == 'adam':
                for g in self.optimizer.param_groups:
                    g['eps'] = 1e-4
                    g['weight_decay'] = 1e-4
            for st in self.optimizer.state.values():
                for k, v in list(st.items()):
                    if isinstance(v, torch.Tensor):
                        try:
                            st[k] = v.to(dev)
                        except Exception:
                            pass
            self.episode = int(ckpt.get('episode', 0))
            self.total_steps = int(ckpt.get('total_steps', 0))
            self.epsilon_steps = int(ckpt.get('epsilon_steps', 0))
            self.epsilon = epsilon_now(self.epsilon_steps)
            self.best_ema = ckpt.get('best_ema')
            self.ema_reward = ckpt.get('ema_reward')
            try:
                if getattr(self, 'use_amp', False) and self.scaler is not None \
                        and isinstance(ckpt.get('scaler'), dict):
                    self.scaler.load_state_dict(ckpt['scaler'])
            except Exception as ex:
                self.logger.warning(f'reaper scaler resume failed, fresh scaler: {ex}')
            self.model.load_state_dict(
                {k: v.cpu() if hasattr(v, 'cpu') else v for k, v in ckpt['q_net'].items()},
                strict=False,
            )
            self.model.eval()
            resumed = True
            self.logger.info(
                f"reaper resumed last.pt ep={self.episode} steps={self.total_steps}"
            )
        except Exception as ex:
            self.logger.warning(f'reaper resume failed, fresh start: {ex}')
    if not resumed:
        try:
            if getattr(self, 'model', None) is not None:
                raw = {k: v for k, v in self.model.state_dict().items()}
                self.q_net.load_state_dict(
                    {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in raw.items()},
                    strict=False,
                )
                self.target_net.load_state_dict(self.q_net.state_dict())
        except Exception:
            pass
        self.logger.info('reaper fresh start (no usable checkpoint)')


def _payload(self):
    import torch
    dev_cpu = torch.device('cpu')
    payload = {
        'q_net': {k: v.detach().to(dev_cpu) for k, v in self.q_net.state_dict().items()},
        'target_net': {k: v.detach().to(dev_cpu) for k, v in self.target_net.state_dict().items()},
        'optimizer': self.optimizer.state_dict(),
        'episode': self.episode,
        'total_steps': self.total_steps,
        'epsilon_steps': self.epsilon_steps,
        'best_ema': self.best_ema,
        'ema_reward': self.ema_reward,
        'config': {'gamma': GAMMA, 'n_step': N_STEP, 'lr': self.opt_base_lr,
                   'batch': self.batch_size, 'opt': type(self.optimizer).__name__,
                   'schedule': self.opt_schedule, 'utd': self.utd,
                   'eor': self.eor_updates, 'eps_decay': EPS_DECAY,
                   'eps_start': EPS_START,
                   'demo_ratio': DEMO_RATIO, 'bc_w': BC_W,
                   'amp': bool(getattr(self, 'use_amp', False))},
    }
    try:
        if getattr(self, 'use_amp', False) and getattr(self, 'scaler', None) is not None:
            payload['scaler'] = self.scaler.state_dict()
    except Exception:
        pass
    return payload


def _export_tournament_model(here, q_net):
    import torch
    sd = {k: v.detach().cpu() for k, v in q_net.state_dict().items()}
    here_pt = os.path.join(here, 'my-saved-model.pt')
    tmp = here_pt + '.tmp'
    torch.save(sd, tmp)
    os.replace(tmp, here_pt)
    # Sweep runs (REAPER_RUN_DIR set) must NOT touch the repo-root copy:
    # K parallel jobs would race on it. Only the main agent dir exports
    # the cwd mirror (legacy behavior).
    if os.environ.get('REAPER_RUN_DIR', '').strip():
        return
    try:
        cwd_pt = os.path.join(os.getcwd(), 'my-saved-model.pt')
        if os.path.abspath(cwd_pt) != os.path.abspath(here_pt):
            torch.save(sd, cwd_pt + '.tmp')
            os.replace(cwd_pt + '.tmp', cwd_pt)
    except Exception:
        pass


def _push(self, feats, action, nfeats, reward, done):
    self.n_step_buf.append((feats, action, reward))
    if len(self.n_step_buf) < N_STEP and not done:
        return
    R = 0.0
    for i, (_, _, r) in enumerate(self.n_step_buf):
        R += (GAMMA ** i) * r
    f0, a0, _ = self.n_step_buf[0]
    self.buffer.add(Transition(f0, a0, nfeats, R, done))
    if done:
        _drain_n_step(self)
    elif len(self.n_step_buf) == N_STEP:
        self.n_step_buf.popleft()


def _drain_n_step(self):
    """Flush the n-step buffer as terminal transitions (done=True)."""
    while len(self.n_step_buf) > 1:
        self.n_step_buf.popleft()
        R = 0.0
        for i, (_, _, r) in enumerate(self.n_step_buf):
            R += (GAMMA ** i) * r
        if len(self.n_step_buf) == 0:
            break
        f0, a0, _ = self.n_step_buf[0]
        self.buffer.add(Transition(f0, a0, None, R, True))
    self.n_step_buf.clear()


def _push_final(self, terminal_bonus):
    """Round-end finalize without double-pushing (E37/P1).

    When the agent SURVIVES the round, game_events_occurred already pushed
    (last_state, last_action) into the n-step buffer — so we must NOT push
    it again. Instead attach the terminal reward (e.g. SURVIVED_ROUND) to
    the buffered tail and drain once as done=True.
    """
    if terminal_bonus and self.n_step_buf:
        f, a, r = self.n_step_buf[-1]
        self.n_step_buf[-1] = (f, a, r + float(terminal_bonus))
    _drain_n_step(self)


def _aug_batch(feats_list, acts_list, next_list=None):
    """Random dihedral augmentation: permute features + map actions."""
    F = np.stack(feats_list).astype(np.float32)
    syms = np.random.randint(0, N_SYMS, size=F.shape[0])
    for i, s in enumerate(syms):
        F[i] = apply_aug(F[i], int(s))
    A = [map_action(a, int(s)) for a, s in zip(acts_list, syms)]
    A = np.array([ACTION_TO_IDX[a] if isinstance(a, str) else int(a) for a in A])
    N = None
    if next_list is not None:
        N = np.stack([f if f is not None else feats_list[0] for f in next_list]).astype(np.float32)
        for i, s in enumerate(syms):
            if next_list[i] is not None:
                N[i] = apply_aug(N[i], int(s))
    return F, A, N


def _update(self):
    if len(self.buffer) < MIN_REPLAY:
        return
    import torch
    self.total_steps += 1
    beta = min(1.0, 0.4 + 0.6 * self.total_steps / 150000)
    if getattr(self, 'opt_schedule', False):
        apply_schedule(self.optimizer, self.opt_base_lr, self.total_steps,
                       warmup=SCHED_WARMUP, total=SCHED_TOTAL, min_ratio=SCHED_MIN)
    dev = self.device
    use_demo = bool(DEMO_RATIO > 0 and self._demo_feats is not None
                    and len(self._demo_feats) > 0)
    n_demo = int(self.batch_size * DEMO_RATIO) if use_demo else 0
    n_rl = self.batch_size - n_demo
    demo_F = demo_A = None
    if n_demo > 0:
        idx = np.random.randint(0, len(self._demo_feats), size=n_demo)
        demo_F = self._demo_feats[idx].copy()
        syms = np.random.randint(0, N_SYMS, size=n_demo)
        for i, s in enumerate(syms):
            demo_F[i] = apply_aug(demo_F[i], int(s))
        demo_A = self._demo_acts[idx]
        for i, s in enumerate(syms):
            a = ACTION_LIST[int(demo_A[i])]
            demo_A[i] = ACTION_TO_IDX[map_action(a, int(s))]
        demo_F = torch.from_numpy(demo_F).to(dev)
        demo_A = torch.tensor(demo_A, dtype=torch.long, device=dev)

    batch, idx, w = self.buffer.sample(max(n_rl, 1), beta)
    F, A, N = _aug_batch(
        [t.feats for t in batch], [t.action for t in batch],
        [t.next_feats for t in batch],
    )
    S = torch.from_numpy(F).to(dev)
    A = torch.tensor(A, device=dev).unsqueeze(1)
    NS = torch.from_numpy(N).to(dev)
    R = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=dev)
    D = torch.tensor([0.0 if t.done else 1.0 for t in batch], dtype=torch.float32, device=dev)
    W = torch.tensor(w, dtype=torch.float32, device=dev)
    gam_n = GAMMA ** N_STEP
    self.q_net.train()

    def loss_fn():
        q = self.q_net(S).gather(1, A).squeeze(1)
        with torch.no_grad():
            a_star = self.q_net(NS).argmax(dim=1, keepdim=True)
            qt = self.target_net(NS).gather(1, a_star).squeeze(1)
            target = R + D * gam_n * qt
        td = target - q
        td_loss = (W * torch.nn.functional.smooth_l1_loss(td, torch.zeros_like(td), reduction='none')).mean()
        bc_loss = torch.tensor(0.0, device=dev)
        if n_demo > 0:
            logits = self.q_net(demo_F)
            bc_loss = torch.nn.functional.cross_entropy(logits, demo_A)
        return td_loss + BC_W * bc_loss, td, bc_loss

    use_amp = bool(getattr(self, 'use_amp', False) and getattr(self, 'scaler', None) is not None)
    if use_amp:
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            loss, td, bc_loss = loss_fn()
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
    else:
        loss, td, bc_loss = loss_fn()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()
    try:
        self.buffer.update(idx, td.detach().cpu().numpy())
    except Exception:
        pass
    try:
        self.last_loss = float(loss.detach().cpu().item())
        self.last_bc_loss = float(bc_loss.detach().cpu().item())
    except Exception:
        pass
    if self.total_steps % TARGET_SYNC == 0:
        self.target_net.load_state_dict(self.q_net.state_dict())
        try:
            self.model.load_state_dict(
                {k: v.cpu() for k, v in self.q_net.state_dict().items()}, strict=False
            )
            self.model.eval()
        except Exception:
            pass


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    try:
        events = list(events) + _custom(old_game_state, self_action, new_game_state)
    except Exception:
        pass
    r = reward_from_events(self, events, old_game_state, self_action, new_game_state)
    self._round_reward += float(r)
    self._round_rew_engine = float(getattr(self, '_round_rew_engine', 0.0)) + \
        float(_engine_score_delta(events))
    try:
        if e.COIN_COLLECTED in events:
            self._round_coins += 1
        if e.KILLED_OPPONENT in events:
            self._round_kills += 1
        if e.KILLED_SELF in events:
            self._round_suicides += 1
            self._round_killed_self += 1
        elif e.GOT_KILLED in events:
            self._round_suicides += 1
            self._round_got_killed += 1
    except Exception:
        pass
    own_before = self._own_bomb
    # Feature cache: this step's new_state IS next step's old_state (the
    # engine passes the same object through store_game_state). Key on
    # identity (strong ref, so no id-reuse) + own-bomb value, since the
    # tracker updates between the two encodings on BOMB/replenish steps.
    feats = None
    try:
        _fc = getattr(self, '_feat_cache', None)
        if _fc is not None and old_game_state is not None \
                and old_game_state is _fc[0] and own_before == _fc[1]:
            feats = _fc[2]
    except Exception:
        feats = None
    if feats is None and old_game_state is not None:
        feats = _encode(old_game_state, own_before)
    nfeats = _encode(new_game_state, own_before) if new_game_state is not None else None
    try:
        self._feat_cache = (new_game_state, own_before, nfeats)
    except Exception:
        pass
    # Potential-based shaping (E37/P3): gamma*Phi(s') - Phi(s), reusing the
    # already-encoded features (free). Policy-invariant by construction.
    try:
        _sh = _phi_shaping(feats, nfeats)
    except Exception:
        _sh = 0.0
    r += _sh
    self._round_rew_shaping = float(getattr(self, '_round_rew_shaping', 0.0)) + float(_sh)
    if feats is not None and self_action in ACTION_TO_IDX:
        _push(self, feats, self_action, nfeats, r, new_game_state is None)
    # update own-bomb tracker AFTER encoding (features describe pre-action state)
    try:
        if self_action == 'BOMB' and old_game_state is not None:
            _, _, _, (x, y) = old_game_state['self']
            self._own_bomb = (int(x), int(y))
        elif old_game_state is not None:
            _, _, bl, _ = old_game_state['self']
            if not bl:
                nbl = new_game_state['self'][2] if new_game_state is not None else True
                if nbl:
                    self._own_bomb = None
    except Exception:
        pass
    # E31/C2 rule: epsilon counts env steps, not gradient steps
    self.epsilon_steps += 1
    self.epsilon = epsilon_now(self.epsilon_steps)
    for _ in range(getattr(self, 'utd', 1)):
        with self.duty.measure():
            _update(self)


def end_of_round(self, last_game_state, last_action, events):
    try:
        events = list(events) + _custom(last_game_state, last_action, None)
    except Exception:
        pass
    r = reward_from_events(self, events, last_game_state, last_action, None)
    self._round_reward += float(r)
    self._round_rew_engine = float(getattr(self, '_round_rew_engine', 0.0)) + \
        float(_engine_score_delta(events))
    try:
        if e.COIN_COLLECTED in events:
            self._round_coins += 1
        if e.KILLED_OPPONENT in events:
            self._round_kills += 1
        if e.KILLED_SELF in events:
            self._round_suicides += 1
            self._round_killed_self += 1
        elif e.GOT_KILLED in events:
            self._round_suicides += 1
            self._round_got_killed += 1
    except Exception:
        pass
    if e.SURVIVED_ROUND in events:
        # Survived: (last_state, last_action) already entered the n-step
        # buffer via game_events_occurred (do_step always sends events for
        # the final step before end_round). Attach the terminal reward to
        # the buffered tail and drain once — pushing again here would
        # duplicate (s, a) with a conflicting return.
        _push_final(self, r)
    elif last_action in ACTION_TO_IDX and last_game_state is not None:
        # Died: the fatal step never reached game_events_occurred (the
        # engine skips dead agents), so push it now as THE terminal
        # transition.
        feats = _encode(last_game_state, self._own_bomb)
        _push(self, feats, last_action, None, r, True)
    else:
        self.n_step_buf.clear()
    self._feat_cache = None
    self.epsilon_steps += 1
    self.epsilon = epsilon_now(self.epsilon_steps)
    for _ in range(getattr(self, 'eor_updates', 4)):
        with self.duty.measure():
            _update(self)
    gpu_ms, wall_ms, duty_pct = self.duty.end_round()
    self.episode += 1
    here = getattr(self, '_reaper_here', os.path.dirname(os.path.abspath(__file__)))
    self._own_bomb = None

    rr = float(self._round_reward)
    self.ema_reward = rr if self.ema_reward is None else (1 - EMA_ALPHA) * self.ema_reward + EMA_ALPHA * rr
    improved = self.best_ema is None or self.ema_reward > self.best_ema
    if improved:
        self.best_ema = float(self.ema_reward)

    try:
        import torch as _tw
        with _tw.no_grad():
            wnorm = float(_tw.cat([p.detach().flatten().float().cpu() for p in self.q_net.parameters()]).norm().item())
    except Exception:
        wnorm = 0.0
    try:
        append_metrics_row(os.path.join(here, 'runs', 'metrics.csv'), {
            'episode': self.episode,
            'total_steps': self.total_steps,
            'epsilon': round(float(self.epsilon), 4),
            'buffer': len(self.buffer),
            'loss': round(float(self.last_loss), 5),
            'bc_loss': round(float(self.last_bc_loss), 5),
            'round_reward': round(rr, 3),
            'rew_engine': round(float(getattr(self, '_round_rew_engine', 0.0)), 3),
            'rew_shaping': round(float(getattr(self, '_round_rew_shaping', 0.0)), 3),
            'ema_reward': round(float(self.ema_reward), 3),
            'coins': int(self._round_coins),
            'kills': int(self._round_kills),
            'suicides': int(self._round_suicides),
            'killed_self': int(self._round_killed_self),
            'got_killed': int(self._round_got_killed),
            'wnorm': round(float(wnorm), 2),
            'device': str(self.device),
            'gpu_ms': gpu_ms,
            'wall_ms': wall_ms,
            'duty_pct': duty_pct,
        })
    except Exception as ex:
        self.logger.warning(f'reaper metrics log failed: {ex}')

    try:
        payload = _payload(self)
        self.ckpts.maybe_save(self.episode, payload, improved)
        _export_tournament_model(here, self.q_net)
    except Exception as ex:
        self.logger.warning(f'reaper save failed: {ex}')
    self.logger.info(
        f'reaper ep={self.episode} buf={len(self.buffer)} loss={self.last_loss:.4f} '
        f'bc={self.last_bc_loss:.4f} r={rr:.2f} ema={self.ema_reward:.2f} '
        f'wnorm={wnorm:.1f} duty={duty_pct}% dev={self.device}'
    )
    self._round_reward = 0.0
    self._round_rew_engine = 0.0
    self._round_rew_shaping = 0.0
    self._round_coins = 0
    self._round_kills = 0
    self._round_suicides = 0
    self._round_killed_self = 0
    self._round_got_killed = 0

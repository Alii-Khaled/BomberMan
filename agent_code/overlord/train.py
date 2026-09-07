"""Overlord training: CNN Double Dueling DQN + PER + aux danger loss on CUDA.

Device: CUDA when available (default, auto), else CPU.
Same shared kit as sentinel (dml_trainkit): env-driven UTD/batch/round-end
updates + duty-cycle logging. Optimizer via dml_optimizer factory
(stock AdamW on CUDA/CPU; Lion available via OVERLORD_OPT).
AMP autocast+GradScaler on CUDA by default (OVERLORD_AMP=0 for fp32);
fp32 + grad clip on CPU.
"""
from collections import deque, namedtuple
import os
import numpy as np
import events as e

try:
    from .callbacks import ACTION_LIST, ACTION_TO_IDX
    from .features_cnn import state_to_tensor, N_CHANNELS
    from .model import build_model, scalars_from_state
    from .safety import action_safety
    from dml_trainkit import (update_config, DutyTimer, get_cuda_device,
                              get_device, is_cuda_device,
                              CheckpointStore, log_metrics_row, apply_schedule,
                              amp_enabled, append_metrics_row,
                              save_every_config)
except ImportError:
    from callbacks import ACTION_LIST, ACTION_TO_IDX
    from features_cnn import state_to_tensor, N_CHANNELS
    from model import build_model, scalars_from_state
    from safety import action_safety
    try:
        from dml_trainkit import (update_config, DutyTimer, get_cuda_device,
                                  get_device, is_cuda_device,
                                  CheckpointStore, log_metrics_row, apply_schedule,
                                  amp_enabled, append_metrics_row,
                                  save_every_config)
    except ImportError:
        import sys
        _root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from dml_trainkit import (update_config, DutyTimer, get_cuda_device,
                                  get_device, is_cuda_device,
                                  CheckpointStore, log_metrics_row, apply_schedule,
                                  amp_enabled, append_metrics_row,
                                  save_every_config)

Transition = namedtuple('Transition', ('img', 'sc', 'action', 'next_img', 'next_sc', 'reward', 'done', 'aux'))

GAMMA = 0.97
N_STEP = 5
LR = 3e-4
BATCH = 512
BATCH_CPU = 128
TARGET_SYNC = 2000
BUFFER_SIZE = 300000
MIN_REPLAY = 5000
AUX_W = 0.1
def _env_int(name, default, lo, hi):
    try:
        v = int(os.environ.get(name, str(default)))
    except ValueError:
        v = default
    return max(lo, min(hi, v))

EPS_START, EPS_END = 1.0, 0.05
EPS_DECAY = _env_int('OVERLORD_EPS_DECAY', 100000, 5000, 1000000)  # epsilon-greedy, sentinel-proven (lengthened for cold CNN; was 50000)
# GPU-throughput switches (env-gated; defaults = aggressive L40S path):
#   OVERLORD_CHANNELS_LAST=1 (default) keeps conv inputs channels-last
#   OVERLORD_COMPILE=1 enables torch.compile on the train nets (default 0:
#     validate first — BatchNorm/amax usually compile cleanly but graph
#     breaks would silently slow the loop)
#   OVERLORD_SAVE_EVERY (default 5): rounds between full last.pt saves
#     (best.pt still on EMA improvement; tournament export every round).
CHANNELS_LAST = os.environ.get('OVERLORD_CHANNELS_LAST', '1') == '1'
USE_COMPILE = os.environ.get('OVERLORD_COMPILE', '0') == '1'
SAVE_EVERY = save_every_config('OVERLORD', 5)
# Optimizer select (env-gated; default = legacy AdamW behavior):
#   OVERLORD_OPT: 'adam' (stock AdamW) | 'lion' (Lion, benchmark winner on sentinel)
#   OVERLORD_LR: base LR override (default LR)
#   OVERLORD_SCHEDULE: '1' enables warmup+cosine schedule on total_steps
#   OVERLORD_TUNED: '1' applies the DQN task preset to AdamW (eps=1e-4,
#     decay stays 1e-4); re-applied after resume (load_state_dict would
#     otherwise restore old groups). Sentinel Arm-A lesson, ported.
#   OVERLORD_AMP: '1' (default) enables autocast+GradScaler on CUDA; '0' = fp32.
#   OVERLORD_DEVICE: 'cuda'|'cpu'|'auto' (default auto = CUDA when available).
OPT_NAME = os.environ.get('OVERLORD_OPT', 'adam')
OPT_LR = float(os.environ.get('OVERLORD_LR', '0.0') or 0.0)
OPT_SCHEDULE = os.environ.get('OVERLORD_SCHEDULE', '0') == '1'
OPT_TUNED = os.environ.get('OVERLORD_TUNED', '0') == '1'
OPT_LR_DEFAULTS = {'adam': LR, 'lion': 1e-3}
SCHED_WARMUP, SCHED_TOTAL, SCHED_MIN = 5000, 200000, 0.1
# GPU-utilization config (env-gated; defaults = legacy behavior):
#   OVERLORD_UTD (default 1), OVERLORD_BATCH (default 0 = BATCH),
#   OVERLORD_EOR_UPDATES (default 6).
UTD, BATCH_EFF, EOR_UPDATES = update_config('OVERLORD', BATCH, 6)
EMA_ALPHA = 0.05


class PERBuffer:
    def __init__(self, cap=BUFFER_SIZE, alpha=0.6):
        self.cap = cap
        self.alpha = alpha
        self.buf = []
        self.prio = []
        self.pos = 0

    def __len__(self):
        return len(self.buf)

    def add(self, tr, td_error=1.0):
        p = (abs(float(td_error)) + 1e-5) ** self.alpha
        if len(self.buf) < self.cap:
            self.buf.append(tr)
            self.prio.append(p)
        else:
            self.buf[self.pos] = tr
            self.prio[self.pos] = p
            self.pos = (self.pos + 1) % self.cap

    def sample(self, n, beta=0.4):
        p = np.asarray(self.prio, dtype=np.float64)
        p = p / p.sum()
        idx = np.random.choice(len(self.buf), size=min(n, len(self.buf)), replace=False, p=p)
        w = (len(self.buf) * p[idx]) ** (-beta)
        w = w / w.max()
        return [self.buf[i] for i in idx], np.asarray(idx), w.astype(np.float32)

    def update(self, idx, td_errors):
        for i, td in zip(idx, td_errors):
            self.prio[int(i)] = (abs(float(td)) + 1e-5) ** self.alpha


def epsilon_now(epsilon_steps):
    frac = min(1.0, epsilon_steps / EPS_DECAY)
    return EPS_END + (EPS_START - EPS_END) * (1.0 - frac)


def _pick_device(logger=None):
    """CUDA-first (OVERLORD_DEVICE=auto default), CPU fallback. Never raises."""
    try:
        return get_cuda_device('OVERLORD_DML', logger, default_on=True)
    except Exception:
        try:
            import torch
            return torch.device('cpu')
        except Exception:
            return 'cpu'


def reward_from_events(self, events, old_state=None, action=None, new_state=None):
    r = 0.0
    for ev in events:
        if ev == e.COIN_COLLECTED:
            r += 1.0
        elif ev == e.KILLED_OPPONENT:
            r += 5.0
        elif ev == e.KILLED_SELF:
            r -= 10.0
        elif ev == e.GOT_KILLED:
            r -= 5.0
        elif ev == e.CRATE_DESTROYED:
            r += 0.25
        elif ev == e.COIN_FOUND:
            r += 0.25
        elif ev == e.INVALID_ACTION:
            r -= 0.6
        elif ev == e.WAITED:
            r -= 0.1
        elif ev == e.SURVIVED_ROUND:
            r += 1.0
    for ev in events:
        if ev == 'MOVE_TOWARD_TARGET':
            r += 0.06
        elif ev == 'MOVE_AWAY_TARGET':
            r -= 0.06
        elif ev == 'BOMB_NO_ESCAPE':
            r -= 6.0
        elif ev == 'BOMB_GOOD':
            r += 0.4
        elif ev == 'CENTER_LATE':
            r += 0.03
    return r


def _custom(old_state, action, new_state):
    out = []
    if old_state is None or new_state is None:
        return out
    try:
        oc = old_state.get('coins', []) or []
        if oc and action in ('UP', 'DOWN', 'LEFT', 'RIGHT'):
            _, _, _, (ox, oy) = old_state['self']
            _, _, _, (nx, ny) = new_state['self']
            od = min(abs(int(a) - ox) + abs(int(b) - oy) for (a, b) in oc)
            nc = new_state.get('coins', []) or []
            nd = min(abs(int(a) - nx) + abs(int(b) - ny) for (a, b) in nc) if nc else 0
            if nd < od:
                out.append('MOVE_TOWARD_TARGET')
            elif nd > od:
                out.append('MOVE_AWAY_TARGET')
        if action == 'BOMB' and old_state is not None:
            try:
                s = action_safety(old_state)
                if not s.get('can_escape_if_bomb', True):
                    out.append('BOMB_NO_ESCAPE')
                elif s.get('crates_hit_if_bomb', 0) > 0 or s.get('opps_hit_if_bomb', 0) > 0:
                    out.append('BOMB_GOOD')
            except Exception:
                pass
        if int(new_state.get('step', 0)) > 280 and action in ('UP', 'DOWN', 'LEFT', 'RIGHT'):
            try:
                _, _, _, (ox, oy) = old_state['self']
                _, _, _, (nx, ny) = new_state['self']
                if abs(nx - 8) + abs(ny - 8) < abs(ox - 8) + abs(oy - 8):
                    out.append('CENTER_LATE')
            except Exception:
                pass
    except Exception:
        pass
    return out


def _encode(state):
    if state is None:
        return None, None, 0.0
    try:
        s = action_safety(state)
    except Exception:
        s = None
    img = state_to_tensor(state, s).astype(np.float32)
    sc = scalars_from_state(state, s).astype(np.float32)
    # aux target: mean future danger (t=1) -> dense supervision
    aux = 0.0
    try:
        if s is not None and 'danger' in s and s['danger'].shape[0] > 1:
            aux = float(s['danger'][1].mean())
    except Exception:
        pass
    return img, sc, aux


def setup_training(self):
    self.logger.info('overlord setup_training (CUDA)')
    import torch
    here = os.path.dirname(os.path.abspath(__file__))
    self._overlord_here = here
    self.device = _pick_device(self.logger)
    # Legacy CPU behavior used the smaller batch; any GPU (CUDA) uses full one.
    if os.environ.get('OVERLORD_BATCH', '0') in ('0', ''):
        self.batch_size = BATCH if is_cuda_device(self.device) else BATCH_CPU
    else:
        self.batch_size = BATCH_EFF
    self.utd = UTD
    self.eor_updates = EOR_UPDATES
    self.duty = DutyTimer()
    self.ckpts = CheckpointStore(here, logger=self.logger, save_every=SAVE_EVERY)
    self.logger.info(
        f'overlord device={self.device} batch={self.batch_size} '
        f'utd={self.utd} eor={self.eor_updates}'
    )
    self.q_net = build_model().to(self.device)
    # L40S throughput: autotune conv algorithms once (fixed 17x17 input
    # shape, so benchmark cost is one-time); channels-last keeps NHWC
    # tensor cores fed on Ampere+ (sm_89 here).
    try:
        import torch as _t
        if is_cuda_device(self.device):
            _t.backends.cudnn.benchmark = True
            if CHANNELS_LAST:
                self.q_net = self.q_net.to(memory_format=_t.channels_last)
    except Exception as ex:
        self.logger.debug(f'overlord cudnn/channels-last setup skipped: {ex}')
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
    if CHANNELS_LAST:
        try:
            import torch as _t2
            if is_cuda_device(self.device):
                self.target_net = self.target_net.to(memory_format=_t2.channels_last)
        except Exception:
            pass
    # Optional torch.compile (env-gated, default off). Compiles the hot
    # train net only; CPU act() model and tournament export stay eager.
    if USE_COMPILE:
        try:
            import torch as _t3
            self.q_net = _t3.compile(self.q_net)
            self.logger.info('overlord torch.compile enabled on q_net')
        except Exception as ex:
            self.logger.warning(f'overlord torch.compile failed, eager fallback: {ex}')
    # Optimizer via shared factory (AdamW default = legacy; Lion via env).
    try:
        from dml_optimizer import build_optimizer_for_device, assert_params_on_device
    except ImportError:  # chdir fallback: shared modules live at repo root
        import sys
        _root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from dml_optimizer import build_optimizer_for_device, assert_params_on_device
    self.opt_base_lr = OPT_LR or OPT_LR_DEFAULTS.get(OPT_NAME, LR)
    self.opt_schedule = OPT_SCHEDULE
    _adamw = (OPT_NAME != 'lion')
    _lion_kw = {} if _adamw else {'name': 'lion', 'betas': (0.9, 0.99)}
    self.optimizer = build_optimizer_for_device(
        self.device, self.q_net.parameters(), lr=self.opt_base_lr,
        weight_decay=1e-4, adamw=_adamw, tuned=OPT_TUNED, **_lion_kw
    )
    try:
        assert_params_on_device(self.q_net, self.device, what="q_net")
    except AssertionError as ex:
        self.logger.warning(f'overlord device mismatch: {ex}')
    # AMP on CUDA by default (OVERLORD_AMP=0 for fp32); fp32 + clip on CPU.
    try:
        _dev_is_cuda = is_cuda_device(self.device)
    except Exception:
        _dev_is_cuda = str(getattr(self.device, 'type', self.device)) == 'cuda'
    self.use_amp = bool(_dev_is_cuda and amp_enabled('OVERLORD', default_on=True))
    self.scaler = None
    if self.use_amp:
        try:
            self.scaler = torch.amp.GradScaler('cuda')
        except Exception:
            try:
                self.scaler = torch.cuda.amp.GradScaler()
            except Exception as ex:
                self.logger.warning(f'overlord AMP scaler init failed, fp32 fallback: {ex}')
                self.use_amp = False
                self.scaler = None
    self.logger.info(
        f'overlord optimizer={type(self.optimizer).__name__} lr={self.opt_base_lr} '
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
    self._round_reward = 0.0
    self._round_coins = 0
    self._round_kills = 0
    self._round_suicides = 0
    self._round_killed_self = 0  # own-bomb deaths (KILLED_SELF)
    self._round_got_killed = 0  # enemy-bomb deaths (GOT_KILLED only)

    # Resume full state if present, else legacy weights-only init.
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
                self.optimizer.load_state_dict(ckpt['optimizer'])
            except Exception as ex:
                self.logger.warning(
                    f'overlord optimizer state incompatible '
                    f'({type(self.optimizer).__name__}), fresh optimizer: {ex}'
                )
            if OPT_TUNED and _adamw:
                # load_state_dict restores checkpoint hyperparams; re-apply
                # the tuned preset on top (momentum kept, groups retuned).
                for g in self.optimizer.param_groups:
                    g['eps'] = 1e-4
                    g['weight_decay'] = 1e-4
                self.logger.info('overlord applied tuned AdamW preset (eps=1e-4, decay=1e-4)')
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
                self.logger.warning(f'overlord scaler resume failed, fresh scaler: {ex}')
            self.model.load_state_dict(
                {k: v.cpu() if hasattr(v, 'cpu') else v for k, v in ckpt['q_net'].items()},
                strict=False,
            )
            self.model.eval()
            resumed = True
            self.logger.info(
                f"overlord resumed last.pt ep={self.episode} steps={self.total_steps}"
            )
        except Exception as ex:
            self.logger.warning(f'overlord resume failed, fresh start: {ex}')
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
        self.logger.info('overlord fresh start (no usable checkpoint)')


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
                   'eor': self.eor_updates,
                   'base': int(os.environ.get('OVERLORD_BASE', '96')),
                   'fc': int(os.environ.get('OVERLORD_FC', '512')),
                   'norm': os.environ.get('OVERLORD_NORM', 'bn'),
                   'deep': os.environ.get('OVERLORD_DEEP', '0'),
                   'eps_decay': EPS_DECAY,
                   'channels_last': CHANNELS_LAST, 'compile': USE_COMPILE,
                   'save_every': SAVE_EVERY,
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
    try:
        cwd_pt = os.path.join(os.getcwd(), 'my-saved-model.pt')
        if os.path.abspath(cwd_pt) != os.path.abspath(here_pt):
            torch.save(sd, cwd_pt + '.tmp')
            os.replace(cwd_pt + '.tmp', cwd_pt)
    except Exception:
        pass


def _push(self, img, sc, action, nimg, nsc, reward, done, aux):
    self.n_step_buf.append((img, sc, action, reward, aux))
    if len(self.n_step_buf) < N_STEP and not done:
        return
    R = 0.0
    for i, (_, _, _, r, _) in enumerate(self.n_step_buf):
        R += (GAMMA ** i) * r
    i0, s0, a0, _, aux0 = self.n_step_buf[0]
    self.buffer.add(Transition(i0, s0, a0, nimg, nsc, R, done, aux0))
    if done:
        # drain remainder with bootstrap=None
        while len(self.n_step_buf) > 1:
            self.n_step_buf.popleft()
            R = 0.0
            for i, (_, _, _, r, _) in enumerate(self.n_step_buf):
                R += (GAMMA ** i) * r
            if len(self.n_step_buf) == 0:
                break
            i0, s0, a0, _, aux0 = self.n_step_buf[0]
            self.buffer.add(Transition(i0, s0, a0, None, None, R, True, aux0))
        self.n_step_buf.clear()
    elif len(self.n_step_buf) == N_STEP:
        self.n_step_buf.popleft()


def _update(self):
    if len(self.buffer) < MIN_REPLAY:
        return
    import torch
    self.total_steps += 1
    self.epsilon_steps += 1
    self.epsilon = epsilon_now(self.epsilon_steps)
    beta = min(1.0, 0.4 + 0.6 * self.total_steps / 150000)
    if getattr(self, 'opt_schedule', False):
        apply_schedule(self.optimizer, self.opt_base_lr, self.total_steps,
                       warmup=SCHED_WARMUP, total=SCHED_TOTAL, min_ratio=SCHED_MIN)
    batch, idx, w = self.buffer.sample(self.batch_size, beta)
    dev = self.device
    # One stacked H2D copy per tensor (was: per-sample from_numpy + per-
    # element zeros_like → N small PCIe transfers per update). Stack on
    # CPU, then a single async copy; channels-last for conv inputs.
    use_cl = bool(CHANNELS_LAST and is_cuda_device(dev))
    I_cpu = np.stack([t.img for t in batch])
    S_cpu = np.stack([t.sc for t in batch])
    I = torch.from_numpy(I_cpu).to(dev, non_blocking=True)
    S = torch.from_numpy(S_cpu).to(dev, non_blocking=True)
    if use_cl:
        I = I.to(memory_format=torch.channels_last)
    has_next = [t.next_img is not None for t in batch]
    if all(has_next):
        NI = torch.from_numpy(np.stack([t.next_img for t in batch])).to(dev, non_blocking=True)
        NS = torch.from_numpy(np.stack([t.next_sc for t in batch])).to(dev, non_blocking=True)
    else:
        NI_cpu = np.stack([t.next_img if t.next_img is not None else batch[0].img for t in batch])
        NS_cpu = np.stack([t.next_sc if t.next_sc is not None else batch[0].sc for t in batch])
        NI = torch.from_numpy(NI_cpu).to(dev, non_blocking=True)
        NS = torch.from_numpy(NS_cpu).to(dev, non_blocking=True)
    if use_cl:
        NI = NI.to(memory_format=torch.channels_last)
    A = torch.tensor([ACTION_TO_IDX[t.action] for t in batch], device=dev).unsqueeze(1)
    R = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=dev)
    D = torch.tensor([0.0 if t.done else 1.0 for t in batch], dtype=torch.float32, device=dev)
    W = torch.tensor(w, dtype=torch.float32, device=dev)
    AUX = torch.tensor([t.aux for t in batch], dtype=torch.float32, device=dev)
    gam_n = GAMMA ** N_STEP
    self.q_net.train()

    def loss_fn():
        q, aux_pred = self.q_net(I, S)
        q = q.gather(1, A).squeeze(1)
        with torch.no_grad():
            qn, _ = self.q_net(NI, NS)
            a_star = qn.argmax(dim=1, keepdim=True)
            qt, _ = self.target_net(NI, NS)
            q_next = qt.gather(1, a_star).squeeze(1)
            target = R + D * gam_n * q_next
        td = target - q
        # Huber (delta=1) both heads: MSE detonated on ±100 TD errors in
        # sentinel Stage 4 (lossMed 480 -> 74k). Bounded gradients instead.
        td_loss = (W * torch.nn.functional.smooth_l1_loss(td, torch.zeros_like(td), reduction='none')).mean()
        aux_loss = torch.nn.functional.smooth_l1_loss(aux_pred, AUX, reduction='none').mean()
        return td_loss + AUX_W * aux_loss, td
    use_amp = bool(getattr(self, 'use_amp', False) and getattr(self, 'scaler', None) is not None)
    if use_amp:
        # AMP on CUDA (default); fp32 path below is bit-identical to legacy.
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            loss, td = loss_fn()
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
    else:
        loss, td = loss_fn()
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
    except Exception:
        self.last_loss = float(loss.item())
    if self.total_steps % TARGET_SYNC == 0:
        self.target_net.load_state_dict(self.q_net.state_dict())
        # keep CPU inference model in sync (act stays on CPU)
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
    try:
        if e.COIN_COLLECTED in events:
            self._round_coins += 1
        if e.KILLED_OPPONENT in events:
            self._round_kills += 1
        # Own bomb takes precedence: suicide rounds emit BOTH events
        # (removal loop tags every death GOT_KILLED) — partition exactly.
        if e.KILLED_SELF in events:
            self._round_suicides += 1
            self._round_killed_self += 1
        elif e.GOT_KILLED in events:
            self._round_suicides += 1
            self._round_got_killed += 1
    except Exception:
        pass
    img, sc, aux = _encode(old_game_state)
    nimg, nsc, _ = _encode(new_game_state) if new_game_state is not None else (None, None, 0.0)
    done = new_game_state is None
    _push(self, img, sc, self_action, nimg, nsc, r, done, aux)
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
    try:
        if e.COIN_COLLECTED in events:
            self._round_coins += 1
        if e.KILLED_OPPONENT in events:
            self._round_kills += 1
        # Own bomb takes precedence: suicide rounds emit BOTH events
        # (removal loop tags every death GOT_KILLED) — partition exactly.
        if e.KILLED_SELF in events:
            self._round_suicides += 1
            self._round_killed_self += 1
        elif e.GOT_KILLED in events:
            self._round_suicides += 1
            self._round_got_killed += 1
    except Exception:
        pass
    img, sc, aux = _encode(last_game_state)
    _push(self, img, sc, last_action, None, None, r, True, aux)
    # _push with done=True already drains n-step buffer
    self.n_step_buf.clear()
    for _ in range(getattr(self, 'eor_updates', 6)):
        with self.duty.measure():
            _update(self)
    gpu_ms, wall_ms, duty_pct = self.duty.end_round()
    self.episode += 1
    here = getattr(self, '_overlord_here', os.path.dirname(os.path.abspath(__file__)))

    rr = float(self._round_reward)
    self.ema_reward = rr if self.ema_reward is None else (1 - EMA_ALPHA) * self.ema_reward + EMA_ALPHA * rr
    improved = self.best_ema is None or self.ema_reward > self.best_ema
    if improved:
        self.best_ema = float(self.ema_reward)

    # Weight norm (divergence guard: E04 lesson — |w| 447→2139 signaled
    # MSE blowup; healthy Huber runs deflate to ~15-30 and hold).
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
            'round_reward': round(rr, 3),
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
        self.logger.warning(f'overlord metrics log failed: {ex}')

    try:
        payload = _payload(self)
        self.ckpts.maybe_save(self.episode, payload, improved)
        _export_tournament_model(here, self.q_net)
    except Exception as ex:
        self.logger.warning(f'overlord save failed: {ex}')
    self.logger.info(
        f'overlord ep={self.episode} buf={len(self.buffer)} loss={self.last_loss:.4f} '
        f'r={rr:.2f} ema={self.ema_reward:.2f} wnorm={wnorm:.1f} duty={duty_pct}% dev={self.device}'
    )
    self._round_reward = 0.0
    self._round_coins = 0
    self._round_kills = 0
    self._round_suicides = 0
    self._round_killed_self = 0
    self._round_got_killed = 0

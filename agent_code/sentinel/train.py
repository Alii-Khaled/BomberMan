"""Sentinel training: N-step Double Dueling DQN + PER on CUDA (Colab NVIDIA GPUs).

Device: CUDA when available (default, auto), else CPU.
act()/callbacks model stays on CPU; only q_net/target_net train on CUDA.
AMP autocast+GradScaler on CUDA by default (SENTINEL_AMP=0 for fp32);
fp32 + grad clip on CPU.

Checkpoints (agent_code/sentinel/checkpoints/): last.pt every 25 rounds,
best.pt on EMA-reward improvement, ep_NNNNNN.pt every 200 rounds.
setup_training() resumes last.pt (optimizer + episode + epsilon), else
legacy my-saved-model.pt weights-only. Metrics -> runs/metrics.csv
(columns double as report figures for Tasks 1-4).
"""
from collections import deque, namedtuple
import csv
import os
import random
import numpy as np

import events as e

try:
    from .callbacks import ACTION_LIST, ACTION_TO_IDX
    from .features_mlp import state_to_features, FEATURE_DIM
    from .model import build_model
    from .safety import action_safety
    from .device import get_device, is_cuda
    from .checkpointing import (
        save_last, save_best, save_snapshot, load_checkpoint,
        last_path, SAVE_EVERY, SNAPSHOT_EVERY,
    )
    from dml_trainkit import update_config, DutyTimer, amp_enabled
except ImportError:  # direct cwd execution fallback (SequentialAgentBackend chdir)
    from callbacks import ACTION_LIST, ACTION_TO_IDX
    from features_mlp import state_to_features, FEATURE_DIM
    from model import build_model
    from safety import action_safety
    from device import get_device, is_cuda
    from checkpointing import (
        save_last, save_best, save_snapshot, load_checkpoint,
        last_path, SAVE_EVERY, SNAPSHOT_EVERY,
    )
    try:
        from dml_trainkit import update_config, DutyTimer, amp_enabled
    except ImportError:
        import sys
        _root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from dml_trainkit import update_config, DutyTimer, amp_enabled

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

# hyperparams (reliable preset — HPO grid for report: LR {3e-4, 1e-3}, N_STEP {1,3,5})
# Stage 5 (Arm A): N_STEP 5 so the -10 suicide penalty (4 steps after BOMB)
# falls inside the n-step return window (was 3 < BOMB_TIMER 4: no direct credit).
GAMMA = 0.95
N_STEP = 5
LR = 1e-3
BATCH = 256
TARGET_SYNC = 1000
EPS_START, EPS_END, EPS_DECAY = 1.0, 0.05, 50000
BUFFER_SIZE = 100000
# Optimizer select (env-gated; defaults preserve Stage 1-3 behavior exactly):
#   SENTINEL_OPT: 'adam' (stock Adam lr=1e-3, foreach on CUDA) | 'lion' (Lion, benchmark winner)
#   SENTINEL_LR: base LR override (default per optimizer)
#   SENTINEL_SCHEDULE: '1' enables warmup+cosine schedule on total_steps
#   SENTINEL_TUNED: '1' applies the DQN task preset to Adam (eps=1e-4,
#     decay=1e-4); re-applied after checkpoint resume (load_state_dict
#     would otherwise restore the old eps=1e-8/decay=0 groups).
#   SENTINEL_AMP: '1' (default) enables autocast+GradScaler on CUDA; '0' = fp32.
#   SENTINEL_DEVICE: 'cuda'|'cpu'|'auto' (default auto = CUDA when available).
# Stage 4 launch: SENTINEL_OPT=lion SENTINEL_SCHEDULE=1.
# Stage 5 Arm A launch: SENTINEL_OPT=adam SENTINEL_TUNED=1 (no schedule: fixed LR).
OPT_NAME = os.environ.get('SENTINEL_OPT', 'adam')
OPT_LR = float(os.environ.get('SENTINEL_LR', '0.0') or 0.0)
OPT_SCHEDULE = os.environ.get('SENTINEL_SCHEDULE', '0') == '1'
OPT_TUNED = os.environ.get('SENTINEL_TUNED', '0') == '1'
OPT_LR_DEFAULTS = {'adam': LR, 'lion': 1e-3}
SCHED_WARMUP, SCHED_TOTAL, SCHED_MIN = 5000, 200000, 0.1
# GPU-utilization config (env-gated; defaults = legacy behavior exactly):
#   SENTINEL_UTD: updates per env step (default 1)
#   SENTINEL_BATCH: batch override (default 0 = BATCH)
#   SENTINEL_EOR_UPDATES: extra updates at round end (default 4)
# Stage 4 launch: SENTINEL_UTD=3 SENTINEL_BATCH=512 SENTINEL_EOR_UPDATES=16.
UTD, BATCH_EFF, EOR_UPDATES = update_config('SENTINEL', BATCH, 4)
MIN_REPLAY = 2000
EMA_ALPHA = 0.05  # best-tracking smoothing for round reward


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
            r += 0.2
        elif ev == e.COIN_FOUND:
            r += 0.2
        elif ev == e.INVALID_ACTION:
            r -= 0.5
        elif ev == e.WAITED:
            r -= 0.08
        elif ev == e.SURVIVED_ROUND:
            r += 0.5
        elif ev == e.BOMB_DROPPED:
            # small shaping decided by context (computed by caller via custom events)
            r += 0.0
    # custom dense events appended by game_events_occurred
    for ev in events:
        if ev == 'MOVE_TOWARD_TARGET':
            r += 0.05
        elif ev == 'MOVE_AWAY_TARGET':
            r -= 0.05
        elif ev == 'BOMB_NO_ESCAPE':
            r -= 5.0
        elif ev == 'BOMB_GOOD':
            r += 0.3
        elif ev == 'LOOP_PENALTY':
            r -= 0.2
    return r


def _custom_events(old_state, action, new_state):
    """Potential-ish dense shaping based on states only (plus bomb safety)."""
    out = []
    if old_state is None or new_state is None:
        return out
    try:
        oc = [c for c in (old_state.get('coins', []) or [])]
        nc = [c for c in (new_state.get('coins', []) or [])]
        _, _, _, (ox, oy) = old_state['self']
        _, _, _, (nx, ny) = new_state['self']
        if oc:
            od = min(abs(int(a) - ox) + abs(int(b) - oy) for (a, b) in oc)
            if nc:
                nd = min(abs(int(a) - nx) + abs(int(b) - ny) for (a, b) in nc)
            else:
                nd = 0  # collected
            if action in ('UP', 'DOWN', 'LEFT', 'RIGHT'):
                if nd < od:
                    out.append('MOVE_TOWARD_TARGET')
                elif nd > od:
                    out.append('MOVE_AWAY_TARGET')
        if action == 'BOMB':
            try:
                s = action_safety(old_state)
                if not s.get('can_escape_if_bomb', True):
                    out.append('BOMB_NO_ESCAPE')
                elif s.get('crates_hit_if_bomb', 0) > 0 or s.get('opps_hit_if_bomb', 0) > 0:
                    out.append('BOMB_GOOD')
            except Exception:
                pass
    except Exception:
        pass
    return out


def _here():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except Exception:
        return os.getcwd()


def _metrics_path(here):
    d = os.path.join(here, 'runs')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, 'metrics.csv')


def _log_metrics(here, row):
    """Append a row; widens the header if new columns appear (e.g. duty
    columns added mid-curriculum — old rows backfill as empty)."""
    path = _metrics_path(here)
    if not os.path.isfile(path):
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)
        return
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        old_fields = list(reader.fieldnames or [])
        old_rows = list(reader)
    fields = old_fields + [k for k in row.keys() if k not in old_fields]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(old_rows)
        w.writerow({k: row.get(k, '') for k in fields})


def _export_tournament_model(here, q_net):
    """Write my-saved-model.pt (state_dict on CPU) for tournament/docker."""
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


def setup_training(self):
    self.logger.info('sentinel setup_training (CUDA)')
    here = _here()
    self._sentinel_here = here
    self.device = get_device(self.logger)
    self.logger.info(f'sentinel device={self.device} cuda={is_cuda(self.device)} batch={BATCH}')
    self.q_net = build_model().to(self.device)
    # callbacks model (act) stays on CPU for 0.5s tournament guarantee
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
    # CUDA-default optimizer: stock Adam/AdamW with foreach kernels on CUDA
    # (legacy DMLAdam classes remain only for old-checkpoint resume).
    try:
        from dml_optimizer import (
            build_optimizer_for_device,
            assert_params_on_device,
        )
    except ImportError:  # SequentialAgentBackend chdir: dml_optimizer lives at repo root
        import sys
        _root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from dml_optimizer import build_optimizer_for_device, assert_params_on_device
    self.optimizer = build_optimizer_for_device(
        self.device, self.q_net.parameters(),
        lr=OPT_LR or OPT_LR_DEFAULTS.get(OPT_NAME, LR),
        name=OPT_NAME if OPT_NAME in ('adam', 'lion') else 'adam',
        tuned=OPT_TUNED,
    )
    self.opt_base_lr = OPT_LR or OPT_LR_DEFAULTS.get(OPT_NAME, LR)
    self.opt_schedule = OPT_SCHEDULE
    # AMP on CUDA by default (SENTINEL_AMP=0 for fp32); fp32 + clip on CPU.
    try:
        _dev_is_cuda = is_cuda(self.device)
    except Exception:
        _dev_is_cuda = str(getattr(self.device, 'type', self.device)) == 'cuda'
    self.use_amp = bool(_dev_is_cuda and amp_enabled('SENTINEL', default_on=True))
    self.scaler = None
    if self.use_amp:
        try:
            import torch
            self.scaler = torch.amp.GradScaler('cuda')
        except Exception:
            try:
                import torch
                self.scaler = torch.cuda.amp.GradScaler()
            except Exception as ex:
                self.logger.warning(f'sentinel AMP scaler init failed, fp32 fallback: {ex}')
                self.use_amp = False
                self.scaler = None
    try:
        assert_params_on_device(self.q_net, self.device, what="q_net")
    except AssertionError as ex:
        self.logger.warning(f'sentinel device mismatch: {ex}')
    self.logger.info(
        f'sentinel optimizer={type(self.optimizer).__name__} lr={self.opt_base_lr} '
        f'schedule={self.opt_schedule} amp={self.use_amp} device={self.device}'
    )
    self.buffer = PERBuffer()
    self.n_step_buf = deque(maxlen=N_STEP)
    self.train_batch = BATCH_EFF
    self.utd = UTD
    self.eor_updates = EOR_UPDATES
    self.duty = DutyTimer()
    self.logger.info(
        f'sentinel utd={self.utd} batch={self.train_batch} eor={self.eor_updates}'
    )
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
    self._round_killed_self = 0  # E14a: own-bomb deaths (KILLED_SELF)
    self._round_got_killed = 0  # E14a: enemy-bomb deaths (GOT_KILLED only)
    # expose epsilon to callbacks.act() for exploration (train mode only)
    self.epsilon = EPS_START

    # Resume full state if present, else legacy weights-only.
    resumed = False
    ckpt = load_checkpoint(last_path(here))
    if ckpt and isinstance(ckpt, dict) and 'q_net' in ckpt:
        try:
            import torch
            dev = self.device
            self.q_net.load_state_dict(
                {k: v.to(dev) if hasattr(v, 'to') else v for k, v in ckpt['q_net'].items()},
                strict=False,
            )
            self.target_net.load_state_dict(self.q_net.state_dict())
            # Best-effort optimizer resume: old ckpts may hold DMLAdam state
            # (int steps) while we now use stock Adam (Tensor steps), or vice
            # versa — momentum transfers when shapes match, else fresh start.
            try:
                self.optimizer.load_state_dict(ckpt['optimizer'])
            except Exception as ex:
                self.logger.warning(
                    f'sentinel optimizer state incompatible '
                    f'({type(self.optimizer).__name__}), fresh optimizer: {ex}'
                )
            if OPT_TUNED:
                # load_state_dict restores the checkpoint's hyperparams
                # (eps=1e-8/decay=0); re-apply the tuned preset on top.
                # State (momentum) is kept, only group hypers change.
                for g in self.optimizer.param_groups:
                    g['eps'] = 1e-4
                    g['weight_decay'] = 1e-4
                self.logger.info('sentinel applied tuned Adam preset (eps=1e-4, decay=1e-4)')
            # optimizer tensors back to training device
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
            # AMP scaler resume (old DML/fp32 ckpts have none — fresh scaler).
            try:
                if getattr(self, 'use_amp', False) and self.scaler is not None \
                        and isinstance(ckpt.get('scaler'), dict):
                    self.scaler.load_state_dict(ckpt['scaler'])
            except Exception as ex:
                self.logger.warning(f'sentinel scaler resume failed, fresh scaler: {ex}')
            self.model.load_state_dict(
                {k: v.cpu() if hasattr(v, 'cpu') else v for k, v in ckpt['q_net'].items()},
                strict=False,
            )
            self.model.eval()
            resumed = True
            self.logger.info(
                f"sentinel resumed {last_path(here)} ep={self.episode} steps={self.total_steps}"
            )
        except Exception as ex:
            self.logger.warning(f'sentinel resume failed, fresh start: {ex}')
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
        self.logger.info('sentinel fresh start (no usable checkpoint)')


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
        'epsilon': self.epsilon,
        'best_ema': self.best_ema,
        'ema_reward': self.ema_reward,
        'config': {'gamma': GAMMA, 'n_step': N_STEP, 'lr': LR, 'batch': BATCH,
                   'eps': (EPS_START, EPS_END, EPS_DECAY),
                   'opt': getattr(self, 'optimizer', None) is not None and type(
                       self.optimizer).__name__,
                   'opt_base_lr': getattr(self, 'opt_base_lr', LR),
                   'opt_schedule': getattr(self, 'opt_schedule', False),
                   'amp': bool(getattr(self, 'use_amp', False))},
    }
    try:
        if getattr(self, 'use_amp', False) and getattr(self, 'scaler', None) is not None:
            payload['scaler'] = self.scaler.state_dict()
    except Exception:
        pass
    return payload


def _push_nstep(self, state, action, next_state, reward, done):
    self.n_step_buf.append((state, action, reward))
    if len(self.n_step_buf) < N_STEP and not done:
        return
    # compute n-step return
    R = 0.0
    for i, (_, _, r) in enumerate(self.n_step_buf):
        R += (GAMMA ** i) * r
    s0, a0, _ = self.n_step_buf[0]
    self.buffer.add(Transition(s0, a0, next_state, R, done))
    if done:
        while len(self.n_step_buf) > 1:
            self.n_step_buf.popleft()
            R = 0.0
            for i, (_, _, r) in enumerate(self.n_step_buf):
                R += (GAMMA ** i) * r
            if len(self.n_step_buf) == 0:
                break
            s0, a0, _ = self.n_step_buf[0]
            self.buffer.add(Transition(s0, a0, None, R, True))
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
    beta = min(1.0, 0.4 + 0.6 * self.total_steps / 100000)
    if getattr(self, 'opt_schedule', False):
        try:
            from dml_optimizer import schedule_factor
        except ImportError:
            import sys
            _root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            if _root not in sys.path:
                sys.path.insert(0, _root)
            from dml_optimizer import schedule_factor
        lr_now = self.opt_base_lr * schedule_factor(
            self.total_steps, warmup=SCHED_WARMUP, total=SCHED_TOTAL,
            min_ratio=SCHED_MIN)
        for g in self.optimizer.param_groups:
            g['lr'] = lr_now
    batch, idx, w = self.buffer.sample(getattr(self, 'train_batch', BATCH), beta)
    dev = self.device
    # Single host->device transfer per tensor (PCIe is the bottleneck, not FLOPs).
    S = torch.from_numpy(np.stack([t.state for t in batch])).to(dev)
    zero = torch.from_numpy(batch[0].state).to(dev)
    NS = torch.stack([
        torch.from_numpy(t.next_state).to(dev) if t.next_state is not None else zero
        for t in batch
    ])
    A = torch.tensor([ACTION_TO_IDX[t.action] for t in batch], device=dev).unsqueeze(1)
    R = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=dev)
    D = torch.tensor([0.0 if t.done else 1.0 for t in batch], dtype=torch.float32, device=dev)
    W = torch.tensor(w, dtype=torch.float32, device=dev)
    gam_n = GAMMA ** N_STEP
    self.q_net.train()
    use_amp = bool(getattr(self, 'use_amp', False) and getattr(self, 'scaler', None) is not None)
    if use_amp:
        # AMP on CUDA (default); fp32 path below is bit-identical to legacy.
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            q = self.q_net(S).gather(1, A).squeeze(1)
            with torch.no_grad():
                a_star = self.q_net(NS).argmax(dim=1, keepdim=True)
                q_next = self.target_net(NS).gather(1, a_star).squeeze(1)
                target = R + D * gam_n * q_next
            td = target - q
            # Huber (delta=1): MSE detonated on ±100 TD errors in Stage 4
            # (lossMed 480 -> 74k); bounded gradient keeps steps sane under PER.
            loss = (W * torch.nn.functional.smooth_l1_loss(td, torch.zeros_like(td), reduction='none')).mean()
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
    else:
        q = self.q_net(S).gather(1, A).squeeze(1)
        with torch.no_grad():
            a_star = self.q_net(NS).argmax(dim=1, keepdim=True)
            q_next = self.target_net(NS).gather(1, a_star).squeeze(1)
            target = R + D * gam_n * q_next
        td = target - q
        # Huber (delta=1): MSE detonated on ±100 TD errors in Stage 4
        # (lossMed 480 -> 74k); bounded gradient keeps steps sane under PER.
        loss = (W * torch.nn.functional.smooth_l1_loss(td, torch.zeros_like(td), reduction='none')).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()
    try:
        self.buffer.update(idx, td.detach().cpu().numpy())
    except Exception:
        pass
    self.last_loss = float(loss.detach().cpu().item()) if hasattr(loss, 'detach') else float(loss)
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
    # augment with dense custom events
    try:
        events = list(events) + _custom_events(old_game_state, self_action, new_game_state)
    except Exception:
        pass
    r = reward_from_events(self, events, old_game_state, self_action, new_game_state)
    self._round_reward += float(r)
    try:
        if e.COIN_COLLECTED in events:
            self._round_coins += 1
        if e.KILLED_OPPONENT in events:
            self._round_kills += 1
        # E14a attribution: suicide rounds emit BOTH KILLED_SELF (own blast)
        # and GOT_KILLED (removal loop tags every death) — own bomb takes
        # precedence so the two counters partition _round_suicides exactly.
        if e.KILLED_SELF in events:
            self._round_suicides += 1
            self._round_killed_self += 1
        elif e.GOT_KILLED in events:
            self._round_suicides += 1
            self._round_got_killed += 1
    except Exception:
        pass
    try:
        s = state_to_features(old_game_state, action_safety(old_game_state) if old_game_state else None)
    except Exception:
        s = state_to_features(old_game_state)
    try:
        ns = state_to_features(new_game_state, action_safety(new_game_state) if new_game_state else None)
    except Exception:
        ns = state_to_features(new_game_state)
    done = new_game_state is None
    if done:
        ns = None
    # flush n-step (store raw arrays)
    _push_nstep(self, s, self_action, ns, r, done)
    for _ in range(getattr(self, 'utd', 1)):
        with self.duty.measure():
            _update(self)


def end_of_round(self, last_game_state, last_action, events):
    try:
        events = list(events) + _custom_events(last_game_state, last_action, None)
    except Exception:
        pass
    r = reward_from_events(self, events, last_game_state, last_action, None)
    self._round_reward += float(r)
    try:
        if e.COIN_COLLECTED in events:
            self._round_coins += 1
        if e.KILLED_OPPONENT in events:
            self._round_kills += 1
        # E14a attribution: suicide rounds emit BOTH KILLED_SELF (own blast)
        # and GOT_KILLED (removal loop tags every death) — own bomb takes
        # precedence so the two counters partition _round_suicides exactly.
        if e.KILLED_SELF in events:
            self._round_suicides += 1
            self._round_killed_self += 1
        elif e.GOT_KILLED in events:
            self._round_suicides += 1
            self._round_got_killed += 1
    except Exception:
        pass
    # suicide already penalized via events; add survival bonus if alive implicitly via SURVIVED_ROUND
    try:
        s = state_to_features(last_game_state, action_safety(last_game_state) if last_game_state else None)
    except Exception:
        s = state_to_features(last_game_state)
    _push_nstep(self, s, last_action, None, r, True)
    # _push_nstep with done=True already drains; ensure empty
    self.n_step_buf.clear()
    for _ in range(getattr(self, 'eor_updates', 4)):
        with self.duty.measure():
            _update(self)
    gpu_ms, wall_ms, duty_pct = self.duty.end_round()
    self.episode += 1
    here = getattr(self, '_sentinel_here', _here())

    # EMA best tracking (report metric: avg round reward trend)
    rr = float(self._round_reward)
    self.ema_reward = rr if self.ema_reward is None else (1 - EMA_ALPHA) * self.ema_reward + EMA_ALPHA * rr
    improved = self.best_ema is None or self.ema_reward > self.best_ema
    if improved:
        self.best_ema = float(self.ema_reward)

    # Metrics row for report figures (Tasks 1-4 progression)
    try:
        _log_metrics(here, {
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
            'device': str(self.device),
            'gpu_ms': gpu_ms,
            'wall_ms': wall_ms,
            'duty_pct': duty_pct,
        })
    except Exception as ex:
        self.logger.warning(f'sentinel metrics log failed: {ex}')

    # Checkpoints: full state (resumable, every round — 642KB, cheap) +
    # tournament export (weights only). best/snapshot stay sparse.
    try:
        payload = _payload(self)
        save_last(here, payload)
        if improved:
            save_best(here, payload)
        if self.episode % SNAPSHOT_EVERY == 0:
            save_snapshot(here, self.episode, payload)
        _export_tournament_model(here, self.q_net)
    except Exception as ex:
        self.logger.warning(f'sentinel save failed: {ex}')
    self.logger.info(
        f'sentinel ep={self.episode} buf={len(self.buffer)} loss={self.last_loss:.4f} '
        f'eps={self.epsilon:.3f} r={rr:.2f} ema={self.ema_reward:.2f} dev={self.device}'
    )
    self._round_reward = 0.0
    self._round_coins = 0
    self._round_kills = 0
    self._round_suicides = 0
    self._round_killed_self = 0
    self._round_got_killed = 0

"""E100 ARBITER-RL: KL-anchored REINFORCE fine-tune of pi.

Only the move fallback is a pi decision (the search owns bombs); steps
where search/tactical acted carry no trace entry. Rewards are the exact
engine objective plus reaper's death/invalid/wait terms; returns-to-go
over the whole round feed the traced steps. KL(pi || frozen BC prior)
anchors calibration. Runs on the callbacks' device (main CUDA device
with CPU fallback, E100c); the net is tiny.
"""
import copy
import os

import numpy as np

import events as e


def _reward(events):
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


def _env(name, default, cast=float):
    try:
        return cast(os.environ.get(name, str(default)))
    except Exception:
        return default


def setup_training(self):
    self._rl = True
    self._rl_trace = []
    self._rl_rewards = []
    self._rl_ep = 0
    self._rl_gamma = _env('ARBITER_RL_GAMMA', 0.99)
    self._rl_lr = _env('ARBITER_RL_LR', 1e-4)
    self._rl_beta = _env('ARBITER_RL_BETA', 0.02)
    self._rl_trunk = os.environ.get('ARBITER_RL_TRUNK', '1') == '1'
    self._rl_save_every = int(_env('ARBITER_RL_SAVE_EVERY', 25, int))
    # E102 RL loop v2 (all env-gated; defaults = E100 behavior):
    # STABLE — adaptive KL anchor + hard revert guard;
    # EPOCHS — extra passes over the round batch; CRITIC — V-as-baseline
    # advantages + value regression.
    self._rl_stable = os.environ.get('ARBITER_RL_STABLE', '0') == '1'
    self._rl_kl_hi = _env('ARBITER_RL_KL_HI', 0.5)
    self._rl_kl_lo = _env('ARBITER_RL_KL_LO', 0.1)
    self._rl_kl_rev = _env('ARBITER_RL_KL_REVERT', 1.0)
    self._rl_beta_min = _env('ARBITER_RL_BETA_MIN', 0.02)
    self._rl_beta_max = _env('ARBITER_RL_BETA_MAX', 1.0)
    self._rl_epochs = int(_env('ARBITER_RL_EPOCHS', 1, int))
    self._rl_critic = os.environ.get('ARBITER_RL_CRITIC', '0') == '1'
    self._rl_vf_coef = _env('ARBITER_RL_VF_COEF', 0.5)
    self._rl_good = None
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    self._rl_out = os.environ.get(
        'ARBITER_RL_OUT', os.path.join(here, 'my-saved-model.pt'))
    if not os.path.isabs(self._rl_out):
        self._rl_out = os.path.join(root, self._rl_out)
    self._rl_csv = os.environ.get('ARBITER_RL_CSV', '')
    if self._rl_csv and not os.path.isabs(self._rl_csv):
        self._rl_csv = os.path.join(root, self._rl_csv)
    self._rl_hist = []
    try:
        self.model.train()
    except Exception:
        pass
    self._rl_ref = copy.deepcopy(self.model).eval()
    for p in self._rl_ref.parameters():
        p.requires_grad_(False)
    import torch
    params = list(self.model.parameters()) if self._rl_trunk else \
        list(self.model.pi.parameters()) + list(self.model.v.parameters())
    self._rl_opt = torch.optim.Adam(params, lr=self._rl_lr)
    self.epsilon = 0.0
    try:
        self.logger.info(
            'arbiter_rl setup lr=%.1e beta=%.3f trunk=%s out=%s '
            'stable=%d critic=%d epochs=%d'
            % (self._rl_lr, self._rl_beta, self._rl_trunk, self._rl_out,
               self._rl_stable, self._rl_critic, self._rl_epochs))
    except Exception:
        pass


def game_events_occurred(self, old_game_state, self_action, new_game_state,
                         events):
    try:
        self._rl_rewards.append(_reward(events))
    except Exception:
        pass


def end_of_round(self, last_game_state, last_action, events):
    try:
        step = int(last_game_state.get('step', 0)) if last_game_state else 0
    except Exception:
        step = 0
    rew = getattr(self, '_rl_rewards', [])
    if step and len(rew) < step:
        rew.append(_reward(events))
    # E107: diag dispatch (ported from agent_code/arbiter/train.py —
    # diag_dump_round was missing here so ARBITER_DIAG jsonl never
    # appeared for the NG agent).
    try:
        from .callbacks import diag_dump_round
        diag_dump_round(self, last_action, events)
    except Exception:
        pass
    trace = getattr(self, '_rl_trace', [])
    self._rl_ep = int(getattr(self, '_rl_ep', 0)) + 1
    if trace and rew:
        try:
            # E104 B2 shaping knob: survived a round in which we planted
            # bombs -> terminal bonus (counter-weights KILLED_SELF -8;
            # default 0 = ship behavior).
            _sb = _env('ARBITER_RL_BOMB_SURVIVE_BONUS', 0.0)
            if _sb > 0 \
                    and 'SURVIVED_ROUND' in ' '.join(
                        str(ev) for ev in (events or [])) \
                    and any(t[3][t[4]] == 5 for t in trace):
                rew[-1] += _sb
        except Exception:
            pass
        try:
            _rl_update(self, trace, rew)
        except Exception as ex:
            try:
                self.logger.warning('arbiter_rl update failed: %s' % ex)
            except Exception:
                pass
    self._rl_trace = []
    self._rl_rewards = []
    if self._rl_save_every and self._rl_ep % self._rl_save_every == 0:
        _rl_save(self)


def _rl_update(self, trace, rew):
    import torch
    import torch.nn.functional as F
    T = len(rew)
    ret = np.zeros(T)
    run = 0.0
    for i in range(T - 1, -1, -1):
        run = rew[i] + self._rl_gamma * run
        ret[i] = run
    ret = np.clip(ret, -20.0, 20.0)
    base = float(np.mean(ret))
    feats, idxs, js, steps_k = [], [], [], []
    for (_rnd, step, f, idx, j) in trace:
        i = min(max(step - 1, 0), T - 1)
        feats.append(f)
        idxs.append(idx)
        js.append(j)
        steps_k.append(i)
    dev = next(self.model.parameters()).device
    xb = torch.from_numpy(np.stack(feats)).to(dev)
    ret_t = torch.tensor([ret[i] for i in steps_k],
                         dtype=torch.float32, device=dev)
    critic = getattr(self, '_rl_critic', False)
    self.model.train()
    logits, vv = self.model(xb)
    if critic:
        adv_arr = (ret_t.detach().cpu().numpy().astype(np.float64)
                   - vv.detach().cpu().numpy().astype(np.float64))
    else:
        adv_arr = np.asarray([ret[i] - base for i in steps_k],
                             dtype=np.float64)
    # E100b stability: unit-variance advantages (raw returns span +-20 and
    # destabilized the first run: KL blew to 12 and G1 fell to 3.55).
    adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-6)
    adv = torch.tensor(adv_arr, dtype=torch.float32, device=dev)
    with torch.no_grad():
        rlogits, _ = self._rl_ref(xb)
        rprob = F.softmax(rlogits, dim=-1)
    stable = getattr(self, '_rl_stable', False)
    epochs = max(1, int(getattr(self, '_rl_epochs', 1)))
    pg_val, kl_val = 0.0, 0.0
    for _ in range(epochs):
        logits, vv = self.model(xb)
        logp = []
        for k in range(len(feats)):
            sel = torch.tensor(idxs[k], dtype=torch.long, device=dev)
            logp.append(F.log_softmax(logits[k][sel], dim=0)[js[k]])
        logp = torch.stack(logp)
        pg = -(logp * adv).mean()
        kl = F.kl_div(F.log_softmax(logits, dim=-1), rprob,
                      reduction='batchmean')
        loss = pg + self._rl_beta * kl
        if critic:
            loss = loss + self._rl_vf_coef * F.mse_loss(vv, ret_t)
        self._rl_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self._rl_opt.step()
        pg_val, kl_val = float(pg.detach()), float(kl.detach())
        if stable:
            _rl_stable_post(self, kl_val)
    self._rl_hist.append((pg_val, kl_val))
    if self._rl_csv:
        try:
            new = not os.path.exists(self._rl_csv)
            with open(self._rl_csv, 'a') as fh:
                if new:
                    fh.write('ep,steps,ret_mean,pg,kl\n')
                fh.write('%d,%d,%.4f,%.4f,%.4f\n'
                         % (self._rl_ep, len(trace), base,
                            pg_val, kl_val))
        except Exception:
            pass


def _rl_stable_post(self, kl_val):
    """E102 stable mode: revert guard + adaptive beta on the measured KL."""
    import torch
    try:
        if kl_val > self._rl_kl_rev and self._rl_good is not None:
            self.model.load_state_dict(self._rl_good)
            params = [p for g in self._rl_opt.param_groups
                      for p in g['params']]
            self._rl_opt = torch.optim.Adam(params, lr=self._rl_lr)
            self._rl_beta = min(self._rl_beta_max, self._rl_beta * 2.0)
            try:
                self.logger.warning(
                    'arbiter_rl KL %.2f > %.2f: reverted to last good '
                    '(beta -> %.3f)' % (kl_val, self._rl_kl_rev,
                                        self._rl_beta))
            except Exception:
                pass
            return
        self._rl_good = {k: v.detach().cpu().clone()
                         for k, v in self.model.state_dict().items()}
        if kl_val > self._rl_kl_hi:
            self._rl_beta = min(self._rl_beta_max, self._rl_beta * 1.5)
        elif kl_val < self._rl_kl_lo:
            self._rl_beta = max(self._rl_beta_min, self._rl_beta * 0.9)
    except Exception:
        pass


def _rl_save(self):
    import torch
    try:
        sd = {k: v.cpu() for k, v in self.model.state_dict().items()}
        tmp = self._rl_out + '.part'
        torch.save(sd, tmp)
        os.replace(tmp, self._rl_out)
        ep = int(getattr(self, '_rl_ep', 0))
        if ep:
            snap = '%s.ep%03d' % (self._rl_out, ep)
            tmp2 = snap + '.part'
            torch.save(sd, tmp2)
            os.replace(tmp2, snap)
        try:
            self.logger.info('arbiter_rl saved %s (ep %d)'
                             % (self._rl_out, ep))
        except Exception:
            pass
    except Exception as ex:
        try:
            self.logger.warning('arbiter_rl save failed: %s' % ex)
        except Exception:
            pass

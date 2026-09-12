"""E100 ARBITER-RL: KL-anchored REINFORCE fine-tune of pi.

Only the move fallback is a pi decision (the search owns bombs); steps
where search/tactical acted carry no trace entry. Rewards are the exact
engine objective plus reaper's death/invalid/wait terms; returns-to-go
over the whole round feed the traced steps. KL(pi || frozen BC prior)
anchors calibration. CPU-only by design (the callbacks forward the model
on CPU; the net is tiny).
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
            'arbiter_rl setup lr=%.1e beta=%.3f trunk=%s out=%s'
            % (self._rl_lr, self._rl_beta, self._rl_trunk, self._rl_out))
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
    trace = getattr(self, '_rl_trace', [])
    self._rl_ep = int(getattr(self, '_rl_ep', 0)) + 1
    if trace and rew:
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
    feats, idxs, js, advs = [], [], [], []
    for (_rnd, step, f, idx, j) in trace:
        i = min(max(step - 1, 0), T - 1)
        feats.append(f)
        idxs.append(idx)
        js.append(j)
        advs.append(ret[i] - base)
    adv_arr = np.asarray(advs, dtype=np.float64)
    # E100b stability: unit-variance advantages (raw returns span +-20 and
    # destabilized the first run: KL blew to 12 and G1 fell to 3.55).
    adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-6)
    xb = torch.from_numpy(np.stack(feats))
    self.model.train()
    logits, _v = self.model(xb)
    logp = []
    for k in range(len(feats)):
        sel = torch.tensor(idxs[k], dtype=torch.long)
        logp.append(F.log_softmax(logits[k][sel], dim=0)[js[k]])
    logp = torch.stack(logp)
    adv = torch.tensor(adv_arr, dtype=torch.float32)
    pg = -(logp * adv).mean()
    with torch.no_grad():
        rlogits, _ = self._rl_ref(xb)
    kl = F.kl_div(F.log_softmax(logits, dim=-1),
                  F.softmax(rlogits, dim=-1), reduction='batchmean')
    loss = pg + self._rl_beta * kl
    self._rl_opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
    self._rl_opt.step()
    self._rl_hist.append((float(pg.detach()), float(kl.detach())))
    if self._rl_csv:
        try:
            new = not os.path.exists(self._rl_csv)
            with open(self._rl_csv, 'a') as fh:
                if new:
                    fh.write('ep,steps,ret_mean,pg,kl\n')
                fh.write('%d,%d,%.4f,%.4f,%.4f\n'
                         % (self._rl_ep, len(trace), base,
                            float(pg.detach()), float(kl.detach())))
        except Exception:
            pass


def _rl_save(self):
    import torch
    try:
        tmp = self._rl_out + '.part'
        torch.save({k: v.cpu() for k, v in self.model.state_dict().items()},
                   tmp)
        os.replace(tmp, self._rl_out)
        try:
            self.logger.info('arbiter_rl saved %s (ep %d)'
                             % (self._rl_out, self._rl_ep))
        except Exception:
            pass
    except Exception as ex:
        try:
            self.logger.warning('arbiter_rl save failed: %s' % ex)
        except Exception:
            pass

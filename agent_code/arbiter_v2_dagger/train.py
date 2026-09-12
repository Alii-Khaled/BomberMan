"""Arbiter_dagger training callbacks: flush per-round (feats, acts).

Recording happens in act() (see callbacks.py) so the fatal step of a
death round is captured too (game_events_occurred never sees deaths).
Rounds flush to: <ARBITER_DEMO_DIR>/round_%06d.npz
  {feats: float32[N,98], acts: uint8[N]}  (reaper-format, pi-only rows)
Consume via scripts/arbiter_extract.py (teacher id 4 = arbiter_self).
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _demo_dir():
    d = os.environ.get('ARBITER_DEMO_DIR', '').strip()
    if not d:
        d = os.path.join(_ROOT, 'results', 'demos', 'arbiter_v2_self')
    if not os.path.isabs(d):
        d = os.path.join(_ROOT, d)
    os.makedirs(d, exist_ok=True)
    return d


def setup_training(self):
    self._dag_demo_dir = _demo_dir()
    self._dag_round_id = 0
    try:
        _ex = [f for f in os.listdir(self._dag_demo_dir)
               if f.startswith('round_') and f.endswith('.npz')]
        _nums = [int(f[6:12]) for f in _ex if f[6:12].isdigit()]
        if _nums:
            self._dag_round_id = max(_nums)
    except Exception:
        pass
    self.logger.info(f'arbiter_dagger recording to {self._dag_demo_dir}')


def game_events_occurred(self, old_game_state, self_action,
                         new_game_state, events):
    return


def end_of_round(self, last_game_state, last_action, events):
    n = len(getattr(self, '_dag_feats', []))
    if n:
        self._dag_round_id += 1
        path = os.path.join(self._dag_demo_dir,
                            f'round_{self._dag_round_id:06d}.npz')
        try:
            payload = {
                'feats': np.asarray(self._dag_feats, dtype=np.float32),
                'acts': np.asarray(self._dag_acts, dtype=np.uint8),
            }
            _aw = getattr(self, '_dag_acts_w', None)
            if _aw:
                payload['acts_w'] = np.asarray(_aw, dtype=np.uint8)
            np.savez_compressed(path, **payload)
            self.logger.info(f'arbiter_dagger saved {n} samples to {path}')
        except Exception as ex:
            self.logger.warning(f'arbiter_dagger save failed: {ex}')
    self._dag_feats = []
    self._dag_acts = []
    self._dag_acts_w = []

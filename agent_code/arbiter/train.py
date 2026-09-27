"""Framework training hooks for the offline-trained Arbiter policy.

Pretraining lives in scripts/pretrain_arbiter.py. These callbacks do not
perform online learning.
"""
import events as e

try:  # diag dump lives in callbacks; train.py is the dispatched hook
    from .callbacks import diag_dump_round, _DIAG
except Exception:  # pragma: no cover
    diag_dump_round = None
    _DIAG = ''


def setup_training(self):
    try:
        self.logger.info('arbiter setup_training: online loop lands in P2; '
                         'P0/P1 train offline via scripts/pretrain_arbiter.py')
    except Exception:
        pass
    self.episode = 0


def game_events_occurred(self, old_game_state, self_action, new_game_state,
                         events):
    if _DIAG:
        try:
            self._diag_last_events = [str(ev) for ev in (events or [])][-24:]
        except Exception:
            pass


def end_of_round(self, last_game_state, last_action, events):
    try:
        self.episode = int(getattr(self, 'episode', 0)) + 1
    except Exception:
        pass
    if diag_dump_round is not None:
        diag_dump_round(self, last_action, events)

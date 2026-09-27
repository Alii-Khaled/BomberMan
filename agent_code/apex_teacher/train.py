"""Apex_teacher training hooks: flush the final round's demo buffer.

act() in callbacks.py buffers (img, sc, act) per step and flushes on
round-change, but the last round of an invocation never sees a next
round and its npz would be dropped. end_of_round() flushes
unconditionally.

game_events_occurred is a no-op: recording happens in act() (the
teacher action is known there; here we would only see our own
delegated action again).
"""


def setup_training(self):
    pass


def game_events_occurred(self, old_game_state, self_action, new_game_state,
                          events):
    pass


def end_of_round(self, last_game_state, last_action, events):
    try:
        from agent_code.apex_teacher.callbacks import _flush
    except ImportError:
        try:
            from callbacks import _flush
        except ImportError:
            return
    try:
        _flush(self)
    except Exception:
        pass

"""Solo-DAgger training hooks: flush the final round's buffer.

Same pattern as apex_teacher/train.py (E51 lesson: the last round of an
invocation never sees a round-change, so flush it explicitly).
"""


def setup_training(self):
    pass


def game_events_occurred(self, old_game_state, self_action, new_game_state,
                          events):
    pass


def end_of_round(self, last_game_state, last_action, events):
    try:
        from agent_code.solo_dagger.callbacks import _flush
    except ImportError:
        try:
            from callbacks import _flush
        except ImportError:
            return
    try:
        _flush(self)
    except Exception:
        pass

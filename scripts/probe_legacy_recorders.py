#!/usr/bin/env python3
"""Smoke-test the training-only recorders against the current Harvey package.

Run from the repository root. Demonstration output uses a temporary directory.
This does not train a model or evaluate tournament performance.
"""

import logging
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def state():
    field = np.zeros((17, 17), dtype=np.int32)
    field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = -1
    return {'field': field, 'coins': [(3, 1)], 'step': 1, 'round': 1,
            'self': ('Harvey', 0, True, (1, 1)), 'others': [], 'bombs': [],
            'explosion_map': np.zeros_like(field)}


def main():
    with tempfile.TemporaryDirectory(prefix='recorder-probe-') as tmp:
        os.environ['APEX_DEMO_OUT'] = str(Path(tmp) / 'solo')
        os.environ['ARBITER_DEMO_DIR'] = str(Path(tmp) / 'exit')
        os.environ['SOLO_LABEL'] = 'coin_collector_agent'
        os.environ['HARVEY_DIAG'] = ''
        os.environ.pop('HARVEY_WEIGHTS', None)
        os.environ.pop('HARVEY_MODEL', None)

        from agent_code.Harvey import callbacks as harvey
        from agent_code.solo_dagger import callbacks as solo, train as solo_train
        from agent_code.exit_recorder import callbacks as exit_cb, train as exit_train

        game_state = state()
        logger = logging.getLogger('recorder-probe')
        actions = {'UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB'}

        solo_agent = SimpleNamespace(train=False, logger=logger)
        solo.setup(solo_agent)
        assert solo_agent._dag_arb is harvey
        solo_action = solo.act(solo_agent, game_state)
        assert solo_action in actions and solo_agent._buf
        solo_train.end_of_round(solo_agent, game_state, solo_action, [])
        solo_files = list((Path(tmp) / 'solo').rglob('round_*.npz'))
        assert len(solo_files) == 1
        with np.load(solo_files[0]) as rows:
            assert rows['img'].shape[1:] == (12, 17, 17)
            assert rows['sc'].shape[1:] == (16,)
            assert len(rows['act']) == len(rows['sc'])

        exit_agent = SimpleNamespace(train=False, logger=logger)
        exit_cb.setup(exit_agent)
        assert exit_agent._arb is harvey
        exit_action = exit_cb.act(exit_agent, game_state)
        assert exit_action in actions
        features = exit_agent._state_to_features(
            game_state, exit_agent._action_safety(game_state))
        assert features.shape == (3566,)
        exit_train.setup_training(exit_agent)
        exit_agent._exit_feats.append(np.asarray(features, dtype=np.float32))
        exit_agent._exit_labels.append(4)
        exit_agent._exit_acts.append(4)
        exit_train.end_of_round(exit_agent, game_state, exit_action, [])
        exit_files = list((Path(tmp) / 'exit').rglob('round_*.npz'))
        assert len(exit_files) == 1
        with np.load(exit_files[0]) as rows:
            assert rows['feats'].shape[1:] == (3566,)
            assert len(rows['labels']) == len(rows['acts']) == len(rows['feats'])
    print('PASS: solo_dagger and exit_recorder delegate to Harvey and flush data')


if __name__ == '__main__':
    main()

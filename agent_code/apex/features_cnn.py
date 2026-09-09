"""CNN spatial features for overlord: 12x17x17 float32. Numpy only, vectorized."""
import numpy as np

N_CHANNELS = 12


def state_to_tensor(game_state, safety_info=None, power=3):
    """Channels:
    0 wall, 1 crate, 2 coin, 3 self, 4 others, 5 bomb_timer/4, 6 bomb_is_self,
    7 explosion, 8 danger_t0, 9 danger_t1, 10 danger_t2, 11 blast_if_bomb_now
    """
    if game_state is None:
        return np.zeros((N_CHANNELS, 17, 17), dtype=np.float32)
    arena = np.asarray(game_state['field'])
    W, H = arena.shape[0], arena.shape[1]
    assert (W, H) == (17, 17), f"unexpected arena {arena.shape}"
    _, _, _, (x, y) = game_state['self']
    bombs = game_state.get('bombs', []) or []
    coins = game_state.get('coins', []) or []
    others = game_state.get('others', []) or []
    exp_map = np.asarray(game_state.get('explosion_map', np.zeros((W, H))))

    t = np.zeros((N_CHANNELS, W, H), dtype=np.float32)
    # arena is indexed [x,y]
    t[0] = (arena == -1).astype(np.float32)
    t[1] = (arena == 1).astype(np.float32)
    for (cx, cy) in coins:
        t[2, int(cx), int(cy)] = 1.0
    t[3, int(x), int(y)] = 1.0
    for (n, s, b, xy) in others:
        t[4, int(xy[0]), int(xy[1])] = 1.0
    # bombs: need owner? game_state bombs lack owner; mark timer + assume not self
    for (xy, timer) in bombs:
        xb, yb = int(xy[0]), int(xy[1])
        t[5, xb, yb] = min(float(timer), 4.0) / 4.0
    t[7] = np.clip(exp_map, 0, 1).astype(np.float32)
    if safety_info is not None and 'danger' in safety_info:
        d = safety_info['danger']
        if d.shape[0] > 0:
            t[8] = d[0].astype(np.float32)
        if d.shape[0] > 1:
            t[9] = d[1].astype(np.float32)
        if d.shape[0] > 2:
            t[10] = d[2].astype(np.float32)
        # blast_if_bomb
        blast = safety_info.get('blast_if_bomb', set())
        for (cx, cy) in blast:
            if 0 <= cx < W and 0 <= cy < H:
                t[11, cx, cy] = 1.0
    else:
        # fallback: compute blast inline (wall-aware)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            t[11, x, y] = 1.0
            for i in range(1, power + 1):
                nx, ny = x + dx * i, y + dy * i
                if not (0 <= nx < W and 0 <= ny < H) or arena[nx, ny] == -1:
                    break
                t[11, nx, ny] = 1.0
    # transpose to [C,H,W]? No: keep [C,x,y] consistently; model treats dim1,2 as spatial.
    return t

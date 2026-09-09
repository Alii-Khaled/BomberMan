"""Apex extra scalars: opponent-model + trap + economy, rotation-INVARIANT.

Deliberately aggregate-only (max/any/min over directions) so train-time
augmentation never needs scalar permutation — the per-direction masks that
caused reaper's probe forensics #1/#2 are excluded by design. All outputs
are invariant under board rotation by construction.

8 dims (float32):
 0 trap_any: 1.0 if any adjacent free spot traps an opponent
 1 min_opp_esc: min opponent escape distance /8 (1.0 if none)
 2 max_crates_adj: max crates hit bombing any adjacent free spot /4
 3 opp_in_deadend: nearest opponent stands in a dead-end (free_nb<=1)
 4 score_margin: (self-best_opp)/10 clipped to [-1,1]
 5 hunt_flag: crates+coins<=6 or step>200 (warden trigger)
 6 own_in_deadend: own free_nb<=1
 7 nearest_opp: min Manhattan dist /16 (capped 1.0; 1.0 if none)
"""
import numpy as np

try:
    from .safety_reaper import (true_blast, escape_bfs, danger_no_explosion,
                                opp_can_escape, with_hypothetical_bomb)
except ImportError:  # direct execution fallback
    from safety_reaper import (true_blast, escape_bfs, danger_no_explosion,
                               opp_can_escape, with_hypothetical_bomb)

N_EXTRA = 8
_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _free_nb(arena, bomb_set, x, y):
    W, H = arena.shape[0], arena.shape[1]
    n = 0
    for dx, dy in _DELTAS:
        nx, ny = x + dx, y + dy
        if 0 <= nx < W and 0 <= ny < H and arena[nx, ny] == 0 \
                and (nx, ny) not in bomb_set:
            n += 1
    return n


def extra_scalars(game_state):
    out = np.zeros(N_EXTRA, dtype=np.float32)
    if game_state is None:
        return out
    try:
        arena = np.asarray(game_state['field'])
        _, self_score, bombs_left, (x, y) = game_state['self']
        x, y = int(x), int(y)
        bombs = game_state.get('bombs', []) or []
        coins = game_state.get('coins', []) or []
        others = game_state.get('others', []) or []
        step = int(game_state.get('step', 0))
        others_xy = [(int(pxy[0]), int(pxy[1])) for (_, _, _, pxy) in others]
        bomb_set = set((int(bxy[0]), int(bxy[1])) for (bxy, _) in bombs)

        # adjacent free spots (bombable positions next step)
        spots = []
        for dx, dy in _DELTAS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1] \
                    and arena[nx, ny] == 0 and (nx, ny) not in bomb_set:
                spots.append((nx, ny))

        max_crates, trap_any, min_oe = 0.0, 0.0, float('inf')
        if bombs_left and spots:
            try:
                shared = danger_no_explosion(arena, bombs)
            except Exception:
                shared = None
            for nx, ny in spots:
                try:
                    hit = sum(1 for (cx, cy) in true_blast(arena, nx, ny)
                              if arena[cx, cy] == 1)
                except Exception:
                    hit = 0
                max_crates = max(max_crates, min(float(hit), 4.0) / 4.0)
                if not others_xy or shared is None:
                    continue
                for ox, oy in others_xy:
                    if (ox, oy) not in set(true_blast(arena, nx, ny)):
                        continue
                    try:
                        can, dist_o = opp_can_escape(
                            arena, bombs, (ox, oy), (nx, ny),
                            others_xy, danger=shared)
                        do_f = float(dist_o)
                    except Exception:
                        continue
                    if do_f < min_oe:
                        min_oe = do_f
                    if not can:
                        trap_any = 1.0
        out[0] = trap_any
        out[1] = min(min_oe, 8.0) / 8.0 if min_oe != float('inf') else 1.0
        out[2] = max_crates

        if others_xy:
            dists = [abs(ox - x) + abs(oy - y) for (ox, oy) in others_xy]
            j = int(np.argmin(dists))
            ox, oy = others_xy[j]
            out[3] = 1.0 if _free_nb(arena, bomb_set, ox, oy) <= 1 else 0.0
            out[7] = min(float(dists[j]), 16.0) / 16.0
            try:
                best_opp = max(float(s) for (_, s, _, _) in others)
                out[4] = max(-1.0, min(1.0, (float(self_score) - best_opp) / 10.0))
            except Exception:
                pass
        n_crates = int((arena == 1).sum())
        out[5] = 1.0 if (n_crates + len(coins) <= 6 or step > 200) else 0.0
        out[6] = 1.0 if _free_nb(arena, bomb_set, x, y) <= 1 else 0.0
    except Exception:
        pass
    return out

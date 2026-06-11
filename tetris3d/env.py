"""A 3D Tetris environment with a *placement* (afterstate) action space.

Board is W x D x H (x, y, z). Pieces fall along -z. A horizontal z-plane that is
completely filled clears, and everything above it drops down by one.

Instead of micro-controlling a falling piece, one RL step = placing the current
piece: the agent chooses among all legal final placements (orientation, x, y),
the piece is hard-dropped and locked, full layers clear, and the next piece
appears. This is the standard, far more learnable formulation for Tetris.

Key methods:
    reset(prefill_p)          -> start a game (optionally with a near-full board)
    enumerate_placements()    -> (list[Placement], feature_matrix[N, PLACEMENT_FEAT_DIM])
    place(i)                  -> apply placement i: (reward, done, info)
    state_features()          -> value-network input for the current state
"""
import os

import numpy as np

from . import pieces as P

# Board dimensions (override with TETRIS_W / TETRIS_D / TETRIS_H env vars).
# Smaller cross-sections clear far more easily, which matters a lot for the
# competitive mode (attacks only happen when layers actually clear).
W = int(os.environ.get("TETRIS_W", 5))
D = int(os.environ.get("TETRIS_D", 5))
H = int(os.environ.get("TETRIS_H", 10))

# Human game score for clearing k layers at once (reported in the GUI/metrics).
LAYER_SCORE = {0: 0, 1: 100, 2: 300, 3: 600, 4: 1000}

# --- RL reward (per placement) -------------------------------------------
# Clearing layers is the dominant, durable reward. Board-health shaping only
# nudges placement quality between clears and is clipped so it never dwarfs a
# clear. There is no "alive" reward (it would reward stalling) and no telescoping
# fill potential (it cancelled out the incentive to actually complete a layer).
# Clearing layers must be the clearly dominant strategy, so the clear reward is
# large and the height penalty is small (otherwise the agent settles for a low,
# clean board and never bothers to clear). Holes are still punished (they block
# clears) and fill-progress densely rewards building toward a completion.
CLEAR_REWARD = {0: 0.0, 1: 100.0, 2: 300.0, 3: 600.0, 4: 1200.0}
GAMEOVER_PENALTY = -10.0
PLACE_REWARD = 0.0           # no constant survival reward (it's just a value offset)
HEALTH_W_HEIGHT = -0.1       # per unit of aggregate-height increase (mild)
HEALTH_W_HOLES = -4.0        # per hole created (holes block clears: punish hard)
HEALTH_W_BUMPY = -0.3        # per unit of surface-roughness increase
HEALTH_CLIP = 60.0          # generous clip (only guards pathological garbage cases)
FILL_PROGRESS_W = 1.0       # dense reward per cell added to the most-filled layer
                            # (one-way: only rewards progress toward a clear, so it
                            #  does not telescope away like a potential term would)

PLACEMENT_FEAT_DIM = 8
STATE_FEAT_DIM = 5 + 2 * P.NUM_PIECE_TYPES

# Versus mode: clearing k layers sends this many garbage layers to the opponent
# (you must clear 2+ at once to attack, which rewards stacking for multi-clears).
GARBAGE_SENT = {0: 0, 1: 0, 2: 1, 3: 2, 4: 4}


_ZRANGE = np.arange(1, H + 1)


def _heightmap(board):
    """Per-column top height (z index of highest filled cell + 1), vectorized."""
    occ = board > 0
    return (occ * _ZRANGE).max(axis=2)


def board_quality(board):
    """Return (agg_height, holes, bumpiness, max_height, total_fill).

    Fully vectorized (this is the hot path: called for every candidate
    placement of every step). A "hole" is an empty cell at or below a column's
    top; since every filled cell lies below the top, holes == height - fill.
    """
    occ = board > 0
    heights = (occ * _ZRANGE).max(axis=2)
    col_fill = occ.sum(axis=2)
    agg = int(heights.sum())
    mx = int(heights.max())
    holes = int((heights - col_fill).sum())
    bump = int(np.abs(np.diff(heights, axis=0)).sum() + np.abs(np.diff(heights, axis=1)).sum())
    fill = int(occ.sum())
    return agg, holes, bump, mx, fill


class Placement:
    __slots__ = ("cells", "pos", "lines")

    def __init__(self, cells, pos, lines):
        self.cells = cells      # (4, 3) oriented offsets
        self.pos = pos          # (x, y, z) resting corner
        self.lines = lines      # layers cleared by this placement


class Tetris3DEnv:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ---- lifecycle -----------------------------------------------------
    def reset(self, prefill_p=0.0):
        self.board = np.zeros((W, D, H), dtype=np.int8)
        self.board_color = -np.ones((W, D, H), dtype=np.int8)
        self.score = 0
        self.cleared_layers = 0
        self.pieces_placed = 0
        self.last_lines = 0
        self.done = False
        if prefill_p > 0 and self.rng.random() < prefill_p:
            self._prefill()
        self.cur_type = int(self.rng.integers(P.NUM_PIECE_TYPES))
        self.next_type = int(self.rng.integers(P.NUM_PIECE_TYPES))
        self.last_pos = None     # corner of the most recent placement (for GUI)
        self.last_cells = None
        if not self.enumerate_placements()[0]:
            self.done = True
        return self.state_features()

    def _prefill(self):
        """Curriculum: a bottom layer that is full except for an OPEN well.

        The well is a contiguous empty strip reachable from above, so dropped
        pieces land at z=0 inside it and progressively fill it -- completing the
        layer clears it. This (unlike scattered covered holes) gives the agent
        frequent, learnable first clears plus dense fill-progress reward. The
        caller scales how often this happens via prefill_p.
        """
        self.board[:, :, 0] = 1
        self.board_color[:, :, 0] = self.rng.integers(P.NUM_PIECE_TYPES, size=(W, D))
        # Empty a small open well so a single well-placed piece can complete the
        # layer (an easy, frequent first clear); the well size grows the variety.
        wx = int(self.rng.integers(2, 4))       # 2-3
        wy = int(self.rng.integers(2, 4))
        x0 = int(self.rng.integers(0, W - wx + 1))
        y0 = int(self.rng.integers(0, D - wy + 1))
        self.board[x0:x0 + wx, y0:y0 + wy, 0] = 0
        self.board_color[x0:x0 + wx, y0:y0 + wy, 0] = -1

    # ---- placement enumeration ----------------------------------------
    def enumerate_placements(self):
        """All legal placements for the current piece + their feature matrix.

        Fully vectorized for speed: resting heights for a whole orientation are
        computed from the column heightmap at once, then every candidate's
        afterstate features are computed in a single batched numpy pass (the
        per-candidate numpy overhead was the bottleneck).
        """
        heights = _heightmap(self.board)            # (W, D)
        cand_cells = []     # list of (4, 3) world-cell arrays
        cand_pos = []       # list of (x, y, pz)
        cand_oriented = []  # the orientation cells (for Placement / place())
        for info in P.ORIENT_INFO[self.cur_type]:
            gx, gy = W - info.ex, D - info.ey
            if gx <= 0 or gy <= 0:
                continue
            # pz_grid[x, y] = resting z for top-left corner (x, y)
            pz_grid = np.full((gx, gy), -(10 ** 6), dtype=np.int64)
            for k in range(info.fox.size):
                sub = heights[info.fox[k]:info.fox[k] + gx,
                              info.foy[k]:info.foy[k] + gy] - info.fbot[k]
                np.maximum(pz_grid, sub, out=pz_grid)
            np.maximum(pz_grid, 0, out=pz_grid)
            valid = (pz_grid + info.max_oz) < H
            xs, ys = np.nonzero(valid)
            for x, y in zip(xs.tolist(), ys.tolist()):
                pz = int(pz_grid[x, y])
                cand_pos.append((x, y, pz))
                cand_cells.append(info.cells + (x, y, pz))
                cand_oriented.append(info.cells)

        n = len(cand_cells)
        if n == 0:
            return [], np.zeros((0, PLACEMENT_FEAT_DIM), np.float32)

        # Batched afterstate occupancy: (n, W, D, H).
        occ = np.broadcast_to(self.board > 0, (n, W, D, H)).copy()
        world = np.stack(cand_cells)                # (n, 4, 3)
        ci = np.repeat(np.arange(n), 4)
        flat = world.reshape(-1, 3)
        occ[ci, flat[:, 0], flat[:, 1], flat[:, 2]] = True

        feats, lines = self._batch_features(occ, np.array([p[2] for p in cand_pos]))
        placements = [Placement(cand_oriented[i], np.asarray(cand_pos[i]), int(lines[i]))
                      for i in range(n)]
        return placements, feats

    def _batch_features(self, occ, landing):
        """Afterstate features for a batch of candidate boards occ (n,W,D,H)."""
        n = occ.shape[0]
        area = W * D
        full = occ.all(axis=(1, 2))                 # (n, H) full layers
        lines = full.sum(axis=1)                    # (n,)
        heights = (occ * _ZRANGE).max(axis=3)       # (n, W, D)
        col_fill = occ.sum(axis=3)
        agg = heights.reshape(n, -1).sum(1)
        mx = heights.reshape(n, -1).max(1)
        holes = (heights - col_fill).reshape(n, -1).sum(1)
        bump = (np.abs(np.diff(heights, axis=1)).reshape(n, -1).sum(1)
                + np.abs(np.diff(heights, axis=2)).reshape(n, -1).sum(1))
        fill = occ.reshape(n, -1).sum(1)
        max_layer = occ.sum(axis=(1, 2)).max(axis=1) / area   # most-full layer fraction
        feats = np.empty((n, PLACEMENT_FEAT_DIM), np.float32)
        feats[:, 0] = agg / (area * H)
        feats[:, 1] = holes / (area * H)
        feats[:, 2] = bump / (area * H)
        feats[:, 3] = mx / H
        feats[:, 4] = fill / (area * H)
        feats[:, 5] = lines / 4.0
        feats[:, 6] = landing / H
        feats[:, 7] = max_layer
        # For layer-clearing candidates the above describes the pre-clear board;
        # the lines feature already flags them, which is what the policy needs.
        return feats, lines

    # ---- state features (value net input) ------------------------------
    def state_features(self):
        agg, holes, bump, mx, fill = board_quality(self.board)
        area = W * D
        base = np.array([
            agg / (area * H), holes / (area * H), bump / (area * H),
            mx / H, fill / (area * H),
        ], dtype=np.float32)
        cur = np.zeros(P.NUM_PIECE_TYPES, np.float32)
        nxt = np.zeros(P.NUM_PIECE_TYPES, np.float32)
        cur[self.cur_type] = 1.0
        nxt[self.next_type] = 1.0
        return np.concatenate([base, cur, nxt])

    def _apply_with_color(self, cells, pos, ptype):
        """Lock `cells` at `pos` into the real board+color and clear layers."""
        wc = cells + np.asarray(pos, dtype=np.int64)
        self.board[wc[:, 0], wc[:, 1], wc[:, 2]] = 1
        self.board_color[wc[:, 0], wc[:, 1], wc[:, 2]] = ptype
        full = (self.board > 0).all(axis=(0, 1))
        n = int(full.sum())
        if n:
            keep = np.nonzero(~full)[0]
            nb = np.zeros_like(self.board)
            nc = -np.ones_like(self.board_color)
            nb[:, :, :keep.size] = self.board[:, :, keep]
            nc[:, :, :keep.size] = self.board_color[:, :, keep]
            self.board, self.board_color = nb, nc
        return n

    # ---- apply a placement --------------------------------------------
    def place(self, placement, placements_cache=None):
        if self.done:
            return 0.0, True, self._info()

        before_h = board_quality(self.board)
        maxlayer_before = int((self.board > 0).sum(axis=(0, 1)).max())
        self.last_cells = placement.cells
        self.last_pos = placement.pos.copy()
        lines = self._apply_with_color(placement.cells, placement.pos, self.cur_type)
        self.last_lines = lines
        self.pieces_placed += 1
        self.cleared_layers += lines
        self.score += LAYER_SCORE[min(lines, 4)]

        after_h = board_quality(self.board)
        maxlayer_after = int((self.board > 0).sum(axis=(0, 1)).max())
        phi_before = (HEALTH_W_HEIGHT * before_h[0] + HEALTH_W_HOLES * before_h[1]
                      + HEALTH_W_BUMPY * before_h[2])
        phi_after = (HEALTH_W_HEIGHT * after_h[0] + HEALTH_W_HOLES * after_h[1]
                     + HEALTH_W_BUMPY * after_h[2])
        health_delta = float(np.clip(phi_after - phi_before, -HEALTH_CLIP, HEALTH_CLIP))
        fill_progress = FILL_PROGRESS_W * max(0, maxlayer_after - maxlayer_before)

        reward = CLEAR_REWARD[min(lines, 4)] + PLACE_REWARD + health_delta + fill_progress

        # Advance to the next piece.
        self.cur_type = self.next_type
        self.next_type = int(self.rng.integers(P.NUM_PIECE_TYPES))
        if not self.enumerate_placements()[0]:
            self.done = True
            reward += GAMEOVER_PENALTY

        return reward, self.done, self._info()

    def add_garbage(self, num):
        """Push `num` garbage layers up from the bottom (versus attack).

        Each garbage layer is full except one random hole, so it cannot clear by
        itself. If the existing stack is shoved above the ceiling, the player
        tops out (returns True) and the game is lost.
        """
        if num <= 0 or self.done:
            return False
        overflow = bool((self.board[:, :, H - num:] > 0).any())
        nb = np.zeros_like(self.board)
        nc = -np.ones_like(self.board_color)
        nb[:, :, num:] = self.board[:, :, :H - num]
        nc[:, :, num:] = self.board_color[:, :, :H - num]
        for z in range(num):
            nb[:, :, z] = 1
            nc[:, :, z] = P.GARBAGE_COLOR_ID
            hx, hy = int(self.rng.integers(W)), int(self.rng.integers(D))
            nb[hx, hy, z] = 0
            nc[hx, hy, z] = -1
        self.board, self.board_color = nb, nc
        if overflow or not self.enumerate_placements()[0]:
            self.done = True
        return self.done

    def _info(self):
        return {
            "score": self.score,
            "cleared_layers": self.cleared_layers,
            "pieces_placed": self.pieces_placed,
            "lines": getattr(self, "last_lines", 0),
        }

    @staticmethod
    def state_dim():
        return STATE_FEAT_DIM

    @staticmethod
    def placement_dim():
        return PLACEMENT_FEAT_DIM

    # ---- GUI snapshot --------------------------------------------------
    def render_state(self, piece_cells=None):
        cells = []
        cb = self.board_color
        occ = self.board > 0
        for x in range(W):
            for y in range(D):
                for z in range(H):
                    if occ[x, y, z]:
                        cells.append([int(x), int(y), int(z), int(cb[x, y, z])])
        piece = []
        if piece_cells is not None:
            for cx, cy, cz in piece_cells:
                if 0 <= cz < H:
                    piece.append([int(cx), int(cy), int(cz)])
        return {
            "dims": [W, D, H],
            "cells": cells,
            "piece": piece,
            "piece_type": int(self.cur_type) if not self.done else -1,
            "next_type": int(self.next_type),
            "score": int(self.score),
            "cleared_layers": int(self.cleared_layers),
            "pieces_placed": int(self.pieces_placed),
            "done": bool(self.done),
            "colors": P.PIECE_COLORS,
        }

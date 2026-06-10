"""A 3D Tetris environment.

Board is W x D x H (x, y, z). Pieces fall along -z. A horizontal z-plane that
is completely filled is cleared (Tetris "line" generalized to a "layer"), and
everything above it drops down by one.

Action space (per step): the agent first applies one move/rotate/drop action,
then gravity pulls the piece down by one. When the piece can no longer descend
it locks, full layers clear, and a new piece spawns.
"""
import numpy as np

from . import pieces as P

# Board dimensions.
W, D, H = 6, 6, 12

# Discrete action set.
ACTIONS = [
    "move_-x", "move_+x", "move_-y", "move_+y",
    "rot_x", "rot_y", "rot_z",
    "noop", "hard_drop",
]
NUM_ACTIONS = len(ACTIONS)

# Score awarded for clearing k layers at once (rewards multi-clears).
LAYER_SCORE = {0: 0, 1: 100, 2: 300, 3: 600, 4: 1000}

# Reward-shaping weights (potential-based, on top of the game score).
# Tuned to push the 5x5 agent toward actually completing & clearing layers:
# survival alone is barely rewarded, while near-complete layers are strongly
# attractive and holes (which block clears) are punished hard.
ALIVE_REWARD = 0.2
GAMEOVER_PENALTY = -50.0
W_HEIGHT = -0.25   # penalize aggregate column height (mild: allow building up)
W_HOLES = -2.5     # penalize covered empty cells (they prevent clears)
W_BUMPY = -0.3     # penalize uneven surface
W_FILL = 30.0      # reward filling layers toward completion (steep: fill_fraction**4)
FILL_POW = 4


class Piece:
    def __init__(self, ptype, cells, pos):
        self.ptype = ptype          # int piece type id
        self.cells = cells          # (4, 3) normalized offsets
        self.pos = np.array(pos, dtype=np.int64)  # (x, y, z) corner position

    def world_cells(self, cells=None, pos=None):
        cells = self.cells if cells is None else cells
        pos = self.pos if pos is None else pos
        return cells + pos


class Tetris3DEnv:
    def __init__(self, seed=None, shaping=True):
        self.rng = np.random.default_rng(seed)
        self.shaping = shaping
        self.reset()

    # ---- core state ----------------------------------------------------
    def reset(self):
        self.board = np.zeros((W, D, H), dtype=np.int8)
        self.board_color = -np.ones((W, D, H), dtype=np.int8)  # piece type per cell for GUI
        self.score = 0
        self.cleared_layers = 0
        self.pieces_placed = 0
        self.steps = 0
        self.done = False
        self.next_type = int(self.rng.integers(P.NUM_PIECE_TYPES))
        self._spawn()
        return self.get_obs()

    def _spawn(self):
        ptype = self.next_type
        self.next_type = int(self.rng.integers(P.NUM_PIECE_TYPES))
        cells = P.spawn_cells(ptype)
        extent = cells.max(axis=0) + 1
        px = (W - extent[0]) // 2
        py = (D - extent[1]) // 2
        pz = H - extent[2]
        self.piece = Piece(ptype, cells, (px, py, pz))
        if self._collides(self.piece.cells, self.piece.pos):
            self.done = True

    # ---- collision / placement ----------------------------------------
    def _collides(self, cells, pos):
        wc = cells + pos
        if (wc[:, 0] < 0).any() or (wc[:, 0] >= W).any():
            return True
        if (wc[:, 1] < 0).any() or (wc[:, 1] >= D).any():
            return True
        if (wc[:, 2] < 0).any() or (wc[:, 2] >= H).any():
            return True
        return bool(self.board[wc[:, 0], wc[:, 1], wc[:, 2]].any())

    def _lock_and_clear(self):
        wc = self.piece.world_cells()
        self.board[wc[:, 0], wc[:, 1], wc[:, 2]] = 1
        self.board_color[wc[:, 0], wc[:, 1], wc[:, 2]] = self.piece.ptype
        self.pieces_placed += 1

        # Clear full z-layers.
        full = [z for z in range(H) if self.board[:, :, z].all()]
        n = len(full)
        if n:
            keep = [z for z in range(H) if z not in full]
            new_board = np.zeros_like(self.board)
            new_color = -np.ones_like(self.board_color)
            for new_z, z in enumerate(keep):
                new_board[:, :, new_z] = self.board[:, :, z]
                new_color[:, :, new_z] = self.board_color[:, :, z]
            self.board = new_board
            self.board_color = new_color
            self.cleared_layers += n
        gained = LAYER_SCORE[min(n, 4)]
        self.score += gained
        return n, gained

    # ---- board features for shaping -----------------------------------
    def _heightmap(self):
        hmap = np.zeros((W, D), dtype=np.int64)
        for x in range(W):
            for y in range(D):
                col = self.board[x, y]
                nz = np.nonzero(col)[0]
                hmap[x, y] = (nz.max() + 1) if nz.size else 0
        return hmap

    def _features(self):
        hmap = self._heightmap()
        agg_height = int(hmap.sum())
        holes = 0
        for x in range(W):
            for y in range(D):
                top = hmap[x, y]
                if top > 0:
                    holes += int(top - self.board[x, y, :top].sum())
        bump = int(np.abs(np.diff(hmap, axis=0)).sum() + np.abs(np.diff(hmap, axis=1)).sum())
        return agg_height, holes, bump

    def _fill_bonus(self):
        # Reward layers that are close to complete (cubic emphasizes near-full
        # layers), steering the agent toward actually clearing layers.
        area = W * D
        frac = self.board.sum(axis=(0, 1)) / area  # fill fraction per z-layer
        return float((frac ** FILL_POW).sum())

    def _potential(self):
        h, holes, bump = self._features()
        return (W_HEIGHT * h + W_HOLES * holes + W_BUMPY * bump
                + W_FILL * self._fill_bonus())

    # ---- step ----------------------------------------------------------
    def step(self, action):
        if self.done:
            return self.get_obs(), 0.0, True, self._info()

        self.steps += 1
        prev_potential = self._potential() if self.shaping else 0.0
        a = ACTIONS[action]
        locked = False

        if a == "hard_drop":
            pos = self.piece.pos.copy()
            while not self._collides(self.piece.cells, pos + [0, 0, -1]):
                pos[2] -= 1
            self.piece.pos = pos
            n, gained = self._lock_and_clear()
            locked = True
        else:
            # Apply the chosen lateral move / rotation if legal.
            if a.startswith("move"):
                delta = {"move_-x": [-1, 0, 0], "move_+x": [1, 0, 0],
                         "move_-y": [0, -1, 0], "move_+y": [0, 1, 0]}[a]
                npos = self.piece.pos + delta
                if not self._collides(self.piece.cells, npos):
                    self.piece.pos = npos
            elif a.startswith("rot"):
                axis = {"rot_x": 0, "rot_y": 1, "rot_z": 2}[a]
                ncells = P.rotate(self.piece.cells, axis)
                if not self._collides(ncells, self.piece.pos):
                    self.piece.cells = ncells

            # Gravity: try to descend one cell.
            if self._collides(self.piece.cells, self.piece.pos + [0, 0, -1]):
                n, gained = self._lock_and_clear()
                locked = True
            else:
                self.piece.pos[2] -= 1
                n, gained = 0, 0

        # Reward.
        reward = float(gained) + ALIVE_REWARD
        if locked:
            self._spawn()
            if self.shaping and not self.done:
                reward += self._potential() - prev_potential
        if self.done:
            reward += GAMEOVER_PENALTY

        return self.get_obs(), reward, self.done, self._info()

    # ---- observation ---------------------------------------------------
    def get_obs(self):
        board = self.board.astype(np.float32)
        piece_grid = np.zeros((W, D, H), dtype=np.float32)
        if not self.done:
            wc = self.piece.world_cells()
            valid = (wc[:, 2] >= 0) & (wc[:, 2] < H)
            wc = wc[valid]
            piece_grid[wc[:, 0], wc[:, 1], wc[:, 2]] = 1.0
        ptype_oh = np.zeros(P.NUM_PIECE_TYPES, dtype=np.float32)
        ntype_oh = np.zeros(P.NUM_PIECE_TYPES, dtype=np.float32)
        cur = self.piece.ptype if not self.done else 0
        ptype_oh[cur] = 1.0
        ntype_oh[self.next_type] = 1.0
        return np.concatenate([board.ravel(), piece_grid.ravel(), ptype_oh, ntype_oh])

    @staticmethod
    def obs_dim():
        return W * D * H * 2 + P.NUM_PIECE_TYPES * 2

    def _info(self):
        return {
            "score": self.score,
            "cleared_layers": self.cleared_layers,
            "pieces_placed": self.pieces_placed,
            "steps": self.steps,
        }

    # ---- GUI snapshot --------------------------------------------------
    def render_state(self):
        """JSON-serializable snapshot for the web GUI."""
        cells = []
        cb = self.board_color
        for x in range(W):
            for y in range(D):
                for z in range(H):
                    if self.board[x, y, z]:
                        cells.append([int(x), int(y), int(z), int(cb[x, y, z])])
        piece = []
        if not self.done:
            for cx, cy, cz in self.piece.world_cells():
                if 0 <= cz < H:
                    piece.append([int(cx), int(cy), int(cz)])
        return {
            "dims": [W, D, H],
            "cells": cells,
            "piece": piece,
            "piece_type": int(self.piece.ptype) if not self.done else -1,
            "next_type": int(self.next_type),
            "score": int(self.score),
            "cleared_layers": int(self.cleared_layers),
            "pieces_placed": int(self.pieces_placed),
            "done": bool(self.done),
            "colors": P.PIECE_COLORS,
        }

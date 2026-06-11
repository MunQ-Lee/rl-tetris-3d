"""3D Tetris pieces: a randomized polycube library + rotation/drop helpers.

Instead of only the 8 tetracubes, the piece set is generated: every distinct
free polycube (connected set of unit cubes, deduplicated under 3D rotation) of
size 3, 4 and 5. Spawns are drawn with a size weighting (4-cube pieces stay the
most common) so shapes are varied and unpredictable while remaining playable.
"""
import colorsys

import numpy as np

# Which polycube sizes to include and their relative spawn weight (within a
# size class every shape is equally likely).
PIECE_SIZES_INCLUDED = (3, 4, 5)
SIZE_WEIGHT = {3: 1.0, 4: 3.0, 5: 1.5}

NEIGHBORS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def _normalize(cells: np.ndarray) -> np.ndarray:
    """Shift cells so the minimum coordinate along each axis is 0."""
    return cells - cells.min(axis=0)


def rotate(cells: np.ndarray, axis: int) -> np.ndarray:
    """Rotate a piece 90 degrees about the given axis (0=x, 1=y, 2=z)."""
    x, y, z = cells[:, 0], cells[:, 1], cells[:, 2]
    if axis == 0:      # about x: (x, y, z) -> (x, -z, y)
        out = np.stack([x, -z, y], axis=1)
    elif axis == 1:    # about y: (x, y, z) -> (z, y, -x)
        out = np.stack([z, y, -x], axis=1)
    else:              # about z: (x, y, z) -> (-y, x, z)
        out = np.stack([-y, x, z], axis=1)
    return _normalize(out)


def _canon(cells: np.ndarray):
    """Canonical hashable key for a normalized cell set (order-independent)."""
    return tuple(sorted(map(tuple, cells.tolist())))


def orientations_of(cells: np.ndarray):
    """All distinct orientations of a cell set under the 24 axis rotations."""
    start = _normalize(cells)
    seen = {_canon(start): start}
    frontier = [start]
    while frontier:
        cur = frontier.pop()
        for axis in (0, 1, 2):
            nxt = rotate(cur, axis)
            key = _canon(nxt)
            if key not in seen:
                seen[key] = nxt
                frontier.append(nxt)
    return list(seen.values())


def _rot_canon(cells: np.ndarray):
    """Rotation-invariant canonical key (min over all orientations)."""
    return min(_canon(o) for o in orientations_of(cells))


def _tkey(cells):
    mx = min(c[0] for c in cells); my = min(c[1] for c in cells); mz = min(c[2] for c in cells)
    return tuple(sorted((c[0] - mx, c[1] - my, c[2] - mz) for c in cells))


def _gen_polycubes(n: int):
    """Every distinct free polycube of size n (deduplicated under rotation)."""
    shapes = {((0, 0, 0),)}
    for _ in range(n - 1):
        nxt = set()
        for shp in shapes:
            s = set(shp)
            for c in s:
                for d in NEIGHBORS:
                    nc = (c[0] + d[0], c[1] + d[1], c[2] + d[2])
                    if nc not in s:
                        nxt.add(_tkey(s | {nc}))
        shapes = nxt
    seen = set()
    out = []
    for shp in shapes:
        arr = np.array(shp, dtype=np.int64)
        rc = _rot_canon(arr)
        if rc not in seen:
            seen.add(rc)
            out.append(_normalize(arr))
    return out


# Build the piece library.
PIECES = []
PIECE_SIZES = []
for _sz in PIECE_SIZES_INCLUDED:
    for _cells in _gen_polycubes(_sz):
        PIECES.append(_cells)
        PIECE_SIZES.append(_sz)

NUM_PIECE_TYPES = len(PIECES)

# Spawn probabilities: each size class gets SIZE_WEIGHT, split evenly within it.
_counts = {s: PIECE_SIZES.count(s) for s in set(PIECE_SIZES)}
_w = np.array([SIZE_WEIGHT[s] / _counts[s] for s in PIECE_SIZES], dtype=np.float64)
PIECE_PROBS = _w / _w.sum()

# Distinct colors for the GUI (HSV around the wheel) + gray for garbage rows.
PIECE_COLORS = [
    "#%02x%02x%02x" % tuple(int(255 * c) for c in colorsys.hsv_to_rgb(i / NUM_PIECE_TYPES, 0.62, 0.92))
    for i in range(NUM_PIECE_TYPES)
]
PIECE_COLORS.append("#6b7280")
GARBAGE_COLOR_ID = NUM_PIECE_TYPES


def spawn_cells(piece_type: int) -> np.ndarray:
    """Return normalized cells for a freshly spawned piece."""
    return _normalize(PIECES[piece_type].copy())


def all_orientations(piece_type: int):
    """All distinct orientations of a piece type."""
    return orientations_of(PIECES[piece_type])


# Precompute orientations for every piece type.
ORIENTATIONS = [all_orientations(t) for t in range(NUM_PIECE_TYPES)]


class OrientInfo:
    """Precomputed drop geometry for one orientation."""
    __slots__ = ("cells", "ex", "ey", "max_oz", "fox", "foy", "fbot")

    def __init__(self, cells):
        self.cells = cells
        self.ex = int(cells[:, 0].max())
        self.ey = int(cells[:, 1].max())
        self.max_oz = int(cells[:, 2].max())
        # Bottom profile: for each footprint column (ox, oy), the lowest oz.
        cols = {}
        for ox, oy, oz in cells.tolist():
            key = (ox, oy)
            cols[key] = min(oz, cols.get(key, oz))
        keys = sorted(cols)
        self.fox = np.array([k[0] for k in keys], dtype=np.int64)
        self.foy = np.array([k[1] for k in keys], dtype=np.int64)
        self.fbot = np.array([cols[k] for k in keys], dtype=np.int64)


# For each piece type, drop geometry for each of its orientations.
ORIENT_INFO = [[OrientInfo(c) for c in ORIENTATIONS[t]] for t in range(NUM_PIECE_TYPES)]

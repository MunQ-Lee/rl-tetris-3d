"""3D Tetris piece (tetracube) definitions and rotation helpers.

A piece is a set of 4 unit cubes given as integer (x, y, z) offsets.
Rotations are exact 90-degree rotations about the x, y, or z axis.
"""
import numpy as np

# 8 distinct tetracubes (flat tetrominoes + genuine 3D shapes).
# Each is a (4, 3) array of (x, y, z) cube offsets.
PIECES = [
    np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.int64),  # I  (line)
    np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 1, 0]], dtype=np.int64),  # L
    np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [1, 1, 0]], dtype=np.int64),  # T
    np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0]], dtype=np.int64),  # S
    np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.int64),  # O (square)
    np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int64),  # tripod (3D)
    np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=np.int64),  # left screw (3D)
    np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 1]], dtype=np.int64),  # right screw (3D)
]

NUM_PIECE_TYPES = len(PIECES)

# Colors (hex) used by the web GUI, one per piece type.
PIECE_COLORS = [
    "#00bcd4", "#ff9800", "#9c27b0", "#4caf50",
    "#ffeb3b", "#f44336", "#2196f3", "#e91e63",
]


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


def spawn_cells(piece_type: int) -> np.ndarray:
    """Return normalized cells for a freshly spawned piece."""
    return _normalize(PIECES[piece_type].copy())

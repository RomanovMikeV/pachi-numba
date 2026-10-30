# pachi-numba: a translation of Pachi (GPL-2.0-only), October 2026; see NOTICE.
"""Translation of Pachi's ``ownermap.h`` and ``ownermap.c``.

Map of board intersection owners, and devices to derive group status
information from the map. A ``struct board_ownermap`` is a named tuple:
``playouts`` (``playouts[0]``) and ``map`` (``[board_size2, S_MAX]``).
"""

from collections import namedtuple

import numpy as np
from numba import njit

from . import board as pb
from .mq import mq_add
from .stone import S_BLACK, S_MAX, S_NONE, S_WHITE, stone_other

Ownermap = namedtuple("Ownermap", ["playouts", "map"])

# enum point_judgement
PJ_DAME, PJ_BLACK, PJ_WHITE, PJ_UNKNOWN = S_NONE, S_BLACK, S_WHITE, 3
# enum gj_state
GS_NONE, GS_DEAD, GS_ALIVE, GS_UNKNOWN = range(4)


@njit(cache=True, nogil=True, _nrt=True)
def ownermap_new(size2):
    return Ownermap(np.zeros(1, np.int64), np.zeros((size2, S_MAX), np.int64))


@njit(cache=True, nogil=True, _nrt=False)
def board_ownermap_fill(ownermap, b):
    """Count the owners of all points of the final position ``b``."""
    ownermap.playouts[0] += 1
    for c in range(pb.board_size2(b)):
        color = b.b[c]
        if color == S_NONE:
            color = pb.board_get_one_point_eye(b, c)
        ownermap.map[c, color] += 1


@njit(cache=True, nogil=True, _nrt=False)
def board_ownermap_merge(bsize2, dst, src):
    dst.playouts[0] += src.playouts[0]
    for i in range(bsize2):
        for j in range(S_MAX):
            dst.map[i, j] += src.map[i, j]


@njit(cache=True, nogil=True, _nrt=False)
def board_ownermap_estimate_point(ownermap, c):
    """(black - white) / playouts, as float32 (C returns float)."""
    b = ownermap.map[c, S_BLACK]
    w = ownermap.map[c, S_WHITE]
    total = ownermap.playouts[0]
    return np.float32(1.0 * (b - w) / total)


@njit(cache=True, nogil=True, _nrt=False)
def board_ownermap_judge_point(ownermap, c, thres):
    n = ownermap.map[c, S_NONE]
    b = ownermap.map[c, S_BLACK]
    w = ownermap.map[c, S_WHITE]
    total = ownermap.playouts[0]
    # C: int * floating_t, compared as float.
    limit = np.float32(np.float32(total) * np.float32(thres))
    if np.float32(n) >= limit:
        return PJ_DAME
    if np.float32(n + b) >= limit:
        return PJ_BLACK
    if np.float32(n + w) >= limit:
        return PJ_WHITE
    return PJ_UNKNOWN


@njit(cache=True, nogil=True, _nrt=False)
def board_ownermap_judge_groups(b, ownermap, thres, gs):
    """Fill ``gs`` (``[board_size2]``) with the status of every group."""
    gs[:] = GS_NONE
    for c in range(pb.board_size2(b)):
        color = b.b[c]
        g = b.g[c]
        if not g:
            continue
        pj = board_ownermap_judge_point(ownermap, c, thres)
        if pj == PJ_UNKNOWN:
            # Fate is uncertain.
            gs[g] = GS_UNKNOWN
        elif gs[g] != GS_UNKNOWN:
            # Update group state.
            if pj == color:
                new = GS_ALIVE
            elif pj == stone_other(color):
                new = GS_DEAD
            else:
                assert pj == PJ_DAME
                new = GS_UNKNOWN  # Exotic!
            if gs[g] == GS_NONE:
                gs[g] = new
            elif gs[g] != new:
                gs[g] = GS_UNKNOWN  # Contradiction. :(


@njit(cache=True, nogil=True, _nrt=False)
def groups_of_status(b, gs, s, q):
    """Add the groups of status ``s`` to the move queue ``q``."""
    for c in range(pb.board_size2(b)):
        g = b.g[c]
        if not g or g != c:
            continue
        assert gs[g] != GS_NONE
        if gs[g] == s:
            mq_add(q, g, 0)

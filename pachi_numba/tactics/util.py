# pachi-numba: a translation of Pachi (GPL-2.0-only), October 2026; see NOTICE.
"""Translation of Pachi's ``tactics/util.h`` and ``tactics/util.c``."""

import numpy as np
from numba import njit

from .. import board as pb
from ..stone import S_BLACK, S_NONE, S_OFFBOARD

# On average 20% of points remain empty at the end of a game
EXPECTED_FINAL_EMPTY_PERCENT = 20
# To avoid running out of time, assume we always have at least 30 more moves
MIN_MOVES_LEFT = 30


@njit(cache=True, nogil=True, _nrt=False)
def c_div(a, b):
    """C integer division: truncates toward zero."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


@njit(cache=True, nogil=True, _nrt=False)
def board_stone_radar(b, coord, distance):
    """Whether there are any stones in ``distance``-vicinity of ``coord``."""
    high = pb.board_size(b) - 2
    x1 = min(max(pb.coord_x(coord, b) - distance, 1), high)
    y1 = min(max(pb.coord_y(coord, b) - distance, 1), high)
    x2 = min(max(pb.coord_x(coord, b) + distance, 1), high)
    y2 = min(max(pb.coord_y(coord, b) + distance, 1), high)
    for x in range(x1, x2 + 1):
        for y in range(y1, y2 + 1):
            if pb.board_atxy(b, x, y) != S_NONE:
                return True
    return False


@njit(cache=True, nogil=True, _nrt=False)
def coord_edge_distance(c, b):
    """Distance from the edge; on edge returns 0."""
    x = pb.coord_x(c, b)
    y = pb.coord_y(c, b)
    dx = pb.board_size(b) - 1 - x if x > pb.board_size(b) // 2 else x
    dy = pb.board_size(b) - 1 - y if y > pb.board_size(b) // 2 else y
    return (dx if dx < dy else dy) - 1  # S_OFFBOARD


@njit(cache=True, nogil=True, _nrt=False)
def coord_gridcular_distance(c1, c2, b):
    dx = abs(pb.coord_x(c1, b) - pb.coord_x(c2, b))
    dy = abs(pb.coord_y(c1, b) - pb.coord_y(c2, b))
    return dx + dy + (dx if dx > dy else dy)


@njit(cache=True, nogil=True, _nrt=False)
def _qinc(x, size2):
    return x + 1 - size2 if x + 1 >= size2 else x + 1


@njit(cache=True, nogil=True, _nrt=False)
def _cfg_one(b, coord, grp, d, distances, queue, qstop):
    distances[coord] = d
    for c in pb.neighbors(b, coord):
        # C macro cfg_one(c, g): its group_at(b, coord) reads the neighbor c.
        if distances[c] < 0 and (not grp or b.g[c] != grp):
            queue[qstop] = c
            qstop = _qinc(qstop, pb.board_size2(b))
    return qstop


@njit(cache=True, nogil=True, _nrt=True)
def cfg_distances(b, start, distances, maxdist):
    """Distances in the "common fate graph" from ``start`` (stone chains cost 0)."""
    size2 = pb.board_size2(b)
    queue = np.zeros(size2, np.int64)
    qstart = 0
    qstop = 0
    for c in range(size2):
        distances[c] = maxdist + 1 if b.b[c] == S_OFFBOARD else -1
    queue[qstop] = start
    qstop += 1
    for d in range(maxdist + 1):
        qa = qstart
        qb = qstop
        qstart = qstop
        q = qa
        while q < qb:
            cq = queue[q]
            if distances[cq] < 0:
                if b.b[cq] == S_NONE:
                    qstop = _cfg_one(b, cq, 0, d, distances, queue, qstop)
                else:
                    g = b.g[cq]
                    c = g
                    c2 = b.p[c]
                    while True:
                        qstop = _cfg_one(b, c, g, d, distances, queue, qstop)
                        c = c2
                        if c == 0:
                            break
                        c2 = b.p[c2]
            q = _qinc(q, size2)
    for c in range(size2):
        if distances[c] < 0:
            distances[c] = maxdist + 1


@njit(cache=True, nogil=True, _nrt=False)
def board_effective_handicap(b, first_move_value):
    """Extra komi for the "effective handicap" of black (0 for even 7.5-komi games)."""
    komi = np.float64(b.komi[0])
    if not b.ints[pb.HANDICAP]:
        return np.float32(0.5 * first_move_value if komi == 0.5 else 7.5 - komi)
    return np.float32(b.ints[pb.HANDICAP] * first_move_value + 0.5 - komi)


@njit(cache=True, nogil=True, _nrt=False)
def pass_is_safe(b, color, dead_groups, num_dead):
    """Whether ``color`` wins by counting, with the given groups dead."""
    score = pb.board_official_score(b, dead_groups, num_dead)
    if color == S_BLACK:
        score = -score
    return score >= 0


@njit(cache=True, nogil=True, _nrt=False)
def board_estimated_moves_left(b):
    """Estimated number of remaining moves for one player until end of game."""
    total_points = (pb.board_size(b) - 2) * (pb.board_size(b) - 2)
    moves_left = c_div(b.ints[pb.FLEN] - total_points * EXPECTED_FINAL_EMPTY_PERCENT // 100, 2)
    return moves_left if moves_left > MIN_MOVES_LEFT else MIN_MOVES_LEFT


@njit(cache=True, nogil=True, _nrt=False)
def board_local_value(scan_neis, b, coord, color):
    """Value of the move ``coord`` by ``color`` in the final position, in [0, 1]."""
    if scan_neis:
        friends = 0
        for c in pb.neighbors(b, coord):
            friends += b.b[c] == color or b.b[c] == S_OFFBOARD \
                or pb.board_is_one_point_eye(b, c, color)
        return np.float64(2 * (b.b[coord] == color) + friends) / np.float64(np.float32(6.0))
    return 1.0 if b.b[coord] == color else 0.0

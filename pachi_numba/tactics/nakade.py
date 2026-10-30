# pachi-numba: a translation of Pachi (GPL-2.0-only), October 2026; see NOTICE.
"""Translation of Pachi's ``tactics/nakade.h`` and ``tactics/nakade.c``: piercing eyes."""

import numpy as np
from numba import njit

from .. import board as pb
from ..move import PASS
from ..stone import S_NONE, stone_other

NAKADE_MAX = 6


@njit(cache=True, nogil=True, _nrt=False)
def nakade_area(b, around, color, area):
    """The empty area around ``around``, within ``color`` groups; -1 if not one."""
    area_n = 0
    area[area_n] = around
    area_n += 1
    i = 0
    while i < area_n:
        for c in pb.neighbors(b, area[i]):
            if b.b[c] == stone_other(color):
                return -1
            if b.b[c] == S_NONE:
                dup = False
                for j in range(area_n):
                    if c == area[j]:
                        dup = True
                        break
                if dup:
                    continue
                if area_n >= NAKADE_MAX:
                    return -1  # Too large nakade area.
                area[area_n] = c
                area_n += 1
        i += 1
    return area_n


@njit(cache=True, nogil=True, _nrt=False)
def get_neighbors(b, area, area_n, neighbors, ptbynei):
    """Adjacency counts of the area points, and their histogram."""
    neighbors[:area_n] = 0
    for i in range(area_n):
        for j in range(i + 1, area_n):
            if abs(area[i] - area[j]) == 1 or abs(area[i] - area[j]) == pb.board_size(b):
                ptbynei[neighbors[i]] -= 1
                neighbors[i] += 1
                ptbynei[neighbors[i]] += 1
                ptbynei[neighbors[j]] -= 1
                neighbors[j] += 1
                ptbynei[neighbors[j]] += 1


@njit(cache=True, nogil=True, _nrt=False)
def nakade_point_(area, area_n, neighbors, ptbynei, coordbynei):
    coordbynei[:] = 0
    for i in range(area_n):
        coordbynei[neighbors[i]] = area[i]
    if area_n == 1 or area_n == 2:
        return PASS
    if area_n == 3:
        assert ptbynei[2] == 1
        return coordbynei[2]  # middle point
    if area_n == 4:
        if ptbynei[3] != 1:
            return PASS  # long line, L shape, or square
        return coordbynei[3]  # tetris four
    if area_n == 5:
        if ptbynei[3] == 1 and ptbynei[1] == 1:
            return coordbynei[3]  # bulky five
        if ptbynei[4] == 1:
            return coordbynei[4]  # cross five
        return PASS  # long line
    if area_n == 6:
        if ptbynei[4] == 1 and ptbynei[2] == 3:
            return coordbynei[4]  # rabbity six
        return PASS
    raise AssertionError("nakade_point_: bad area")


@njit(cache=True, nogil=True, _nrt=False)
def _area_and_neighbors(b, around, color):
    """The area and adjacency arrays, in ``b.scratch[48:79]`` (C: on the stack)."""
    b.scratch[48:79] = 0
    area = b.scratch[48:54]
    area_n = nakade_area(b, around, color, area)
    neighbors = b.scratch[54:60]
    ptbynei = b.scratch[60:69]
    if area_n > 0:
        ptbynei[0] = area_n
    return area, area_n, neighbors, ptbynei


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def nakade_point(b, around, color):
    """Normalizes the argument types (one compiled ``_nakade_point``)."""
    return _nakade_point(b, np.int64(around), np.int64(color))


@njit(cache=True, nogil=True, _nrt=False)
def _nakade_point(b, around, color):
    """The eye-piercing point of the empty area around ``around``, or pass."""
    assert b.b[around] == S_NONE
    area, area_n, neighbors, ptbynei = _area_and_neighbors(b, around, color)
    if area_n == -1:
        return PASS
    get_neighbors(b, area, area_n, neighbors, ptbynei)
    return nakade_point_(area, area_n, neighbors, ptbynei, b.scratch[70:79])


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def nakade_dead_shape(b, around, color):
    """Normalizes the argument types (one compiled ``_nakade_dead_shape``)."""
    return _nakade_dead_shape(b, np.int64(around), np.int64(color))


@njit(cache=True, nogil=True, _nrt=False)
def _nakade_dead_shape(b, around, color):
    """Whether the big eyespace can be reduced to one eye."""
    assert b.b[around] == S_NONE
    area, area_n, neighbors, ptbynei = _area_and_neighbors(b, around, color)
    if area_n == -1:
        return False
    if area_n <= 3:
        return True
    get_neighbors(b, area, area_n, neighbors, ptbynei)
    if area_n == 4 and ptbynei[2] == 4:  # square 4
        return True
    return nakade_point_(area, area_n, neighbors, ptbynei, b.scratch[70:79]) != PASS

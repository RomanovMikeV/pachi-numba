# pachi-numba: a translation of Pachi (GPL-2.0-only), October 2026; see NOTICE.
"""Translation of Pachi's ``stone.h`` and ``stone.c``."""

import numpy as np
from numba import njit

# enum stone
S_NONE = 0
S_BLACK = 1
S_WHITE = 2
S_OFFBOARD = 3
S_MAX = 4

_STONE_OTHER = np.array([S_NONE, S_WHITE, S_BLACK, S_OFFBOARD], dtype=np.int8)


def stone2char(s):
    return ".XO#"[s]


def char2stone(s):
    return {".": S_NONE, "X": S_BLACK, "O": S_WHITE, "#": S_OFFBOARD}.get(s, S_NONE)


def stone2str(s):
    if s == S_BLACK:
        return "black"
    if s == S_WHITE:
        return "white"
    return "none"


def str2stone(text):
    first = text[:1].lower()
    if first == "b":
        return S_BLACK
    if first == "w":
        return S_WHITE
    return S_NONE


@njit(cache=True, nogil=True, _nrt=False)
def stone_other(s):
    return _STONE_OTHER[s]

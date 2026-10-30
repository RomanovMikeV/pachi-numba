# pachi-numba: a translation of Pachi (GPL-2.0-only), October 2026; see NOTICE.
"""Translation of Pachi's ``tactics/nlib.h`` and ``tactics/nlib.c``.

N-liberty semeai defense tactical checks.
"""

from numba import njit

from .. import board as pb
from ..stone import S_NONE, S_OFFBOARD, stone_other
from . import two_lib


@njit(cache=True, nogil=True, _nrt=False)
def group_nlib_defense_check(b, group, to_play, q, tag, rng):
    """Moves for ``to_play`` to atari 2-lib enemy neighbors of its own n-lib group."""
    color = to_play
    assert color != S_OFFBOARD and color != S_NONE and color == b.b[pb.group_base(group)]
    # It is possible that we must start capturing a 2-lib neighbor right now,
    # because of approach liberties. Therefore, we will check for this case.
    stone = group
    stone_next = b.p[stone]
    while True:
        for c in pb.neighbors(b, stone):
            if b.b[c] != stone_other(color):
                continue
            g2 = b.g[c]
            if b.gi_libs[g2] != 2:
                continue
            two_lib.can_atari_group(b, g2, stone_other(color), to_play, q, tag, True, rng)
        stone = stone_next
        if stone == 0:
            break
        stone_next = b.p[stone_next]

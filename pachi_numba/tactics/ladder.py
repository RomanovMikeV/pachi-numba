# pachi-numba: a translation of Pachi (GPL-2.0-only), October 2026; see NOTICE.
"""Translation of Pachi's ``tactics/ladder.h`` and ``tactics/ladder.c``: reading ladders.

C reads the ladder on a preallocated stack of boards (``bset``); here
every copy is a new board, which gives the same results. The recursion of
``middle_ladder_walk`` is an explicit stack (numba cannot cache recursion).
"""

import numpy as np
from numba import njit

from .. import board as pb
from .. import mq
from ..move import coord_is_8adjecent, coord_is_adjecent
from ..stone import S_NONE, S_OFFBOARD, stone_other
from . import one_lib


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def is_ladder(b, coord, laddered, test_middle):
    """Normalizes the argument types (one compiled ``_is_ladder``)."""
    return _is_ladder(b, np.int64(coord), np.int64(laddered), bool(test_middle))


@njit(cache=True, nogil=True, _nrt=False)
def _is_ladder(b, coord, laddered, test_middle):
    """Whether escaping at ``coord`` by the group in atari would play out a simple ladder.

    First-line ladders always; trivial middle-board ladders if ``test_middle``.
    """
    lcolor = b.b[pb.group_base(laddered)]
    # First, special-case first-line "ladders". This is a huge chunk
    # of ladders we actually meet and want to play.
    if pb.neighbor_count_at(b, coord, S_OFFBOARD) == 1 \
            and pb.neighbor_count_at(b, coord, lcolor) == 1:
        return is_border_ladder(b, coord, lcolor)
    return test_middle and is_middle_ladder(b, coord, laddered, lcolor)


@njit(cache=True, nogil=True, _nrt=False)
def is_border_ladder(b, coord, lcolor):
    x = pb.coord_x(coord, b)
    y = pb.coord_y(coord, b)
    # Direction along border; xd is horiz. border, yd vertical.
    xd = 0
    yd = 0
    if pb.board_atxy(b, x + 1, y) == S_OFFBOARD or pb.board_atxy(b, x - 1, y) == S_OFFBOARD:
        yd = 1
    else:
        xd = 1
    # Direction from the border; -1 is above/left, 1 is below/right.
    dd = 1 if pb.board_atxy(b, x + yd, y + xd) == S_OFFBOARD else -1
    # This is normally caught, unless we have friends both above and below...
    if pb.board_atxy(b, x + xd * 2, y + yd * 2) == lcolor \
            and pb.board_atxy(b, x - xd * 2, y - yd * 2) == lcolor:
        return False
    # ...or can't block where we need because of shortage of liberties.
    g1 = pb.group_atxy(b, x + xd - yd * dd, y + yd - xd * dd)
    libs1 = b.gi_libs[g1]
    g2 = pb.group_atxy(b, x - xd - yd * dd, y - yd - xd * dd)
    libs2 = b.gi_libs[g2]
    # Already in atari?
    if libs1 < 2 or libs2 < 2:
        return False
    size = pb.board_size(b)
    # Would be self-atari?
    if libs1 < 3 and (pb.board_atxy(b, x + xd * 2, y + yd * 2) != S_NONE
                      or coord_is_adjecent(b.gi_lib[g1, 0], b.gi_lib[g1, 1], size)):
        return False
    if libs2 < 3 and (pb.board_atxy(b, x - xd * 2, y - yd * 2) != S_NONE
                      or coord_is_adjecent(b.gi_lib[g2, 0], b.gi_lib[g2, 1], size)):
        return False
    return True


# The results of the escape move of middle_ladder_walk().
_CAPTURED, _FREE, _CHASED = 0, 1, 2


@njit(cache=True, nogil=True, _nrt=False)
def _ladder_escape(b, laddered, nextmove, lcolor, alternatives):
    """The start of C's middle_ladder_walk(): escape, then list the chasing moves.

    Returns (result, laddered group); with ``_CHASED``, ``alternatives``
    holds the liberty indices to try and their number.
    """
    assert b.gi_libs[laddered] == 1
    # First, escape.
    res = pb.board_play(b, nextmove, lcolor)
    laddered = b.g[laddered]
    assert res >= 0
    if b.gi_libs[laddered] == 1:
        return _CAPTURED, laddered  # we can capture now
    if b.gi_libs[laddered] > 2:
        return _FREE, laddered  # we are free now
    for c in pb.neighbors(b, nextmove):
        if b.b[c] == stone_other(lcolor) and b.gi_libs[b.g[c]] == 1:
            # We can capture one of the ladder stones anytime later.
            return _FREE, laddered
    # Now, consider alternatives.
    libs = 0
    size = pb.board_size(b)
    for i in range(2):
        ataristone = b.gi_lib[laddered, i]
        escape = b.gi_lib[laddered, 1 - i]
        if pb.immediate_liberty_count(b, escape) > 2 + coord_is_adjecent(ataristone, escape, size):
            continue  # Too much free space, ignore.
        alternatives[libs] = i
        libs += 1
    alternatives[2] = libs
    alternatives[3] = 0  # the next alternative to try
    return _CHASED, laddered


@njit(cache=True, nogil=True, _nrt=True)
def middle_ladder_walk(b, laddered, nextmove, lcolor):
    """Whether some chase sequence, keeping the group at two liberties, captures it.

    C recurses: escape, then try each chasing move and recurse after it;
    true if any branch captures. numba cannot cache recursive functions,
    so the branches are on an explicit stack; the answer is the same, as
    the search has no side effects outside its board copies.
    """
    alternatives = np.zeros(4, np.int64)
    result, laddered = _ladder_escape(b, laddered, nextmove, lcolor, alternatives)
    if result != _CHASED:
        return result == _CAPTURED
    stack = [(b, laddered, alternatives)]
    while len(stack) > 0:
        frame_board, frame_laddered, frame_alternatives = stack[-1]
        i = frame_alternatives[3]
        if i >= frame_alternatives[2]:
            stack.pop()  # no branch captures
            continue
        frame_alternatives[3] = i + 1
        b2 = pb.board_duplicate(frame_board)
        ataristone = b2.gi_lib[frame_laddered, frame_alternatives[i]]
        res = pb.board_play(b2, ataristone, stone_other(lcolor))
        # If we just played self-atari, abandon ship.
        if res < 0 or b2.gi_libs[b2.g[ataristone]] <= 1:
            continue
        next_alternatives = np.zeros(4, np.int64)
        result, next_laddered = _ladder_escape(b2, frame_laddered, b2.gi_lib[frame_laddered, 0],
                                               lcolor, next_alternatives)
        if result == _CAPTURED:
            return True
        if result == _CHASED:
            stack.append((b2, next_laddered, next_alternatives))
    return False


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def is_middle_ladder(b, coord, laddered, lcolor):
    """Normalizes the argument types (one compiled ``_is_middle_ladder``)."""
    return _is_middle_ladder(b, np.int64(coord), np.int64(laddered), np.int64(lcolor))


@njit(cache=True, nogil=True, _nrt=True)
def _is_middle_ladder(b, coord, laddered, lcolor):
    assert b.gi_libs[laddered] == 1
    assert b.gi_lib[laddered, 0] == coord
    assert b.b[laddered] == lcolor
    # If we can move into empty space or do not have enough space
    # to escape, this is obviously not a ladder.
    if pb.immediate_liberty_count(b, coord) != 2:
        return False
    # A fair chance for a ladder. Group in atari, with some but limited
    # space to escape. Time for the expensive stuff.
    ccq = mq.mq_new()
    if one_lib.can_countercapture(b, lcolor, laddered, lcolor, ccq, True, 0):
        # We could escape by countercapturing a group. Investigate.
        assert ccq.moves[0] > 0
        for i in range(ccq.moves[0]):
            b2 = pb.board_duplicate(b)
            if not middle_ladder_walk(b2, laddered, ccq.move[i], lcolor):
                return False
    b2 = pb.board_duplicate(b)
    return middle_ladder_walk(b2, laddered, b2.gi_lib[laddered, 0], lcolor)


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def wouldbe_ladder(b, group, escapelib, chaselib, lcolor):
    """Normalizes the argument types (one compiled ``_wouldbe_ladder``)."""
    return _wouldbe_ladder(b, np.int64(group), np.int64(escapelib), np.int64(chaselib),
                           np.int64(lcolor))


@njit(cache=True, nogil=True, _nrt=True)
def _wouldbe_ladder(b, group, escapelib, chaselib, lcolor):
    """Whether the 2-lib group escaping at ``escapelib`` is caught by a stone at ``chaselib``."""
    assert b.gi_libs[group] == 2
    assert b.b[group] == lcolor
    if not coord_is_8adjecent(escapelib, chaselib, pb.board_size(b)):
        return False  # cannot determine ladder for remote simulated stone
    if pb.neighbor_count_at(b, chaselib, lcolor) != 1 \
            or pb.immediate_liberty_count(b, chaselib) != 2:
        return False  # overly trivial for a ladder
    is_ladder_found = False
    b2 = pb.board_duplicate(b)
    res = pb.board_play(b2, chaselib, stone_other(lcolor))
    if res >= 0:
        is_ladder_found = middle_ladder_walk(b2, group, b2.gi_lib[group, 0], lcolor)
    return is_ladder_found

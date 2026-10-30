# pachi-numba: a translation of Pachi (GPL-2.0-only), October 2026; see NOTICE.
"""Translation of Pachi's ``tactics/1lib.h`` and ``tactics/1lib.c``.

One-liberty tactical checks (i.e. dealing with atari situations). C passes
``NULL`` for a missing move queue or ladder pointer; here a flag says
whether the argument is used.
"""

import numpy as np
from numba import njit

from .. import board as pb
from .. import random as fast
from ..mq import mq_add, mq_nodup
from ..stone import S_NONE, S_OFFBOARD, stone_other
from . import ladder, selfatari

NO_LADDER = -3  # group_atari_check(): no ladder move


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def can_play_on_lib(b, g, to_play):
    capture = b.gi_lib[g, 0]
    # Does playing on the liberty usefully capture the group?
    return pb.board_is_valid_play(b, to_play, capture) \
        and not selfatari.is_bad_selfatari(b, to_play, capture)


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def capturable_group(b, capturer, c, to_play):
    """Whether the group at ``c`` is in danger from ``capturer`` and ``to_play``
    can play at its last liberty (to capture or to escape)."""
    g = b.g[c]
    if b.b[c] != stone_other(capturer) or b.gi_libs[g] > 1:
        return False
    return can_play_on_lib(b, g, to_play)


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def can_countercapture(b, owner, g, to_play, q, use_queue, tag):
    """Normalizes the argument types (one compiled ``_can_countercapture``)."""
    return _can_countercapture(b, np.int64(owner), np.int64(g), np.int64(to_play), q,
                               bool(use_queue), np.int64(tag))


@njit(cache=True, nogil=True, _nrt=False)
def _can_countercapture(b, owner, g, to_play, q, use_queue, tag):
    """Whether ``to_play`` can save (or keep in danger) the atari group ``g`` of
    ``owner`` by capturing a neighbor group; the moves go to ``q``."""
    if b.ints[pb.CLEN] < 2:
        return False
    qmoves_prev = q.moves[0] if use_queue else 0
    stone = g
    stone_next = b.p[stone]
    while True:
        for c in pb.neighbors(b, stone):
            if not capturable_group(b, owner, c, to_play):
                continue
            if not use_queue:
                return True
            mq_add(q, b.gi_lib[b.g[c], 0], tag)
            mq_nodup(q)
        stone = stone_next
        if stone == 0:
            break
        stone_next = b.p[stone_next]
    return q.moves[0] > qmoves_prev if use_queue else False


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def group_atari_check(alwaysccaprate, b, group, to_play, q, use_ladder, middle_ladder, tag, rng):
    """Normalizes the argument types (one compiled ``_group_atari_check``)."""
    return _group_atari_check(np.int64(alwaysccaprate), b, np.int64(group), np.int64(to_play), q,
                              bool(use_ladder), bool(middle_ladder), np.int64(tag), rng)


@njit(cache=True, nogil=True, _nrt=False)
def _group_atari_check(alwaysccaprate, b, group, to_play, q, use_ladder, middle_ladder, tag, rng):
    """Moves for ``to_play`` to deal with the group in atari (rescue or capture).

    With ``use_ladder``, a ladder escape is enqueued anyway and returned
    (C: stored in ``*ladder``); otherwise, and when there is none, returns
    ``NO_LADDER``.
    """
    ladder_move = NO_LADDER
    color = b.b[pb.group_base(group)]
    lib = b.gi_lib[group, 0]
    assert color != S_OFFBOARD and color != S_NONE
    assert b.b[lib] == S_NONE

    if to_play != color:
        # We are the attacker! In that case, do not try defending
        # our group, since we can capture the culprit.
        if can_play_on_lib(b, group, to_play):
            mq_add(q, lib, tag)
            mq_nodup(q)
        return ladder_move

    # Can we capture some neighbor?
    ccap = can_countercapture(b, color, group, to_play, q, True, tag)
    if ccap and not use_ladder and alwaysccaprate > fast.fast_random(rng, 100):
        return ladder_move

    # Otherwise, do not save kos.
    if pb.group_is_onestone(b, group) \
            and pb.neighbor_count_at(b, lib, color) + pb.neighbor_count_at(b, lib, S_OFFBOARD) == 4:
        # Except when the ko is for an eye!
        eyeconnect = False
        c = lib
        for i in range(4):
            c += b.dnei[i]
            if b.b[c] == S_NONE and \
                    pb.neighbor_count_at(b, c, color) + pb.neighbor_count_at(b, c, S_OFFBOARD) == 4:
                eyeconnect = True
                break
        if not eyeconnect:
            return ladder_move

    # Do not suicide...
    if not can_play_on_lib(b, group, to_play):
        return ladder_move

    # ...or play out ladders (unless we can counter-capture anytime).
    if not ccap:
        if ladder.is_ladder(b, lib, group, middle_ladder):
            # Sometimes we want to keep the ladder move in the
            # queue in order to discourage it.
            if not use_ladder:
                return ladder_move
            ladder_move = lib

    mq_add(q, lib, tag)
    mq_nodup(q)
    return ladder_move

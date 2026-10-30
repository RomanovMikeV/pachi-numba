# pachi-numba: a translation of Pachi (GPL-2.0-only), October 2026; see NOTICE.
"""Translation of Pachi's ``tactics/2lib.h`` and ``tactics/2lib.c``.

Two-liberty tactical checks (i.e. dealing with two-step capturing races,
preventing atari).
"""

import numpy as np
from numba import njit

from .. import board as pb
from ..move import PASS, coord_is_adjecent, is_pass
from ..mq import mq_add, mq_nodup
from ..stone import S_NONE, S_OFFBOARD, stone_other
from . import selfatari


@njit(cache=True, nogil=True, _nrt=False)
def miai_2lib(b, group, color):
    """Whether we can connect on both libs, or connect on one and escape on the other."""
    can_connect = False
    can_pull_out = False
    lib0 = b.gi_lib[group, 0]
    lib1 = b.gi_lib[group, 1]
    for c in pb.neighbors(b, lib0):
        cc = b.b[c]
        if cc == S_NONE and cc != b.b[lib1]:
            can_pull_out = True
        elif cc != color:
            continue
        cg = b.g[c]
        if cg and cg != group and b.gi_libs[cg] > 1:
            can_connect = True
    for c in pb.neighbors(b, lib1):
        cc = b.b[c]
        if c == lib0:
            continue
        if cc == S_NONE and can_connect:
            return True
        elif cc != color:
            continue
        cg = b.g[c]
        if cg and cg != group and b.gi_libs[cg] > 1:
            return can_connect or can_pull_out
    return False


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def defense_is_hopeless(b, group, owner, to_play, lib, otherlib, use):
    """Normalizes the argument types (one compiled ``_defense_is_hopeless``)."""
    return _defense_is_hopeless(b, np.int64(group), np.int64(owner), np.int64(to_play),
                                np.int64(lib), np.int64(otherlib), bool(use))


@njit(cache=True, nogil=True, _nrt=False)
def _defense_is_hopeless(b, group, owner, to_play, lib, otherlib, use):
    """Whether the defender escapes with a move that does not gain liberties anyway."""
    if not use:
        return False
    if to_play == owner and pb.neighbor_count_at(b, lib, owner) == 1:
        if pb.immediate_liberty_count(b, lib) == 1:
            return True
        if pb.immediate_liberty_count(b, lib) == 2 \
                and coord_is_adjecent(lib, otherlib, pb.board_size(b)):
            return True
    return False


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def can_atari_group(b, group, owner, to_play, q, tag, use_def_no_hopeless, rng):
    """Normalizes the argument types (one compiled ``_can_atari_group``)."""
    return _can_atari_group(b, np.int64(group), np.int64(owner), np.int64(to_play), q,
                            np.int64(tag), bool(use_def_no_hopeless), rng)


@njit(cache=True, nogil=True, _nrt=False)
def _can_atari_group(b, group, owner, to_play, q, tag, use_def_no_hopeless, rng):
    have0 = False
    preference0 = True
    preference1 = True
    for i in range(2):
        lib = b.gi_lib[group, i]
        assert b.b[lib] == S_NONE
        if not pb.board_is_valid_play(b, to_play, lib):
            continue

        # Prevent hopeless escape attempts.
        if defense_is_hopeless(b, group, owner, to_play, lib, b.gi_lib[group, 1 - i],
                               use_def_no_hopeless):
            continue

        # Of course we don't want to play bad selfatari ourselves, if we are the attacker...
        if selfatari.is_bad_selfatari(b, to_play, lib):
            coord = PASS
            bygroup = 0
            if to_play != owner:
                # Okay! We are attacker; maybe we just need to connect a false
                # eye before atari - this is very common in the corner.
                coord, bygroup = selfatari.selfatari_cousin(b, to_play, lib, rng)
            if is_pass(coord):
                continue
            # Ok, connect, but prefer not to.
            byowner = b.b[bygroup]
            # One more thing - is the cousin sensible defense for the other group?
            if defense_is_hopeless(b, bygroup, byowner, to_play, coord, lib, use_def_no_hopeless):
                continue
            lib = coord
            if i == 0:
                preference0 = False
            else:
                preference1 = False

        if i == 0:
            have0 = True

        # If the move is too "lumpy", prefer the alternative.
        if pb.neighbor_count_at(b, lib, to_play) + pb.neighbor_count_at(b, lib, S_OFFBOARD) >= 3:
            if i == 0:
                preference0 = False
            else:
                preference1 = False

        # If we prefer only one of the moves, pick that one.
        if i == 1 and have0 and preference0 != preference1:
            if not preference0:
                if q.move[q.moves[0] - 1] == b.gi_lib[group, 0]:
                    q.moves[0] -= 1
                # ...else{ may happen, since we call mq_nodup() and the
                # move might have been there earlier.
            else:
                assert not preference1
                continue

        # Tasty! Crispy! Good!
        mq_add(q, lib, tag)
        mq_nodup(q)


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def group_2lib_check(b, group, to_play, q, tag, use_miaisafe, use_def_no_hopeless, rng):
    """Normalizes the argument types (one compiled ``_group_2lib_check``)."""
    return _group_2lib_check(b, np.int64(group), np.int64(to_play), q, np.int64(tag),
                             bool(use_miaisafe), bool(use_def_no_hopeless), rng)


@njit(cache=True, nogil=True, _nrt=False)
def _group_2lib_check(b, group, to_play, q, tag, use_miaisafe, use_def_no_hopeless, rng):
    color = b.b[pb.group_base(group)]
    assert color != S_OFFBOARD and color != S_NONE

    # Do not try to atari groups that cannot be harmed.
    if use_miaisafe and miai_2lib(b, group, color):
        return

    can_atari_group(b, group, color, to_play, q, tag, use_def_no_hopeless, rng)

    # Can we counter-atari another group, if we are the defender?
    if to_play != color:
        return
    stone = group
    stone_next = b.p[stone]
    while True:
        for c in pb.neighbors(b, stone):
            if b.b[c] != stone_other(color):
                continue
            g2 = b.g[c]
            if b.gi_libs[g2] == 1:
                # We can capture a neighbor.
                mq_add(q, b.gi_lib[g2, 0], tag)
                mq_nodup(q)
                continue
            if b.gi_libs[g2] != 2:
                continue
            can_atari_group(b, g2, stone_other(color), to_play, q, tag, use_def_no_hopeless, rng)
        stone = stone_next
        if stone == 0:
            break
        stone_next = b.p[stone_next]

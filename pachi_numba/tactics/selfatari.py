# pachi-numba: a translation of Pachi (GPL-2.0-only), October 2026; see NOTICE.
"""Translation of Pachi's ``tactics/selfatari.h`` and ``tactics/selfatari.c``.

A fairly reliable self-atari detector. The C helpers return ``true``,
``false`` or ``-1`` (no decision yet); here 1, 0 and -1.
"""

from collections import namedtuple

import numpy as np
from numba import njit

from .. import board as pb
from .. import mq
from .. import random as fast
from ..move import PASS, coord_is_adjecent, is_pass
from ..stone import S_NONE, S_OFFBOARD, stone_other
from . import nakade, one_lib

# struct selfatari_state; the three scalars are in ``flags``.
SelfatariState = namedtuple("SelfatariState", ["groupcts", "groupids", "groupneis", "flags"])
FRIEND_HAS_NO_LIBS, NEEDS_MORE_LIB, NEEDS_MORE_LIB_EXCEPT = 0, 1, 2


@njit(cache=True, nogil=True, _nrt=False)
def _selfatari_state(b):
    """A zeroed state in ``b.scratch[0:39]`` (C: a struct on the stack)."""
    b.scratch[0:39] = 0
    # groupids and groupneis are [S_MAX][4] in C, flat here: [color * 4 + i].
    return SelfatariState(b.scratch[0:4], b.scratch[4:20], b.scratch[20:36], b.scratch[36:39])


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def is_bad_selfatari(b, color, to):
    """Whether the move is an undesirable self-atari (nakade, eye falsification
    and throw-ins are allowed)."""
    # More than one immediate liberty, thumbs up!
    if pb.immediate_liberty_count(b, to) > 1:
        return False
    return is_bad_selfatari_slow(b, color, to)


@njit(cache=True, nogil=True, _nrt=False)
def three_liberty_suicide(b, g, color, to, s):
    """Whether, after this move, the opponent can unconditionally capture the 3-lib group."""
    size = pb.board_size(b)
    # Extract the other two liberties.
    other_lib0 = 0
    other_lib1 = 0
    other_lib0_adj = False
    other_lib1_adj = False
    j = 0
    for i in range(3):
        lib = b.gi_lib[g, i]
        if lib != to:
            if j == 0:
                other_lib0_adj = coord_is_adjecent(lib, to, size)
                other_lib0 = lib
            else:
                other_lib1_adj = coord_is_adjecent(lib, to, size)
                other_lib1 = lib
            j += 1

    # Make sure this move is not useful by gaining liberties,
    # splitting the other two liberties or connecting to a different group.
    if pb.immediate_liberty_count(b, to) - int(other_lib0_adj or other_lib1_adj) > 0:
        return False
    assert not (other_lib0_adj and other_lib1_adj)
    if s.groupcts[color] > 1:
        return False

    # Playing on the third liberty might be useful if it enables
    # capturing some group (are we doing nakade or semeai?).
    other = stone_other(color)
    for i in range(s.groupcts[other]):
        if b.gi_libs[s.groupids[other * 4 + i]] <= 3:
            return False

    # The final suicidal test is: (After filling this liberty,) when opponent
    # fills liberty [0], playing liberty [1] will not help the group, or vice versa.
    other_libs_neighbors = coord_is_adjecent(other_lib0, other_lib1, size)
    for i in range(2):
        lib_i = other_lib0 if i == 0 else other_lib1
        lib_other = other_lib1 if i == 0 else other_lib0
        adj_i = other_lib0_adj if i == 0 else other_lib1_adj
        null_libs = int(other_libs_neighbors) + int(adj_i)
        if pb.board_is_one_point_eye(b, lib_other, color):
            # The other liberty is an eye, happily go ahead.
            return False
        if pb.immediate_liberty_count(b, lib_i) - null_libs > 1:
            continue  # Gains liberties.
        can_connect = False
        for c in pb.neighbors(b, lib_i):
            if b.b[c] == color and b.g[c] != g and b.gi_libs[b.g[c]] > 1:
                can_connect = True  # Can connect to a friend.
        if can_connect:
            continue
        # Ok, the last liberty has no way to get out.
        return True
    return False


@njit(cache=True, nogil=True, _nrt=False)
def examine_friendly_groups(b, color, to, s):
    for i in range(s.groupcts[color]):
        # We can escape by connecting to this group if it's not in atari.
        g = s.groupids[color * 4 + i]
        if b.gi_libs[g] == 1:
            if not s.flags[NEEDS_MORE_LIB]:
                s.flags[FRIEND_HAS_NO_LIBS] = 1
            # or we already have a friend with 1 lib
            continue

        # Could we self-atari the group here?
        if b.gi_libs[g] > 2:
            if b.gi_libs[g] == 3 and three_liberty_suicide(b, g, color, to, s):
                return 1
            return 0

        # We need to have another liberty, and it must not be the other
        # liberty of the group.
        lib2 = pb.board_group_other_lib(b, g, to)
        # Maybe we already looked at another group providing one liberty?
        if s.flags[NEEDS_MORE_LIB] and s.flags[NEEDS_MORE_LIB] != g \
                and s.flags[NEEDS_MORE_LIB_EXCEPT] != lib2:
            return 0

        # Can we get the liberty locally?
        # Yes if we are route to more liberties...
        if s.groupcts[S_NONE] > 1:
            return 0
        # ...or one liberty, but not lib2.
        if s.groupcts[S_NONE] > 0 and not coord_is_adjecent(lib2, to, pb.board_size(b)):
            return 0

        # ...ok, then we can still contribute a liberty later by capturing something.
        s.flags[NEEDS_MORE_LIB] = g
        s.flags[NEEDS_MORE_LIB_EXCEPT] = lib2
        s.flags[FRIEND_HAS_NO_LIBS] = 0
    return -1


@njit(cache=True, nogil=True, _nrt=False)
def examine_enemy_groups(b, color, to, s):
    # We may be able to gain a liberty by capturing this group.
    can_capture = 0
    other = stone_other(color)
    for i in range(s.groupcts[other]):
        # We can escape by capturing this group if it's in atari.
        g = s.groupids[other * 4 + i]
        if b.gi_libs[g] > 1:
            continue
        # But we need to get to at least two liberties by this; we already have
        # one outside liberty, or the group is more than 1 stone.
        if s.groupcts[S_NONE] > 0 or not pb.group_is_onestone(b, g):
            return 0
        # ...or, it's a ko stone,
        if pb.neighbor_count_at(b, g, color) + pb.neighbor_count_at(b, g, S_OFFBOARD) == 3:
            # and we don't have a group to save: then, just taking
            # single stone means snapback!
            if not s.flags[FRIEND_HAS_NO_LIBS]:
                return 0
        # ...or, we already have one indirect liberty provided by another group.
        if s.flags[NEEDS_MORE_LIB] or (can_capture and can_capture != g):
            return 0
        can_capture = g

    if not s.flags[NEEDS_MORE_LIB] and not can_capture and not s.groupcts[S_NONE]:
        # This move is simply a suicide, not even a self-atari.
        return 1
    return -1


@njit(cache=True, nogil=True, _nrt=False)
def is_neighbor_group(b, color, g, s):
    for i in range(s.groupcts[color]):
        if g == s.groupids[color * 4 + i]:
            return True
    return False


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def is_bad_nakade(b, color, to, lib2, check_enemy, s):
    """Normalizes the argument types (one compiled ``_is_bad_nakade``)."""
    return _is_bad_nakade(b, np.int64(color), np.int64(to), np.int64(lib2), bool(check_enemy), s)


@njit(cache=True, nogil=True, _nrt=False)
def _is_bad_nakade(b, color, to, lib2, check_enemy, s):
    """Instead of this self-atari, could we connect or escape at the other
    liberty ``lib2`` of a neighbor group? Or is there a strong enemy group there?"""
    for c in pb.neighbors(b, lib2):
        # If the other liberty has an empty neighbor, it must be the original
        # liberty; otherwise we are nakade'ing an eyeless group from outside.
        if b.b[c] == S_NONE:
            if c == to:
                continue
            return True
    for c in pb.neighbors(b, lib2):
        if b.b[c] != color:
            continue
        g2 = b.g[c]
        # Looking for a group we don't know about
        if is_neighbor_group(b, color, g2, s):
            continue
        # Should connect these groups instead of self-atari on the other side.
        if b.b[c] == color:
            return True
        # (C: unreachable checks of enemy groups follow.)
        if not check_enemy:
            continue
        if b.gi_libs[g2] == 1:
            continue
        if b.gi_libs[g2] == 2 and (b.gi_lib[g2, 0] == to or b.gi_lib[g2, 1] == to):
            continue
        return True  # Stronger enemy group. No nakade.
    return False


@njit(cache=True, nogil=True, _nrt=False)
def can_escape_instead(b, color, to, s):
    """Could we have connected or escaped at the other liberty of a neighbor group?"""
    for i in range(s.groupcts[color]):
        g = s.groupids[color * 4 + i]
        if b.gi_libs[g] != 2:
            continue
        other = pb.board_group_other_lib(b, g, to)
        # Let's look at the other liberty of that group.
        if pb.immediate_liberty_count(b, other) >= 2 \
                or is_bad_nakade(b, color, to, other, False, s):
            return True
    return False


@njit(cache=True, nogil=True, _nrt=False)
def unreachable_lib_from_neighbors(b, color, to, s, lib):
    for i in range(s.groupcts[color]):
        g = s.groupids[color * 4 + i]
        for j in range(b.gi_libs[g]):
            if b.gi_lib[g, j] == lib:
                return False
    return True


@njit(cache=True, nogil=True, _nrt=False)
def capture_would_make_extra_eye(b, color, to, s):
    """This only looks at existing empty spots, not captures."""
    for c in pb.neighbors(b, to):
        if b.b[c] == S_NONE and unreachable_lib_from_neighbors(b, color, to, s, c):
            return True
    return False


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def nakade_making_dead_shape(b, color, to, s, atariing_group, stones):
    """Normalizes the argument types (one compiled ``_nakade_making_dead_shape``)."""
    return _nakade_making_dead_shape(b, np.int64(color), np.int64(to), s, bool(atariing_group),
                                     np.int64(stones))


@njit(cache=True, nogil=True, _nrt=True)
def _nakade_making_dead_shape(b, color, to, s, atariing_group, stones):
    if not stones:
        return False
    assert stones != 1
    assert stones <= 5

    # If not atariing surrounding group it's a good move if:
    #   - Shape after capturing us is dead AND
    #     - Oponent gets extra eye if he plays first OR
    #     - would create living shape
    # If atariing surrounding group we only care about dead shape.
    cap_would_make_eye = False
    if not atariing_group:
        cap_would_make_eye = capture_would_make_extra_eye(b, color, to, s)

    # Can oponent make living shape if we don't play ?
    # (don't bother killing stuff that's already dead...)
    if not atariing_group and not cap_would_make_eye and s.groupcts[color] == 1:
        b2 = pb.board_duplicate(b)
        # Play opponent color where we want to play
        if pb.board_play(b2, to, stone_other(color)) != -1:
            # Had 2 libs ? One more move to capture us then
            if pb.neighbor_count_at(b, to, color) == pb.neighbor_count_at(b2, to, color):
                standing = -1
                for c in pb.neighbors(b2, to):
                    g = b2.g[c]
                    if b2.b[c] == color:
                        assert b2.gi_libs[g] == 1  # Should be in atari
                        standing = g
                assert standing != -1
                r = pb.board_play(b2, b2.gi_lib[standing, 0], stone_other(color))
                assert r != -1
            # Empty now since it's been captured
            empty = pb.group_base(s.groupids[color * 4 + 0])
            would_live = not nakade.nakade_dead_shape(b2, empty, stone_other(color))
            if not would_live:  # And !cap_would_make_eye here
                return False  # Bad nakade

    b2 = pb.board_duplicate(b)
    # Play self-atari
    r = pb.board_play(b2, to, color)
    assert r != -1
    # Play capture
    g = b2.g[to]
    r = pb.board_play(b2, b2.gi_lib[g, 0], stone_other(color))
    assert r != -1
    return nakade.nakade_dead_shape(b2, to, stone_other(color))


@njit(cache=True, nogil=True, _nrt=False)
def _check_throw_in_or_inside_capture(b, color, s, capturing):
    """C macro check_throw_in_or_inside_capture: 1/0, or -1 to go on."""
    if s.groupcts[color] == 1 and pb.group_is_onestone(b, s.groupids[color * 4 + 0]):
        g2 = s.groupids[color * 4 + 0]
        assert b.gi_libs[g2] <= 2
        if b.gi_libs[g2] == 1:
            return 0
        return 0 if capturing else 1
    return -1


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def setup_nakade_or_snapback(b, color, to, s):
    """Normalizes the argument types (one compiled ``_setup_nakade_or_snapback``)."""
    return _setup_nakade_or_snapback(b, np.int64(color), np.int64(to), s)


@njit(cache=True, nogil=True, _nrt=False)
def _setup_nakade_or_snapback(b, color, to, s):
    # Look at the enemy groups and determine the other contended liberty.
    other = stone_other(color)
    lib2 = PASS
    for i in range(s.groupcts[other]):
        g = s.groupids[other * 4 + i]
        if b.gi_libs[g] != 2:
            continue
        this_lib2 = pb.board_group_other_lib(b, g, to)
        if is_pass(lib2):
            lib2 = this_lib2
        elif this_lib2 != lib2:
            # Two neighboring groups that do not share the other liberty:
            # for sure not a good nakade.
            return -1

    if is_pass(lib2):
        # Not putting any group in atari. Could be creating dead shape though.
        # Before checking if it's a useful nakade make sure it can't connect out!
        if can_escape_instead(b, color, to, s):
            return -1
        d = _check_throw_in_or_inside_capture(b, color, s, False)
        if d >= 0:
            return d
        stones = 0
        for j in range(s.groupcts[color]):
            stones += pb.group_stone_count(b, s.groupids[color * 4 + j], 6)
            if stones > 5:
                return 1
        return 0 if nakade_making_dead_shape(b, color, to, s, False, stones) else -1

    # Let's look at neighbors of the other liberty:
    if is_bad_nakade(b, color, to, lib2, True, s) or can_escape_instead(b, color, to, s):
        return -1

    # Now, we must distinguish between nakade and eye falsification; moreover,
    # we must not falsify an eye by more than two stones.
    if s.groupcts[color] < 1:
        return 0  # Simple throw-in, an easy case
    d = _check_throw_in_or_inside_capture(b, color, s, True)
    if d >= 0:
        return d

    # We would create more than 2-stone group; in that case, the liberty of
    # our result must be lib2, indicating this really is a nakade.
    stones = 0
    for j in range(s.groupcts[color]):
        g2 = s.groupids[color * 4 + j]
        assert b.gi_libs[g2] <= 2
        if b.gi_libs[g2] == 2:
            if b.gi_lib[g2, 0] != lib2 and b.gi_lib[g2, 1] != lib2:
                return -1
        else:
            assert b.gi_lib[g2, 0] == to
        stones += pb.group_stone_count(b, g2, 6)
        if stones > 5:
            return 1
    return 0 if nakade_making_dead_shape(b, color, to, s, True, stones) else 1


@njit(cache=True, nogil=True, _nrt=False)
def check_throwin(b, color, to, s):
    """We can be throwing-in to false eye; we cannot sensibly throw-in into a corner."""
    other = stone_other(color)
    if pb.neighbor_count_at(b, to, S_OFFBOARD) < 2 \
            and pb.neighbor_count_at(b, to, other) + pb.neighbor_count_at(b, to, S_OFFBOARD) == 3 \
            and pb.board_is_false_eyelike(b, to, other):
        assert s.groupcts[color] <= 1
        # Single-stone throw-in may be ok...
        if s.groupcts[color] == 0:
            # There is one problem - when it's actually not a throw-in!
            for c in pb.neighbors(b, to):
                if b.b[c] == S_NONE:
                    # Is the empty neighbor an escape path?
                    if pb.neighbor_count_at(b, c, other) \
                            + pb.neighbor_count_at(b, c, S_OFFBOARD) < 2:
                        return -1
            return 0

        # Multi-stone throwin...?
        assert s.groupcts[color] == 1
        g = s.groupids[color * 4 + 0]
        assert b.gi_libs[g] <= 2
        # Suicide is definitely NOT ok, no matter what else we could test.
        if b.gi_libs[g] == 1:
            return 1
        # In that case, we must be connected to at most one stone,
        # or throwin will not destroy any eyes.
        if pb.group_is_onestone(b, g):
            return 0
    return -1


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def is_bad_selfatari_slow(b, color, to):
    """Normalizes the argument types (one compiled ``_is_bad_selfatari_slow``)."""
    return _is_bad_selfatari_slow(b, np.int64(color), np.int64(to))


@njit(cache=True, nogil=True, _nrt=False)
def _is_bad_selfatari_slow(b, color, to):
    # Assess if we actually gain any liberties by this escape route.
    s = _selfatari_state(b)
    for c in pb.neighbors(b, to):
        neighbor_color = b.b[c]
        group = b.g[c]
        dup = False
        for i in range(s.groupcts[neighbor_color]):
            if s.groupids[neighbor_color * 4 + i] == group:
                dup = True
                break
        if not dup:
            s.groupneis[neighbor_color * 4 + s.groupcts[neighbor_color]] = c
            s.groupids[neighbor_color * 4 + s.groupcts[neighbor_color]] = group
            s.groupcts[neighbor_color] += 1

    # We have shortage of liberties; that's the point.
    assert s.groupcts[S_NONE] <= 1

    d = examine_friendly_groups(b, color, to, s)
    if d >= 0:
        return d != 0
    d = examine_enemy_groups(b, color, to, s)
    if d >= 0:
        return d != 0
    d = check_throwin(b, color, to, s)
    if d >= 0:
        return d != 0
    d = setup_nakade_or_snapback(b, color, to, s)
    if d >= 0:
        return d != 0
    # No way to pull out, no way to connect out. This really is a bad self-atari!
    return True


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def selfatari_cousin(b, color, coord, rng):
    """Normalizes the argument types (one compiled ``_selfatari_cousin``)."""
    return _selfatari_cousin(b, np.int64(color), np.int64(coord), rng)


@njit(cache=True, nogil=True, _nrt=True)
def _selfatari_cousin(b, color, coord, rng):
    """The other liberty of a neighbor group the self-atari ``coord`` puts in atari,
    if that is not a self-atari. Returns (move or pass, the group)."""
    groups = np.zeros(4, np.int64)
    groups_n = 0
    groupsbycolor = np.zeros(4, np.int64)
    for c in pb.neighbors(b, coord):
        stone = b.b[c]
        g = b.g[c]
        if b.gi_libs[g] == 2:
            groups[groups_n] = g
            groups_n += 1
            groupsbycolor[stone] += 1
    if not groups_n:
        return PASS, 0

    other = stone_other(color)
    if groupsbycolor[other]:
        # Prefer to fill the other liberty of an opponent group
        # to filling own approach liberties.
        gl = np.int64(fast.fast_random(rng, groups_n))
        found = False
        gn = gl
        while gn < groups_n:
            if b.b[groups[gn]] == other:
                found = True
                break
            gn += 1
        if not found:
            gn = 0
            while gn < gl:
                if b.b[groups[gn]] == other:
                    break
                gn += 1
    else:
        gn = np.int64(fast.fast_random(rng, groups_n))

    gl = gn
    ccq = mq.mq_new()
    while gn - gl < groups_n:
        gnm = gn % groups_n
        group = groups[gnm]
        gn += 1
        # Can we get liberties by capturing a neighbor?
        mq.mq_clear(ccq)
        if one_lib.can_countercapture(b, color, group, color, ccq, True, 0):
            lib2 = mq.mq_pick(ccq, rng)
        else:
            lib2 = pb.board_group_other_lib(b, group, coord)
            if pb.board_is_one_point_eye(b, lib2, b.b[group]):
                continue
            if is_bad_selfatari(b, color, lib2):
                continue
        return lib2, group
    return PASS, 0

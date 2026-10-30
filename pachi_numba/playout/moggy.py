# pachi-numba: a translation of Pachi (GPL-2.0-only), October 2026; see NOTICE.
"""Translation of Pachi's ``playout/moggy.c`` and ``playout/light.c``.

Heuristical playout (and tree prior) policy modelled primarily after the
description of the Mogo engine. The ``struct moggy_policy`` is a ``Moggy``
named tuple; its integer settings are in ``params`` at the indices below.
The light policy (``playout_light_init``) is a ``Moggy`` of kind
``KIND_LIGHT``: it chooses no moves and permits all.

There is no joseki dictionary: C loads ``joseki<size>.pdict`` from the
working directory, which Pachi does not ship, so ``joseki_check`` never
adds moves (its random number is still drawn, as in C).
"""

import re
import threading
from collections import namedtuple

import numpy as np
from numba import njit

from .. import board as pb
from .. import mq
from .. import pattern3
from .. import random as fast
from ..move import PASS, coord_is_8adjecent, is_pass
from ..stone import S_NONE, S_OFFBOARD, stone_other
from ..tactics import ladder, nakade, n_lib, one_lib, selfatari, two_lib
from ..uct import prior

# enum mq_tag
MQ_KO, MQ_LATARI, MQ_L2LIB, MQ_LNLIB, MQ_PAT3, MQ_GATARI, MQ_JOSEKI, MQ_NAKADE, MQ_MAX = range(9)
MQ_LADDER = MQ_L2LIB  # XXX: We want to fit in char still!

PAT3_N = 15
KIND_LIGHT, KIND_MOGGY = 0, 1

(KIND, LCAPTURERATE, ATARIRATE, NLIBRATE, LADDERRATE, CAPTURERATE, PATTERNRATE, KORATE,
 JOSEKIRATE, NAKADERATE, EYEFIXRATE, SELFATARIRATE, EYEFILLRATE, ALWAYSCCAPRATE,
 FILLBOARDTRIES, KOAGE, PATTERN2, SELFATARI_OTHER, MIDDLE_LADDER, CAPCHECKALL, CAP_STONE_MIN,
 CAP_STONE_MAX, CAP_STONE_DENOM, ATARI_DEF_NO_HOPELESS, ATARI_MIAISAFE, NLIB_COUNT, FULLCHOOSE,
 CHOOSE_FULL, DEBUG_LEVEL, NUM_PARAMS) = range(30)

Moggy = namedtuple("Moggy", ["params", "pattern", "value", "pat3_gammas", "mq_prob", "tenuki_prob"])

# The default 3x3 patterns ("#if 0" ones left out); see pattern3.py for the encoding.
MOGGY_PATTERNS_SRC = [
    "XOX...???",  # hane pattern - enclosing hane
    "YO....?.?",  # hane pattern - non-cutting hane
    "XO?X..x.?",  # hane pattern - magari
    "XOO...?.?X",  # hane pattern - thin hane
    ".Q.Y.....",  # generic pattern - katatsuke or diagonal attachment; similar to magari
    "XO?O.o?o?",  # cut1 pattern (kiri) - unprotected cut
    "XO?O.X???",  # cut1 pattern (kiri) - peeped cut
    "?X?O.Oooo",  # cut2 pattern (de)
    "OX??.O?o?",  # cut keima (not in Mogo)
    "X.?O.?##?",  # side pattern - chase
    "OX?X.O###",  # side pattern - block side cut
    "?X?x.O###",  # side pattern - block side connection
    "?XQx.x###",  # side pattern - sagari (SUSPICIOUS)
    "?OYY.O###",  # side pattern - cut (SUSPICIOUS)
    "?X.Q.X###",  # side pattern - make eye
]

_UNSET = 0xFFFFFFFF  # C: -1U, "use rate"


# ---- initialization (plain Python)

def _atoi(text):
    match = re.match(r"\s*[+-]?\d+", text or "")
    return int(match.group()) if match else 0


def _atof(text):
    match = re.match(r"\s*[+-]?(\d+\.?\d*([eE][+-]?\d+)?|\.\d+([eE][+-]?\d+)?)", text or "")
    return float(match.group()) if match else 0.0


def _unsigned(value):
    return value & 0xFFFFFFFF


# The pattern table is built once and then only read, by all policies (and threads).
_PATTERN_CACHE = {}
_PATTERN_LOCK = threading.Lock()


def _patterns():
    with _PATTERN_LOCK:
        if "moggy" not in _PATTERN_CACHE:
            _PATTERN_CACHE["moggy"] = pattern3.pattern3s_init(MOGGY_PATTERNS_SRC)
        return _PATTERN_CACHE["moggy"]


def playout_light_init(arg, b):
    """The light policy: uniformly random moves."""
    params = np.zeros(NUM_PARAMS, np.int64)
    params[KIND] = KIND_LIGHT
    return Moggy(params, np.zeros(1, np.uint32), np.zeros(1, np.uint8), np.zeros(PAT3_N),
                 np.zeros(MQ_MAX), np.zeros(1))


def playout_moggy_init(arg, b):
    """The moggy policy with the options ``arg`` ("name=value:name=value", or None)."""
    params = np.zeros(NUM_PARAMS, np.int64)
    params[KIND] = KIND_MOGGY
    # These settings are tuned for 19x19 play with several threads on reasonable
    # time limits. XXX: no 9x9 tuning has been done recently.
    rate = 80 if pb.board_large(b) else 90
    params[PATTERNRATE] = params[EYEFIXRATE] = 100
    params[LCAPTURERATE] = 90
    params[ATARIRATE] = params[JOSEKIRATE] = _UNSET
    params[NAKADERATE] = 60
    params[KORATE] = 40
    params[KOAGE] = 4
    params[ALWAYSCCAPRATE] = 40
    params[EYEFILLRATE] = 60
    params[NLIBRATE] = 25
    params[SELFATARIRATE] = 60
    params[SELFATARI_OTHER] = 1
    params[PATTERN2] = 1
    params[CAP_STONE_MIN] = 2
    params[CAP_STONE_MAX] = 15
    params[CAP_STONE_DENOM] = 200
    params[ATARI_DEF_NO_HOPELESS] = not pb.board_large(b)
    params[ATARI_MIAISAFE] = 1
    params[NLIB_COUNT] = 4

    mq_prob = np.zeros(MQ_MAX)
    mq_prob[MQ_KO] = 6.0
    mq_prob[MQ_NAKADE] = 5.5
    mq_prob[MQ_LATARI] = 5.0
    mq_prob[MQ_L2LIB] = 4.0
    mq_prob[MQ_LNLIB] = 3.5
    mq_prob[MQ_PAT3] = 3.0
    mq_prob[MQ_GATARI] = 2.0
    mq_prob[MQ_JOSEKI] = 1.0
    # Default 3x3 pattern gammas tuned on 15x15 with 500s/game on i7-3770
    # single thread using 40000 CLOP games.
    pat3_gammas = np.array([0.52, 0.53, 0.32, 0.22, 0.37, 0.28, 0.21, 0.19, 0.82,
                            0.12, 0.20, 0.11, 0.16, 0.57, 0.44])
    tenuki_prob = np.zeros(1)

    rates = {"lcapturerate": LCAPTURERATE, "ladderrate": LADDERRATE, "atarirate": ATARIRATE,
             "nlibrate": NLIBRATE, "capturerate": CAPTURERATE, "patternrate": PATTERNRATE,
             "selfatarirate": SELFATARIRATE, "eyefillrate": EYEFILLRATE, "korate": KORATE,
             "josekirate": JOSEKIRATE, "nakaderate": NAKADERATE, "eyefixrate": EYEFIXRATE,
             "alwaysccaprate": ALWAYSCCAPRATE}
    integers = {"debug": DEBUG_LEVEL, "koage": KOAGE, "cap_stone_min": CAP_STONE_MIN,
                "cap_stone_max": CAP_STONE_MAX, "cap_stone_denom": CAP_STONE_DENOM,
                "nlib_count": NLIB_COUNT}
    flags = {"pattern2": PATTERN2, "selfatari_other": SELFATARI_OTHER,
             "capcheckall": CAPCHECKALL, "atari_miaisafe": ATARI_MIAISAFE,
             "atari_def_no_hopeless": ATARI_DEF_NO_HOPELESS, "middle_ladder": MIDDLE_LADDER}
    for optspec in (arg.split(":") if arg else []):
        optname, _, optval = optspec.partition("=")
        has_value = "=" in optspec
        name = optname.lower()
        if name in rates and has_value:
            params[rates[name]] = _unsigned(_atoi(optval))
        elif name in integers and has_value:
            params[integers[name]] = _atoi(optval)
        elif name == "rate" and has_value:
            rate = _atoi(optval)
        elif name == "fillboardtries":
            params[FILLBOARDTRIES] = _unsigned(_atoi(optval))
        elif name in flags:
            params[flags[name]] = not (has_value and optval[:1] == "0")
        elif name == "fullchoose":
            params[FULLCHOOSE] = 1
            params[CHOOSE_FULL] = not (has_value and optval[:1] == "0")
        elif name == "mqprob" and has_value:
            # KO%LATARI%L2LIB%LNLIB%PAT3%GATARI%JOSEKI%NAKADE
            for i, value in enumerate(optval.split("%")[:MQ_MAX]):
                mq_prob[i] = _atof(value)
        elif name == "pat3gammas" and has_value:
            for i, value in enumerate(optval.split("%")[:PAT3_N]):
                pat3_gammas[i] = _atof(value)
        elif name == "tenukiprob" and has_value:
            tenuki_prob[0] = _atof(optval)
        else:
            raise ValueError(f"playout-moggy: Invalid policy argument {optname} or missing value")

    for index in (LCAPTURERATE, ATARIRATE, NLIBRATE, CAPTURERATE, PATTERNRATE, SELFATARIRATE,
                  EYEFILLRATE, KORATE, JOSEKIRATE, LADDERRATE, NAKADERATE, EYEFIXRATE,
                  ALWAYSCCAPRATE):
        if params[index] == _UNSET:
            params[index] = _unsigned(rate)
    patterns = _patterns()
    return Moggy(params, patterns.pattern, patterns.value, pat3_gammas, mq_prob, tenuki_prob)


# ---- move generators

@njit(cache=True, nogil=True, inline="always", _nrt=False)
def _random_below(rate, rng):
    """C: ``rate > fast_random(100)``."""
    return rate > fast.fast_random(rng, 100)


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def test_pattern3_here(p, b, coord, color, middle_ladder, q):
    """Whether a 3x3 pattern matches the move and it is not obviously stupid; and its gamma.

    ``q`` is any move queue: it is passed as C's NULL queue and not touched.
    """
    matches, pi = pattern3.pattern3_move_here(pattern3.Pattern3s(p.pattern, p.value), b, coord,
                                              color)
    if not matches:
        return False, 0.0
    if selfatari.is_bad_selfatari(b, color, coord):
        return False, 0.0
    # Ladder moves are stupid.
    atari_neighbor = pb.board_get_atari_neighbor(b, coord, color)
    if atari_neighbor and ladder.is_ladder(b, coord, atari_neighbor, middle_ladder) \
            and not one_lib.can_countercapture(b, b.b[pb.group_base(atari_neighbor)],
                                               atari_neighbor, color, q, False, 0):
        return False, 0.0
    return True, p.pat3_gammas[pi]


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def apply_pattern_here(p, b, c, color, q, gammas):
    if pb.board_is_valid_play(b, color, c):
        matches, gamma = test_pattern3_here(p, b, c, color, p.params[MIDDLE_LADDER] != 0, q)
        if matches:
            mq.mq_gamma_add(q, gammas, c, gamma, 1 << MQ_PAT3)


@njit(cache=True, nogil=True, _nrt=False)
def apply_pattern(p, b, m_coord, m_color, has_mm, mm_coord, q, gammas):
    """Check if we match any pattern around given move (with the other color to play)."""
    # Suicides do not make any patterns and confuse us.
    if b.b[m_coord] == S_NONE or b.b[m_coord] == S_OFFBOARD:
        return
    c = m_coord
    for i in range(8):
        c += b.nei8[i]
        apply_pattern_here(p, b, c, stone_other(m_color), q, gammas)
    if has_mm:  # Second move for pattern searching
        c = mm_coord
        for i in range(8):
            c += b.nei8[i]
            if coord_is_8adjecent(m_coord, c, pb.board_size(b)):
                continue
            apply_pattern_here(p, b, c, stone_other(m_color), q, gammas)


@njit(cache=True, nogil=True, _nrt=False)
def joseki_check(p, b, to_play, q):
    """No joseki dictionary (see the module docstring)."""


@njit(cache=True, nogil=True, _nrt=False)
def global_atari_check(p, b, to_play, q, rng):
    if b.ints[pb.CLEN] == 0:
        return
    rate = p.params[ALWAYSCCAPRATE]
    middle = p.params[MIDDLE_LADDER] != 0
    if p.params[CAPCHECKALL]:
        for g in range(b.ints[pb.CLEN]):
            one_lib.group_atari_check(rate, b, b.g[pb.group_base(b.c[g])], to_play, q,
                                      False, middle, 1 << MQ_GATARI, rng)
        if p.params[FULLCHOOSE]:
            return
    g_base = np.int64(fast.fast_random(rng, b.ints[pb.CLEN]))
    for g in range(g_base, b.ints[pb.CLEN]):
        one_lib.group_atari_check(rate, b, b.g[pb.group_base(b.c[g])], to_play, q,
                                  False, middle, 1 << MQ_GATARI, rng)
        if q.moves[0] > 0 and p.params[FULLCHOOSE]:
            return  # XXX: Try carrying on.
    for g in range(0, g_base):
        one_lib.group_atari_check(rate, b, b.g[pb.group_base(b.c[g])], to_play, q,
                                  False, middle, 1 << MQ_GATARI, rng)
        if q.moves[0] > 0 and p.params[FULLCHOOSE]:
            return


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def local_atari_check(p, b, m_coord, m_color, q, rng):
    rate = p.params[ALWAYSCCAPRATE]
    middle = p.params[MIDDLE_LADDER] != 0
    # Did the opponent play a self-atari?
    if b.gi_libs[b.g[m_coord]] == 1:
        one_lib.group_atari_check(rate, b, b.g[m_coord], stone_other(m_color), q,
                                  False, middle, 1 << MQ_LATARI, rng)
    for c in pb.neighbors(b, m_coord):
        g = b.g[c]
        if not g or b.gi_libs[g] != 1:
            continue
        one_lib.group_atari_check(rate, b, g, stone_other(m_color), q, False, middle,
                                  1 << MQ_LATARI, rng)


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def local_ladder_check(p, b, m_coord, q):
    group = b.g[m_coord]
    if b.gi_libs[group] != 2:
        return
    for i in range(2):
        chase = b.gi_lib[group, i]
        escape = b.gi_lib[group, 1 - i]
        if ladder.wouldbe_ladder(b, group, escape, chase, b.b[group]):
            mq.mq_add(q, chase, 1 << MQ_LADDER)


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def local_2lib_check(p, b, m_coord, m_color, q, rng):
    group = b.g[m_coord]
    group2 = 0
    miaisafe = p.params[ATARI_MIAISAFE] != 0
    no_hopeless = p.params[ATARI_DEF_NO_HOPELESS] != 0
    # Does the opponent have just two liberties?
    if b.gi_libs[group] == 2:
        two_lib.group_2lib_check(b, group, stone_other(m_color), q, 1 << MQ_L2LIB, miaisafe,
                                 no_hopeless, rng)
    # Then he took a third liberty from neighboring chain?
    for c in pb.neighbors(b, m_coord):
        g = b.g[c]
        if not g or g == group or g == group2 or b.gi_libs[g] != 2:
            continue
        two_lib.group_2lib_check(b, g, stone_other(m_color), q, 1 << MQ_L2LIB, miaisafe,
                                 no_hopeless, rng)
        group2 = g  # prevent trivial repeated checks


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def local_nlib_check(p, b, m_coord, m_color, q, rng):
    """Defense of our few-liberty groups next to the last move (see C for the reasoning)."""
    color = stone_other(m_color)
    group2 = 0
    c = m_coord
    for i in range(8):
        c += b.nei8[i]
        g = b.g[c]
        if not g or group2 == g or b.b[c] != color:
            continue
        if b.gi_libs[g] < 3 or b.gi_libs[g] > p.params[NLIB_COUNT]:
            continue
        n_lib.group_nlib_defense_check(b, g, color, q, 1 << MQ_LNLIB, rng)
        group2 = g  # prevent trivial repeated checks


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def nakade_check(p, b, m_coord, to_play):
    empty = PASS
    for c in pb.neighbors(b, m_coord):
        if b.b[c] != S_NONE:
            continue
        if is_pass(empty):
            empty = c
            continue
        if not coord_is_8adjecent(c, empty, pb.board_size(b)):
            return PASS  # Seems like impossible nakade shape!
    assert not is_pass(empty)
    return nakade.nakade_point(b, empty, stone_other(to_play))


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def eye_fix_check(p, b, m_coord, m_color, to_play, q):
    """The opponent could have filled an approach liberty for falsifying an eye."""
    size = pb.board_size(b)
    # Iterator for walking coordinates in a clockwise fashion
    # (nei8 jumps "over" the middle point, inst. of "around").
    nei8_clockwise = (-size - 1, 1, 1, size, size, -1, -1, -size, -size, 1)
    c = m_coord
    for dni in range(0, 8, 2):
        c0 = c + nei8_clockwise[dni]  # one diagonal neighbor
        c1 = c0 + nei8_clockwise[dni + 1]  # adjecent staight neighbor
        c2 = c1 + nei8_clockwise[dni + 2]  # and adjecent another diagonal neighbor
        # The last move must have a pair of unfriendly diagonal neighbors
        # separated by a friendly stone.
        if (b.b[c0] == to_play or b.b[c0] == S_OFFBOARD) and b.b[c1] == m_color \
                and (b.b[c2] == to_play or b.b[c2] == S_OFFBOARD) \
                and pb.immediate_liberty_count(b, c1) > 0:
            for n in pb.neighbors(b, c1):
                if n == m_coord or b.b[n] != S_NONE:
                    continue
                # ...and the neighbor must potentially falsify an eye.
                falsifying = n
                d = falsifying
                for i in range(4):
                    d += b.dnei[i]
                    if b.b[d] != S_NONE:
                        continue
                    if not pb.board_is_eyelike(b, d, to_play):
                        continue
                    falsified = d
                    # C: color_diag_libs[S_MAX]; only two of the counts are read.
                    opponent_diag = 0
                    offboard_diag = 0
                    e = falsified
                    for j in range(4):
                        e += b.dnei[j]
                        if b.b[e] == m_color and b.gi_libs[b.g[e]] == 1:
                            # Suggest capturing a falsifying stone in atari.
                            mq.mq_add(q, b.gi_lib[b.g[e], 0], 0)
                        else:
                            opponent_diag += b.b[e] == m_color
                            offboard_diag += b.b[e] == S_OFFBOARD
                    if opponent_diag == 1 or (opponent_diag == 0 and offboard_diag == 2):
                        # That's it. Fill the falsifying liberty before it's too late!
                        mq.mq_add(q, falsifying, 0)
        c = c1


@njit(cache=True, nogil=True, _nrt=False)
def fillboard_check(p, b, rng):
    fbtries = b.ints[pb.FLEN] // 8
    if p.params[FILLBOARDTRIES] < fbtries:
        fbtries = p.params[FILLBOARDTRIES]
    for _ in range(fbtries):
        coord = b.f[fast.fast_random(rng, b.ints[pb.FLEN])]
        if pb.immediate_liberty_count(b, coord) != 4:
            continue
        is_empty_around = True
        c = coord
        for i in range(4):
            c += b.dnei[i]
            if b.b[c] != S_NONE:
                is_empty_around = False
                break
        if is_empty_around:
            return coord
    return PASS


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def _has_mm(p, b):
    return p.params[PATTERN2] != 0 and b.ints[pb.LAST_MOVE2_COORD] >= 0


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def playout_moggy_seqchoose(p, b, to_play, rng, q, gammas):
    """Apply the fixed set of decision rules in order, each with its probability."""
    mq.mq_clear(q)
    last_coord = b.ints[pb.LAST_MOVE_COORD]
    last_color = b.ints[pb.LAST_MOVE_COLOR]

    # Ko fight check
    if not is_pass(b.ints[pb.LAST_KO_COORD]) and is_pass(b.ints[pb.KO_COORD]) \
            and b.ints[pb.MOVES] - b.ints[pb.LAST_KO_AGE] < p.params[KOAGE] \
            and _random_below(p.params[KORATE], rng):
        ko = b.ints[pb.LAST_KO_COORD]
        if pb.board_is_valid_play(b, to_play, ko) \
                and not selfatari.is_bad_selfatari(b, to_play, ko):
            return ko

    # Local checks
    if not is_pass(last_coord):
        # Local group in atari?
        if _random_below(p.params[LCAPTURERATE], rng):
            mq.mq_clear(q)
            local_atari_check(p, b, last_coord, last_color, q, rng)
            if q.moves[0] > 0:
                return mq.mq_pick(q, rng)
        # Local group trying to escape ladder?
        if _random_below(p.params[LADDERRATE], rng):
            mq.mq_clear(q)
            local_ladder_check(p, b, last_coord, q)
            if q.moves[0] > 0:
                return mq.mq_pick(q, rng)
        # Local group can be PUT in atari?
        if _random_below(p.params[ATARIRATE], rng):
            mq.mq_clear(q)
            local_2lib_check(p, b, last_coord, last_color, q, rng)
            if q.moves[0] > 0:
                return mq.mq_pick(q, rng)
        # Local group reduced some of our groups to 3 libs?
        if _random_below(p.params[NLIBRATE], rng):
            mq.mq_clear(q)
            local_nlib_check(p, b, last_coord, last_color, q, rng)
            if q.moves[0] > 0:
                return mq.mq_pick(q, rng)
        # Some other semeai-ish shape checks
        if _random_below(p.params[EYEFIXRATE], rng):
            mq.mq_clear(q)
            eye_fix_check(p, b, last_coord, last_color, to_play, q)
            if q.moves[0] > 0:
                return mq.mq_pick(q, rng)
        # Nakade check
        if _random_below(p.params[NAKADERATE], rng) \
                and pb.immediate_liberty_count(b, last_coord) > 0:
            nakade_move = nakade_check(p, b, last_coord, to_play)
            if not is_pass(nakade_move):
                return nakade_move
        # Check for patterns we know
        if _random_below(p.params[PATTERNRATE], rng):
            mq.mq_clear(q)
            apply_pattern(p, b, last_coord, last_color, _has_mm(p, b),
                          b.ints[pb.LAST_MOVE2_COORD], q, gammas)
            if q.moves[0] > 0:
                return mq.mq_gamma_pick(q, gammas, rng)

    # Global checks
    # Any groups in atari?
    if _random_below(p.params[CAPTURERATE], rng):
        mq.mq_clear(q)
        global_atari_check(p, b, to_play, q, rng)
        if q.moves[0] > 0:
            return mq.mq_pick(q, rng)
    # Joseki moves?
    if _random_below(p.params[JOSEKIRATE], rng):
        mq.mq_clear(q)
        joseki_check(p, b, to_play, q)
        if q.moves[0] > 0:
            return mq.mq_pick(q, rng)
    # Fill board
    if p.params[FILLBOARDTRIES] > 0:
        c = fillboard_check(p, b, rng)
        if not is_pass(c):
            return c
    return PASS


@njit(cache=True, nogil=True, _nrt=False)
def mq_tagged_choose(p, b, to_play, q, rng, pd):
    """Pick a move from the queue, with likelihoods given by the tags.

    ``pd`` is work space (C: an array on the stack).
    """
    # First, merge all entries for a move (C's naive O(N^2) loop, as is).
    i = 0
    while i < q.moves[0]:
        j = i + 1
        while j < q.moves[0]:
            if q.move[i] == q.move[j]:
                q.tag[i] |= q.tag[j]
                q.moves[0] -= 1
                q.tag[j] = q.tag[q.moves[0]]
                q.move[j] = q.move[q.moves[0]]
            j += 1
        i += 1
    # Now, construct a probdist.
    total = np.uint64(0)
    for i in range(q.moves[0]):
        val = 1.0
        assert q.tag[i] != 0
        for j in range(MQ_MAX):
            if q.tag[i] & (1 << j):
                val *= p.mq_prob[j]
        pd[i] = mq.double_to_fixp(val)
        total += pd[i]
    total += mq.double_to_fixp(p.tenuki_prob[0])
    # Finally, pick a move!
    stab = np.uint64(fast.fast_irandom(rng, np.uint32(total & np.uint64(0xFFFFFFFF))))
    for i in range(q.moves[0]):
        if stab < pd[i]:
            return q.move[i]
        stab -= pd[i]
    # Tenuki.
    assert stab < mq.double_to_fixp(p.tenuki_prob[0])
    return PASS


@njit(cache=True, nogil=True, _nrt=False)
def playout_moggy_fullchoose(p, b, to_play, rng, q, gammas):
    """Build a queue of tagged candidates and pick from it."""
    mq.mq_clear(q)
    last_coord = b.ints[pb.LAST_MOVE_COORD]
    last_color = b.ints[pb.LAST_MOVE_COLOR]

    # Ko fight check
    if p.params[KORATE] > 0 and not is_pass(b.ints[pb.LAST_KO_COORD]) \
            and is_pass(b.ints[pb.KO_COORD]) \
            and b.ints[pb.MOVES] - b.ints[pb.LAST_KO_AGE] < p.params[KOAGE]:
        ko = b.ints[pb.LAST_KO_COORD]
        if pb.board_is_valid_play(b, to_play, ko) \
                and not selfatari.is_bad_selfatari(b, to_play, ko):
            mq.mq_add(q, ko, 1 << MQ_KO)

    # Local checks
    if not is_pass(last_coord):
        if p.params[LCAPTURERATE] > 0:
            local_atari_check(p, b, last_coord, last_color, q, rng)
        if p.params[LADDERRATE] > 0:
            local_ladder_check(p, b, last_coord, q)
        if p.params[ATARIRATE] > 0:
            local_2lib_check(p, b, last_coord, last_color, q, rng)
        if p.params[NLIBRATE] > 0:
            local_nlib_check(p, b, last_coord, last_color, q, rng)
        if p.params[EYEFIXRATE] > 0:
            eye_fix_check(p, b, last_coord, last_color, to_play, q)
        if p.params[NAKADERATE] > 0 and pb.immediate_liberty_count(b, last_coord) > 0:
            nakade_move = nakade_check(p, b, last_coord, to_play)
            if not is_pass(nakade_move):
                mq.mq_add(q, nakade_move, 1 << MQ_NAKADE)
        if p.params[PATTERNRATE] > 0:
            apply_pattern(p, b, last_coord, last_color, _has_mm(p, b),
                          b.ints[pb.LAST_MOVE2_COORD], q, gammas)
            # FIXME: Use the gammas.

    # Global checks
    if p.params[CAPTURERATE] > 0:
        global_atari_check(p, b, to_play, q, rng)
    if p.params[JOSEKIRATE] > 0:
        joseki_check(p, b, to_play, q)

    if q.moves[0] > 0:
        return mq_tagged_choose(p, b, to_play, q, rng, gammas)

    # Fill board
    if p.params[FILLBOARDTRIES] > 0:
        c = fillboard_check(p, b, rng)
        if not is_pass(c):
            return c
    return PASS


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def choose(p, b, to_play, rng, q, gammas):
    """``policy->choose``: the next playout move, or pass to play a random one.

    ``q`` and ``gammas`` are work space of the caller (C: on the stack).
    """
    if p.params[KIND] == KIND_LIGHT:
        return PASS
    if p.params[CHOOSE_FULL]:
        return playout_moggy_fullchoose(p, b, to_play, rng, q, gammas)
    return playout_moggy_seqchoose(p, b, to_play, rng, q, gammas)


# ---- assessment of tree priors

@njit(cache=True, nogil=True, _nrt=True)
def playout_moggy_assess_group(p, prior_map, g, games, rng):
    b = prior_map.b
    to_play = prior_map.ints[prior.TO_PLAY]
    q = mq.mq_new()
    if b.gi_libs[g] > p.params[NLIB_COUNT]:
        return

    if b.gi_libs[g] > 2:
        if not p.params[NLIBRATE]:
            return
        if b.b[g] != to_play:
            return  # we do only defense
        n_lib.group_nlib_defense_check(b, g, to_play, q, 0, rng)
        while q.moves[0] > 0:
            q.moves[0] -= 1
            coord = q.move[q.moves[0]]
            assess = games // 2
            prior.add_prior_value(prior_map, coord, 1, assess)
        return

    if b.gi_libs[g] == 2:
        if p.params[LADDERRATE]:
            # Make sure to play the correct liberty in case
            # this is a group that can be caught in a ladder.
            ladderable = False
            for i in range(2):
                chase = b.gi_lib[g, i]
                escape = b.gi_lib[g, 1 - i]
                if ladder.wouldbe_ladder(b, g, escape, chase, b.b[g]):
                    prior.add_prior_value(prior_map, chase, 1, games)
                    ladderable = True
            if ladderable:
                return  # do not suggest the other lib at all
        if not p.params[ATARIRATE]:
            return
        two_lib.group_2lib_check(b, g, to_play, q, 0, p.params[ATARI_MIAISAFE] != 0,
                                 p.params[ATARI_DEF_NO_HOPELESS] != 0, rng)
        while q.moves[0] > 0:
            q.moves[0] -= 1
            coord = q.move[q.moves[0]]
            assess = games // 2
            prior.add_prior_value(prior_map, coord, 1, assess)
        return

    # This group, sir, is in atari!
    ladder_move = one_lib.group_atari_check(p.params[ALWAYSCCAPRATE], b, g, to_play, q, True,
                                            True, 0, rng)
    if ladder_move == one_lib.NO_LADDER:
        ladder_move = PASS  # C: coord_t ladder = pass
    while q.moves[0] > 0:
        q.moves[0] -= 1
        coord = q.move[q.moves[0]]
        # _Never_ play here if this move plays out a caught ladder.
        if coord == ladder_move and not pb.board_playing_ko_threat(b):
            if to_play != b.b[g]:
                continue
            prior.add_prior_value(prior_map, coord, 0, games)
            continue
        if not p.params[CAPTURERATE] and not p.params[LCAPTURERATE]:
            continue
        assess = games * 2
        if p.params[CAP_STONE_DENOM] > 0:
            stones = pb.group_stone_count(b, g, p.params[CAP_STONE_MAX]) \
                - (p.params[CAP_STONE_MIN] - 1)
            assess += (stones if stones > 0 else 0) * games * 100 // p.params[CAP_STONE_DENOM]
        prior.add_prior_value(prior_map, coord, 1, assess)


@njit(cache=True, nogil=True, _nrt=True)
def playout_moggy_assess_one(p, prior_map, coord, games, rng):
    b = prior_map.b
    to_play = prior_map.ints[prior.TO_PLAY]
    # Is this move a self-atari?
    if p.params[SELFATARIRATE]:
        if not pb.board_playing_ko_threat(b) and selfatari.is_bad_selfatari(b, to_play, coord):
            prior.add_prior_value(prior_map, coord, 0, games)
            if not p.params[SELFATARI_OTHER]:
                return
            # If we can play on the other liberty of the endangered group, do!
            coord, _ = selfatari.selfatari_cousin(b, to_play, coord, rng)
            if is_pass(coord):
                return
            prior.add_prior_value(prior_map, coord, 1.0, games)
            return
    # Pattern check
    if p.params[PATTERNRATE]:
        matches, _ = test_pattern3_here(p, b, coord, to_play, True, mq.mq_new())
        if matches:
            prior.add_prior_value(prior_map, coord, 1, games)


@njit(cache=True, nogil=True, _nrt=False)
def playout_moggy_assess(p, prior_map, games, rng):
    """``policy->assess``: prior values of the moves of ``prior_map``."""
    if p.params[KIND] == KIND_LIGHT:
        return  # C: the light policy has no assess
    b = prior_map.b
    # First, go through all endangered groups.
    for g in range(1, pb.board_size2(b)):
        if b.g[g] == g:
            playout_moggy_assess_group(p, prior_map, g, games, rng)
    # Then, assess individual moves.
    if not p.params[PATTERNRATE] and not p.params[SELFATARIRATE]:
        return
    for fi in range(b.ints[pb.FLEN]):
        c = b.f[fi]
        if prior_map.consider[c + 1]:
            playout_moggy_assess_one(p, prior_map, c, games, rng)


# ---- random moves

@njit(cache=True, nogil=True, inline="always", _nrt=False)
def playout_moggy_permit(p, b, coord, color, rng):
    """Whether the random move may be played; returns (permitted, the move to play)."""
    # The idea is simple for now - never allow self-atari moves.
    if fast.fast_random(rng, 100) < p.params[SELFATARIRATE]:
        if selfatari.is_bad_selfatari(b, color, coord):
            if p.params[SELFATARI_OTHER]:
                # Ok, try the other liberty of the atari'd group.
                c, _ = selfatari.selfatari_cousin(b, color, coord, rng)
                if is_pass(c):
                    return False, coord
                return True, c
            return False, coord
    # Check if we don't seem to be filling our eye.
    if fast.fast_random(rng, 100) >= p.params[EYEFILLRATE]:
        return True, coord
    if pb.board_is_eyelike(b, coord, color):
        c = coord
        for i in range(4):
            c += b.dnei[i]
            if b.b[c] != stone_other(color):
                continue
            libs = b.gi_libs[b.g[c]]
            if libs == 1:  # Capture!
                return True, b.gi_lib[b.g[c], 0]
            if libs == 2:  # Try to switch to some 2-lib neighbor.
                for j in range(2):
                    lib = b.gi_lib[b.g[c], j]
                    if pb.board_is_one_point_eye(b, lib, b.b[c]):
                        continue
                    if selfatari.is_bad_selfatari(b, color, lib):
                        continue
                    return True, lib
    return True, coord


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def _try_random_move(p, b, color, f, rng):
    """C board_try_random_move() with the policy's permit; the move played or -3."""
    coord = b.f[f]
    if pb.board_is_one_point_eye(b, coord, color) or not pb.board_is_valid_play(b, color, coord):
        return -3
    if p.params[KIND] == KIND_MOGGY:
        permitted, permitted_coord = playout_moggy_permit(p, b, coord, color, rng)
        if not permitted:
            return -3
    else:
        permitted_coord = coord
    if permitted_coord == coord:
        return coord if pb.board_play_f(b, coord, color, f) >= 0 else -3
    # permit modified the coordinate
    return permitted_coord if pb.board_play(b, permitted_coord, color) >= 0 else -3


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def board_play_random_permit(b, color, p, rng):
    """C board_play_random(b, color, &coord, policy->permit, policy)."""
    if b.ints[pb.FLEN] != 0:
        base = np.int64(fast.fast_random(rng, b.ints[pb.FLEN]))
        for f in range(base, b.ints[pb.FLEN]):
            coord = _try_random_move(p, b, color, f, rng)
            if coord != -3:
                return coord
        for f in range(0, base):
            coord = _try_random_move(p, b, color, f, rng)
            if coord != -3:
                return coord
    pb.board_play(b, PASS, color)
    return PASS

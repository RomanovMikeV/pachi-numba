# pachi-numba: a translation of Pachi (GPL-2.0-only), October 2026; see NOTICE.
"""Translation of Pachi's ``playout.h`` and ``playout.c``: random games.

The playout policy is a ``Moggy`` named tuple (``playout/moggy.py``); the
light policy is a ``Moggy`` of kind ``KIND_LIGHT``. C's UCT engine passes
pre- and post-policy hooks that always return pass (``uct_playout_hook``);
they are left out. A ``struct playout_amafmap`` is a named tuple with the
two lengths in ``ints``.
"""

from collections import namedtuple

import numpy as np
from numba import njit

from .. import board as pb
from .. import mq
from .. import ownermap as om
from ..move import is_pass
from ..stone import S_BLACK, S_WHITE, stone_other
from . import moggy

MAX_GAMELEN = 600

PlayoutSetup = namedtuple("PlayoutSetup", ["gamelen", "mercymin"])
Amafmap = namedtuple("Amafmap", ["game", "is_ko_capture", "ints"])
GAMELEN, GAME_BASELEN = 0, 1


@njit(cache=True, nogil=True, _nrt=True)
def amafmap_new():
    # Only the first ints[GAMELEN] entries are ever read.
    return Amafmap(np.empty(MAX_GAMELEN, np.int32), np.empty(MAX_GAMELEN, np.bool_),
                   np.zeros(2, np.int64))


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def play_random_move(setup, b, color, policy, rng, q, gammas):
    """Play the move of the policy, else a uniformly random one; returns it."""
    coord = moggy.choose(policy, b, color, rng, q, gammas)
    if not is_pass(coord):
        if pb.board_play(b, coord, color) >= 0:
            return coord
        # Pre-picked move is ILLEGAL: defer to uniformly random move choice.
    return moggy.board_play_random_permit(b, color, policy, rng)


@njit(cache=True, nogil=True, _nrt=True)
def play_random_game(setup, b, starting_color, amafmap, use_amafmap, ownermap, use_ownermap,
                     policy, rng):
    """Play the game out; >0: ``starting_color`` wins, <0: loses (double of the score)."""
    # Work space of the policy, once per game. The game itself is in a
    # function without reference counting (see tools/set_nrt_flags.py).
    q = mq.mq_new()
    gammas = np.empty(mq.MQL, np.uint64)  # written with each queued move
    return _play_random_game(setup, b, starting_color, amafmap, use_amafmap, ownermap,
                             use_ownermap, policy, rng, q, gammas)


@njit(cache=True, nogil=True, _nrt=False)
def _play_random_game(setup, b, starting_color, amafmap, use_amafmap, ownermap, use_ownermap,
                      policy, rng, q, gammas):
    gamelen = setup.gamelen - b.ints[pb.MOVES]
    color = starting_color
    passes = 1 if is_pass(b.ints[pb.LAST_MOVE_COORD]) and b.ints[pb.MOVES] > 0 else 0

    while gamelen != 0 and passes < 2:
        gamelen -= 1
        coord = play_random_move(setup, b, color, policy, rng, q, gammas)
        if is_pass(coord):
            passes += 1
        else:
            passes = 0
        if use_amafmap:
            n = amafmap.ints[GAMELEN]
            assert n < MAX_GAMELEN
            amafmap.is_ko_capture[n] = pb.board_playing_ko_threat(b)
            amafmap.game[n] = coord
            amafmap.ints[GAMELEN] = n + 1
        if setup.mercymin and abs(b.captures[S_BLACK] - b.captures[S_WHITE]) > setup.mercymin:
            break
        color = stone_other(color)

    score = pb.board_fast_score(b)
    doubled = np.float32(score * np.float32(2))
    result = np.int64(doubled) if starting_color == S_WHITE else np.int64(-doubled)
    if use_ownermap:
        om.board_ownermap_fill(ownermap, b)
    return result


# pachi-numba: a translation of Pachi (GPL-2.0-only), October 2026; see NOTICE.
"""Translation of Pachi's ``uct/prior.h`` and ``uct/prior.c``: prior values of new tree nodes.

A ``struct prior_map`` is a named tuple; ``ints`` holds ``to_play`` and
``parity``, and the ``prior`` move stats are two arrays (``prior_value``,
``prior_playouts``). As in C (``map.prior = &map_prior[1]``), ``prior_*``
and ``consider`` are indexed by ``coord + 1``, so that pass has a slot.
"""

from collections import namedtuple

import numpy as np
from numba import njit

from .. import board as pb
from .. import stats
from ..move import PASS, is_pass, is_resign
from ..tactics import ladder, util

PriorMap = namedtuple("PriorMap", ["b", "ints", "prior_value", "prior_playouts", "consider",
                                   "distances"])
TO_PLAY, PARITY = 0, 1


@njit(cache=True, nogil=True, _nrt=True)
def prior_map_new(b, to_play, parity):
    size2 = pb.board_size2(b)
    ints = np.zeros(2, np.int64)
    ints[TO_PLAY] = to_play
    ints[PARITY] = parity
    return PriorMap(b, ints, np.zeros(size2 + 1, np.float32), np.zeros(size2 + 1, np.int64),
                    np.zeros(size2 + 1, np.bool_), np.zeros(size2, np.int64))


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def add_prior_value(prior_map, c, value, playouts):
    """Normalizes the argument types (one compiled ``_add_prior_value``)."""
    return _add_prior_value(prior_map, np.int64(c), np.float64(value), np.int64(playouts))


@njit(cache=True, nogil=True, _nrt=False)
def _add_prior_value(prior_map, c, value, playouts):
    """``value`` is the value, ``playouts`` is its weight."""
    value = np.float32(value)
    v = value if prior_map.ints[PARITY] > 0 else np.float32(np.float32(1) - value)
    stats.stats_merge(prior_map.prior_value, prior_map.prior_playouts, c + 1, v, playouts)


# ---- struct uct_prior and uct_prior() (the default priors)

(EQEX, EVEN_EQEX, POLICY_EQEX, B19_EQEX, EYE_EQEX, KO_EQEX, PLUGIN_EQEX, JOSEKI_EQEX,
 PATTERN_EQEX, DCNN_EQEX, CFGDN, PRUNE_LADDERS, NUM_SETTINGS) = range(13)

UctPrior = namedtuple("UctPrior", ["ints", "cfgd_eqex"])


def uct_prior_init(b):
    """C's uct_prior_init(NULL, b, u): the default equivalent experiences.

    Negative values are hundredths of ``eqex``. There is no DCNN, and
    joseki, pattern and plugin priors have nothing to add (no dictionary,
    no pattern files, no plugins), as in the default build.
    """
    ints = np.zeros(NUM_SETTINGS, np.int64)
    for index in (EVEN_EQEX, POLICY_EQEX, B19_EQEX, EYE_EQEX, KO_EQEX, PLUGIN_EQEX):
        ints[index] = -100
    ints[PATTERN_EQEX] = -800
    ints[DCNN_EQEX] = 1300
    ints[JOSEKI_EQEX] = -200
    ints[CFGDN] = -1
    ints[EQEX] = 20 if pb.real_board_size(b) >= 15 else 14  # Even number!
    ints[PRUNE_LADDERS] = 1
    for index in (EVEN_EQEX, POLICY_EQEX, B19_EQEX, EYE_EQEX, KO_EQEX, JOSEKI_EQEX,
                  PATTERN_EQEX, PLUGIN_EQEX, DCNN_EQEX):
        if ints[index] < 0:
            ints[index] = ints[EQEX] * -ints[index] // 100
    ints[DCNN_EQEX] = 0  # not using dcnn
    ints[CFGDN] = 3
    if pb.real_board_size(b) >= 15:
        cfgd_eqex = np.array([0, 55, 50, 15], np.int64)
    else:
        cfgd_eqex = np.array([0, 45, 40, 15], np.int64)
    return UctPrior(ints, cfgd_eqex)


@njit(cache=True, nogil=True, _nrt=False)
def uct_prior_even(p, prior_map):
    """Q_{even}: essential for reading stability of RAVE, it appears."""
    b = prior_map.b
    add_prior_value(prior_map, PASS, 0.5, p.ints[EVEN_EQEX])
    for fi in range(b.ints[pb.FLEN]):
        c = b.f[fi]
        if prior_map.consider[c + 1]:
            add_prior_value(prior_map, c, 0.5, p.ints[EVEN_EQEX])


@njit(cache=True, nogil=True, _nrt=False)
def uct_prior_eye(p, prior_map):
    """Discourage playing into our own eyes."""
    b = prior_map.b
    for fi in range(b.ints[pb.FLEN]):
        c = b.f[fi]
        if not prior_map.consider[c + 1]:
            continue
        if not pb.board_is_one_point_eye(b, c, prior_map.ints[TO_PLAY]):
            continue
        add_prior_value(prior_map, c, 0, p.ints[EYE_EQEX])


@njit(cache=True, nogil=True, _nrt=False)
def uct_prior_ko(p, prior_map):
    """Favor fighting ko, if we took it le 10 moves ago."""
    b = prior_map.b
    ko = b.ints[pb.LAST_KO_COORD]
    if is_pass(ko) or b.ints[pb.MOVES] - b.ints[pb.LAST_KO_AGE] > 10 \
            or not prior_map.consider[ko + 1]:
        return
    add_prior_value(prior_map, ko, 1, p.ints[KO_EQEX])


@njit(cache=True, nogil=True, _nrt=False)
def uct_prior_b19(p, prior_map):
    """Q_{b19}: priors for certain edge distances (first line 0, third line 1)."""
    b = prior_map.b
    for fi in range(b.ints[pb.FLEN]):
        c = b.f[fi]
        if not prior_map.consider[c + 1]:
            continue
        d = util.coord_edge_distance(c, b)
        if d != 0 and d != 2:
            continue
        # The bonus applies only with no stones in immediate vincinity.
        if util.board_stone_radar(b, c, 2):
            continue
        add_prior_value(prior_map, c, 1 if d == 2 else 0, p.ints[B19_EQEX])


@njit(cache=True, nogil=True, _nrt=False)
def uct_prior_cfgd(p, prior_map):
    """Q_{common_fate_graph_distance}: bonus to moves local to the last move."""
    b = prior_map.b
    if is_pass(b.ints[pb.LAST_MOVE_COORD]) or is_resign(b.ints[pb.LAST_MOVE_COORD]):
        return
    for fi in range(b.ints[pb.FLEN]):
        c = b.f[fi]
        if not prior_map.consider[c + 1]:
            continue
        if prior_map.distances[c] > p.ints[CFGDN]:
            continue
        assert prior_map.distances[c] != 0
        add_prior_value(prior_map, c, 1, p.cfgd_eqex[prior_map.distances[c]])


@njit(cache=True, nogil=True, _nrt=False)
def uct_prior(p, playout_policy, prior_map, rng):
    """Fill the prior map of a node to expand."""
    b = prior_map.b
    if p.ints[PRUNE_LADDERS] and not pb.board_playing_ko_threat(b):
        for fi in range(b.ints[pb.FLEN]):
            c = b.f[fi]
            if not prior_map.consider[c + 1]:
                continue
            atari_neighbor = pb.board_get_atari_neighbor(b, c, prior_map.ints[TO_PLAY])
            if atari_neighbor and ladder.is_ladder(b, c, atari_neighbor, True):
                prior_map.consider[c + 1] = False
    if p.ints[EVEN_EQEX]:
        uct_prior_even(p, prior_map)
    if p.ints[EYE_EQEX]:
        uct_prior_eye(p, prior_map)
    if p.ints[KO_EQEX]:
        uct_prior_ko(p, prior_map)
    if p.ints[B19_EQEX]:
        uct_prior_b19(p, prior_map)
    if p.ints[POLICY_EQEX]:
        # Q_{playout-policy}
        moggy.playout_moggy_assess(playout_policy, prior_map, p.ints[POLICY_EQEX], rng)
    if p.ints[CFGDN] >= 0 and p.cfgd_eqex.shape[0] > 0:
        uct_prior_cfgd(p, prior_map)
    # joseki (no dictionary), pattern (no pattern files) and plugin (no plugins)
    # priors add nothing.


# At the end: moggy imports this module (for add_prior_value).
from ..playout import moggy  # noqa: E402

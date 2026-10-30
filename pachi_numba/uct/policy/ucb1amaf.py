# pachi-numba: a translation of Pachi (GPL-2.0-only), October 2026; see NOTICE.
"""Translation of Pachi's ``uct/policy/ucb1amaf.c`` and ``uct/policy/generic.c``.

The UCB1 policy with an extra AMAF heuristics, with its default settings.
Settings are in the ``Ucb1Amaf`` named tuple. Local-tree RAVE
(``ltree_rave``) needs local trees, which are not translated; distributed
virtual wins (``virtual_win``) need slaves, which are not translated either.

The arithmetic follows C's float (``floating_t``) and int rules: float32
operations, truncating int conversions, and division by zero giving inf or
NaN (``error_model="numpy"``).
"""

from collections import namedtuple

import numpy as np
from numba import njit

from ... import random as fast
from ...move import is_pass
from ...stone import S_BLACK, S_WHITE
from ...tactics import util
from .. import tree as tr
from ... import stats

INT_MAX = 2147483647
FLT_EPSILON = np.float32(1.1920929e-07)
DESCENT_MAX = 512  # C: dbest[BOARD_MAX_MOVES + 1]

# Ucb1Amaf.ints / .floats fields
(EQUIV_RAVE, SYLVAIN_RAVE, DISTANCE_RAVE, THREAT_RAVE, CRIT_MIN_PLAYOUTS, CRIT_NEGATIVE,
 CRIT_NEGFLIP, CRIT_AMAF, CRIT_LVALUE, VLOSS_SQRT, VIRTUAL_LOSS, THREADS, AMAF_PRIOR,
 NUM_INTS) = range(14)
EXPLORE_P, FPU, CRIT_RAVE, CRIT_PLTHRES_COEF, NUM_FLOATS = range(5)

Ucb1Amaf = namedtuple("Ucb1Amaf", ["ints", "floats"])


def policy_ucb1amaf_init(b, virtual_loss=1, threads=1, amaf_prior=False):
    """The default settings of C's policy_ucb1amaf_init() (no options)."""
    ints = np.zeros(NUM_INTS, np.int64)
    floats = np.zeros(NUM_FLOATS, np.float32)
    floats[EXPLORE_P] = 0
    ints[EQUIV_RAVE] = 4000 if b.ints[0] - 2 >= 15 else 3000  # board_large(board)
    floats[FPU] = np.inf
    ints[SYLVAIN_RAVE] = 1
    ints[DISTANCE_RAVE] = 3
    ints[THREAT_RAVE] = 0
    floats[CRIT_RAVE] = 1.1
    ints[CRIT_MIN_PLAYOUTS] = 2000
    ints[CRIT_NEGATIVE] = 1
    ints[CRIT_AMAF] = 0
    ints[VLOSS_SQRT] = 1
    ints[VIRTUAL_LOSS] = virtual_loss
    ints[THREADS] = threads
    ints[AMAF_PRIOR] = amaf_prior
    return Ucb1Amaf(ints, floats)


@njit(cache=True, nogil=True, error_model="numpy", _nrt=False)
def _merge(value, playouts, src_value, src_playouts):
    """stats_merge() on local (value, playouts) pairs."""
    if src_playouts:
        playouts += src_playouts
        value = np.float32(value + np.float32(np.float32(np.float32(src_value - value)
                                                         * np.float32(src_playouts))
                                              / np.float32(playouts)))
    return value, playouts


@njit(cache=True, nogil=True, error_model="numpy", _nrt=False)
def ucb1rave_evaluate(p, t, node, parity):
    """Returns (value to maximize, descent value, descent playouts)."""
    n_value = t.u_value[node]
    n_playouts = np.int64(t.u_playouts[node])
    r_value = t.amaf_value[node]
    r_playouts = np.int64(t.amaf_playouts[node])
    if p.ints[AMAF_PRIOR]:
        r_value, r_playouts = _merge(r_value, r_playouts, t.prior_value[node],
                                     t.prior_playouts[node])
    else:
        n_value, n_playouts = _merge(n_value, n_playouts, t.prior_value[node],
                                     t.prior_playouts[node])

    if p.ints[VIRTUAL_LOSS]:
        # Add virtual loss to discourage other threads from visiting this node.
        threads = p.ints[THREADS]
        vloss_coeff = np.sqrt(np.float64(threads)) / threads if p.ints[VLOSS_SQRT] else 1.0
        c_value = np.float32(0.0 if parity > 0 else 1.0)
        c_playouts = np.int64(t.descents[node] * vloss_coeff)
        n_value, n_playouts = _merge(n_value, n_playouts, c_value, c_playouts)

    # Criticality heuristics.
    crit_rave = p.floats[CRIT_RAVE]
    if p.floats[CRIT_PLTHRES_COEF] > 0:
        is_critical_enough = np.float32(t.u_playouts[node]) \
            > np.float32(np.float32(t.u_playouts[t.ints[tr.ROOT]]) * p.floats[CRIT_PLTHRES_COEF])
    else:
        is_critical_enough = t.u_playouts[node] > p.ints[CRIT_MIN_PLAYOUTS]
    if crit_rave > 0 and is_critical_enough:
        crit = tr.tree_node_criticality(t, node)
        if p.ints[CRIT_NEGATIVE] or crit > 0:
            val = np.float32(1.0)
            if p.ints[CRIT_NEGFLIP] and crit < 0:
                val = np.float32(0)
                crit = -crit
            c_value = tr.tree_node_get_value(t, parity, val)
            c_playouts = np.int64(np.float32(np.float32(crit * np.float32(r_playouts)) * crit_rave))
            r_value, r_playouts = _merge(r_value, r_playouts, c_value, c_playouts)

    value = np.float32(0)
    if n_playouts:
        if r_playouts:
            # At the beginning, beta is at 1 and RAVE is used.
            # At b->equiv_rate, beta is at 1/3 and gets steeper on.
            if p.ints[SYLVAIN_RAVE]:
                nr = np.float32(np.float32(np.float32(n_playouts) * np.float32(r_playouts))
                                / np.float32(p.ints[EQUIV_RAVE]))
                beta = np.float32(np.float32(r_playouts)
                                  / np.float32(np.float32(r_playouts + n_playouts) + nr))
            else:
                parent_playouts = t.u_playouts[t.parent[node]]
                beta = np.float32(np.sqrt(np.float64(p.ints[EQUIV_RAVE])
                                          / (3 * parent_playouts + p.ints[EQUIV_RAVE])))
            value = np.float32(np.float32(beta * r_value)
                               + np.float32(np.float32(np.float32(1) - beta) * n_value))
        else:
            value = n_value
    elif r_playouts:
        value = r_value
    return tr.tree_node_get_value(t, parity, value), value, r_playouts + n_playouts


@njit(cache=True, nogil=True, error_model="numpy", _nrt=False)
def _try_children(p, t, node, parity, allow_pass, rng, use_fpu, best_nodes, best_values,
                  best_playouts):
    """C's uctd_try_node_children / uctd_set_best_child / uctd_get_best_child.

    Returns (chosen node, its descent value, its descent playouts). The
    ``best_*`` arrays are work space (C: on the stack).
    """
    first = t.children[node]
    best_nodes[0] = first
    best_values[0] = 0
    best_playouts[0] = 0
    dbests = 1
    best_urgency = np.float32(-9999)
    ni = first
    while ni != tr.NONE:
        if (not allow_pass and is_pass(t.coord[ni])) or (t.hints[ni] & tr.TREE_HINT_INVALID):
            ni = t.sibling[ni]
            continue
        urgency, d_value, d_playouts = ucb1rave_evaluate(p, t, ni, parity)
        if use_fpu:
            if t.u_playouts[ni] > 0 and p.floats[EXPLORE_P] > 0:
                raise NotImplementedError("explore_p > 0 is not translated")
            if t.u_playouts[ni] + t.amaf_playouts[ni] + t.prior_playouts[ni] == 0:
                urgency = p.floats[FPU]
        if np.float32(urgency - best_urgency) > FLT_EPSILON:  # urgency > best_urgency
            best_urgency = urgency
            dbests = 0
        if np.float32(urgency - best_urgency) > -FLT_EPSILON:  # urgency >= best_urgency
            # We want to always choose something else than a pass in case of a tie.
            if dbests == 1 and is_pass(t.coord[best_nodes[0]]):
                dbests -= 1
            best_nodes[dbests] = ni
            best_values[dbests] = d_value
            best_playouts[dbests] = d_playouts
            dbests += 1
        ni = t.sibling[ni]
    pick = fast.fast_random(rng, dbests)
    return best_nodes[pick], best_values[pick], best_playouts[pick]


@njit(cache=True, nogil=True, _nrt=True)
def descent_buffers():
    """Work space of ucb1rave_descend(), for all children of a node."""
    return (np.empty(DESCENT_MAX, np.int64), np.empty(DESCENT_MAX, np.float32),
            np.empty(DESCENT_MAX, np.int64))


@njit(cache=True, nogil=True, error_model="numpy", _nrt=False)
def ucb1rave_descend(p, t, node, parity, allow_pass, rng, buffers):
    """The child to descend to (with explore_p = 0, as by default)."""
    return _try_children(p, t, node, parity, allow_pass, rng, True, buffers[0], buffers[1],
                         buffers[2])


@njit(cache=True, nogil=True, error_model="numpy", _nrt=True)
def uctp_generic_winner(p, t, node, rng):
    """The child with the best value (not the best explored)."""
    parity = tr.tree_node_parity(t, node)
    buffers = descent_buffers()
    chosen, _, _ = _try_children(p, t, node, parity, False, rng, False, buffers[0], buffers[1],
                                 buffers[2])
    return chosen


@njit(cache=True, nogil=True, _nrt=False)
def ko_length(ko_capture_map, start, map_length):
    """The length of the current ko (moves up to the last ko capture), 0 if none."""
    if map_length <= 0 or not ko_capture_map[start]:
        return 0
    length = 1
    while length + 2 < map_length and ko_capture_map[start + length + 2]:
        length += 3
    return length


@njit(cache=True, nogil=True, error_model="numpy", _nrt=True)
def ucb1amaf_update(p, t, node, node_color, player_color, amaf, final_board, result):
    """Record the result in ``node`` and its ancestors, and AMAF in their children."""
    result = np.float32(result)
    winner_color = S_BLACK if result > np.float32(0.5) else S_WHITE
    size2 = final_board.b.shape[0]
    # first_move[coord + 1]: the index in amaf.game of the first move at coord.
    first_move = np.full(size2 + 1, INT_MAX, np.int64)
    gamelen = amaf.ints[0]
    game_baselen = amaf.ints[1]
    assert gamelen > 0
    move = gamelen - 1
    while move >= game_baselen:
        first_move[amaf.game[move] + 1] = move
        move -= 1

    crit_lvalue = p.ints[CRIT_LVALUE] != 0
    while node != tr.NONE:
        if not p.ints[CRIT_AMAF] and not is_pass(t.coord[node]):
            stats.stats_add_result(t.winner_owner_value, t.winner_owner_playouts, node,
                                   util.board_local_value(crit_lvalue, final_board, t.coord[node],
                                                          winner_color), 1)
            stats.stats_add_result(t.black_owner_value, t.black_owner_playouts, node,
                                   util.board_local_value(crit_lvalue, final_board, t.coord[node],
                                                          S_BLACK), 1)
        stats.stats_add_result(t.u_value, t.u_playouts, node, result, 1)

        if p.ints[THREAT_RAVE] <= 0:
            max_threat_dist = ko_length(amaf.is_ko_capture, move + 1, gamelen - (move + 1))
        else:
            max_threat_dist = -1
        ni = t.children[node]
        while ni != tr.NONE:
            coord = t.coord[ni]
            if is_pass(coord):
                ni = t.sibling[ni]
                continue
            # Use the child move only if it was first played by the same color.
            first = first_move[coord + 1]
            if first == INT_MAX:
                ni = t.sibling[ni]
                continue
            assert first > move and first < gamelen
            distance = first - (move + 1)
            if distance & 1:
                ni = t.sibling[ni]
                continue
            weight = 1
            res = result
            # Don't give amaf bonus to a ko threat before taking the ko.
            if distance <= max_threat_dist and distance % 6 == 4:
                weight = -p.ints[THREAT_RAVE]
                res = np.float32(np.float32(1.0) - res)
            elif p.ints[DISTANCE_RAVE] != 0:
                # Give more weight to moves played earlier
                weight += util.c_div(p.ints[DISTANCE_RAVE] * (gamelen - first), gamelen - move)
            stats.stats_add_result(t.amaf_value, t.amaf_playouts, ni, res, weight)
            if p.ints[CRIT_AMAF]:
                stats.stats_add_result(t.winner_owner_value, t.winner_owner_playouts, ni,
                                       util.board_local_value(crit_lvalue, final_board, coord,
                                                              winner_color), 1)
                stats.stats_add_result(t.black_owner_value, t.black_owner_playouts, ni,
                                       util.board_local_value(crit_lvalue, final_board, coord,
                                                              S_BLACK), 1)
            ni = t.sibling[ni]

        if t.parent[node] != tr.NONE:
            assert move >= 0 and amaf.game[move] == t.coord[node] \
                and first_move[t.coord[node] + 1] > move
            first_move[t.coord[node] + 1] = move
            move -= 1
        node = t.parent[node]

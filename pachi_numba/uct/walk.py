# pachi-numba: a translation of Pachi (GPL-2.0-only), October 2026; see NOTICE.
"""Translation of Pachi's ``uct/walk.c``: one simulation through the tree.

The engine settings that the walk reads are in ``params`` (indices below)
with their default values; local trees, value scaling (``val_scale``),
a random tree policy and dynamic komi per simulation are off by default and
not translated.
"""

import numpy as np
from numba import njit

from .. import board as pb
from .. import ownermap as om
from .. import playout
from ..move import PASS, is_pass, is_resign
from ..stone import S_NONE, S_WHITE, stone_other
from ..tactics import util
from . import prior
from . import tree as tr
from .policy import ucb1amaf

# Engine settings read by the walk: indices into params.
(GAMELEN, MERCYMIN, EXPAND_P, VIRTUAL_LOSS, PLAYOUT_AMAF, PLAYOUT_AMAF_CUTOFF,
 MAX_TREE_SIZE, NUM_PARAMS) = range(8)

MC_GAMELEN = playout.MAX_GAMELEN


def walk_params_default():
    params = np.zeros(NUM_PARAMS, np.int64)
    params[GAMELEN] = MC_GAMELEN
    params[MERCYMIN] = 0
    params[EXPAND_P] = 8
    params[VIRTUAL_LOSS] = 1
    params[PLAYOUT_AMAF] = 1
    params[PLAYOUT_AMAF_CUTOFF] = 0
    return params


@njit(cache=True, nogil=True, _nrt=False)
def min_free_nodes(b):
    """Nodes one simulation may add: the root and one leaf expansion."""
    return 2 * (pb.board_size2(b) + 1)


@njit(cache=True, nogil=True, error_model="numpy", _nrt=True)
def tree_expand_node(t, node, b, color, parity, prior_settings, playout_policy, rng):
    """C's tree_expand_node(): priors of the moves of ``color``, then the children."""
    size2 = pb.board_size2(b)
    # Get a Common Fate Graph distance map from parent node.
    distances = np.empty(size2, np.int64)
    last_move = b.ints[pb.LAST_MOVE_COORD]
    if not is_pass(last_move) and not is_resign(last_move) and not is_pass(t.coord[node]):
        util.cfg_distances(b, t.coord[node], distances, tr.TREE_NODE_D_MAX)
    else:
        # C runs cfg_distances() from the virtual pass root of a fresh tree
        # too, reading outside the board (undefined behavior); all real
        # points end up at the maximum distance, as here.
        distances[:] = tr.TREE_NODE_D_MAX + 1

    # Get a map of prior values to initialize the new nodes with.
    prior_map = prior.prior_map_new(b, color, tr.tree_parity(t, parity))
    prior_map.distances[:] = distances
    prior_map.consider[PASS + 1] = True
    child_count = 1  # for pass
    for fi in range(b.ints[pb.FLEN]):
        c = b.f[fi]
        assert b.b[c] == S_NONE
        if not pb.board_is_valid_play_no_suicide(b, color, c):
            continue
        prior_map.consider[c + 1] = True
        child_count += 1
    prior.uct_prior(prior_settings, playout_policy, prior_map, rng)

    tr.tree_expand_node_children(t, node, b, prior_map.consider, prior_map.prior_value,
                                 prior_map.prior_playouts, distances, child_count)


@njit(cache=True, nogil=True, error_model="numpy", _nrt=False)
def _undo_virtual_loss(t, n, virtual_loss):
    while t.parent[n] != tr.NONE:
        t.descents[n] -= virtual_loss
        n = t.parent[n]


@njit(cache=True, nogil=True, error_model="numpy", _nrt=True)
def uct_playout(params, policy, prior_settings, playout_policy, ownermap, t, b, player_color,
                rng):
    """One simulation: descend the tree, expand, play a random game, update.

    Returns the result from black's perspective (double of the score).
    """
    # The copies and work space; the simulation itself is in a function
    # without reference counting (see tools/set_nrt_flags.py).
    b2 = pb.board_duplicate(b)
    amaf = playout.amafmap_new()
    buffers = ucb1amaf.descent_buffers()
    return _uct_playout(params, policy, prior_settings, playout_policy, ownermap, t, b,
                        player_color, rng, b2, amaf, buffers)


@njit(cache=True, nogil=True, error_model="numpy", _nrt=False)
def _uct_playout(params, policy, prior_settings, playout_policy, ownermap, t, b, player_color,
                 rng, b2, amaf, buffers):
    virtual_loss = params[VIRTUAL_LOSS]

    # Walk the tree until we find a leaf, then expand it and do a random playout.
    n = t.ints[tr.ROOT]
    node_color = stone_other(player_color)
    assert node_color == t.ints[tr.ROOT_COLOR]

    # Make sure the root node is expanded.
    if tr.tree_leaf_node(t, n) and not t.is_expanded[n]:
        t.is_expanded[n] = True
        tree_expand_node(t, n, b2, player_color, 1, prior_settings, playout_policy, rng)

    pass_limit = (pb.board_size(b2) - 2) * (pb.board_size(b2) - 2) // 2
    passes = 1 if is_pass(b.ints[pb.LAST_MOVE_COORD]) and b.ints[pb.MOVES] > 0 else 0

    while not tr.tree_leaf_node(t, n) and passes < 2:
        # Parity is chosen already according to the child color.
        node_color = stone_other(node_color)
        parity = 1 if node_color == player_color else -1
        n, _, _ = ucb1amaf.ucb1rave_descend(policy, t, n, parity,
                                            b2.ints[pb.MOVES] > pass_limit, rng, buffers)
        if virtual_loss:
            t.descents[n] += virtual_loss

        coord = t.coord[n]
        res = pb.board_play(b2, coord, node_color)
        if res < 0 or (not is_pass(coord) and not b2.g[coord]) \
                or b2.ints[pb.SUPERKO_VIOLATION]:
            t.hints[n] |= tr.TREE_HINT_INVALID
            if virtual_loss:
                _undo_virtual_loss(t, n, virtual_loss)
            return 0

        assert coord >= -1
        assert amaf.ints[playout.GAMELEN] < playout.MAX_GAMELEN
        amaf.is_ko_capture[amaf.ints[playout.GAMELEN]] = pb.board_playing_ko_threat(b2)
        amaf.game[amaf.ints[playout.GAMELEN]] = coord
        amaf.ints[playout.GAMELEN] += 1
        if is_pass(coord):
            passes += 1
        else:
            passes = 0

        next_color = stone_other(node_color)
        if tr.tree_leaf_node(t, n) and t.u_playouts[n] - virtual_loss >= params[EXPAND_P] \
                and t.ints[tr.NODES_SIZE] < params[MAX_TREE_SIZE] and not t.is_expanded[n]:
            t.is_expanded[n] = True
            tree_expand_node(t, n, b2, next_color, -parity, prior_settings, playout_policy, rng)

    amaf.ints[playout.GAME_BASELEN] = amaf.ints[playout.GAMELEN]

    # The result is from black's perspective here (positive: black wins).
    if passes >= 2:
        # XXX: No dead groups support.
        score = pb.board_official_score(b2, amaf.game, 0)  # no dead groups
        result = np.int64(-np.float32(score * np.float32(2)))
        om.board_ownermap_fill(ownermap, b2)
    else:
        # uct_leaf_node()
        next_color = stone_other(node_color)
        setup = playout.PlayoutSetup(params[GAMELEN], params[MERCYMIN])
        result = playout.play_random_game(setup, b2, next_color, amaf, params[PLAYOUT_AMAF] != 0,
                                          ownermap, True, playout_policy, rng)
        if next_color == S_WHITE:
            result = -result  # We need the result from black's perspective.

    if params[PLAYOUT_AMAF_CUTOFF]:
        base = amaf.ints[playout.GAME_BASELEN]
        amaf.ints[playout.GAMELEN] = base + (amaf.ints[playout.GAMELEN] - base) \
            * params[PLAYOUT_AMAF_CUTOFF] // 100

    # Record the result. (scale_value() with val_scale = 0.)
    if result > 0:
        rval = np.float32(1.0)
    elif result < 0:
        rval = np.float32(0.0)
    else:
        rval = np.float32(0.5)
    ucb1amaf.ucb1amaf_update(policy, t, n, node_color, player_color, amaf, b2, rval)
    # stats_add_result(&t->avg_score, result / 2, 1)
    avg_playouts = t.ints[tr.AVG_SCORE_PLAYOUTS] + 1
    avg_value = t.floats[tr.AVG_SCORE_VALUE]
    t.floats[tr.AVG_SCORE_VALUE] = np.float32(avg_value + np.float32(
        np.float32(np.float32(util.c_div(result, 2)) - avg_value) / np.float32(avg_playouts)))
    t.ints[tr.AVG_SCORE_PLAYOUTS] = avg_playouts

    if virtual_loss:
        _undo_virtual_loss(t, n, virtual_loss)
    return result


@njit(cache=True, nogil=True, error_model="numpy", _nrt=False)
def uct_playouts(params, policy, prior_settings, playout_policy, ownermap, t, b, color, games,
                 rng):
    """Simulations while the root has at most ``games`` playouts.

    Stops early, returning the number done so far, when the tree arrays may
    not hold another simulation; the caller grows them and calls again.
    """
    i = 0
    need = min_free_nodes(b)
    while t.u_playouts[t.ints[tr.ROOT]] <= games:
        if tr.free_slots(t) < need:
            break
        uct_playout(params, policy, prior_settings, playout_policy, ownermap, t, b, color, rng)
        i += 1
    return i

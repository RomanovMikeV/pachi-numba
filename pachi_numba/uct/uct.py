# pachi-numba: a translation of Pachi (GPL-2.0-only), October 2026; see NOTICE.
"""Translation of Pachi's ``uct/uct.c`` and ``uct/search.c``: the UCT engine.

The engine runs with its default settings and one search thread, as the
C engine created with ``threads=1,pondering=0``, and a fixed number of
simulations per move (``=N`` time settings). Random numbers follow C:
the calling ("main") thread has its own state; each search seeds the
thread manager from it and the worker from the manager.

C's main thread polls the search every 100 ms and, when the most-visited
move is a pass, may run extra simulations on its own random state while
the worker runs. That depends on timing; here the search is polled once,
when the worker has finished, which is what C does when a search takes
less than 100 ms. C also stops large searches (5000 or more simulations
of the best move) early when it is sure to win, at a poll; this is not
done here.

Only ``dynkomi=none`` is translated, which is the default on boards up to
9x9; larger boards raise ``NotImplementedError``.
"""

from collections import namedtuple

import numpy as np
from numba import njit

from .. import board as pb
from .. import mq
from .. import ownermap as om
from .. import random as fast
from ..move import PASS, RESIGN, is_pass, is_resign
from ..playout import moggy
from ..stone import S_BLACK, S_OFFBOARD, S_WHITE, stone_other
from ..tactics import util
from . import prior
from . import tree as tr
from . import walk
from .policy import ucb1amaf

# How big proportion of ownermap counts must be of one color to consider the point sure.
GJ_THRES = 0.8
# How many games to consider at minimum before judging groups.
GJ_MINGAMES = 500

# Default tree sizes (bytes), as uct_state_init() computes them for fast_alloc.
_MAX_TREE_SIZE_OPTION = 1408 * 1048576
PRUNING_THRESHOLD = _MAX_TREE_SIZE_OPTION // 10
MAX_PRUNED_SIZE = _MAX_TREE_SIZE_OPTION // 5
MAX_TREE_SIZE = _MAX_TREE_SIZE_OPTION - MAX_PRUNED_SIZE

RESIGN_THRESHOLD = 0.2
INITIAL_TREE_CAPACITY = 4096

# Uct.flags
PASS_ALL_ALIVE, ALLOW_LOSING_PASS, TERRITORY_SCORING, DYNKOMI_MASK, NUM_FLAGS = range(5)

Uct = namedtuple("Uct", ["params", "policy", "prior_settings", "playout_policy", "ownermap",
                         "rng", "flags"])


# ---- compiled parts

@njit(cache=True, nogil=True, error_model="numpy", _nrt=True)
def dead_group_list(u, b, q):
    """The groups that the ownermap of the simulations judges dead."""
    gs = np.zeros(pb.board_size2(b), np.int64)
    om.board_ownermap_judge_groups(b, u.ownermap, np.float32(GJ_THRES), gs)
    om.groups_of_status(b, gs, om.GS_DEAD, q)


@njit(cache=True, nogil=True, error_model="numpy", _nrt=True)
def uct_pass_is_safe(u, t, b, color, pass_all_alive):
    # Make sure enough playouts are simulated to get a reasonable dead group list.
    while u.ownermap.playouts[0] < GJ_MINGAMES:
        walk.uct_playout(u.params, u.policy, u.prior_settings, u.playout_policy, u.ownermap, t, b,
                         color, u.rng)
    q = mq.mq_new()
    dead_group_list(u, b, q)
    if pass_all_alive:
        for i in range(q.moves[0]):
            if b.b[q.move[i]] == stone_other(color):
                return False  # We need to remove opponent dead groups first.
        q.moves[0] = 0  # our dead stones are alive when pass_all_alive is true
    if u.flags[ALLOW_LOSING_PASS]:
        for c in range(pb.board_size2(b)):
            if b.b[c] == S_OFFBOARD:
                continue
            if om.board_ownermap_judge_point(u.ownermap, c, np.float32(GJ_THRES)) == om.PJ_UNKNOWN:
                return False  # Unclear point, clarify first.
        return True
    return util.pass_is_safe(b, color, q.move, q.moves[0])


@njit(cache=True, nogil=True, error_model="numpy", _nrt=False)
def uctp_generic_choose(u, t, node, b, color, exclude):
    """The best-explored child (pass only if it is safe); NONE if no children."""
    nbest = t.children[node]
    if nbest == tr.NONE:
        return tr.NONE
    nbest2 = t.sibling[nbest]
    ni = nbest2
    while ni != tr.NONE:
        # we compare playouts and choose the best-explored child
        if t.coord[ni] == exclude or t.hints[ni] & tr.TREE_HINT_INVALID:
            ni = t.sibling[ni]
            continue
        if t.u_playouts[ni] > t.u_playouts[nbest]:
            nbest2 = nbest
            nbest = ni
        elif t.u_playouts[ni] > t.u_playouts[nbest2]:
            nbest2 = ni
        ni = t.sibling[ni]
    # Play pass only if we can afford scoring.
    if is_pass(t.coord[nbest]) \
            and not uct_pass_is_safe(u, t, b, color, u.flags[PASS_ALL_ALIVE] != 0):
        return nbest2
    return nbest


@njit(cache=True, nogil=True, error_model="numpy", _nrt=False)
def uct_search_poll(u, t, b, color):
    """The poll of uct_search_check_stop() after the worker has finished: the
    two choose() calls, which may run simulations (pass safety)."""
    i = t.u_playouts[t.ints[tr.ROOT]]
    if i < GJ_MINGAMES:
        return
    best = uctp_generic_choose(u, t, t.ints[tr.ROOT], b, color, RESIGN)
    if best != tr.NONE:
        uctp_generic_choose(u, t, t.ints[tr.ROOT], b, color, t.coord[best])


@njit(cache=True, nogil=True, error_model="numpy", _nrt=False)
def uct_search_result(u, t, b, color, pass_all_alive, resign_threshold):
    """The move to play: (coordinate, best node or NONE for pass and resignation)."""
    root = t.ints[tr.ROOT]
    best = uctp_generic_choose(u, t, root, b, color, RESIGN)
    if best == tr.NONE:
        return PASS, tr.NONE

    # Do not resign if the evaluation of best is unreliable, or with dynkomi.
    extra_komi = t.floats[tr.EXTRA_KOMI]
    komi_by_color = extra_komi if color == S_WHITE else -extra_komi
    if tr.tree_node_get_value(t, 1, t.u_value[best]) < np.float32(resign_threshold) \
            and not is_pass(t.coord[best]) \
            and (t.u_playouts[best] > GJ_MINGAMES or t.u_playouts[root] > GJ_MINGAMES * 2) \
            and (not t.ints[tr.USE_EXTRA_KOMI] or komi_by_color < 0.5) \
            and not t.ints[tr.UNTRUSTWORTHY_TREE]:
        return RESIGN, tr.NONE

    # If the opponent just passed and we win counting, always pass as well.
    if b.ints[pb.MOVES] > 1 and is_pass(b.ints[pb.LAST_MOVE_COORD]) \
            and b.ints[pb.RULES] != pb.RULES_STONES_ONLY:
        if uct_pass_is_safe(u, t, b, color, pass_all_alive):
            best = t.children[root]  # pass is the first child
            assert is_pass(t.coord[best])
            return PASS, best
    return t.coord[best], best


@njit(cache=True, nogil=True, _nrt=True)
def search_seeds(main_rng):
    """C: mctx.seed from the main thread, then the worker's from the manager's."""
    manager_seed = fast.fast_random(main_rng, 65536)
    manager_rng = fast.make_state_from(manager_seed)
    worker_seed = np.uint64(fast.fast_random(manager_rng, 65536)) + np.uint64(0)  # + thread id
    return fast.make_state_from(worker_seed)


# ---- the engine (Python side: tree ownership and growth)

class UctEngine:
    """The UCT engine of one game, for the board ``b`` it was created with.

    The methods follow the C engine interface: ``notify_play`` (before the
    move is played on the board), ``genmove`` (the caller then plays the
    move) and ``dead_group_list``.
    """

    def __init__(self, b, seed=None, resign_threshold=RESIGN_THRESHOLD):
        if pb.real_board_size(b) > 9:
            raise NotImplementedError("dynkomi linear (boards over 9x9) is not translated")
        params = walk.walk_params_default()
        params[walk.MAX_TREE_SIZE] = MAX_TREE_SIZE
        flags = np.zeros(NUM_FLAGS, np.int64)
        flags[DYNKOMI_MASK] = S_BLACK | S_WHITE
        rng = fast.make_state() if seed is None else fast.make_state(seed)
        self.u = Uct(params, ucb1amaf.policy_ucb1amaf_init(b), prior.uct_prior_init(b),
                     moggy.playout_moggy_init(None, b),
                     om.ownermap_new(int(pb.board_size2(b))), rng, flags)
        # C option resign_threshold: resign below this win chance of the best move.
        self.resign_threshold = float(resign_threshold)
        self.tree = None
        self.initial_extra_komi = np.float32(0)

    @property
    def rng(self):
        return self.u.rng

    # -- state

    def _setup_state(self, b, color):
        self.tree = tr.tree_init(b, color, MAX_TREE_SIZE, MAX_PRUNED_SIZE, PRUNING_THRESHOLD,
                                 np.float32(1.0), INITIAL_TREE_CAPACITY)
        if self.initial_extra_komi:
            self.tree.floats[tr.EXTRA_KOMI] = self.initial_extra_komi
        # force_seed is 0 and there is no opening book (tree_load).

    def _reset_state(self):
        self.tree = None

    def prepare_move(self, b, color):
        if self.tree is not None:
            if color != stone_other(self.tree.ints[tr.ROOT_COLOR]):
                raise RuntimeError(f"Non-alternating play detected {color} "
                                   f"{self.tree.ints[tr.ROOT_COLOR]}")
        else:
            self._setup_state(b, color)
        self.u.ownermap.playouts[0] = 0
        self.u.ownermap.map[:] = 0

    def _grow(self, b, simulations):
        """Make room for ``simulations`` more simulations."""
        need = int(walk.min_free_nodes(b)) * max(simulations, 1)
        self.tree = tr.tree_grow(self.tree, need)

    def _grow_for_pass_check(self, b):
        missing = GJ_MINGAMES - int(self.u.ownermap.playouts[0])
        self._grow(b, max(missing, 0) * 2 + 1)

    # -- engine interface

    def notify_play(self, b, coord, color):
        """A move is about to be played on ``b`` (by anyone)."""
        if self.tree is None:
            # No state, create one - this is probably game beginning.
            self.prepare_move(b, color)
        if is_resign(coord):
            self._reset_state()
            return
        # Promote node of the appropriate move to the tree root.
        if self.tree.ints[tr.UNTRUSTWORTHY_TREE]:
            found = False
        else:
            self.tree, found = tr.tree_promote_at(self.tree, int(pb.board_size(b)), coord,
                                                  tr.capacity(self.tree))
        if not found:
            # Preserve dynamic komi information, though, that is important.
            self.initial_extra_komi = self.tree.floats[tr.EXTRA_KOMI]
            self._reset_state()

    def genmove(self, b, color, games, pass_all_alive=False):
        """The move for ``color`` after ``games`` simulations in total in the tree."""
        if games < GJ_MINGAMES:
            raise ValueError(f"Pachi needs at least {GJ_MINGAMES} simulations per move")
        if pass_all_alive:
            self.u.flags[PASS_ALL_ALIVE] = 1

        # uct_genmove_setup()
        b.ints[pb.SUPERKO_VIOLATION] = 0
        self.prepare_move(b, color)
        t = self.tree
        t.ints[tr.USE_EXTRA_KOMI] = (self.u.flags[DYNKOMI_MASK] & color) != 0
        # setup_dynkomi(): dynkomi none has no permove, extra komi stays.
        if b.ints[pb.RULES] == pb.RULES_JAPANESE:
            self.u.flags[TERRITORY_SCORING] = 1
        if self.u.flags[TERRITORY_SCORING] \
                and (int(np.floor(b.komi[0])) + int(pb.board_size(b))) & 1:
            b.komi[0] = np.float32(b.komi[0] + (1.0 if color == S_BLACK else -1.0))

        # uct_search(): the worker's simulations, then the final poll.
        worker_rng = search_seeds(self.u.rng)
        while self.tree.u_playouts[self.tree.ints[tr.ROOT]] <= games:
            self._grow(b, 1)  # doubles the arrays when they are full
            walk.uct_playouts(self.u.params, self.u.policy, self.u.prior_settings,
                              self.u.playout_policy, self.u.ownermap, self.tree, b, color, games,
                              worker_rng)
        self._grow_for_pass_check(b)
        uct_search_poll(self.u, self.tree, b, color)

        self._grow_for_pass_check(b)
        coord, best = uct_search_result(self.u, self.tree, b, color,
                                        self.u.flags[PASS_ALL_ALIVE] != 0,
                                        self.resign_threshold)
        coord, best = int(coord), int(best)
        if best == tr.NONE:
            # Pass or resign.
            if is_pass(coord):
                self.initial_extra_komi = self.tree.floats[tr.EXTRA_KOMI]
            self._reset_state()
            return coord
        if not self.tree.ints[tr.UNTRUSTWORTHY_TREE]:
            self.tree, _ = tr.tree_promote_node(self.tree, best, tr.capacity(self.tree))
        else:
            self.initial_extra_komi = self.tree.floats[tr.EXTRA_KOMI]
            self._reset_state()
        return coord

    def dead_group_list(self, b):
        """The dead groups (their base coordinates), as C's uct_dead_group_list()."""
        q = mq.mq_new()
        if self.u.flags[PASS_ALL_ALIVE]:
            return q.move[:0].copy()  # no dead groups
        mock_state = False
        if self.tree is None:
            self.prepare_move(b, S_BLACK)
            mock_state = True
        # Make sure the ownermap is well-seeded.
        while self.u.ownermap.playouts[0] < GJ_MINGAMES:
            self._grow(b, 1)
            walk.uct_playout(self.u.params, self.u.policy, self.u.prior_settings,
                             self.u.playout_policy, self.u.ownermap, self.tree, b, S_BLACK,
                             self.u.rng)
        dead_group_list(self.u, b, q)
        if mock_state:
            self._reset_state()
        return q.move[:q.moves[0]].copy()

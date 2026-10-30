# pachi-numba: a translation of Pachi (GPL-2.0-only), October 2026; see NOTICE.
"""A Pachi player: a board and the UCT engine, like the C interface of pachi-py.

Points are numbered row by row from the top-left corner (``0`` is A9 on
9x9); ``-1`` is a pass, ``-2`` a resignation. Colors are Pachi's: 1 black,
2 white.
"""

import numpy as np

from . import board as pb
from . import random as fast
from .move import PASS, is_pass, is_resign
from .uct.uct import UctEngine

MIN_PLAYOUTS = 500


class PachiPlayer:
    """Pachi's UCT engine (one thread, no pondering, Chinese rules).

    Parameters
    ----------
    size : int
        Board side.
    komi : float
        Komi for white.
    seed : int, optional
        Seed of the random numbers (C: ``fast_srandom`` of the main thread).

    Examples
    --------
    >>> player = PachiPlayer(size=9, komi=7.5, seed=1)
    >>> point = player.genmove(color=1, playouts=500)
    """

    def __init__(self, size=9, komi=7.5, seed=None):
        self.board = pb.board_init(size)
        self.board.komi[0] = komi
        pb.board_set_rules(self.board, "chinese")
        self.engine = UctEngine(self.board)
        if seed is not None:
            self.seed(seed)

    @property
    def size(self):
        return int(pb.real_board_size(self.board))

    def seed(self, seed):
        fast.fast_srandom(self.engine.rng, np.uint64(seed))

    def _coord(self, point):
        if point == -1:
            return PASS
        size = self.size
        return pb.board_size(self.board) * (size - point // size) + point % size + 1

    def _point(self, coord):
        if is_pass(coord):
            return -1
        if is_resign(coord):
            return -2
        size = self.size
        return (size - int(coord) // (size + 2)) * size + int(coord) % (size + 2) - 1

    def play(self, point, color):
        """Play a move not chosen by Pachi; returns False if it is illegal."""
        coord = self._coord(point)
        trial = pb.board_duplicate(self.board)
        if pb.board_play(trial, coord, color) < 0:
            return False
        self.engine.notify_play(self.board, coord, color)
        pb.board_play(self.board, coord, color)
        return True

    def genmove(self, color, playouts):
        """Let Pachi choose and play a move with ``playouts`` simulations."""
        if playouts < MIN_PLAYOUTS:
            raise ValueError(f"Pachi needs at least {MIN_PLAYOUTS} playouts")
        coord = self.engine.genmove(self.board, color, playouts)
        if not is_resign(coord):
            pb.board_play(self.board, coord, color)
        return self._point(coord)

    def black_margin(self):
        """Black's area minus white's area minus komi, dead groups removed."""
        dead = np.asarray(self.engine.dead_group_list(self.board), np.int32)
        return -float(pb.board_official_score(self.board, dead, len(dead)))

    def stone(self, point):
        """0 empty, 1 black, 2 white."""
        return int(self.board.b[self._coord(point)])

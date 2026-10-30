"""The UCT engine: the same moves as the C original over whole games."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import reference  # noqa: E402
from pachi_numba.player import PachiPlayer  # noqa: E402


@pytest.fixture(scope="module")
def c():
    return reference.load()


def play_game(c, seed, playouts, opponent_share, max_moves=200):
    """Pachi against itself, or against random moves for some of the moves.

    Returns the moves; every move and the final margin must equal C's.
    """
    handle = c.pachi_new(9, 7.5, 1)
    c.pachi_seed(seed)
    ours = PachiPlayer(9, 7.5, seed=seed)
    generator = np.random.default_rng(seed)
    color = 1
    passes = 0
    moves = []
    for step in range(max_moves):
        if generator.random() < opponent_share:
            # An outside move: a random legal point (or pass), told to both.
            candidates = [p for p in range(81) if ours.stone(p) == 0]
            generator.shuffle(candidates)
            point = -1
            for candidate in candidates[:20]:
                if ours.play(candidate, color):
                    point = candidate
                    break
            if point == -1:
                assert ours.play(-1, color)
            assert c.pachi_play(handle, point, color) == 1, (seed, step)
            move = point
        else:
            theirs = c.pachi_genmove(handle, color, playouts)
            move = ours.genmove(color, playouts)
            assert move == theirs, (seed, step, moves)
        moves.append(move)
        if move == -2:
            break
        passes = passes + 1 if move == -1 else 0
        if passes == 2:
            break
        color = 3 - color
    for point in range(81):
        assert ours.stone(point) == c.pachi_stone(handle, point)
    assert ours.black_margin() == c.pachi_black_margin(handle), (seed, moves)
    c.pachi_free(handle)
    return moves


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_self_play_games_equal_c(c, seed):
    moves = play_game(c, seed, playouts=500, opponent_share=0.0)
    assert len(moves) > 20


@pytest.mark.parametrize("seed", [4, 5, 6])
def test_games_against_random_moves_equal_c(c, seed):
    """Outside moves exercise tree promotion, symmetry flips and resets."""
    play_game(c, seed, playouts=500, opponent_share=0.4)


def test_more_playouts_equal_c(c):
    play_game(c, 7, playouts=1500, opponent_share=0.2, max_moves=40)


def test_resign_threshold_option():
    """C's option resign_threshold: 1 resigns on the first search, 0 never."""
    eager = PachiPlayer(9, 7.5, seed=3, resign_threshold=1.0)
    assert eager.genmove(1, 1000) == -2  # every win chance is below 1
    never = PachiPlayer(9, 7.5, seed=3, resign_threshold=0.0)
    color = 1
    for _ in range(12):
        assert never.genmove(color, 1000) != -2
        color = 3 - color

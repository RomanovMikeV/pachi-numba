"""The C reference builds and plays; the package imports."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pachi_numba  # noqa: E402
import reference  # noqa: E402


def test_package_imports():
    assert pachi_numba.__version__


def test_reference_plays_a_move():
    library = reference.load()
    library.pachi_seed(1)
    game = library.pachi_new(9, 7.5, 1)
    try:
        assert library.pachi_play(game, 40, 1)  # black takes the center
        point = library.pachi_genmove(game, 2, 500)  # white answers
        assert 0 <= point < 81 and point != 40
        assert library.pachi_stone(game, point) == 2
    finally:
        library.pachi_free(game)


def test_reference_is_deterministic_with_a_seed():
    """The basis of all comparisons: one thread, same seed, same moves."""
    library = reference.load()
    games = []
    for _ in range(2):
        library.pachi_seed(7)
        game = library.pachi_new(9, 7.5, 1)
        moves = []
        for _ in range(6):
            moves.append(library.pachi_genmove(game, 1 + len(moves) % 2, 500))
        library.pachi_free(game)
        games.append(moves)
    assert games[0] == games[1], games

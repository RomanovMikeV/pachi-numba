"""Games in threads: the same moves as the same games one after another."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pachi_numba.parallel import map_games  # noqa: E402
from pachi_numba.player import PachiPlayer  # noqa: E402


def self_play(seed, moves=24, playouts=500):
    """A short game of Pachi against itself, with some outside moves."""
    player = PachiPlayer(size=9, komi=7.5, seed=seed)
    record = []
    color = 1
    for step in range(moves):
        if step % 5 == 4:
            point = (seed * 7 + step * 13) % 81
            record.append(point if player.play(point, color) else player.play(-1, color) and -1)
        else:
            record.append(player.genmove(color, playouts))
        color = 3 - color
    return record, player.black_margin()


def test_threads_give_the_same_games():
    seeds = list(range(1, 7))
    one_by_one = [self_play(seed) for seed in seeds]
    in_threads = map_games(self_play, seeds, max_workers=6)
    assert in_threads == one_by_one


def test_map_games_keeps_order_and_runs_serially():
    assert map_games(lambda x: x * x, [3, 1, 2], max_workers=1) == [9, 1, 4]
    assert map_games(lambda x: x * x, [3, 1, 2], max_workers=3) == [9, 1, 4]
    assert map_games(lambda x: x, []) == []

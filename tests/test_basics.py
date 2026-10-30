"""Random numbers, stones and coordinates: the translation against the C original."""

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import reference  # noqa: E402
from pachi_numba import move, random, stone  # noqa: E402

SEEDS = [0, 1, 29264, 12345, 2**31 - 1, 2**31, 2**40 + 3, 2**63 + 5, 2**64 - 1]
# A mixed sequence of calls, as the engine makes them: (function, max).
CALLS = [("random", 1), ("random", 7), ("frandom", 0), ("random", 65536), ("irandom", 65537),
         ("irandom", 10**6), ("frandom", 0), ("irandom", 2**32 - 1), ("random", 81), ("random", 3)]


@pytest.fixture(scope="module")
def c():
    return reference.load()


@pytest.mark.parametrize("seed", SEEDS)
def test_random_sequences_equal_c(c, seed):
    c.pachi_seed(seed)
    state = random.make_state()
    random.fast_srandom(state, np.uint64(seed))
    assert int(random.fast_getseed(state)) == c.ref_fast_getseed()
    for step in range(400):
        name, max_value = CALLS[step % len(CALLS)]
        if name == "random":
            ours, theirs = int(random.fast_random(state, max_value)), c.ref_fast_random(max_value)
        elif name == "irandom":
            ours, theirs = int(random.fast_irandom(state, max_value)), c.ref_fast_irandom(max_value)
        else:
            ours, theirs = float(random.fast_frandom(state)), c.ref_fast_frandom()
            if math.isnan(theirs):
                assert math.isnan(ours), (seed, step)
                continue
        assert ours == theirs, (seed, step, name, max_value)
    assert int(random.fast_getseed(state)) == c.ref_fast_getseed()


def test_frandom_range_and_dtype():
    state = random.make_state(5)
    values = [random.fast_frandom(state) for _ in range(1000)]
    # numba returns a Python float; inside compiled code it is a float32, so
    # every value must be exactly a float32.
    assert all(float(np.float32(v)) == v for v in values if not math.isnan(v))
    assert all(0.0 <= v < 1.0 for v in values if not math.isnan(v))


def test_stone_other_equals_c(c):
    for s in range(stone.S_MAX):
        assert stone.stone_other(s) == c.ref_stone_other(s)
    assert stone.str2stone("Black") == stone.S_BLACK and stone.str2stone("w") == stone.S_WHITE
    assert stone.stone2char(stone.S_WHITE) == "O" and stone.char2stone("#") == stone.S_OFFBOARD


@pytest.mark.parametrize("side", [9, 19])
def test_coordinates_equal_c(c, side):
    size = side + 2  # board_size(board): the side with the margin
    buffer = __import__("ctypes").create_string_buffer(16)
    for y in range(1, side + 1):
        for x in range(1, side + 1):
            point = move.coord_xy(size, x, y)
            c.ref_coord2str(point, side, buffer, 16)
            name = buffer.value.decode()
            assert move.coord2str(point, size) == name, (point, name)
            assert move.str2coord(name, size) == c.ref_str2coord(name.encode(), size) == point
            assert move.str2coord(name.lower(), size) == point
    for special in ("pass", "PASS", "resign"):
        assert move.str2coord(special, size) == c.ref_str2coord(special.encode(), size)
    assert move.coord2str(move.PASS, size) == "pass"
    assert move.coord2str(move.RESIGN, size) == "resign"


def test_adjacency_and_quadrants():
    size = 11
    center = move.coord_xy(size, 5, 5)
    assert move.coord_is_adjecent(center, center + 1, size)
    assert move.coord_is_adjecent(center, center + size, size)
    assert not move.coord_is_adjecent(center, center + size + 1, size)
    assert move.coord_is_8adjecent(center, center + size + 1, size)
    assert move.coord_quadrant(move.coord_xy(size, 1, 1), size) == 0
    assert move.coord_quadrant(move.coord_xy(size, 9, 9), size) == 3


def test_nrt_flags_are_up_to_date():
    """Compiled functions without reference counting must allocate nothing."""
    import subprocess
    script = pathlib.Path(__file__).resolve().parents[1] / "tools" / "set_nrt_flags.py"
    result = subprocess.run([sys.executable, str(script), "--check"], capture_output=True,
                            text=True)
    assert result.returncode == 0, result.stderr

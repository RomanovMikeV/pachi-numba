"""The board: every field equal to the C original after every move."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import reference  # noqa: E402
from pachi_numba import board as pb  # noqa: E402
from pachi_numba import random  # noqa: E402
from pachi_numba.move import PASS  # noqa: E402
from pachi_numba.stone import S_BLACK, S_WHITE, stone_other  # noqa: E402


@pytest.fixture(scope="module")
def c():
    return reference.board_functions(reference.load())


def dump(c, handle, like):
    out = pb.board_new_like(like)
    c.ref_board_dump(handle, *out[:-1])  # all but scratch
    return out


def assert_equal_boards(ours, theirs, context):
    flen, clen = theirs.ints[pb.FLEN], theirs.ints[pb.CLEN]
    for name in pb.Board._fields:
        if name == "scratch":
            continue
        a, b = getattr(ours, name), getattr(theirs, name)
        if name == "f":
            a, b = a[:flen], b[:flen]
        if name == "c":
            a, b = a[:clen], b[:clen]
        assert np.array_equal(a, b), (context, name, np.flatnonzero(a.ravel() != b.ravel())[:10])


def new_pair(c, side, komi=7.5, rules=b"chinese"):
    handle = c.ref_board_new(side)
    c.ref_board_set_komi(handle, komi)
    ours = pb.board_init(side)
    ours.komi[0] = komi
    if rules == b"stones_only":  # no name in board_set_rules()
        c.ref_board_set_rules_id(handle, pb.RULES_STONES_ONLY)
        ours.ints[pb.RULES] = pb.RULES_STONES_ONLY
    else:
        assert c.ref_board_set_rules(handle, rules)
        assert pb.board_set_rules(ours, rules.decode())
    return ours, handle


def check_points(c, ours, handle, context):
    """The inline checks of board.h at every point, for both colors."""
    for coord in range(int(pb.board_size2(ours))):
        for color in (S_BLACK, S_WHITE):
            if ours.b[coord] == 3:  # S_OFFBOARD: C reads outside the board for some checks
                continue
            expected = [c.ref_board_check(handle, which, coord, color) for which in range(11)]
            got = [
                pb.board_is_valid_play(ours, color, coord),
                pb.board_is_valid_play_no_suicide(ours, color, coord),
                pb.board_safe_to_play(ours, coord, color),
                pb.board_get_atari_neighbor(ours, coord, color),
                pb.board_is_one_point_eye(ours, coord, color),
                pb.board_is_false_eyelike(ours, coord, color),
                pb.board_get_one_point_eye(ours, coord),
                pb.board_coord_in_symmetry(ours, coord),
                pb.group_stone_count(ours, ours.g[coord], color) if ours.g[coord] else -1,
                pb.board_is_eyelike(ours, coord, color),
                pb.immediate_liberty_count(ours, coord),
            ]
            if ours.b[coord] != 0:  # the move checks are for empty points only
                expected, got = expected[3:], got[3:]
            assert [int(v) for v in got] == expected, (context, coord, color)


def test_empty_board_equals_c(c):
    for side in (5, 9, 13, 19):
        ours, handle = new_pair(c, side)
        assert_equal_boards(ours, dump(c, handle, ours), side)
        c.ref_board_free(handle)


@pytest.mark.parametrize("side", [5, 9, 19])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_random_games_equal_c(c, side, seed):
    """board_play_random with the same seed: the same moves and boards, to the end."""
    ours, handle = new_pair(c, side)
    rng = random.make_state()
    random.fast_srandom(rng, seed)
    c.pachi_seed(seed)
    color = S_BLACK
    passes = 0
    for step in range(side * side * 3):
        theirs_move = c.ref_board_play_random(handle, color)
        ours_move = pb.board_play_random(ours, color, rng)
        assert ours_move == theirs_move, (step,)
        assert_equal_boards(ours, dump(c, handle, ours), (side, seed, step))
        if step % 17 == 0:
            check_points(c, ours, handle, (side, seed, step))
        passes = passes + 1 if ours_move == PASS else 0
        if passes == 2:
            break
        color = stone_other(color)
    assert c.ref_board_fast_score(handle) == pb.board_fast_score(ours)
    no_dead = np.zeros(1, np.int32)
    assert c.ref_board_official_score(handle, no_dead, 0) == \
        pb.board_official_score(ours, no_dead, 0)
    groups = np.unique(ours.g[ours.g > 0]).astype(np.int32)[:3]
    assert c.ref_board_official_score(handle, groups, len(groups)) == \
        pb.board_official_score(ours, groups, len(groups))
    c.ref_board_free(handle)


RULES = [b"chinese", b"japanese", b"siming", b"stones_only"]


@pytest.mark.parametrize("seed", range(24))
def test_any_moves_equal_c(c, seed):
    """Uniformly random points: occupied, suicides, ko, superko, passes and undo.

    Small boards (3x3, 4x4) repeat positions, which tests the superko history;
    even sides have no center point, which tests the symmetry of even boards.
    """
    generator = np.random.default_rng(seed)
    side = [3, 4, 5, 6, 7][seed % 5]
    rules = RULES[seed % len(RULES)]
    ours, handle = new_pair(c, side, komi=0.5, rules=rules)
    color = S_BLACK
    for step in range(600):
        x, y = generator.integers(1, side + 1, size=2)
        coord = int(x + y * (side + 2)) if generator.random() > 0.03 else PASS
        theirs_result = c.ref_board_play(handle, coord, color)
        ours_result = pb.board_play(ours, coord, color)
        assert ours_result == theirs_result, (seed, step)
        assert_equal_boards(ours, dump(c, handle, ours), (seed, step))
        if coord == PASS and generator.random() < 0.5:
            assert pb.board_undo(ours) == c.ref_board_undo(handle)
            assert_equal_boards(ours, dump(c, handle, ours), (seed, step, "undo"))
        if ours_result >= 0:
            color = stone_other(color)
        if step % 50 == 0:
            assert c.ref_board_fast_score(handle) == pb.board_fast_score(ours)
            groups = np.unique(ours.g[ours.g > 0]).astype(np.int32)[:2]
            for num_dead in (0, len(groups)):
                assert c.ref_board_official_score(handle, groups, num_dead) == \
                    pb.board_official_score(ours, groups, num_dead), (seed, step)
    check_points(c, ours, handle, seed)
    c.ref_board_free(handle)


def test_superko_is_detected():
    """Some of the small random games above must repeat a position."""
    violations = 0
    for seed in range(24):
        generator = np.random.default_rng(seed)
        ours = pb.board_init(3)
        color = S_BLACK
        for _ in range(600):
            x, y = generator.integers(1, 4, size=2)
            if pb.board_play(ours, int(x + y * 5), color) >= 0:
                color = stone_other(color)
        violations += int(ours.ints[pb.SUPERKO_VIOLATION])
    assert violations > 0


def test_empty_board_official_score_is_komi(c):
    ours, handle = new_pair(c, 9, komi=6.5)
    no_dead = np.zeros(1, np.int32)
    assert c.ref_board_official_score(handle, no_dead, 0) == 6.5
    assert pb.board_official_score(ours, no_dead, 0) == 6.5
    assert not pb.board_set_rules(ours, "unknown")
    assert pb.board_print(ours).count(".") == 81
    c.ref_board_free(handle)


@pytest.mark.parametrize("stones", [2, 5, 7, 9])
@pytest.mark.parametrize("side", [9, 13, 19])
def test_handicap_equals_c(c, side, stones):
    ours, handle = new_pair(c, side, rules=b"japanese")
    c.ref_board_handicap(handle, stones)
    pb.board_handicap(ours, stones)
    assert_equal_boards(ours, dump(c, handle, ours), (side, stones))
    assert c.ref_board_fast_score(handle) == pb.board_fast_score(ours)
    c.ref_board_free(handle)


def test_copy_and_clear(c):
    ours, handle = new_pair(c, 9)
    rng = random.make_state(7)
    for i in range(30):
        pb.board_play_random(ours, S_BLACK if i % 2 == 0 else S_WHITE, rng)
    copy = pb.board_copy(pb.board_new_like(ours), ours)
    for name in pb.Board._fields:
        assert np.array_equal(getattr(copy, name), getattr(ours, name))
    cleared = pb.board_clear(ours)
    fresh = pb.board_init(9)
    fresh.komi[0] = 7.5
    assert_equal_boards(cleared, fresh, "clear")
    c.ref_board_free(handle)

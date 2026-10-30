"""3x3 patterns and tactics: equal answers to the C original on many positions."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import reference  # noqa: E402
from pachi_numba import board as pb  # noqa: E402
from pachi_numba import mq, pattern3, random  # noqa: E402
from pachi_numba.move import PASS  # noqa: E402
from pachi_numba.stone import S_BLACK, S_NONE, S_WHITE, stone_other  # noqa: E402
from pachi_numba.tactics import ladder, n_lib, nakade, one_lib, selfatari, two_lib, util  # noqa: E402

# The patterns of playout/moggy.c (without the "#if 0" ones).
MOGGY_PATTERNS = ["XOX...???", "YO....?.?", "XO?X..x.?", "XOO...?.?X", ".Q.Y.....", "XO?O.o?o?",
                  "XO?O.X???", "?X?O.Oooo", "OX??.O?o?", "X.?O.?##?", "OX?X.O###", "?X?x.O###",
                  "?XQx.x###", "?OYY.O###", "?X.Q.X###"]


@pytest.fixture(scope="module")
def c():
    return reference.tactics_functions(reference.load())


@pytest.fixture(scope="module")
def patterns(c):
    ours = pattern3.pattern3s_init(MOGGY_PATTERNS)
    padded = "".join(p.ljust(10) for p in MOGGY_PATTERNS).encode()
    theirs = c.ref_pattern3s_new(padded, len(MOGGY_PATTERNS))
    return ours, theirs


def positions(c, side, seed, every):
    """Positions of a random game: (our board, C board), the C board replayed from ours."""
    ours = pb.board_init(side)
    ours.komi[0] = 7.5
    handle = c.ref_board_new(side)
    c.ref_board_set_komi(handle, 7.5)
    rng = random.make_state(seed)
    color = S_BLACK
    passes = 0
    for step in range(side * side * 3):
        move = pb.board_play_random(ours, color, rng)
        assert c.ref_board_play(handle, move, color) >= 0
        passes = passes + 1 if move == PASS else 0
        if passes == 2:
            break
        color = stone_other(color)
        if step % every == 0:
            yield ours, handle
    yield ours, handle
    c.ref_board_free(handle)


def c_queue(function, *args):
    moves = np.zeros(mq.MQL, np.int32)
    tags = np.zeros(mq.MQL, np.int32)
    n = np.zeros(1, np.int32)
    result = function(*args, moves, tags, n)
    return result, list(zip(moves[:n[0]], tags[:n[0]]))


def our_queue(q):
    return list(zip(q.move[:q.moves[0]], q.tag[:q.moves[0]].astype(np.int32)))


def seeded(c, seed):
    c.pachi_seed(seed)
    return random.make_state(seed)


def groups_of(b):
    return [int(g) for g in np.unique(b.g[b.g > 0])]


def empty_points(b):
    return [int(p) for p in np.flatnonzero(b.b == S_NONE)]


def test_pattern_hashes_equal_c(c):
    transposed = np.zeros(8, np.uint32)
    generator = np.random.default_rng(0)
    for pat in generator.integers(0, 1 << 20, size=500):
        pat = int(pat)
        assert int(pattern3.hash3_to_hash(pat)) == c.ref_hash3_to_hash(pat)
        assert int(pattern3.pattern3_reverse(pat)) == c.ref_pattern3_reverse(pat)
        c.ref_pattern3_transpose(pat, transposed)
        assert pattern3.pattern3_transpose(pat) == [int(t) for t in transposed]


def test_pattern_table_equals_c(c, patterns):
    ours, theirs = patterns
    pattern = np.zeros(pattern3.PATTERN3_HASH_SIZE, np.uint32)
    value = np.zeros(pattern3.PATTERN3_HASH_SIZE, np.uint8)
    c.ref_pattern3s_dump(theirs, pattern, value)
    assert np.array_equal(ours.value, value)
    assert np.array_equal(ours.pattern[value != 0], pattern[value != 0])
    assert np.count_nonzero(value) > 1000


def check_points(c, patterns, b, handle, context):
    """Self-atari, cousins, nakade, 3x3 patterns and the util functions at every empty point."""
    ours_p, theirs_p = patterns
    bygroup = np.zeros(1, np.int32)
    side = int(pb.real_board_size(b))
    seed = context
    checked = 0
    for point in empty_points(b):
        for color in (S_BLACK, S_WHITE):
            context = (side, seed, int(b.ints[pb.MOVES]), point, color)
            assert selfatari.is_bad_selfatari(b, color, point) == \
                bool(c.ref_is_bad_selfatari(handle, color, point)), context
            rng = seeded(c, point * 7 + color)
            cousin, group = selfatari.selfatari_cousin(b, color, point, rng)
            bygroup[0] = 0
            assert cousin == c.ref_selfatari_cousin(handle, color, point, bygroup), context
            assert group == bygroup[0] or cousin == PASS, context
            assert int(random.fast_getseed(rng)) == c.ref_fast_getseed(), context
            assert nakade.nakade_point(b, point, color) == \
                c.ref_nakade_point(handle, point, color)
            assert nakade.nakade_dead_shape(b, point, color) == \
                bool(c.ref_nakade_dead_shape(handle, point, color)), context
            matches, index = pattern3.pattern3_move_here(ours_p, b, point, color)
            assert (index if matches else -1) == \
                c.ref_pattern3_move_here(theirs_p, handle, point, color), context
            for scan in (False, True):
                assert util.board_local_value(scan, b, point, color) == \
                    c.ref_board_local_value(scan, handle, point, color)
            checked += 1
        for distance in (1, 2, 3):
            assert util.board_stone_radar(b, point, distance) == \
                bool(c.ref_board_stone_radar(handle, point, distance))
        assert util.coord_edge_distance(point, b) == c.ref_coord_edge_distance(handle, point)
        assert util.coord_gridcular_distance(point, 2 * side + 5, b) == \
            c.ref_coord_gridcular_distance(handle, point, 2 * side + 5)
    start = empty_points(b)[0] if empty_points(b) else groups_of(b)[0]
    for maxdist in (2, 5):
        ours_d = np.zeros(int(pb.board_size2(b)), np.int64)
        theirs_d = np.zeros(int(pb.board_size2(b)), np.int32)
        util.cfg_distances(b, start, ours_d, maxdist)
        c.ref_cfg_distances(handle, start, theirs_d, maxdist)
        assert np.array_equal(ours_d, theirs_d)
    assert util.board_estimated_moves_left(b) == c.ref_board_estimated_moves_left(handle)
    for value in (0, 7):
        assert util.board_effective_handicap(b, value) == \
            c.ref_board_effective_handicap(handle, value)
    return checked


@pytest.mark.parametrize("side,seed", [(9, 1), (9, 2), (13, 3), (19, 4)])
def test_point_checks_equal_c(c, patterns, side, seed):
    checked = 0
    for b, handle in positions(c, side, seed, every=7):
        checked += check_points(c, patterns, b, handle, seed)
    assert checked > 1000


def check_groups(c, b, handle, seed):
    """Atari, 2-lib and n-lib checks, ladders and counter-captures of every group."""
    side = int(pb.real_board_size(b))
    ladders = 0
    for group in groups_of(b):
        owner = int(b.b[group])
        libs = int(b.gi_libs[group])
        for to_play in (S_BLACK, S_WHITE):
            context = (side, seed, int(b.ints[pb.MOVES]), group, to_play)
            for use_queue in (False, True):
                q = mq.mq_new()
                ours = one_lib.can_countercapture(b, owner, group, to_play, q, use_queue, 3)
                theirs, theirs_q = c_queue(c.ref_can_countercapture, handle, owner, group,
                                           to_play, use_queue, 3)
                assert ours == bool(theirs) and our_queue(q) == theirs_q, context
            if libs == 1:
                for rate in (0, 50, 100):
                    for use_ladder in (False, True):
                        for middle in (False, True):
                            rng = seeded(c, group + rate)
                            q = mq.mq_new()
                            ladder_move = one_lib.group_atari_check(rate, b, group, to_play, q,
                                                                    use_ladder, middle, 1, rng)
                            theirs, theirs_q = c_queue(c.ref_group_atari_check, rate, handle,
                                                       group, to_play, use_ladder, middle, 1)
                            assert ladder_move == theirs, context
                            assert our_queue(q) == theirs_q, context
                            assert int(random.fast_getseed(rng)) == c.ref_fast_getseed()
                lib = int(b.gi_lib[group, 0])
                for middle in (False, True):
                    is_ladder = ladder.is_ladder(b, lib, group, middle)
                    assert is_ladder == bool(c.ref_is_ladder(handle, lib, group, middle))
                    ladders += is_ladder
            if libs == 2:
                for use_def in (False, True):
                    rng = seeded(c, group)
                    q = mq.mq_new()
                    two_lib.can_atari_group(b, group, owner, to_play, q, 2, use_def, rng)
                    _, theirs_q = c_queue(c.ref_can_atari_group, handle, group, owner,
                                          to_play, 2, use_def)
                    assert our_queue(q) == theirs_q, context
                    for miai in (False, True):
                        rng = seeded(c, group + 1)
                        q = mq.mq_new()
                        two_lib.group_2lib_check(b, group, to_play, q, 2, miai, use_def, rng)
                        _, theirs_q = c_queue(c.ref_group_2lib_check, handle, group, to_play,
                                              2, miai, use_def)
                        assert our_queue(q) == theirs_q, context
                        assert int(random.fast_getseed(rng)) == c.ref_fast_getseed()
                for i in (0, 1):
                    escape, chase = int(b.gi_lib[group, i]), int(b.gi_lib[group, 1 - i])
                    assert ladder.wouldbe_ladder(b, group, escape, chase, owner) == \
                        bool(c.ref_wouldbe_ladder(handle, group, escape, chase, owner)), context
            if to_play == owner:
                rng = seeded(c, group + 2)
                q = mq.mq_new()
                n_lib.group_nlib_defense_check(b, group, to_play, q, 4, rng)
                _, theirs_q = c_queue(c.ref_group_nlib_defense_check, handle, group, to_play, 4)
                assert our_queue(q) == theirs_q, context
    return ladders


@pytest.mark.parametrize("side,seed", [(9, 5), (9, 6), (9, 7), (13, 8), (19, 9)])
def test_group_checks_equal_c(c, side, seed):
    ladders = 0
    for b, handle in positions(c, side, seed, every=3):
        ladders += check_groups(c, b, handle, seed)
    assert ladders > 0


def dense_position(c, side, seed, density, black_share):
    """Random stones, mostly of one color: many small eye spaces and nakade shapes."""
    generator = np.random.default_rng(seed)
    b = pb.board_init(side)
    handle = c.ref_board_new(side)
    for point in generator.permutation(empty_points(b)):
        if generator.random() > density:
            continue
        color = S_BLACK if generator.random() < black_share else S_WHITE
        if pb.board_is_valid_play_no_suicide(b, color, int(point)):
            assert pb.board_play(b, int(point), color) >= 0
            assert c.ref_board_play(handle, int(point), color) >= 0
    return b, handle


@pytest.mark.parametrize("seed", range(12))
def test_dense_positions_equal_c(c, patterns, seed):
    side = (7, 9, 13)[seed % 3]
    b, handle = dense_position(c, side, seed, density=(0.6, 0.75, 0.9)[seed % 3],
                               black_share=(0.8, 0.9, 0.65, 0.95)[seed % 4])
    check_points(c, patterns, b, handle, seed)
    check_groups(c, b, handle, seed)
    for color in (S_BLACK, S_WHITE):
        groups = np.array(groups_of(b)[:2], np.int32)
        for num_dead in (0, len(groups)):
            score = c.ref_board_official_score(handle, groups, num_dead)
            expected = (-score if color == S_BLACK else score) >= 0
            assert util.pass_is_safe(b, color, groups, num_dead) == expected
    c.ref_board_free(handle)


# Eye shapes (x, y offsets): 3 to 6 points, nakade and living shapes.
SHAPES = [
    [(0, 0), (1, 0), (2, 0)], [(0, 0), (1, 0), (1, 1)],
    [(0, 0), (1, 0), (2, 0), (1, 1)], [(0, 0), (1, 0), (0, 1), (1, 1)],
    [(0, 0), (1, 0), (2, 0), (3, 0)], [(0, 0), (1, 0), (2, 0), (2, 1)],
    [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1)], [(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)],
    [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)], [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1)],
    [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)],
    [(1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2)],
    [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)],
    [(0, 0), (1, 0), (2, 0), (1, 1), (1, 2), (0, 2)],
]


def shape_position(c, shape, filled, corner):
    """Black everywhere on 9x9 but the shape; the first ``filled`` shape points white."""
    side = 9
    b = pb.board_init(side)
    handle = c.ref_board_new(side)
    origin = (1, 1) if corner else (3, 3)
    points = [origin[0] + dx + (origin[1] + dy) * (side + 2) for dx, dy in shape]
    for point in points[:filled]:
        assert pb.board_play(b, point, S_WHITE) >= 0
        assert c.ref_board_play(handle, point, S_WHITE) >= 0
    for point in empty_points(b):
        if point in points or b.b[point] != S_NONE:
            continue
        if pb.board_is_valid_play_no_suicide(b, S_BLACK, point):
            assert pb.board_play(b, point, S_BLACK) >= 0
            assert c.ref_board_play(handle, point, S_BLACK) >= 0
    return b, handle


@pytest.mark.parametrize("corner", [False, True])
@pytest.mark.parametrize("filled", [0, 1, 2, 3])
def test_eye_shapes_equal_c(c, patterns, corner, filled):
    for shape in SHAPES:
        b, handle = shape_position(c, shape, filled, corner)
        check_points(c, patterns, b, handle, (corner, filled))
        check_groups(c, b, handle, (corner, filled))
        c.ref_board_free(handle)

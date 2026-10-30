"""Playouts: the same moves, boards, results and priors as the C original."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import reference  # noqa: E402
from pachi_numba import board as pb  # noqa: E402
from pachi_numba import ownermap as om  # noqa: E402
from pachi_numba import mq, playout, random  # noqa: E402
from pachi_numba.move import PASS  # noqa: E402
from pachi_numba.playout import moggy  # noqa: E402
from pachi_numba.stone import S_BLACK, S_MAX, S_NONE, S_WHITE, stone_other  # noqa: E402
from pachi_numba.uct import prior  # noqa: E402
from test_board import assert_equal_boards, dump  # noqa: E402

# Policy options: the default and options that switch on the other code paths.
OPTIONS = [None, "fullchoose", "fullchoose:capturerate=90:tenukiprob=0.5",
           "capturerate=90:capcheckall", "capturerate=90:capcheckall:fullchoose",
           "ladderrate=60:middle_ladder", "fillboardtries=5", "fullchoose:fillboardtries=5",
           "selfatari_other=0:pattern2=0:eyefillrate=100", "rate=50:korate=-1:nlibrate=-1",
           "mqprob=1%2%3%4%5%6%7%8:pat3gammas=1%1%1%1%1%1%1%1%1%1%1%1%1%1%1"]


@pytest.fixture(scope="module")
def c():
    return reference.playout_functions(reference.load())


def new_pair(c, side, komi=7.5):
    ours = pb.board_init(side)
    ours.komi[0] = komi
    handle = c.ref_board_new(side)
    c.ref_board_set_komi(handle, komi)
    return ours, handle


def new_policies(c, option, ours, handle):
    if option == "light":
        return moggy.playout_light_init(None, ours), c.ref_light_new(handle)
    encoded = option.encode() if option else None
    return moggy.playout_moggy_init(option, ours), c.ref_moggy_new(encoded, handle)


def seeded(c, seed):
    c.pachi_seed(seed)
    return random.make_state(seed)


@pytest.mark.parametrize("option", OPTIONS + ["light"])
@pytest.mark.parametrize("side", [9, 19])
def test_playout_moves_equal_c(c, side, option):
    """play_random_move from the empty board to the end: the same moves and boards."""
    ours, handle = new_pair(c, side)
    ours_p, theirs_p = new_policies(c, option, ours, handle)
    rng = seeded(c, 11 + side)
    queue = mq.mq_new()
    gammas = np.zeros(mq.MQL, np.uint64)
    color = S_BLACK
    passes = 0
    for step in range(side * side * 3):
        theirs_move = c.ref_play_random_move(theirs_p, handle, color)
        ours_move = playout.play_random_move(None, ours, color, ours_p, rng, queue, gammas)
        assert ours_move == theirs_move, (option, step)
        assert int(random.fast_getseed(rng)) == c.ref_fast_getseed(), (option, step)
        if step % 10 == 0:
            assert_equal_boards(ours, dump(c, handle, ours), (option, step))
        passes = passes + 1 if ours_move == PASS else 0
        if passes == 2:
            break
        color = stone_other(color)
    assert_equal_boards(ours, dump(c, handle, ours), option)
    c.ref_board_free(handle)


@pytest.mark.parametrize("option", [None, "fullchoose", "light"])
@pytest.mark.parametrize("seed", range(6))
def test_random_games_equal_c(c, option, seed):
    """play_random_game from positions of a game: result, AMAF record and owners."""
    side = 9 if seed < 4 else 13
    ours, handle = new_pair(c, side)
    ours_p, theirs_p = new_policies(c, option, ours, handle)
    rng = random.make_state(seed)
    color = S_BLACK
    for move in range(0, 60, 6 + seed):
        for _ in range(6 + seed):  # advance the game with light random moves
            coord = pb.board_play_random(ours, color, rng)
            assert c.ref_board_play(handle, coord, color) >= 0
            color = stone_other(color)
        ours_copy = pb.board_duplicate(ours)
        theirs_copy = c.ref_board_clone(handle)
        size2 = int(pb.board_size2(ours))
        game, ko = np.zeros(playout.MAX_GAMELEN, np.int32), np.zeros(playout.MAX_GAMELEN, np.uint8)
        game_len, owner_playouts = np.zeros(1, np.int32), np.zeros(1, np.int32)
        owner_map = np.zeros(size2 * S_MAX, np.int32)
        gamelen, mercymin = (playout.MAX_GAMELEN, 0) if move % 2 else (move + 40, 3)
        play_rng = seeded(c, 1000 * seed + move)
        theirs = c.ref_play_random_game(theirs_p, theirs_copy, color, gamelen, mercymin, game, ko,
                                        game_len, owner_map, owner_playouts)
        amaf = playout.amafmap_new()
        owners = om.ownermap_new(size2)
        setup = playout.PlayoutSetup(gamelen, mercymin)
        result = playout.play_random_game(setup, ours_copy, color, amaf, True, owners, True,
                                          ours_p, play_rng)
        context = (option, seed, move)
        assert result == theirs, context
        assert int(random.fast_getseed(play_rng)) == c.ref_fast_getseed(), context
        n = amaf.ints[playout.GAMELEN]
        assert n == game_len[0], context
        assert np.array_equal(amaf.game[:n], game[:n]), context
        assert np.array_equal(amaf.is_ko_capture[:n], ko[:n].astype(bool)), context
        assert owners.playouts[0] == owner_playouts[0]
        assert np.array_equal(owners.map.ravel(), owner_map), context
        assert_equal_boards(ours_copy, dump(c, theirs_copy, ours_copy), context)
        c.ref_board_free(theirs_copy)
    c.ref_board_free(handle)


@pytest.mark.parametrize("option", [None, "ladderrate=50", "selfatari_other=0",
                                    "rate=0:atarirate=0", "light"])
def test_assess_equals_c(c, option):
    """The prior values of every move, from positions of random games."""
    for seed in range(4):
        side = 9 if seed % 2 else 13
        ours, handle = new_pair(c, side)
        ours_p, theirs_p = new_policies(c, option, ours, handle)
        rng = random.make_state(seed)
        color = S_BLACK
        for step in range(side * side * 2):
            coord = pb.board_play_random(ours, color, rng)
            assert c.ref_board_play(handle, coord, color) >= 0
            color = stone_other(color)
            if step % 9:
                continue
            for parity in (1, -1):
                size2 = int(pb.board_size2(ours))
                prior_map = prior.prior_map_new(ours, color, parity)
                prior_map.consider[1:] = ours.b == S_NONE
                assess_rng = seeded(c, 2 * step + (parity > 0))
                moggy.playout_moggy_assess(ours_p, prior_map, 20, assess_rng)
                values = np.zeros(size2, np.float32)
                playouts = np.zeros(size2, np.int32)
                c.ref_policy_assess(theirs_p, handle, color, parity,
                                    prior_map.consider[1:].astype(np.uint8), 20, values, playouts)
                context = (option, seed, step, parity)
                assert np.array_equal(prior_map.prior_playouts[1:], playouts), context
                assert np.array_equal(prior_map.prior_value[1:], values), context
                assert int(random.fast_getseed(assess_rng)) == c.ref_fast_getseed(), context
        c.ref_board_free(handle)


def test_permit_equals_c(c):
    """The permit of random moves: the same answers and redirections."""
    ours, handle = new_pair(c, 9)
    ours_p, theirs_p = new_policies(c, "eyefillrate=100:selfatarirate=100", ours, handle)
    rng = random.make_state(3)
    out = np.zeros(1, np.int32)
    color = S_BLACK
    redirected = 0
    for step in range(150):
        for point in np.flatnonzero(ours.b == S_NONE):
            point = int(point)
            if not pb.board_is_valid_play(ours, color, point):
                continue
            check_rng = seeded(c, point)
            permitted, coord = moggy.playout_moggy_permit(ours_p, ours, point, color, check_rng)
            assert permitted == bool(c.ref_moggy_permit(theirs_p, handle, point, color, out))
            assert coord == out[0]
            redirected += coord != point
        coord = pb.board_play_random(ours, color, rng)
        assert c.ref_board_play(handle, coord, color) >= 0
        color = stone_other(color)
    assert redirected > 0
    c.ref_board_free(handle)


def test_ownermap_judgement():
    """Group status from an ownermap of playouts (C code without a C test hook: by hand)."""
    b = pb.board_init(5)
    rng = random.make_state(1)
    for i in range(16):
        pb.board_play_random(b, S_BLACK if i % 2 == 0 else S_WHITE, rng)
    size2 = int(pb.board_size2(b))
    owners = om.ownermap_new(size2)
    for _ in range(4):
        om.board_ownermap_fill(owners, b)
    gs = np.zeros(size2, np.int64)
    om.board_ownermap_judge_groups(b, owners, 0.8, gs)
    for g in np.unique(b.g[b.g > 0]):
        assert gs[g] == om.GS_ALIVE
    q = __import__("pachi_numba.mq", fromlist=["mq"]).mq_new()
    om.groups_of_status(b, gs, om.GS_ALIVE, q)
    assert q.moves[0] == len(np.unique(b.g[b.g > 0]))
    stone = int(np.flatnonzero(b.b == S_BLACK)[0])
    assert om.board_ownermap_estimate_point(owners, stone) == 1.0
    assert om.board_ownermap_judge_point(owners, stone, 0.8) == om.PJ_BLACK

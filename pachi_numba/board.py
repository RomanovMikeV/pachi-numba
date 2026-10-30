# pachi-numba: a translation of Pachi (GPL-2.0-only), October 2026; see NOTICE.
"""Translation of Pachi's ``board.h`` and ``board.c``.

The C ``struct board`` is a named tuple of NumPy arrays (``Board``), so that
numba can compile functions that take it. The maps of the board are arrays of
``board_size2`` points, as in C; the scalar fields are in ``ints`` at the
indices below. Configuration of this snapshot: ``WANT_BOARD_C`` and
``BOARD_PAT3`` on, ``BOARD_TRAITS`` and ``BOARD_SPATHASH`` off, no fixed
``BOARD_SIZE``. Debug printing is left out. ``fbook`` (opening books), ``es``
and ``ps`` (engine and playout state) are kept by their owners instead.
"""

from collections import namedtuple

import numpy as np
from numba import njit

from . import random as fast
from .move import PASS, is_pass, is_resign
from .stone import S_BLACK, S_MAX, S_NONE, S_OFFBOARD, S_WHITE, stone_other

BOARD_MAX_SIZE = 19
BOARD_MAX_MOVES = BOARD_MAX_SIZE * BOARD_MAX_SIZE
BOARD_MAX_GROUPS = BOARD_MAX_SIZE * BOARD_MAX_SIZE // 2
GROUP_KEEP_LIBS = 10
GROUP_REFILL_LIBS = 5
HISTORY_HASH_BITS = 12
HISTORY_HASH_MASK = (1 << HISTORY_HASH_BITS) - 1
# Board.scratch: C keeps these small arrays on the stack; here they are
# allocated once per board, so that the hot tactics allocate nothing.
SCRATCH_SIZE = 128

# enum e_sym
SYM_FULL, SYM_DIAG_UP, SYM_DIAG_DOWN, SYM_HORIZ, SYM_VERT, SYM_NONE = range(6)
# enum go_ruleset
(RULES_CHINESE, RULES_AGA, RULES_NEW_ZEALAND, RULES_JAPANESE, RULES_STONES_ONLY,
 RULES_SIMING) = range(6)

# Scalar fields of struct board: indices into Board.ints.
(SIZE, SIZE2, BITS2, HANDICAP, RULES, MOVES, SUPERKO_VIOLATION, FLEN, CLEN, LAST_KO_AGE,
 LAST_MOVE_COORD, LAST_MOVE_COLOR, LAST_MOVE2_COORD, LAST_MOVE2_COLOR,
 LAST_MOVE3_COORD, LAST_MOVE3_COLOR, LAST_MOVE4_COORD, LAST_MOVE4_COLOR,
 LAST_KO_COORD, LAST_KO_COLOR, KO_COORD, KO_COLOR,
 SYM_X1, SYM_X2, SYM_Y1, SYM_Y2, SYM_D, SYM_TYPE, NUM_INTS) = range(29)

Board = namedtuple("Board", [
    "ints",          # int64[NUM_INTS]: the scalar fields above
    "captures",      # int64[S_MAX]
    "komi",          # float32[1] (floating_t is float in this build)
    "nei8",          # int64[8]: offsets of foreach_8neighbor
    "dnei",          # int64[4]: offsets of foreach_diag_neighbor
    "b",             # int8[size2]: stones
    "g",             # int32[size2]: group of every stone, 0 = none
    "p",             # int32[size2]: next stone of the group, 0 = last
    "n",             # int8[size2, S_MAX]: neighbor counts per color
    "h",             # uint64[2 * size2]: Zobrist hashes
    "pat3",          # uint32[size2]: 3x3 pattern codes (empty points)
    "coord",         # uint8[size2, 2]: x, y of every point
    "gi_lib",        # int32[size2, GROUP_KEEP_LIBS]: liberties of groups
    "gi_libs",       # int32[size2]: number of known liberties
    "f",             # int32[size2]: queue of free points, FLEN long
    "c",             # int32[size2]: queue of capturable groups, CLEN long
    "history_hash",  # uint64[1 << HISTORY_HASH_BITS]
    "hash",          # uint64[1]
    "qhash",         # uint64[4]
    "scratch",       # int64[SCRATCH_SIZE]: work space of tactics (not in C, not copied)
])


@njit(cache=True, nogil=True, _nrt=True)
def board_alloc(size2):
    """A zeroed board of ``size2`` points (with the margin); also in compiled code."""
    return Board(
        np.zeros(NUM_INTS, np.int64), np.zeros(S_MAX, np.int64), np.zeros(1, np.float32),
        np.zeros(8, np.int64), np.zeros(4, np.int64), np.zeros(size2, np.int8),
        np.zeros(size2, np.int32), np.zeros(size2, np.int32), np.zeros((size2, S_MAX), np.int8),
        np.zeros(2 * size2, np.uint64), np.zeros(size2, np.uint32),
        np.zeros((size2, 2), np.uint8), np.zeros((size2, GROUP_KEEP_LIBS), np.int32),
        np.zeros(size2, np.int32), np.zeros(size2, np.int32), np.zeros(size2, np.int32),
        np.zeros(1 << HISTORY_HASH_BITS, np.uint64), np.zeros(1, np.uint64),
        np.zeros(4, np.uint64), np.zeros(SCRATCH_SIZE, np.int64),
    )


def _allocate(size2):
    return board_alloc(size2)


@njit(cache=True, nogil=True, _nrt=False)
def board_size(b):
    return b.ints[SIZE]


@njit(cache=True, nogil=True, _nrt=False)
def board_size2(b):
    return b.ints[SIZE2]


@njit(cache=True, nogil=True, _nrt=False)
def real_board_size(b):
    return b.ints[SIZE] - 2


@njit(cache=True, nogil=True, _nrt=False)
def board_large(b):
    return board_size(b) - 2 >= 15


@njit(cache=True, nogil=True, _nrt=False)
def board_small(b):
    return board_size(b) - 2 <= 9


@njit(cache=True, nogil=True, _nrt=False)
def coord_x(c, b):
    return np.int64(b.coord[c, 0])


@njit(cache=True, nogil=True, _nrt=False)
def coord_y(c, b):
    return np.int64(b.coord[c, 1])


@njit(cache=True, nogil=True, _nrt=False)
def coord_quadrant(c, b):
    return (coord_x(c, b) > board_size(b) // 2) + 2 * (coord_y(c, b) > board_size(b) // 2)


@njit(cache=True, nogil=True, _nrt=False)
def board_atxy(b, x, y):
    return b.b[x + board_size(b) * y]


@njit(cache=True, nogil=True, _nrt=False)
def group_atxy(b, x, y):
    return b.g[x + board_size(b) * y]


@njit(cache=True, nogil=True, _nrt=False)
def immediate_liberty_count(b, coord):
    return 4 - b.n[coord, S_BLACK] - b.n[coord, S_WHITE] - b.n[coord, S_OFFBOARD]


@njit(cache=True, nogil=True, _nrt=False)
def group_is_onestone(b, g):
    return b.p[g] == 0


@njit(cache=True, nogil=True, _nrt=False)
def board_group_captured(b, g):
    return b.gi_libs[g] == 0


@njit(cache=True, nogil=True, _nrt=False)
def board_group_other_lib(b, g, lib):
    """Makes sense only for groups with two liberties."""
    return b.gi_lib[g, 0] if b.gi_lib[g, 0] != lib else b.gi_lib[g, 1]


@njit(cache=True, nogil=True, _nrt=False)
def hash_at(b, coord, color):
    return b.h[(board_size2(b) if color == S_BLACK else 0) + coord]


@njit(cache=True, nogil=True, _nrt=False)
def neighbors(b, coord):
    """The four neighbors in the order of C's foreach_neighbor: up, left, right, down."""
    size = board_size(b)
    return (coord - size, coord - 1, coord + 1, coord + size)


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def pattern3_hash(b, c):
    """pattern3.h: the 3x3 pattern code of the point ``c``."""
    x = coord_x(c, b)
    y = coord_y(c, b)
    pat = np.uint32(0)
    pat |= np.uint32(board_atxy(b, x - 1, y - 1)) << 14
    pat |= np.uint32(board_atxy(b, x, y - 1)) << 12
    pat |= np.uint32(board_atxy(b, x + 1, y - 1)) << 10
    pat |= np.uint32(board_atxy(b, x - 1, y)) << 8
    pat |= np.uint32(board_atxy(b, x + 1, y)) << 6
    pat |= np.uint32(board_atxy(b, x - 1, y + 1)) << 4
    pat |= np.uint32(board_atxy(b, x, y + 1)) << 2
    pat |= np.uint32(board_atxy(b, x + 1, y + 1))
    pat |= np.uint32(_atari_atxy(b, x, y - 1)) << 19
    pat |= (np.uint32(_atari_atxy(b, x - 1, y)) << 18) | (np.uint32(_atari_atxy(b, x + 1, y)) << 17)
    pat |= np.uint32(_atari_atxy(b, x, y + 1)) << 16
    return pat


@njit(cache=True, nogil=True, _nrt=False)
def _atari_atxy(b, x, y):
    g = group_atxy(b, x, y)
    return g != 0 and b.gi_libs[g] == 1


# ---- board_setup, board_init, board_resize, board_init_data, board_clear, board_copy


def _board_setup(b):
    for array in b:
        array[...] = 0
    moves = ((LAST_MOVE_COORD, LAST_MOVE_COLOR), (LAST_MOVE2_COORD, LAST_MOVE2_COLOR),
             (LAST_MOVE3_COORD, LAST_MOVE3_COLOR), (LAST_MOVE4_COORD, LAST_MOVE4_COLOR),
             (LAST_KO_COORD, LAST_KO_COLOR), (KO_COORD, KO_COLOR))
    for coord_index, color_index in moves:
        b.ints[coord_index] = PASS
        b.ints[color_index] = S_NONE


def board_init(size=9):
    """A new, cleared board; ``size`` without the margin (C: 9 by default)."""
    b = _allocate((size + 2) * (size + 2))
    b.ints[SIZE] = size + 2
    return board_clear(b)


def board_resize(b, size):
    """``size`` without the S_OFFBOARD margin; returns the (new) board."""
    assert size <= BOARD_MAX_SIZE
    new = _allocate((size + 2) * (size + 2))
    new.ints[:] = b.ints
    new.captures[:] = b.captures
    new.komi[:] = b.komi
    new.ints[SIZE] = size + 2
    new.ints[SIZE2] = (size + 2) * (size + 2)
    bits2 = 1
    while (1 << bits2) < new.ints[SIZE2]:
        bits2 += 1
    new.ints[BITS2] = bits2
    return new


def _board_init_data(b):
    size = int(board_size(b))
    komi, rules, handicap = b.komi[0], b.ints[RULES], b.ints[HANDICAP]
    _board_setup(b)
    b.ints[SIZE] = size
    b = board_resize(b, size - 2)
    b.komi[0], b.ints[RULES], b.ints[HANDICAP] = komi, rules, handicap
    b.nei8[:] = [-size - 1, 1, 1, size - 2, 2, size - 2, 1, 1]
    b.dnei[:] = [-size - 1, 2, size * 2 - 2, 2]
    if size % 2:
        b.ints[SYM_D] = 1
        b.ints[SYM_X1] = b.ints[SYM_Y1] = size // 2
        b.ints[SYM_X2] = b.ints[SYM_Y2] = size - 1
        b.ints[SYM_TYPE] = SYM_FULL
    else:
        b.ints[SYM_D] = 0
        b.ints[SYM_X1] = b.ints[SYM_Y1] = 1
        b.ints[SYM_X2] = b.ints[SYM_Y2] = size - 1
        b.ints[SYM_TYPE] = SYM_NONE
    size2 = size * size
    for c in range(size2):
        b.coord[c, 0] = c % size
        b.coord[c, 1] = c // size
    top_row = size2 - size
    for i in range(size):
        b.b[i] = b.b[top_row + i] = S_OFFBOARD
    for i in range(0, top_row + 1, size):
        b.b[i] = b.b[size - 1 + i] = S_OFFBOARD
    for coord in range(size2):
        if b.b[coord] == S_OFFBOARD:
            continue
        for c in (coord - size, coord - 1, coord + 1, coord + size):
            b.n[coord, b.b[c]] += 1
    flen = 0
    for i in range(size, (size - 1) * size):
        if i % size != 0 and i % size != size - 1:
            b.f[flen] = i
            flen += 1
    b.ints[FLEN] = flen
    # Zobrist hashes: stable across runs, not from fast_random().
    mask = (1 << 64) - 1
    hseed = 0x3121110101112131
    for c in range(size2):
        hseed = (hseed * 16807) & mask
        b.h[c * 2] = hseed if hseed else 1
        hseed = (hseed * 16807) & mask
        b.h[c * 2 + 1] = hseed if hseed else 1
    for c in range(size2):
        if b.b[c] == S_NONE:
            b.pat3[c] = pattern3_hash(b, c)
    return b


def board_clear(b):
    """Clear the board, keeping komi and rules; returns the (new) board.

    C copies a cached cleared board of the same size, which is the same as
    building it again.
    """
    komi, rules = b.komi[0], b.ints[RULES]
    b = _board_init_data(b)
    b.ints[HANDICAP] = 0
    b.komi[0] = komi
    b.ints[RULES] = rules
    return b


@njit(cache=True, nogil=True, _nrt=True)
def board_new_like(b):
    """An allocated board of the same size, to copy into (C: struct board b2)."""
    return board_alloc(board_size2(b))


@njit(cache=True, nogil=True, _nrt=True)
def board_duplicate(b):
    """C: ``struct board b2; board_copy(&b2, b);``"""
    return board_copy(board_new_like(b), b)


@njit(cache=True, nogil=True, _nrt=False)
def neighbor_count_at(b, coord, color):
    return np.int64(b.n[coord, color])


@njit(cache=True, nogil=True, _nrt=False)
def group_base(g):
    return g


@njit(cache=True, nogil=True, _nrt=True)
def board_copy(b2, b1):
    """Copy ``b1`` into ``b2`` (of the same size)."""
    b2.ints[:] = b1.ints
    b2.captures[:] = b1.captures
    b2.komi[:] = b1.komi
    b2.nei8[:] = b1.nei8
    b2.dnei[:] = b1.dnei
    b2.b[:] = b1.b
    b2.g[:] = b1.g
    b2.p[:] = b1.p
    b2.n[:] = b1.n
    b2.h[:] = b1.h
    b2.pat3[:] = b1.pat3
    b2.coord[:] = b1.coord
    b2.gi_lib[:] = b1.gi_lib
    b2.gi_libs[:] = b1.gi_libs
    b2.f[:] = b1.f
    b2.c[:] = b1.c
    b2.history_hash[:] = b1.history_hash
    b2.hash[:] = b1.hash
    b2.qhash[:] = b1.qhash
    return b2


def board_set_rules(b, name):
    rules = {"japanese": RULES_JAPANESE, "chinese": RULES_CHINESE, "aga": RULES_AGA,
             "new_zealand": RULES_NEW_ZEALAND, "siming": RULES_SIMING,
             "simplified_ing": RULES_SIMING}.get(name.lower())
    if rules is None:
        return False
    b.ints[RULES] = rules
    return True


# ---- hashes and symmetry

_ATARIBITS = np.array([-1, 0, -1, 1, 2, -1, 3, -1], dtype=np.int64)


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def board_hash_update(b, coord, color):
    b.hash[0] ^= hash_at(b, coord, color)
    b.qhash[coord_quadrant(coord, b)] ^= hash_at(b, coord, color)
    new_color = b.b[coord]
    in_atari = False
    if new_color == S_NONE:
        b.pat3[coord] = pattern3_hash(b, coord)
    else:
        in_atari = b.gi_libs[b.g[coord]] == 1
    c = coord
    for i in range(8):
        c += b.nei8[i]
        if b.b[c] != S_NONE:
            continue
        pat = np.int64(b.pat3[c])
        pat &= ~(3 << (i * 2))
        pat |= np.int64(new_color) << (i * 2)
        if _ATARIBITS[i] >= 0:
            pat &= ~(1 << (16 + _ATARIBITS[i]))
            pat |= np.int64(in_atari) << (16 + _ATARIBITS[i])
        b.pat3[c] = np.uint32(pat & 0xFFFFFFFF)


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def board_hash_commit(b):
    hash_value = b.hash[0]
    mask = np.uint64(HISTORY_HASH_MASK)
    if b.history_hash[hash_value & mask] == 0:
        b.history_hash[hash_value & mask] = hash_value
    else:
        i = hash_value
        while b.history_hash[i & mask]:
            if b.history_hash[i & mask] == hash_value:
                b.ints[SUPERKO_VIOLATION] = 1
                return
            i = (i + np.uint64(1)) & mask
        b.history_hash[i & mask] = hash_value


@njit(cache=True, nogil=True, _nrt=False)
def _break_symmetry(b):
    b.ints[SYM_TYPE] = SYM_NONE
    b.ints[SYM_X1] = b.ints[SYM_Y1] = 1
    b.ints[SYM_X2] = b.ints[SYM_Y2] = board_size(b) - 1
    b.ints[SYM_D] = 0


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def board_symmetry_update(b, c):
    kind = b.ints[SYM_TYPE]
    if kind == SYM_NONE:
        return
    x = coord_x(c, b)
    y = coord_y(c, b)
    t = board_size(b) // 2
    dx = board_size(b) - 1 - x
    if kind == SYM_FULL:
        if x == t and y == t:
            return
        if x == y:
            b.ints[SYM_TYPE] = SYM_DIAG_UP
            b.ints[SYM_X1] = b.ints[SYM_Y1] = 1
            b.ints[SYM_X2] = b.ints[SYM_Y2] = board_size(b) - 1
            b.ints[SYM_D] = 1
        elif dx == y:
            b.ints[SYM_TYPE] = SYM_DIAG_DOWN
            b.ints[SYM_X1] = b.ints[SYM_Y1] = 1
            b.ints[SYM_X2] = b.ints[SYM_Y2] = board_size(b) - 1
            b.ints[SYM_D] = 1
        elif x == t:
            b.ints[SYM_TYPE] = SYM_HORIZ
            b.ints[SYM_Y1] = 1
            b.ints[SYM_Y2] = board_size(b) - 1
            b.ints[SYM_D] = 0
        elif y == t:
            b.ints[SYM_TYPE] = SYM_VERT
            b.ints[SYM_X1] = 1
            b.ints[SYM_X2] = board_size(b) - 1
            b.ints[SYM_D] = 0
        else:
            _break_symmetry(b)
    elif kind == SYM_DIAG_UP:
        if x != y:
            _break_symmetry(b)
    elif kind == SYM_DIAG_DOWN:
        if dx != y:
            _break_symmetry(b)
    elif kind == SYM_HORIZ:
        if x != t:
            _break_symmetry(b)
    elif kind == SYM_VERT:
        if y != t:
            _break_symmetry(b)


@njit(cache=True, nogil=True, _nrt=False)
def board_coord_in_symmetry(b, c):
    if coord_y(c, b) < b.ints[SYM_Y1] or coord_y(c, b) > b.ints[SYM_Y2]:
        return False
    if coord_x(c, b) < b.ints[SYM_X1] or coord_x(c, b) > b.ints[SYM_X2]:
        return False
    if b.ints[SYM_D]:
        x = coord_x(c, b)
        if b.ints[SYM_TYPE] == SYM_DIAG_DOWN:
            x = board_size(b) - 1 - x
        if x > coord_y(c, b):
            return False
    return True


# ---- liberties and groups

@njit(cache=True, nogil=True, inline="always", _nrt=False)
def board_capturable_add(b, group, lib, onestone):
    i = 0
    for c in neighbors(b, lib):
        b.pat3[lib] |= np.uint32(b.g[c] == group) << np.uint32(16 + 3 - i)
        i += 1
    b.c[b.ints[CLEN]] = group
    b.ints[CLEN] += 1


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def board_capturable_rm(b, group, lib, onestone):
    i = 0
    for c in neighbors(b, lib):
        b.pat3[lib] &= ~(np.uint32(b.g[c] == group) << np.uint32(16 + 3 - i))
        i += 1
    for i in range(b.ints[CLEN]):
        if b.c[i] == group:
            b.ints[CLEN] -= 1
            b.c[i] = b.c[b.ints[CLEN]]
            return
    raise AssertionError("rm of bad group")


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def board_group_addlib(b, group, coord):
    onestone = group_is_onestone(b, group)
    libs = b.gi_libs[group]
    if libs < GROUP_KEEP_LIBS:
        for i in range(GROUP_KEEP_LIBS):
            if b.gi_lib[group, i] == coord:
                return
        if libs == 0:
            board_capturable_add(b, group, coord, onestone)
        elif libs == 1:
            board_capturable_rm(b, group, b.gi_lib[group, 0], onestone)
        b.gi_lib[group, b.gi_libs[group]] = coord
        b.gi_libs[group] += 1


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def _is_known_lib(b, group, c, avoid):
    if c == avoid:
        return True
    for i in range(b.gi_libs[group]):
        if b.gi_lib[group, i] == c:
            return True
    return False


@njit(cache=True, nogil=True, _nrt=False)
def board_group_find_extra_libs(b, group, avoid):
    """Refill the liberties of ``group`` up to GROUP_KEEP_LIBS.

    C marks the known liberties (and ``avoid``) in a board-sized watermark
    on the stack; the marks are exactly the current liberty list (the unused
    entries are 0, an S_OFFBOARD point) and ``avoid``, so they are looked up
    there instead of allocating a map.
    """
    c_stone = group
    c2 = b.p[c_stone]
    while True:
        for c in neighbors(b, c_stone):
            if b.b[c] != S_NONE or _is_known_lib(b, group, c, avoid):
                continue
            b.gi_lib[group, b.gi_libs[group]] = c
            b.gi_libs[group] += 1
            if b.gi_libs[group] >= GROUP_KEEP_LIBS:
                return
        c_stone = c2
        if c_stone == 0:
            break
        c2 = b.p[c2]


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def board_group_rmlib(b, group, coord):
    onestone = group_is_onestone(b, group)
    for i in range(GROUP_KEEP_LIBS):
        if b.gi_lib[group, i] != coord:
            continue
        b.gi_libs[group] -= 1
        lib = b.gi_lib[group, i] = b.gi_lib[group, b.gi_libs[group]]
        b.gi_lib[group, b.gi_libs[group]] = 0
        libs = b.gi_libs[group]
        if libs > GROUP_REFILL_LIBS:
            return
        if libs == GROUP_REFILL_LIBS:
            board_group_find_extra_libs(b, group, coord)
        libs = b.gi_libs[group]
        if libs == 1:
            board_capturable_add(b, group, b.gi_lib[group, 0], onestone)
        elif libs == 0:
            board_capturable_rm(b, group, lib, onestone)
        return


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def board_remove_stone(b, group, c):
    """Low level: does not keep all the board data consistent."""
    color = b.b[c]
    b.b[c] = S_NONE
    b.g[c] = 0
    board_hash_update(b, c, color)
    coord = c
    for n in neighbors(b, coord):
        b.n[n, color] -= 1
        g = b.g[n]
        if g and g != group:
            board_group_addlib(b, g, coord)
    b.pat3[c] = pattern3_hash(b, c)
    b.f[b.ints[FLEN]] = c
    b.ints[FLEN] += 1


@njit(cache=True, nogil=True, _nrt=False)
def board_group_capture(b, group):
    stones = 0
    c = group
    c2 = b.p[c]
    while True:
        b.captures[stone_other(b.b[c])] += 1
        board_remove_stone(b, group, c)
        stones += 1
        c = c2
        if c == 0:
            break
        c2 = b.p[c2]
    b.gi_lib[group, :] = 0
    b.gi_libs[group] = 0
    return stones


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def add_to_group(b, group, prevstone, coord):
    b.g[coord] = group
    b.p[coord] = b.p[prevstone]
    b.p[prevstone] = coord
    for c in neighbors(b, coord):
        if b.b[c] == S_NONE:
            board_group_addlib(b, group, c)


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def merge_groups(b, group_to, group_from):
    onestone_from = group_is_onestone(b, group_from)
    onestone_to = group_is_onestone(b, group_to)
    if b.gi_libs[group_from] == 1:
        board_capturable_rm(b, group_from, b.gi_lib[group_from, 0], onestone_from)
    if b.gi_libs[group_to] < GROUP_KEEP_LIBS:
        for i in range(b.gi_libs[group_from]):
            lib_from = b.gi_lib[group_from, i]
            known = False
            for j in range(b.gi_libs[group_to]):
                if b.gi_lib[group_to, j] == lib_from:
                    known = True
                    break
            if known:
                continue
            if b.gi_libs[group_to] == 0:
                board_capturable_add(b, group_to, lib_from, onestone_to)
            elif b.gi_libs[group_to] == 1:
                board_capturable_rm(b, group_to, b.gi_lib[group_to, 0], onestone_to)
            b.gi_lib[group_to, b.gi_libs[group_to]] = lib_from
            b.gi_libs[group_to] += 1
            if b.gi_libs[group_to] >= GROUP_KEEP_LIBS:
                break
    if b.gi_libs[group_to] == 1:
        lib = b.gi_lib[group_to, 0]
        if b.gi_libs[group_from] == 1:
            i = 0
            for c in neighbors(b, lib):
                b.pat3[lib] |= np.uint32(b.g[c] == group_from) << np.uint32(16 + 3 - i)
                i += 1
    last_in_group = 0
    c = group_from
    c2 = b.p[c]
    while True:
        last_in_group = c
        b.g[c] = group_to
        c = c2
        if c == 0:
            break
        c2 = b.p[c2]
    b.p[last_in_group] = b.p[group_to]
    b.p[group_to] = group_from
    b.gi_lib[group_from, :] = 0
    b.gi_libs[group_from] = 0


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def new_group(b, coord):
    group = coord
    for c in neighbors(b, coord):
        if b.b[c] == S_NONE:
            b.gi_lib[group, b.gi_libs[group]] = c
            b.gi_libs[group] += 1
    b.g[coord] = group
    b.p[coord] = 0
    if b.gi_libs[group] == 1:
        board_capturable_add(b, group, b.gi_lib[group, 0], True)
    return group


# ---- playing moves

@njit(cache=True, nogil=True, inline="always", _nrt=False)
def play_one_neighbor(b, coord, color, other_color, c, group):
    ncolor = b.b[c]
    ngroup = b.g[c]
    b.n[c, color] += 1
    if not ngroup:
        return group
    board_group_rmlib(b, ngroup, coord)
    if ncolor == color and ngroup != group:
        if not group:
            group = ngroup
            add_to_group(b, group, c, coord)
        else:
            merge_groups(b, group, ngroup)
    elif ncolor == other_color:
        if board_group_captured(b, ngroup):
            board_group_capture(b, ngroup)
    return group


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def _shift_last_moves(b, coord, color):
    b.ints[LAST_MOVE4_COORD] = b.ints[LAST_MOVE3_COORD]
    b.ints[LAST_MOVE4_COLOR] = b.ints[LAST_MOVE3_COLOR]
    b.ints[LAST_MOVE3_COORD] = b.ints[LAST_MOVE2_COORD]
    b.ints[LAST_MOVE3_COLOR] = b.ints[LAST_MOVE2_COLOR]
    b.ints[LAST_MOVE2_COORD] = b.ints[LAST_MOVE_COORD]
    b.ints[LAST_MOVE2_COLOR] = b.ints[LAST_MOVE_COLOR]
    b.ints[LAST_MOVE_COORD] = coord
    b.ints[LAST_MOVE_COLOR] = color


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def board_play_outside(b, coord, color, f):
    other_color = stone_other(color)
    group = 0
    b.ints[FLEN] -= 1
    b.f[f] = b.f[b.ints[FLEN]]
    for c in neighbors(b, coord):
        group = play_one_neighbor(b, coord, color, other_color, c, group)
    b.b[coord] = color
    if not group:
        group = new_group(b, coord)
    _shift_last_moves(b, coord, color)
    b.ints[MOVES] += 1
    board_hash_update(b, coord, color)
    board_symmetry_update(b, coord)
    b.ints[KO_COORD] = PASS
    b.ints[KO_COLOR] = S_NONE
    return group


@njit(cache=True, nogil=True, _nrt=False)
def board_play_in_eye(b, coord, color, f):
    if color == b.ints[KO_COLOR] and coord == b.ints[KO_COORD]:
        return -1
    ko_coord = PASS
    ko_color = S_NONE
    captured_groups = 0
    for c in neighbors(b, coord):
        g = b.g[c]
        captured_groups += b.gi_libs[g] == 1
    if captured_groups == 0:
        return -1
    b.ints[FLEN] -= 1
    b.f[f] = b.f[b.ints[FLEN]]
    ko_caps = 0
    cap_at = PASS
    for c in neighbors(b, coord):
        b.n[c, color] += 1
        group = b.g[c]
        if not group:
            continue
        board_group_rmlib(b, group, coord)
        if board_group_captured(b, group):
            ko_caps += board_group_capture(b, group)
            cap_at = c
    if ko_caps == 1:
        ko_color = stone_other(color)
        ko_coord = cap_at
        b.ints[LAST_KO_COORD] = ko_coord
        b.ints[LAST_KO_COLOR] = ko_color
        b.ints[LAST_KO_AGE] = b.ints[MOVES]
    b.b[coord] = color
    group = new_group(b, coord)
    _shift_last_moves(b, coord, color)
    b.ints[MOVES] += 1
    board_hash_update(b, coord, color)
    board_hash_commit(b)
    board_symmetry_update(b, coord)
    b.ints[KO_COORD] = ko_coord
    b.ints[KO_COLOR] = ko_color
    return 1 if group else 0


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def board_play_f(b, coord, color, f):
    """Normalizes the argument types (one compiled ``_board_play_f``)."""
    return _board_play_f(b, np.int64(coord), np.int64(color), np.int64(f))


@njit(cache=True, nogil=True, _nrt=False)
def _board_play_f(b, coord, color, f):
    if not board_is_eyelike(b, coord, stone_other(color)):
        group = board_play_outside(b, coord, color, f)
        if board_group_captured(b, group):
            board_group_capture(b, group)
        board_hash_commit(b)
        return 0
    return board_play_in_eye(b, coord, color, f)


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def board_play(b, coord, color):
    """Normalizes the argument types (one compiled ``_board_play``)."""
    return _board_play(b, np.int64(coord), np.int64(color))


@njit(cache=True, nogil=True, _nrt=False)
def _board_play(b, coord, color):
    """Returns group id, 0 on allowed suicide, pass or resign, -1 on error."""
    if is_pass(coord) or is_resign(coord):
        if is_pass(coord) and b.ints[RULES] == RULES_SIMING:
            b.captures[stone_other(color)] += 1
        b.ints[KO_COORD] = PASS
        b.ints[KO_COLOR] = S_NONE
        _shift_last_moves(b, coord, color)
        return 0
    for f in range(b.ints[FLEN]):
        if b.f[f] == coord:
            return board_play_f(b, coord, color, f)
    return -1


@njit(cache=True, nogil=True, _nrt=False)
def board_undo(b):
    """Undo, supported only for pass moves."""
    if not is_pass(b.ints[LAST_MOVE_COORD]):
        return -1
    if b.ints[RULES] == RULES_SIMING:
        b.captures[stone_other(b.ints[LAST_MOVE_COLOR])] -= 1
    b.ints[LAST_MOVE_COORD] = b.ints[LAST_MOVE2_COORD]
    b.ints[LAST_MOVE_COLOR] = b.ints[LAST_MOVE2_COLOR]
    b.ints[LAST_MOVE2_COORD] = b.ints[LAST_MOVE3_COORD]
    b.ints[LAST_MOVE2_COLOR] = b.ints[LAST_MOVE3_COLOR]
    b.ints[LAST_MOVE3_COORD] = b.ints[LAST_MOVE4_COORD]
    b.ints[LAST_MOVE3_COLOR] = b.ints[LAST_MOVE4_COLOR]
    if b.ints[LAST_KO_AGE] == b.ints[MOVES]:
        b.ints[KO_COORD] = b.ints[LAST_KO_COORD]
        b.ints[KO_COLOR] = b.ints[LAST_KO_COLOR]
    return 0


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def board_try_random_move(b, color, f):
    """Without a permit function; returns the move played or -3 if none."""
    coord = b.f[f]
    if board_is_one_point_eye(b, coord, color) or not board_is_valid_play(b, color, coord):
        return -3
    if board_play_f(b, coord, color, f) >= 0:
        return coord
    return -3


@njit(cache=True, nogil=True, _nrt=False)
def board_play_random(b, color, rng):
    """A random move that never fills an own eye; pass if none. Returns the move."""
    if b.ints[FLEN] != 0:
        base = np.int64(fast.fast_random(rng, b.ints[FLEN]))
        for f in range(base, b.ints[FLEN]):
            coord = board_try_random_move(b, color, f)
            if coord != -3:
                return coord
        for f in range(0, base):
            coord = board_try_random_move(b, color, f)
            if coord != -3:
                return coord
    board_play(b, PASS, color)
    return PASS


def board_handicap_stone(b, x, y):
    coord = x + y * int(board_size(b))
    board_play(b, coord, S_BLACK)
    b.ints[MOVES] += 1  # simulate white passing
    return coord


def board_handicap(b, stones):
    """Place handicap stones; returns their coordinates (C prints them)."""
    size = int(board_size(b))
    margin = 3 + (size >= 13)
    low, mid, high = margin, size // 2, size - 1 - margin
    places = [(low, low), (high, high), (low, high), (high, low), (low, mid), (high, mid),
              (mid, low), (mid, high), (mid, mid)]
    b.ints[HANDICAP] = stones
    played = []
    if stones in (5, 7):
        played.append(board_handicap_stone(b, mid, mid))
        stones -= 1
    for i in range(stones):
        played.append(board_handicap_stone(b, places[i][0], places[i][1]))
    return played


# ---- inline checks of board.h

@njit(cache=True, nogil=True, _nrt=False)
def board_is_eyelike(b, coord, eye_color):
    return b.n[coord, eye_color] + b.n[coord, S_OFFBOARD] == 4


@njit(cache=True, nogil=True, _nrt=False)
def board_is_valid_play(b, color, coord):
    """Group suicides allowed."""
    if b.b[coord] != S_NONE:
        return False
    if not board_is_eyelike(b, coord, stone_other(color)):
        return True
    if b.ints[KO_COORD] == coord and b.ints[KO_COLOR] == color:
        return False
    groups_in_atari = 0
    for c in neighbors(b, coord):
        g = b.g[c]
        groups_in_atari += b.gi_libs[g] == 1
    return groups_in_atari != 0


@njit(cache=True, nogil=True, _nrt=False)
def board_is_valid_play_no_suicide(b, color, coord):
    if b.b[coord] != S_NONE:
        return False
    if immediate_liberty_count(b, coord) >= 1:
        return True
    if board_is_eyelike(b, coord, stone_other(color)) and b.ints[KO_COORD] == coord \
            and b.ints[KO_COLOR] == color:
        return False
    for c in neighbors(b, coord):
        if b.b[c] == stone_other(color) and b.gi_libs[b.g[c]] == 1:
            return True
    for c in neighbors(b, coord):
        if b.b[c] == color and b.gi_libs[b.g[c]] > 1:
            return True
    return False


@njit(cache=True, nogil=True, _nrt=False)
def board_is_valid_move(b, coord, color):
    return board_is_valid_play(b, color, coord)


@njit(cache=True, nogil=True, _nrt=False)
def board_playing_ko_threat(b):
    return not is_pass(b.ints[KO_COORD])


@njit(cache=True, nogil=True, _nrt=False)
def board_get_atari_neighbor(b, coord, group_color):
    for c in neighbors(b, coord):
        g = b.g[c]
        if g and b.b[c] == group_color and b.gi_libs[g] == 1:
            return g
    return 0


@njit(cache=True, nogil=True, _nrt=False)
def board_safe_to_play(b, coord, color):
    libs = immediate_liberty_count(b, coord)
    if libs > 1:
        return True
    onelib = -1
    for c in neighbors(b, coord):
        if b.b[c] == stone_other(color) and b.gi_libs[b.g[c]] == 1:
            return True
        if b.b[c] != color:
            continue
        g = b.g[c]
        if b.gi_libs[g] == 1:
            continue
        if b.gi_libs[g] == 2:
            if libs > 0:
                return True
            if onelib >= 0 and c != onelib:
                return True
            onelib = board_group_other_lib(b, g, c)
            continue
        return True
    return False


@njit(cache=True, nogil=True, _nrt=False)
def group_stone_count(b, group, max_count):
    n = 0
    c = group
    c2 = b.p[c]
    while True:
        n += 1
        if n >= max_count:
            return max_count
        c = c2
        if c == 0:
            break
        c2 = b.p[c2]
    return n


@njit(cache=True, nogil=True, _nrt=False)
def board_is_false_eyelike(b, coord, eye_color):
    """Two enemy stones diagonally in the middle of the board, or one at the edge."""
    other = stone_other(eye_color)
    enemies = 0
    offboard = 0
    c = coord
    for i in range(4):
        c += b.dnei[i]
        color = b.b[c]
        enemies += color == other
        offboard += color == S_OFFBOARD
    # C: color_diag_libs[stone_other(eye_color)] += !!color_diag_libs[S_OFFBOARD]
    return enemies + (offboard != 0) >= 2


@njit(cache=True, nogil=True, _nrt=False)
def board_is_one_point_eye(b, coord, eye_color):
    return board_is_eyelike(b, coord, eye_color) and not board_is_false_eyelike(b, coord, eye_color)


@njit(cache=True, nogil=True, _nrt=False)
def board_get_one_point_eye(b, coord):
    if board_is_one_point_eye(b, coord, S_WHITE):
        return S_WHITE
    if board_is_one_point_eye(b, coord, S_BLACK):
        return S_BLACK
    return S_NONE


# ---- scoring

@njit(cache=True, nogil=True, _nrt=False)
def _komi_score(b, white, black):
    """komi + handicap + white - black, in float like C (floating_t)."""
    score = b.komi[0]
    if b.ints[RULES] != RULES_SIMING:
        score = np.float32(score + np.float32(b.ints[HANDICAP]))
    score = np.float32(score + np.float32(white))
    return np.float32(score - np.float32(black))


@njit(cache=True, nogil=True, _nrt=False)
def board_fast_score(b):
    """Stones and 1-point eyes; positive: white wins."""
    white = 0
    black = 0
    for c in range(board_size2(b)):
        color = b.b[c]
        if color == S_NONE and b.ints[RULES] != RULES_STONES_ONLY:
            color = board_get_one_point_eye(b, c)
        white += color == S_WHITE
        black += color == S_BLACK
    return _komi_score(b, white, black)


@njit(cache=True, nogil=True, _nrt=True)
def board_tromp_taylor_iter(b, ownermap):
    needs_update = False
    for fi in range(b.ints[FLEN]):
        c = b.f[fi]
        if b.ints[RULES] == RULES_STONES_ONLY:
            ownermap[c] = 3
        if ownermap[c] == 3:
            continue
        nei = np.zeros(4, np.int64)
        for n in neighbors(b, c):
            nei[ownermap[n]] += 1
        if (nei[1] and nei[2]) or nei[3]:
            ownermap[c] = 3
            for n in neighbors(b, c):
                if b.b[n] == S_NONE:
                    ownermap[n] = 3
            needs_update = True
            continue
        if not ownermap[c] and (nei[1] or nei[2]):
            newowner = 1 if nei[1] else 2
            ownermap[c] = newowner
            for n in neighbors(b, c):
                if b.b[n] == S_NONE and not ownermap[n]:
                    ownermap[n] = newowner
            needs_update = True
            continue
    return needs_update


@njit(cache=True, nogil=True, _nrt=True)
def board_official_score(b, dead_groups, num_dead):
    """Tromp-Taylor counting, the groups ``dead_groups[:num_dead]`` taken as dead.

    C passes a ``struct move_queue``; here its array and length.
    """
    size2 = board_size2(b)
    ownermap = np.zeros(size2, np.int64)
    s = np.zeros(4, np.int64)
    o = (0, 1, 2, 0)
    for c in range(size2):
        ownermap[c] = o[b.b[c]]
        s[b.b[c]] += 1
    for i in range(num_dead):
        c = dead_groups[i]
        c2 = b.p[c]
        while True:
            color = b.b[c]
            ownermap[c] = o[stone_other(color)]
            s[color] -= 1
            s[stone_other(color)] += 1
            c = c2
            if c == 0:
                break
            c2 = b.p[c2]
    if not s[S_BLACK] and not s[S_WHITE]:
        return b.komi[0]
    while board_tromp_taylor_iter(b, ownermap):
        pass
    scores = np.zeros(S_MAX, np.int64)
    for c in range(size2):
        if ownermap[c] == 3:
            continue
        scores[ownermap[c]] += 1
    return _komi_score(b, scores[S_WHITE], scores[S_BLACK])


def board_print(b):
    """A plain text picture of the board (for debugging; not C's layout)."""
    size = int(board_size(b))
    rows = []
    for y in range(size - 2, 0, -1):
        rows.append(f"{y:2d} " + " ".join(".XO#"[b.b[x + y * size]] for x in range(1, size - 1)))
    rows.append("   " + " ".join("ABCDEFGHJKLMNOPQRST"[:size - 2]))
    return "\n".join(rows)


# pachi-numba: a translation of Pachi (GPL-2.0-only), October 2026; see NOTICE.
"""Translation of Pachi's ``pattern3.h`` and ``pattern3.c``: 3x3 patterns.

``pattern3_hash`` (the code of a point) is in ``board.py``, which keeps the
codes up to date. A ``struct pattern3s`` (an open-addressing hash table) is
a named tuple of two arrays, ``pattern`` and ``value``. Building the table
runs once, in plain Python. C first tries to read ``moggy.patterns`` from
the working directory; this translation always uses the given patterns.
"""

from collections import namedtuple

import numpy as np
from numba import njit

from .stone import S_BLACK, S_MAX, S_NONE, S_OFFBOARD, S_WHITE

PATTERN3_HASH_BITS = 19
PATTERN3_HASH_SIZE = 1 << PATTERN3_HASH_BITS
PATTERN3_HASH_MASK = PATTERN3_HASH_SIZE - 1

Pattern3s = namedtuple("Pattern3s", ["pattern", "value"])

_MASK64 = (1 << 64) - 1
_ATARIBITS = (-1, 0, -1, 1, 2, -1, 3, -1)


def _p3hashes_init():
    """Zobrist hashes of the points: [point][is_atari][color]; tuned for 11482 collisions."""
    hashes = np.zeros((8, 2, S_MAX), np.uint64)
    h = 0x35373C
    for i in range(8):
        for a in range(2):
            h = (h * 16803 - 7) & _MASK64
            hashes[i, a, S_NONE] = h
            h = (h * 16805 - 2) & _MASK64
            hashes[i, a, S_BLACK] = h
            h = (h * 16807 - 11) & _MASK64
            hashes[i, a, S_WHITE] = h
            h = (h * 16809 + 7) & _MASK64
            hashes[i, a, S_OFFBOARD] = h
    return hashes


P3HASHES = _p3hashes_init()
_ATARIBITS_ARRAY = np.array(_ATARIBITS, np.int64)


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def hash3_to_hash(pat):
    h = np.uint64(0)
    pat = np.int64(pat)
    for i in range(8):
        is_atari = (pat >> (16 + _ATARIBITS_ARRAY[i])) & 1 if _ATARIBITS_ARRAY[i] >= 0 else 0
        h ^= P3HASHES[i, is_atari, (pat >> (i * 2)) & 3]
    return h


@njit(cache=True, nogil=True, inline="always", _nrt=False)
def pattern3_move_here(p, b, coord, color):
    """Whether a pattern matches the move; returns (matches, pattern index)."""
    pat = b.pat3[coord]
    h = hash3_to_hash(pat)
    mask = np.uint64(PATTERN3_HASH_MASK)
    while p.pattern[h & mask] != pat and p.value[h & mask] != 0:
        h += np.uint64(1)
    if p.value[h & mask] & color:
        return True, np.int64(p.value[h & mask] >> 2)
    return False, np.int64(0)


@njit(cache=True, nogil=True, _nrt=False)
def pattern3_reverse(pat):
    """Reverse color assignment - achieved by swapping odd and even bits."""
    pat = np.uint32(pat)
    return ((pat >> 1) & np.uint32(0x5555)) | ((pat & np.uint32(0x5555)) << 1) \
        | (pat & np.uint32(0xF0000))


def _pattern_record(p, pi, pat, fixed_color):
    h = int(hash3_to_hash(pat))
    while p.pattern[h & PATTERN3_HASH_MASK] != pat and p.value[h & PATTERN3_HASH_MASK] != 0:
        h += 1
    p.pattern[h & PATTERN3_HASH_MASK] = pat
    p.value[h & PATTERN3_HASH_MASK] = ((fixed_color if fixed_color else 3) | (pi << 2)) & 0xFF


def _pat_vmirror(pat):
    """V mirror pattern; reverse order of 3-2-3 color chunks and 1-2-1 atari chunks."""
    return ((pat & 0xFC00) >> 10) | (pat & 0x03C0) | ((pat & 0x003F) << 10) \
        | ((pat & 0x80000) >> 3) | (pat & 0x60000) | ((pat & 0x10000) << 3)


def _rev3(p):
    return (p >> 4) | (p & 0xC) | ((p & 0x3) << 4)


def _rev2(p):
    return (p >> 2) | ((p & 0x3) << 2)


def _pat_hmirror(pat):
    """H mirror pattern; reverse order of 2-bit values within the chunks,
    and the 2-bit middle atari chunk."""
    return (_rev3((pat & 0xFC00) >> 10) << 10) | (_rev2((pat & 0x03C0) >> 6) << 6) \
        | _rev3(pat & 0x003F) | ((pat & 0x20000) << 1) | ((pat & 0x40000) >> 1) | (pat & 0x90000)


def _pat_90rot(pat):
    """Rotate by 90 degrees."""
    vals = [(pat >> (i * 2)) & 0x3 for i in range(8)]
    vals2 = [vals[5], vals[3], vals[0], vals[6], vals[1], vals[7], vals[4], vals[2]]
    p2 = 0
    for i in range(8):
        p2 |= vals2[i] << (i * 2)
    avals = [(pat >> (16 + i)) & 0x1 for i in range(4)]
    avals2 = [avals[1], avals[3], avals[0], avals[2]]
    for i in range(4):
        p2 |= avals2[i] << (16 + i)
    return p2


def pattern3_transpose(pat):
    """All 8 transpositions of the pattern."""
    vm, hm = _pat_vmirror, _pat_hmirror
    return [pat, vm(pat), hm(pat), vm(hm(pat)), _pat_90rot(pat), _pat_90rot(vm(pat)),
            _pat_90rot(hm(pat)), _pat_90rot(vm(hm(pat)))]


# Expansions of the wildcards, in the order of C's recursion.
_WILDCARDS = {
    "?": ".XO#", "x": ".O#", "o": ".X#", "y": ".|O#", "q": ".@X#", "=": ".YO#", "0": ".QX#",
}


def _pattern_gen(p, pi, pat, src, position, fixed_color):
    """``src`` is the list of the 9 characters; ``position`` the C pointer offset."""
    while position < 9:
        srclen = 9 - position
        if srclen == 5:
            position += 1
            continue
        patofs = (srclen - 1 if srclen > 5 else srclen) - 1
        char = src[position]
        if char in _WILDCARDS or char in "XO":
            if char in _WILDCARDS:
                options = _WILDCARDS[char]
            elif char == "X":
                options = "Y|" if _ATARIBITS[patofs] >= 0 else "Y"
            else:
                options = "Q@" if _ATARIBITS[patofs] >= 0 else "Q"
            for option in options:
                src[position] = option
                _pattern_gen(p, pi, pat, src, position, fixed_color)
            src[position] = char
            return
        if char == "Y":
            pat |= S_BLACK << (patofs * 2)
        elif char == "Q":
            pat |= S_WHITE << (patofs * 2)
        elif char == "|":
            pat |= (S_BLACK << (patofs * 2)) | (1 << (16 + _ATARIBITS[patofs]))
        elif char == "@":
            pat |= (S_WHITE << (patofs * 2)) | (1 << (16 + _ATARIBITS[patofs]))
        elif char == "#":
            pat |= S_OFFBOARD << (patofs * 2)
        position += 1

    # Original pattern, all transpositions and rotations
    for transposed in pattern3_transpose(pat):
        _pattern_record(p, pi, transposed, fixed_color)
        # C flips fixed_color here, and the flip carries over to the next
        # transposition.
        if fixed_color:
            fixed_color = 2 - (fixed_color == 2)
        _pattern_record(p, pi, int(pattern3_reverse(transposed)), fixed_color)


def pattern3s_init(src):
    """Build the table from patterns of 10 characters (9 points and the color)."""
    p = Pattern3s(np.zeros(PATTERN3_HASH_SIZE, np.uint32), np.zeros(PATTERN3_HASH_SIZE, np.uint8))
    for i, line in enumerate(src):
        fixed_color = {"X": S_BLACK, "O": S_WHITE}.get(line[9:10], 0)
        _pattern_gen(p, i, 0, list(line[:9]), 0, fixed_color)
    return p

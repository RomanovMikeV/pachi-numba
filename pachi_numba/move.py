# pachi-numba: a translation of Pachi (GPL-2.0-only), October 2026; see NOTICE.
"""Translation of Pachi's ``move.h`` and ``move.c``.

A coordinate (``coord_t``) is an index into the board with its margin of
``S_OFFBOARD`` points: ``x + y * size``, where ``size`` is the side of the
board *with* the margin (Pachi's ``board_size(board)``, 11 for 9x9), ``x``
and ``y`` count from 1 and ``y`` grows upward. C looks ``coord_x`` and
``coord_y`` up in a table of the board; here they are computed, which gives
the same numbers.

C allocates coordinates on the heap (``coord_init``, ``coord_copy``,
``coord_pass``, ``coord_resign``, ``coord_done``); plain integers replace
them.
"""

from numba import njit

# static coord_t pass = -1, resign = -2 ("pass" is a keyword in Python)
PASS = -1
RESIGN = -2

_LETTERS = "abcdefghjklmnopqrstuvwxyz"  # C: asdf[], no "i"


@njit(cache=True, nogil=True, _nrt=False)
def is_pass(c):
    return c == PASS


@njit(cache=True, nogil=True, _nrt=False)
def is_resign(c):
    return c == RESIGN


@njit(cache=True, nogil=True, _nrt=False)
def coord_xy(size, x, y):
    return x + y * size


@njit(cache=True, nogil=True, _nrt=False)
def coord_x(c, size):
    return c % size


@njit(cache=True, nogil=True, _nrt=False)
def coord_y(c, size):
    return c // size


@njit(cache=True, nogil=True, _nrt=False)
def coord_dx(c1, c2, size):
    return coord_x(c1, size) - coord_x(c2, size)


@njit(cache=True, nogil=True, _nrt=False)
def coord_dy(c1, c2, size):
    return coord_y(c1, size) - coord_y(c2, size)


@njit(cache=True, nogil=True, _nrt=False)
def coord_is_adjecent(c1, c2, size):
    return abs(c1 - c2) == 1 or abs(c1 - c2) == size


@njit(cache=True, nogil=True, _nrt=False)
def coord_is_8adjecent(c1, c2, size):
    return abs(c1 - c2) == 1 or abs(abs(c1 - c2) - size) < 2


@njit(cache=True, nogil=True, _nrt=False)
def coord_quadrant(c, size):
    """0 1 / 2 3; the middle belongs to the lower quadrants."""
    return (coord_x(c, size) > size // 2) + 2 * (coord_y(c, size) > size // 2)


def coord2bstr(c, size):
    """``"C7"``, ``"pass"`` or ``"resign"`` (C writes into a buffer)."""
    if is_pass(c):
        return "pass"
    if is_resign(c):
        return "resign"
    return f"{_LETTERS[coord_x(c, size) - 1].upper()}{coord_y(c, size)}"


coord2str = coord2bstr  # C: the same string, allocated
coord2sstr = coord2bstr  # C: the same string, in a static buffer


def _atoi(text):
    """C ``atoi``: the leading digits (after an optional sign), else 0."""
    digits = ""
    for i, character in enumerate(text):
        if character.isdigit() or (i == 0 and character in "+-"):
            digits += character
        else:
            break
    return int(digits) if digits not in ("", "+", "-") else 0


def str2scoord(text, size):
    """No sanity checking, like C."""
    if text.lower() == "pass":
        return PASS
    if text.lower() == "resign":
        return RESIGN
    xc = text[0].lower()
    return ord(xc) - ord("a") - (xc > "i") + 1 + _atoi(text[1:]) * size


str2coord = str2scoord  # C: the same value, allocated

# pachi-numba: a translation of Pachi (GPL-2.0-only), October 2026; see NOTICE.
"""Translation of Pachi's ``random.h`` and ``random.c``.

Pachi keeps one random seed per thread (``static __thread unsigned long
pmseed``). Here the seed is an explicit state: an array of one ``uint64``,
made by ``make_state``. Every function takes it as its first argument, so
that every search thread can have its own, as in C. ``unsigned long`` is 64
bits on Linux and macOS, so the state is ``uint64`` and wraps around at
2**64 like the C value.
"""

import numpy as np
from numba import njit

# The seed of every thread before ``fast_srandom`` (C: ``pmseed = 29264``).
INITIAL_SEED = 29264


def make_state(seed=INITIAL_SEED):
    """A new random state, as a new thread of Pachi starts with."""
    return np.array([seed], dtype=np.uint64)


@njit(cache=True, nogil=True, _nrt=False)
def fast_srandom(state, seed):
    state[0] = np.uint64(seed)


@njit(cache=True, nogil=True, _nrt=False)
def fast_getseed(state):
    return state[0]


@njit(cache=True, nogil=True, _nrt=False)
def fast_random(state, max_value):
    """A number in ``0 .. max_value - 1``; only 16-bit numbers (C: uint16_t)."""
    pmseed = (state[0] * np.uint64(1103515245) + np.uint64(12345)) & np.uint64(0x7FFFFFFF)
    state[0] = pmseed
    return np.uint16(((pmseed & np.uint64(0xFFFF)) * np.uint64(max_value)) >> np.uint64(16))


@njit(cache=True, nogil=True, _nrt=False)
def fast_irandom(state, max_value):
    """Like ``fast_random``, for larger numbers (C: uint32_t)."""
    if max_value <= 65536:
        return np.uint32(fast_random(state, max_value))
    himax = (max_value - 1) // 65536
    hi = fast_random(state, himax + 1)
    low_max = 65536 if hi < himax else max_value % 65536
    return np.uint32((np.uint32(hi) << np.uint32(16)) | np.uint32(fast_random(state, low_max)))


@njit(cache=True, nogil=True, _nrt=False)
def fast_frandom(state):
    """A number in [0, 1) (C: float).

    C builds the float from bits: ``((pmseed & 0x007fffff) - 1) | 0x3f800000``
    read as an IEEE float is ``1 + (m - 1) / 2**23`` for ``m >= 1``, and minus
    ``1.0f`` that is ``(m - 1) / 2**23`` exactly. For ``m == 0`` the
    subtraction wraps around and the bits are a NaN.
    """
    pmseed = state[0] * np.uint64(16807)
    state[0] = pmseed
    mantissa = pmseed & np.uint64(0x007FFFFF)
    if mantissa == 0:
        return np.float32(np.nan)
    return np.float32(mantissa - np.uint64(1)) / np.float32(8388608.0)


@njit(cache=True, nogil=True, _nrt=True)
def make_state_from(seed):
    """``make_state`` inside compiled code (C: ``fast_srandom`` in a new thread)."""
    state = np.zeros(1, np.uint64)
    state[0] = np.uint64(seed)
    return state

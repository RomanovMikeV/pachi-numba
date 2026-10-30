# pachi-numba: a translation of Pachi (GPL-2.0-only), October 2026; see NOTICE.
"""Translation of Pachi's ``mq.h`` (move queues) and ``fixp.h``.

A ``struct move_queue`` is a named tuple of arrays: ``moves`` holds the
length in ``moves[0]``. The random functions take the random state of the
calling thread (see ``random.py``).
"""

from collections import namedtuple

import numpy as np
from numba import njit

from . import random as fast
from .move import PASS

MQL = 512  # XXX: On larger board this might not be enough.

MoveQueue = namedtuple("MoveQueue", ["moves", "move", "tag"])

# fixp.h: unsigned fixed-point numbers with 15 fractional bits.
FIXP_BITS = 15
FIXP_SCALE = 1 << FIXP_BITS


@njit(cache=True, nogil=True, _nrt=True)
def mq_new():
    # Only the first moves[0] entries are ever read.
    return MoveQueue(np.zeros(1, np.int64), np.empty(MQL, np.int32), np.empty(MQL, np.uint8))


@njit(cache=True, nogil=True, _nrt=False)
def mq_clear(q):
    q.moves[0] = 0


@njit(cache=True, nogil=True, _nrt=False)
def mq_pick(q, rng):
    """Pick a random move from the queue."""
    return q.move[fast.fast_random(rng, q.moves[0])] if q.moves[0] else PASS


@njit(cache=True, nogil=True, _nrt=False)
def mq_add(q, c, tag):
    assert q.moves[0] < MQL
    q.tag[q.moves[0]] = tag
    q.move[q.moves[0]] = c
    q.moves[0] += 1


@njit(cache=True, nogil=True, _nrt=False)
def mq_append(qd, qs):
    """Cat two queues together."""
    assert qd.moves[0] + qs.moves[0] < MQL
    n = qs.moves[0]
    qd.tag[qd.moves[0]:qd.moves[0] + n] = qs.tag[:n]
    qd.move[qd.moves[0]:qd.moves[0] + n] = qs.move[:n]
    qd.moves[0] += n


@njit(cache=True, nogil=True, _nrt=False)
def mq_nodup(q):
    """If the last move is among the three before it, merge the two."""
    for i in range(1, 4):
        if q.moves[0] <= i:
            return
        if q.move[q.moves[0] - 1 - i] == q.move[q.moves[0] - 1]:
            q.tag[q.moves[0] - 1 - i] |= q.tag[q.moves[0] - 1]
            q.moves[0] -= 1
            return


@njit(cache=True, nogil=True, _nrt=False)
def double_to_fixp(n):
    return np.uint64(n * FIXP_SCALE)


@njit(cache=True, nogil=True, _nrt=False)
def fixp_to_double(n):
    return np.float64(n) / FIXP_SCALE


@njit(cache=True, nogil=True, _nrt=False)
def mq_gamma_pick(q, gammas, rng):
    """Pick a move with the probability proportional to its gamma."""
    if not q.moves[0]:
        return PASS
    total = np.uint64(0)
    for i in range(q.moves[0]):
        total += gammas[i]
    if not total:
        return PASS
    stab = np.uint64(fast.fast_irandom(rng, total))
    for i in range(q.moves[0]):
        if stab < gammas[i]:
            return q.move[i]
        stab -= gammas[i]
    raise AssertionError("mq_gamma_pick: no move")


@njit(cache=True, nogil=True, _nrt=False)
def mq_gamma_add(q, gammas, c, gamma, tag):
    mq_add(q, c, tag)
    gammas[q.moves[0] - 1] = double_to_fixp(gamma)

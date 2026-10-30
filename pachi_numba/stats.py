# pachi-numba: a translation of Pachi (GPL-2.0-only), October 2026; see NOTICE.
"""Translation of Pachi's ``stats.h``: move statistics.

A ``struct move_stats`` is a pair (``value``: float32, ``playouts``: int)
kept in two arrays at the same index. ``floating_t`` is float in C, and
every operation is done in float32, as C does (a float times an int is a
float in C, a float64 in numba). Division by zero gives inf or NaN as in C
(``error_model="numpy"``): Pachi adds results with zero weight, which turns
a value with zero playouts into NaN.
"""

import numpy as np
from numba import njit


@njit(cache=True, nogil=True, error_model="numpy", _nrt=False)
def f32(x):
    return np.float32(x)


@njit(cache=True, nogil=True, error_model="numpy", _nrt=False)
def stats_add_result(values, playouts, i, result, add_playouts):
    """Add a result to the stats ``i``."""
    s_playouts = playouts[i] + add_playouts
    s_value = values[i]
    delta = f32(f32(f32(result) - s_value) * f32(add_playouts))
    s_value = f32(s_value + f32(delta / f32(s_playouts)))
    values[i] = s_value
    playouts[i] = s_playouts


@njit(cache=True, nogil=True, error_model="numpy", _nrt=False)
def stats_rm_result(values, playouts, i, result, rm_playouts):
    """Remove a result from the stats ``i``."""
    if playouts[i] > rm_playouts:
        s_playouts = playouts[i] - rm_playouts
        s_value = values[i]
        delta = f32(f32(s_value - f32(result)) * f32(rm_playouts))
        s_value = f32(s_value + f32(delta / f32(s_playouts)))
        values[i] = s_value
        playouts[i] = s_playouts
    else:
        playouts[i] = 0


@njit(cache=True, nogil=True, error_model="numpy", _nrt=False)
def stats_merge(values, playouts, i, src_value, src_playouts):
    """Merge the stats (``src_value``, ``src_playouts``) into the stats ``i``."""
    if src_playouts:
        playouts[i] += src_playouts
        values[i] = f32(values[i] + f32(f32(f32(f32(src_value) - values[i]) * f32(src_playouts))
                                        / f32(playouts[i])))


@njit(cache=True, nogil=True, error_model="numpy", _nrt=False)
def stats_reverse_parity(values, i):
    values[i] = f32(f32(1) - values[i])

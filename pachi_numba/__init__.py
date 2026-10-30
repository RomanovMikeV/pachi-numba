# pachi-numba: a translation of Pachi (GPL-2.0-only), October 2026; see NOTICE.
"""Pachi, the Go engine, rebuilt in pure Python (compiled with numba) for education.

A line-by-line translation of Pachi 11.99 (the version in OpenAI's
pachi-py 0.0.21) that plays the same moves as the C original, under the
GNU General Public License version 2, like Pachi. Start with
``pachi_numba.player.PachiPlayer``; ``pachi_numba.parallel.map_games``
runs several games at once.
"""

__version__ = "0.1.1"

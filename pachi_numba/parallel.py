# pachi-numba: a translation of Pachi (GPL-2.0-only), October 2026; see NOTICE.
"""Several independent games at once, in threads.

Every ``PachiPlayer`` owns its board, search tree and random state, and the
search runs in compiled code that releases the GIL (``nogil=True``). So
games in different threads run on different cores, and each one plays
exactly the moves it plays alone, in any order of execution. A search
within one move stays single-threaded (several search threads would not
give reproducible moves, in C either).
"""

import concurrent.futures
import os


def map_games(play_game, arguments, max_workers=None):
    """Run ``play_game(argument)`` for every argument, several at a time.

    Parameters
    ----------
    play_game : callable
        Plays one game; it creates its own ``PachiPlayer`` objects (players
        must not be shared between threads).
    arguments : iterable
        One argument per game, e.g. its seed.
    max_workers : int, optional
        Number of threads; by default the number of CPUs.

    Returns
    -------
    list
        The results of ``play_game``, in the order of ``arguments``.

    Examples
    --------
    >>> def margin(seed):
    ...     player = PachiPlayer(size=9, komi=7.5, seed=seed)
    ...     ...  # play the game
    ...     return player.black_margin()
    >>> margins = map_games(margin, range(10))
    """
    arguments = list(arguments)
    if max_workers is None:
        max_workers = os.cpu_count() or 1
    max_workers = max(1, min(max_workers, len(arguments)))
    if max_workers == 1:
        return [play_game(argument) for argument in arguments]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(play_game, arguments))

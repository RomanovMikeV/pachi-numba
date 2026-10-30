# pachi-numba

**The Go engine [Pachi](https://github.com/pasky/pachi), refactored and rebuilt
in pure Python for educational purposes.**

pachi-numba is a line-by-line translation of Pachi's C code to Python. 

Features:
- Python 3 support
- Pip installable on any platform
- No compilation required
- Numba speedup (the performance is close to C code)
- Parallel games
- Exact correspondence to the original code

The Python code is compiled with [numba](https://numba.pydata.org/), so it runs
at close to C speed, and it installs with `pip` alone: no C compiler, no
build step.

It is adapted to Python 3 (3.10 and later): the Python binding
it comes from, OpenAI's [pachi-py](https://github.com/openai/pachi-py), no
longer builds on current Python versions.

Pachi is a strong Monte Carlo tree search (UCT with RAVE) engine. This
package keeps its algorithm exactly: with the same random seed it plays the
same moves as the C original. The code is meant to be read, studied and
used as an opponent, for example to rate agents trained by reinforcement
learning.

## Quick start

```bash
pip install git+https://github.com/RomanovMikeV/pachi-numba
```

```python
import random

from pachi_numba.player import PachiPlayer

pachi = PachiPlayer(size=9, komi=7.5, seed=1)
opponent = random.Random(0)
color, passes = 1, 0  # 1 is black, 2 is white
while passes < 2:
    if color == 1:  # Pachi plays black ...
        move = pachi.genmove(color, playouts=1000)
    else:  # ... against random moves
        empty = [point for point in range(81) if pachi.stone(point) == 0]
        opponent.shuffle(empty)
        move = next((point for point in empty if pachi.play(point, color)), -1)
        if move == -1:
            pachi.play(-1, color)
    if move == -2:  # Pachi resigned
        break
    passes = passes + 1 if move == -1 else 0
    color = 3 - color
print("black's margin:", pachi.black_margin())
```

- Points are numbered row by row from the top-left corner (`0` is A9 on
  9x9, `80` is J1); `-1` is a pass and `-2` a resignation.
- `genmove(color, playouts)` lets Pachi choose and play a move after that
  many simulations (at least 500, as in Pachi); `play(point, color)` plays
  a move of the opponent (returns `False` if it is illegal);
  `black_margin()` is black's area minus white's area and komi, with the
  groups Pachi considers dead removed; `stone(point)` is 0, 1 or 2.
- The first use compiles the engine, which takes about a minute and a
  half; numba caches the result, and later programs start in about a
  second.

## Parallel games

Every `PachiPlayer` has its own board, search tree and random numbers, and
the search releases Python's GIL. Games in different threads therefore run
on different cores, and each game plays exactly the moves it plays alone:

```python
from pachi_numba.parallel import map_games

def play_game(seed):
    pachi = PachiPlayer(size=9, komi=7.5, seed=seed)
    ...  # play the game as above
    return pachi.black_margin()

margins = map_games(play_game, range(12))  # one thread per CPU by default
```

12 games of 20 moves at 2000 playouts per move took 22.2 s one after
another, 5.8 s in 4 threads and 2.3 s in 12 threads (on 18 cores). The
search of a single move stays single-threaded: several search threads,
in C too, do not give reproducible moves.

## Speed

On 9x9, one thread, arm64 Linux, against the C original built from the same
sources:

| | pachi-numba | C Pachi |
|---|---|---|
| random games (light playouts) per second | 40,000 | 55,000 |
| heuristic games (moggy playouts) per second | 13,000 | 26,000 |
| one move at 2000 playouts | 0.12 s | 0.10 s |
| one move at 8000 playouts | 0.24 s | 0.15 s |

C's move times include its 100 ms polling of the search. The numba code
reaches this speed by three means:

- most compiled functions run without numba's reference counting
  (`_nrt=False`); `tools/set_nrt_flags.py` sets the flags from the call
  graph, and a test checks that they are up to date;
- the small functions of the hot path are inlined (`inline="always"`),
  because a call that is not inlined copies the board, a tuple of 20 arrays;
- heavy functions are split into a body compiled once and a small wrapper
  that fixes the argument types, which halves the compile time.

## What changed compared with Pachi

- **Language**: C to Python, compiled with numba; no C compiler is needed.
  Every module says which Pachi file it translates, and the code follows
  the C line by line, with Pachi's comments.
- **Scope**: Pachi's default engine (`uct`), as pachi-py runs it: one search
  thread, no pondering, a fixed number of playouts per move, the `moggy`
  and `light` playout policies, RAVE (UCB1-AMAF), priors, tree reuse
  between moves, pass, resignation and dead-group decisions. Left out:
  GTP, opening books, joseki dictionaries and pattern files (Pachi does not
  ship them), DCNN, the distributed engine, several search threads, local
  trees and other non-default options. Dynamic komi is only `none`, which
  is Pachi's default up to 9x9; larger boards are not supported yet.
- **Data**: C structs and pointers became named tuples of NumPy arrays
  (`Board`, `Tree`, ...), and tree nodes are numbered instead of linked by
  pointers. Small arrays that C keeps on the stack live in a per-board
  work area or are allocated once per game.
- **Control flow**: recursion became explicit stacks (numba cannot cache
  recursive functions); thread-local random numbers became one random
  state per engine.
- **Interface**: `PachiPlayer` (play, genmove, black_margin, stone), with the
  same point numbering as pachi-py, and `map_games` for parallel games.
- **Timing**: C's main thread checks the search every 100 ms. When a pass
  is the most visited move, it may run extra simulations during these
  checks, and it stops searches of 5000 or more simulations early when the
  game looks won. Both depend on timing; pachi-numba does what C does when
  a search ends before its first check.

## How it is checked

`reference/` holds Pachi's C sources and a small C interface. The tests
compile them and compare, with the same seeds: random numbers, the whole
board after every move, every tactical check at every point of many
positions, playout move sequences, prior values, and `genmove` over whole
games (also with outside moves, tree reuse and symmetry flips) and the
final score. The C reference is compiled with `-ffp-contract=off`, because
fused multiply-adds (the default of GCC on arm64 and of clang) round some
float operations differently, and then a C build can choose other moves.

```bash
pip install -e ".[test]"
pytest
```

The tests need a C compiler (`cc`) for the reference, the package itself
does not. When changing the code, delete numba's cache files (`*.nbi`,
`*.nbc`) or let the tests do it: numba checks only the file of a cached
function, not the modules it calls.

## Licence and credits

pachi-numba is a work based on Pachi and is distributed, like Pachi, under
the GNU General Public License version 2 (`LICENSE`). Pachi is by Petr
Baudiš, Jean-loup Gailly and the other authors listed in `CREDITS`; the
translated version is Pachi 11.99 as included in pachi-py 0.0.21. `NOTICE`
describes the origin and the changes.

"""Drop numba caches older than the newest source file of the package.

numba checks only the source file of the cached function itself, not the
modules it calls; a cache compiled against an older callee can crash.
"""

import pathlib

PACKAGE = pathlib.Path(__file__).resolve().parents[1] / "pachi_numba"


def _drop_stale_numba_caches():
    newest_source = max(path.stat().st_mtime for path in PACKAGE.rglob("*.py"))
    for cache in list(PACKAGE.rglob("*.nbi")) + list(PACKAGE.rglob("*.nbc")):
        if cache.stat().st_mtime < newest_source:
            cache.unlink(missing_ok=True)


_drop_stale_numba_caches()

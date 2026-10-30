"""The original C Pachi, built for the tests that compare the translation with it.

``load()`` compiles ``pachi_src`` and ``shim.c`` into ``_build`` on first use
(this needs ``cc``) and returns the library through ``ctypes``. Floating
point contraction is off (``-ffp-contract=off``): fused multiply-adds (the
default of GCC on arm64) round differently from numba and from x86-64.
"""

import concurrent.futures
import ctypes
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

DIRECTORY = pathlib.Path(__file__).resolve().parent
SOURCES = sorted((DIRECTORY / "pachi_src").rglob("*.c")) + [DIRECTORY / "shim.c"]
LIBRARY = DIRECTORY / "_build" / (
    "libpachi_reference.dylib" if sys.platform == "darwin" else "libpachi_reference.so"
)

_library = None


def _compile(source, objects):
    target = objects / (str(source.relative_to(DIRECTORY)).replace(os.sep, "_") + ".o")
    subprocess.run(
        ["cc", "-O2", "-std=gnu99", "-fcommon", "-fPIC", "-w", "-D_GNU_SOURCE", "-ffp-contract=off",
         "-I", str(DIRECTORY / "pachi_src"), "-c", str(source), "-o", str(target)],
        check=True,
    )
    return target


def _build():
    if shutil.which("cc") is None:
        raise RuntimeError("The reference C Pachi needs a C compiler (cc).")
    LIBRARY.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as objects:
        with concurrent.futures.ThreadPoolExecutor(os.cpu_count()) as pool:
            object_files = list(pool.map(lambda s: _compile(s, pathlib.Path(objects)), SOURCES))
        partial = LIBRARY.with_suffix(f".{os.getpid()}.tmp")
        subprocess.run(["cc", "-shared", "-o", str(partial), *map(str, object_files),
                        "-lm", "-lpthread"], check=True)
        os.replace(partial, LIBRARY)


def load():
    """Build (when needed) and load the reference library."""
    global _library
    if _library is not None:
        return _library
    newest = max(source.stat().st_mtime for source in SOURCES + [pathlib.Path(__file__)])
    if not LIBRARY.exists() or LIBRARY.stat().st_mtime < newest:
        _build()
    library = ctypes.CDLL(str(LIBRARY))
    library.pachi_new.restype = ctypes.c_void_p
    library.pachi_new.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_int]
    library.pachi_free.argtypes = [ctypes.c_void_p]
    library.pachi_seed.argtypes = [ctypes.c_ulong]
    library.pachi_play.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    library.pachi_genmove.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    library.pachi_black_margin.restype = ctypes.c_double
    library.pachi_black_margin.argtypes = [ctypes.c_void_p]
    library.pachi_stone.argtypes = [ctypes.c_void_p, ctypes.c_int]
    library.ref_fast_random.restype = ctypes.c_uint
    library.ref_fast_random.argtypes = [ctypes.c_uint]
    library.ref_fast_irandom.restype = ctypes.c_uint
    library.ref_fast_irandom.argtypes = [ctypes.c_uint]
    library.ref_fast_frandom.restype = ctypes.c_float
    library.ref_fast_getseed.restype = ctypes.c_ulong
    library.ref_stone_other.argtypes = [ctypes.c_int]
    library.ref_str2coord.argtypes = [ctypes.c_char_p, ctypes.c_int]
    library.ref_coord2str.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    _library = library
    return library


def board_functions(library):
    """Set the ctypes signatures of the board exports (the ``ref_board_*``)."""
    import numpy as np

    pointer = ctypes.c_void_p
    arrays = {
        "int64": np.ctypeslib.ndpointer(np.int64, flags="C"),
        "float32": np.ctypeslib.ndpointer(np.float32, flags="C"),
        "int8": np.ctypeslib.ndpointer(np.int8, flags="C"),
        "int32": np.ctypeslib.ndpointer(np.int32, flags="C"),
        "uint64": np.ctypeslib.ndpointer(np.uint64, flags="C"),
        "uint32": np.ctypeslib.ndpointer(np.uint32, flags="C"),
        "uint8": np.ctypeslib.ndpointer(np.uint8, flags="C"),
    }
    library.ref_board_new.restype = pointer
    library.ref_board_new.argtypes = [ctypes.c_int]
    library.ref_board_free.argtypes = [pointer]
    library.ref_board_clone.restype = pointer
    library.ref_board_clone.argtypes = [pointer]
    library.ref_board_set_komi.argtypes = [pointer, ctypes.c_float]
    library.ref_board_set_rules.argtypes = [pointer, ctypes.c_char_p]
    library.ref_board_set_rules_id.argtypes = [pointer, ctypes.c_int]
    library.ref_board_handicap.argtypes = [pointer, ctypes.c_int]
    library.ref_board_play.argtypes = [pointer, ctypes.c_int, ctypes.c_int]
    library.ref_board_play_random.argtypes = [pointer, ctypes.c_int]
    library.ref_board_undo.argtypes = [pointer]
    library.ref_board_fast_score.restype = ctypes.c_float
    library.ref_board_fast_score.argtypes = [pointer]
    library.ref_board_official_score.restype = ctypes.c_float
    library.ref_board_official_score.argtypes = [pointer, arrays["int32"], ctypes.c_int]
    library.ref_board_check.argtypes = [pointer, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    library.ref_board_dump.argtypes = [pointer] + [arrays[name] for name in (
        "int64", "int64", "float32", "int64", "int64", "int8", "int32", "int32", "int8",
        "uint64", "uint32", "uint8", "int32", "int32", "int32", "int32", "uint64", "uint64",
        "uint64")]
    return library


def tactics_functions(library):
    """Set the ctypes signatures of the pattern and tactics exports."""
    import numpy as np

    board_functions(library)
    pointer = ctypes.c_void_p
    i = ctypes.c_int
    ints = np.ctypeslib.ndpointer(np.int32, flags="C")
    queue = [ints, ints, ints]  # moves, tags, length
    library.ref_hash3_to_hash.restype = ctypes.c_ulong
    library.ref_hash3_to_hash.argtypes = [ctypes.c_uint]
    library.ref_pattern3_reverse.restype = ctypes.c_uint
    library.ref_pattern3_reverse.argtypes = [ctypes.c_uint]
    library.ref_pattern3_transpose.argtypes = [
        ctypes.c_uint, np.ctypeslib.ndpointer(np.uint32, flags="C")]
    library.ref_pattern3s_new.restype = pointer
    library.ref_pattern3s_new.argtypes = [ctypes.c_char_p, i]
    library.ref_pattern3s_dump.argtypes = [pointer, np.ctypeslib.ndpointer(np.uint32, flags="C"),
                                           np.ctypeslib.ndpointer(np.uint8, flags="C")]
    library.ref_pattern3_move_here.argtypes = [pointer, pointer, i, i]
    library.ref_is_bad_selfatari.argtypes = [pointer, i, i]
    library.ref_selfatari_cousin.argtypes = [pointer, i, i, ints]
    library.ref_can_countercapture.argtypes = [pointer, i, i, i, i, i] + queue
    library.ref_group_atari_check.argtypes = [ctypes.c_uint, pointer, i, i, i, i, i] + queue
    library.ref_can_atari_group.argtypes = [pointer, i, i, i, i, i] + queue
    library.ref_group_2lib_check.argtypes = [pointer, i, i, i, i, i] + queue
    library.ref_group_nlib_defense_check.argtypes = [pointer, i, i, i] + queue
    library.ref_is_ladder.argtypes = [pointer, i, i, i]
    library.ref_wouldbe_ladder.argtypes = [pointer, i, i, i, i]
    library.ref_nakade_point.argtypes = [pointer, i, i]
    library.ref_nakade_dead_shape.argtypes = [pointer, i, i]
    library.ref_board_stone_radar.argtypes = [pointer, i, i]
    library.ref_cfg_distances.argtypes = [pointer, i, ints, i]
    library.ref_coord_edge_distance.argtypes = [pointer, i]
    library.ref_coord_gridcular_distance.argtypes = [pointer, i, i]
    library.ref_board_effective_handicap.restype = ctypes.c_float
    library.ref_board_effective_handicap.argtypes = [pointer, i]
    library.ref_board_estimated_moves_left.argtypes = [pointer]
    library.ref_board_local_value.restype = ctypes.c_double
    library.ref_board_local_value.argtypes = [i, pointer, i, i]
    return library


def playout_functions(library):
    """Set the ctypes signatures of the policy and playout exports."""
    import numpy as np

    tactics_functions(library)
    pointer = ctypes.c_void_p
    i = ctypes.c_int
    ints = np.ctypeslib.ndpointer(np.int32, flags="C")
    uint8s = np.ctypeslib.ndpointer(np.uint8, flags="C")
    library.ref_moggy_new.restype = pointer
    library.ref_moggy_new.argtypes = [ctypes.c_char_p, pointer]
    library.ref_light_new.restype = pointer
    library.ref_light_new.argtypes = [pointer]
    library.ref_policy_choose.argtypes = [pointer, pointer, i]
    library.ref_moggy_permit.argtypes = [pointer, pointer, i, i, ints]
    library.ref_play_random_move.argtypes = [pointer, pointer, i]
    library.ref_play_random_game.argtypes = [pointer, pointer, i, i, i, ints, uint8s, ints, ints,
                                             ints]
    library.ref_policy_assess.argtypes = [pointer, pointer, i, i, uint8s, i,
                                          np.ctypeslib.ndpointer(np.float32, flags="C"), ints]
    return library

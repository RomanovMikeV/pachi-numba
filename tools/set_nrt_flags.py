# pachi-numba: a translation of Pachi (GPL-2.0-only), October 2026; see NOTICE.
"""Set numba's ``_nrt`` flag on every compiled function of the package.

numba counts references to arrays with atomic operations. A board is a
named tuple of 19 arrays, so every call that passes it costs dozens of
them; this made the engine 3 times slower. Functions that allocate
nothing do not need the counting at all:

- direct allocation (numpy constructors, lists, appends) -> ``_nrt=True``;
- a call to a function that returns a new object (an owned reference)
  -> ``_nrt=True``, so that the object is freed;
- otherwise -> ``_nrt=False``.

A function compiled without counting may call one with counting only if
the callee says so explicitly (``_nrt=True``), which is why every function
gets the flag. Functions returning views (into ``Board.scratch``) must not
be called from counting functions; the script checks that.

Usage: ``python tools/set_nrt_flags.py`` (rewrites the sources) or
``python tools/set_nrt_flags.py --check`` (exit status 1 if a flag is not
up to date; run by the tests).
"""
import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1] / "pachi_numba"
NUMPY_ALLOC = {"zeros", "empty", "full", "array", "ones", "zeros_like", "empty_like", "copy"}
RETURNS_NEW = {"board_alloc", "board_new_like", "board_duplicate", "mq_new", "amafmap_new",
               "ownermap_new", "prior_map_new", "tree_alloc", "tree_init", "tree_garbage_collect",
               "tree_promote_node", "tree_promote_at", "make_state_from", "search_seeds",
               "descent_buffers", "board_copy", "tree_copy_into"}
RETURNS_VIEW = {"_selfatari_state", "_area_and_neighbors"}
KEEP_NRT = {"board_copy", "tree_copy_into"}  # return their array argument

funcs = {}
for f in sorted(ROOT.rglob("*.py")):
    for node in ast.parse(f.read_text()).body:
        if isinstance(node, ast.FunctionDef) and any(
                isinstance(d, ast.Call) and getattr(d.func, "id", None) == "njit"
                for d in node.decorator_list):
            funcs[node.name] = node

def analyze(node):
    allocates, calls = False, set()
    for n in ast.walk(node):
        if isinstance(n, (ast.List, ast.ListComp)):
            allocates = True
        if isinstance(n, ast.Attribute) and n.attr == "append":
            allocates = True
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Attribute) and getattr(n.func.value, "id", None) == "np" \
                    and n.func.attr in NUMPY_ALLOC:
                allocates = True
            name = n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", None)
            if name:
                calls.add(name)
    return allocates, calls

nrt = {}
for name, node in funcs.items():
    allocates, calls = analyze(node)
    nrt[name] = allocates or bool(calls & RETURNS_NEW) or name in KEEP_NRT
bad = [(caller, callee) for caller, node in funcs.items() if nrt[caller]
       for callee in analyze(node)[1] & RETURNS_VIEW]
if bad:
    sys.exit(f"view-returning functions called from NRT functions: {bad}")



def mark(m):
    deco, name = m.group(1), m.group(2)
    if name not in nrt:
        return m.group(0)
    return f"@njit({deco}, _nrt={nrt[name]})\ndef {name}("


outdated = []
for f in sorted(ROOT.rglob("*.py")):
    s = f.read_text()
    marked = re.sub(r"@njit\(([^)]*)\)\ndef (\w+)\(", mark,
                    s.replace(", _nrt=False", "").replace(", _nrt=True", ""))
    if marked != s:
        outdated.append(str(f))
        if "--check" not in sys.argv:
            f.write_text(marked)
if "--check" in sys.argv:
    if outdated:
        sys.exit(f"_nrt flags are not up to date in: {outdated}")
else:
    print(f"with reference counting: {sum(nrt.values())}, without: {len(nrt) - sum(nrt.values())}")

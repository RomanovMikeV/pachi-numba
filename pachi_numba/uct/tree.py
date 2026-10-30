# pachi-numba: a translation of Pachi (GPL-2.0-only), October 2026; see NOTICE.
"""Translation of Pachi's ``uct/tree.h`` and ``uct/tree.c``: the search tree.

C links ``struct tree_node`` records by pointers. Here a tree is a named
tuple of arrays indexed by node number, with ``NONE`` (-1) for a missing
link; ``struct move_stats`` fields are pairs of arrays (``*_value``,
``*_playouts``). Only the ``fast_alloc`` mode of the default engine is
translated: nodes are taken in order from preallocated arrays, and
``nodes_size`` counts bytes as C does (``NODE_SIZE`` per node), so that the
garbage collection runs at the same moments. When the arrays fill up, the
Python wrapper copies the tree into larger ones (``tree_grow``); this does
not change the search. Local trees (``local_tree``) are off by default and
not translated; their two root nodes are still counted in ``nodes_size``.

C recurses over subtrees (``tree_prune``, ``tree_fix_node_symmetry``);
numba cannot cache recursive functions, so these use explicit stacks and
visit the nodes in the same order.
"""

from collections import namedtuple

import numpy as np
from numba import njit

from .. import board as pb
from ..move import PASS, is_pass
from ..stone import S_WHITE, stone_other

NONE = -1
NODE_SIZE = 88  # sizeof(struct tree_node) on 64-bit Linux and macOS
TREE_NODE_D_MAX = 3
TREE_HINT_INVALID = 1

# See tree_garbage_collect().
LARGE_TREE_PLAYOUTS = 40000
DEEP_PLAYOUTS_THRESHOLD = 40
SMALL_TREE_PLAYOUTS = 5000

# Scalar fields of struct tree: indices into Tree.ints.
(ROOT, ROOT_COLOR, USE_EXTRA_KOMI, UNTRUSTWORTHY_TREE, MAX_DEPTH, NODES_SIZE, MAX_TREE_SIZE,
 MAX_PRUNED_SIZE, PRUNING_THRESHOLD, USED, AVG_SCORE_PLAYOUTS,
 SYM_X1, SYM_X2, SYM_Y1, SYM_Y2, SYM_D, SYM_TYPE, BOARD_SIZE, NUM_INTS) = range(19)
# Float fields: indices into Tree.floats (float32, floating_t).
EXTRA_KOMI, AVG_SCORE_VALUE, LTREE_AGING, NUM_FLOATS = range(4)

Tree = namedtuple("Tree", [
    "ints", "floats",
    "parent", "sibling", "children",
    "u_value", "u_playouts", "prior_value", "prior_playouts", "amaf_value", "amaf_playouts",
    "pu_value", "pu_playouts", "winner_owner_value", "winner_owner_playouts",
    "black_owner_value", "black_owner_playouts",
    "coord", "depth", "descents", "d", "hints", "is_expanded",
])


@njit(cache=True, nogil=True, _nrt=True)
def tree_alloc(capacity):
    """Arrays for ``capacity`` nodes."""
    return Tree(
        np.zeros(NUM_INTS, np.int64), np.zeros(NUM_FLOATS, np.float32),
        np.full(capacity, NONE, np.int32), np.full(capacity, NONE, np.int32),
        np.full(capacity, NONE, np.int32),
        np.zeros(capacity, np.float32), np.zeros(capacity, np.int32),
        np.zeros(capacity, np.float32), np.zeros(capacity, np.int32),
        np.zeros(capacity, np.float32), np.zeros(capacity, np.int32),
        np.zeros(capacity, np.float32), np.zeros(capacity, np.int32),
        np.zeros(capacity, np.float32), np.zeros(capacity, np.int32),
        np.zeros(capacity, np.float32), np.zeros(capacity, np.int32),
        np.zeros(capacity, np.int32), np.zeros(capacity, np.int32),
        np.zeros(capacity, np.int32), np.zeros(capacity, np.uint8),
        np.zeros(capacity, np.uint8), np.zeros(capacity, np.bool_),
    )


@njit(cache=True, nogil=True, _nrt=False)
def capacity(t):
    return t.parent.shape[0]


@njit(cache=True, nogil=True, _nrt=False)
def free_slots(t):
    return capacity(t) - t.ints[USED]


@njit(cache=True, nogil=True, _nrt=False)
def _clear_node(t, n):
    t.parent[n] = NONE
    t.sibling[n] = NONE
    t.children[n] = NONE
    t.u_value[n] = 0
    t.u_playouts[n] = 0
    t.prior_value[n] = 0
    t.prior_playouts[n] = 0
    t.amaf_value[n] = 0
    t.amaf_playouts[n] = 0
    t.pu_value[n] = 0
    t.pu_playouts[n] = 0
    t.winner_owner_value[n] = 0
    t.winner_owner_playouts[n] = 0
    t.black_owner_value[n] = 0
    t.black_owner_playouts[n] = 0
    t.coord[n] = 0
    t.depth[n] = 0
    t.descents[n] = 0
    t.d[n] = 0
    t.hints[n] = 0
    t.is_expanded[n] = False


@njit(cache=True, nogil=True, _nrt=False)
def _copy_node(dst, m, src, n):
    """C: ``*n2 = *node`` (the links are set by the caller)."""
    dst.parent[m] = src.parent[n]
    dst.sibling[m] = src.sibling[n]
    dst.children[m] = src.children[n]
    dst.u_value[m] = src.u_value[n]
    dst.u_playouts[m] = src.u_playouts[n]
    dst.prior_value[m] = src.prior_value[n]
    dst.prior_playouts[m] = src.prior_playouts[n]
    dst.amaf_value[m] = src.amaf_value[n]
    dst.amaf_playouts[m] = src.amaf_playouts[n]
    dst.pu_value[m] = src.pu_value[n]
    dst.pu_playouts[m] = src.pu_playouts[n]
    dst.winner_owner_value[m] = src.winner_owner_value[n]
    dst.winner_owner_playouts[m] = src.winner_owner_playouts[n]
    dst.black_owner_value[m] = src.black_owner_value[n]
    dst.black_owner_playouts[m] = src.black_owner_playouts[n]
    dst.coord[m] = src.coord[n]
    dst.depth[m] = src.depth[n]
    dst.descents[m] = src.descents[n]
    dst.d[m] = src.d[n]
    dst.hints[m] = src.hints[n]
    dst.is_expanded[m] = src.is_expanded[n]


@njit(cache=True, nogil=True, _nrt=False)
def tree_alloc_node(t, count):
    """Allocate ``count`` zeroed nodes; the first one, or NONE if not enough memory.

    C adds to ``nodes_size`` before the check, also when it fails.
    """
    nsize = count * NODE_SIZE
    old_size = t.ints[NODES_SIZE]
    t.ints[NODES_SIZE] = old_size + nsize
    if old_size + nsize > t.ints[MAX_TREE_SIZE]:
        return NONE
    if t.ints[USED] + count > capacity(t):
        raise MemoryError("tree arrays are full (the caller grows them between playouts)")
    n = t.ints[USED]
    t.ints[USED] += count
    for i in range(count):
        _clear_node(t, n + i)
    return n


@njit(cache=True, nogil=True, _nrt=False)
def tree_setup_node(t, n, coord, depth):
    """C also sets a debugging hash, which nothing reads."""
    t.coord[n] = coord
    t.depth[n] = depth
    if depth > t.ints[MAX_DEPTH]:
        t.ints[MAX_DEPTH] = depth


@njit(cache=True, nogil=True, _nrt=False)
def tree_init_node(t, coord, depth):
    n = tree_alloc_node(t, 1)
    if n == NONE:
        return NONE
    tree_setup_node(t, n, coord, depth)
    return n


@njit(cache=True, nogil=True, _nrt=True)
def tree_init(b, color, max_tree_size, max_pruned_size, pruning_threshold, ltree_aging,
              capacity_nodes):
    """A tree with the root (a virtual pass) for ``color`` to play."""
    t = tree_alloc(capacity_nodes)
    t.ints[MAX_TREE_SIZE] = max_tree_size
    t.ints[MAX_PRUNED_SIZE] = max_pruned_size
    t.ints[PRUNING_THRESHOLD] = pruning_threshold
    # The root PASS move is only virtual, we never play it.
    t.ints[ROOT] = tree_init_node(t, PASS, 0)
    t.ints[SYM_X1] = b.ints[pb.SYM_X1]
    t.ints[SYM_X2] = b.ints[pb.SYM_X2]
    t.ints[SYM_Y1] = b.ints[pb.SYM_Y1]
    t.ints[SYM_Y2] = b.ints[pb.SYM_Y2]
    t.ints[SYM_D] = b.ints[pb.SYM_D]
    t.ints[SYM_TYPE] = b.ints[pb.SYM_TYPE]
    t.ints[BOARD_SIZE] = pb.board_size(b)  # C: tree->board
    t.ints[ROOT_COLOR] = stone_other(color)  # to research black moves, root will be white
    # C allocates ltree_black and ltree_white outside the buffer, but still
    # counts them in nodes_size.
    t.ints[NODES_SIZE] += 2 * NODE_SIZE
    t.floats[LTREE_AGING] = ltree_aging
    return t


@njit(cache=True, nogil=True, _nrt=True)
def tree_copy_into(dst, src):
    """Copy all nodes and fields of ``src`` into the larger ``dst`` (same numbers)."""
    dst.ints[:] = src.ints
    dst.floats[:] = src.floats
    for n in range(src.ints[USED]):
        _copy_node(dst, n, src, n)
    return dst


def tree_grow(t, min_free):
    """A copy of the tree with at least ``min_free`` free nodes (Python side)."""
    if free_slots(t) >= min_free:
        return t
    new_capacity = max(2 * capacity(t), int(t.ints[USED]) + min_free)
    return tree_copy_into(tree_alloc(new_capacity), t)


# ---- values

@njit(cache=True, nogil=True, _nrt=False)
def tree_leaf_node(t, n):
    return t.children[n] == NONE


@njit(cache=True, nogil=True, _nrt=False)
def tree_node_parity(t, n):
    return -1 if (t.depth[n] ^ t.depth[t.ints[ROOT]]) & 1 else 1


@njit(cache=True, nogil=True, _nrt=False)
def tree_parity(t, parity):
    """Black parity from parity within the tree."""
    return parity if t.ints[ROOT_COLOR] == S_WHITE else -1 * parity


@njit(cache=True, nogil=True, _nrt=False)
def tree_node_get_value(t, parity, value):
    """A 0..1 value to maximize; ``parity`` is parity within the tree."""
    value = np.float32(value)
    return value if tree_parity(t, parity) > 0 else np.float32(np.float32(1) - value)


@njit(cache=True, nogil=True, _nrt=False)
def tree_node_criticality(t, n):
    """cov(player_gets, player_wins); see C for the derivation."""
    bo = t.black_owner_value[n]
    uv = t.u_value[n]
    inner = np.float32(np.float32(np.float32(np.float32(np.float32(2) * bo) * uv) - bo) - uv)
    return np.float32(t.winner_owner_value[n] - np.float32(inner + np.float32(1)))


# ---- children

@njit(cache=True, nogil=True, _nrt=False)
def tree_expand_node_children(t, node, b, consider, prior_value, prior_playouts, distances,
                              child_count):
    """The node creation part of C's tree_expand_node(); priors are computed by the caller.

    ``consider``, ``prior_value`` and ``prior_playouts`` are indexed by
    ``coord + 1`` (pass at 0), as C's ``map.prior = &map_prior[1]``.
    """
    ni = tree_alloc_node(t, child_count)
    # In fast_alloc mode we might temporarily run out of nodes but this should be rare.
    if ni == NONE:
        t.is_expanded[node] = False
        return
    tree_setup_node(t, ni, PASS, t.depth[node] + 1)
    first_child = ni
    t.parent[ni] = node
    t.prior_value[ni] = prior_value[PASS + 1]
    t.prior_playouts[ni] = prior_playouts[PASS + 1]
    t.d[ni] = TREE_NODE_D_MAX + 1

    # The loop considers only the symmetry playground.
    child = 1
    size = pb.board_size(b)
    for j in range(b.ints[pb.SYM_Y1], b.ints[pb.SYM_Y2] + 1):
        for i in range(b.ints[pb.SYM_X1], b.ints[pb.SYM_X2] + 1):
            if b.ints[pb.SYM_D]:
                x = size - 1 - i if b.ints[pb.SYM_TYPE] == pb.SYM_DIAG_DOWN else i
                if x > j:
                    continue
            c = i + j * size
            if not consider[c + 1]:  # Filter out invalid moves
                continue
            assert c != t.coord[node]
            nj = first_child + child
            child += 1
            tree_setup_node(t, nj, c, t.depth[node] + 1)
            t.parent[nj] = node
            t.sibling[ni] = nj
            ni = nj
            t.prior_value[ni] = prior_value[c + 1]
            t.prior_playouts[ni] = prior_playouts[c + 1]
            t.d[ni] = distances[c]
    t.children[node] = first_child  # must be done at the end to avoid race


@njit(cache=True, nogil=True, _nrt=False)
def tree_get_node(t, parent, c, create):
    """The child of ``parent`` at ``c`` (children are sorted by coordinate)."""
    first = t.children[parent]
    if first == NONE or t.coord[first] >= c:
        # Special case: Insertion at the beginning.
        if first != NONE and t.coord[first] == c:
            return first
        if not create:
            return NONE
        nn = tree_init_node(t, c, t.depth[parent] + 1)
        t.parent[nn] = parent
        t.sibling[nn] = first
        t.children[parent] = nn
        return nn
    # No candidate at the beginning, look through all the children.
    ni = first
    while t.sibling[ni] != NONE:
        if t.coord[t.sibling[ni]] >= c:
            break
        ni = t.sibling[ni]
    if t.sibling[ni] != NONE and t.coord[t.sibling[ni]] == c:
        return t.sibling[ni]
    assert t.coord[ni] < c
    if not create:
        return NONE
    nn = tree_init_node(t, c, t.depth[parent] + 1)
    t.parent[nn] = parent
    t.sibling[nn] = t.sibling[ni]
    t.sibling[ni] = nn
    return nn


# ---- garbage collection

@njit(cache=True, nogil=True, _nrt=True)
def tree_prune(dest, src, node, threshold, depth):
    """Copy the subtree of ``node``: all nodes at or above ``depth`` or with at least
    ``threshold`` playouts. Returns the copy, or NONE if ``dest`` is full.

    The relative order of children is preserved. C recurses; the explicit
    stack allocates the nodes in the same (depth-first) order.
    """
    n2 = tree_alloc_node(dest, 1)
    if n2 == NONE:
        return NONE
    _copy_node(dest, n2, src, node)
    if dest.depth[n2] > dest.ints[MAX_DEPTH]:
        dest.ints[MAX_DEPTH] = dest.depth[n2]
    dest.children[n2] = NONE
    dest.is_expanded[n2] = False
    if src.depth[node] >= depth and src.u_playouts[node] < threshold:
        return n2
    if src.children[node] == NONE:
        return n2

    # Each frame: (source node, its copy, the next source child, the last copied child).
    stack = [(node, n2, src.children[node], NONE)]
    while len(stack) > 0:
        s_node, d_node, s_child, d_prev = stack[-1]
        if s_child == NONE:
            # All children copied.
            dest.is_expanded[d_node] = True
            stack.pop()
            continue
        c2 = tree_alloc_node(dest, 1)
        if c2 == NONE:
            # dest is full: avoid partially expanded nodes. C's caller then
            # goes on with the next sibling of this node.
            dest.children[d_node] = NONE
            stack.pop()
            continue
        _copy_node(dest, c2, src, s_child)
        if dest.depth[c2] > dest.ints[MAX_DEPTH]:
            dest.ints[MAX_DEPTH] = dest.depth[c2]
        dest.children[c2] = NONE
        dest.is_expanded[c2] = False
        if d_prev == NONE:
            dest.children[d_node] = c2
        else:
            dest.sibling[d_prev] = c2
        dest.sibling[c2] = NONE
        dest.parent[c2] = d_node
        stack[-1] = (s_node, d_node, src.sibling[s_child], c2)
        if not (src.depth[s_child] >= depth and src.u_playouts[s_child] < threshold) \
                and src.children[s_child] != NONE:
            stack.append((s_child, c2, src.children[s_child], NONE))
    return n2


@njit(cache=True, nogil=True, _nrt=True)
def tree_garbage_collect(t, node, capacity_nodes):
    """A new tree holding only the subtree of ``node`` (pruned if large).

    Returns (the new tree, the new number of ``node``). C copies the subtree
    to a temporary tree and back into its buffer; one copy into new arrays
    gives the same nodes and sizes.
    """
    assert t.parent[node] == NONE and t.sibling[node] == NONE
    # Find the maximum depth at which we can copy all nodes.
    max_nodes = 1
    ni = t.children[node]
    while ni != NONE:
        max_nodes += 1
        ni = t.sibling[ni]
    nodes_size = max_nodes * NODE_SIZE
    max_depth = t.depth[node]
    while nodes_size < t.ints[MAX_PRUNED_SIZE] and max_nodes > 1:
        max_nodes -= 1
        nodes_size += max_nodes * nodes_size
        max_depth += 1

    # Copy all nodes for small trees; for large trees, deep nodes only with enough playouts.
    threshold = (t.u_playouts[node] - LARGE_TREE_PLAYOUTS) * DEEP_PLAYOUTS_THRESHOLD \
        // LARGE_TREE_PLAYOUTS
    if t.u_playouts[node] - LARGE_TREE_PLAYOUTS < 0:
        threshold = 0  # C: negative, then clamped
    if threshold > DEEP_PLAYOUTS_THRESHOLD:
        threshold = DEEP_PLAYOUTS_THRESHOLD

    new = tree_alloc(capacity_nodes)
    new.ints[:] = t.ints
    new.floats[:] = t.floats
    new.ints[NODES_SIZE] = 0
    new.ints[USED] = 0
    new.ints[MAX_DEPTH] = 0
    # The temporary tree of C is limited to max_pruned_size.
    new.ints[MAX_TREE_SIZE] = t.ints[MAX_PRUNED_SIZE]
    new_node = tree_prune(new, t, node, threshold, max_depth)
    assert new_node != NONE
    new.ints[MAX_TREE_SIZE] = t.ints[MAX_TREE_SIZE]
    return new, new_node


# ---- promotion

@njit(cache=True, nogil=True, _nrt=False)
def tree_unlink_node(t, node):
    ni = t.parent[node]
    if t.children[ni] == node:
        t.children[ni] = t.sibling[node]
    else:
        ni = t.children[ni]
        while t.sibling[ni] != node:
            ni = t.sibling[ni]
        t.sibling[ni] = t.sibling[node]
    t.sibling[node] = NONE
    t.parent[node] = NONE


@njit(cache=True, nogil=True, _nrt=False)
def _symmetry_update(t, size, c):
    """board_symmetry_update() on the tree's root_symmetry."""
    kind = t.ints[SYM_TYPE]
    if kind == pb.SYM_NONE:
        return
    x = c % size
    y = c // size
    half = size // 2
    dx = size - 1 - x
    broken = False
    if kind == pb.SYM_FULL:
        if x == half and y == half:
            return
        if x == y:
            t.ints[SYM_TYPE] = pb.SYM_DIAG_UP
            t.ints[SYM_X1] = t.ints[SYM_Y1] = 1
            t.ints[SYM_X2] = t.ints[SYM_Y2] = size - 1
            t.ints[SYM_D] = 1
        elif dx == y:
            t.ints[SYM_TYPE] = pb.SYM_DIAG_DOWN
            t.ints[SYM_X1] = t.ints[SYM_Y1] = 1
            t.ints[SYM_X2] = t.ints[SYM_Y2] = size - 1
            t.ints[SYM_D] = 1
        elif x == half:
            t.ints[SYM_TYPE] = pb.SYM_HORIZ
            t.ints[SYM_Y1] = 1
            t.ints[SYM_Y2] = size - 1
            t.ints[SYM_D] = 0
        elif y == half:
            t.ints[SYM_TYPE] = pb.SYM_VERT
            t.ints[SYM_X1] = 1
            t.ints[SYM_X2] = size - 1
            t.ints[SYM_D] = 0
        else:
            broken = True
    elif kind == pb.SYM_DIAG_UP:
        broken = x != y
    elif kind == pb.SYM_DIAG_DOWN:
        broken = dx != y
    elif kind == pb.SYM_HORIZ:
        broken = x != half
    elif kind == pb.SYM_VERT:
        broken = y != half
    if broken:
        t.ints[SYM_TYPE] = pb.SYM_NONE
        t.ints[SYM_X1] = t.ints[SYM_Y1] = 1
        t.ints[SYM_X2] = t.ints[SYM_Y2] = size - 1
        t.ints[SYM_D] = 0


@njit(cache=True, nogil=True, _nrt=True)
def tree_promote_node(t, node, capacity_nodes):
    """Make ``node`` (a child of the root) the root. Returns (tree, node): the
    tree is new if it was garbage collected."""
    assert t.parent[node] == t.ints[ROOT]
    tree_unlink_node(t, node)
    # Garbage collect if we run out of memory, or it is cheap to do so now:
    if t.ints[NODES_SIZE] >= t.ints[PRUNING_THRESHOLD] \
            or (t.ints[NODES_SIZE] >= t.ints[MAX_TREE_SIZE] // 10
                and t.u_playouts[node] < SMALL_TREE_PLAYOUTS):
        t, node = tree_garbage_collect(t, node, capacity_nodes)
    t.ints[ROOT] = node
    t.ints[ROOT_COLOR] = stone_other(t.ints[ROOT_COLOR])
    _symmetry_update(t, t.ints[BOARD_SIZE], t.coord[node])
    t.ints[AVG_SCORE_PLAYOUTS] = 0
    # ltree aging: local trees are not translated (ltree_aging is 1 without them).
    return t, node



@njit(cache=True, nogil=True, _nrt=False)
def _flip_coord(size, c, flip_horiz, flip_vert, flip_diag):
    x = c % size
    y = c // size
    if flip_diag:
        x, y = y, x
    if flip_horiz:
        x = size - 1 - x
    if flip_vert:
        y = size - 1 - y
    return x + y * size


@njit(cache=True, nogil=True, _nrt=True)
def tree_fix_symmetry(t, size, c):
    """Flip the tree so that its playground covers the move ``c``."""
    if is_pass(c):
        return
    cx = c % size
    cy = c // size
    flip_horiz = cx < t.ints[SYM_X1] or cx > t.ints[SYM_X2]
    flip_vert = cy < t.ints[SYM_Y1] or cy > t.ints[SYM_Y2]
    flip_diag = False
    if t.ints[SYM_D]:
        direction = t.ints[SYM_TYPE] == pb.SYM_DIAG_DOWN
        x = size - 1 - cx if direction ^ flip_horiz ^ flip_vert else cx
        if (x < cy) if flip_vert else (x > cy):
            flip_diag = True
    if not (flip_horiz or flip_vert or flip_diag):
        return
    # C: tree_fix_node_symmetry() recursively from the root.
    stack = [t.ints[ROOT]]
    while len(stack) > 0:
        n = stack.pop()
        if not is_pass(t.coord[n]):
            t.coord[n] = _flip_coord(size, t.coord[n], flip_horiz, flip_vert, flip_diag)
        ni = t.children[n]
        while ni != NONE:
            stack.append(ni)
            ni = t.sibling[ni]


@njit(cache=True, nogil=True, _nrt=True)
def tree_promote_at(t, size, c, capacity_nodes):
    """Promote the root child at ``c``; returns (tree, found)."""
    tree_fix_symmetry(t, size, c)
    ni = t.children[t.ints[ROOT]]
    while ni != NONE:
        if t.coord[ni] == c:
            t, _ = tree_promote_node(t, ni, capacity_nodes)
            return t, True
        ni = t.sibling[ni]
    return t, False

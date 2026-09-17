"""Depression hierarchy: synthetic worlds with known answers, and brute force."""

import os

import numpy as np
import pytest
from scipy.ndimage import label as nd_label, minimum_filter

from kvterrain import core, hierarchy as H

NONE = H.NONE


def build(codes, **kw):
    codes = np.asarray(codes, dtype=np.uint16)
    return H.build_hierarchy(codes, 0.0, 65535.0, **kw)


def rootchild(nodes, v):
    while nodes["parent"][v] != H.ROOT:
        v = int(nodes["parent"][v])
    return v


def ancestors(nodes, v):
    out = [v]
    while nodes["parent"][v] != NONE:
        v = int(nodes["parent"][v])
        out.append(v)
    return out


# --------------------------------------------------------------------------- #
# Synthetic worlds                                                             #
# --------------------------------------------------------------------------- #

def test_single_bowl_known_spill():
    g = np.full((9, 9), 10)
    g[2:7, 2:7] = 5
    g[4, 1] = 7
    g[4, 0] = 7                       # a notch through the rim to the map edge
    h = build(g)
    n = h.nodes
    assert h.leaf_count == 1 and h.node_count == 2
    assert n["floor_code"][1] == 5 and n["spill_code"][1] == 7
    assert n["parent"][1] == H.ROOT and n["overflow_to"][1] == H.ROOT
    assert n["spill_cell"][1] == 4 * 9 + 1 and n["outlet_cell"][1] == 4 * 9 + 0
    assert n["flags"][1] & H.FLAG_SPILLS_OFF_MAP
    assert n["flags"][1] & H.FLAG_LEAF
    assert (h.labels[2:7, 2:7] == 1).all()
    assert n["cell_count"].sum() == g.size
    assert n["subtree_cells"][H.ROOT] == g.size


def test_two_bowls_merge_at_saddle():
    g = np.full((9, 15), 20)
    g[2:7, 2:6] = 4                   # bowl A, floor 4
    g[2:7, 9:13] = 6                  # bowl B, floor 6
    g[4, 6:9] = 8                     # saddle between them at 8
    g[4, 13:15] = 9                   # B spills off the east edge at 9
    h = build(g)
    n = h.nodes
    assert h.leaf_count == 2 and h.node_count == 4
    a, b = int(h.labels[4, 3]), int(h.labels[4, 10])
    assert {a, b} == {1, 2}
    m = int(n["parent"][a])
    assert n["parent"][b] == m and m == 3
    assert n["spill_code"][a] == n["spill_code"][b] == 8
    assert n["overflow_to"][a] == b and n["overflow_to"][b] == a
    assert n["floor_code"][m] == 4 and n["floor_cell"][m] == n["floor_cell"][a]
    assert n["parent"][m] == H.ROOT and n["spill_code"][m] == 9


def test_draining_flat_is_not_a_pit():
    g = np.full((8, 8), 30)
    g[2:6, 2:6] = 10                  # a flat terrace ...
    g[3, 1] = 5                       # ... with one lower sample beside it ...
    g[3, 0] = 5                       # ... that reaches the edge
    h = build(g)
    assert h.leaf_count == 0
    assert (h.labels == 0).all()


def test_flat_bottomed_pit_is_one_leaf():
    g = np.full((10, 10), 50)
    g[2:8, 2:8] = 12                  # 36 equal samples, no lower neighbour
    h = build(g)
    assert h.leaf_count == 1
    assert h.nodes["cell_count"][1] >= 36


def test_edge_cropped_bowl_drains_off_map():
    g = np.full((8, 8), 40)
    g[2:6, 0:4] = 10                  # the bowl touches the west edge
    h = build(g)
    assert h.leaf_count == 0


def test_minor_flag_thresholds():
    g = np.full((9, 9), 1000)
    g[4, 4] = 999                     # one sample, one code deep
    h = H.build_hierarchy(g.astype(np.uint16), 0.0, 655.35, spacing_m=5.0)
    assert h.leaf_count == 1
    assert h.nodes["flags"][1] & H.FLAG_MINOR


def test_ocean_outlet_flag():
    g = np.full((7, 7), 10)
    g[2:5, 2:5] = 3
    g[3, 0:2] = 5                     # a sea inlet (sea level 5) against the bowl
    ocean = g == 5
    h = build(g, ocean=ocean)
    assert h.nodes["flags"][1] & H.FLAG_OUTLET_OCEAN


# --------------------------------------------------------------------------- #
# Brute force                                                                  #
# --------------------------------------------------------------------------- #

def _neigh(SY, SX, r, c):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if (dr or dc) and 0 <= r + dr < SY and 0 <= c + dc < SX:
                yield r + dr, c + dc


def minimax_from(g, sources):
    """Lowest possible max height along an 8-connected path from `sources`."""
    SY, SX = g.shape
    F = np.full(g.shape, np.inf)
    for r, c in sources:
        F[r, c] = g[r, c]
    changed = True
    while changed:
        changed = False
        for r in range(SY):
            for c in range(SX):
                best = F[r, c]
                for rr, cc in _neigh(SY, SX, r, c):
                    v = max(g[r, c], F[rr, cc])
                    if v < best:
                        best = v
                if best < F[r, c]:
                    F[r, c] = best
                    changed = True
    return F


def brute_pits(g):
    SY, SX = g.shape
    lower = minimum_filter(g, size=3, mode="nearest") < g
    count = 0
    for v in np.unique(g):
        lbl, n = nd_label(g == v, structure=np.ones((3, 3)))
        for k in range(1, n + 1):
            m = lbl == k
            touches_edge = m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any()
            if not touches_edge and not (m & lower).any():
                count += 1
    return count


@pytest.mark.parametrize("seed", range(12))
def test_hierarchy_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    SY, SX = rng.integers(6, 14, size=2)
    # A handful of distinct values forces plenty of ties and flats.
    g = rng.integers(0, 6 + seed % 5, size=(SY, SX)).astype(np.int64)
    h = build(g)
    n = h.nodes
    lab = h.labels.astype(np.int64)

    assert h.leaf_count == brute_pits(g)
    assert int(n["cell_count"].sum()) == g.size

    # Fill level: what a Priority-Flood from the edges would fill each sample to.
    edges = [(r, c) for r in range(SY) for c in range(SX)
             if r in (0, SY - 1) or c in (0, SX - 1)]
    F = minimax_from(g, edges)
    for r in range(SY):
        for c in range(SX):
            l = lab[r, c]
            if l == 0:
                want = g[r, c]
            else:
                want = max(g[r, c], int(n["spill_code"][rootchild(n, l)]))
            assert F[r, c] == want, (r, c)

    # Structure.
    for v in range(1, h.node_count):
        p = int(n["parent"][v])
        assert p != NONE
        assert p == H.ROOT or p > v
        assert n["spill_code"][v] >= n["floor_code"][v]
        sc, oc = int(n["spill_cell"][v]), int(n["outlet_cell"][v])
        assert n["spill_code"][v] == max(g.flat[sc], g.flat[oc])
        assert v in ancestors(n, int(lab.flat[sc]))
        ov = int(n["overflow_to"][v])
        assert ov in ancestors(n, int(lab.flat[oc]))
        if p != H.ROOT:
            assert n["spill_code"][v] <= n["spill_code"][p]
            assert n["floor_code"][p] <= n["floor_code"][v]

    # Pairs of pits: the lowest pass between them is where they merge.
    leaves = list(range(1, h.leaf_count + 1))
    for a in leaves:
        fc = int(n["floor_cell"][a])
        Fa = minimax_from(g, [divmod(fc, SX)])
        anc_a = ancestors(n, a)
        for b in leaves:
            if b <= a:
                continue
            anc_b = set(ancestors(n, b))
            lca = next(x for x in anc_a if x in anc_b)
            if lca == H.ROOT:
                continue
            child = anc_a[anc_a.index(lca) - 1]
            fb = divmod(int(n["floor_cell"][b]), SX)
            assert Fa[fb] == n["spill_code"][child], (a, b)


def test_subtree_intervals():
    rng = np.random.default_rng(3)
    g = rng.integers(0, 8, size=(20, 20))
    h = build(g)
    tin, tout = H.subtree_intervals(h.nodes)
    for v in range(h.node_count):
        for a in ancestors(h.nodes, v):
            assert tin[a] <= tin[v] < tout[a]


# --------------------------------------------------------------------------- #
# Labels pyramid, atlas and file round trips                                   #
# --------------------------------------------------------------------------- #

def test_label_downsample_lowest_child():
    labels = np.arange(25, dtype=np.uint32).reshape(5, 5)
    codes = np.full((5, 5), 100, dtype=np.uint16)
    codes[1, 1] = 3                   # inside parent (0,0)'s gather and (1,1)'s
    codes[2, 3] = 3                   # a tie for parent (1,1): lower label wins
    out = H.build_label_pyramid(labels, [codes, None])[1]
    assert out.shape == (3, 3)
    assert out[0, 0] == labels[1, 1]
    assert out[1, 1] == min(labels[1, 1], labels[2, 3])
    assert out[2, 2] == labels[3, 3]  # all equal: lowest label in the gather


def test_atlas_offsets_bytes_per_sample():
    lx, ly, nl, ts = 4, 2, 2, 17
    assert core.atlas_tile_bytes(ts) == ts * ts * 2
    assert core.atlas_tile_bytes(ts, 4) == ts * ts * 4
    for lvl in range(nl):
        tx, ty = core.tiles_at_level(lx, ly, lvl)
        for y in range(ty):
            for x in range(tx):
                o2 = core.atlas_tile_offset(lx, ly, nl, ts, lvl, x, y)
                o4 = core.atlas_tile_offset(lx, ly, nl, ts, lvl, x, y, 4)
                assert o4 == 2 * o2
    assert core.atlas_total_bytes(lx, ly, nl, ts, 4) == 2 * core.atlas_total_bytes(
        lx, ly, nl, ts)


def test_hierarchy_bin_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    h = build(rng.integers(0, 20, size=(30, 40)))
    info = H.write_hierarchy_bin(h, str(tmp_path / "hierarchy.bin"))
    assert info["bytes"] == H.HEADER_BYTES + H.NODE_DTYPE.itemsize * h.node_count
    header, nodes = H.read_hierarchy_bin(str(tmp_path / "hierarchy.bin"))
    assert header["node_count"] == h.node_count
    assert header["leaf_count"] == h.leaf_count
    assert (header["samples_x"], header["samples_y"]) == (40, 30)
    assert header["trailing_bytes"] == 0
    assert nodes.tobytes() == h.nodes.tobytes()


def test_labels_atlas_round_trip(tmp_path):
    plan = core.GridPlan(epsg=25833, spacing_m=5.0, origin_x=0.0, origin_y=0.0,
                         tile_cells=8, leaf_tiles_x=4, leaf_tiles_y=2, num_levels=2)
    rng = np.random.default_rng(2)
    codes = rng.integers(0, 50, size=(plan.samples_y, plan.samples_x)).astype(np.uint16)
    h = build(codes)
    code_levels = [codes, codes[::2, ::2]]
    levels = H.build_label_pyramid(h.labels, code_levels)
    info = H.export_labels_atlas(plan, levels, str(tmp_path))
    ts = plan.tile_cells + 1
    raw = np.fromfile(tmp_path / H.LABELS_FILE, dtype="<u4")
    for lvl, arr in enumerate(levels):
        tx_n, ty_n = core.tiles_at_level(plan.leaf_tiles_x, plan.leaf_tiles_y, lvl)
        for ty in range(ty_n):
            for tx in range(tx_n):
                off = core.atlas_tile_offset(plan.leaf_tiles_x, plan.leaf_tiles_y,
                                             plan.num_levels, ts, lvl, tx, ty, 4)
                tile = raw[off // 4: off // 4 + ts * ts].reshape(ts, ts)
                r0, c0, _ = core.north_up_tile_slice(arr.shape[0], plan.tile_cells, tx, ty)
                assert (tile == arr[r0:r0 + ts, c0:c0 + ts]).all()
    assert info["bytes"] == raw.nbytes

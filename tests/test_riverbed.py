"""Downhill river bed: profiles, links, and a river through a pit."""

import numpy as np

from kvterrain import bathymetry, core, hierarchy as H, riverbed, rivernet, water

ROW = 16


def world():
    plan = core.GridPlan(epsg=25833, spacing_m=5.0, origin_x=0.0, origin_y=0.0,
                         tile_cells=32, leaf_tiles_x=1, leaf_tiles_y=1, num_levels=1)
    SY, SX = plan.samples_y, plan.samples_x
    cols = np.arange(SX, dtype=np.float32)
    ground = np.tile(100.0 - 0.5 * cols, (SY, 1)).astype(np.float32)
    ground[:, 12:16] += 3.0                  # a ridge across the valley floor
    t = np.zeros((SY, SX), np.uint8)
    t[ROW, 2:33] = water.TYPE_RIVER      # runs off the east edge
    w = np.where(t == water.TYPE_RIVER, 1, 0).astype(np.uint8)
    wg = water.WaterGrid(type=t, weight=w, lake_id=np.zeros((SY, SX), np.uint16),
                         lake_table={}, weight_raw=w,
                         lake_island=np.zeros((SY, SX), bool))
    return plan, ground, wg


def seg(seg_id, cols, row=ROW, order=1, **kw):
    y = (32 - row) * 5.0
    xy = np.array([[c * 5.0, y] for c in cols], dtype=np.float64)
    return rivernet.RiverSegment(seg_id=seg_id, geom_hash="", order=order, xy=xy,
                                 z=None, level=None, **kw)


def traced(segments):
    return rivernet.TracedNetwork(segments=segments, stride_m=5.0, clip={},
                                  span_stats={})


def carve(plan, ground, wg, *, downhill):
    level = np.where(wg.type == water.TYPE_RIVER, ground, np.nan).astype(np.float32)
    if downhill:
        tn = traced([seg(0, range(2, 33))])
        low, target, rep = riverbed.downhill_river_bed(plan, tn, wg, ground, level)
        assert rep["vertices_bed_lowered"] > 0
        lv = low[ROW, 2:33]
        lv = lv[np.isfinite(lv)]
        assert np.all(np.diff(lv) <= 1e-4)
        bed = bathymetry.carve_river_beds(ground, wg.type, wg.weight, plan.spacing_m,
                                          level=level, bed_target=target)
        riverbed.burn_downhill_paths(plan, tn, wg, bed)
        return bed
    return bathymetry.carve_river_beds(ground, wg.type, wg.weight, plan.spacing_m,
                                       level=level)


def test_bed_never_rises_downstream():
    plan, ground, wg = world()
    before = carve(plan, ground, wg, downhill=False)[ROW, 2:33]
    after = carve(plan, ground, wg, downhill=True)[ROW, 2:33]
    assert (np.diff(before) > 0.01).any()
    assert np.all(np.diff(after) <= 1e-4)
    assert np.all(after <= before + 1e-4)     # only ever lowers
    tn = traced([seg(0, range(2, 33))])
    bed = carve(plan, ground, wg, downhill=True)
    assert riverbed.pixel_descent(plan, tn, bed)["pixel_steps_rising"] == 0
    assert riverbed.pixel_descent(
        plan, tn, carve(plan, ground, wg, downhill=False))["pixel_steps_rising"] > 0


def test_river_through_a_pit_leaves_no_pit_in_the_channel():
    plan, ground, wg = world()

    def channel_pits(bed):
        hmin, hmax = core.resolve_height_range(bed)
        h = H.build_hierarchy(core.pack_leaf_codes(bed, hmin, hmax), hmin, hmax)
        floors = h.nodes["floor_cell"][1:h.leaf_count + 1].astype(np.int64)
        return int(((floors // plan.samples_x) == ROW).sum())

    assert channel_pits(carve(plan, ground, wg, downhill=False)) > 0
    assert channel_pits(carve(plan, ground, wg, downhill=True)) == 0


def test_minimum_carries_across_links_and_through_lake_spans():
    plan, ground, wg = world()
    ground = ground.copy()
    trib = seg(0, range(2, 8), row=10)
    trunk = seg(1, range(8, 20), row=10, lake_spans=[(2, 5, 1)])
    trib.downstream, trunk.upstream = 1, [0]
    ground[10, 2:8] = 50.0                   # the tributary arrives far below
    prof = riverbed.downhill_profiles(plan, traced([trunk, trib]), ground)
    assert prof["cyclic_segments"] == 0
    _, _, lm, bm = prof["profiles"][1]
    t_lm, t_bm = prof["profiles"][0][2], prof["profiles"][0][3]
    assert lm[0] <= t_lm[-1] and bm[0] <= t_bm[-1]
    assert np.all(lm <= 50.0 + 1e-9)          # through the lake span too
    lvl, bed, _, _ = prof["profiles"][1]
    assert np.all(lvl - bed > 0)


def test_link_cycles_are_reported_not_fatal():
    plan, ground, wg = world()
    a, b = seg(0, range(2, 8)), seg(1, range(8, 14))
    a.downstream, b.downstream = 1, 0
    a.upstream, b.upstream = [1], [0]
    prof = riverbed.downhill_profiles(plan, traced([a, b]), ground)
    assert prof["cyclic_segments"] == 2


def test_path_burn_reaches_samples_outside_the_river_mask():
    plan, ground, wg = world()
    wg.type[ROW, 10:20] = water.TYPE_LAND     # a gappy mask across the ridge
    tn = traced([seg(0, range(2, 33))])
    bed = carve(plan, ground, wg, downhill=True)
    assert riverbed.pixel_descent(plan, tn, bed)["pixel_steps_rising"] == 0


def test_path_burn_carries_through_lake_spans_without_writing_them():
    plan, ground, wg = world()
    bed = ground.copy()
    bed[ROW, 5] = 80.0                                  # a low sample just upstream ...
    tn = traced([seg(0, range(2, 33), lake_spans=[(6, 9, 1)])])   # ... of a lake span
    before = bed[ROW, 8:12].copy()
    rep = riverbed.burn_downhill_paths(plan, tn, wg, bed)
    assert np.array_equal(bed[ROW, 8:12], before)       # span samples untouched
    assert rep["path_samples_lowered_river"] > 0
    assert np.all(bed[ROW, 12:33] <= 80.0)              # the minimum carried across

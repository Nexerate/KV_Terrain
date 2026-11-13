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


def test_shape_uses_symmetric_windows_and_restores_both_invariants():
    slope = np.array([10., 9., 8., 7., 6.])
    assert np.allclose(riverbed._shape_profile(slope, np.full(5, np.inf), 3), slope)

    raw = np.array([10., 10., 10., 5., 5., 5., 5.])
    cap = np.array([10., 9., 10., 5., 5., 5., 5.])
    shaped = riverbed._shape_profile(raw, cap, 2)
    assert np.all(shaped <= np.minimum(raw, cap) + 1e-12)
    assert np.all(np.diff(shaped) <= 1e-12)
    assert np.count_nonzero(np.abs(np.diff(shaped)) < 1e-9) < 4


def test_bank_ceiling_reads_across_the_channel_and_skips_water_samples():
    plan, ground, wg = world()
    ground = np.full_like(ground, 100.0)
    s = seg(0, range(4, 24))

    # North bank is low but water: it must not sink the profile. South bank is
    # dry at 97 m and is therefore the ceiling.
    ground[14:16, :] = 20.0
    wg.type[14:16, :] = water.TYPE_LAKE
    ground[17:19, :] = 97.0
    prof = riverbed.downhill_profiles(plan, traced([s]), ground, wg=wg)
    ceiling = prof["ceilings"][0]
    assert np.allclose(ceiling, 97.0)
    assert np.all(prof["profiles"][0][2] <= 97.0 + 1e-9)
    assert prof["vertices_no_dry_bank_sample"] == 0


def test_bank_ceiling_starts_beyond_the_carved_strip():
    plan, ground, wg = world()
    ground = np.full_like(ground, 100.0)
    s = seg(0, range(4, 24))

    # With a 5 m modelled half-width the old samples at 5 m and 10 m could
    # include the all-touched strip.  The runtime-visible edge starts at 8.54 m,
    # toward these 97 m banks.  The exact bilinear value confirms that the low
    # inner strip is no longer sampled directly as the bank.
    ground[15, :] = 20.0
    ground[17, :] = 20.0
    ground[14, :] = 97.0
    ground[18, :] = 97.0
    ceiling = riverbed._bank_ceiling(plan, s.xy, ground, wg.type, 5.0)
    expected = 20.0 * (1.0 - riverbed.MASK_EDGE_DIAGONAL_CELLS) + \
               97.0 * riverbed.MASK_EDGE_DIAGONAL_CELLS
    assert np.allclose(ceiling, expected)


def test_bank_ceiling_reads_the_measurements_half_sample_position():
    plan, ground, wg = world()
    ground = np.full_like(ground, 100.0)
    s = seg(0, range(4, 24))
    edge = 0.5 * plan.spacing_m + riverbed.MASK_EDGE_DIAGONAL_CELLS * plan.spacing_m
    normal = riverbed._vertex_normals(s.xy)
    midpoint = s.xy + normal * (edge + 0.5 * plan.spacing_m)
    row, col = rivernet.world_to_grid(plan, midpoint[:, 0], midpoint[:, 1])
    rr = np.rint(row).astype(int)
    ground[rr, np.rint(col).astype(int)] = 90.0
    ceiling = riverbed._bank_ceiling(plan, s.xy, ground, wg.type, 2.5)
    measured, ok = riverbed._dry_bilinear(plan, ground, wg.type, midpoint)
    assert ok.all()
    assert np.all(ceiling <= measured + 1e-12)


def test_confluence_tail_is_accordant_and_bounded_by_max_grade():
    plan, ground, wg = world()
    ground = np.full_like(ground, 91.0)
    ground[:, 10:] = 90.0
    wg.type[:] = water.TYPE_LAND
    wg.type[ROW, 2:21] = water.TYPE_RIVER
    wg.weight[:] = 0
    wg.weight[ROW, 2:21] = 1
    u = seg(0, range(2, 11))
    main = seg(1, range(10, 21))
    u.downstream, main.upstream = 1, [0]

    ungraded = riverbed.downhill_profiles(plan, traced([u, main]), ground)
    prof = riverbed.downhill_profiles(plan, traced([u, main]), ground, wg=wg)
    ul = prof["profiles"][0][2]
    ml = prof["profiles"][1][2]
    assert np.allclose(ul[-3:], ml[0])       # touching channel-width zone
    depth = bathymetry.DEFAULT_RIVER_DEPTH_BY_ORDER[1]
    assert np.max(ungraded["profiles"][0][2] - ul) <= depth + 1e-9


def test_steep_grade_stops_at_carve_depth_and_tapers_upstream():
    surface = np.linspace(100.0, 10.0, 11)
    original = surface.copy()
    ceiling = np.full(surface.shape, np.inf)
    xy = np.stack([np.arange(11) * 5.0, np.zeros(11)], axis=1)

    riverbed._grade_tail(
        surface, ceiling, xy, 0, 10, junction_level=10.0, touch_m=0.0,
        max_grade=riverbed.CONFLUENCE_MAX_GRADE, max_lowering_m=1.5)

    lowering = original - surface
    assert lowering.max() <= 1.5 + 1e-9
    assert np.allclose(lowering[:5], 0.0)    # no grade carried to the source
    assert np.all(np.diff(surface) <= 1e-9)


def test_gentle_drop_ramp_is_backward_tapered_and_depth_bounded():
    xy = np.stack([np.arange(5) * 5.0, np.zeros(5)], axis=1)
    raw = np.array([10.0, 10.0, 10.0, 8.0, 8.0])
    ceiling = np.full(5, np.inf)

    surface = np.array([10.0, 10.0, 10.0, 8.0, 8.0])
    before = surface.copy()
    rep = riverbed._soften_gentle_drops(
        surface, raw, ceiling, xy, [], 1.5, 5.0)
    assert rep["candidates"] == rep["ramped"] == 1
    assert before[2] - before[3] > 1.75
    assert surface[2] - surface[3] < 1.75
    assert np.max(before - surface) <= 1.5 + 1e-12
    assert np.all(np.diff(surface) <= 1e-12)

    # A larger real fall can only spend the same carve-depth budget, so the
    # residual step remains white-water steep.
    large = np.array([10.0, 10.0, 10.0, 5.0, 5.0])
    original = large.copy()
    riverbed._soften_gentle_drops(
        large, raw, ceiling, xy, [], 1.5, 5.0)
    assert np.max(original - large) <= 1.5 + 1e-12
    assert large[2] - large[3] > 1.75


def test_fall_ramp_does_not_cross_lake_spans():
    xy = np.stack([np.arange(7) * 5.0, np.zeros(7)], axis=1)
    raw = np.array([10.0, 10.0, 8.0, 8.0, 8.0, 6.0, 6.0])
    surface = raw.copy()
    riverbed._soften_gentle_drops(
        surface, raw, np.full(7, np.inf), xy, [(3, 3, 1)], 1.5, 5.0)
    assert surface[3] == raw[3]


def test_flat_tilt_is_downstream_only_and_depth_bounded():
    surface = np.array([10.0, 10.0, 10.0, 8.0, 8.0, 8.0])
    before = surface.copy()
    n = riverbed._tilt_flat_runs(
        surface, before, np.full(6, np.inf), [], 1.5,
        np.full(6, -np.inf))
    assert n == 2
    assert np.all(-np.diff(surface[:3]) > 0.01)
    assert np.allclose(surface[3:], before[3:])
    assert np.all(np.diff(surface) <= 0.0)
    assert np.max(before - surface) <= 1.5


def test_lake_inlet_is_graded_down_and_lake_expands_into_low_mouth():
    plan, ground, wg = world()
    wg.lake_table = {1: {"level_m": 91.0}}
    s = seg(0, range(2, 23), lake_spans=[(15, 20, 1)])
    prof = riverbed.downhill_profiles(plan, traced([s]), ground, wg=wg)
    level = prof["profiles"][0][2]
    assert level[14] == 91.0
    ungraded = riverbed.downhill_profiles(plan, traced([s]), ground)
    depth = bathymetry.DEFAULT_RIVER_DEPTH_BY_ORDER[1]
    assert np.max(ungraded["profiles"][0][2] - level) <= depth + 1e-9

    # At an inlet the lake expands into a mouth whose banks stand within a metre
    # of its level, even over the river's own lower ground...
    ground2 = np.full_like(ground, 94.5)
    ground2[ROW, :] = 85.0
    wg.lake_table = {1: {"level_m": 95.0}}
    prof2 = riverbed.downhill_profiles(plan, traced([s]), ground2, wg=wg)
    level2 = prof2["profiles"][0][2]
    assert np.allclose(level2[:15], 95.0)
    assert np.all(np.diff(level2) <= 1e-9)

    # ...but not over banks far below it: that would stand the water in the air.
    ground3 = np.full_like(ground, 92.0)
    ground3[ROW, :] = 85.0
    prof3 = riverbed.downhill_profiles(plan, traced([s]), ground3, wg=wg)
    assert np.all(prof3["profiles"][0][2][:15] <= 92.0 + 1e-9)


def test_backwater_is_flat_to_banks_near_the_lake_and_stops_at_a_low_one():
    surface = np.array([89.0, 88.0, 87.0])
    ceiling = np.array([90.0, 94.0, 96.0])
    xy = np.stack([np.arange(3) * 5.0, np.zeros(3)], axis=1)
    reached = riverbed._grade_tail(
        surface, ceiling, xy, 0, 2, junction_level=95.0, touch_m=0.0,
        max_grade=riverbed.CONFLUENCE_MAX_GRADE, max_lowering_m=1.5,
        backwater=True)
    # 94 is within a metre of the lake: flooded. 90 is five below: not.
    assert np.allclose(surface, [89.0, 95.0, 95.0])
    assert not reached


def test_lake_backwater_propagates_across_upstream_links():
    plan, ground, wg = world()
    ground = np.full_like(ground, 94.5)
    wg.lake_table = {1: {"level_m": 95.0}}
    upstream = seg(0, range(2, 9))
    main = seg(1, range(8, 23), lake_spans=[(8, 14, 1)])
    upstream.downstream, main.upstream = 1, [0]
    prof = riverbed.downhill_profiles(
        plan, traced([upstream, main]), ground, wg=wg)
    assert np.allclose(prof["profiles"][1][2][:8], 95.0)
    assert np.allclose(prof["profiles"][0][2], 95.0)


def test_shaped_profile_is_written_directly_to_network_levels():
    plan, ground, wg = world()
    s = seg(0, range(2, 10))
    tn = rivernet.TracedNetwork(
        segments=[s], stride_m=5.0,
        clip={"features": 1, "clipped": 0, "split": 0,
              "vertices_in": s.n, "vertices_dropped": 0, "features_dropped": 0},
        span_stats={"raw": 0, "merged": 0, "dropped": 0, "kept": 0})
    shaped = np.linspace(91.0, 88.0, s.n)
    profiles = {0: (shaped + 9.0, shaped + 7.5, shaped, shaped - 1.5)}
    raster = np.full_like(ground, 123.0)
    net = rivernet.build_river_network(
        plan, water.WaterFeatures(), wg, ground, water_surface=raster,
        shaped_profiles=profiles, traced=tn)
    assert np.allclose(net.segments[0].level, shaped)


def test_lowered_surface_cuts_only_a_shallow_dry_band_inside_the_channel():
    plan, ground, wg = world()
    old_level = np.where(wg.type == water.TYPE_RIVER, 100.0, np.nan).astype(np.float32)
    new_level = np.where(wg.type == water.TYPE_RIVER, 90.0, np.nan).astype(np.float32)
    terrain = np.full_like(ground, 100.0)
    terrain[ROW, 2:6] = 91.0
    bed = bathymetry.carve_river_beds(
        terrain, wg.type, wg.weight, plan.spacing_m,
        level=old_level, surface_target=new_level)
    assert np.all(bed[ROW, 2:6] <= 90.0)
    assert np.all(bed[ROW, 6:33] > 90.0)


def test_unmistakable_open_channel_uphill_run_is_reversed():
    xy = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
    oriented, changed = rivernet.orient_clear_uphill_run(
        xy, np.array([100.0, 120.0, 130.0]), [])
    assert changed
    assert np.array_equal(oriented, xy[::-1])
    assert not rivernet.orient_clear_uphill_run(
        xy, np.array([100.0, 120.0, 130.0]), [(0, 2, 1)])[1]
    assert not rivernet.orient_clear_uphill_run(
        xy, np.array([100.0, 105.0, 110.0]), [])[1]

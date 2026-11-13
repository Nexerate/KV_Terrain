"""Streams onto the scanned channel, ridges kept as barriers, widths inside N50."""

from types import SimpleNamespace

import numpy as np
import shapely

from kvterrain import core, riverbed, rivernet, riverwidth, thalweg, water


def plan64():
    return core.GridPlan(epsg=25833, spacing_m=5.0, origin_x=0.0, origin_y=0.0,
                         tile_cells=64, leaf_tiles_x=1, leaf_tiles_y=1, num_levels=1)


def valley(plan, bottom_y: float, side_slope: float = 0.3):
    """A valley running east, its bottom at world y = bottom_y, sloping down east."""
    SY, SX = plan.samples_y, plan.samples_x
    x = np.arange(SX) * plan.spacing_m
    y = plan.origin_y + (SY - 1 - np.arange(SY)) * plan.spacing_m
    X, Y = np.meshgrid(x, y)
    return (300.0 - 0.02 * X + side_slope * np.abs(Y - bottom_y)).astype(np.float32)


def wgrid(plan):
    SY, SX = plan.samples_y, plan.samples_x
    return SimpleNamespace(type=np.zeros((SY, SX), np.uint8))


def seg(sid, xy, width=2.0, spans=(), down=None, up=()):
    return SimpleNamespace(seg_id=sid, xy=np.asarray(xy, float), width_m=width,
                           vertex_width_m=None, lake_spans=list(spans),
                           downstream=down, upstream=list(up), n=len(xy))


def test_a_stream_mapped_up_the_valley_side_moves_to_its_bottom():
    plan = plan64()
    ground = valley(plan, bottom_y=160.0)
    xs = np.arange(20.0, 300.0, 5.0)
    s = seg(1, np.stack([xs, np.full_like(xs, 170.0)], 1))
    rep = thalweg.snap_segments(plan, [s], ground, wgrid(plan))
    # Within one sample of the bottom: the bank is read at the channel strip's
    # edge, so any point closer than half of it already needs no cut.
    mid = s.xy[5:-5, 1]
    assert np.all(np.abs(mid - 160.0) <= 3.5), mid
    assert rep["cut_sum_after_m"] < 0.2 * rep["cut_sum_before_m"]


def test_a_stream_on_an_even_hillside_keeps_its_line():
    plan = plan64()
    SY, SX = plan.samples_y, plan.samples_x
    y = (SY - 1 - np.arange(SY)) * plan.spacing_m
    ground = np.tile((200.0 + 0.3 * y)[:, None], (1, SX)).astype(np.float32)
    xs = np.arange(20.0, 300.0, 5.0)
    s = seg(1, np.stack([xs, np.full_like(xs, 160.0)], 1))
    thalweg.snap_segments(plan, [s], ground, wgrid(plan))
    assert np.allclose(s.xy[:, 1], 160.0)


def test_a_tributary_follows_the_stream_it_joins():
    plan = plan64()
    ground = valley(plan, bottom_y=160.0)
    xs = np.arange(100.0, 300.0, 5.0)
    main = seg(2, np.stack([xs, np.full_like(xs, 170.0)], 1), up=[1])
    ty = np.arange(260.0, 170.0 - 1e-9, -5.0)
    trib = seg(1, np.stack([np.full_like(ty, 100.0), ty], 1), down=2)
    thalweg.snap_segments(plan, [trib, main], ground, wgrid(plan))
    assert np.hypot(*(trib.xy[-1] - main.xy[0])) < 1e-6


def test_lake_vertices_and_wide_rivers_keep_their_line():
    plan = plan64()
    ground = valley(plan, bottom_y=160.0)
    xs = np.arange(20.0, 300.0, 5.0)
    s = seg(1, np.stack([xs, np.full_like(xs, 170.0)], 1), spans=[(10, 20, 1)])
    s.vertex_width_m = np.full(len(xs), 2.0)
    s.vertex_width_m[40:] = 30.0
    thalweg.snap_segments(plan, [s], ground, wgrid(plan))
    assert np.allclose(s.xy[10:21, 1], 170.0)
    assert np.allclose(s.xy[40:, 1], 170.0)


def test_a_line_over_a_hill_is_a_barrier_but_an_embankment_is_breached():
    xs = np.arange(0.0, 500.0, 5.0)
    xy = np.stack([xs, np.zeros_like(xs)], 1)
    ceiling = np.full(xs.size, np.inf)
    # 20 m hill, 200 m across, on a gentle downhill.
    level = 100.0 - 0.01 * xs + 20.0 * np.exp(-((xs - 250.0) / 60.0) ** 2)
    shaped = riverbed._shape_profile(level, ceiling, 2)
    bars = riverbed._find_barriers(level, ceiling, shaped, xy, [])
    assert len(bars) == 1
    first, crest = bars[0]
    assert abs(xs[crest] - 250.0) <= 10.0
    redo = riverbed._shape_with_barriers(level, ceiling, 2, np.inf, bars)
    assert np.max(np.minimum(level, ceiling) - redo) < 5.0
    # A 6 m road embankment 20 m across stays breached.
    road = 100.0 - 0.01 * xs + np.where(np.abs(xs - 250.0) <= 10.0, 6.0, 0.0)
    shaped = riverbed._shape_profile(road, ceiling, 2)
    assert riverbed._find_barriers(road, ceiling, shaped, xy, []) == []


def _poly(x0, y0, x1, y1):
    return {"type": "Feature", "properties": {"objtype": "Elv"},
            "geometry": {"type": "Polygon", "coordinates": [
                [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}}


def test_an_off_centre_line_is_centred_in_its_n50_area():
    area = _poly(0, -20, 1000, 20)
    xs = np.arange(100, 900, 5.0)
    s = SimpleNamespace(seg_id=1, xy=np.stack([xs, np.full_like(xs, 6.0)], 1),
                        width_m=12.0, lake_spans=[])
    centres = {}
    riverwidth.vertex_widths([s], {"features": [area]}, centres=centres)
    # +normal points north for an eastward line: the middle is 6 m south.
    assert np.allclose(centres[1][20:-20], -6.0, atol=0.5)


def test_a_width_is_capped_so_its_circle_stays_in_the_area():
    area = shapely.Polygon([(0, -20), (1000, -20), (1000, 20), (0, 20)])
    xs = np.arange(100, 900, 5.0)
    s = SimpleNamespace(seg_id=1, xy=np.stack([xs, np.full_like(xs, 12.0)], 1),
                        vertex_width_m=np.full(xs.size, 40.0, np.float32), n=xs.size)
    rep = riverwidth.fit_to_areas([s], area, margin_m=0.0)
    assert rep["vertices_capped"] > 0
    assert np.all(s.vertex_width_m <= 16.0 + 1e-3)

from types import SimpleNamespace

import numpy as np
from scipy.ndimage import binary_dilation

from kvterrain import rivermouth, rivernet, water

SP = 5.0
N = 121
LEVEL = 300.0
C = N // 2                                  # the river runs down column C


def _scene(river_depth=2.0, depth=None):
    """A round lake with a river entering from the north and leaving to the south.
    The bed is a plain bowl (or `depth(r)`) with its 8-connected shore ring on the
    waterline, as lakebed leaves it."""
    plan = SimpleNamespace(origin_x=0.0, origin_y=0.0, spacing_m=SP,
                           samples_x=N, samples_y=N)
    yy, xx = np.mgrid[0:N, 0:N]
    r = np.hypot(yy - C, xx - C)
    lake = r < 30
    river = (np.abs(xx - C) <= 0) & ~lake & ((yy < C) | (yy > C))
    typ = np.where(lake, water.TYPE_LAKE, np.where(river, water.TYPE_RIVER,
                                                   water.TYPE_LAND)).astype(np.uint8)
    lake_id = np.where(lake, 1, 0).astype(np.uint16)
    ground = LEVEL + 0.2 * np.maximum(r - 30, 0) * SP + 0.5
    depth = (8.0 * np.clip((30 - r) / 30, 0, 1) if depth is None
             else np.where(lake, depth(np.hypot(yy - C, xx - C - 15)), 0.0))
    ring = lake & binary_dilation(~lake, structure=np.ones((3, 3), bool))
    depth[ring] = 0.0
    bed = np.where(lake, LEVEL - depth, ground)
    # The river trench: the inflow climbs upstream, the outflow falls away.
    inflow = river & (yy < C)
    outflow = river & (yy > C)
    bed = np.where(inflow, LEVEL - river_depth + 0.15 * (C - 30 - yy), bed)
    bed = np.where(outflow, LEVEL - river_depth - 0.05 * (yy - C - 30), bed)
    surf = np.where(lake, LEVEL, np.nan).astype(np.float32)

    xs = np.full(N, C * SP)
    ys = (N - 1 - np.arange(N)) * SP        # row 0 (north) first: downstream order
    xy = np.stack([xs, ys], 1)
    ri, ci = rivernet._nearest_idx(plan, xy)
    spans, _ = rivernet._runs_of_lake(lake_id[ri, ci].astype(np.int64))
    seg = SimpleNamespace(seg_id=0, xy=xy, lake_spans=spans, width_m=5.0, order=3)
    return plan, bed.astype(np.float32), typ, lake_id, surf, seg, lake, ring


def _carve(plan, bed, typ, lake_id, surf, seg, raised=None, **kw):
    mouths = rivermouth.find_mouths(plan, [seg], bed, surf, spacing_m=SP)
    out, rep = rivermouth.carve_river_mouths(
        bed.copy(), mouths, plan=plan, water_type=typ, lake_id=lake_id,
        params=rivermouth.MouthParams(**kw), raised_out=raised)
    return out, rep


def _highest_on_the_way_out(out, lake):
    """The height water in the lake must reach to leave it by the south river: the
    lowest possible highest sample on any path from the lake's deepest point to the
    bottom edge (a brute-force minimax flood)."""
    allowed = np.ones_like(lake)
    r0, c0 = np.unravel_index(np.argmin(np.where(lake, out, np.inf)), out.shape)
    conn = rivermouth._connect_heights(out.astype(np.float64), allowed,
                                       np.array([r0]), np.array([c0]))
    return float(conn[-1, C])


def test_an_inflow_runs_into_its_lake_without_a_dry_line():
    plan, bed, typ, lake_id, surf, seg, lake, ring = _scene()
    before = LEVEL - bed[C - 32:C - 20, C]
    assert before.min() == 0.0                         # the ring across the mouth
    out, rep = _carve(plan, bed, typ, lake_id, surf, seg)
    after = LEVEL - out[C - 32:C - 20, C]              # river, throat, delta
    assert after.min() > 1.5, after
    assert rep["inflows"] == 1 and rep["outflows"] == 1 and rep["inlets_opened"] == 1
    assert rep["inlets_kept_shut"] == {}
    # The delta spreads: wider into the lake than at the mouth.
    wet = lambda row: int(((LEVEL - out[row]) - (LEVEL - bed[row]) > 0.05).sum())
    assert wet(C - 25) > wet(C - 28) >= 1


def test_an_outlet_is_held_by_a_riffle_on_the_river():
    plan, bed, typ, lake_id, surf, seg, lake, ring = _scene()
    raised = []
    out, rep = _carve(plan, bed, typ, lake_id, surf, seg, raised=raised)
    changed = out != bed
    # Only lake samples are lowered, and the only ring samples that move are the
    # two throats; the only samples raised are river samples by the outlet.
    assert not ((out < bed) & ~lake).any()
    rr, _ = np.nonzero(changed & ring)
    assert rr.size and ((rr < C - 20) | (rr > C + 20)).all()
    up = out > bed
    assert up.any() and (typ[up] == water.TYPE_RIVER).all()
    # The lake runs wet through its outlet into the river...
    assert (LEVEL - out[C + 20:C + 36, C]).min() > 1.5
    # ...and the riffle, a river sample 10 cm above the river's own bed, is the
    # highest point on the way out: the lake's spill, which carries the head.
    crest = LEVEL - 2.0 + 0.1
    assert np.allclose(out[up], crest)
    assert abs(_highest_on_the_way_out(out, lake) - crest) < 1e-4
    assert rep["outlets_opened"] == 1 and rep["outlets_kept_shut"] == {}
    (r, c, lvl), = raised
    assert lvl == LEVEL and set(zip(r, c)) >= set(zip(*np.nonzero(up)))


def test_a_riffle_is_shallow_enough_to_keep_the_lake_one_basin():
    # A shallow lake 1 m deep with a hollow off the river's line, ringed by a
    # ridge 0.5 m down: a riffle at the river's 2 m would drop the lake below that
    # ridge and cut the hollow off as a basin of its own.
    def depth(r):
        return np.where(r < 5, 3.0, np.where(r < 8, 0.5, 1.0))
    plan, bed, typ, lake_id, surf, seg, lake, ring = _scene(depth=depth)
    out, rep = _carve(plan, bed, typ, lake_id, surf, seg)
    assert rep["riffle_depth_m"]["p50"] == 0.4        # rim 0.5 m, minus the margin
    assert abs(_highest_on_the_way_out(out, lake) - (LEVEL - 0.4)) < 1e-4


def test_an_outlet_stays_shut_when_the_lake_has_no_room_for_a_riffle():
    def depth(r):
        return np.where(r < 5, 3.0, np.where(r < 8, 0.05, 1.0))
    plan, bed, typ, lake_id, surf, seg, lake, ring = _scene(depth=depth)
    out, rep = _carve(plan, bed, typ, lake_id, surf, seg)
    assert rep["outlets_kept_shut"] == {"no_room": 1}
    south_ring = ring & (np.arange(N)[:, None] > C)
    assert np.array_equal(out[south_ring], bed[south_ring])
    assert not (out > bed).any()


def test_an_inlet_opens_onto_an_enclosed_hollow_beside_its_trench():
    plan, bed, typ, lake_id, surf, seg, lake, ring = _scene()
    # A hollow beside the inflow's trench, a metre below the lake's level: the lake
    # fills it through the opened throat, but cannot drain through it.
    bed[C - 34, C + 1:C + 4] = LEVEL - 1.0
    out, rep = _carve(plan, bed, typ, lake_id, surf, seg)
    assert rep["inlets_kept_shut"] == {}
    assert (LEVEL - out[C - 32:C - 20, C]).min() > 1.5


def test_an_inlet_stays_shut_where_the_lake_would_drain_through_it():
    plan, bed, typ, lake_id, surf, seg, lake, ring = _scene()
    # A ditch below the lake's level from beside the trench to the map edge.
    bed[C - 34, C + 1:] = LEVEL - 1.0
    out, rep = _carve(plan, bed, typ, lake_id, surf, seg)
    assert rep["inlets_kept_shut"] == {"map_edge": 1}
    north_ring = ring & (np.arange(N)[:, None] < C)
    assert np.array_equal(out[north_ring], bed[north_ring])   # the inlet's sill
    # The channel behind the sill is still cut.
    assert (LEVEL - out[C - 26:C - 20, C]).min() > 1.5


def test_a_trenched_river_does_not_carry_its_trench_into_the_lake():
    plan, bed, typ, lake_id, surf, seg, lake, ring = _scene(river_depth=20.0)
    mouths = rivermouth.find_mouths(plan, [seg], bed, surf, spacing_m=SP)
    assert {m.kind for m in mouths} == {rivermouth.INFLOW, rivermouth.OUTFLOW}
    by = {m.kind: m for m in mouths}
    assert by[rivermouth.INFLOW].depth_m == 3.0        # order 3's modelled depth
    # An outlet's funnel may follow its river deeper, to twice that at most.
    assert by[rivermouth.OUTFLOW].depth_m == 6.0 and by[rivermouth.OUTFLOW].capped


def test_mouths_can_be_switched_off():
    plan, bed, typ, lake_id, surf, seg, lake, ring = _scene()
    out, rep = _carve(plan, bed, typ, lake_id, surf, seg, enabled=False)
    assert np.array_equal(out, bed) and rep["carved"] == 0


def test_an_outlet_stays_shut_where_its_banks_stand_below_the_riffle():
    # The ground either side of the outlet river lies below where the riffle must
    # stand: the lake would run round it, so the sill holds it instead.
    plan, bed, typ, lake_id, surf, seg, lake, ring = _scene()
    yy, xx = np.mgrid[0:N, 0:N]
    banks = (typ == water.TYPE_LAND) & (np.abs(xx - C) <= 3) & (yy > C + 28) & (yy < C + 36)
    bed[banks] = LEVEL - 2.5
    out, rep = _carve(plan, bed, typ, lake_id, surf, seg)
    assert rep["outlets_kept_shut"] == {"low_banks": 1}
    south_ring = ring & (np.arange(N)[:, None] > C)
    assert np.array_equal(out[south_ring], bed[south_ring])
    assert not (out > bed).any()

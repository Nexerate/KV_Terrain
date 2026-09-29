from types import SimpleNamespace

import numpy as np
from scipy.ndimage import binary_dilation

from kvterrain import rivermouth, rivernet, water

SP = 5.0
N = 121
LEVEL = 300.0
C = N // 2                                  # the river runs down column C


def _scene(river_depth=2.0):
    """A round lake with a river entering from the north and leaving to the south.
    The bed is a plain bowl with its 8-connected shore ring on the waterline, as
    lakebed leaves it."""
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
    depth = 8.0 * np.clip((30 - r) / 30, 0, 1)
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


def _carve(plan, bed, typ, lake_id, surf, seg, **kw):
    mouths = rivermouth.find_mouths(plan, [seg], bed, surf, spacing_m=SP)
    out, rep = rivermouth.carve_river_mouths(
        bed.copy(), mouths, plan=plan, water_type=typ, lake_id=lake_id,
        params=rivermouth.MouthParams(**kw))
    return out, rep


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


def test_the_outlet_sill_and_the_shoreline_stay_on_the_waterline():
    plan, bed, typ, lake_id, surf, seg, lake, ring = _scene()
    out, _ = _carve(plan, bed, typ, lake_id, surf, seg)
    changed = out != bed
    assert not (changed & ~lake).any()                 # land and rivers untouched
    assert (out <= bed).all()                          # only ever lowers
    # The only ring samples that moved are the inlet's throat, north of the lake.
    rr, _ = np.nonzero(changed & ring)
    assert rr.size and (rr < C).all()
    # The outlet is fed by a funnel behind its sill...
    assert (LEVEL - out[C + 20:C + 28, C]).min() > 1.5
    # ...and the sill itself is still the waterline.
    south_ring = ring & (np.arange(N)[:, None] > C)
    assert np.array_equal(out[south_ring], bed[south_ring])


def test_an_inlet_stays_shut_where_the_lake_could_spill_through_it():
    plan, bed, typ, lake_id, surf, seg, lake, ring = _scene()
    # A hollow beside the inflow's trench, a metre below the lake's level: opening
    # the throat would let the lake drain into it.
    bed[C - 40, C + 1:C + 4] = LEVEL - 1.0
    out, rep = _carve(plan, bed, typ, lake_id, surf, seg)
    assert rep["inlets_kept_shut"] == {"land_below_level": 1}
    assert np.array_equal(out[ring], bed[ring])        # every sill on the waterline
    # The channel behind the sill is still cut.
    assert (LEVEL - out[C - 26:C - 20, C]).min() > 1.5


def test_a_trenched_river_does_not_carry_its_trench_into_the_lake():
    plan, bed, typ, lake_id, surf, seg, lake, ring = _scene(river_depth=20.0)
    mouths = rivermouth.find_mouths(plan, [seg], bed, surf, spacing_m=SP)
    assert {m.kind for m in mouths} == {rivermouth.INFLOW, rivermouth.OUTFLOW}
    assert all(m.depth_m == 3.0 for m in mouths)       # order 3's modelled depth


def test_mouths_can_be_switched_off():
    plan, bed, typ, lake_id, surf, seg, lake, ring = _scene()
    out, rep = _carve(plan, bed, typ, lake_id, surf, seg, enabled=False)
    assert np.array_equal(out, bed) and rep["carved"] == 0

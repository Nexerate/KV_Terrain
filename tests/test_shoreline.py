import numpy as np
from scipy.ndimage import binary_dilation

from kvterrain import shoreline, water

N = 60
LEVEL = 300.0
C = N // 2


def _world(plane_offset=-0.4):
    """A round lake outline whose scan shows a flat water plane `plane_offset` from
    the level, and a scanned shore rising out of it. The outline overshoots the
    scanned shore on the east: those samples are land standing above the level."""
    yy, xx = np.mgrid[0:N, 0:N]
    r = np.hypot(yy - C, xx - C)
    outline = r < 16
    water_r = np.where(xx > C, 13.0, 16.0)            # scanned water edge
    ground = np.where(r < water_r, LEVEL + plane_offset,
                      LEVEL + plane_offset + 0.6 * (r - water_r)).astype(np.float32)
    typ = np.where(outline, water.TYPE_LAKE, water.TYPE_LAND).astype(np.uint8)
    wg = water.WaterGrid(type=typ, weight=np.zeros_like(typ),
                         lake_id=np.where(outline, 1, 0).astype(np.uint16),
                         lake_table={1: {"hoyde_moh": LEVEL}},
                         lake_island=np.zeros(typ.shape, bool))
    return wg, ground, outline


def test_scanned_land_inside_the_outline_is_shore_not_lake():
    wg, ground, outline = _world()
    sh = shoreline.split_shores(wg, ground)
    land = outline & (ground >= LEVEL)
    assert land.any()
    assert (wg.type[land] == water.TYPE_LAND).all() and (wg.lake_id[land] == 0).all()
    assert not sh.plane[land].any()
    # The plane is the lake's scanned water, and only it is modelled...
    lake = wg.type == water.TYPE_LAKE
    plane = lake & (np.abs(ground - (LEVEL - 0.4)) <= 0.15)
    assert np.array_equal(sh.plane, plane)
    assert not sh.plane[lake & (ground > LEVEL - 0.25)].any()
    # ...from its own height when that lies below the level.
    assert np.allclose(sh.anchor[plane], LEVEL - 0.4)
    assert sh.report["land_samples_to_shore"] == int(land.sum())


def test_a_plane_above_the_level_is_anchored_below_the_waterline():
    wg, ground, outline = _world(plane_offset=+0.3)
    sh = shoreline.split_shores(wg, ground,
                                params=shoreline.ShoreParams(edge_drop_m=0.3))
    assert sh.plane.any()
    assert np.allclose(sh.anchor[sh.plane], LEVEL - 0.3)


def test_scanned_land_inside_the_lake_is_an_island():
    wg, ground, outline = _world()
    ground[C - 1:C + 2, C - 1:C + 2] = LEVEL + 1.0     # a rock the outline missed
    ground[C, C] = LEVEL + 2.0
    shoreline.split_shores(wg, ground)
    assert wg.lake_island[C, C] and wg.type[C, C] == water.TYPE_LAKE


def test_flat_water_off_the_lakes_median_height_stays_lake():
    # A big lake's scan: a patch of its water flown most of a metre higher than
    # the rest, and above the level. It is water, not land.
    wg, ground, outline = _world()
    patch = outline & (np.hypot(*np.mgrid[0:N, 0:N] - C) < 8) & \
        (np.mgrid[0:N, 0:N][1] < C)
    ground[patch] = LEVEL + 0.4
    sh = shoreline.split_shores(wg, ground)
    assert (wg.type[patch] == water.TYPE_LAKE).all()
    assert sh.plane[patch].all() and not wg.lake_island[patch].any()


def test_low_dry_bumps_inside_a_boggy_lake_are_water():
    # Vegetation on the water: rough scan a hand's breadth above the level, far
    # from the shore. It is water, not a scatter of dry islands.
    wg, ground, outline = _world()
    rr, cc = np.mgrid[C - 3:C + 4, C - 3:C + 4]
    ground[rr, cc] = LEVEL + 0.1 + 0.2 * ((rr + cc) % 2)
    sh = shoreline.split_shores(wg, ground)
    assert (wg.type[rr, cc] == water.TYPE_LAKE).all()
    assert not wg.lake_island[rr, cc].any() and sh.plane[rr, cc].all()


def test_enclosed_low_ground_outside_the_outline_joins_the_lake():
    # West of the outline the scanned shore climbs from the plane, 0.4 m under the
    # level, and stays below the level for a sample or so: ground the lake covers.
    wg, ground, outline = _world()
    low_out = ~outline & (ground < LEVEL)
    assert low_out.any()
    sh = shoreline.split_shores(wg, ground)
    assert (wg.type[low_out] == water.TYPE_LAKE).all() and (wg.lake_id[low_out] == 1).all()
    assert sh.report["ground_samples_joined"] == int(low_out.sum())


def test_low_ground_that_drains_away_is_not_joined():
    wg, ground, outline = _world()
    ground[C, :C - 15] = LEVEL - 1.0                   # a ditch to the map edge
    shoreline.split_shores(wg, ground)
    assert (wg.type[C, :C - 15] == water.TYPE_LAND).all()


def test_only_samples_that_would_let_the_lake_drain_are_sealed():
    wg, ground, outline = _world()
    ground = np.where(outline, ground, np.maximum(ground, LEVEL + 0.2)).astype(np.float32)
    shoreline.split_shores(wg, ground)
    surf = wg.lake_surface_moh()
    bed = np.where(wg.type == water.TYPE_LAKE, LEVEL - 2.0, ground).astype(np.float32)
    lake = wg.type == water.TYPE_LAKE
    # A ditch below the level leading away from the west shore.
    bed[C, :C - 15] = LEVEL - 1.0
    before = bed.copy()
    n = shoreline.seal_shores(bed, wg, surf)
    sealed = bed != before
    assert n == int(sealed.sum()) and n > 0
    assert (bed[sealed] == LEVEL).all()
    # Every sealed sample is a lake sample beside the ditch; the rest of the shore
    # (land above the level) needs no seal and keeps its depth.
    ditch = np.zeros_like(lake)
    ditch[C, :C - 15] = True
    assert (sealed <= (lake & binary_dilation(ditch, structure=np.ones((3, 3), bool)))).all()
    ring = lake & binary_dilation(~lake, structure=np.ones((3, 3), bool))
    assert (bed[ring & ~sealed] < LEVEL).all()

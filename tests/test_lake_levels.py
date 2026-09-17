"""Outflow head on the hierarchy, and correcting published levels the terrain contradicts."""

import numpy as np

from kvterrain import core, hierarchy as H, wateraudit, water


def bowl_with_outlet(outlet_is_river=True):
    """15x15 at 20 m; a lake bowl at 10 m; a flat outlet channel at 12 m to the east edge."""
    h = np.full((15, 15), 20.0)
    h[4:11, 4:11] = 10.0
    h[7, 11:15] = 12.0
    t = np.zeros(h.shape, np.uint8)
    t[4:11, 4:11] = water.TYPE_LAKE
    if outlet_is_river:
        t[7, 11:15] = water.TYPE_RIVER
    lake_id = np.where(t == water.TYPE_LAKE, 1, 0).astype(np.uint16)
    wg = water.WaterGrid(type=t, weight=(t == water.TYPE_RIVER).astype(np.uint8),
                         lake_id=lake_id, lake_table={1: {"hoyde_moh": None}},
                         weight_raw=None, lake_island=np.zeros(h.shape, bool))
    channel = np.where(t == water.TYPE_RIVER, 14.0, np.nan).astype(np.float32)
    return h, wg, channel


def audit(level, outlet_is_river=True):
    h, wg, channel = bowl_with_outlet(outlet_is_river)
    wg.lake_table[1]["level_m"] = level
    plan = core.GridPlan(epsg=25833, spacing_m=5.0, origin_x=0.0, origin_y=0.0,
                         tile_cells=14, leaf_tiles_x=1, leaf_tiles_y=1, num_levels=1)
    codes = np.rint(h * 100).astype(np.uint16)       # 1 cm steps
    hier = H.build_hierarchy(codes, 0.0, 655.35, spacing_m=5.0)
    H.set_outflow_heads(hier, wg.type == water.TYPE_RIVER, channel)
    tin, tout = H.subtree_intervals(hier.nodes)
    rows = wateraudit.audit_lakes(plan, hier, codes, wg, inflow_lakes=set(),
                                  tin=tin, tout=tout)
    return rows[0], hier


def test_outflow_head_is_river_depth_at_the_spill():
    row, hier = audit(11.0)
    v = row["node_id"]
    assert abs(float(hier.nodes["spill_m"][v]) - 12.0) < 0.02
    assert abs(float(hier.nodes["outflow_head_m"][v]) - 2.0) < 0.02
    assert row["status"] == "ok"


def test_level_within_outflow_head_is_held_and_reproduced():
    row, hier = audit(13.5)
    v = row["node_id"]
    assert row["status"] == "held_by_outflow"
    n = hier.nodes
    assert float(n["spill_m"][v] + n["outflow_head_m"][v]) >= 13.5 - 1e-3
    assert n["flags"][v] & H.FLAG_AUTHORED_LAKE


def test_level_above_the_head_is_a_finding():
    row, _ = audit(16.0)
    assert row["status"] == "level_above_spill"
    assert row["spill_through"] in ("river_channel", "map_edge")
    assert abs(row["excess_m"] - 2.0) < 0.1


def test_spill_over_land_has_no_head():
    row, hier = audit(13.5, outlet_is_river=False)
    assert float(hier.nodes["outflow_head_m"][row["node_id"]]) == 0.0
    assert row["status"] == "level_above_spill"


def correction_world(hoyde):
    h = np.full((30, 30), 105.0, np.float32)          # shore at ~105 m
    h[10:20, 10:20] = 100.0                           # flown water surface at 100 m
    t = np.zeros(h.shape, np.uint8)
    t[10:20, 10:20] = water.TYPE_LAKE
    wg = water.WaterGrid(type=t, weight=np.zeros(h.shape, np.uint8),
                         lake_id=np.where(t == water.TYPE_LAKE, 1, 0).astype(np.uint16),
                         lake_table={1: {"hoyde_moh": hoyde}},
                         lake_island=np.zeros(h.shape, bool))
    return h, wg


def test_published_level_above_all_the_shore_is_corrected():
    h, wg = correction_world(118.0)
    rep = water.apply_estimated_levels(wg, h)
    info = wg.lake_table[1]
    assert info["level_source"] == water.LEVEL_SOURCE_CORRECTED
    assert info["hoyde_moh"] == 118.0
    assert info["level_m"] <= 105.0
    assert rep["levels_corrected"] == 1


def test_published_level_near_the_shore_is_kept():
    for hoyde in (104.0, 106.5):                      # below and within 2 m of the ring
        h, wg = correction_world(hoyde)
        water.apply_estimated_levels(wg, h)
        assert wg.lake_table[1]["level_source"] == water.LEVEL_SOURCE_NVE
        assert wg.lake_table[1]["level_m"] == hoyde


def test_correction_can_be_switched_off():
    h, wg = correction_world(118.0)
    rep = water.apply_estimated_levels(wg, h, max_above_shore_m=None)
    assert wg.lake_table[1]["level_m"] == 118.0
    assert rep["levels_corrected"] == 0

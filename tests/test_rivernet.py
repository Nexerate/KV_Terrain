"""Discharge, hydraulic width and hierarchy attachment on the river graph."""

import numpy as np

from kvterrain import core, hierarchy, rivernet


def segment(seg_id, x0, x1, *, unit, area_km2):
    return rivernet.RiverSegment(
        seg_id=seg_id, geom_hash="", order=1,
        xy=np.array([[x0, 0.0], [x1, 0.0]], dtype=np.float64),
        z=None, level=None, vassdragsnr=unit, regine_area_km2=area_km2)


def test_discharge_accumulates_each_regine_area_once():
    a = segment(0, 0.0, 10.0, unit="A", area_km2=10.0)
    b = segment(1, 0.0, 10.0, unit="A", area_km2=10.0)
    trunk = segment(2, 10.0, 20.0, unit="B", area_km2=20.0)
    a.downstream = b.downstream = 2
    trunk.upstream = [0, 1]
    traced = rivernet.TracedNetwork(
        segments=[a, b, trunk], stride_m=5.0, clip={}, span_stats={})

    report = rivernet.assign_discharge_and_width(traced)
    assert a.local_catchment_m2 == b.local_catchment_m2 == 5e6
    assert trunk.local_catchment_m2 == 20e6
    assert trunk.accumulated_catchment_m2 == 30e6
    assert np.isclose(
        trunk.discharge_m3_s,
        30e6 / rivernet.SECONDS_PER_MEAN_YEAR)
    assert report["represented_regine_area_m2"] == 30e6
    assert report["sink_accumulated_area_m2"] == 30e6
    assert a.width_m == float(np.float32(
        rivernet.width_from_discharge(a.discharge_m3_s)))


def test_segment_node_is_downstream_vertex_leaf_label():
    plan = core.GridPlan(epsg=25833, spacing_m=1.0, origin_x=0.0, origin_y=0.0,
                         tile_cells=4, leaf_tiles_x=1, leaf_tiles_y=1, num_levels=1)
    h = hierarchy.build_hierarchy(
        np.array([[9, 9, 9, 9, 9],
                  [9, 2, 3, 4, 9],
                  [9, 3, 1, 3, 9],
                  [9, 4, 3, 2, 9],
                  [9, 9, 9, 9, 9]], dtype=np.uint16),
        0.0, 65535.0)
    s = segment(0, 1.0, 2.0, unit="A", area_km2=1.0)
    # y=2 maps to north-up row 2; the downstream vertex is cell (2,2).
    s.xy[:, 1] = 2.0
    net = rivernet.RiverNetwork(segments=[s])
    report = rivernet.attach_hierarchy_nodes(plan, net, h)
    assert s.node_id == int(h.labels[2, 2])
    assert report["leaf"] == 1 and report["missing"] == 0


def test_cycle_conserves_area_and_gives_each_member_the_same_q():
    a = segment(0, 0.0, 10.0, unit="A", area_km2=4.0)
    b = segment(1, 10.0, 20.0, unit="A", area_km2=4.0)
    a.downstream, b.downstream = 1, 0
    a.upstream, b.upstream = [1], [0]
    traced = rivernet.TracedNetwork(
        segments=[a, b], stride_m=5.0, clip={}, span_stats={})
    report = rivernet.assign_discharge_and_width(traced)
    assert report["graph_cycle_segments"] == 2
    assert report["graph_cycle_components"] == 1
    assert report["sink_accumulated_area_m2"] == 4e6
    assert a.accumulated_catchment_m2 == b.accumulated_catchment_m2 == 4e6

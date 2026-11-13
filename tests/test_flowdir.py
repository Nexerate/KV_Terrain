"""Flat-resolved D8 directions, accumulation and wire formats."""

import numpy as np

from kvterrain import core, flowdir as F, hierarchy as H


def _walk(flow, start):
    seen = set()
    SY, SX = flow.shape
    i = int(start)
    while int(flow.flat[i]) != 0:
        assert i not in seen
        seen.add(i)
        d = int(flow.flat[i]) - 1
        r, c = divmod(i, SX)
        r += int(F.D8_DR[d])
        c += int(F.D8_DC[d])
        assert 0 <= r < SY and 0 <= c < SX
        i = r * SX + c
    return i


def test_strict_slope_uses_steepest_d8_and_edges_terminate():
    r, c = np.indices((5, 5))
    codes = (100 - r - c).astype(np.uint16)
    flow = F.build_flow_directions(codes)
    assert (flow[0] == 0).all() and (flow[-1] == 0).all()
    assert (flow[:, 0] == 0).all() and (flow[:, -1] == 0).all()
    assert (flow[1:-1, 1:-1] == 4).all()  # south-east is steepest


def test_closed_flat_has_one_row_major_terminal_and_no_cycles():
    codes = np.full((7, 7), 9, dtype=np.uint16)
    codes[1:6, 1:6] = 3
    flow = F.build_flow_directions(codes)
    terminals = np.flatnonzero((flow == 0) & (codes == 3))
    assert terminals.tolist() == [1 * 7 + 1]
    for i in np.flatnonzero(codes.ravel() == 3):
        assert _walk(flow, i) == terminals[0]


def test_drainable_flat_routes_to_lower_outlet():
    codes = np.full((7, 7), 9, dtype=np.uint16)
    codes[1:6, 1:5] = 5
    codes[3, 5] = 4
    codes[3, 6] = 3
    flow = F.build_flow_directions(codes)
    for i in np.flatnonzero(codes.ravel() == 5):
        assert _walk(flow, i) == 3 * 7 + 6


def test_node_catchments_are_composable_and_cover_the_lattice():
    codes = np.full((9, 15), 20, dtype=np.uint16)
    codes[2:7, 2:6] = 4
    codes[2:7, 9:13] = 6
    codes[4, 6:9] = 8
    codes[4, 13:15] = 9
    h = H.build_hierarchy(codes, 0.0, 65535.0, spacing_m=5.0)
    flow = F.build_flow_directions(codes)
    area, report = F.node_catchment_areas(flow, h.labels, h.nodes, 5.0)
    assert report["processed"] == codes.size
    assert area[0] == codes.size * 25.0
    parent = h.leaf_count + 1
    children = np.flatnonzero(h.nodes["parent"] == parent)
    assert area[parent] == area[children].sum()


def test_flowdir_atlas_and_catchments_round_trip(tmp_path):
    plan = core.GridPlan(epsg=25833, spacing_m=5.0, origin_x=0.0, origin_y=0.0,
                         tile_cells=4, leaf_tiles_x=2, leaf_tiles_y=2, num_levels=2)
    r, c = np.indices((plan.samples_y, plan.samples_x))
    codes = (100 - r - c).astype(np.uint16)
    flow = F.build_flow_directions(codes)
    info = F.export_flowdir_atlas(plan, flow, str(tmp_path))
    assert info["bytes"] == 4 * 5 * 5

    h = H.build_hierarchy(codes, 0.0, 65535.0, spacing_m=5.0)
    area, report = F.node_catchment_areas(flow, h.labels, h.nodes, 5.0)
    cinfo = F.write_catchments_bin(area, str(tmp_path / F.CATCHMENTS_FILE))
    header, got = F.read_catchments_bin(str(tmp_path / F.CATCHMENTS_FILE))
    assert cinfo["bytes"] == F.CATCHMENTS_HEADER_BYTES + 8 * h.node_count
    assert header["node_count"] == h.node_count
    assert got.tobytes() == area.tobytes()
    manifest = F.catchments_manifest(h.node_count, cinfo, report, 5.0)
    assert manifest["parent_includes_children"] is True

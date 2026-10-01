"""Side channels traced from N50's river areas where no mapped line runs."""

import os
import struct
from types import SimpleNamespace

import numpy as np
import shapely

from kvterrain import core, riverbed, riverbranch, rivernet, water


def plan128():
    return core.GridPlan(epsg=25833, spacing_m=5.0, origin_x=0.0, origin_y=0.0,
                         tile_cells=128, leaf_tiles_x=1, leaf_tiles_y=1, num_levels=1)


def wgrid(plan, lake_from_x=None, level=90.0):
    SY, SX = plan.samples_y, plan.samples_x
    t = np.zeros((SY, SX), np.uint8)
    lid = np.zeros((SY, SX), np.int32)
    table = {}
    if lake_from_x is not None:
        cols = np.arange(SX) * plan.spacing_m >= lake_from_x
        t[:, cols] = water.TYPE_LAKE
        lid[:, cols] = 1
        table = {1: {"level_m": level}}
    return SimpleNamespace(type=t, lake_id=lid, lake_table=table,
                           lake_level=lambda info: info.get("level_m"))


def main_line(x0, x1, y=300.0, width=40.0, spans=()):
    xs = np.arange(x0, x1 + 1e-9, 5.0)
    s = rivernet.RiverSegment(seg_id=0, geom_hash="m", order=4,
                              xy=np.stack([xs, np.full_like(xs, y)], 1),
                              z=None, level=None, lake_spans=list(spans))
    s.vertex_width_m = np.full(xs.size, width, np.float32)
    s.width_m, s.discharge_m3_s, s.depth_m = 10.0, 10.0, 2.0
    return s


def test_an_arm_round_an_island_becomes_a_side_channel():
    plan = plan128()
    main = shapely.box(0, 280, 640, 320)
    loop = shapely.box(150, 280, 450, 380).difference(shapely.box(200, 320, 400, 355))
    areas = shapely.union_all([main, loop])
    seg = main_line(20, 620)
    traced = SimpleNamespace(segments=[seg], stride_m=5.0)
    rep = riverbranch.add_side_channels(plan, traced, areas, wgrid(plan))
    assert rep["side_arms"] == 1, rep
    b = traced.segments[1]
    info = b.branch_of
    assert info.parent == 0 and info.rejoin == 0
    # A Y at each end: a river half-width plus BRANCH_LEAD_M beyond the arm.
    assert 100 <= seg.xy[info.parent_vertex, 0] <= 200
    assert 400 <= seg.xy[info.rejoin_vertex, 0] <= 500
    assert np.allclose(b.xy[0], seg.xy[info.parent_vertex])
    assert np.allclose(b.xy[-1], seg.xy[info.rejoin_vertex])
    # Down the middle of the 25 m arm, about as wide as it.
    mid = (b.xy[:, 0] > 230) & (b.xy[:, 0] < 370)
    assert np.all(np.abs(b.xy[mid, 1] - 367.5) <= 3.0), b.xy[mid, 1]
    assert abs(float(np.median(b.vertex_width_m[mid])) - 25.0) <= 4.0
    wb = float(np.median(b.vertex_width_m))
    assert abs(info.share - wb ** 2 / (wb ** 2 + 40.0 ** 2)) < 0.05
    assert b.upstream == [] and b.downstream is None
    assert np.isclose(b.discharge_m3_s, info.share * 10.0, rtol=1e-5)


def test_a_second_mouth_into_the_lake_becomes_a_delta_channel():
    plan = plan128()
    main = shapely.box(0, 280, 455, 320)
    arm = shapely.union_all([shapely.box(200, 300, 240, 370),
                             shapely.box(200, 340, 455, 370)])
    areas = shapely.union_all([main, arm])
    seg = main_line(20, 600, spans=[(86, 116, 1)])
    traced = SimpleNamespace(segments=[seg], stride_m=5.0)
    rep = riverbranch.add_side_channels(plan, traced, areas,
                                        wgrid(plan, lake_from_x=450.0))
    assert rep["delta_channels"] == 1, rep
    b = traced.segments[1]
    assert b.branch_of.rejoin == riverbranch.NO_SEGMENT
    assert b.branch_of.lake_id == 1
    assert b.lake_spans and b.lake_spans[-1][1] == b.n - 1
    assert b.xy[-1, 0] > 450.0


def test_a_bulge_with_one_contact_is_left_alone():
    plan = plan128()
    areas = shapely.union_all([shapely.box(0, 280, 640, 320),
                               shapely.box(250, 300, 330, 380)])
    traced = SimpleNamespace(segments=[main_line(20, 620)], stride_m=5.0)
    rep = riverbranch.add_side_channels(plan, traced, areas, wgrid(plan))
    assert len(traced.segments) == 1
    assert rep["skipped_one_contact"] >= 1


def test_a_side_channel_level_runs_from_split_to_rejoin():
    seg = main_line(20, 620)
    xs = np.arange(150.0, 451.0, 5.0)
    br = rivernet.RiverSegment(seg_id=1, geom_hash="b", order=4,
                               xy=np.stack([xs, np.full_like(xs, 360.0)], 1),
                               z=None, level=None)
    br.branch_of = riverbranch.BranchOf(parent=0, parent_vertex=26, share=0.3,
                                        rejoin=0, rejoin_vertex=86)
    lm = np.linspace(100.0, 90.0, seg.n)
    out = {0: (lm, lm - 2, lm.copy(), lm - 2),
           1: (np.full(br.n, 120.0), None, np.full(br.n, 120.0), None)}
    assert riverbed.tie_branches(out, {0: seg, 1: br}, wgrid(plan128())) == 1
    b = out[1][2]
    assert np.isclose(b[0], lm[26]) and np.isclose(b[-1], lm[86])
    assert np.all(np.diff(b) <= 1e-9)


def test_branches_bin_round_trips(tmp_path):
    seg = main_line(20, 620)
    br = main_line(150, 450)
    br.seg_id = 1
    br.branch_of = riverbranch.BranchOf(parent=0, parent_vertex=26, share=0.25,
                                        rejoin=-1, rejoin_vertex=0, lake_id=3)
    path = os.path.join(tmp_path, "b.bin")
    man = riverbranch.write_branches_bin([seg, br], path)
    assert man["branches"] == 1
    raw = open(path, "rb").read()
    magic, ver, n = struct.unpack_from("<8sII", raw, 0)
    assert magic == riverbranch.BRANCHES_MAGIC and n == 1
    rec = struct.unpack_from("<IIIfiIHH", raw, 16)
    assert rec[:3] == (1, 0, 26) and abs(rec[3] - 0.25) < 1e-6
    assert rec[4] == -1 and rec[6] == 3


def test_late_level_changes_are_taken_out_of_a_side_channel_end():
    seg = main_line(20, 620)
    seg.level = np.linspace(100.0, 90.0, seg.n).astype(np.float32)
    xs = np.arange(150.0, 451.0, 5.0)
    br = rivernet.RiverSegment(seg_id=1, geom_hash="b", order=4,
                               xy=np.stack([xs, np.full_like(xs, 360.0)], 1),
                               z=None, level=np.linspace(98.0, 92.0, xs.size).astype(np.float32))
    br.branch_of = riverbranch.BranchOf(parent=0, parent_vertex=26, share=0.3,
                                        rejoin=0, rejoin_vertex=86)
    assert rivernet._retie_branch_ends([seg, br]) == 2
    assert np.isclose(br.level[0], seg.level[26]) and np.isclose(br.level[-1], seg.level[86])


def test_a_line_beside_its_own_area_gets_no_second_channel():
    plan = plan128()
    # The mapped line runs in its area at both ends and 35 m north of it in the
    # middle, where the area alone carries the river.
    areas = shapely.union_all([shapely.box(0, 285, 250, 325),
                               shapely.box(410, 285, 640, 325),
                               shapely.box(200, 220, 250, 325),
                               shapely.box(410, 220, 460, 325),
                               shapely.box(200, 220, 460, 260)])
    traced = SimpleNamespace(segments=[main_line(20, 620, y=305.0, width=40.0)],
                             stride_m=5.0)
    rep = riverbranch.add_side_channels(plan, traced, areas, wgrid(plan))
    assert len(traced.segments) == 1, rep
    assert rep["skipped_line_beside_area"] == 1, rep


def test_two_mouths_from_one_reach_share_the_river_between_them():
    plan = plan128()
    # Two arms leave the river together and reach the lake on either side of it.
    main = shapely.box(0, 280, 455, 320)
    north = shapely.union_all([shapely.box(200, 300, 240, 370),
                               shapely.box(200, 340, 455, 370)])
    south = shapely.union_all([shapely.box(200, 230, 240, 300),
                               shapely.box(200, 230, 455, 260)])
    areas = shapely.union_all([main, north, south])
    seg = main_line(20, 600, spans=[(86, 116, 1)])
    traced = SimpleNamespace(segments=[seg], stride_m=5.0)
    rep = riverbranch.add_side_channels(plan, traced, areas,
                                        wgrid(plan, lake_from_x=450.0))
    assert rep["delta_channels"] == 2, rep
    assert rep["channels_sharing_a_reach"] == 2
    shares = [b.branch_of.share for b in traced.segments[1:]]
    assert sum(shares) < 1.0
    w = [float(np.median(b.vertex_width_m)) for b in traced.segments[1:]]
    total = 40.0 ** 2 + sum(x ** 2 for x in w)
    assert np.allclose(shares, [x ** 2 / total for x in w], atol=0.03)

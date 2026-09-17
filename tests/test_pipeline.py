"""The whole process stage on a small synthetic world, read back from disk."""

import json
import os

import numpy as np
import pytest

from kvterrain import core, exports, water
from kvterrain.fetch import synthetic_height_fetcher


@pytest.fixture(scope="module")
def export_dir(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("export"))
    plan = core.plan_grid(8.30, 61.28, 8.36, 61.31, spacing_m=25.0, tile_cells=32)
    core.run_export(plan, out, fetcher=synthetic_height_fetcher, server_url="(demo)",
                    include_water=True,
                    water_opts={"fetcher": water.synthetic_water_features})
    return out


def test_new_products_and_manifest(export_dir):
    m = json.load(open(os.path.join(export_dir, "manifest.json")))
    atlas = m["atlas"]
    assert atlas["labels_file"] == "labels.atlas"
    assert atlas["bytes_per_sample"] == {"heights.atlas": 2, "surface.atlas": 2,
                                         "water_id.atlas": 2, "labels.atlas": 4}
    for f in ("hierarchy.bin", "labels.atlas", "water_audit.json"):
        assert os.path.exists(os.path.join(export_dir, f))
    assert m["hierarchy"]["record_bytes"] == 52
    assert m["water_surface"]["river_bathymetry"]["downhill_bed"]["enabled"]
    assert m["water_surface"]["lake_bathymetry"]["edge_is_shore"]
    assert m["water_surface"]["lake_bathymetry"]["shore_8_connected"]
    audit = json.load(open(os.path.join(export_dir, "water_audit.json")))
    assert audit["summary"]["lakes"] == len(audit["lakes"])


def test_all_checks_pass(export_dir):
    exp = exports.load(export_dir)
    assert exports.check_atlas(exp)["ok"]
    assert exports.check_water(exp)["ok"]
    rep = exports.check_hierarchy(exp)
    assert rep["ok"], rep["errors"]


def test_check_hierarchy_catches_a_corrupt_label(export_dir, tmp_path):
    import shutil
    bad = str(tmp_path / "bad")
    shutil.copytree(export_dir, bad)
    exp = exports.load(bad)
    lx, ly = exp.leaf_tiles
    off = core.atlas_tile_offset(lx, ly, exp.num_levels, exp.tile_samples, 1, 0, 0, 4)
    with open(os.path.join(bad, "labels.atlas"), "r+b") as fh:
        fh.seek(off + 4 * (exp.tile_samples + 1))
        v = np.frombuffer(fh.read(4), "<u4")[0]
        fh.seek(off + 4 * (exp.tile_samples + 1))
        fh.write(np.uint32(v + 1).tobytes())
    rep = exports.check_hierarchy(exports.load(bad))
    assert not rep["ok"]


def test_cropped_lake_ramps_to_its_level_at_the_map_edge():
    SY = SX = 9
    t = np.zeros((SY, SX), np.uint8)
    t[2:7, 0:5] = 2                            # the west edge cuts through the lake
    h = np.full((SY, SX), 50.0, np.float32)
    surf = np.where(t == 2, 50.0, np.nan).astype(np.float32)
    bed = __import__("kvterrain.bathymetry", fromlist=["x"]).carve_lake_beds(
        h, t, 5.0, surface_moh=surf, estimate_missing=False, edge_is_shore=True)
    assert np.allclose(bed[2:7, 0], 50.0)
    assert bed[4, 2] < 50.0


def test_diagonal_lake_sliver_is_not_stepped_at_the_shore():
    from kvterrain import bathymetry
    t = np.zeros((20, 20), np.uint8)
    t[5:15, 5:15] = 2
    t[15, 15] = 2                              # touches the lake only diagonally
    h = np.full(t.shape, 50.0, np.float32)
    surf = np.where(t == 2, 50.0, np.nan).astype(np.float32)
    kw = dict(surface_moh=surf, estimate_missing=False)
    old = bathymetry.carve_lake_beds(h, t, 5.0, **kw)
    new = bathymetry.carve_lake_beds(h, t, 5.0, shore_8_connected=True, **kw)
    assert old[15, 15] <= 48.0                 # stepped to the 2 m floor
    assert new[15, 15] == 50.0                 # on the waterline, like any shore sample


def test_shore_is_watertight_diagonally():
    from kvterrain import bathymetry
    t = np.zeros((20, 20), np.uint8)
    t[3:17, 3:17] = 2
    t[3, 3] = 0                                # a dry corner notch
    h = np.full(t.shape, 50.0, np.float32)
    surf = np.where(t == 2, 50.0, np.nan).astype(np.float32)
    kw = dict(surface_moh=surf, estimate_missing=False)
    old = bathymetry.carve_lake_beds(h, t, 5.0, **kw)
    new = bathymetry.carve_lake_beds(h, t, 5.0, shore_8_connected=True, **kw)
    assert old[4, 4] < 49.0                    # diagonal to dry ground, carved ~2 m
    assert new[4, 4] == 50.0
    assert new[8, 8] == old[8, 8]              # the interior is unchanged

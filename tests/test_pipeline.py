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
    assert atlas["flowdir_file"] == "flowdir.atlas"
    assert atlas["bytes_per_sample"] == {"heights.atlas": 2, "surface.atlas": 2,
                                         "water_id.atlas": 2, "labels.atlas": 4,
                                         "flowdir.atlas": 1}
    for f in ("hierarchy.bin", "hypsometry.bin", "catchments.bin", "labels.atlas",
              "flowdir.atlas", "water_audit.json", "lake_islands.bin"):
        assert os.path.exists(os.path.join(export_dir, f))
    assert m["hierarchy"]["record_bytes"] == 52
    hypso = m["hierarchy"]["hypsometry"]
    assert hypso["file"] == "hypsometry.bin"
    assert hypso["format"] == "kvterrain-hypsometry/1"
    assert hypso["samples_per_node"] == 64
    assert hypso["axis"]["kind"] == "absolute_heights_atlas_u16_code"
    catch = m["hierarchy"]["catchments"]
    assert catch["file"] == "catchments.bin"
    assert catch["record_bytes"] == 8
    assert catch["parent_includes_children"] is True
    assert m["flow_direction"]["file"] == "flowdir.atlas"
    assert m["flow_direction"]["levels"] == 1
    rivers = m["water_vector"]["rivers"]
    assert rivers["format"] == "kvterrain-rivers-bin/2"
    assert rivers["discharge"]["kind"] == "modelled_mean_annual_discharge"
    assert rivers["width_model"]["formula"].startswith("width_m = clamp")
    assert rivers["hierarchy_attachment"]["missing"] == 0
    assert m["water_surface"]["river_bathymetry"]["downhill_bed"]["enabled"]
    assert m["water_surface"]["lake_bathymetry"]["edge_is_shore"]
    assert m["water_surface"]["lake_bathymetry"]["shore_8_connected"]
    islands = m["water_id"]["lake_islands"]
    assert islands["file"] == "lake_islands.bin"
    assert islands["bytes"] == 4 + 4 * islands["count"]
    cells = m["water_id"]["lake_cells"]
    assert cells["file"] == "lake_cells.bin"
    assert cells["bytes"] == 4 + 8 * cells["lake_count"] + 8 * cells["run_count"]
    samples = m["water_id"]["lake_samples"]
    assert samples["file"] == "lake_samples.bin"
    assert samples["bytes"] == 4 + 12 * samples["record_count"]
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
    # Bump one level-1 label, writing the atlas back in the dense_v1 layout
    # (which the reader still accepts) so the edit is a plain byte change.
    target = core.atlas_tile_index(lx, ly, 1, 0, 0)
    with exp.open_atlas("labels") as r:
        tiles = [r.read_tile(i).copy() for i in range(r.tile_count)]
    tiles[target].flat[exp.tile_samples + 1] += 1
    with open(os.path.join(bad, "labels.atlas"), "wb") as fh:
        for t in tiles:
            fh.write(t.astype("<u4").tobytes())
    rep = exports.check_hierarchy(exports.load(bad))
    assert not rep["ok"]


def test_atlases_are_kvtiles(export_dir):
    from kvterrain import tilepack
    m = json.load(open(os.path.join(export_dir, "manifest.json")))
    atlas = m["atlas"]
    assert atlas["format"] == "kvtiles_v1"
    assert m["flow_direction"]["format"] == "kvtiles_v1_level0"
    assert "offset index" in m["flow_direction"]["tile_offset"]
    assert atlas["tile_filter"]["heights.atlas"] == "delta_split"
    for fn in atlas["bytes_per_sample"]:
        path = os.path.join(export_dir, fn)
        assert tilepack.is_kvtiles(path)
        assert atlas["file_bytes"][fn] == os.path.getsize(path)
        if fn != "heights.atlas":
            assert atlas["tile_filter"][fn] == "none"
    assert atlas["height_bytes"] == core.atlas_total_bytes(
        *m["leaf_tiles"], m["num_levels"], m["tile_samples"])


def _write_dense_copy(src, dst):
    """The same export with every atlas written in the old dense_v1 layout."""
    import shutil
    from kvterrain import tilepack
    shutil.copytree(src, dst)
    exp = exports.load(dst)
    for kind in exports.ATLAS_KINDS:
        path = exp.atlas_path(kind)
        if not path:
            continue
        with exp.open_atlas(kind) as r:
            raw = b"".join(r.read_tile_bytes(i) for i in range(r.tile_count))
        with open(path, "wb") as fh:
            fh.write(raw)
        assert not tilepack.is_kvtiles(path)
    return exp


def test_dense_exports_still_read_and_pack_back(export_dir, tmp_path):
    from kvterrain import tilepack
    dense = _write_dense_copy(export_dir, str(tmp_path / "dense"))
    assert exports.check_atlas(dense)["ok"]
    assert exports.check_water(dense)["ok"]
    assert exports.check_hierarchy(dense)["ok"]
    new = exports.load(export_dir)
    for kind in exports.ATLAS_KINDS:
        if new.atlas_path(kind):
            for lvl in range(new.atlas_levels(kind)):
                assert np.array_equal(dense.read_level(kind, lvl), new.read_level(kind, lvl))

    # A Unity-style sidecar must ride along unchanged.
    with open(os.path.join(dense.root, "heights.atlas.meta"), "w") as fh:
        fh.write("guid: 0123\n")
    before = {fn: open(os.path.join(dense.root, fn), "rb").read()
              for fn in os.listdir(dense.root)}
    out = str(tmp_path / "packed")
    rep = tilepack.pack_export(dense.root, out)
    after = {fn: open(os.path.join(dense.root, fn), "rb").read()
             for fn in os.listdir(dense.root)}
    assert before == after                                  # the source is untouched
    assert set(rep["atlases"]) == set(dense.manifest["atlas"]["bytes_per_sample"])
    for fn, data in before.items():
        if fn in rep["atlases"] or fn == "manifest.json":
            continue
        assert open(os.path.join(out, fn), "rb").read() == data
    packed = exports.load(out)
    assert packed.manifest["atlas"]["format"] == "kvtiles_v1"
    assert packed.manifest["flow_direction"]["format"] == "kvtiles_v1_level0"
    assert exports.check_atlas(packed)["ok"]
    assert exports.check_water(packed)["ok"]
    assert exports.check_hierarchy(packed)["ok"]
    for fn in rep["atlases"]:
        assert tilepack.is_kvtiles(os.path.join(out, fn))
    with pytest.raises(ValueError):
        tilepack.pack_export(dense.root, out)               # refuses a non-empty dst


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


def test_repeated_nve_records_become_one_segment():
    def feat(oid, date, coords, order=1):
        return {"type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {"objectid": oid, "statusdatofra": date,
                               "strekninglnr": "7", "elveordenstrahler": order}}
    a = [[0.0, 0.0], [10.0, 0.0]]
    b = [[10.0, 0.0], [20.0, 5.0]]
    raw = [feat(1, 100, a), feat(2, 100, b), feat(3, 300, a, order=2),
           feat(4, 200, a), feat(5, 300, a, order=3)]
    wf = water.features_from_geojson(raw, [], [])
    assert len(wf.rivers) == 2
    first = wf.rivers[0]
    assert first.feature_index == 4 and first.order == 3      # latest, then highest id
    assert np.array_equal(first.xy, np.asarray(a))             # at a's first position
    assert wf.rivers[1].feature_index == 1

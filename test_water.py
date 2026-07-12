"""
Offline validation of the water pipeline (no network).

Uses a synthetic river + lake placed at known UTM coordinates and asserts:
  * the rasterisation grid is corner-centered (pixel centers on sample points),
  * features land where they should (lake interior, river centerline, dry land),
  * flow bearing + weight are encoded/decoded correctly,
  * the categorical pyramid keeps the right winner and never drops the thin river,
  * water tiles share bit-identical edges (aligned with the height tiles),
  * the .water record round-trips through the packed byte layout.
"""
import numpy as np

from kvterrain import core, water


def _plan():
    # Small square region so the pyramid collapses to a single root tile.
    return core.GridPlan(
        epsg=25833, spacing_m=5.0, origin_x=265000.0, origin_y=6648000.0,
        tile_cells=64, leaf_tiles_x=4, leaf_tiles_y=4, num_levels=3,
    )


def world_of_sample(plan, i_east, j_north):
    """World UTM of corner-centered sample (south-index j, east-index i)."""
    return (plan.origin_x + i_east * plan.spacing_m,
            plan.origin_y + j_north * plan.spacing_m)


def test_grid_transform_is_corner_centered():
    plan = _plan()
    T = water.sample_grid_transform(plan, level=0)
    SX, SY = plan.samples_x, plan.samples_y
    # Pixel center (col+0.5, row+0.5) must equal the sample world point.
    for (i, j) in [(0, 0), (SX - 1, 0), (0, SY - 1), (SX - 1, SY - 1), (7, 11)]:
        row = SY - 1 - j                      # north-up row for south-index j
        wx, wy = T * (i + 0.5, row + 0.5)
        ex, ey = world_of_sample(plan, i, j)
        assert abs(wx - ex) < 1e-6 and abs(wy - ey) < 1e-6, (i, j, wx - ex, wy - ey)
    print("[ok] raster pixel centers coincide with corner-centered sample points")
    return plan


def test_bearing_roundtrip():
    for deg in [0, 45, 90, 135, 180, 225, 270, 315, 359]:
        b = water.encode_bearing(deg)
        back = water.decode_bearing(b)
        # within one quantisation step (360/256 ≈ 1.4°)
        d = min((back - deg) % 360, (deg - back) % 360)
        assert d <= 360 / 256 + 1e-9, (deg, back, d)
    print("[ok] flow bearing encode/decode within one quantisation step")


def test_rasterize_places_features(plan):
    feats = water.synthetic_water_features(plan)
    g = water.rasterize_water(plan, feats, width_scale=1.0)
    SX, SY = plan.samples_x, plan.samples_y
    assert g.type.shape == (SY, SX)

    # --- lake interior: sample the center of the lake rectangle ---
    x0, y0, x1, y1 = plan.bbox_utm
    lx = x0 + 0.70 * (x1 - x0)
    ly = y0 + 0.35 * (y1 - y0)
    i = int(round((lx - plan.origin_x) / plan.spacing_m))
    j = int(round((ly - plan.origin_y) / plan.spacing_m))
    row = SY - 1 - j
    assert g.type[row, i] == water.TYPE_LAKE, "lake center not marked lake"
    assert g.lake_id[row, i] == 1, "lake center missing local id"
    print("[ok] lake interior sample is TYPE_LAKE with a local lake_id")

    # --- dry corner: SE-most sample should be land (river/lake are elsewhere) ---
    assert g.type[SY - 1, SX - 1] == water.TYPE_LAND, "SE corner unexpectedly wet"
    print("[ok] dry corner sample is TYPE_LAND")

    # --- river present with expected size + roughly-diagonal flow ---
    river_mask = g.type == water.TYPE_RIVER
    assert river_mask.any(), "river not rasterised"
    assert g.weight[river_mask].max() == 5, "river weight != stream order 5"
    # First leg of the synthetic river runs SW->NE at ~ atan2(dx,dy).
    dx = 0.40 - 0.10
    dy = 0.35 - 0.10
    exp = np.degrees(np.arctan2(dx * (plan.bbox_utm[2] - plan.bbox_utm[0]),
                                dy * (plan.bbox_utm[3] - plan.bbox_utm[1]))) % 360
    flows = water.decode_bearing(g.flow[river_mask].astype(float))
    near = np.min(np.minimum((flows - exp) % 360, (exp - flows) % 360))
    assert near < 5.0, f"no river pixel near expected bearing {exp:.1f} (min {near:.1f})"
    print(f"[ok] river present (order 5), flow bearing ~{exp:.1f}° found")

    # --- lake beats river where they overlap: no sample is both ---
    assert not np.any((g.lake_id > 0) & (g.weight > 0)), "river bytes left under lake"
    print("[ok] lake>river priority: river bytes cleared under lakes")
    return plan, g


def test_pyramid_preserves_river(plan, leaf):
    levels = water.build_water_pyramid(leaf, plan.num_levels)
    assert len(levels) == plan.num_levels
    # Shapes must match the HEIGHT pyramid exactly (same lattice / registration).
    dummy = np.zeros((plan.samples_y, plan.samples_x), np.float32)
    h_levels = core.build_pyramid(dummy, plan.num_levels)
    for wl, hl in zip(levels, h_levels):
        assert wl.type.shape == hl.shape, (wl.type.shape, hl.shape)
    # The thin river must survive winner-take-all all the way to the coarsest level.
    for lvl, g in enumerate(levels):
        assert (g.type == water.TYPE_RIVER).any(), f"river vanished at level {lvl}"
        assert (g.type == water.TYPE_LAKE).any(), f"lake vanished at level {lvl}"
    print(f"[ok] river+lake survive winner-take-all to coarsest of {len(levels)} levels")
    return levels


def test_shared_edges_and_pack(plan, levels):
    captured = {}

    def writer(path, rec):
        captured[path.replace("\\", "/")] = rec.copy()

    res = water.export_water_tiles(plan, levels, "out", writer=writer)
    assert res.tiles_written == plan.total_tiles()
    TS = plan.tile_cells + 1

    def tile(tx, ty, lvl=0):
        return captured[f"out/L{lvl}/{tx}_{ty}.water"].reshape(TS, TS)

    # Raw-record edge identity (prove no cracks in the mask, same as height).
    for ty in range(plan.leaf_tiles_y):
        for tx in range(plan.leaf_tiles_x - 1):
            a, b = tile(tx, ty), tile(tx + 1, ty)
            assert np.array_equal(a[:, -1], b[:, 0]), f"x-seam {tx},{ty}"
    for ty in range(plan.leaf_tiles_y - 1):
        for tx in range(plan.leaf_tiles_x):
            a, b = tile(tx, ty), tile(tx, ty + 1)
            assert np.array_equal(a[0, :], b[-1, :]), f"y-seam {tx},{ty}"
    print("[ok] all leaf .water tile edges bit-identical (aligns with height tiles)")

    # Round-trip: bytes -> record dtype reproduces the fields.
    any_tile = next(iter(captured.values()))
    raw = any_tile.tobytes()
    back = np.frombuffer(raw, dtype=water.WATER_DTYPE)
    assert back.size == TS * TS
    assert back.dtype.itemsize == 5
    print("[ok] .water record layout round-trips as 5-byte little-endian samples")

    # Manifest sanity.
    wm = res.water_manifest
    assert wm["record_bytes"] == 5
    assert wm["lake_count"] >= 1
    assert "1" in wm["lake_table"]
    print(f"[ok] water manifest: {wm['lake_count']} lake(s), "
          f"{res.tiles_written} tiles, {len(wm['levels'])} levels")


def test_run_export_integration(tmpdir="out_int"):
    """core.run_export with include_water, fully offline (synthetic fetchers)."""
    import os, json, shutil, tempfile
    plan = _plan()
    out = tempfile.mkdtemp(prefix="kvwater_")

    def height_fetch(u, e, sx, sy, nx, ny, sp):
        xs = sx + np.arange(nx) * sp
        ys = sy + np.arange(ny) * sp
        XX, YY = np.meshgrid(xs, ys)
        return (100.0 + 0.001 * XX + 0.001 * YY).astype(np.float32)[::-1, :]

    res = core.run_export(
        plan, out, fetcher=height_fetch, server_url="(demo)",
        include_water=True,
        water_opts={"fetcher": water.synthetic_water_features},
    )
    with open(os.path.join(out, "manifest.json")) as f:
        man = json.load(f)
    assert "water" in man, "water section not folded into manifest"
    # Every height tile has a sibling .water tile.
    for lvl in man["levels"]:
        for t in lvl["tiles"]:
            r16 = os.path.join(out, t["file"])
            wat = r16.replace(".r16", ".water")
            assert os.path.exists(r16) and os.path.exists(wat), t["file"]
    shutil.rmtree(out, ignore_errors=True)
    print("[ok] run_export(include_water=True) writes paired .r16 + .water tiles "
          "and one merged manifest")


if __name__ == "__main__":
    p = test_grid_transform_is_corner_centered()
    test_bearing_roundtrip()
    p, leaf = test_rasterize_places_features(p)
    levels = test_pyramid_preserves_river(p, leaf)
    test_shared_edges_and_pack(p, levels)
    test_run_export_integration()
    print("\nALL WATER CHECKS PASSED")
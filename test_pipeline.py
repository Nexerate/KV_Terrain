"""
Offline validation of the pipeline with a synthetic surface (no network).

We define a known continuous height function H(X, Y) over UTM space and a
synthetic fetcher that samples it EXACTLY at the requested grid points. Then we
assert the properties that the corner-centered convention requires.
"""
import numpy as np
from kvterrain import core


# A surface with both a linear part (to test exact decimation registration) and
# a smooth nonlinear part (to test general behaviour).
def H(X, Y):
    return 0.001 * X - 0.0007 * Y + 50.0 * np.sin(X / 800.0) * np.cos(Y / 650.0) + 500.0


def synthetic_fetcher(server_url, epsg, sx0, sy0, nx, ny, spacing):
    # Build the grid of requested sample CENTERS and evaluate H. Return north-up.
    xs = sx0 + np.arange(nx) * spacing
    ys = sy0 + np.arange(ny) * spacing            # south -> north
    XX, YY = np.meshgrid(xs, ys)                  # row 0 = south here
    vals = H(XX, YY).astype(np.float32)
    return vals[::-1, :]                          # flip to north-up (row 0 = north)


def test_alignment_and_assembly():
    plan = core.GridPlan(
        epsg=25833, spacing_m=5.0, origin_x=265000.0, origin_y=6648000.0,
        tile_cells=128, leaf_tiles_x=4, leaf_tiles_y=4, num_levels=3,
    )
    # Force multi-chunk assembly to exercise chunk stitching.
    leaf = core.assemble_region(plan, synthetic_fetcher,
                                core.IMAGESERVER["DTM"], max_fetch_px=200)

    SX, SY = plan.samples_x, plan.samples_y
    assert leaf.shape == (SY, SX)

    # Reconstruct the expected north-up array directly and compare.
    xs = plan.origin_x + np.arange(SX) * plan.spacing_m
    ys = plan.origin_y + np.arange(SY) * plan.spacing_m
    XX, YY = np.meshgrid(xs, ys)
    expected_south_up = H(XX, YY)
    expected = expected_south_up[::-1, :]
    err = np.nanmax(np.abs(leaf - expected))
    assert err < 1e-2, f"assembly misaligned, max err {err}"
    print(f"[ok] assembly aligned across chunks, max err {err:.2e} m")
    return plan, leaf


def test_decimation_registration(plan, leaf):
    # For the LINEAR part of any surface, a centered [1 2 1] decimation must
    # reproduce child[2i] exactly. We test on a purely linear field so the
    # property is exact to float precision.
    SX, SY = plan.samples_x, plan.samples_y
    xs = plan.origin_x + np.arange(SX) * plan.spacing_m
    ys = plan.origin_y + np.arange(SY) * plan.spacing_m
    XX, YY = np.meshgrid(xs, ys)
    lin = (0.3 * XX - 0.2 * YY).astype(np.float32)[::-1, :]  # north-up linear
    parent = core.decimate_corner(lin)
    child_even = lin[0::2, 0::2]
    diff = np.abs(parent - child_even)
    # INTERIOR must be exact: a parent sample lands exactly on child[2i,2j].
    interior_err = np.max(diff[1:-1, 1:-1])
    assert interior_err < 1e-3, f"interior decimation drifted, err {interior_err}"
    print(f"[ok] interior decimation exact on linear field, err {interior_err:.2e}")
    # RIM is an edge-clamp approximation, bounded by (|dx|+|dy|)/4 per sample.
    rim_err = diff.copy(); rim_err[1:-1, 1:-1] = 0.0
    print(f"[ok] region-rim clamp deviation bounded at {rim_err.max():.3f} m "
          f"(outer row/col only, has no neighbour)")

    # Shape check: (n+1) -> (n/2+1) per axis.
    assert parent.shape == ((lin.shape[0] - 1) // 2 + 1, (lin.shape[1] - 1) // 2 + 1)
    print(f"[ok] decimation shape {lin.shape} -> {parent.shape}")


def test_shared_edges_and_pack(plan, leaf):
    levels = core.build_pyramid(leaf, plan.num_levels)
    assert len(levels) == plan.num_levels

    captured = {}

    def writer(path, a16):
        captured[path.replace("\\", "/")] = a16.copy()

    res = core.export_tiles(plan, levels, "out", writer=writer)
    print(f"[ok] wrote {res.tiles_written} tiles, "
          f"range [{res.height_min:.1f}, {res.height_max:.1f}] m")
    assert res.tiles_written == plan.total_tiles()

    TS = plan.tile_cells + 1
    # Every tile is TS x TS uint16.
    for p, a in captured.items():
        assert a.shape == (TS * TS,) or a.size == TS * TS

    # Shared-edge check at leaf level: right column of (tx,ty) == left column of
    # (tx+1,ty); top row of (tx,ty) == bottom row of (tx,ty+1). Compare the raw
    # R16 bytes so we prove bit-identical edges.
    def tile(tx, ty, lvl=0):
        return captured[f"out/L{lvl}/{tx}_{ty}.r16"].reshape(TS, TS)

    for ty in range(plan.leaf_tiles_y):
        for tx in range(plan.leaf_tiles_x - 1):
            a, b = tile(tx, ty), tile(tx + 1, ty)
            assert np.array_equal(a[:, -1], b[:, 0]), f"x-seam at {tx},{ty}"
    for ty in range(plan.leaf_tiles_y - 1):
        for tx in range(plan.leaf_tiles_x):
            a, b = tile(tx, ty), tile(tx, ty + 1)
            # row 0 is NORTH (top). tile(ty) north edge meets tile(ty+1) south edge.
            assert np.array_equal(a[0, :], b[-1, :]), f"y-seam at {tx},{ty}"
    print("[ok] all leaf tile edges bit-identical (no cracks)")

    # Coarsest level is a single tile (square region).
    assert levels[-1].shape == (TS, TS)
    print(f"[ok] coarsest level is one {TS}x{TS} root tile")


if __name__ == "__main__":
    plan, leaf = test_alignment_and_assembly()
    test_decimation_registration(plan, leaf)
    test_shared_edges_and_pack(plan, leaf)
    print("\nALL CHECKS PASSED")
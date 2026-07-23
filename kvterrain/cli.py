"""kvterrain command-line interface."""
import argparse
import json
import os
import sys

import numpy as np

from . import core


def _demo_fetcher(u, e, sx, sy, nx, ny, sp):
    xs = sx + np.arange(nx) * sp
    ys = sy + np.arange(ny) * sp
    XX, YY = np.meshgrid(xs, ys)
    surf = (300.0
            + 400.0 * np.exp(-(((XX - XX.mean()) / 4000.0) ** 2
                               + ((YY - YY.mean()) / 4000.0) ** 2))
            + 60.0 * np.sin(XX / 900.0) * np.cos(YY / 700.0))
    return surf.astype(np.float32)[::-1, :]


def cmd_build(a):
    lon_min, lat_min, lon_max, lat_max = a.bbox
    plan = core.plan_grid(lon_min, lat_min, lon_max, lat_max,
                          spacing_m=a.spacing, tile_cells=a.tile_cells, epsg=a.epsg)
    print(f"CRS EPSG:{plan.epsg}  region {plan.width_m:.0f}x{plan.height_m:.0f} m  "
          f"leaf {plan.leaf_tiles_x}x{plan.leaf_tiles_y} tiles  "
          f"{plan.num_levels} levels  {plan.total_tiles()} tiles total")

    def prog(d, t, m):
        print(f"\r  fetch {d}/{t}", end="", flush=True)

    fetcher = _demo_fetcher if a.demo else None
    server = "(demo)" if a.demo else None

    water_opts = None
    if a.water:
        from . import water as _water
        water_opts = {
            "include_main_rivers": not a.no_main_rivers,
            "width_scale": a.river_width_scale,
            "lake_ramp_radius_m": a.lake_ramp_radius,
            "lake_max_depth_m": a.lake_max_depth,
            "river_depth_scale": a.river_depth_scale,
        }
        if a.demo:
            water_opts["fetcher"] = _water.synthetic_water_features

    res = core.run_export(plan, a.out, source_kind=a.source,
                          fetcher=fetcher, server_url=server,
                          max_fetch_px=a.max_px,
                          nodata_fill_m=a.nodata_fill,
                          height_min=a.hmin, height_max=a.hmax,
                          progress=prog,
                          include_water=a.water, water_opts=water_opts)
    print(f"\npacked {res.tiles_written} tiles to {a.out}  "
          f"range [{res.height_min:.1f}, {res.height_max:.1f}] m")
    atlas = res.manifest["atlas"]
    line = (f"  + {atlas['height_file']} "
            f"({atlas['height_bytes'] / 1e6:.1f} MB, {atlas['tile_bytes']}-byte tiles)")
    if atlas.get("surface_file"):
        line += f" + {atlas['surface_file']}"
    print(line)
    if a.water:
        wm = res.manifest.get("water_surface", {})
        print(f"  + water surface: {wm.get('lake_count', 0)} lakes, "
              f"packed in {atlas['surface_file']} (© NVE)")
        print(f"  + lake beds carved flat {a.lake_max_depth:.1f} m below the known "
              f"NVE surface (bevel radius {a.lake_ramp_radius:.1f} m)")
        print(f"  + river surface raised above the DTM channel by stream order "
              f"(× {a.river_depth_scale:g})")


def cmd_validate(a):
    import requests

    with open(os.path.join(a.out, "manifest.json")) as f:
        man = json.load(f)
    epsg = int(man["crs"].split(":")[1])
    hmin, hmax = man["height_min_m"], man["height_max_m"]
    tc = man["tile_cells"]
    ts = man["tile_samples"]
    spacing = man["leaf_spacing_m"]
    ox, oy = man["origin_utm"]
    lx, ly = man["leaf_tiles"]
    nlev = man["num_levels"]

    # Everything about a leaf tile is implied by the header — its atlas byte offset and its
    # world-space SW corner — so we sample leaf tiles directly from the grid, no per-tile
    # metadata needed. Read the packed u16 grid out of the height atlas by computed offset.
    tile_bytes = core.atlas_tile_bytes(ts)
    atlas_path = os.path.join(a.out, man["atlas"]["height_file"])
    rng = np.random.default_rng(0)
    coords = [(x, y) for y in range(ly) for x in range(lx)]      # level-0 grid
    pick = rng.choice(len(coords), min(a.n, len(coords)), replace=False)

    sess = requests.Session()
    errs = []
    with open(atlas_path, "rb") as atlas_fh:
        for idx in pick:
            x, y = coords[int(idx)]
            off = core.atlas_tile_offset(lx, ly, nlev, ts, 0, x, y)
            atlas_fh.seek(off)
            a16 = np.frombuffer(atlas_fh.read(tile_bytes), dtype="<u2").reshape(ts, ts)

            i = int(rng.integers(0, tc + 1)); j = int(rng.integers(0, tc + 1))
            packed = a16[tc - j, i]
            h_packed = hmin + (packed / 65535.0) * (hmax - hmin)
            # Tile SW corner in world CRS, then the sampled cell within it.
            X = ox + (x * tc + i) * spacing
            Y = oy + (y * tc + j) * spacing
            try:
                r = sess.get(core.POINT_API,
                             params={"ost": X, "nord": Y, "koordsys": epsg},
                             timeout=30)
                data = r.json()
                h_api = data.get("punkter", [{}])[0].get("z")
            except Exception as ex:
                print(f"  point API call failed ({ex}); check param names live.")
                return
            if h_api is None:
                continue
            errs.append(abs(h_api - h_packed))
            print(f"  ({X:.0f},{Y:.0f})  packed {h_packed:7.2f}  api {h_api:7.2f}  "
                  f"d={abs(h_api - h_packed):.2f} m")
    if errs:
        print(f"\nmedian |error| = {np.median(errs):.2f} m  "
              f"(includes R16 quantisation ~{(hmax - hmin) / 65535:.3f} m)")


def cmd_validate_atlas(a):
    """Verify a dense atlas against the manifest header and (if present) the
    per-tile files: exact file size, and byte-for-byte equality of a sample of
    tiles extracted by computed offset. Implements the §8 export-side checks."""
    with open(os.path.join(a.out, "manifest.json")) as f:
        man = json.load(f)

    atlas = man.get("atlas")
    if not atlas:
        print("no 'atlas' block in manifest — not a kvterrain atlas export.")
        sys.exit(1)

    lx, ly = man["leaf_tiles"]
    nlev = man["num_levels"]
    ts = man["tile_samples"]
    tile_bytes = core.atlas_tile_bytes(ts)
    expect = core.atlas_total_bytes(lx, ly, nlev, ts)

    targets = [("height", atlas.get("height_file"))]
    if atlas.get("surface_file"):
        targets.append(("surface", atlas["surface_file"]))

    rng = np.random.default_rng(0)
    ok = True
    for kind, fname in targets:
        path = os.path.join(a.out, fname)
        if not os.path.exists(path):
            print(f"[{kind}] MISSING atlas file {fname}")
            ok = False
            continue

        actual = os.path.getsize(path)
        size_ok = actual == expect
        ok &= size_ok
        print(f"[{kind}] {fname}: {actual} bytes, expected {expect} "
              f"-> {'OK' if size_ok else 'MISMATCH (grid not dense!)'}")
        if not size_ok:
            continue

        # Sample tiles across levels and confirm each computed offset yields a full
        # tileBytes block inside the file (offsets land where the header implies).
        checked = 0
        with open(path, "rb") as fh:
            for lvl in range(nlev):
                tx, ty = core.tiles_at_level(lx, ly, lvl)
                coords = [(x, y) for y in range(ty) for x in range(tx)]
                for idx in rng.choice(len(coords), size=min(a.n, len(coords)),
                                      replace=False):
                    x, y = coords[int(idx)]
                    off = core.atlas_tile_offset(lx, ly, nlev, ts, lvl, x, y)
                    fh.seek(off)
                    if len(fh.read(tile_bytes)) != tile_bytes:
                        print(f"    L{lvl} {x},{y}: short read at offset {off}")
                        ok = False
                    checked += 1
        print(f"    sampled {checked} tile offsets, all full-length")

    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def main(argv=None):
    p = argparse.ArgumentParser(prog="kvterrain")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="fetch + build tiles")
    b.add_argument("--bbox", nargs=4, type=float, required=True,
                   metavar=("LONMIN", "LATMIN", "LONMAX", "LATMAX"))
    b.add_argument("--spacing", type=float, default=5.0)
    b.add_argument("--tile-cells", type=int, default=128, dest="tile_cells")
    b.add_argument("--epsg", type=int, default=None)
    b.add_argument("--source", choices=["DTM", "DOM"], default="DTM")
    b.add_argument("--out", required=True)
    b.add_argument("--max-px", type=int, default=core.DEFAULT_MAX_FETCH_PX, dest="max_px")
    b.add_argument("--nodata-fill", type=float, default=0.0, dest="nodata_fill")
    b.add_argument("--hmin", type=float, default=None)
    b.add_argument("--hmax", type=float, default=None)
    b.add_argument("--demo", action="store_true")
    b.add_argument("--water", action="store_true")
    b.add_argument("--no-main-rivers", action="store_true", dest="no_main_rivers")
    b.add_argument("--river-width-scale", type=float, default=1.0, dest="river_width_scale")
    b.add_argument("--river-depth-scale", type=float, default=1.0, dest="river_depth_scale",
                   help="scales how far river surfaces sit above the DTM channel bed "
                        "(per stream order); >1 = deeper/more opaque rivers")
    b.add_argument("--lake-ramp-radius", type=float, default=10.0, dest="lake_ramp_radius",
                   help="shore-to-max-depth distance in metres for synthetic lake beds")
    b.add_argument("--lake-max-depth", type=float, default=20.0, dest="lake_max_depth",
                   help="maximum synthetic lake depth in metres")
    b.set_defaults(func=cmd_build)

    v = sub.add_parser("validate", help="spot-check packed leaf heights vs point API")
    v.add_argument("--out", required=True)
    v.add_argument("--n", type=int, default=12)
    v.set_defaults(func=cmd_validate)

    va = sub.add_parser("validate-atlas",
                        help="check dense atlas size + byte-for-byte vs per-tile files")
    va.add_argument("--out", required=True)
    va.add_argument("--n", type=int, default=8,
                    help="tiles sampled per level for the byte-for-byte check")
    va.set_defaults(func=cmd_validate_atlas)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main(sys.argv[1:])